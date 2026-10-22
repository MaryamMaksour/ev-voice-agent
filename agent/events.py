"""Event-driven post-call processing (producer/consumer).

The agent publishes a CallEnded event; a consumer runs extraction, fills missing CRM
fields, writes the summary note, gathers the call's logs and saves the call record.

The consumer runs inside the agent's call process (InProcessBus), after the caller has
hung up: it takes a second or two on vLLM, and the call process stays open until it is done.
No Celery: at ~2,800 calls/day one process per call already runs this work in parallel.

Safety net: the agent first writes data/calls/<call_id>.raw.json. If the process dies before
the final <call_id>.json replaces it, `python -m agent.postcall` finishes the work later
(the agent container runs it at every start and then every few minutes).

If the post-call work itself fails (the LLM is down, the CRM rejects a write), the raw file is
kept and <call_id>.json is written with post_call_status "failed", so the dashboard shows the call.
Each retry counts in post_call_attempts; after POST_CALL_MAX_ATTEMPTS the raw file is removed and
the status becomes "needs_review": someone must update that lead by hand."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from openai import AsyncOpenAI

from crm.audited import AuditedCRM
from crm.port import CRMClient

from .calllog import CallLogger, bundle_call_logs, recording_seconds
from .extraction import extract_outcome, reconcile

logger = logging.getLogger("voice-agent.events")


# the agent asking the caller to say something again: a sign that speech-to-text missed it
_REPEAT = re.compile(r"\b(sorry|pardon|apologi[sz]e)\b.{0,40}\b(repeat|again|catch that|hear you|didn'?t get)"
                     r"|\bcould you (please )?(repeat|say that again)", re.I)


def repeat_requests(transcript: list[dict]) -> int:
    return sum(1 for t in transcript if t.get("role") == "agent" and _REPEAT.search(t.get("text", "")))


@dataclass
class CallEnded:
    call_id: str
    lead_id: str
    transcript: list[dict]
    actions: list[dict]
    metrics: dict
    started_at: str
    ended_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    room: str = ""
    models: dict = field(default_factory=dict)          # llm / stt / tts model names used on this call
    ended_by: str = ""                                  # agent (end_call) / lead / error / not_reached / llm_failure ...
    close_reason: str = ""                              # LiveKit's reason the session closed
    tool_calls: list[dict] = field(default_factory=list)  # every tool the LLM called, with its output
    line: dict | None = None                            # phone line result (3CX); None for browser calls
    recording: dict | None = None                       # where the audio recording is, if recording is on
    transfer: dict | None = None                        # caller asked for a person: what happened
    versions: dict = field(default_factory=dict)        # code (git sha), prompt, livekit-agents, vLLM model
    settings: dict = field(default_factory=dict)        # temperature, turn detection, VAD, timeouts...
    errors: list[dict] = field(default_factory=list)    # every pipeline error, with its stage (stt/llm/tts...)
    log_files: dict = field(default_factory=dict)       # this call's log files under data/calls/
    fallback: dict | None = None                        # the LLM failed mid-call: what the agent did (fallback.py)
    post_call_attempts: int = 0                         # failed post-call runs so far (kept in the raw file)


def write_raw(event: CallEnded, calls_dir: str | Path) -> Path:
    """The call as the agent saw it, before post-call work: replaced by <call_id>.json when done."""
    folder = Path(calls_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{event.call_id}.raw.json"
    path.write_text(json.dumps(asdict(event), indent=2, default=str))
    return path


class PostCallProcessor:
    def __init__(self, crm: CRMClient, llm: AsyncOpenAI, model: str, calls_dir: str, *,
                 logs_dir: str = "logs", recordings_dir: str | None = None,
                 wait_for_recording: float = 0.0, max_attempts: int = 5, tz: str = "UTC") -> None:
        self.crm = crm
        self.llm = llm
        self.model = model
        self.calls_dir = Path(calls_dir)
        self.logs_dir = logs_dir
        self.recordings_dir = recordings_dir
        self.wait_for_recording = wait_for_recording  # Egress closes the file a few seconds after the call
        self.max_attempts = max_attempts
        self.tz = ZoneInfo(tz)  # callback times the lead said are in this zone (CALL_TIMEZONE)

    async def handle(self, event: CallEnded) -> dict:
        record = asdict(event)
        record["agent_closed"] = event.ended_by == "agent"
        record["repeat_requests"] = repeat_requests(event.transcript)
        record["duration_seconds"] = _seconds_between(event.started_at, event.ended_at)
        record["post_call_status"] = "done"
        # post-call CRM writes go to the same crm_changes.jsonl as the ones made during the call
        call_log = CallLogger(self.calls_dir, event.call_id, started_at=event.started_at)
        crm = AuditedCRM(self.crm, call_log.crm_change, source="post_call")
        t0 = asyncio.get_running_loop().time()
        llm_stats: list[dict] = []
        if event.line and event.line.get("result") != "answered":
            # nobody picked up: nothing to extract, just note the attempt
            record["classification"] = "not_reached"
            try:
                await crm.add_note(event.lead_id, f"Call attempt: {event.line.get('label')} "
                                                  f"(SIP {event.line.get('sip_status_code')})")
            except Exception:
                logger.exception("could not note the call attempt for %s", event.lead_id)
            return await self._finish(event, record, call_log, t0)
        try:
            lead = await crm.get_lead(event.lead_id)
            outcome = await extract_outcome(self.llm, self.model, lead, event.transcript, stats=llm_stats,
                                            call_time=self._call_time(event))
            fill, disagreements = reconcile(lead, outcome)
            if fill.changes():
                await crm.update_lead(event.lead_id, fill)
            # safety net: never keep dialing someone who asked us to stop
            if outcome.outcome.value == "do_not_call" and not lead.do_not_call:
                await crm.mark_do_not_call(event.lead_id)
            record["callback"] = await self._book_callback(crm, event, outcome)
            if outcome.outcome.value != "no_conversation":
                await crm.add_note(event.lead_id, outcome.summary)
            record.update(
                extraction=outcome.model_dump(mode="json"),
                filled_after_call=fill.changes(),
                disagreements=disagreements,
                classification=outcome.outcome.value,
            )
        except Exception as e:  # never lose the call record because extraction failed
            logger.exception("post-call processing failed for %s", event.call_id)
            record["error"] = repr(e)
            record["errors"] = [*record.get("errors", []), {"stage": "post_call", "error": repr(e),
                                                            "at": datetime.now(timezone.utc).isoformat()}]
            record["post_call_attempts"] = event.post_call_attempts = event.post_call_attempts + 1
            record["post_call_status"] = "failed" if event.post_call_attempts < self.max_attempts else "needs_review"
        # the post-call LLM check is timed and token-counted like the in-call requests
        for row in llm_stats:
            call_log.post_call_llm(row)
        record["post_call_llm"] = post_call_llm_summary(llm_stats)
        return await self._finish(event, record, call_log, t0)

    def _call_time(self, event: CallEnded) -> datetime:
        try:
            return datetime.fromisoformat(event.ended_at).astimezone(self.tz)
        except (TypeError, ValueError):
            return datetime.now(self.tz)

    async def _book_callback(self, crm: CRMClient, event: CallEnded, outcome) -> dict | None:
        """The lead asked for a callback but the agent never booked one during the call: book what the
        check found, or leave a note for a person when there is no usable time."""
        if not outcome.callback_requested:
            return None
        if any(a.get("action") == "schedule_callback" for a in event.actions):
            return {"booked": True, "by": "call"}
        said = outcome.callback_time_text or outcome.callback_at or "no time given"
        when = None
        try:
            when = datetime.fromisoformat(outcome.callback_at) if outcome.callback_at else None
            if when is not None and when.tzinfo is None:
                when = when.replace(tzinfo=self.tz)
        except ValueError:
            pass
        if when is not None and when > datetime.now(self.tz):
            await crm.schedule_callback(event.lead_id, when)
            return {"booked": True, "by": "post_call", "callback_at": when.isoformat(timespec="minutes"),
                    "said": said}
        await crm.add_note(event.lead_id, f"The lead asked to be called back ({said}), but no exact time "
                                          f"could be set. Please follow up.")
        return {"booked": False, "by": "post_call", "said": said}

    async def _finish(self, event: CallEnded, record: dict, call_log: CallLogger, t0: float) -> dict:
        record["post_call_seconds"] = round(asyncio.get_running_loop().time() - t0, 3)
        rec_file = (event.recording or {}).get("file") if (event.recording or {}).get("ok") else None
        if rec_file and self.recordings_dir:
            local = Path(self.recordings_dir) / Path(rec_file).name
            waited = 0.0
            while not local.exists() and waited < self.wait_for_recording:
                await asyncio.sleep(1)
                waited += 1
            record["recording"] = {**event.recording, "path": str(local) if local.exists() else None,
                                   "seconds": recording_seconds(local) if local.exists() else None}
        call_log.event("post_call_done", classification=record.get("classification"),
                       seconds=record["post_call_seconds"], error=record.get("error"))
        record["log_files"] = bundle_call_logs(event.call_id, calls_dir=self.calls_dir, logs_dir=self.logs_dir,
                                               recordings_dir=self.recordings_dir, recording_file=rec_file)
        return self._save(event, record)

    def _save(self, event: CallEnded, record: dict) -> dict:
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        path = self.calls_dir / f"{event.call_id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2, default=str))
        tmp.replace(path)  # the dashboard never reads a half-written file
        if record.get("post_call_status") == "failed":
            write_raw(event, self.calls_dir)  # keep it (new mtime = the wait before the next try)
            logger.warning("post-call work failed for %s (attempt %s of %s); will retry", event.call_id,
                           event.post_call_attempts, self.max_attempts)
        else:
            (self.calls_dir / f"{event.call_id}.raw.json").unlink(missing_ok=True)
            if record.get("post_call_status") == "needs_review":
                logger.error("post-call work failed %s times for %s; giving up, update the lead by hand",
                             event.post_call_attempts, event.call_id)
        logger.info("call record saved: %s", path)
        return record


def post_call_llm_summary(rows: list[dict]) -> dict:
    """Totals of the post-call LLM requests (more than one when the first JSON was invalid)."""
    return {
        "requests": len(rows),
        "seconds": round(sum(r.get("seconds") or 0 for r in rows), 3),
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in rows),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in rows),
        "ok": bool(rows) and bool(rows[-1].get("ok")),
    }


def load_raw(path: str | Path) -> CallEnded:
    """A <call_id>.raw.json back into the CallEnded event (unknown keys are ignored)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {f.name for f in fields(CallEnded)}
    return CallEnded(**{k: v for k, v in data.items() if k in known})


async def process_pending(processor: PostCallProcessor, *, min_age_seconds: float = 300) -> list[str]:
    """Finish calls whose post-call work never completed (agent crashed, was restarted, or the
    work failed and is waiting for a retry).

    Only raw files older than `min_age_seconds` are taken, so a call that is being
    processed right now is left alone. Returns the call ids that were finished; a call that
    failed again stays (or, after the last attempt, is marked needs_review) and is not returned."""
    done: list[str] = []
    now = time.time()
    for raw in sorted(processor.calls_dir.glob("*.raw.json")):
        if now - raw.stat().st_mtime < min_age_seconds:
            continue
        try:
            event = load_raw(raw)
        except (OSError, ValueError, TypeError):
            logger.exception("unreadable raw call file %s; left in place", raw)
            continue
        logger.info("finishing post-call work for %s", event.call_id)
        record = await processor.handle(event)
        if record.get("post_call_status") == "done":
            done.append(event.call_id)
    return done


def _seconds_between(start: str, end: str) -> float | None:
    try:
        return round((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds(), 1)
    except (TypeError, ValueError):
        return None


class InProcessBus:
    def __init__(self, processor: PostCallProcessor) -> None:
        self._processor = processor
        self._queue: asyncio.Queue[CallEnded] = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    def publish(self, event: CallEnded) -> None:
        write_raw(event, self._processor.calls_dir)
        if self._worker is None:
            self._worker = asyncio.create_task(self._consume())
        self._queue.put_nowait(event)

    async def _consume(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await self._processor.handle(event)
            finally:
                self._queue.task_done()

    async def drain(self) -> None:
        """Wait until every published event is processed (called at job shutdown)."""
        await self._queue.join()
        if self._worker:
            self._worker.cancel()
            self._worker = None
