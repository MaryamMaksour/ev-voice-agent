"""Event-driven post-call processing (producer/consumer).

The agent publishes a CallEnded event; a consumer runs extraction, fills missing CRM
fields, writes the summary note, gathers the call's logs and saves the call record.

  CeleryBus     (REDIS_URL set): the event goes to Redis; the postcall Celery worker
                runs it (agent/tasks.py). Retries, survives an agent restart.
  InProcessBus  (no REDIS_URL): same work in the agent process (laptop, tests).

Either way the agent first writes data/calls/<call_id>.raw.json, so a call is never lost
even if Redis or the worker is down; the final <call_id>.json replaces it."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

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
    ended_by: str = ""                                  # agent (end_call) / lead / error / not_reached ...
    close_reason: str = ""                              # LiveKit's reason the session closed
    tool_calls: list[dict] = field(default_factory=list)  # every tool the LLM called, with its output
    line: dict | None = None                            # phone line result (3CX); None for browser calls
    recording: dict | None = None                       # where the audio recording is, if recording is on
    transfer: dict | None = None                        # caller asked for a person: what happened
    versions: dict = field(default_factory=dict)        # code (git sha), prompt, livekit-agents, vLLM model
    settings: dict = field(default_factory=dict)        # temperature, turn detection, VAD, timeouts...
    errors: list[dict] = field(default_factory=list)    # every pipeline error, with its stage (stt/llm/tts...)
    log_files: dict = field(default_factory=dict)       # this call's log files under data/calls/


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
                 wait_for_recording: float = 0.0) -> None:
        self.crm = crm
        self.llm = llm
        self.model = model
        self.calls_dir = Path(calls_dir)
        self.logs_dir = logs_dir
        self.recordings_dir = recordings_dir
        self.wait_for_recording = wait_for_recording  # Egress closes the file a few seconds after the call

    async def handle(self, event: CallEnded) -> dict:
        record = asdict(event)
        record["agent_closed"] = event.ended_by == "agent"
        record["repeat_requests"] = repeat_requests(event.transcript)
        record["duration_seconds"] = _seconds_between(event.started_at, event.ended_at)
        # post-call CRM writes go to the same crm_changes.jsonl as the ones made during the call
        call_log = CallLogger(self.calls_dir, event.call_id, started_at=event.started_at)
        crm = AuditedCRM(self.crm, call_log.crm_change, source="post_call")
        t0 = asyncio.get_running_loop().time()
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
            outcome = await extract_outcome(self.llm, self.model, lead, event.transcript)
            fill, disagreements = reconcile(lead, outcome)
            if fill.changes():
                await crm.update_lead(event.lead_id, fill)
            # safety net: never keep dialing someone who asked us to stop
            if outcome.outcome.value == "do_not_call" and not lead.do_not_call:
                await crm.mark_do_not_call(event.lead_id)
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
        return await self._finish(event, record, call_log, t0)

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
        (self.calls_dir / f"{event.call_id}.raw.json").unlink(missing_ok=True)
        logger.info("call record saved: %s", path)
        return record


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


class CeleryBus:
    """Sends CallEnded to the postcall Celery worker through Redis.

    If Redis can't be reached, the event is processed in this process instead
    (`fallback`), so a broker outage never loses a call."""

    def __init__(self, calls_dir: str, fallback: InProcessBus | None = None) -> None:
        self.calls_dir = calls_dir
        self._fallback = fallback

    def publish(self, event: CallEnded) -> None:
        from .tasks import process_call

        write_raw(event, self.calls_dir)
        try:
            result = process_call.apply_async(args=[asdict(event)], retry=True,
                                              retry_policy={"max_retries": 3, "interval_start": 0.5,
                                                            "interval_step": 1.0, "interval_max": 3.0})
            logger.info("call %s sent to the postcall worker (task %s)", event.call_id, result.id)
        except Exception:
            logger.exception("could not reach Redis; processing call %s in the agent instead", event.call_id)
            if self._fallback is not None:
                self._fallback.publish(event)

    async def drain(self) -> None:
        if self._fallback is not None:
            await self._fallback.drain()
