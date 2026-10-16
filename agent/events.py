"""Event-driven post-call processing (producer/consumer).

The agent publishes a CallEnded event; a consumer runs extraction, fills missing CRM
fields, writes the summary note and saves the call record. For the demo the bus is an
in-process asyncio.Queue; in production the same `publish` goes to Redis Streams and
the consumer runs as its own worker."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

from crm.port import CRMClient

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


class PostCallProcessor:
    def __init__(self, crm: CRMClient, llm: AsyncOpenAI, model: str, calls_dir: str) -> None:
        self.crm = crm
        self.llm = llm
        self.model = model
        self.calls_dir = Path(calls_dir)

    async def handle(self, event: CallEnded) -> dict:
        record = asdict(event)
        record["agent_closed"] = event.ended_by == "agent"
        record["repeat_requests"] = repeat_requests(event.transcript)
        if event.line and event.line.get("result") != "answered":
            # nobody picked up: nothing to extract, just note the attempt
            record["classification"] = "not_reached"
            try:
                await self.crm.add_note(event.lead_id, f"Call attempt: {event.line.get('label')} "
                                                       f"(SIP {event.line.get('sip_status_code')})")
            except Exception:
                logger.exception("could not note the call attempt for %s", event.lead_id)
            return self._save(event, record)
        try:
            lead = await self.crm.get_lead(event.lead_id)
            outcome = await extract_outcome(self.llm, self.model, lead, event.transcript)
            fill, disagreements = reconcile(lead, outcome)
            if fill.changes():
                await self.crm.update_lead(event.lead_id, fill)
            # safety net: never keep dialing someone who asked us to stop
            if outcome.outcome.value == "do_not_call" and not lead.do_not_call:
                await self.crm.mark_do_not_call(event.lead_id)
            if outcome.outcome.value != "no_conversation":
                await self.crm.add_note(event.lead_id, outcome.summary)
            record.update(
                extraction=outcome.model_dump(mode="json"),
                filled_after_call=fill.changes(),
                disagreements=disagreements,
                classification=outcome.outcome.value,
            )
        except Exception as e:  # never lose the call record because extraction failed
            logger.exception("post-call processing failed for %s", event.call_id)
            record["error"] = repr(e)
        return self._save(event, record)

    def _save(self, event: CallEnded, record: dict) -> dict:
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        path = self.calls_dir / f"{event.call_id}.json"
        path.write_text(json.dumps(record, indent=2, default=str))
        logger.info("call record saved: %s", path)
        return record


class InProcessBus:
    def __init__(self, processor: PostCallProcessor) -> None:
        self._processor = processor
        self._queue: asyncio.Queue[CallEnded] = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    def publish(self, event: CallEnded) -> None:
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
