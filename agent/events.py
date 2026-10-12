"""Event-driven post-call processing (producer/consumer).

The agent publishes a CallEnded event; a consumer runs extraction, fills missing CRM
fields, writes the summary note and saves the call record. For the demo the bus is an
in-process asyncio.Queue; in production the same `publish` goes to Redis Streams and
the consumer runs as its own worker."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

from crm.port import CRMClient

from .extraction import extract_outcome, reconcile

logger = logging.getLogger("voice-agent.events")


@dataclass
class CallEnded:
    call_id: str
    lead_id: str
    transcript: list[dict]
    actions: list[dict]
    metrics: dict
    started_at: str
    ended_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PostCallProcessor:
    def __init__(self, crm: CRMClient, llm: AsyncOpenAI, model: str, calls_dir: str) -> None:
        self.crm = crm
        self.llm = llm
        self.model = model
        self.calls_dir = Path(calls_dir)

    async def handle(self, event: CallEnded) -> dict:
        record = asdict(event)
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
            )
        except Exception as e:  # never lose the call record because extraction failed
            logger.exception("post-call processing failed for %s", event.call_id)
            record["error"] = repr(e)

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
