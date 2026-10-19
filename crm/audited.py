"""AuditedCRM: wraps any CRM adapter and reports every write with the lead before and after.

    crm = AuditedCRM(make_crm(), on_change=call_logger.crm_change, source="call")

Reads pass straight through. A failed write is still reported (ok=False) and re-raised,
so the caller behaves exactly as with the bare adapter.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Callable

from .port import CRMClient, Lead, LeadUpdate

logger = logging.getLogger("voice-agent.crm")

# on_change(source=..., action=..., lead_id=..., before=..., after=..., args=..., ok=..., error=..., seconds=...)
OnChange = Callable[..., None]


def _snapshot(lead: Lead | None) -> dict | None:
    return None if lead is None else lead.model_dump(mode="json")


class AuditedCRM:
    def __init__(self, inner: CRMClient, on_change: OnChange, source: str = "call") -> None:
        self.inner = inner
        self.on_change = on_change
        self.source = source

    async def get_lead(self, lead_id: str) -> Lead:
        return await self.inner.get_lead(lead_id)

    async def _write(self, action: str, lead_id: str, args: dict, fn) -> Lead:
        try:
            before = _snapshot(await self.inner.get_lead(lead_id))
        except Exception:
            before = None
        t0 = time.perf_counter()
        try:
            after = await fn()
        except Exception as e:
            self._report(action, lead_id, args, before, None, ok=False, error=repr(e), seconds=time.perf_counter() - t0)
            raise
        self._report(action, lead_id, args, before, _snapshot(after), ok=True, seconds=time.perf_counter() - t0)
        return after

    def _report(self, action: str, lead_id: str, args: dict, before, after, **kw: Any) -> None:
        try:
            self.on_change(source=self.source, action=action, lead_id=lead_id, before=before, after=after,
                           args=args, **kw)
        except Exception:  # logging must never break a CRM write
            logger.exception("could not record CRM change %s for %s", action, lead_id)

    async def update_lead(self, lead_id: str, update: LeadUpdate) -> Lead:
        return await self._write("update_lead", lead_id, update.changes(),
                                 lambda: self.inner.update_lead(lead_id, update))

    async def schedule_callback(self, lead_id: str, when: datetime) -> Lead:
        return await self._write("schedule_callback", lead_id, {"when": when.isoformat()},
                                 lambda: self.inner.schedule_callback(lead_id, when))

    async def mark_do_not_call(self, lead_id: str) -> Lead:
        return await self._write("mark_do_not_call", lead_id, {}, lambda: self.inner.mark_do_not_call(lead_id))

    async def add_note(self, lead_id: str, note: str) -> Lead:
        return await self._write("add_note", lead_id, {"note": note}, lambda: self.inner.add_note(lead_id, note))
