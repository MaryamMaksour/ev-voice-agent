"""Tool logic, kept free of LiveKit so it can be unit-tested directly.

The lead id is bound when the call starts, so the LLM can only ever touch the lead
it is calling."""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from crm.port import CRMClient, LeadUpdate

logger = logging.getLogger("voice-agent.tools")


class LeadTools:
    def __init__(self, crm: CRMClient, lead_id: str, tz: str = "UTC") -> None:
        self.crm = crm
        self.lead_id = lead_id
        self.tz = ZoneInfo(tz)
        self.actions: list[dict] = []  # audit log of every CRM write in this call

    def _log(self, action: str, **data) -> None:
        self.actions.append({"action": action, **data})
        logger.info("crm %s lead=%s %s", action, self.lead_id, data)

    async def update_lead(
        self,
        status: str | None = None,
        budget: int | None = None,
        preferred_area: str | None = None,
        property_type: str | None = None,
        timeline: str | None = None,
        interest_level: str | None = None,
    ) -> str:
        try:
            update = LeadUpdate(
                status=status,
                budget=budget,
                preferred_area=preferred_area,
                property_type=property_type.lower() if property_type else None,
                timeline=timeline,
                interest_level=interest_level.lower() if interest_level else None,
            )
        except ValidationError as e:
            # tell the model what was wrong so it can retry with valid values
            errors = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            return f"Not saved, invalid values: {errors}"
        changes = update.changes()
        if not changes:
            return "Nothing to save."
        await self.crm.update_lead(self.lead_id, update)
        self._log("update_lead", **changes)
        return f"Saved: {', '.join(changes)}."

    async def schedule_callback(self, when: str) -> str:
        try:
            dt = datetime.fromisoformat(when)
        except ValueError:
            return "Not saved: 'when' must be an ISO 8601 datetime like 2026-10-14T16:00."
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.tz)
        if dt < datetime.now(self.tz):
            return "Not saved: that time is in the past. Confirm the day and time again."
        await self.crm.schedule_callback(self.lead_id, dt)
        self._log("schedule_callback", when=dt.isoformat())
        return f"Callback booked for {dt.strftime('%A %d %B at %H:%M')}."

    async def mark_do_not_call(self) -> str:
        await self.crm.mark_do_not_call(self.lead_id)
        self._log("mark_do_not_call")
        return "Marked as do-not-call."
