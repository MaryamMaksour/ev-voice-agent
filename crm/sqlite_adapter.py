"""SQLite CRM adapter: a stand-in for Zoho during the demo.

    python -m crm.sqlite_adapter --seed     # create data/crm.db with demo leads
    python -m crm.sqlite_adapter --show     # print all leads
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .port import Lead, LeadNotFoundError, LeadStatus, LeadUpdate

DEFAULT_DB = Path("data/crm.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS leads (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
"""

DEMO_LEADS = [
    ("L001", "Sarah Johnson", "+15550100001", "Marina Heights"),
    ("L002", "Michael Chen", "+15550100002", "Palm Gardens Villas"),
    ("L003", "Emily Davis", "+15550100003", "Downtown Lofts"),
    ("L004", "James Wilson", "+15550100004", "Marina Heights"),
    ("L005", "Olivia Brown", "+15550100005", "Lakeside Townhouses"),
    ("L006", "David Miller", "+15550100006", "Business Bay Offices"),
    ("L007", "Sophia Garcia", "+15550100007", "Palm Gardens Villas"),
    ("L008", "Daniel Martinez", "+15550100008", "Downtown Lofts"),
    ("L009", "Ava Anderson", "+15550100009", "Lakeside Townhouses"),
    ("L010", "Matthew Thomas", "+15550100010", "Marina Heights"),
    ("L011", "Isabella Taylor", "+15550100011", "Hillside Plots"),
    ("L012", "Ethan Moore", "+15550100012", "Downtown Lofts"),
    ("L013", "Mia Jackson", "+15550100013", "Palm Gardens Villas"),
    ("L014", "Alexander White", "+15550100014", "Business Bay Offices"),
    ("L015", "Charlotte Harris", "+15550100015", "Marina Heights"),
    ("L016", "Benjamin Clark", "+15550100016", "Lakeside Townhouses"),
    ("L017", "Amelia Lewis", "+15550100017", "Downtown Lofts"),
    ("L018", "Lucas Walker", "+15550100018", "Hillside Plots"),
    ("L019", "Harper Hall", "+15550100019", "Palm Gardens Villas"),
    ("L020", "Henry Young", "+15550100020", "Marina Heights"),
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SQLiteCRM:
    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    # -- sync helpers (run in a thread so the event loop never blocks) --

    def _get(self, lead_id: str) -> Lead:
        with self._connect() as conn:
            row = conn.execute("SELECT data FROM leads WHERE id = ?", (lead_id,)).fetchone()
        if row is None:
            raise LeadNotFoundError(lead_id)
        return Lead.model_validate_json(row[0])

    def _put(self, lead: Lead) -> Lead:
        lead.updated_at = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO leads (id, data) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET data = excluded.data",
                (lead.id, lead.model_dump_json()),
            )
        return lead

    def _modify(self, lead_id: str, **changes) -> Lead:
        lead = self._get(lead_id)
        return self._put(lead.model_copy(update=changes))

    def list_leads(self) -> list[Lead]:
        with self._connect() as conn:
            rows = conn.execute("SELECT data FROM leads ORDER BY id").fetchall()
        return [Lead.model_validate_json(r[0]) for r in rows]

    def seed(self, reset: bool = True) -> None:
        if reset:
            with self._connect() as conn:
                conn.execute("DELETE FROM leads")
        for lead_id, name, phone, project in DEMO_LEADS:
            self._put(Lead(id=lead_id, name=name, phone=phone, project_interest=project))

    def reset_lead(self, lead_id: str) -> Lead:
        """Put one demo lead back to its seeded values (no answers, no notes, callable again)."""
        for demo_id, name, phone, project in DEMO_LEADS:
            if demo_id == lead_id:
                return self._put(Lead(id=demo_id, name=name, phone=phone, project_interest=project))
        raise LeadNotFoundError(lead_id)

    # -- CRMClient --

    async def get_lead(self, lead_id: str) -> Lead:
        return await asyncio.to_thread(self._get, lead_id)

    async def update_lead(self, lead_id: str, update: LeadUpdate) -> Lead:
        changes = {k: v for k, v in update if v is not None}
        return await asyncio.to_thread(self._modify, lead_id, **changes)

    async def schedule_callback(self, lead_id: str, when: datetime) -> Lead:
        return await asyncio.to_thread(
            self._modify, lead_id, callback_at=when, status=LeadStatus.CALLBACK
        )

    async def mark_do_not_call(self, lead_id: str) -> Lead:
        return await asyncio.to_thread(
            self._modify, lead_id, do_not_call=True, status=LeadStatus.DO_NOT_CALL
        )

    async def add_note(self, lead_id: str, note: str) -> Lead:
        def _add() -> Lead:
            lead = self._get(lead_id)
            return self._put(lead.model_copy(update={"notes": [*lead.notes, note]}))

        return await asyncio.to_thread(_add)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo SQLite CRM")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--seed", action="store_true", help="reset and load demo leads")
    parser.add_argument("--show", action="store_true", help="print all leads")
    parser.add_argument("--lead", help="print one lead as JSON")
    args = parser.parse_args()

    crm = SQLiteCRM(args.db)
    if args.seed:
        crm.seed()
        print(f"Seeded {len(DEMO_LEADS)} leads into {args.db}")
    if args.lead:
        print(json.dumps(crm._get(args.lead).model_dump(mode="json"), indent=2))
    if args.show:
        for lead in crm.list_leads():
            cb = lead.callback_at.strftime("%Y-%m-%d %H:%M") if lead.callback_at else "-"
            print(
                f"{lead.id}  {lead.name:<18} {lead.status.value:<14} "
                f"budget={lead.budget or '-'} area={lead.preferred_area or '-'} "
                f"type={lead.property_type.value if lead.property_type else '-'} "
                f"interest={lead.interest_level.value if lead.interest_level else '-'} "
                f"callback={cb} notes={len(lead.notes)}"
            )


if __name__ == "__main__":
    main()
