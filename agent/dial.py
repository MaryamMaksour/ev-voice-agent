"""Phone a lead through 3CX: creates the call room; the agent joins, dials, and talks once the lead answers.

    python -m agent.dial --lead L003                  # the number saved in the CRM
    python -m agent.dial --lead L003 --number +9715…  # another number (e.g. your own, to test)

Needs, in .env: LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET, SIP_OUTBOUND_TRUNK_ID (the LiveKit
outbound trunk that points at 3CX), and the agent worker running (python -m agent.main dev).
The line result (answered / busy / no answer / ...) is saved in the call record in data/calls/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from livekit import api

from crm import make_crm

from .compliance import check_dial_with, record_blocked
from .config import Settings
from .telephony import mask_number


async def start_call(lead_id: str, number: str | None) -> str:
    settings = Settings()
    if not settings.sip_outbound_trunk_id:
        raise SystemExit("SIP_OUTBOUND_TRUNK_ID is not set in .env: create the 3CX outbound trunk in LiveKit first.")
    lead = await make_crm().get_lead(lead_id)
    decision = check_dial_with(settings, lead)
    if not decision.ok:
        record_blocked(settings.blocked_dir, lead_id=lead_id, number=number or lead.phone, rule=decision.rule,
                       detail=decision.detail, source="dial")
        raise SystemExit(f"Not calling {lead_id}: {decision.detail} (rule: {decision.rule}).")
    meta = {"lead_id": lead_id, **({"phone_number": number} if number else {"dial": True})}
    room = f"lead-{lead_id}-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    async with api.LiveKitAPI() as lk:  # reads LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
        await lk.room.create_room(api.CreateRoomRequest(name=room, empty_timeout=120, metadata=json.dumps(meta)))
    print(f"calling {lead.name} ({lead_id}) on {mask_number(number or lead.phone)} in room {room}")
    print(f"the result will be in {settings.calls_dir}/ when the call ends")
    return room


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lead", required=True, help="lead id, e.g. L003")
    p.add_argument("--number", help="call this number instead of the one in the CRM")
    args = p.parse_args()
    try:
        asyncio.run(start_call(args.lead, args.number))
    except SystemExit as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
