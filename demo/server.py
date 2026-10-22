"""The demo website: pick a lead, talk to the agent in the browser, watch the lead change.

    python -m demo.server        # then open http://localhost:8090

It never talks to the models. It lists the demo leads, hands the browser a LiveKit token for a room
named lead-<ID>-<time> (the agent already knows which lead that is), and shows the call record the
agent saves afterwards. Demo CRM only; listens on 127.0.0.1 only (docs/SRS.md 3.10).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from crm.port import LeadNotFoundError
from crm.sqlite_adapter import SQLiteCRM

PAGE = Path(__file__).with_name("index.html")
ROOM = re.compile(r"^lead-[A-Za-z0-9]{1,20}-\d{14}$")
PORT = 8090


class LeadRef(BaseModel):
    lead_id: str


def create_app(*, crm: SQLiteCRM, calls_dir: str, livekit_url: str, api_key: str, api_secret: str) -> FastAPI:
    app = FastAPI(title="Voice agent demo", docs_url=None, redoc_url=None)

    def token_for(room: str) -> str:
        from livekit import api

        return (api.AccessToken(api_key, api_secret).with_identity("demo-caller").with_name("Demo caller")
                .with_grants(api.VideoGrants(room_join=True, room=room)).to_jwt())

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(PAGE)

    @app.get("/api/leads")
    def leads() -> list[dict]:
        return [lead.model_dump(mode="json") for lead in crm.list_leads()]

    @app.post("/api/call")
    async def start_call(body: LeadRef) -> dict:
        try:
            await crm.get_lead(body.lead_id)
        except LeadNotFoundError:
            raise HTTPException(404, "That lead does not exist.")
        room = f"lead-{body.lead_id}-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
        return {"room": room, "token": token_for(room), "url": livekit_url}

    @app.get("/api/call/{room}")
    def call_result(room: str) -> dict:
        """The saved record of a finished call, or {"done": false} while the agent is still finishing."""
        if not ROOM.match(room):
            raise HTTPException(400, "That is not a call room.")
        folder = Path(calls_dir)
        # the agent names the record <room>-<start time>.json; <room>-<...>.raw.json is "still being processed"
        for path in folder.glob(f"{room}-*.json"):
            if path.name.endswith(".raw.json"):
                continue
            record = _read_json(path)
            if record is None:
                continue
            extraction = record.get("extraction") or {}
            return {"done": True, "classification": record.get("classification"),
                    "summary": extraction.get("summary"), "ended_by": record.get("ended_by"),
                    "transcript": record.get("transcript") or [], "error": record.get("error")}
        return {"done": False}

    @app.post("/api/reset")
    def reset(body: LeadRef) -> dict:
        try:
            return crm.reset_lead(body.lead_id).model_dump(mode="json")
        except LeadNotFoundError:
            raise HTTPException(404, "That lead does not exist.")

    return app


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()
    if os.getenv("CRM_BACKEND", "sqlite").lower() != "sqlite":
        raise SystemExit("The demo website only works with the demo CRM: set CRM_BACKEND=sqlite.")
    app = create_app(
        crm=SQLiteCRM(os.getenv("CRM_SQLITE_PATH", "data/crm.db")),
        calls_dir=os.getenv("CALLS_DIR", "data/calls"),
        # where the browser reaches LiveKit: through the SSH tunnel it is always localhost
        livekit_url=os.getenv("DEMO_LIVEKIT_URL", "ws://localhost:7880"),
        api_key=os.environ["LIVEKIT_API_KEY"],
        api_secret=os.environ["LIVEKIT_API_SECRET"],
    )
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
