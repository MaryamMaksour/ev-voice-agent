"""The demo website's server.

    python -m demo.server        # then open http://localhost:8090

It never talks to the models itself. It lists the demo leads, creates a LiveKit room for the call
(with the chosen prompt, voice and STT server in the room's metadata) and hands the browser a token.
The agent worker (started with DEMO_MODE=1) does the rest, and the call record it saves is shown
here afterwards. Listens on 127.0.0.1 only and changes only the demo CRM (see docs/SRS.md 3.10).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from agent import demo_mode
from agent.config import Settings
from agent.prompts import SYSTEM_PROMPT
from crm.port import LeadNotFoundError
from crm.sqlite_adapter import DEMO_LEADS, SQLiteCRM

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
STATIC_FILES = {"app.js": "text/javascript", "app.css": "text/css"}
ROOM_RE = re.compile(r"^lead-[A-Za-z0-9]{1,20}-\d{14}$")
FEMALE, MALE = ("af", "bf"), ("am", "bm")


@dataclass(frozen=True)
class DemoConfig:
    crm_path: str
    calls_dir: str
    livekit_url: str          # where the server (and the agent) reach LiveKit
    livekit_public_url: str   # where the browser reaches it
    api_key: str
    api_secret: str
    options_path: Path
    host: str = "127.0.0.1"
    port: int = 8090

    @classmethod
    def from_env(cls) -> "DemoConfig":
        url = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
        return cls(
            crm_path=os.getenv("CRM_SQLITE_PATH", "data/demo_crm.db"),
            calls_dir=Settings().calls_dir,
            livekit_url=url,
            livekit_public_url=os.getenv("DEMO_LIVEKIT_PUBLIC_URL", url),
            api_key=os.getenv("LIVEKIT_API_KEY", ""),
            api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
            options_path=Path(os.getenv("DEMO_OPTIONS", str(HERE / "options.json"))),
            port=int(os.getenv("DEMO_PORT", "8090")),
        )


class LiveKitGateway:
    """The three LiveKit calls the demo needs. A fake stands in for it in the tests."""

    def __init__(self, url: str, key: str, secret: str) -> None:
        self._url, self._key, self._secret = url, key, secret

    def _http(self) -> str:
        return self._url.replace("wss://", "https://").replace("ws://", "http://")

    async def create_room(self, room: str, metadata: dict) -> None:
        from livekit import api

        async with api.LiveKitAPI(url=self._http(), api_key=self._key, api_secret=self._secret) as lk:
            await lk.room.create_room(api.CreateRoomRequest(
                name=room, empty_timeout=60, departure_timeout=5, metadata=json.dumps(metadata)))

    async def delete_room(self, room: str) -> None:
        from livekit import api

        async with api.LiveKitAPI(url=self._http(), api_key=self._key, api_secret=self._secret) as lk:
            await lk.room.delete_room(api.DeleteRoomRequest(room=room))

    def token(self, room: str, identity: str) -> str:
        from livekit import api

        return (api.AccessToken(self._key, self._secret).with_identity(identity).with_name("Demo caller")
                .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True))
                .to_jwt())


def _get_json(url: str, timeout: float = 0.8):
    """GET `url` and parse the JSON, or None if it does not answer."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (local service URLs only)
            return json.loads(resp.read().decode("utf-8") or "null")
    except Exception:
        return None


def _root(base_url: str) -> str:
    return re.sub(r"/v1/?$", "", base_url.rstrip("/"))


def _voice_info(voice: str) -> dict:
    prefix = voice.split("_", 1)[0]
    kind = "Female" if prefix in FEMALE else "Male" if prefix in MALE else "Other"
    accent = "British" if prefix.startswith("b") else "American"
    return {"id": voice, "label": f"{voice.split('_', 1)[-1].title()} ({kind}, {accent})", "gender": kind.lower()}


def lead_json(lead) -> dict:
    return lead.model_dump(mode="json")


def find_record(calls_dir: str, room: str) -> tuple[str, Path | None]:
    """("done", file) once the call record is saved, "processing" while the post-call check runs, else "waiting"."""
    folder = Path(calls_dir)
    done = [p for p in sorted(folder.glob(f"{room}-*.json")) if not p.name.endswith(".raw.json")]
    if done:
        return "done", done[-1]
    if list(folder.glob(f"{room}-*.raw.json")):
        return "processing", None
    return "waiting", None


def _read_jsonl(path: Path) -> list:
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def call_result(calls_dir: str, room: str) -> dict:
    status, path = find_record(calls_dir, room)
    if path is None:
        return {"status": status}
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "processing"}
    folder = path.with_suffix("")
    keys = ("call_id", "lead_id", "transcript", "tool_calls", "summary", "classification", "duration_seconds",
            "models", "versions", "settings", "errors", "filled_after_call", "ended_by", "close_reason")
    out = {k: rec.get(k) for k in keys}
    out["status"] = "done"
    out["crm_changes"] = _read_jsonl(folder / "crm_changes.jsonl")
    return out


class CallRequest(BaseModel):
    lead_id: str = Field(pattern=r"^[A-Za-z0-9]{1,20}$")
    prompt: str | None = None
    tts_voice: str | None = Field(default=None, pattern=r"^[a-z]{2}_[a-z0-9]{2,20}$")
    stt_id: str | None = Field(default=None, max_length=40)


class PromptCheck(BaseModel):
    prompt: str


def create_app(config: DemoConfig, *, livekit=None, crm: SQLiteCRM | None = None, probe=None, voices=None) -> FastAPI:
    """`livekit`, `crm`, `probe` (STT health) and `voices` (TTS voice list) can be replaced in tests."""
    crm = crm or SQLiteCRM(config.crm_path)
    if not crm.list_leads():
        crm.seed()
    livekit = livekit or LiveKitGateway(config.livekit_url, config.api_key, config.api_secret)
    probe = probe or (lambda base_url: _get_json(_root(base_url) + "/health") is not None)
    settings = Settings()
    app = FastAPI(title="EV voice agent demo", docs_url=None, redoc_url=None, openapi_url=None)

    def lk_origins() -> str:
        host = re.sub(r"^\w+://", "", config.livekit_public_url).rstrip("/")
        return f"ws://{host} wss://{host} http://{host} https://{host}"

    @app.middleware("http")
    async def headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; style-src 'self'; img-src 'self' data:; "
            f"connect-src 'self' {lk_origins()}; media-src 'self' blob:; worker-src 'self' blob:; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["Permissions-Policy"] = "microphone=(self)"
        return resp

    def options() -> dict:
        try:
            return json.loads(config.options_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"stt": [{"id": "default", "label": "Server default", "base_url": None, "model": None}], "voices": []}

    async def stt_options() -> list:
        opts = options().get("stt", [])
        # all servers are asked at once, so a server that is down costs one timeout, not one per server
        up = await asyncio.gather(*(asyncio.to_thread(probe, o.get("base_url") or settings.stt_base_url) for o in opts))
        return [{"id": o["id"], "label": o["label"], "available": bool(ok)} for o, ok in zip(opts, up)]

    async def voice_list() -> list:
        names = voices() if voices else None
        if names is None:
            data = await asyncio.to_thread(_get_json, _root(settings.tts_base_url) + "/v1/audio/voices")
            names = data.get("voices") if isinstance(data, dict) else None
        names = [v for v in (names or options().get("voices", [])) if isinstance(v, str) and re.match(r"^[a-z]{2}_\w+$", v)]
        # English voices only (the product is English only)
        names = [v for v in names if v.split("_", 1)[0] in FEMALE + MALE]
        return [_voice_info(v) for v in names]

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @app.get("/static/{name}")
    async def static(name: str):
        if name not in STATIC_FILES:
            raise HTTPException(404, "Not found")
        return FileResponse(STATIC / name, media_type=STATIC_FILES[name])

    @app.get("/api/config")
    async def get_config():
        voice_names, stt = await asyncio.gather(voice_list(), stt_options())
        return {
            "default_prompt": SYSTEM_PROMPT, "placeholders": list(demo_mode.PLACEHOLDERS),
            "tool_names": list(demo_mode.TOOL_NAMES), "max_prompt_chars": demo_mode.MAX_PROMPT_CHARS,
            "voices": voice_names, "default_voice": settings.tts_voice,
            "stt": stt,
            "models": {"llm": settings.llm_model, "stt": settings.stt_model, "tts": settings.tts_model},
            "livekit_url": config.livekit_public_url,
        }

    @app.get("/api/leads")
    async def leads():
        return [lead_json(x) for x in await asyncio.to_thread(crm.list_leads)]

    @app.post("/api/leads/{lead_id}/reset")
    async def reset_lead(lead_id: str):
        try:
            return lead_json(await asyncio.to_thread(crm.reset_lead, lead_id))
        except LeadNotFoundError:
            raise HTTPException(404, f"No demo lead {lead_id}")

    @app.post("/api/prompt/check")
    async def prompt_check(body: PromptCheck):
        error = demo_mode.validate_template(body.prompt)
        missing = [t for t in demo_mode.TOOL_NAMES if t not in body.prompt]
        return {"ok": error is None, "error": error,
                "warning": ("These tools are not mentioned: " + ", ".join(missing)
                            + ". The agent may not save anything to the CRM.") if missing and error is None else None}

    @app.post("/api/call")
    async def start_call(body: CallRequest):
        try:
            lead = await crm.get_lead(body.lead_id)
        except LeadNotFoundError:
            raise HTTPException(404, f"No demo lead {body.lead_id}")
        meta: dict = {"lead_id": lead.id, "source": "demo"}
        prompt = (body.prompt or "").strip()
        if prompt and prompt != SYSTEM_PROMPT.strip():
            error = demo_mode.validate_template(prompt)
            if error:
                raise HTTPException(400, error)
            meta["prompt"] = prompt
        if body.tts_voice:
            known = {v["id"] for v in await voice_list()}
            if body.tts_voice not in known:
                raise HTTPException(400, f"Unknown voice {body.tts_voice}")
            meta["tts_voice"] = body.tts_voice
        if body.stt_id and body.stt_id != "default":
            option = next((o for o in options().get("stt", []) if o["id"] == body.stt_id), None)
            if option is None or not option.get("base_url"):
                raise HTTPException(400, f"Unknown speech-to-text choice {body.stt_id}")
            if not await asyncio.to_thread(probe, option["base_url"]):
                raise HTTPException(400, f"The speech-to-text server for {option['label']} is not running.")
            meta["stt_base_url"], meta["stt_model"] = option["base_url"], option["model"]
        room = f"lead-{lead.id}-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
        try:
            await livekit.create_room(room, meta)
        except Exception as e:  # LiveKit not running or wrong keys: say so plainly
            raise HTTPException(502, f"Could not reach LiveKit at {config.livekit_url}: {e}")
        return {"room": room, "token": livekit.token(room, "demo-caller"), "url": config.livekit_public_url,
                "lead": lead_json(lead), "overrides": sorted(k for k in meta if k not in ("lead_id", "source"))}

    @app.post("/api/call/{room}/end")
    async def end_call(room: str):
        if not ROOM_RE.match(room):
            raise HTTPException(400, "Bad room name")
        try:
            await livekit.delete_room(room)
        except Exception:
            pass  # already closed
        return {"ok": True}

    @app.get("/api/call/{room}/result")
    async def result(room: str):
        if not ROOM_RE.match(room):
            raise HTTPException(400, "Bad room name")
        data = await asyncio.to_thread(call_result, config.calls_dir, room)
        if data["status"] == "done":
            try:
                data["lead_now"] = lead_json(await crm.get_lead(data["lead_id"]))
            except Exception:
                data["lead_now"] = None
        return JSONResponse(data)

    return app


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()
    if os.getenv("CRM_BACKEND", "sqlite").lower() != "sqlite":
        raise SystemExit("The demo site only works with the demo CRM. Remove CRM_BACKEND (or set it to sqlite).")
    config = DemoConfig.from_env()
    print(f"Demo CRM: {config.crm_path}  (start the agent worker with the same CRM_SQLITE_PATH and DEMO_MODE=1)")
    print(f"Open http://localhost:{config.port}  ({len(DEMO_LEADS)} demo leads)")
    uvicorn.run(create_app(config), host=config.host, port=config.port, log_level="warning")


if __name__ == "__main__":
    main()
