"""The LLM fails mid-call: the agent says sorry, books a callback and ends the call (agent/fallback.py).
Also the quick "is the LLM up?" check made before dialing."""

import asyncio
from datetime import datetime, timedelta, timezone

from livekit.agents import AgentSession, APIConnectOptions
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import openai

from agent import fallback
from agent.config import Settings, llm_reachable, no_think_body
from agent.main import LeadQualifierAgent
from agent.tools import LeadTools
from tests.fake_openai import FakeOpenAI

NOW = datetime(2026, 10, 19, 10, 0, tzinfo=timezone.utc)


async def test_fallback_books_a_callback_and_says_sorry(crm):
    said = []

    async def say(text):
        said.append(text)

    out = await fallback.handle_llm_failure(say=say, crm=crm, lead_id="L001", callback_minutes=60, now=NOW)
    assert said == [fallback.FALLBACK_LINE]
    assert out["said"] and out["callback_booked"] and out["note_added"]
    assert out["callback_at"] == (NOW + timedelta(minutes=60)).isoformat(timespec="seconds")
    lead = await crm.get_lead("L001")
    assert lead.status.value == "callback"
    assert lead.callback_at.replace(tzinfo=lead.callback_at.tzinfo or timezone.utc) == NOW + timedelta(minutes=60)
    assert fallback.NOTE in lead.notes


async def test_fallback_still_books_the_callback_when_speech_is_down(crm):
    def say(text):
        raise RuntimeError("TTS is down too")

    out = await fallback.handle_llm_failure(say=say, crm=crm, lead_id="L002", callback_minutes=30, now=NOW)
    assert not out["said"] and "TTS is down" in out["say_error"]
    assert out["callback_booked"]


async def test_fallback_does_not_wait_forever_to_speak(crm):
    async def say(text):
        await asyncio.sleep(10)

    out = await fallback.handle_llm_failure(say=say, crm=crm, lead_id="L003", callback_minutes=60, now=NOW,
                                            say_timeout=0.1)
    assert not out["said"] and out["callback_booked"]


async def test_fallback_never_raises_when_the_crm_fails():
    class BrokenCRM:
        async def schedule_callback(self, *a):
            raise ConnectionError("CRM down")

        async def add_note(self, *a):
            raise ConnectionError("CRM down")

    async def say(text):
        return None

    out = await fallback.handle_llm_failure(say=say, crm=BrokenCRM(), lead_id="L001", callback_minutes=60, now=NOW)
    assert out["said"] and not out["callback_booked"] and not out["note_added"]


async def test_a_broken_llm_raises_the_error_the_fallback_listens_for(crm):
    """The real AgentSession + OpenAI plugin: an LLM answering HTTP 500 gives an unrecoverable llm_error."""
    errors = []
    with FakeOpenAI(lambda messages: {"http_error": 500}) as fake:
        model = "qwen3.5-9b"
        llm = openai.LLM(base_url=fake.base_url, api_key="x", model=model, extra_body=no_think_body(model, "vllm"))
        lead = await crm.get_lead("L001")
        session = AgentSession(llm=llm, conn_options=SessionConnectOptions(
            llm_conn_options=APIConnectOptions(max_retry=0, timeout=5)))
        session.on("error", lambda ev: errors.append(ev.error))
        async with session:
            await session.start(LeadQualifierAgent(lead, LeadTools(crm, "L001"), Settings()))
            try:
                await session.run(user_input="Hello?")
            except Exception:
                pass  # the run fails; what matters is the error event
            for _ in range(50):
                if errors:
                    break
                await asyncio.sleep(0.05)
    assert any(fallback.is_llm_failure(e) for e in errors), errors


def test_is_llm_failure_ignores_other_errors():
    class E:
        def __init__(self, type_, recoverable):
            self.type, self.recoverable = type_, recoverable

    assert fallback.is_llm_failure(E("llm_error", False))
    assert not fallback.is_llm_failure(E("llm_error", True))     # LiveKit will retry it
    assert not fallback.is_llm_failure(E("tts_error", False))
    assert not fallback.is_llm_failure(None)


def test_llm_reachable(monkeypatch):
    with FakeOpenAI(lambda m: "hi") as fake:
        monkeypatch.setenv("LLM_BASE_URL", fake.base_url)
        assert llm_reachable(Settings()) == (True, "")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")      # nothing listens on port 9
    ok, error = llm_reachable(Settings(), timeout=1)
    assert not ok and error


# ---------- health file for the dashboard (python -m agent.check --write data/health.json --watch 60) ----------
def test_health_file(tmp_path, monkeypatch):
    import json

    from agent.check import run_checks, write_health

    skip = {"stt", "tts", "livekit", "redis"}
    with FakeOpenAI(lambda m: "Hello.") as fake:
        monkeypatch.setenv("LLM_BASE_URL", fake.base_url)
        monkeypatch.setenv("LLM_MODEL", "fake")
        up, _ = run_checks(Settings(), skip)
    assert up["llm"]["ok"] is True and up["llm"]["error"] is None and up["llm"]["seconds"] is not None
    assert up["stt"] == {"ok": None, "seconds": None, "error": None}          # skipped

    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    down, _ = run_checks(Settings(), skip)
    assert down["llm"]["ok"] is False and down["llm"]["error"].startswith("[FAIL]")

    path = tmp_path / "data" / "health.json"
    write_health(str(path), down)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["ts"].endswith("+00:00") and saved["services"]["llm"]["ok"] is False
    assert not (tmp_path / "data" / "health.json.tmp").exists()
