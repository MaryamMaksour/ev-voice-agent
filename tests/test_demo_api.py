"""The demo website's API, with LiveKit and the speech servers faked."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from agent import demo_mode
from agent.prompts import SYSTEM_PROMPT
from crm.port import LeadUpdate
from crm.sqlite_adapter import SQLiteCRM
from demo import server


class FakeLiveKit:
    def __init__(self):
        self.rooms, self.deleted, self.fail = {}, [], False

    async def create_room(self, room, metadata):
        if self.fail:
            raise RuntimeError("connection refused")
        self.rooms[room] = metadata

    async def delete_room(self, room):
        self.deleted.append(room)

    def token(self, room, identity):
        return f"token-for-{room}-{identity}"


@pytest.fixture
def env(tmp_path):
    opts = tmp_path / "options.json"
    opts.write_text(json.dumps({
        "stt": [{"id": "default", "label": "Default", "base_url": None, "model": None},
                {"id": "parakeet", "label": "Parakeet", "base_url": "http://stt-a/v1", "model": "parakeet"},
                {"id": "whisper", "label": "Whisper", "base_url": "http://stt-b/v1", "model": "small"}],
        "voices": ["af_heart", "am_adam"]}))
    calls = tmp_path / "calls"
    calls.mkdir()
    config = server.DemoConfig(crm_path=str(tmp_path / "demo.db"), calls_dir=str(calls),
                               livekit_url="ws://localhost:7880", livekit_public_url="ws://localhost:7880",
                               api_key="k", api_secret="s", options_path=opts)
    lk = FakeLiveKit()
    app = server.create_app(config, livekit=lk, crm=SQLiteCRM(config.crm_path),
                            probe=lambda url: "stt-a" in url, voices=lambda: ["af_heart", "am_adam", "xx_other"])
    return TestClient(app), lk, calls, config


def test_leads_are_listed_and_seeded(env):
    client, *_ = env
    leads = client.get("/api/leads").json()
    assert len(leads) == 20 and leads[2]["id"] == "L003" and leads[2]["status"] == "new"


def test_config_has_prompt_voices_and_stt_availability(env):
    client, *_ = env
    cfg = client.get("/api/config").json()
    assert cfg["default_prompt"] == SYSTEM_PROMPT
    assert [v["id"] for v in cfg["voices"]] == ["af_heart", "am_adam"]  # English voices only
    assert {v["id"]: v["gender"] for v in cfg["voices"]} == {"af_heart": "female", "am_adam": "male"}
    avail = {o["id"]: o["available"] for o in cfg["stt"]}
    assert avail == {"default": False, "parakeet": True, "whisper": False}


def test_reset_lead(env):
    client, _lk, _calls, config = env
    asyncio.run(SQLiteCRM(config.crm_path).update_lead("L003", LeadUpdate(status="interested", budget=500000)))
    assert client.get("/api/leads").json()[2]["status"] == "interested"
    assert client.post("/api/leads/L003/reset").json()["status"] == "new"
    assert client.post("/api/leads/L999/reset").status_code == 404


def test_prompt_check(env):
    client, *_ = env
    assert client.post("/api/prompt/check", json={"prompt": SYSTEM_PROMPT}).json()["ok"] is True
    bad = client.post("/api/prompt/check", json={"prompt": "Hi {nope}"}).json()
    assert bad["ok"] is False and "{nope}" in bad["error"]
    warn = client.post("/api/prompt/check", json={"prompt": "Call {name}."}).json()
    assert warn["ok"] and "update_lead" in warn["warning"]


def test_start_call_with_defaults_sends_only_lead_and_source(env):
    client, lk, *_ = env
    body = client.post("/api/call", json={"lead_id": "L003", "prompt": SYSTEM_PROMPT}).json()
    assert body["room"].startswith("lead-L003-") and body["token"].startswith("token-for-lead-L003-")
    assert body["lead"]["name"] == "Emily Davis" and body["overrides"] == []
    assert lk.rooms[body["room"]] == {"lead_id": "L003", "source": "demo"}


def test_start_call_with_overrides(env):
    client, lk, *_ = env
    prompt = "You are {agent_name}. Call {name}. Use update_lead and end_call."
    body = client.post("/api/call", json={"lead_id": "L001", "prompt": prompt, "tts_voice": "am_adam",
                                          "stt_id": "parakeet"}).json()
    meta = lk.rooms[body["room"]]
    assert meta == {"lead_id": "L001", "source": "demo", "prompt": prompt, "tts_voice": "am_adam",
                    "stt_base_url": "http://stt-a/v1", "stt_model": "parakeet"}
    # and the agent accepts exactly this metadata in demo mode
    got = demo_mode.overrides_from_metadata([json.dumps(meta)], on=True)
    assert got["prompt"] == prompt and got["tts_voice"] == "am_adam" and got["stt_model"] == "parakeet"
    assert sorted(body["overrides"]) == ["prompt", "stt_base_url", "stt_model", "tts_voice"]


@pytest.mark.parametrize("payload,fragment", [
    ({"lead_id": "L999"}, "No demo lead"),
    ({"lead_id": "L003", "prompt": "Hi {nope}"}, "{nope}"),
    ({"lead_id": "L003", "tts_voice": "xx_other"}, "Unknown voice"),
    ({"lead_id": "L003", "stt_id": "whisper"}, "not running"),
    ({"lead_id": "L003", "stt_id": "nonsense"}, "Unknown speech-to-text"),
])
def test_start_call_refusals(env, payload, fragment):
    client, lk, *_ = env
    resp = client.post("/api/call", json=payload)
    assert resp.status_code in (400, 404) and fragment in resp.json()["detail"]
    assert lk.rooms == {}


def test_start_call_bad_lead_id_shape_is_refused(env):
    client, *_ = env
    assert client.post("/api/call", json={"lead_id": "../x"}).status_code == 422


def test_livekit_down_gives_a_plain_message(env):
    client, lk, *_ = env
    lk.fail = True
    resp = client.post("/api/call", json={"lead_id": "L003"})
    assert resp.status_code == 502 and "Could not reach LiveKit" in resp.json()["detail"]


def test_end_call_and_room_name_check(env):
    client, lk, *_ = env
    assert client.post("/api/call/lead-L003-20261021100000/end").json() == {"ok": True}
    assert lk.deleted == ["lead-L003-20261021100000"]
    assert client.post("/api/call/evil/end").status_code == 400
    assert client.get("/api/call/evil/result").status_code == 400


def test_result_goes_waiting_processing_done(env):
    client, _lk, calls, _ = env
    room = "lead-L003-20261021100000"
    assert client.get(f"/api/call/{room}/result").json() == {"status": "waiting"}
    (calls / f"{room}-2026-10-21T100001.raw.json").write_text("{}")
    assert client.get(f"/api/call/{room}/result").json() == {"status": "processing"}
    call_id = f"{room}-2026-10-21T100001"
    (calls / f"{call_id}.json").write_text(json.dumps({
        "call_id": call_id, "lead_id": "L003", "transcript": [{"role": "agent", "text": "Hi"}],
        "tool_calls": [{"name": "update_lead", "arguments": "{}", "output": "Saved: status.", "is_error": False}],
        "summary": "Interested.", "models": {"llm": "q"}, "versions": {"prompt": "abc"}}))
    (calls / call_id).mkdir()
    change = {"action": "update_lead", "ok": True, "changed": {"status": {"before": "new", "after": "interested"}}}
    (calls / call_id / "crm_changes.jsonl").write_text(json.dumps(change) + "\n")
    done = client.get(f"/api/call/{room}/result").json()
    assert done["status"] == "done" and done["summary"] == "Interested."
    assert done["crm_changes"][0]["changed"]["status"]["after"] == "interested"
    assert done["lead_now"]["id"] == "L003"


def test_static_whitelist_and_headers(env):
    client, *_ = env
    assert client.get("/static/server.py").status_code == 404
    assert client.get("/static/secret.txt").status_code == 404
    resp = client.get("/api/leads")
    assert "script-src 'self' https://cdn.jsdelivr.net" in resp.headers["content-security-policy"]
    assert "ws://localhost:7880" in resp.headers["content-security-policy"]
    assert resp.headers["permissions-policy"] == "microphone=(self)"


def test_empty_voice_is_not_an_error(env):
    client, lk, *_ = env
    resp = client.post("/api/call", json={"lead_id": "L003", "tts_voice": "", "stt_id": None})
    assert resp.status_code == 200 and "tts_voice" not in lk.rooms[resp.json()["room"]]


def test_unexpected_voice_list_falls_back_to_the_built_in_list(env, tmp_path):
    _client, lk, calls, config = env
    # the speech server answered, but not with plain voice names (e.g. a list of objects)
    app = server.create_app(config, livekit=lk, crm=SQLiteCRM(config.crm_path), probe=lambda url: True,
                            voices=lambda: [{"name": "af_heart"}, {"name": "am_adam"}])
    cfg = TestClient(app).get("/api/config").json()
    assert [v["id"] for v in cfg["voices"]] == ["af_heart", "am_adam"]  # from options.json
