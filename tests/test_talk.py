"""Terminal client (python -m agent.talk): arguments, room and token. The live call itself is run
against a real LiveKit server in the dry run described in the README."""

import json
import re

import pytest
from livekit import api

from agent import talk


def test_args_and_room_name():
    args = talk.parse_args(["--lead", "L007", "--input-device", "2", "--output-device", "Headphones",
                            "--seconds", "30"])
    assert args.lead == "L007" and args.input_device == 2 and args.output_device == "Headphones"
    assert args.seconds == 30 and args.audio_file is None
    assert re.fullmatch(r"lead-L007-\d{14}", talk.room_name("L007"))


def test_token_can_join_and_publish(monkeypatch):
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "a-long-secret-for-tests-0123456789abcdef")
    token = talk.make_token("lead-L001-1", "terminal-caller")
    claims = api.TokenVerifier("devkey", "a-long-secret-for-tests-0123456789abcdef").verify(token)
    assert claims.identity == "terminal-caller"
    assert claims.video.room == "lead-L001-1" and claims.video.room_join and claims.video.can_publish


async def test_room_is_created_with_the_lead(monkeypatch):
    created = {}

    class FakeRooms:
        async def create_room(self, req):
            created["req"] = req

    class FakeAPI:
        def __init__(self, url=None):
            created["url"] = url
            self.room = FakeRooms()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(api, "LiveKitAPI", FakeAPI)
    await talk.create_room("ws://10.0.0.5:7880", "lead-L003-1", "L003")
    assert created["url"] == "http://10.0.0.5:7880"  # the server API is plain HTTP on the same port
    assert json.loads(created["req"].metadata)["lead_id"] == "L003"


def test_speaker_buffer_pads_with_silence():
    s = talk.Speaker()
    s.push(b"\x01\x02" * 3)
    assert s.pull(8) == b"\x01\x02" * 3 + b"\0\0"
    assert s.pull(4) == b"\0" * 4


def test_missing_keys_stop_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.setattr(talk, "load_dotenv", lambda: None)
    with pytest.raises(SystemExit, match="LIVEKIT_API_KEY"):
        talk.main(["--lead", "L001"])
