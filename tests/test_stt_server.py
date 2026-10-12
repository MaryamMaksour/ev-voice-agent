"""Contract test: LiveKit's OpenAI STT plugin <-> our faster-whisper server.

The Whisper model is stubbed (no weights needed); this checks the HTTP contract."""

import importlib
import sys
import threading
import time
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("faster_whisper")

from livekit import rtc  # noqa: E402
from livekit.plugins import openai  # noqa: E402


class StubWhisper:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        data = audio.read()
        self.calls.append((len(data), kwargs))
        assert data[:4] == b"RIFF"  # the plugin sends WAV
        return [types.SimpleNamespace(text=" My budget is eight hundred thousand. ")], None


@pytest.fixture
def stt_server(monkeypatch):
    import faster_whisper
    import uvicorn

    monkeypatch.setattr(faster_whisper, "WhisperModel", StubWhisper)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))
    sys.modules.pop("stt_server", None)
    mod = importlib.import_module("stt_server")
    server = uvicorn.Server(uvicorn.Config(mod.app, port=18901, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.02)
    yield mod
    server.should_exit = True
    sys.path.pop(0)


async def test_plugin_transcribes_through_server(stt_server):
    stt = openai.STT(base_url="http://127.0.0.1:18901/v1", api_key="x", model="whisper-1",
                     language="en", use_realtime=False)
    frame = rtc.AudioFrame(data=b"\x00\x00" * 16000, sample_rate=16000, num_channels=1, samples_per_channel=16000)
    event = await stt.recognize([frame])
    assert event.alternatives[0].text == "My budget is eight hundred thousand."
    _, kwargs = stt_server.model.calls[0]
    assert kwargs["language"] == "en"
