"""Generate the simulated caller's voice with Kokoro (run once, on the box with TTS).

    python -m loadtest.make_audio            # writes loadtest/audio/01.wav ... 07.wav
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from .script import CALLER_LINES

AUDIO_DIR = Path(__file__).parent / "audio"


def main() -> None:
    load_dotenv()
    base = os.getenv("TTS_BASE_URL", "http://localhost:8880/v1")
    voice = os.getenv("SIM_CALLER_VOICE", "am_michael")  # different voice from the agent
    AUDIO_DIR.mkdir(exist_ok=True)
    with httpx.Client(timeout=60) as client:
        for i, line in enumerate(CALLER_LINES, 1):
            resp = client.post(
                f"{base}/audio/speech",
                json={"model": "kokoro", "input": line, "voice": voice, "response_format": "wav"},
            )
            resp.raise_for_status()
            path = AUDIO_DIR / f"{i:02d}.wav"
            path.write_bytes(resp.content)
            print(f"{path}  {line!r}")


if __name__ == "__main__":
    main()
