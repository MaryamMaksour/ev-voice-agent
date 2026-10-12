"""Check every service the agent needs, and say which one is broken.

    python -m agent.check            # LLM (Ollama), STT, TTS
    python -m agent.check --play     # also play a test sentence on your speakers
"""

from __future__ import annotations

import argparse
import io
import math
import struct
import time
import wave

import httpx
from dotenv import load_dotenv

from .config import Settings, no_think_body

OK, FAIL = "[ OK ]", "[FAIL]"


def _silence_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(200 * math.sin(i / 10))) for i in range(int(seconds * rate))))
    return buf.getvalue()


def check_llm(s: Settings, client: httpx.Client) -> bool:
    print(f"\nLLM  {s.llm_base_url}  model={s.llm_model}")
    try:
        r = client.get(f"{s.llm_base_url}/models")
        if r.status_code == 200:
            names = [m.get("id") for m in r.json().get("data", [])]
            if names and s.llm_model not in names:
                print(f"{FAIL} model '{s.llm_model}' is not pulled. Available: {names}")
                print(f"       fix: ollama pull {s.llm_model}")
                return False
        t0 = time.perf_counter()
        r = client.post(
            f"{s.llm_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {s.llm_api_key}"},
            json={"model": s.llm_model, "messages": [{"role": "user", "content": "Say hello in five words. /no_think"}],
                  "max_tokens": 40, **no_think_body(s.llm_model)},
            timeout=180,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        print(f"{OK} replied in {time.perf_counter() - t0:.1f}s: {text.strip()[:80]!r}")
        if time.perf_counter() - t0 > 10:
            print("       note: slow (model partly on CPU). The agent will pause this long before each reply.")
        return True
    except httpx.ConnectError:
        print(f"{FAIL} cannot connect. Is Ollama running? fix: start the Ollama app, or `ollama serve`")
    except Exception as e:
        print(f"{FAIL} {e!r}")
    return False


def check_stt(s: Settings, client: httpx.Client) -> bool:
    print(f"\nSTT  {s.stt_base_url}")
    try:
        t0 = time.perf_counter()
        r = client.post(
            f"{s.stt_base_url}/audio/transcriptions",
            files={"file": ("test.wav", _silence_wav(), "audio/wav")},
            data={"model": s.stt_model, "language": "en"},
            timeout=120,
        )
        r.raise_for_status()
        print(f"{OK} responded in {time.perf_counter() - t0:.1f}s")
        return True
    except httpx.ConnectError:
        print(f"{FAIL} cannot connect. fix: docker compose -f services/docker-compose.local.yml --env-file .env up -d stt")
        print("       (first start downloads the whisper model; check: docker compose -f services/docker-compose.local.yml logs stt)")
    except Exception as e:
        print(f"{FAIL} {e!r}")
    return False


def check_tts(s: Settings, client: httpx.Client) -> bytes | None:
    print(f"\nTTS  {s.tts_base_url}  voice={s.tts_voice}")
    try:
        t0 = time.perf_counter()
        r = client.post(
            f"{s.tts_base_url}/audio/speech",
            json={"model": s.tts_model, "voice": s.tts_voice, "response_format": "wav",
                  "input": "Hi, this is Alex from Horizon Realty. Can you hear me?"},
            timeout=120,
        )
        r.raise_for_status()
        if len(r.content) < 1000:
            print(f"{FAIL} returned almost no audio ({len(r.content)} bytes)")
            return None
        print(f"{OK} {len(r.content) // 1024} KB of audio in {time.perf_counter() - t0:.1f}s")
        return r.content
    except httpx.ConnectError:
        print(f"{FAIL} cannot connect. fix: docker compose -f services/docker-compose.local.yml --env-file .env up -d tts")
    except Exception as e:
        print(f"{FAIL} {e!r}")
    return None


def play(wav_bytes: bytes) -> None:
    import array

    import sounddevice as sd

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        rate, data = w.getframerate(), array.array("h", w.readframes(w.getnframes()))
    dev = sd.query_devices(kind="output")
    print(f"\nplaying on: {dev['name']}  (wrong device? python -m agent.main console --list-devices)")
    sd.play(data, rate, blocking=True)
    print("did you hear 'Hi, this is Alex from Horizon Realty'? If not, check volume / output device.")


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--play", action="store_true", help="play a test sentence on the speakers")
    args = p.parse_args()
    s = Settings()
    with httpx.Client(timeout=30, trust_env=False) as client:
        llm = check_llm(s, client)
        stt = check_stt(s, client)
        audio = check_tts(s, client)
    if args.play and audio:
        try:
            play(audio)
        except Exception as e:
            print(f"{FAIL} could not play audio: {e!r}")
    print("\nsummary:", "LLM", OK if llm else FAIL, "| STT", OK if stt else FAIL, "| TTS", OK if audio else FAIL)
    if llm and not (stt and audio):
        print("text mode still works now:  python -m agent.main console --text")


if __name__ == "__main__":
    main()
