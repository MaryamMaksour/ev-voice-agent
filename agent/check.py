"""Check every service the agent needs, and say which one is broken.

    python -m agent.check            # LLM (vLLM or Ollama), STT, TTS, LiveKit, Redis
    python -m agent.check --play     # also play a test sentence on your speakers
    python -m agent.check --calls    # where the time goes in your recent calls
    python -m agent.check --skip livekit,redis   # text-mode testing: only the models

Exit code 1 if a needed service fails, so the deploy (deploy/deploy.sh) can roll back.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import socket
import struct
import sys
import time
import wave
from urllib.parse import urlparse

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
    backend = s.llm_backend
    print(f"\nLLM  {s.llm_base_url}  model={s.llm_model}  backend={backend}")
    try:
        r = client.get(f"{s.llm_base_url}/models", headers={"Authorization": f"Bearer {s.llm_api_key}"})
        if r.status_code == 200:
            models = r.json().get("data", [])
            names = [m.get("id") for m in models]
            if names and s.llm_model not in names:
                print(f"{FAIL} {backend} serves {names}, not '{s.llm_model}'.")
                if backend == "ollama":
                    print(f"       fix: ollama pull {s.llm_model}   (or set LLM_MODEL in .env to one of the above)")
                else:
                    print("       fix: set LLM_MODEL in .env to vLLM's --served-model-name")
                return False
            for m in models:
                if m.get("id") == s.llm_model and m.get("max_model_len"):
                    print(f"       serving {m.get('root')}  max context {m.get('max_model_len')} tokens")
        t0 = time.perf_counter()
        r = client.post(
            f"{s.llm_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {s.llm_api_key}"},
            json={"model": s.llm_model, "messages": [{"role": "user", "content": "Say hello in five words."}],
                  "max_tokens": 40, **no_think_body(s.llm_model, backend)},
            timeout=300,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"] or ""
        print(f"{OK} replied in {time.perf_counter() - t0:.1f}s: {text.strip()[:80]!r}")
        if "<think>" in text or "</think>" in text:
            print("       note: the model is still thinking: " + ("update Ollama" if backend == "ollama"
                                                                    else "check the vLLM --reasoning-parser flag"))
        if time.perf_counter() - t0 > 3:
            print("       note: slow. The first request loads the model (Ollama) or CUDA graphs (vLLM); try again.")
        return True
    except httpx.ConnectError:
        if backend == "ollama":
            print(f"{FAIL} cannot connect. Is Ollama running? fix: start the Ollama app, then ollama pull {s.llm_model}")
        else:
            print(f"{FAIL} cannot connect. Is vLLM running? fix: docker compose -f services/docker-compose.yml "
                  "--env-file .env up -d vllm   (first start downloads the model: docker compose ... logs -f vllm)")
    except httpx.HTTPStatusError as e:
        print(f"{FAIL} {backend} answered HTTP {e.response.status_code}: {e.response.text.strip()[:300]}")
        if backend == "ollama" and e.response.status_code >= 500:
            print("       fix: Ollama can't run the model. Restart Ollama; if that doesn't help, reinstall or "
                  "update it from ollama.com")
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
                  "input": f"Hi, this is {s.agent_name} from {s.company_name}. Can you hear me?"},
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


def check_livekit(client: httpx.Client) -> bool:
    """The self-hosted LiveKit server answers, and the API key/secret in .env are accepted."""
    import asyncio
    import os

    url = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    print(f"\nLiveKit  {url}")
    http = url.replace("wss://", "https://").replace("ws://", "http://")
    try:
        client.get(http, timeout=5)
    except httpx.ConnectError:
        print(f"{FAIL} cannot connect. fix: docker compose -f services/docker-compose.yml --env-file .env up -d livekit")
        print("       from a laptop: LIVEKIT_URL must be ws://<server-ip>:7880 and ports 7880, 7881/tcp, 7882/udp open")
        return False
    except Exception:
        pass  # any HTTP answer means the server is up

    async def rooms() -> int:
        from livekit import api

        async with api.LiveKitAPI(url=http) as lk:
            return len((await lk.room.list_rooms(api.ListRoomsRequest())).rooms)

    try:
        n = asyncio.run(rooms())
        print(f"{OK} API keys accepted, {n} room(s) open")
        return True
    except Exception as e:
        print(f"{FAIL} server is up but the API call failed: {e!r}")
        print("       fix: LIVEKIT_API_KEY / LIVEKIT_API_SECRET in .env must match the server's LIVEKIT_KEYS")
    return False


def check_redis(s: Settings) -> bool | None:
    """Redis is only needed by LiveKit Egress (call recording). None = recording is off, not checked."""
    if not s.recording_enabled:
        print("\nRedis  not needed (RECORDING_ENABLED=0)")
        return None
    url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    print(f"\nRedis  {url}  (LiveKit Egress: call recording)")
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 6379), timeout=3) as sock:
            sock.sendall(b"PING\r\n")
            reply = sock.recv(64)
        if reply.startswith(b"+PONG") or reply.startswith(b"-NOAUTH"):
            print(f"{OK} reachable")
            return True
        print(f"{FAIL} unexpected reply {reply!r}")
    except OSError as e:
        print(f"{FAIL} {e!r}  fix: docker compose -f services/docker-compose.yml --env-file .env up -d redis")
    return False


def play(wav_bytes: bytes) -> None:
    import array

    import sounddevice as sd

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        rate, data = w.getframerate(), array.array("h", w.readframes(w.getnframes()))
    dev = sd.query_devices(kind="output")
    print(f"\nplaying on: {dev['name']}  (wrong device? python -m agent.main console --list-devices)")
    sd.play(data, rate, blocking=True)
    print("did you hear the test sentence? If not, check volume / output device.")


def latency_report(calls_dir: str, last: int = 5) -> None:
    """Average time per stage for the most recent calls, and the slowest stage."""
    import json
    from pathlib import Path

    files = sorted(Path(calls_dir).glob("*.json"), key=lambda p: p.stat().st_mtime)[-last:]
    if not files:
        print(f"no call records in {calls_dir} yet - make a call first")
        return
    labels = {
        "eou": "waiting for you to finish + speech-to-text",
        "llm_ttft": f"LLM first word ({Settings().llm_backend})",
        "tts_ttfb": "text-to-speech first audio",
    }
    print(f"\naverage time per reply, last {len(files)} call(s):")
    totals: dict[str, list[float]] = {k: [] for k in labels}
    for f in files:
        m = json.loads(f.read_text()).get("metrics", {})
        if not m.get("turns"):
            continue
        for k in labels:
            totals[k].append(m[k]["mean"])
    if not totals["eou"]:
        print("no completed turns recorded (text mode has no audio timings)")
        return
    avg = {k: sum(v) / len(v) for k, v in totals.items()}
    for k, label in labels.items():
        bar = "#" * max(1, int(avg[k] * 10))
        print(f"  {label:<45} {avg[k]:5.2f}s  {bar}")
    print(f"  {'total (you stop -> agent starts)':<45} {sum(avg.values()):5.2f}s")
    slowest = max(avg, key=avg.get)
    print(f"\nslowest: {labels[slowest]}")
    if slowest == "llm_ttft" and Settings().llm_backend == "ollama":
        print("  Ollama is slow: the model probably doesn't fit in GPU memory and runs partly on the CPU.")
        print("  Fine for testing; real-time speed needs the vLLM server.")
    elif slowest == "llm_ttft":
        print("  vLLM is slow: check GPU memory (nvidia-smi), give it more with VLLM_GPU_UTIL, or")
        print("  lower VLLM_MAX_MODEL_LEN so more calls fit in the KV cache.")
    elif slowest == "eou":
        print("  speech-to-text is slow: use STT_ENGINE=parakeet (fast on CPU).")
    else:
        print("  TTS is slow: make sure the Kokoro container is running and not starved of CPU.")


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--play", action="store_true", help="play a test sentence on the speakers")
    p.add_argument("--calls", action="store_true", help="show where the time goes in recent calls")
    p.add_argument("--skip", default="", help="comma-separated services not to check: llm,stt,tts,livekit,redis")
    args = p.parse_args()
    s = Settings()
    if args.calls:
        latency_report(s.calls_dir)
        return
    skip = {x.strip().lower() for x in args.skip.split(",") if x.strip()}
    results: dict[str, bool | None] = {}
    audio = None
    with httpx.Client(timeout=30, trust_env=False) as client:
        results["LLM"] = check_llm(s, client) if "llm" not in skip else None
        results["STT"] = check_stt(s, client) if "stt" not in skip else None
        if "tts" not in skip:
            audio = check_tts(s, client)
            results["TTS"] = audio is not None
        else:
            results["TTS"] = None
        results["LiveKit"] = check_livekit(client) if "livekit" not in skip else None
    results["Redis"] = check_redis(s) if "redis" not in skip else None
    if args.play and audio:
        try:
            play(audio)
        except Exception as e:
            print(f"{FAIL} could not play audio: {e!r}")
    mark = lambda ok: "[ -- ]" if ok is None else OK if ok else FAIL  # noqa: E731
    print("\nsummary:", " | ".join(f"{name} {mark(ok)}" for name, ok in results.items()))
    if results["LLM"] and not (results["STT"] and results["TTS"]):
        print("text mode still works now:  python -m agent.main console --text")
    if any(ok is False for ok in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
