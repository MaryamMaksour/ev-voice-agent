"""Check every service the agent needs, and say which one is broken.

    python -m agent.check            # LLM (vLLM), STT, TTS, LiveKit, Redis
    python -m agent.check --play     # also play a test sentence on your speakers
    python -m agent.check --calls    # where the time goes in your recent calls
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
        r = client.get(f"{s.llm_base_url}/models", headers={"Authorization": f"Bearer {s.llm_api_key}"})
        if r.status_code == 200:
            models = r.json().get("data", [])
            names = [m.get("id") for m in models]
            if names and s.llm_model not in names:
                print(f"{FAIL} vLLM serves {names}, not '{s.llm_model}'.")
                print("       fix: set LLM_MODEL in .env to vLLM's --served-model-name (VLLM_SERVED_NAME)")
                return False
            for m in models:
                if m.get("id") == s.llm_model:
                    print(f"       serving {m.get('root')}  max context {m.get('max_model_len')} tokens")
        t0 = time.perf_counter()
        r = client.post(
            f"{s.llm_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {s.llm_api_key}"},
            json={"model": s.llm_model, "messages": [{"role": "user", "content": "Say hello in five words."}],
                  "max_tokens": 40, **no_think_body(s.llm_model)},
            timeout=180,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"] or ""
        print(f"{OK} replied in {time.perf_counter() - t0:.1f}s: {text.strip()[:80]!r}")
        if "<think>" in text or "</think>" in text:
            print("       note: the model is still thinking: check the vLLM --reasoning-parser flag")
        if time.perf_counter() - t0 > 3:
            print("       note: slow for a GPU. First request after start is slower (CUDA graphs); try again.")
        return True
    except httpx.ConnectError:
        print(f"{FAIL} cannot connect. Is vLLM running? fix: docker compose -f services/docker-compose.yml "
              "--env-file .env up -d vllm   (first start downloads the model: docker compose ... logs -f vllm)")
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
    """Redis (Celery broker for post-call work). None = not configured (post-call runs in the agent)."""
    if not s.redis_url:
        print("\nRedis  not set (REDIS_URL empty): post-call work runs inside the agent")
        return None
    print(f"\nRedis  {s.redis_url}")
    try:
        import redis

        redis.Redis.from_url(s.redis_url, socket_connect_timeout=3).ping()
        print(f"{OK} reachable")
        return True
    except Exception as e:
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
    print("did you hear 'Hi, this is Alex from Horizon Realty'? If not, check volume / output device.")


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
        "llm_ttft": "LLM first word (vLLM)",
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
    if slowest == "llm_ttft":
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
    args = p.parse_args()
    s = Settings()
    if args.calls:
        latency_report(s.calls_dir)
        return
    with httpx.Client(timeout=30, trust_env=False) as client:
        llm = check_llm(s, client)
        stt = check_stt(s, client)
        audio = check_tts(s, client)
        livekit = check_livekit(client)
    redis_ok = check_redis(s)
    if args.play and audio:
        try:
            play(audio)
        except Exception as e:
            print(f"{FAIL} could not play audio: {e!r}")
    mark = lambda ok: OK if ok else FAIL  # noqa: E731
    print("\nsummary:", "LLM", mark(llm), "| STT", mark(stt), "| TTS", mark(audio), "| LiveKit", mark(livekit),
          "| Redis", "[ -- ]" if redis_ok is None else mark(redis_ok))
    if llm and not (stt and audio):
        print("text mode still works now:  python -m agent.main console --text")


if __name__ == "__main__":
    main()
