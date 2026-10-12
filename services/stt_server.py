"""Minimal OpenAI-compatible speech-to-text server with two engines.

    POST /v1/audio/transcriptions  (multipart: file, model, language, prompt, response_format)

STT_ENGINE=parakeet (laptop default): NVIDIA Parakeet TDT 0.6B v2 via sherpa-onnx.
    English only, more accurate than Whisper large and fast on a plain CPU.
    The int8 model (~650 MB) is downloaded on first start into PARAKEET_DIR.
    Env: PARAKEET_DIR, PARAKEET_THREADS (default 4).

STT_ENGINE=whisper (GPU servers): faster-whisper.
    Env: WHISPER_MODEL (default large-v3-turbo), WHISPER_DEVICE (cuda|cpu|auto),
         WHISPER_COMPUTE_TYPE (float16 on GPU, int8 on CPU), WHISPER_BEAM_SIZE (default 1),
         WHISPER_PROMPT (vocabulary hint).

Run without Docker:
    pip install -e ".[stt]"
    STT_ENGINE=parakeet python -m uvicorn --app-dir services stt_server:app --port 8001
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tarfile
import time
import urllib.request
import wave
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt")

ENGINE = os.getenv("STT_ENGINE", "whisper").lower()
# concurrent transcriptions per server
MAX_PARALLEL = int(os.getenv("STT_MAX_PARALLEL", os.getenv("WHISPER_MAX_PARALLEL", "4")))

PARAKEET_NAME = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
PARAKEET_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{PARAKEET_NAME}.tar.bz2"


class WhisperEngine:
    def __init__(self) -> None:
        from faster_whisper import WhisperModel

        self.name = os.getenv("WHISPER_MODEL", "large-v3-turbo")
        self.beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
        self.default_prompt = os.getenv("WHISPER_PROMPT", "")
        device = os.getenv("WHISPER_DEVICE", "auto")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "default")
        self.model = WhisperModel(self.name, device=device, compute_type=compute_type, num_workers=MAX_PARALLEL)
        logger.info("loaded whisper %s on %s (%s)", self.name, device, compute_type)

    def transcribe(self, audio: bytes, language: str | None, prompt: str | None) -> str:
        segments, _ = self.model.transcribe(
            io.BytesIO(audio),
            language=language or None,
            beam_size=self.beam_size,
            initial_prompt=prompt or self.default_prompt or None,
            vad_filter=False,  # the agent already cut the audio with Silero VAD
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()


def ensure_parakeet(model_dir: Path) -> Path:
    """Download and unpack the Parakeet model once (from the sherpa-onnx GitHub releases)."""
    target = model_dir / PARAKEET_NAME
    if (target / "tokens.txt").exists():
        return target
    model_dir.mkdir(parents=True, exist_ok=True)
    archive = model_dir / f"{PARAKEET_NAME}.tar.bz2"
    logger.info("downloading Parakeet model (~480 MB, first start only) from %s", PARAKEET_URL)
    urllib.request.urlretrieve(PARAKEET_URL, archive)
    with tarfile.open(archive, "r:bz2") as tar:
        try:
            tar.extractall(model_dir, filter="data")
        except TypeError:  # Python < 3.11.4 has no extraction filters
            tar.extractall(model_dir)
    archive.unlink()
    return target


class ParakeetEngine:
    def __init__(self) -> None:
        import sherpa_onnx

        model_dir = Path(os.getenv("PARAKEET_DIR", Path.home() / ".cache" / "parakeet"))
        path = ensure_parakeet(model_dir)
        self.name = "parakeet-tdt-0.6b-v2"
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(path / "encoder.int8.onnx"),
            decoder=str(path / "decoder.int8.onnx"),
            joiner=str(path / "joiner.int8.onnx"),
            tokens=str(path / "tokens.txt"),
            model_type="nemo_transducer",
            num_threads=int(os.getenv("PARAKEET_THREADS", "4")),
        )
        logger.info("loaded %s from %s", self.name, path)

    def transcribe(self, audio: bytes, language: str | None, prompt: str | None) -> str:
        import numpy as np

        try:
            with wave.open(io.BytesIO(audio), "rb") as w:
                if w.getsampwidth() != 2:
                    raise HTTPException(400, "expected 16-bit PCM WAV")
                rate, channels = w.getframerate(), w.getnchannels()
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        except wave.Error as e:
            raise HTTPException(400, f"expected a WAV file: {e}") from e
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(rate, pcm.astype(np.float32) / 32768.0)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()


def load_engine():
    if ENGINE == "parakeet":
        return ParakeetEngine()
    if ENGINE == "whisper":
        return WhisperEngine()
    raise SystemExit(f"unknown STT_ENGINE={ENGINE!r} (use parakeet or whisper)")


app = FastAPI(title="speech-to-text")
engine = load_engine()
_slots = asyncio.Semaphore(MAX_PARALLEL)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine": ENGINE, "model": engine.name}


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [{"id": engine.name, "object": "model"}]}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model_name: str = Form("whisper-1", alias="model"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float | None = Form(None),
):
    audio = await file.read()
    t0 = time.perf_counter()
    async with _slots:
        text = await asyncio.to_thread(engine.transcribe, audio, language, prompt)
    logger.info("transcribed %d bytes in %.2fs: %r", len(audio), time.perf_counter() - t0, text[:80])
    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "verbose_json":
        return {"text": text, "language": language or "en", "duration": 0.0, "segments": []}
    return {"text": text}
