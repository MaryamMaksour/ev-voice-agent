"""Minimal OpenAI-compatible speech-to-text server on faster-whisper.

    POST /v1/audio/transcriptions  (multipart: file, model, language, prompt, response_format)

Env: WHISPER_MODEL (default large-v3-turbo), WHISPER_DEVICE (cuda|cpu|auto),
     WHISPER_COMPUTE_TYPE (float16 on GPU, int8 on CPU), WHISPER_BEAM_SIZE (default 1).
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import PlainTextResponse
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt")

MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3-turbo")
DEVICE = os.getenv("WHISPER_DEVICE", "auto")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "default")
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
# real-estate vocabulary hint; improves names of projects and areas
DEFAULT_PROMPT = os.getenv("WHISPER_PROMPT", "")
# concurrent transcriptions; whisper on one GPU handles a few at once
MAX_PARALLEL = int(os.getenv("WHISPER_MAX_PARALLEL", "4"))

app = FastAPI(title="faster-whisper STT")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE, num_workers=MAX_PARALLEL)
_slots = asyncio.Semaphore(MAX_PARALLEL)
logger.info("loaded %s on %s (%s)", MODEL_NAME, DEVICE, COMPUTE_TYPE)


def _transcribe(audio: bytes, language: str | None, prompt: str | None) -> str:
    segments, _ = model.transcribe(
        io.BytesIO(audio),
        language=language or None,
        beam_size=BEAM_SIZE,
        initial_prompt=prompt or DEFAULT_PROMPT or None,
        vad_filter=False,  # the agent already cut the audio with Silero VAD
        condition_on_previous_text=False,
    )
    return " ".join(s.text.strip() for s in segments).strip()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


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
        text = await asyncio.to_thread(_transcribe, audio, language, prompt)
    logger.info("transcribed %d bytes in %.2fs: %r", len(audio), time.perf_counter() - t0, text[:80])
    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "verbose_json":
        return {"text": text, "language": language or "en", "duration": 0.0, "segments": []}
    return {"text": text}
