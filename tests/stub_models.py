"""Stub STT + LLM + TTS on one port, OpenAI-compatible, for dry-running the whole
pipeline (LiveKit, agent, load test) without any GPU or model weights.

    python -m tests.stub_models --port 9000
    # .env: LLM_BASE_URL=STT_BASE_URL=TTS_BASE_URL=http://localhost:9000/v1  TURN_DETECTION=vad
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import math
import struct
import wave

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from tests.fake_openai import _chunk

# what the fake caller says, turn by turn (a typical qualified lead)
CALLER_LINES = [
    "Yes, speaking.",
    "Yes, I'm still interested.",
    "My budget is around eight hundred thousand dollars.",
    "I prefer the Marina area.",
    "A two bedroom apartment.",
    "Within the next three months.",
    "Great, thank you. Bye.",
]

app = FastAPI()
_lines = itertools.cycle(CALLER_LINES)


def _tone_wav(seconds: float = 0.8, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(int(seconds * rate))
        ))
    return buf.getvalue()


TONE = _tone_wav()


@app.post("/v1/audio/transcriptions")
async def stt() -> dict:
    return {"text": next(_lines)}


@app.post("/v1/audio/speech")
async def tts() -> Response:
    return Response(TONE, media_type="audio/wav")


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    messages = body["messages"]
    if not body.get("stream"):  # post-call extraction
        content = json.dumps({"outcome": "interested", "lead_update": {"status": "interested", "budget": 800000},
                              "summary": "Stub summary."})
        return JSONResponse({"id": "x", "object": "chat.completion", "model": body["model"], "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]})

    last = messages[-1]
    text = str(last.get("content", "")).lower()
    if last["role"] != "tool" and "budget" in text:
        args = {"status": "interested", "budget": 800000, "preferred_area": None,
                "property_type": None, "timeline": None, "interest_level": None}
        chunks = [_chunk({"role": "assistant", "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                  "function": {"name": "update_lead", "arguments": json.dumps(args)}}]}), _chunk({}, "tool_calls")]
    else:
        reply = "Hi, this is Alex from Horizon Realty." if len(messages) <= 2 else "Great, thank you."
        chunks = [_chunk({"role": "assistant", "content": reply}), _chunk({}, "stop")]
    return StreamingResponse(iter([*chunks, "data: [DONE]\n\n"]), media_type="text/event-stream")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=9000)
    uvicorn.run(app, host="127.0.0.1", port=p.parse_args().port, log_level="warning")
