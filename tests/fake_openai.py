"""A scripted OpenAI-compatible chat server (streaming), used to drive the real
LiveKit AgentSession + OpenAI plugin end to end without a GPU."""

from __future__ import annotations

import json
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def _chunk(delta: dict, finish: str | None = None, rid: str = "chatcmpl-fake") -> str:
    body = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "fake",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body)}\n\n"


class FakeOpenAI:
    """`policy(messages) -> str | dict` returns text, or {"tool": name, "args": {...}}."""

    def __init__(self, policy) -> None:
        self.policy = policy
        self.requests: list[dict] = []
        self.app = FastAPI()
        self.app.post("/v1/chat/completions")(self._chat)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._server = uvicorn.Server(uvicorn.Config(self.app, port=self.port, log_level="warning"))

    async def _chat(self, request: Request):
        body = await request.json()
        self.requests.append(body)
        reply = self.policy(body["messages"])
        if not body.get("stream"):  # post-call extraction: one JSON answer
            content = reply if isinstance(reply, str) else json.dumps(reply)
            return JSONResponse({"id": f"chatcmpl-{len(self.requests)}", "object": "chat.completion",
                                 "model": body.get("model"), "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": content}}],
                                 "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})

        rid = f"chatcmpl-{len(self.requests)}"  # unique per request, as vLLM does

        def stream():
            if isinstance(reply, dict):
                yield _chunk({"role": "assistant", "tool_calls": [{
                    "index": 0, "id": f"call_{len(self.requests)}", "type": "function",
                    "function": {"name": reply["tool"], "arguments": json.dumps(reply["args"])},
                }]}, rid=rid)
                yield _chunk({}, "tool_calls", rid=rid)
            else:
                yield _chunk({"role": "assistant", "content": reply}, rid=rid)
                yield _chunk({}, "stop", rid=rid)
            if (body.get("stream_options") or {}).get("include_usage"):  # token counts, as vLLM sends them
                yield "data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": 0,
                                             "model": "fake", "choices": [],
                                             "usage": {"prompt_tokens": 120, "completion_tokens": 12,
                                                       "total_tokens": 132}}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    def __enter__(self) -> "FakeOpenAI":
        threading.Thread(target=self._server.run, daemon=True).start()
        while not self._server.started:
            time.sleep(0.02)
        return self

    def __exit__(self, *exc) -> None:
        self._server.should_exit = True
