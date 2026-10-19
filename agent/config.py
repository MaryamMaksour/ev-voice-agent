"""All runtime settings come from the environment (.env).

The LLM is Qwen3.5-9B served by vLLM (OpenAI-compatible API) on the company GPU server;
only the URLs change between machines."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    company_name: str = field(default_factory=lambda: _env("COMPANY_NAME", "Horizon Realty"))
    agent_name: str = field(default_factory=lambda: _env("AGENT_PERSONA_NAME", "Alex"))
    timezone: str = field(default_factory=lambda: _env("CALL_TIMEZONE", "UTC"))

    # vLLM: LLM_MODEL must equal vLLM's --served-model-name
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:8000/v1"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3.5-9b"))
    # empty = vLLM runs without a key; the OpenAI client still needs some value
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "") or "not-needed")
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.3")))

    stt_base_url: str = field(default_factory=lambda: _env("STT_BASE_URL", "http://localhost:8001/v1"))
    stt_model: str = field(default_factory=lambda: _env("STT_MODEL", "whisper-1"))

    tts_base_url: str = field(default_factory=lambda: _env("TTS_BASE_URL", "http://localhost:8880/v1"))
    tts_model: str = field(default_factory=lambda: _env("TTS_MODEL", "kokoro"))
    tts_voice: str = field(default_factory=lambda: _env("TTS_VOICE", "af_heart"))

    # seconds to wait for the LLM's first token (LiveKit's default of 10 s is too short at cold start)
    llm_timeout: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT", "30")))
    llm_max_retry: int = field(default_factory=lambda: int(_env("LLM_MAX_RETRY", "1")))

    turn_detection: str = field(default_factory=lambda: _env("TURN_DETECTION", "model"))

    default_lead_id: str = field(default_factory=lambda: _env("DEFAULT_LEAD_ID", "L001"))
    calls_dir: str = field(default_factory=lambda: _env("CALLS_DIR", "data/calls"))

    # --- logs: JSON lines per service in LOGS_DIR/<service>/<date>.jsonl (see agent/logging_setup.py)
    logs_dir: str = field(default_factory=lambda: _env("LOGS_DIR", "logs"))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    # save every LLM request's full messages in data/calls/<call>/llm.jsonl (0 = only replies and numbers)
    log_llm_messages: bool = field(default_factory=lambda: _env("LOG_LLM_MESSAGES", "1") in ("1", "true", "yes"))

    # --- post-call work: Celery on Redis. Empty = run it inside the agent process (laptop, tests)
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", ""))

    # --- phone line: 3CX through a LiveKit SIP outbound trunk (empty = browser / console calls only)
    sip_outbound_trunk_id: str = field(default_factory=lambda: _env("SIP_OUTBOUND_TRUNK_ID", ""))
    ring_timeout: int = field(default_factory=lambda: int(_env("RING_TIMEOUT", "30")))
    # where "transfer to a person" goes: tel:+9714xxxxxxx or sip:201@your-3cx-host (empty = offer a callback)
    human_transfer_to: str = field(default_factory=lambda: _env("HUMAN_TRANSFER_TO", ""))

    # --- recording with LiveKit Egress (needs the egress service; see README)
    recording_enabled: bool = field(default_factory=lambda: _env("RECORDING_ENABLED", "0") in ("1", "true", "yes"))
    # path as the egress container sees it; {room} and {lead} are filled in
    recording_filepath: str = field(default_factory=lambda: _env("RECORDING_FILEPATH", "/out/{room}.ogg"))
    # the same folder as this machine sees it, to link the file into the call's log folder
    recordings_dir: str = field(default_factory=lambda: _env("RECORDINGS_DIR", "data/recordings"))
    recording_s3_bucket: str = field(default_factory=lambda: _env("RECORDING_S3_BUCKET", ""))
    recording_s3_region: str = field(default_factory=lambda: _env("RECORDING_S3_REGION", ""))

    def recording_s3(self) -> dict | None:
        """S3 upload settings for recordings, or None to keep them on the egress server's disk.
        Credentials come from the egress server's own AWS role or environment, never from here."""
        if not self.recording_s3_bucket:
            return None
        return {"bucket": self.recording_s3_bucket, "region": self.recording_s3_region}


def no_think_body(model: str) -> dict:
    """Qwen3 / Qwen3.5 'thinking' adds seconds of latency; switch it off for voice turns.

    vLLM passes chat_template_kwargs to the model's chat template, which then skips the
    thinking block. As a safety net any <think> text is still stripped before TTS
    (main.py), so a model that thinks anyway is never read aloud."""
    if "qwen3" in model.lower():
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


def session_conn_options(settings: Settings):
    """Timeouts for the model calls; generous for the LLM so slow hardware doesn't fail."""
    from livekit.agents import APIConnectOptions
    from livekit.agents.voice.agent_session import SessionConnectOptions

    return SessionConnectOptions(
        llm_conn_options=APIConnectOptions(timeout=settings.llm_timeout, max_retry=settings.llm_max_retry),
        stt_conn_options=APIConnectOptions(timeout=30),
        tts_conn_options=APIConnectOptions(timeout=30),
    )


def warm_up_llm(settings: Settings) -> None:
    """Ask vLLM for one token so CUDA graphs and the prefix cache are warm before the first call."""
    import httpx

    try:
        httpx.post(
            f"{settings.llm_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            json={"model": settings.llm_model, "max_tokens": 1,
                  "messages": [{"role": "user", "content": "hi"}], **no_think_body(settings.llm_model)},
            timeout=120,
        )
    except Exception:
        pass  # warm-up is best effort; the call itself will report real errors


def llm_server_info(settings: Settings) -> dict:
    """What vLLM is serving (model id, max context): saved in every call record. Best effort."""
    import httpx

    try:
        r = httpx.get(f"{settings.llm_base_url}/models", timeout=3,
                      headers={"Authorization": f"Bearer {settings.llm_api_key}"})
        for m in r.json().get("data", []):
            if m.get("id") == settings.llm_model:
                return {"served_by": "vllm", "model": m.get("id"), "root": m.get("root"),
                        "max_model_len": m.get("max_model_len")}
    except Exception:
        pass
    return {"served_by": "vllm", "model": settings.llm_model}
