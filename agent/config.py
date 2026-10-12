"""All runtime settings come from the environment (.env). The LLM is served by Ollama
everywhere (laptop, RunPod, EC2); only the URLs and the model tag change."""

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

    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:11434/v1"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3:8b"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "ollama"))

    stt_base_url: str = field(default_factory=lambda: _env("STT_BASE_URL", "http://localhost:8001/v1"))
    stt_model: str = field(default_factory=lambda: _env("STT_MODEL", "whisper-1"))

    tts_base_url: str = field(default_factory=lambda: _env("TTS_BASE_URL", "http://localhost:8880/v1"))
    tts_model: str = field(default_factory=lambda: _env("TTS_MODEL", "kokoro"))
    tts_voice: str = field(default_factory=lambda: _env("TTS_VOICE", "af_heart"))

    # a laptop running Qwen3 partly on CPU can take >10 s (LiveKit's default) to answer
    llm_timeout: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT", "60")))
    llm_max_retry: int = field(default_factory=lambda: int(_env("LLM_MAX_RETRY", "1")))

    turn_detection: str = field(default_factory=lambda: _env("TURN_DETECTION", "model"))

    default_lead_id: str = field(default_factory=lambda: _env("DEFAULT_LEAD_ID", "L001"))
    calls_dir: str = field(default_factory=lambda: _env("CALLS_DIR", "data/calls"))


def no_think_body(model: str) -> dict:
    """Qwen3 'thinking' adds seconds of latency; switch it off for voice turns.

    Ollama reads the top-level "think" flag. As belt and braces the system prompt also
    ends with /no_think (prompts.py) and any <think> text is stripped before TTS
    (main.py), so a model that thinks anyway is never read aloud."""
    if "qwen3" in model.lower():
        return {"think": False}
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
    """Ask the LLM for one token so Ollama loads the model before the first call."""
    import httpx

    try:
        httpx.post(
            f"{settings.llm_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            json={"model": settings.llm_model, "max_tokens": 1,
                  "messages": [{"role": "user", "content": "hi /no_think"}], **no_think_body(settings.llm_model)},
            timeout=300,
        )
    except Exception:
        pass  # warm-up is best effort; the call itself will report real errors
