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
