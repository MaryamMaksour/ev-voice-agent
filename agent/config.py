"""All runtime settings come from the environment (.env).

The LLM is reached through the OpenAI-compatible API, so the same code talks to:
  - vLLM   in production (company GPU server): LLM_BASE_URL=http://<server>:8000/v1, LLM_MODEL=qwen3.5-9b
  - Ollama for testing (laptop):              LLM_BASE_URL=http://localhost:11434/v1, LLM_MODEL=qwen3.5:9b
Switching is a .env change only. LLM_BACKEND (vllm / ollama) is worked out from the URL
(port 11434 = Ollama) and only needs setting if Ollama runs on another port."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

BACKENDS = ("vllm", "ollama")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def llm_backend(base_url: str | None = None) -> str:
    """"vllm" or "ollama": LLM_BACKEND if set, else Ollama when the URL uses its port 11434."""
    explicit = os.getenv("LLM_BACKEND", "").strip().lower()
    if explicit:
        if explicit not in BACKENDS:
            raise ValueError(f"LLM_BACKEND must be one of {BACKENDS}, not {explicit!r}")
        return explicit
    url = base_url if base_url is not None else os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    return "ollama" if ":11434" in url else "vllm"


@dataclass(frozen=True)
class Settings:
    company_name: str = field(default_factory=lambda: _env("COMPANY_NAME", "Evolutions"))
    agent_name: str = field(default_factory=lambda: _env("AGENT_PERSONA_NAME", "Alex"))
    timezone: str = field(default_factory=lambda: _env("CALL_TIMEZONE", "UTC"))

    # vLLM: LLM_MODEL must equal vLLM's --served-model-name. Ollama: the model tag (ollama list)
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:8000/v1"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3.5-9b"))
    # empty = no key (vLLM without --api-key, or Ollama); the OpenAI client still needs some value
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
    # the LLM failed mid-call: the agent says sorry, ends the call and books a callback this far ahead
    llm_failure_callback_minutes: int = field(default_factory=lambda: int(_env("LLM_FAILURE_CALLBACK_MINUTES", "60")))

    turn_detection: str = field(default_factory=lambda: _env("TURN_DETECTION", "model"))

    default_lead_id: str = field(default_factory=lambda: _env("DEFAULT_LEAD_ID", "L001"))
    calls_dir: str = field(default_factory=lambda: _env("CALLS_DIR", "data/calls"))

    # --- rules checked before dialing a lead (agent/compliance.py); blocked dials go to BLOCKED_DIR
    max_calls_per_day: int = field(default_factory=lambda: int(_env("MAX_CALLS_PER_DAY", "1")))
    max_calls_per_week: int = field(default_factory=lambda: int(_env("MAX_CALLS_PER_WEEK", "2")))
    blocked_statuses: frozenset = field(default_factory=lambda: frozenset(
        s.strip() for s in _env("BLOCKED_STATUSES", "not_interested,do_not_call,wrong_number").split(",") if s.strip()))
    # "09:00-18:00" in CALL_TIMEZONE; empty = any time (24/7)
    call_hours: str = field(default_factory=lambda: _env("CALL_HOURS", ""))
    blocked_dir: str = field(default_factory=lambda: _env("BLOCKED_DIR", "data/blocked"))

    # --- logs: JSON lines per service in LOGS_DIR/<service>/<date>.jsonl (see agent/logging_setup.py)
    logs_dir: str = field(default_factory=lambda: _env("LOGS_DIR", "logs"))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    # save every LLM request's full messages in data/calls/<call>/llm.jsonl (0 = only replies and numbers)
    log_llm_messages: bool = field(default_factory=lambda: _env("LOG_LLM_MESSAGES", "1") in ("1", "true", "yes"))

    # --- post-call work runs in the agent; this is how long it waits for Egress to close the recording
    recording_wait_seconds: float = field(default_factory=lambda: float(_env("RECORDING_WAIT_SECONDS", "10")))
    # if that work fails (LLM or CRM down) it is retried; after this many failures the call is marked needs_review
    post_call_max_attempts: int = field(default_factory=lambda: int(_env("POST_CALL_MAX_ATTEMPTS", "5")))

    # --- limits on one call (agent/limits.py); 0 turns each off
    max_call_seconds: float = field(default_factory=lambda: float(_env("MAX_CALL_SECONDS", "600")))
    # silence before the agent asks "are you still there?", then how long before it hangs up
    silence_timeout_seconds: float = field(default_factory=lambda: float(_env("SILENCE_TIMEOUT_SECONDS", "20")))
    silence_grace_seconds: float = field(default_factory=lambda: float(_env("SILENCE_GRACE_SECONDS", "15")))

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

    @property
    def llm_backend(self) -> str:
        return llm_backend(self.llm_base_url)

    def recording_s3(self) -> dict | None:
        """S3 upload settings for recordings, or None to keep them on the egress server's disk.
        Credentials come from the egress server's own AWS role or environment, never from here."""
        if not self.recording_s3_bucket:
            return None
        return {"bucket": self.recording_s3_bucket, "region": self.recording_s3_region}


def no_think_body(model: str, backend: str | None = None) -> dict:
    """Qwen3 / Qwen3.5 'thinking' adds seconds of latency; switch it off for voice turns.

    vLLM: chat_template_kwargs reaches the model's chat template, which then skips thinking.
    Ollama: reasoning_effort "none" on its OpenAI-compatible API. Ollama 0.40 ignores its own
    "think": false there for qwen3.5 (a 5-word reply took 120 s of hidden thinking), so both are
    sent; the agent also adds /no_think to the prompt for older Qwen3 models.
    As a safety net any <think> text is still stripped before TTS (main.py), so a model
    that thinks anyway is never read aloud."""
    if "qwen3" not in model.lower():
        return {}
    if (backend or llm_backend()) == "ollama":
        return {"think": False, "reasoning_effort": "none"}
    return {"chat_template_kwargs": {"enable_thinking": False}}


def no_think_prompt_suffix(model: str, backend: str | None = None) -> str:
    """Ollama only: Qwen3's /no_think switch at the end of the system prompt, a second way to
    turn thinking off. vLLM doesn't need it (chat_template_kwargs is exact)."""
    if "qwen3" in model.lower() and (backend or llm_backend()) == "ollama":
        return "\n/no_think"
    return ""


def session_conn_options(settings: Settings):
    """Timeouts for the model calls; generous for the LLM so slow hardware doesn't fail."""
    from livekit.agents import APIConnectOptions
    from livekit.agents.voice.agent_session import SessionConnectOptions

    return SessionConnectOptions(
        llm_conn_options=APIConnectOptions(timeout=settings.llm_timeout, max_retry=settings.llm_max_retry),
        stt_conn_options=APIConnectOptions(timeout=30),
        tts_conn_options=APIConnectOptions(timeout=30),
    )


def llm_reachable(settings: Settings, timeout: float = 3.0) -> tuple[bool, str]:
    """Is the LLM server answering? A quick GET /models (vLLM and Ollama both have it): (ok, error)."""
    import httpx

    try:
        r = httpx.get(f"{settings.llm_base_url}/models", timeout=timeout, trust_env=False,
                      headers={"Authorization": f"Bearer {settings.llm_api_key}"})
        r.raise_for_status()
        return True, ""
    except Exception as e:
        return False, repr(e)[:300]


def warm_up_llm(settings: Settings) -> None:
    """Ask the LLM for one token before the first call: vLLM warms its CUDA graphs and prefix
    cache; Ollama loads the model into memory (10-30 s on a laptop)."""
    import httpx

    try:
        httpx.post(
            f"{settings.llm_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            json={"model": settings.llm_model, "max_tokens": 1,
                  "messages": [{"role": "user", "content": "hi"}],
                  **no_think_body(settings.llm_model, settings.llm_backend)},
            timeout=300,
        )
    except Exception:
        pass  # warm-up is best effort; the call itself will report real errors


def llm_server_info(settings: Settings) -> dict:
    """Which server and model answered (vLLM adds the max context): saved in every call record."""
    import httpx

    backend = settings.llm_backend
    try:
        r = httpx.get(f"{settings.llm_base_url}/models", timeout=3,
                      headers={"Authorization": f"Bearer {settings.llm_api_key}"})
        for m in r.json().get("data", []):
            if m.get("id") == settings.llm_model:
                return {"served_by": backend, "model": m.get("id"), "root": m.get("root"),
                        "max_model_len": m.get("max_model_len")}
    except Exception:
        pass
    return {"served_by": backend, "model": settings.llm_model}
