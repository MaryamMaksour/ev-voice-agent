"""Per-call overrides for the demo website (demo/): a custom prompt, TTS voice and STT server.

Honoured only when DEMO_MODE=1 AND the room's metadata says {"source": "demo"}. A production
worker (DEMO_MODE unset) ignores them, so nobody can change the prompt of a real call."""

from __future__ import annotations

import json
import os
import string
from urllib.parse import urlparse

DEMO_KEYS = ("prompt", "tts_voice", "stt_base_url", "stt_model")
MAX_PROMPT_CHARS = 8000
MAX_VALUE_CHARS = 200
# the names a prompt template may use (see build_system_prompt); anything else is refused
PLACEHOLDERS = ("agent_name", "company", "now", "tz", "name", "first_name", "project", "status",
                "budget", "area", "ptype", "timeline")
# the agent can only act through these; a prompt that never mentions them cannot save anything to the CRM
TOOL_NAMES = ("update_lead", "schedule_callback", "mark_do_not_call", "transfer_to_human", "end_call")


def enabled() -> bool:
    return os.getenv("DEMO_MODE", "0").strip().lower() in ("1", "true", "yes")


def validate_template(template) -> str | None:
    """None if `template` can be used as the system prompt, else a message saying what to fix."""
    if not isinstance(template, str) or not template.strip():
        return "The prompt is empty."
    if len(template) > MAX_PROMPT_CHARS:
        return f"The prompt is too long ({len(template)} characters; the limit is {MAX_PROMPT_CHARS})."
    try:
        fields = [f for _, f, _, _ in string.Formatter().parse(template) if f is not None]
    except ValueError as e:
        return f"A curly bracket is not closed or matched ({e}). To write a real {{ or }}, type it twice: {{{{ or }}}}."
    unknown = sorted({f for f in fields if f not in PLACEHOLDERS})
    if unknown:
        shown = ", ".join("{" + (f or " ") + "}" for f in unknown)
        return (f"Unknown placeholder {shown}. Allowed: " + ", ".join("{" + p + "}" for p in PLACEHOLDERS)
                + ". To write a real { or }, type it twice.")
    return None


def _clean_url(value: str) -> str | None:
    parsed = urlparse(value)
    return value if parsed.scheme in ("http", "https") and parsed.netloc else None


def overrides_from_metadata(raw_metadata: list, *, on: bool | None = None) -> dict:
    """The allowed overrides from the job / room metadata JSON strings, or {} (never raises)."""
    if not (enabled() if on is None else on):
        return {}
    for raw in raw_metadata:
        try:
            meta = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(meta, dict) or meta.get("source") != "demo":
            continue
        out: dict = {}
        for key in DEMO_KEYS:
            value = meta.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            if key == "prompt":
                if validate_template(value) is None:
                    out[key] = value
            elif len(value) <= MAX_VALUE_CHARS and (key != "stt_base_url" or _clean_url(value)):
                out[key] = value.strip()
        return out
    return {}


def summary(overrides: dict) -> dict | None:
    """What goes into the call record: which settings were changed (the prompt only as a yes/no)."""
    if not overrides:
        return None
    out = {k: v for k, v in overrides.items() if k != "prompt"}
    out["custom_prompt"] = "prompt" in overrides
    return out
