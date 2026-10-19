"""What the agent does when the LLM fails in the middle of a call.

LiveKit retries a failed LLM request (LLM_MAX_RETRY), then reports an unrecoverable error and the
caller hears nothing; it only ends the session after 3 such errors in a row. Instead, on the
first one, the agent:
  1. says a short fixed sentence (text-to-speech doesn't need the LLM),
  2. books a callback in the CRM (LLM_FAILURE_CALLBACK_MINUTES from now) and adds a note,
  3. ends the call. The call record gets ended_by = "llm_failure" and a "fallback" block.
If text-to-speech is down too, the sentence is skipped and the callback is still booked.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

logger = logging.getLogger("voice-agent.fallback")

FALLBACK_LINE = ("I'm sorry, I'm having a technical problem on my side. "
                 "A colleague will call you back shortly. Goodbye.")
NOTE = "Call stopped: the AI agent had a technical problem (LLM). Callback booked for a person to follow up."


def is_llm_failure(error: Any) -> bool:
    """An unrecoverable LLM error from LiveKit's session "error" event."""
    return getattr(error, "type", None) == "llm_error" and not getattr(error, "recoverable", True)


async def handle_llm_failure(*, say: Callable[[str], Any], crm, lead_id: str, callback_minutes: int,
                             now: datetime | None = None, say_timeout: float = 15.0) -> dict:
    """Run steps 1 and 2; returns the "fallback" block for the call record. Never raises."""
    now = now or datetime.now(timezone.utc)
    out: dict = {"reason": "llm", "at": now.isoformat(timespec="seconds"), "said": False,
                 "callback_at": None, "callback_booked": False, "note_added": False}
    try:
        handle = say(FALLBACK_LINE)
        if isinstance(handle, Awaitable) or hasattr(handle, "__await__"):
            await asyncio.wait_for(asyncio.ensure_future(handle), timeout=say_timeout)
        elif hasattr(handle, "wait_for_playout"):
            await asyncio.wait_for(handle.wait_for_playout(), timeout=say_timeout)
        out["said"] = True
    except Exception as e:  # TTS down too, or the caller already left
        logger.warning("could not say the fallback line: %r", e)
        out["say_error"] = repr(e)
    when = now + timedelta(minutes=callback_minutes)
    try:
        await crm.schedule_callback(lead_id, when)
        out["callback_at"], out["callback_booked"] = when.isoformat(timespec="seconds"), True
    except Exception as e:
        logger.exception("could not book the callback for %s", lead_id)
        out["callback_error"] = repr(e)
    try:
        await crm.add_note(lead_id, NOTE)
        out["note_added"] = True
    except Exception as e:
        logger.exception("could not add the note for %s", lead_id)
        out["note_error"] = repr(e)
    return out
