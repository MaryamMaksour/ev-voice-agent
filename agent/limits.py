"""Keeps one call from running (and costing) forever: a maximum length, and a silent caller.

  - MAX_CALL_SECONDS: at this length the agent says a short fixed line and hangs up (0 = no limit).
  - SILENCE_TIMEOUT_SECONDS: LiveKit marks the caller "away" after this much silence. The agent asks
    "are you still there?"; if the caller still says nothing for SILENCE_GRACE_SECONDS it says goodbye
    and hangs up. Any speech in between cancels the hang-up.

The call record gets ended_by = "max_duration" or "silence". Kept apart from main.py so it can be
tested with fakes (no LiveKit)."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

logger = logging.getLogger("voice-agent.limits")

MAX_DURATION_LINE = ("I'm sorry, we've reached the time limit for this call. "
                     "A colleague will follow up with you. Goodbye.")
STILL_THERE_LINE = "Hello, are you still there?"
SILENCE_LINE = "I can't hear you, so I'll end the call now. We'll try again later. Goodbye."


class CallLimits:
    def __init__(self, *, max_seconds: float, silence_grace_seconds: float,
                 say: Callable[[str], Awaitable[None]], hang_up: Callable[[], Awaitable[None]],
                 on_end: Callable[[str], bool], say_timeout: float = 15.0) -> None:
        self.max_seconds = max_seconds
        self.silence_grace_seconds = silence_grace_seconds
        self._say = say
        self._hang_up = hang_up
        self._on_end = on_end            # records ended_by; False = the call is already ending, do nothing
        self._say_timeout = say_timeout
        self._max_task: asyncio.Task | None = None
        self._silence_task: asyncio.Task | None = None
        self._asked = False
        self._ending = False

    def start(self) -> None:
        if self.max_seconds > 0 and self._max_task is None:
            self._max_task = asyncio.ensure_future(self._after(self.max_seconds, "max_duration", MAX_DURATION_LINE))

    def user_state(self, new_state: str) -> None:
        """Call with every LiveKit user_state_changed: "speaking" / "listening" / "away"."""
        if new_state != "away":
            self._asked = False
            self._cancel_silence()
        elif not self._asked and not self._ending:
            self._asked = True
            self._silence_task = asyncio.ensure_future(self._ask_then_hang_up())

    def close(self) -> None:
        for task in (self._max_task, self._silence_task):
            if task is not None and task is not asyncio.current_task():
                task.cancel()
        self._max_task = self._silence_task = None

    def _cancel_silence(self) -> None:
        if self._silence_task is not None and self._silence_task is not asyncio.current_task():
            self._silence_task.cancel()
        self._silence_task = None

    async def _after(self, seconds: float, reason: str, line: str) -> None:
        await asyncio.sleep(seconds)
        await self._end(reason, line)

    async def _ask_then_hang_up(self) -> None:
        await self._speak(STILL_THERE_LINE)
        await asyncio.sleep(self.silence_grace_seconds)
        await self._end("silence", SILENCE_LINE)

    async def _speak(self, line: str) -> None:
        try:
            await asyncio.wait_for(self._say(line), timeout=self._say_timeout)
        except Exception as e:  # TTS down, or the caller already left: still hang up
            logger.warning("could not say %r: %r", line, e)

    async def _end(self, reason: str, line: str) -> None:
        if self._ending or not self._on_end(reason):
            return
        self._ending = True
        logger.info("ending the call: %s", reason)
        self._cancel_silence()
        await self._speak(line)
        await self._hang_up()
