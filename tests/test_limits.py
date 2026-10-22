"""Maximum call length and silent callers (agent/limits.py), with a fake phone line."""

import asyncio

from agent.limits import MAX_DURATION_LINE, SILENCE_LINE, STILL_THERE_LINE, CallLimits


class Line:
    def __init__(self, ended_by: str = "") -> None:
        self.said: list[str] = []
        self.hung_up = 0
        self.ended_by = ended_by

    async def say(self, text: str) -> None:
        self.said.append(text)

    async def hang_up(self) -> None:
        self.hung_up += 1

    def on_end(self, reason: str) -> bool:
        if self.ended_by:
            return False
        self.ended_by = reason
        return True

    def limits(self, **kw) -> CallLimits:
        kw = {"max_seconds": 0, "silence_grace_seconds": 0.05, **kw}
        return CallLimits(say=self.say, hang_up=self.hang_up, on_end=self.on_end, **kw)


async def test_call_is_ended_at_the_maximum_length():
    line = Line()
    limits = line.limits(max_seconds=0.05)
    limits.start()
    await asyncio.sleep(0.2)
    assert line.said == [MAX_DURATION_LINE] and line.hung_up == 1 and line.ended_by == "max_duration"


async def test_no_limit_when_max_seconds_is_zero():
    line = Line()
    limits = line.limits(max_seconds=0)
    limits.start()
    await asyncio.sleep(0.1)
    assert line.said == [] and line.hung_up == 0
    limits.close()


async def test_silent_caller_is_asked_then_hung_up_on():
    line = Line()
    limits = line.limits()
    limits.user_state("away")
    await asyncio.sleep(0.2)
    assert line.said == [STILL_THERE_LINE, SILENCE_LINE] and line.hung_up == 1 and line.ended_by == "silence"


async def test_caller_who_answers_keeps_the_call():
    line = Line()
    limits = line.limits(silence_grace_seconds=0.1)
    limits.user_state("away")
    await asyncio.sleep(0.03)
    limits.user_state("speaking")  # "yes, I'm here"
    await asyncio.sleep(0.2)
    assert line.said == [STILL_THERE_LINE] and line.hung_up == 0 and line.ended_by == ""


async def test_a_second_silence_is_asked_again():
    line = Line()
    limits = line.limits(silence_grace_seconds=0.1)
    limits.user_state("away")
    await asyncio.sleep(0.03)
    limits.user_state("listening")
    limits.user_state("away")
    await asyncio.sleep(0.03)
    assert line.said == [STILL_THERE_LINE, STILL_THERE_LINE]
    limits.close()


async def test_a_call_already_ending_is_not_cut_again():
    line = Line(ended_by="agent")  # the agent said goodbye itself
    limits = line.limits(max_seconds=0.05)
    limits.start()
    await asyncio.sleep(0.2)
    assert line.said == [] and line.hung_up == 0 and line.ended_by == "agent"


async def test_hangs_up_even_if_the_goodbye_cannot_be_spoken():
    line = Line()

    async def broken(text: str) -> None:
        raise RuntimeError("TTS down")

    limits = CallLimits(max_seconds=0.05, silence_grace_seconds=0, say=broken, hang_up=line.hang_up,
                        on_end=line.on_end)
    limits.start()
    await asyncio.sleep(0.2)
    assert line.hung_up == 1
