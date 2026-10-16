"""Per-turn latency and tokens: how long from the caller going quiet to the agent's first audio.

    end_to_end = end_of_utterance_delay (VAD/turn detector + STT)
               + llm.ttft (time to first token)
               + tts.ttfb (time to first audio byte)

Every turn is kept (turn_details) so the dashboard can show the shortest, typical and longest
wait and which stage made a slow turn slow. Text mode has no audio, so its turns only carry
the LLM numbers (time to first token, tokens).
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict

from livekit.agents import metrics as lk_metrics

logger = logging.getLogger("voice-agent.metrics")

STAGES = {"eou": "speech-to-text", "llm_ttft": "LLM first word", "tts_ttfb": "text-to-speech first audio"}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _r(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


class CallMetrics:
    def __init__(self) -> None:
        self._turns: dict[str, dict] = defaultdict(dict)
        self._order: list[str] = []
        self._stt: list[dict] = []   # STT metrics carry no speech_id, so they are kept per request
        self._llm_calls = 0

    def _turn(self, speech_id: str, timestamp: float) -> dict:
        if speech_id not in self._turns:
            self._order.append(speech_id)
            self._turns[speech_id]["ts"] = timestamp
        return self._turns[speech_id]

    def on_metrics(self, m) -> None:
        if isinstance(m, lk_metrics.STTMetrics):
            self._stt.append({"seconds": m.duration, "audio_seconds": m.audio_duration})
            return
        speech_id = getattr(m, "speech_id", None)
        if not speech_id:
            return
        turn = self._turn(speech_id, getattr(m, "timestamp", 0.0))
        if isinstance(m, lk_metrics.EOUMetrics):
            turn["eou"] = m.end_of_utterance_delay
            turn["stt"] = m.transcription_delay
        elif isinstance(m, lk_metrics.LLMMetrics):
            self._llm_calls += 1
            turn.setdefault("llm_ttft", m.ttft)  # the first LLM call is what the caller waits for
            turn["llm_calls"] = turn.get("llm_calls", 0) + 1
            turn["llm_seconds"] = turn.get("llm_seconds", 0.0) + m.duration
            turn["prompt_tokens"] = turn.get("prompt_tokens", 0) + m.prompt_tokens
            turn["completion_tokens"] = turn.get("completion_tokens", 0) + m.completion_tokens
            turn["tokens_per_second"] = m.tokens_per_second
        elif isinstance(m, lk_metrics.TTSMetrics):
            turn.setdefault("tts_ttfb", m.ttfb)
            turn["tts_seconds"] = turn.get("tts_seconds", 0.0) + m.duration
            turn["tts_characters"] = turn.get("tts_characters", 0) + m.characters_count
            turn["tts_audio_seconds"] = turn.get("tts_audio_seconds", 0.0) + m.audio_duration
        if {"eou", "llm_ttft", "tts_ttfb"} <= turn.keys() and "e2e" not in turn:
            turn["e2e"] = turn["eou"] + turn["llm_ttft"] + turn["tts_ttfb"]
            logger.info(
                "turn latency %.2fs (eou %.2f, llm %.2f, tts %.2f)",
                turn["e2e"], turn["eou"], turn["llm_ttft"], turn["tts_ttfb"],
            )

    def turns(self) -> list[dict]:
        """Completed voice turns (all three stages measured)."""
        return [self._turns[s] for s in self._order if "e2e" in self._turns[s]]

    def turn_details(self) -> list[dict]:
        """Every turn the LLM answered, in order, with the stage that took longest."""
        out = []
        for n, sid in enumerate(self._order, 1):
            t = self._turns[sid]
            if "llm_ttft" not in t:
                continue
            stages = {k: t[k] for k in STAGES if k in t}
            slowest = max(stages, key=stages.get) if len(stages) == len(STAGES) else None
            out.append({
                "turn": n,
                "end_to_end": _r(t.get("e2e")),
                "eou": _r(t.get("eou")), "stt": _r(t.get("stt")),
                "llm_ttft": _r(t["llm_ttft"]), "llm_seconds": _r(t.get("llm_seconds")),
                "llm_calls": t.get("llm_calls", 0),
                "prompt_tokens": t.get("prompt_tokens", 0), "completion_tokens": t.get("completion_tokens", 0),
                "tokens_per_second": _r(t.get("tokens_per_second"), 1),
                "tts_ttfb": _r(t.get("tts_ttfb")), "tts_characters": t.get("tts_characters"),
                "slowest_stage": STAGES[slowest] if slowest else None,
            })
        return out

    def summary(self) -> dict:
        turns = self.turns()
        llm_turns = [self._turns[s] for s in self._order if "llm_ttft" in self._turns[s]]

        def stats(key: str, rows: list[dict]) -> dict:
            vals = [t[key] for t in rows if key in t]
            return {
                "min": round(min(vals), 3) if vals else 0.0,
                "p50": round(percentile(vals, 50), 3),
                "p95": round(percentile(vals, 95), 3),
                "max": round(max(vals), 3) if vals else 0.0,
                "mean": round(statistics.fmean(vals), 3) if vals else 0.0,
            }

        prompt = sum(t.get("prompt_tokens", 0) for t in llm_turns)
        completion = sum(t.get("completion_tokens", 0) for t in llm_turns)
        tps = [t["tokens_per_second"] for t in llm_turns if t.get("tokens_per_second")]
        return {
            "turns": len(turns),
            "end_to_end": stats("e2e", turns),
            "eou": stats("eou", turns),
            "llm_ttft": stats("llm_ttft", llm_turns),  # text mode too: every turn the LLM answered
            "tts_ttfb": stats("tts_ttfb", turns),
            "llm_turns": len(llm_turns),
            "llm_calls": self._llm_calls,
            "tokens": {
                "prompt": prompt, "completion": completion, "total": prompt + completion,
                "per_turn": round((prompt + completion) / len(llm_turns), 1) if llm_turns else 0.0,
                "tokens_per_second": round(statistics.fmean(tps), 1) if tps else 0.0,
            },
            "stt": {
                "requests": len(self._stt),
                "audio_seconds": round(sum(s["audio_seconds"] for s in self._stt), 2),
                "seconds_avg": round(statistics.fmean(s["seconds"] for s in self._stt), 3) if self._stt else 0.0,
            },
            "tts": {
                "characters": sum(t.get("tts_characters", 0) for t in llm_turns),
                "audio_seconds": round(sum(t.get("tts_audio_seconds", 0.0) for t in llm_turns), 2),
            },
            "turn_details": self.turn_details(),
        }
