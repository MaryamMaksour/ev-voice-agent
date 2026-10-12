"""Per-turn latency: how long from the caller going quiet to the agent's first audio.

    end_to_end = end_of_utterance_delay (VAD/turn detector + STT)
               + llm.ttft (time to first token)
               + tts.ttfb (time to first audio byte)
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict

from livekit.agents import metrics as lk_metrics

logger = logging.getLogger("voice-agent.metrics")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


class CallMetrics:
    def __init__(self) -> None:
        self._turns: dict[str, dict[str, float]] = defaultdict(dict)

    def on_metrics(self, m) -> None:
        speech_id = getattr(m, "speech_id", None)
        if not speech_id:
            return
        turn = self._turns[speech_id]
        if isinstance(m, lk_metrics.EOUMetrics):
            turn["eou"] = m.end_of_utterance_delay
            turn["stt"] = m.transcription_delay
        elif isinstance(m, lk_metrics.LLMMetrics):
            turn.setdefault("llm_ttft", m.ttft)
        elif isinstance(m, lk_metrics.TTSMetrics):
            turn.setdefault("tts_ttfb", m.ttfb)
        if {"eou", "llm_ttft", "tts_ttfb"} <= turn.keys() and "e2e" not in turn:
            turn["e2e"] = turn["eou"] + turn["llm_ttft"] + turn["tts_ttfb"]
            logger.info(
                "turn latency %.2fs (eou %.2f, llm %.2f, tts %.2f)",
                turn["e2e"], turn["eou"], turn["llm_ttft"], turn["tts_ttfb"],
            )

    def turns(self) -> list[dict[str, float]]:
        return [t for t in self._turns.values() if "e2e" in t]

    def summary(self) -> dict:
        turns = self.turns()

        def stats(key: str) -> dict:
            vals = [t[key] for t in turns]
            return {
                "p50": round(percentile(vals, 50), 3),
                "p95": round(percentile(vals, 95), 3),
                "mean": round(statistics.fmean(vals), 3) if vals else 0.0,
            }

        return {
            "turns": len(turns),
            "end_to_end": stats("e2e"),
            "eou": stats("eou"),
            "llm_ttft": stats("llm_ttft"),
            "tts_ttfb": stats("tts_ttfb"),
        }
