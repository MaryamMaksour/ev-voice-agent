"""Everything that happens on one call, saved as it happens, in one folder per call:

    data/calls/<call_id>.json            the call record (dashboard), written after the call
    data/calls/<call_id>/events.jsonl    timeline: speech, replies, states, tools, CRM, metrics, errors
    data/calls/<call_id>/llm.jsonl       every LLM request: messages sent, reply, tool calls, tokens, timings
    data/calls/<call_id>/crm_changes.jsonl   every CRM write: before -> after
    data/calls/<call_id>/agent.log       this call's own log lines (copied in after the call)
    data/calls/<call_id>/recording.ogg   the audio of both sides (linked in after the call)

Lines are appended the moment they happen, so a crash mid-call still leaves its trail.
Every line has `ts` (UTC) and `t_ms` (milliseconds since the call started).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("voice-agent.calllog")

LOG_FILES = {"events": "events.jsonl", "llm": "llm.jsonl", "crm_changes": "crm_changes.jsonl",
             "agent_log": "agent.log", "recording": "recording.ogg"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds") if ts else None


def _r(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


class CallLogger:
    def __init__(self, calls_dir: str | Path, call_id: str, *, log_llm_messages: bool = True,
                 started_at: str | None = None) -> None:
        self.call_id = call_id
        self.folder = Path(calls_dir) / call_id
        self.folder.mkdir(parents=True, exist_ok=True)
        self.log_llm_messages = log_llm_messages
        self._t0 = time.monotonic()
        if started_at:  # post-call: keep t_ms counting from the start of the call
            try:
                elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).total_seconds()
                self._t0 -= max(elapsed, 0.0)
            except ValueError:
                pass
        # LLM requests and their LiveKit metrics arrive separately; joined by request id
        self._llm_pending: dict[str, list[dict]] = {}
        self._llm_metrics: dict[str, list[dict]] = {}
        self.counts = {"user_turns": 0, "agent_replies": 0, "interruptions": 0, "false_interruptions": 0,
                       "llm_requests": 0, "llm_errors": 0, "llm_cancelled": 0, "tool_calls": 0,
                       "tool_errors": 0, "crm_writes": 0, "stt_interim": 0, "thinking_removed": 0}
        self.errors: list[dict] = []
        self.stt_per_utterance: list[dict] = []
        self.user_speech_seconds = 0.0
        self._user_started: float | None = None

    # ---------------------------------------------------------------- writing
    def _append(self, name: str, line: dict) -> None:
        line = {"ts": _now(), "t_ms": int((time.monotonic() - self._t0) * 1000), "call_id": self.call_id, **line}
        try:
            with open(self.folder / LOG_FILES[name], "a", encoding="utf-8") as f:
                f.write(json.dumps(line, default=str, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("could not write %s for %s", name, self.call_id)

    def event(self, kind: str, **data: Any) -> None:
        self._append("events", {"event": kind, **data})

    # ---------------------------------------------------------------- CRM
    def crm_change(self, *, source: str, action: str, lead_id: str, before: dict | None, after: dict | None,
                   args: dict | None = None, ok: bool = True, error: str | None = None,
                   seconds: float | None = None) -> None:
        """One CRM write. `source` is "call" (a tool during the call) or "post_call" (after it)."""
        changed = {k: {"before": (before or {}).get(k), "after": (after or {}).get(k)}
                   for k in sorted(set(before or {}) | set(after or {}))
                   if (before or {}).get(k) != (after or {}).get(k) and k != "updated_at"}
        line = {"source": source, "action": action, "lead_id": lead_id, "args": args, "ok": ok,
                "error": error, "seconds": _r(seconds), "changed": changed}
        self.counts["crm_writes"] += 1
        self._append("crm_changes", line)
        self.event("crm_change", **{k: v for k, v in line.items() if k != "lead_id"})

    # ---------------------------------------------------------------- LLM
    def llm_started(self, *, chat_ctx: Any, tools: list) -> dict:
        """Called by the agent's llm_node before the request; returns the record to fill in."""
        self.counts["llm_requests"] += 1
        names = sorted(getattr(t, "id", None) or getattr(t, "name", None) or type(t).__name__ for t in tools)
        rec: dict = {"n": self.counts["llm_requests"], "started_at": _now(), "_t": time.monotonic(),
                     "tools": names, "tools_hash": hashlib.sha1(",".join(names).encode()).hexdigest()[:10],
                     "text": "", "tool_calls": [], "request_id": None, "ttft": None, "cancelled": False,
                     "error": None, "thinking_removed_chars": 0}
        try:
            messages = chat_ctx.to_dict(exclude_timestamp=True).get("items", [])
        except Exception:
            messages = []
        rec["messages_count"] = len(messages)
        if self.log_llm_messages:
            rec["messages"] = messages
        return rec

    def llm_chunk(self, rec: dict, chunk: Any) -> None:
        if rec["ttft"] is None:
            rec["ttft"] = time.monotonic() - rec["_t"]
        if isinstance(chunk, str):
            rec["text"] += chunk
            return
        rec["request_id"] = rec["request_id"] or getattr(chunk, "id", None)
        delta = getattr(chunk, "delta", None)
        if delta is not None:
            if delta.content:
                rec["text"] += delta.content
            for tc in delta.tool_calls or []:
                rec["tool_calls"].append({"name": tc.name, "arguments": tc.arguments, "tool_call_id": tc.call_id})
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            rec["usage"] = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                            "cached_tokens": usage.prompt_cached_tokens, "total_tokens": usage.total_tokens}

    def llm_finished(self, rec: dict, *, cancelled: bool = False, error: BaseException | None = None) -> None:
        rec["seconds"] = _r(time.monotonic() - rec.pop("_t"))
        rec["ttft"] = _r(rec["ttft"])
        rec["cancelled"] = cancelled
        if cancelled:
            self.counts["llm_cancelled"] += 1
        if error is not None:
            rec["error"] = repr(error)
            self.counts["llm_errors"] += 1
            self.errors.append({"stage": "llm", "at": _now(), "error": repr(error)})
        if rec["thinking_removed_chars"]:
            self.counts["thinking_removed"] += 1
        rid = rec["request_id"]
        if rid and self._llm_metrics.get(rid):
            rec["metrics"] = self._llm_metrics[rid].pop(0)
            self._append("llm", rec)
        elif rid:
            # metrics not here yet; written when they arrive, or at close
            self._llm_pending.setdefault(rid, []).append(rec)
        else:
            self._append("llm", rec)

    def _llm_metrics_arrived(self, m: Any) -> None:
        data = {"ttft": _r(m.ttft), "duration": _r(m.duration), "prompt_tokens": m.prompt_tokens,
                "prompt_cached_tokens": m.prompt_cached_tokens, "completion_tokens": m.completion_tokens,
                "total_tokens": m.total_tokens, "tokens_per_second": _r(m.tokens_per_second, 1),
                # > 0 means the model thought despite enable_thinking=False (slower replies)
                "reasoning_tokens": getattr(m, "reasoning_tokens", 0),
                "cancelled": m.cancelled, "speech_id": m.speech_id}
        waiting = self._llm_pending.get(m.request_id)
        if not waiting:
            self._llm_metrics.setdefault(m.request_id, []).append(data)
        else:
            rec = waiting.pop(0)
            rec["metrics"] = data
            self._append("llm", rec)

    # ---------------------------------------------------------------- LiveKit session
    def on_metrics(self, m: Any) -> None:
        from livekit.agents import metrics as lk

        if isinstance(m, lk.LLMMetrics):
            self._llm_metrics_arrived(m)
            self.event("metrics.llm", request_id=m.request_id, speech_id=m.speech_id, ttft=_r(m.ttft),
                       seconds=_r(m.duration), prompt_tokens=m.prompt_tokens,
                       cached_tokens=m.prompt_cached_tokens, completion_tokens=m.completion_tokens,
                       tokens_per_second=_r(m.tokens_per_second, 1), cancelled=m.cancelled)
        elif isinstance(m, lk.STTMetrics):
            row = {"request_id": m.request_id, "seconds": _r(m.duration), "audio_seconds": _r(m.audio_duration),
                   "streamed": m.streamed}
            self.stt_per_utterance.append(row)
            self.event("metrics.stt", **row)
        elif isinstance(m, lk.TTSMetrics):
            self.event("metrics.tts", request_id=m.request_id, speech_id=m.speech_id, ttfb=_r(m.ttfb),
                       seconds=_r(m.duration), audio_seconds=_r(m.audio_duration),
                       characters=m.characters_count, cancelled=m.cancelled)
        elif isinstance(m, lk.EOUMetrics):
            self.event("metrics.eou", speech_id=m.speech_id, end_of_utterance_delay=_r(m.end_of_utterance_delay),
                       transcription_delay=_r(m.transcription_delay),
                       on_user_turn_completed_delay=_r(m.on_user_turn_completed_delay))
        elif isinstance(m, lk.VADMetrics):
            pass  # sent many times a second; adds nothing per call

    def attach(self, session: Any) -> None:
        """Subscribe to the AgentSession events worth keeping."""

        @session.on("user_input_transcribed")
        def _transcribed(ev) -> None:
            if not ev.is_final:
                self.counts["stt_interim"] += 1
                return
            self.event("user_transcript", text=ev.transcript, language=ev.language, speaker_id=ev.speaker_id)

        @session.on("conversation_item_added")
        def _item(ev) -> None:
            item = ev.item
            if getattr(item, "type", None) != "message" or item.role not in ("user", "assistant"):
                return
            role = "agent" if item.role == "assistant" else "user"
            if role == "user":
                self.counts["user_turns"] += 1
            else:
                self.counts["agent_replies"] += 1
                if item.interrupted:
                    self.counts["interruptions"] += 1
            self.event("message", role=role, text=item.text_content, interrupted=bool(item.interrupted),
                       stt_confidence=_r(item.transcript_confidence) if role == "user" else None,
                       timings=_message_metrics(item))

        @session.on("agent_state_changed")
        def _agent_state(ev) -> None:
            self.event("agent_state", old=ev.old_state, new=ev.new_state)

        @session.on("user_state_changed")
        def _user_state(ev) -> None:
            now = time.monotonic()
            if ev.new_state == "speaking":
                self._user_started = now
            elif ev.old_state == "speaking" and self._user_started is not None:
                self.user_speech_seconds += now - self._user_started
                self._user_started = None
            self.event("user_state", old=ev.old_state, new=ev.new_state)

        @session.on("function_tools_executed")
        def _tools(ev) -> None:
            for fc, out in zip(ev.function_calls, ev.function_call_outputs):
                self.counts["tool_calls"] += 1
                is_error = bool(out and out.is_error)
                if is_error:
                    self.counts["tool_errors"] += 1
                seconds = (out.created_at - fc.created_at) if out and fc.created_at else None
                self.event("tool_call", name=fc.name, arguments=fc.arguments, tool_call_id=fc.call_id,
                           output=(out.output if out else None), ok=not is_error, seconds=_r(seconds))

        @session.on("agent_false_interruption")
        def _false_interruption(ev) -> None:
            self.counts["false_interruptions"] += 1
            self.event("false_interruption", resumed=ev.resumed)

        @session.on("speech_created")
        def _speech(ev) -> None:
            self.event("speech_created", source=ev.source, user_initiated=ev.user_initiated)

        @session.on("error")
        def _error(ev) -> None:
            err = ev.error
            source = type(ev.source).__name__ if ev.source is not None else None
            row = {"stage": _stage(source), "source": source, "at": _now(), "error": repr(err),
                   "recoverable": getattr(err, "recoverable", None)}
            self.errors.append(row)
            self.event("error", **row)

        @session.on("metrics_collected")
        def _metrics(ev) -> None:
            self.on_metrics(ev.metrics)

    # ---------------------------------------------------------------- end of call
    def close(self, **data: Any) -> None:
        """Flush LLM requests whose metrics never arrived (cancelled turns) and mark the end."""
        for recs in self._llm_pending.values():
            for rec in recs:
                self._append("llm", rec)
        self._llm_pending.clear()
        if self._user_started is not None:
            self.user_speech_seconds += time.monotonic() - self._user_started
            self._user_started = None
        self.event("call_closed", **data)

    def summary(self) -> dict:
        """What goes into the call record next to CallMetrics.summary()."""
        return {
            "counts": dict(self.counts),
            "interruptions": self.counts["interruptions"],
            "false_interruptions": self.counts["false_interruptions"],
            "llm_requests": {"total": self.counts["llm_requests"], "errors": self.counts["llm_errors"],
                             "cancelled": self.counts["llm_cancelled"],
                             "thinking_removed": self.counts["thinking_removed"]},
            "stt_per_utterance": self.stt_per_utterance,
            "user_speech_seconds": round(self.user_speech_seconds, 2),
        }

    def files(self) -> dict:
        """Paths of this call's log files that exist (relative to the calls folder)."""
        return {key: f"{self.call_id}/{name}" for key, name in LOG_FILES.items() if (self.folder / name).exists()}


def _message_metrics(item: Any) -> dict | None:
    """LiveKit attaches per-message timings (STT delay, LLM ttft, TTS ttfb...) to chat items."""
    metrics = getattr(item, "metrics", None)
    if not metrics:
        return None
    try:
        return {k: (_r(v) if isinstance(v, float) else v) for k, v in dict(metrics).items()}
    except Exception:
        return None


def _stage(source: str | None) -> str:
    name = (source or "").lower()
    for stage in ("stt", "llm", "tts", "vad"):
        if stage in name:
            return stage
    return "session"


def bundle_call_logs(call_id: str, *, calls_dir: str | Path, logs_dir: str | Path,
                     recordings_dir: str | Path | None = None, recording_file: str | None = None,
                     services: tuple[str, ...] = ("agent", "postcall")) -> dict:
    """After the call: copy this call's service log lines into agent.log and link the recording.

    Returns the call's log files (relative paths) for the call record."""
    folder = Path(calls_dir) / call_id
    folder.mkdir(parents=True, exist_ok=True)
    needle = f'"call_id": "{call_id}"'
    lines: list[str] = []
    for service in services:
        for path in sorted((Path(logs_dir) / service).glob("*.jsonl"))[-2:]:  # today and yesterday
            try:
                with open(path, encoding="utf-8") as f:
                    lines.extend(line for line in f if needle in line)
            except OSError:
                continue
    if lines:
        lines.sort(key=lambda line: line[:40])  # every line starts with {"ts": "<iso time>"
        (folder / LOG_FILES["agent_log"]).write_text("".join(lines), encoding="utf-8")

    if recording_file and recordings_dir:
        src = Path(recordings_dir) / Path(recording_file).name
        dst = folder / LOG_FILES["recording"]
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                import shutil

                shutil.copy2(src, dst)
    return {key: f"{call_id}/{name}" for key, name in LOG_FILES.items() if (folder / name).exists()}


def recording_seconds(path: str | Path) -> float | None:
    """Length of an OGG/Opus recording, read from the last page's granule position (48 kHz)."""
    try:
        data = Path(path).read_bytes()
        last = data.rfind(b"OggS")
        if last < 0:
            return None
        granule = int.from_bytes(data[last + 6:last + 14], "little")
        return round(granule / 48000, 1)
    except OSError:
        return None
