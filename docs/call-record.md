# Call record and per-call logs

What the agent saves for every call, for the [EV Agent Dashboard](https://github.com/MaryamMaksour/ev-agent-dashboard)
and for debugging. Times are UTC ISO 8601. Durations are seconds unless the name ends in `_ms`.

## `data/calls/<call_id>.json`: the call record

`call_id` = `<room>-<start time>`, e.g. `lead-L003-20261019074417-2026-10-19T074417`.
While post-call work is still running, the file is `<call_id>.raw.json`.

| Field | What it says |
|---|---|
| `call_id`, `lead_id`, `room` | Which call, which lead, which LiveKit room |
| `started_at`, `ended_at`, `duration_seconds` | When, and how long |
| `ended_by`, `close_reason`, `agent_closed` | Who ended it (`agent` / `lead` / `transferred` / `not_reached` / `error`) and LiveKit's reason |
| `transcript` | Every line, `{role: agent|user, text}` |
| `tool_calls` | Every tool the LLM called: name, arguments, output, error, time |
| `actions` | The CRM writes the agent's tools made |
| `classification` | `qualified`, `interested`, `callback`, `not_interested`, `do_not_call`, `wrong_number`, `no_conversation`, `not_reached` |
| `extraction`, `filled_after_call`, `disagreements` | The post-call LLM check: what it read from the transcript, which CRM fields it filled, where it disagreed with the agent |
| `repeat_requests` | How often the agent asked the caller to repeat (a sign STT missed something) |
| `line` | Phone line result (3CX): answered / busy / no answer / declined / invalid number, SIP code, ring time, masked number. `null` for terminal calls. |
| `recording` | `ok`, `egress_id`, `file`, `path` on disk, `seconds`, or the error |
| `transfer` | Caller asked for a person: reason, where it went, whether it worked |
| `models` | `llm`, `stt`, `tts`, `tts_voice`, `served_by: vllm`, `llm_server` (model id, `max_model_len`) |
| `versions` | `code` (git sha), `prompt` (hash of the prompt text), `livekit_agents`, `livekit_plugins_openai`, `voice_agent` |
| `settings` | Temperature, LLM timeout and retries, thinking off, turn detection, interruption mode, VAD thresholds, recording on/off, post-call mode |
| `errors` | Every pipeline error: `stage` (stt / llm / tts / vad / session / post_call), `error`, `at`, `recoverable` |
| `post_call_seconds`, `error` | How long post-call work took; its error if it failed |
| `log_files` | Paths of this call's files below `data/calls/` |

### `metrics`

| Field | What it says |
|---|---|
| `end_to_end` | Caller stops talking → agent's first audio: `min` / `p50` / `p95` / `max` / `mean` |
| `eou` | End of turn: VAD + turn detector + STT |
| `llm_ttft` | LLM time to first token |
| `tts_ttfb` | TTS time to first audio |
| `turn_details` | Every turn: each stage's time, LLM calls, prompt and completion tokens, tokens per second, TTS characters, `slowest_stage` |
| `tokens` | `prompt`, `completion`, `total`, `per_turn`, `tokens_per_second` |
| `stt` | Requests, audio seconds, average processing seconds |
| `stt_per_utterance` | Each STT request: processing seconds, audio seconds |
| `tts` | Characters spoken, audio seconds |
| `interruptions`, `false_interruptions` | Caller talked over the agent; false alarms (noise) the agent recovered from |
| `llm_requests` | `total`, `errors`, `cancelled` (cut off by an interruption), `thinking_removed` |
| `user_speech_seconds` | How long the caller spoke |
| `counts` | User turns, agent replies, tool calls and errors, CRM writes, interim transcripts |

## `data/calls/<call_id>/events.jsonl`: the timeline

One JSON object per line. Each line has `ts`, `t_ms` (milliseconds since the call started), `call_id` and `event`:

| `event` | Extra fields |
|---|---|
| `call_started` | `lead_id`, `lead_name`, `lead_status`, `room`, `job_id` |
| `recording` | `ok`, `egress_id`, `file` or `error` |
| `user_state`, `agent_state` | `old`, `new` (listening / thinking / speaking / away) |
| `user_transcript` | `text`, `language` |
| `message` | `role`, `text`, `interrupted`, `stt_confidence`, `timings` (LiveKit's per-message metrics) |
| `speech_created` | `source`, `user_initiated` |
| `tool_call` | `name`, `arguments`, `tool_call_id`, `output`, `ok`, `seconds` |
| `crm_change` | `source` (call / post_call), `action`, `args`, `ok`, `error`, `seconds`, `changed` |
| `metrics.stt` | `seconds`, `audio_seconds` |
| `metrics.eou` | `end_of_utterance_delay`, `transcription_delay`, `on_user_turn_completed_delay` |
| `metrics.llm` | `request_id`, `ttft`, `seconds`, `prompt_tokens`, `cached_tokens`, `completion_tokens`, `tokens_per_second` |
| `metrics.tts` | `ttfb`, `seconds`, `audio_seconds`, `characters` |
| `false_interruption` | `resumed` |
| `error` | `stage`, `source`, `error`, `recoverable` |
| `call_closed` | `ended_by`, `close_reason` |
| `post_call_done` | `classification`, `seconds`, `error` |

## `data/calls/<call_id>/llm.jsonl`: every LLM request

| Field | What it says |
|---|---|
| `n`, `started_at`, `request_id` | Order in the call, start time, vLLM's request id |
| `messages`, `messages_count` | The full conversation sent (system prompt, turns, tool results). Turn off with `LOG_LLM_MESSAGES=0`. |
| `tools`, `tools_hash` | Tool names offered |
| `text`, `tool_calls` | The reply, and tool calls (`name`, `arguments`, `tool_call_id`) |
| `ttft`, `seconds` | Time to first token, total time |
| `metrics` | From LiveKit: `prompt_tokens`, `prompt_cached_tokens`, `completion_tokens`, `reasoning_tokens` (> 0 means the model thought despite thinking off), `tokens_per_second`, `speech_id` |
| `cancelled`, `error`, `thinking_removed_chars` | Cut off by an interruption; failure; `<think>` text removed before speech |

## `data/calls/<call_id>/crm_changes.jsonl`: every CRM write

`source` (`call` = a tool during the call, `post_call` = the Celery worker), `action`
(`update_lead`, `schedule_callback`, `mark_do_not_call`, `add_note`), `args`, `ok`, `error`, `seconds`,
and `changed`: `{field: {before, after}}` for each field that changed.

## Service logs

`logs/<service>/<YYYY-MM-DD>.jsonl`, one JSON object per line: `ts`, `level`, `logger`, `msg`, `service`,
plus `call_id`, `lead_id`, `room`, `job_id` on every line written during a call. Kept `LOG_KEEP_DAYS` days (30).

| Service | Writes |
|---|---|
| `agent` | The worker and every call (LiveKit, pipeline, tools, CRM) |
| `postcall` | Celery post-call tasks |
| `stt` | One line per transcription: engine, model, audio seconds, processing seconds, text |
