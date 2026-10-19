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
| `models` | `llm`, `stt`, `tts`, `tts_voice`, `served_by` (`vllm` in production, `ollama` in tests), `llm_server` (model id; `max_model_len` on vLLM) |
| `versions` | `code` (git sha), `prompt` (hash of the prompt text), `livekit_agents`, `livekit_plugins_openai`, `voice_agent` |
| `settings` | Temperature, LLM timeout and retries, thinking off, `llm_backend` (vllm / ollama), turn detection, interruption mode, VAD thresholds, recording on/off, post-call mode |
| `errors` | Every pipeline error: `stage` (stt / llm / tts / vad / session / post_call), `error`, `at`, `recoverable` |
| `post_call_seconds`, `error` | How long post-call work took; its error if it failed |
| `post_call_llm` | The post-call LLM check: `requests` (more than 1 if the first JSON was invalid), `seconds`, `prompt_tokens`, `completion_tokens`, `ok` |
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
| `metrics.llm_post_call` | `attempt`, `model`, `seconds`, `prompt_tokens`, `completion_tokens`, `ok`, `error` |
| `false_interruption` | `resumed` |
| `error` | `stage`, `source`, `error`, `recoverable` |
| `call_closed` | `ended_by`, `close_reason` |
| `post_call_done` | `classification`, `seconds`, `error` |

## `data/calls/<call_id>/llm.jsonl`: every LLM request

`stage` is `call` (a request during the call) or `post_call` (the transcript check after it).
A `post_call` line has `attempt`, `model`, `request_id`, `seconds`, `prompt_tokens`, `completion_tokens`,
`reply` (the JSON the model returned), `ok` and `error`. A `call` line has:

| Field | What it says |
|---|---|
| `n`, `started_at`, `request_id` | Order in the call, start time, the LLM server's request id |
| `messages`, `messages_count` | The full conversation sent (system prompt, turns, tool results). Turn off with `LOG_LLM_MESSAGES=0`. |
| `tools`, `tools_hash` | Tool names offered |
| `text`, `tool_calls` | The reply, and tool calls (`name`, `arguments`, `tool_call_id`) |
| `ttft`, `seconds` | Time to first token, total time |
| `metrics` | From LiveKit: `prompt_tokens`, `prompt_cached_tokens`, `completion_tokens`, `reasoning_tokens` (> 0 means the model thought despite thinking off), `tokens_per_second`, `speech_id` |
| `cancelled`, `error`, `thinking_removed_chars` | Cut off by an interruption; failure; `<think>` text removed before speech |

## `data/calls/<call_id>/crm_changes.jsonl`: every CRM write

`source` (`call` = a tool during the call, `post_call` = the check after the call), `action`
(`update_lead`, `schedule_callback`, `mark_do_not_call`, `add_note`), `args`, `ok`, `error`, `seconds`,
and `changed`: `{field: {before, after}}` for each field that changed.

## When the LLM fails mid-call

LiveKit retries a failed LLM request (`LLM_MAX_RETRY`); if it still fails, the agent (`agent/fallback.py`) says a fixed sorry line, books a callback `LLM_FAILURE_CALLBACK_MINUTES` ahead (60), adds a CRM note and ends the call. The call record then has:

| Field | Example |
|---|---|
| `ended_by` | `llm_failure` |
| `fallback` | `{"reason": "llm", "at": "...", "said": true, "callback_at": "2026-10-19T11:00:00+00:00", "callback_booked": true, "note_added": true}` (plus `say_error` / `callback_error` / `note_error` when a step failed) |

`events.jsonl` gets an `llm_failure` event with the same fields. Before ringing a lead, the worker also checks that the LLM answers (`GET /v1/models`, 3 s); if not, the dial is blocked with rule `llm_down` (below).

## `data/health.json`: are the services up?

The agent container runs `python -m agent.check --write /app/data/health.json --watch 60`: every minute it checks the LLM, STT, TTS, LiveKit and Redis and rewrites this file (atomically):

```json
{"ts": "2026-10-19T10:00:00+00:00",
 "services": {"llm": {"ok": true, "seconds": 0.41, "error": null},
              "stt": {"ok": false, "seconds": 3.0, "error": "[FAIL] cannot connect. fix: ..."},
              "redis": {"ok": null, "seconds": null, "error": null}}}
```

`ok` is `null` when a service is skipped or not needed. The dashboard treats a file older than 5 minutes as "the checker stopped".

## `data/blocked/<YYYY-MM-DD>.jsonl`: dials the calling rules stopped

Before any phone call (`agent.dial` and the agent worker), `agent/compliance.py` checks the calling rules. A blocked dial is not made and leaves no call record; it adds one line here (UTC date):

| Field | Example | Meaning |
|---|---|---|
| `ts` | `2026-10-19T10:00:00+00:00` | When the dial was stopped (UTC) |
| `lead_id` | `L003` | The lead |
| `number` | `+97150*****67` | Masked number |
| `rule` | `daily_limit` | `do_not_call`, `blocked_status` (not_interested / do_not_call / wrong_number), `daily_limit`, `weekly_limit`, `outside_hours`, `llm_down` |
| `detail` | `1 call(s) today; the limit is 1 per day` | Plain-English reason |
| `source` | `worker` | `dial` (the `agent.dial` command) or `worker` (the agent, just before ringing) |

Only phone calls count towards the limits (a call record with a `line`); browser and console tests don't, and blocked dials don't. "Per day" is the calendar day in `CALL_TIMEZONE`; "per week" is the last 7 days. The EV Agent Dashboard reads this folder.

## Service logs

`logs/<service>/<YYYY-MM-DD>.jsonl`, one JSON object per line: `ts`, `level`, `logger`, `msg`, `service`,
plus `call_id`, `lead_id`, `room`, `job_id` on every line written during a call. Kept `LOG_KEEP_DAYS` days (30).

| Service | Writes |
|---|---|
| `agent` | The worker and every call (LiveKit, pipeline, tools, CRM) |
| `postcall` | `python -m agent.postcall`: calls finished after a crash or restart |
| `stt` | One line per transcription: engine, model, audio seconds, processing seconds, text |
