# EV Voice Agent

## The three projects (context)

1. **WhatsApp AI** (`projects/me/ev-proto`): in production, **not being worked on**. Do not touch it.
2. **Voice agent** (this repo, `Desktop/voice agent/ev-voice-agent-1`): being worked on.
3. **Dashboard** (`Desktop/projects/ev-agent-dashboard`): being worked on. One dashboard for both agents. Only **admins** use it, and they see everything (call text included).

**Why the voice agent is self-hosted: cost.** At about 3,000 calls a day, our own GPU server is cheaper than paid cloud AI APIs. It was not chosen for privacy, so do not justify decisions, or add work, in the name of privacy.

## Current stage (read this first)

- **In-company test version.** Not serving real customers yet.
- **English only.** There is no Arabic: STT is fixed to `en`, the turn detector is English-only, and the prompt says "speaking English". Do not add Arabic work unless asked.
- **Prompts are placeholders.** We are waiting for the sales scenarios, then the prompts in `agent/prompts.py` get rewritten.
- **CRM is still the SQLite demo** (`crm/sqlite_adapter.py`, 20 fake leads). Still to do: connect the real Zoho CRM and finalize the status mapping (`STATUS_TO_ZOHO` / `ZOHO_TO_STATUS` in `crm/zoho_adapter.py`). Treat the current mapping as a draft.
- **The agent runs 24/7.** `CALL_HOURS` is empty on purpose. Do not add calling-hours or day-of-week restrictions unless asked.
- We improve the project one item at a time. Do only the item asked, not neighbouring ones.

## The main idea

An AI phone agent ("Alex" from Evolutions, a real-estate company) that calls a lead, **checks the lead in the CRM before speaking**, asks qualifying questions (interest, budget, area, property type, timeline), and **keeps the CRM up to date**: during the call, and again after it from the full transcript. Everything runs self-hosted on the company GPU server (for cost, see above). Nothing goes to a cloud API.

## The core

One call = one LiveKit room = one agent process.

```
caller (terminal now, 3CX phone later)
   -> LiveKit room "lead-<ID>-<time>"
   -> agent worker:  Silero VAD -> STT -> turn detector -> LLM (+ tools) -> TTS
        tools (agent/tools.py): update_lead, schedule_callback, mark_do_not_call,
                                transfer_to_human, end_call
        every CRM write goes through AuditedCRM (before -> after is logged)
   -> caller hangs up
   -> post-call (same process): LLM extraction of the transcript -> fill missing CRM fields
        -> enforce do-not-call -> write summary note -> save the call record
```

- **Models:** Qwen3.5-9B on vLLM (production) or Ollama (testing); faster-whisper STT; Kokoro TTS. All speak OpenAI-compatible APIs, so switching is `.env` only (`LLM_BASE_URL`, `LLM_MODEL`). Thinking is switched off for voice speed (`no_think_body` in `agent/config.py`).
- **CRM port** (`crm/port.py`): the agent only knows `CRMClient`. Adapters: SQLite (demo), Zoho (production, `CRM_BACKEND=zoho`).
- **Calling rules** (`agent/compliance.py`): checked before every phone dial: do-not-call, blocked statuses, max calls per lead per day/week. Blocked dials go to `data/blocked/`.
- **Phone** (`agent/telephony.py`, `agent/dial.py`): 3CX through a LiveKit SIP outbound trunk. The code is ready; the trunk is not set up yet. Today the callers are the demo website (`demo/`, voice in the browser) or `python -m agent.main console --text`.
- **Failure handling:** if the LLM fails mid-call, `agent/fallback.py` says a fixed sorry line, books a callback and ends the call. If the process dies, `data/calls/<id>.raw.json` is finished later by `python -m agent.postcall`. No Celery on purpose (one process per call already runs post-call work in parallel).
- **Observability:** every call leaves a folder in `data/calls/<id>/` (events, every LLM request, CRM changes, recording) plus `<id>.json`, which the separate EV Agent Dashboard repo reads. Field reference: `docs/call-record.md`.
- **Deploy:** GitHub Actions tests every push; a push to `main` deploys with `deploy/deploy.sh` through a self-hosted runner on the server, runs `agent.check`, and rolls back on failure. Docker compose files are in `services/`.

## Where things are

| Path | What |
|---|---|
| `agent/main.py` | Worker entrypoint: lead lookup, dial rules, session, tools, shutdown |
| `agent/prompts.py` | System prompt and fixed greeting (**to be rewritten with the scenarios**) |
| `agent/tools.py` | Tool logic, free of LiveKit so it is unit-testable |
| `agent/extraction.py`, `agent/events.py` | Post-call extraction and processing |
| `agent/compliance.py` | Dial rules |
| `agent/calllog.py`, `agent/metrics.py`, `agent/logging_setup.py` | Per-call logs and metrics |
| `crm/` | `port.py`, `audited.py`, `sqlite_adapter.py`, `zoho_adapter.py` |
| `demo/` | Demo website: `server.py` (tiny API: leads, LiveKit token, call record) and `index.html`. Demo CRM only, no change to the agent. `DEMO=1` in `.env` starts it on the server. |
| `services/` | Compose files, LiveKit config, STT server |
| `evals/`, `loadtest/` | CRM-extraction and speech accuracy evals; calls-per-GPU load test |
| `docs/architecture.md`, `docs/call-record.md` | Full architecture and log field reference |
| `docs/SRS.md`, `docs/SAD.md` | Requirements (with status of each) and architecture document. Update them when behaviour changes. |

## Commands

```bash
pip install -e ".[dev]"
pytest                                   # no GPU or server needed (fake LLM)
RUN_LLM_TESTS=1 pytest tests/test_scenarios_llm.py   # real LLM from .env
python -m crm.sqlite_adapter --seed      # load the 20 demo leads
python -m agent.check                    # are LLM, STT, TTS, LiveKit, Redis up?
python -m agent.main dev                 # run the worker; then: python -m demo.server, open http://localhost:8090
python -m evals.run_extraction           # CRM-update accuracy
```

## Working rules

- Keep tool logic in `agent/tools.py` and CRM access behind `CRMClient`. Do not call a CRM API from the agent code directly.
- Every CRM write must go through `AuditedCRM`.
- Never commit `.env`, `data/`, recordings or logs (they are gitignored).
- The prompt and greeting are versioned by hash (`prompt_version()`); changing them is intentional and shows up in call records.
- Add or update tests with every change; CI runs `pytest` with a fake LLM.
