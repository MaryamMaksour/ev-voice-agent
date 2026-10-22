# Software Requirements Specification (SRS)

**Product:** EV AI Voice Agent: real-estate lead qualification
**Version:** 0.1 (in-company test version) · **Date:** 2026-10-20 · **Status:** Draft
**Companion document:** [SAD.md](SAD.md) (how it is built). Field reference: [call-record.md](call-record.md).

Requirement status: **Done** (in the code and tested), **Partial** (works with a known limit), **Planned** (not built yet).

---

## 1. Introduction

### 1.1 Purpose
This document says what the voice agent must do and under which limits. It is for the project team, the sales team (who supply the call scenarios) and whoever approves the move from test to production.

### 1.2 Scope
The product is an AI phone agent for Evolutions (a Dubai real-estate company). It calls a sales lead, checks the lead in the CRM before speaking, asks qualifying questions, and keeps the CRM up to date during and after the call. It runs on the company's own GPU server. It records and logs every call so a separate dashboard can track it.

**In scope (this version):** outbound qualification calls in English, CRM read/update, call logging and recording, calling rules, transfer to a person, deployment to the company server.
**Out of scope:** Arabic (the product is English only), inbound calls, WhatsApp/chat, automatic campaign scheduling, payment or contract handling. See section 7.

### 1.3 Definitions

| Term | Meaning |
|---|---|
| Lead | A person who enquired about a property and is stored in the CRM |
| Qualification | Learning the lead's interest, budget, area, property type and timeline |
| Call record | The JSON file and folder of logs saved for each call |
| Post-call check | An LLM pass over the transcript after the call that fills in CRM fields and writes a summary |
| LiveKit | Open-source real-time audio server used for rooms and SIP |
| 3CX | The company phone system the calls go out through |
| vLLM / Ollama | LLM servers (production / testing) that expose the same OpenAI-compatible API |
| STT / TTS / VAD | Speech to text / text to speech / voice activity detection |

### 1.4 References
`README.md`, `docs/architecture.md`, `docs/call-record.md`, `CLAUDE.md`; the EV Agent Dashboard repository (reads this product's data).

---

## 2. Overall description

### 2.1 Product perspective
The agent is one part of a small system: a LiveKit media server, three model servers (LLM, STT, TTS), the CRM (SQLite demo, Zoho for production), the 3CX phone system (later), and the EV Agent Dashboard. The dashboard only reads the files the agent writes. See SAD section 3.

### 2.2 Users

| User | What they need |
|---|---|
| **Lead** (the person called) | A short, polite, accurate call; to be able to stop it; never to be called against their wishes |
| **Sales team** | Qualified leads with answers and a summary in the CRM; callbacks; the ability to take over |
| **Operations / IT** | Easy deploy, health checks, clear failures, recordings and logs |
| **Management** | Call volume, outcomes and cost visible on the dashboard |

### 2.3 Operating environment
One company GPU server (Docker, NVIDIA container toolkit; 48 GB fits everything, 24 GB with FP8). Laptops on the company network for testing. Python 3.10+.

### 2.4 Constraints
- **C1.** Everything is self-hosted, because it is cheaper than cloud AI APIs at this volume. No call audio, transcript or lead data is sent to a cloud AI service.
- **C2.** The LLM is Qwen3.5-9B; vLLM in production, Ollama for tests. Only `.env` changes between them.
- **C3.** English only (STT, turn detection and prompts are English).
- **C4.** The agent works 24/7 in this version: no calling-hours restriction (`CALL_HOURS` empty).
- **C5.** The CRM is the SQLite demo until the real Zoho connection and status mapping are finished.
- **C6.** The prompts are placeholders until the sales team provides the call scenarios.

### 2.5 Assumptions and dependencies
- The sales team will provide scenarios and the Zoho status/field mapping.
- The 3CX trunk will be created by the phone-system administrator.
- The company server is reachable by GitHub's self-hosted runner for deploys.
- Target volume used for sizing: about 2,800 calls a day.

---

## 3. Functional requirements

### 3.1 Starting a call

| ID | Requirement | Status |
|---|---|---|
| FR-1 | The system shall start a call from a LiveKit room named `lead-<id>-<time>` (or job metadata with the lead id), joined by a terminal caller or a phone line. | Done |
| FR-2 | Before speaking, the agent shall load the lead from the CRM. If the lead cannot be loaded, the call shall not start and the error shall be logged. | Done |
| FR-3 | The agent shall not start a call with a lead marked do-not-call. | Done |
| FR-4 | Before dialing a phone number, the system shall check the calling rules (FR-30 to FR-33) and the LLM's health; a blocked dial shall be logged to `data/blocked/` and not made. | Done |
| FR-5 | A dial shall ring the lead through the 3CX trunk, wait for an answer (default ring timeout 30 s), and record the line result: answered, busy, no answer, declined, invalid number, trunk refused, line error, failed. | Done (needs the 3CX trunk) |
| FR-6 | The system shall detect an answering machine and not run the sales conversation with it. | Planned |
| FR-7 | Phone numbers from the CRM shall be normalized to international format before dialing. | Planned |

### 3.2 The conversation

| ID | Requirement | Status |
|---|---|---|
| FR-10 | The agent shall greet the lead by first name with fixed wording (not LLM-written) and ask to confirm who it is talking to. | Done |
| FR-11 | The agent shall mention that the call is recorded when recording is on. | Done |
| FR-12 | The agent shall state that it is an AI assistant. | Planned (decision pending; not in the current greeting) |
| FR-13 | The agent shall ask, one question at a time, for what the CRM does not already hold: interest, budget, preferred area, property type, timeline. | Done |
| FR-14 | The agent shall handle: wrong person, not interested, do not call again, callback request, request for a person. | Done |
| FR-15 | The agent shall keep replies short (one or two sentences), speak plain English with no lists or markdown, and never invent prices, availability or project details. | Done (prompt-enforced) |
| FR-16 | Model "thinking" text shall never be spoken. | Done |
| FR-17 | The caller shall be able to interrupt the agent (barge-in). | Done |
| FR-18 | Call scripts and wording shall follow the sales team's approved scenarios. | Planned (waiting for scenarios) |
| FR-19 | The agent speaks English only. | Done |

### 3.3 Tools the agent can use during a call

| ID | Requirement | Status |
|---|---|---|
| FR-20 | `update_lead`: save status, budget, preferred area, property type, timeline, interest level as soon as the caller says them. Invalid values shall be refused with a message the model can correct. | Done |
| FR-21 | `schedule_callback`: book a callback at an ISO datetime; past or malformed times shall be refused. | Done |
| FR-22 | `mark_do_not_call`: set the lead do-not-call. | Done |
| FR-23 | `transfer_to_human`: hand a phone caller to a person (cold transfer to `HUMAN_TRANSFER_TO`); if impossible, add a note and offer a callback. | Done (needs the 3CX trunk) |
| FR-24 | `end_call`: say goodbye and hang up. | Done |
| FR-25 | Tools shall only act on the lead being called (the lead id is bound when the call starts). | Done |

### 3.4 Calling rules

| ID | Requirement | Status |
|---|---|---|
| FR-30 | Never dial a lead who is do-not-call or whose status is in `BLOCKED_STATUSES` (default not_interested, do_not_call, wrong_number). | Done |
| FR-31 | At most `MAX_CALLS_PER_DAY` (default 1) and `MAX_CALLS_PER_WEEK` (default 2) phone calls per lead. | Done (see NFR-14) |
| FR-32 | Optionally restrict dialing to `CALL_HOURS` in `CALL_TIMEZONE`. Empty means any time. | Done (empty by design in this version) |
| FR-33 | Every blocked dial shall be recorded with its rule and reason. | Done |
| FR-34 | Check the UAE do-not-call registry and the lead's consent. | Planned (legal to decide) |

### 3.5 Limits on one call

| ID | Requirement | Status |
|---|---|---|
| FR-40 | A call shall end with a short goodbye after `MAX_CALL_SECONDS` (default 600; 0 = off). `ended_by` = `max_duration`. | Done |
| FR-41 | A silent caller shall be asked "are you still there?" after `SILENCE_TIMEOUT_SECONDS` (20) and hung up on after `SILENCE_GRACE_SECONDS` (15) more. `ended_by` = `silence`. Speech in between cancels the hang-up. | Done |
| FR-42 | If the LLM fails mid-call, the agent shall say a fixed apology, book a callback `LLM_FAILURE_CALLBACK_MINUTES` ahead, add a CRM note and end the call. | Done |

### 3.6 After the call (post-call check)

| ID | Requirement | Status |
|---|---|---|
| FR-50 | When the call ends (agent, lead hang-up, transfer or error), the system shall save the raw call first, then run an LLM check of the transcript. | Done |
| FR-51 | The check shall fill CRM fields the agent missed and never overwrite a value the agent confirmed live; disagreements shall be logged. | Done |
| FR-52 | The check shall enforce do-not-call if the lead asked to stop. | Done |
| FR-53 | The check shall write a 2 to 3 sentence summary note to the CRM (not for calls with no conversation). | Done |
| FR-54 | If the lead asked for a callback that the agent did not book, the check shall book it from the extracted date and time, or leave a note asking a person to follow up when no usable time exists. | Done |
| FR-55 | If post-call work fails, the raw file shall be kept, the call record written with `post_call_status = failed`, and the work retried every 5 minutes up to `POST_CALL_MAX_ATTEMPTS` (5); after that `needs_review`. | Done |
| FR-56 | If the process dies before post-call work finishes, `agent.postcall` shall finish it at the next start. | Done |

### 3.7 CRM integration

| ID | Requirement | Status |
|---|---|---|
| FR-60 | The agent shall use only the CRM port (`get_lead`, `update_lead`, `schedule_callback`, `mark_do_not_call`, `add_note`). | Done |
| FR-61 | Every CRM write shall be recorded with the lead before and after, the source (call or post_call), success and timing. | Done |
| FR-62 | A demo CRM (SQLite, 20 leads) shall be available for testing. | Done |
| FR-63 | A Zoho adapter shall read and update Leads and add notes, selected by `CRM_BACKEND=zoho`, sandbox by default. | Partial (not tested against the real CRM) |
| FR-64 | The mapping between agent statuses and the company's Zoho `Lead_Status` values shall be agreed and implemented. | Planned (waiting for the sales team) |
| FR-65 | The Zoho adapter shall treat an error returned inside a successful HTTP response as a failure, retry on rate limits, and share its access token between calls. | Planned |

### 3.8 Logging, recording and tracking

| ID | Requirement | Status |
|---|---|---|
| FR-70 | Each call shall produce `data/calls/<id>.json` plus a folder with `events.jsonl`, `llm.jsonl`, `crm_changes.jsonl`, `agent.log` and `recording.ogg`, as described in `call-record.md`. | Done |
| FR-71 | The call record shall hold the transcript, outcome, tool calls, CRM changes, latency per turn, tokens, models, code and prompt versions, settings and errors. | Done |
| FR-72 | Both sides of a phone call shall be recorded (LiveKit Egress) when recording is on; it is on by default on the server. | Done |
| FR-73 | Service logs shall be JSON lines tagged with the call id, kept `LOG_KEEP_DAYS` (30). | Done |
| FR-74 | The agent shall write `data/health.json` every minute with the state of the LLM, STT, TTS, LiveKit and Redis. | Done |
| FR-75 | Record files shall be written atomically so the dashboard never reads a half-written file. | Done |

### 3.9 Deployment and operation

| ID | Requirement | Status |
|---|---|---|
| FR-80 | Every push shall run the tests, validate the compose files and build the agent image. | Done |
| FR-81 | A push to `main` shall deploy to the GPU server, check every service, and roll back automatically if the check fails. | Done |
| FR-82 | A deploy shall let running calls finish (up to 10 minutes) before stopping the old agent. | Done |
| FR-83 | `agent.check` shall report which service is broken and how to fix it. | Done |
| FR-84 | The system shall run without a GPU for tests (`SERVER_GPU=0`, Ollama on the server). | Done |
| FR-85 | A campaign runner shall choose leads from the CRM, spread the calls over the day, retry no-answers and place scheduled callbacks. | Planned |

### 3.10 Demo website

A single page for showing the agent: talk to it from the browser and watch the lead change. It has no settings to change (no prompt, voice or speech-server choice) and needs no change to the agent.

| ID | Requirement | Status |
|---|---|---|
| FR-90 | The page shall list the 20 demo leads with name, status, budget, area, property type, timeline and interest. | Done |
| FR-91 | The user shall pick a lead, press **Start call**, and talk to the agent through the browser microphone. **End call** ends it. | Built, not yet tested on the server |
| FR-92 | While a call runs, the picked lead's row shall refresh every 2 seconds and highlight the fields the agent changed. | Built, not yet tested on the server |
| FR-93 | After the call the page shall show the transcript, the summary and the classification from the call record. | Done |
| FR-94 | **Reset lead** shall put the picked lead back to its starting values so the demo can be repeated. | Done |
| FR-95 | The website shall use the demo SQLite CRM only, shall refuse to start with `CRM_BACKEND` set to anything else, and shall listen on `127.0.0.1` only. | Done |
| FR-96 | The website shall start the call by creating the room `lead-<ID>-<time>` and giving the browser a token; the agent shall need no demo-specific code. | Built, not yet tested on the server |
| FR-97 | The website shall run on the company server and be reached through the SSH tunnel at `http://localhost:8090`. | Built, not yet tested on the server |

---

## 4. External interface requirements

| Interface | Detail |
|---|---|
| **LiveKit** | Rooms, WebRTC audio, SIP participants, Egress recording. Ports 7880/tcp (API), 7881/tcp and 7882/udp (audio). |
| **LLM** | OpenAI-compatible chat API with tool calls and JSON-schema output (`:8000` vLLM, `:11434` Ollama). |
| **STT** | OpenAI-compatible `/v1/audio/transcriptions` (`:8001`, faster-whisper or Parakeet). |
| **TTS** | OpenAI-compatible `/v1/audio/speech` (`:8880`, Kokoro). |
| **CRM** | `CRMClient` port; SQLite file `data/crm.db`; Zoho CRM REST API (v6, Leads and Notes). |
| **3CX** | SIP trunk through the LiveKit SIP service; transfer by SIP REFER. |
| **Dashboard** | Reads `data/calls/`, `data/recordings/`, `data/blocked/`, `data/health.json` (read-only). |
| **User interface** | Terminal voice client `python -m agent.talk --lead <id>`; text console `python -m agent.main console --text`; CLI `agent.dial`, `agent.check`, `agent.postcall`. |

---

## 5. Non-functional requirements

### 5.1 Performance and capacity

| ID | Requirement | Status |
|---|---|---|
| NFR-1 | The agent shall reply fast enough for natural conversation. Time per step (STT, LLM first token, TTS first audio) is measured for every turn. The target value is to be set from the load test. | Partial (measured, target not fixed) |
| NFR-2 | Model thinking shall be off for voice turns. | Done |
| NFR-3 | Size for about 2,800 calls a day on the GPU server. `loadtest` reports p50/p95 latency per concurrency level and the GPUs needed. | Partial (tool exists; results not recorded) |
| NFR-4 | Post-call work shall run in parallel for every call without a queue. | Done |

### 5.2 Reliability

| ID | Requirement | Status |
|---|---|---|
| NFR-10 | No call result shall be lost if a process dies, the LLM fails or the CRM is down (FR-42, FR-50 to FR-56). | Done |
| NFR-11 | The system shall not ring a lead while the LLM is down. | Done |
| NFR-12 | A bad deploy shall roll back automatically. | Done |
| NFR-13 | The agent shall run 24/7 and restart itself (`restart: unless-stopped`). | Done |
| NFR-14 | The per-lead call limits shall stay correct under concurrent dials and a large history of calls. | Partial (counted from files; a lead can be dialed twice at the same moment) |

### 5.3 Security and access
Lead names, numbers, transcripts and recordings are call data that only admins see on the dashboard.

| ID | Requirement | Status |
|---|---|---|
| NFR-20 | No customer data shall leave the company server for a cloud AI provider. | Done |
| NFR-21 | Secrets live in `.env` only, never in git. Phone numbers are masked in logs. | Done |
| NFR-22 | The model servers (LLM, STT, TTS) shall not be reachable from outside the server. | Partial (only by firewall; no authentication) |
| NFR-23 | Recording consent, retention period and deletion shall follow a legal-approved policy. | Planned |
| NFR-24 | Call data under `data/` shall not sync to a personal cloud folder. | Partial (the working folder is currently inside OneDrive) |

### 5.4 Maintainability and portability

| ID | Requirement | Status |
|---|---|---|
| NFR-30 | The CRM shall be replaceable without changing the agent (ports and adapters). | Done |
| NFR-31 | The LLM backend shall be switchable by two `.env` lines. | Done |
| NFR-32 | Tool logic shall be testable without LiveKit; tests shall run with no GPU and a fake LLM. | Done |
| NFR-33 | Quality of CRM updates shall be measurable (`evals.run_extraction`, 15 sample transcripts) and of speech (`evals.run_speech`). | Done |
| NFR-34 | Every call shall record the code version, prompt version and settings, so runs can be compared. | Done |

### 5.5 Observability

| ID | Requirement | Status |
|---|---|---|
| NFR-40 | Failures, blocked dials, stuck post-call work and service health shall be visible on the dashboard. | Done |
| NFR-41 | Operators shall be alerted (not only able to look) when the LLM, SIP or CRM fails. | Planned |

---

## 6. Acceptance criteria for moving from test to production

1. The sales scenarios are in the prompt and pass the scenario tests (`tests/test_scenarios_llm.py`) and `evals.run_extraction`.
2. The real Zoho CRM is connected, the status mapping is approved, and a test lead round-trips correctly.
3. A call through 3CX to a test phone works end to end, including no-answer, busy and a person transfer.
4. The load test shows the target concurrency within the agreed latency.
5. Legal has approved the greeting (AI and recording disclosure), consent, and retention.
6. Alerts exist for LLM, SIP and CRM failure.

## 7. Out of scope now, planned later

Campaign runner and automatic callbacks (FR-85); voicemail detection (FR-6); inbound calls; warm transfer with a summary for the consultant; failover across several GPU servers; a database for call state in place of files (a PostgreSQL index is planned).

## 8. Open questions

| # | Question | Owner |
|---|---|---|
| 1 | The final call scenarios and wording | Sales team |
| 2 | The Zoho `Lead_Status` values, custom fields and mapping | Sales team / CRM admin |
| 3 | Should the agent say it is an AI at the start? | Management / Legal |
| 4 | Budget currency: USD (current) or AED | Sales team |
| 5 | Recording consent wording and retention period | Legal |
| 6 | Target reply latency and calls at the same time | Management / IT |
