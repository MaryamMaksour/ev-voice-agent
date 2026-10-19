# AI Voice Agent: real-estate lead qualification (self-hosted LiveKit + vLLM)

The agent calls a lead, **checks the lead in the CRM before it speaks**, asks the qualifying questions,
and **updates the CRM** during and after the call. Everything runs on the company GPU server:

- **Self-hosted LiveKit** for real-time audio. You talk to the agent **by voice from a terminal** today; a **3CX** phone line plugs in later through LiveKit SIP.
- **Qwen3.5-9B on vLLM** as the LLM, with tool calls for CRM updates and thinking turned off.
- **faster-whisper** speech-to-text and **Kokoro** text-to-speech.
- **Celery on Redis** for the work after a call: transcript check, CRM fill-in, summary note, logs and recording.
- **Every call fully logged**: recording, transcript, every LLM request and reply, tool calls, CRM changes before and after, timing of each STT, LLM and TTS step, tokens, and errors. This is what the [EV Agent Dashboard](https://github.com/MaryamMaksour/ev-agent-dashboard) reads.

**Architecture: [docs/architecture.md](docs/architecture.md)** · **Every logged field: [docs/call-record.md](docs/call-record.md)**

```
 Laptop terminal (mic + speakers)                 Company GPU server  (services/docker-compose.yml)
 python -m agent.talk --lead L003 ──WebRTC──►  LiveKit :7880 ◄──► Egress ──► data/recordings/
   (later: 3CX ──SIP──► LiveKit SIP)              │
                                                  ▼
                                       Agent worker (one process per call)
          Silero VAD → STT :8001 → turn detector → vLLM Qwen3.5-9B :8000 (+ CRM tools) → Kokoro TTS :8880
                                                  │ every CRM write ─► AuditedCRM ─► SQLite / Zoho
                                                  │ timeline, LLM requests, CRM changes ─► data/calls/<call>/
                                                  ▼ call ends
                                   Redis :6379 ─► Celery post-call worker
                     extraction (vLLM) → fill CRM → summary note → logs + recording → data/calls/<call>.json
```

## What happens on a call

1. `agent.talk` (or, later, 3CX) puts the caller in a LiveKit room named after the lead (`lead-L003-…`).
2. LiveKit sends the room to the agent worker. The agent:
   - loads the lead (name, project of interest, what's already known),
   - skips leads marked do-not-call,
   - starts the recording,
   - greets the lead by name.
3. It confirms who it is talking to, then asks about interest, budget, area, property type and timeline.
4. It saves each fact with `update_lead` as soon as it hears it. Other tools: `schedule_callback`, `mark_do_not_call`, `transfer_to_human`, `end_call`.
5. After the call, the Celery worker checks the transcript with the LLM:
   - fills fields the agent missed and enforces do-not-call,
   - writes a summary note to the CRM,
   - saves the call record next to its logs and recording.

**Thinking is off** for voice speed: every request sends `chat_template_kwargs: {enable_thinking: false}`. vLLM's `qwen3` reasoning parser keeps any thinking out of the reply, and the agent strips `<think>` text before speech as a last safety net. `reasoning_tokens` in `llm.jsonl` shows whether the model thought anyway.

---

## 1. Start the server (company GPU server)

Needs:
- Docker with the NVIDIA container toolkit.
- One NVIDIA GPU. 48 GB fits everything. On 24 GB, use FP8 (see `.env.example`).
- Ports 7880/tcp, 7881/tcp and 7882/udp reachable from the laptops that will call.

```bash
git clone https://github.com/MaryamMaksour/ev-voice-agent && cd ev-voice-agent
cp .env.example .env                  # set LIVEKIT_API_SECRET (32+ chars); HF_TOKEN if needed
mkdir -p data/recordings logs && chmod 777 data/recordings     # Egress writes recordings here
GIT_SHA=$(git rev-parse --short HEAD) docker compose -f services/docker-compose.yml --env-file .env up -d --build
docker compose -f services/docker-compose.yml --env-file .env logs -f vllm   # first start downloads Qwen3.5-9B (~19 GB)
```

Seed the demo CRM (20 leads, L001–L020) once, inside the agent container:
```bash
docker compose -f services/docker-compose.yml --env-file .env exec agent python -m crm.sqlite_adapter --seed
```

Check every service. It prints which one is broken and how to fix it:
```bash
docker compose -f services/docker-compose.yml --env-file .env exec agent python -m agent.check
# LLM [ OK ] | STT [ OK ] | TTS [ OK ] | LiveKit [ OK ] | Redis [ OK ]
```

| Service | What | Port |
|---|---|---|
| `livekit` | LiveKit server (rooms, audio) | 7880, 7881/tcp, 7882/udp |
| `redis` | Celery queue, LiveKit ↔ Egress | 6379 (localhost only) |
| `egress` | Call recording | — |
| `vllm` | Qwen3.5-9B | 8000 |
| `stt` | faster-whisper large-v3-turbo | 8001 |
| `tts` | Kokoro | 8880 |
| `agent` | LiveKit agent worker | — |
| `postcall-worker` | Celery post-call worker | — |

**GPU memory:**
- `VLLM_GPU_UTIL` (default 0.55) is the share vLLM takes. STT and TTS use the rest.
- On a 24 GB card, set `VLLM_EXTRA_ARGS=--quantization fp8` and `VLLM_GPU_UTIL=0.6`.
- **No Docker on the server?** `bash deploy/vllm_start.sh` starts vLLM with the same flags.

## 2. Talk to the agent by voice from your laptop

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[talk]"
cp .env.example .env     # same LIVEKIT_API_KEY / SECRET as the server, and:
                         # LIVEKIT_URL=ws://<server-ip>:7880
python -m agent.talk --lead L003
```

- You hear the agent and see live captions of both sides. **Ctrl+C** hangs up.
- Use headphones. Echo cancellation is on, but headphones are cleaner.
- Wrong mic or speaker? Run `python -m agent.talk --list-devices`, then `--input-device N --output-device M`.
- Pick another lead with `--lead L007`.
- **No sound card** (a server, a CI box): `--audio-file caller.wav --save-audio agent.wav --seconds 60` sends a recording as the caller and saves what the agent says.
- **Quick test without the LiveKit server:** `python -m agent.main console` runs the agent inside your terminal and still uses the server's models. Set `LLM_BASE_URL` / `STT_BASE_URL` / `TTS_BASE_URL` to the server's address and open ports 8000, 8001 and 8880 to the laptop. `console --text` lets you type instead of speak.

## 3. Where everything is saved

On the server, after each call:

```
data/calls/<call_id>.json              the call record (dashboard): transcript, outcome, CRM changes,
                                       latency per turn, tokens, models, versions, settings, errors
data/calls/<call_id>/events.jsonl      timeline: who spoke when, every state change, tool call, CRM change, metric, error
data/calls/<call_id>/llm.jsonl         every LLM request: full messages, reply, tool calls, tokens, time to first token
data/calls/<call_id>/crm_changes.jsonl every CRM write during and after the call, before -> after
data/calls/<call_id>/agent.log         this call's log lines (agent + post-call worker)
data/calls/<call_id>/recording.ogg     both sides of the call
logs/agent|postcall|stt/<date>.jsonl   all service logs as JSON, each line tagged with call_id
```

```bash
python -m agent.check --calls                                  # where the time goes in recent calls
grep '"call_id": "<call_id>"' logs/agent/$(date -u +%F).jsonl  # one call's lines from the service log
docker compose -f services/docker-compose.yml logs vllm        # vLLM, LiveKit, Egress (rotated 10 × 50 MB)
```

Every field: [docs/call-record.md](docs/call-record.md).

## 4. Phone calls through 3CX (later)

The agent already handles phone calls:
- `agent/dial.py` rings a lead.
- `agent/telephony.py` handles the line result (busy, no answer…), transfer to a person, and recording.

What's left is adding the LiveKit SIP service and the 3CX trunk. The steps are in [docs/architecture.md](docs/architecture.md#adding-3cx-later). Then:
```bash
python -m agent.dial --lead L003                     # ring the lead's CRM number
python -m agent.dial --lead L003 --number +9715...   # ring your own number to test
```

## Settings (`.env`)

| Setting | Default | What |
|---|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | | The LiveKit server and its keys |
| `LLM_MODEL` | `qwen3.5-9b` | vLLM's served model name |
| `VLLM_MODEL` | `Qwen/Qwen3.5-9B` | The Hugging Face model vLLM serves |
| `LLM_TEMPERATURE`, `LLM_TIMEOUT` | 0.3, 30 | |
| `VLLM_GPU_UTIL`, `VLLM_MAX_MODEL_LEN`, `VLLM_MAX_NUM_SEQS` | 0.55, 16384, 48 | GPU share, context length, calls answered at once |
| `TURN_DETECTION` | `model` | `model` = LiveKit's local turn detector, `vad` = silence only |
| `RECORDING_ENABLED` | 1 on the server | Record calls with Egress |
| `REDIS_URL` | set by compose | Empty = post-call work runs inside the agent |
| `LOGS_DIR`, `LOG_LEVEL`, `LOG_KEEP_DAYS` | `logs`, INFO, 30 | Service logs |
| `LOG_LLM_MESSAGES` | 1 | Save the full messages of each LLM request |
| `CRM_BACKEND` | `sqlite` | `zoho` for Zoho CRM (see below) |
| `SIP_OUTBOUND_TRUNK_ID`, `HUMAN_TRANSFER_TO` | | 3CX, later |

## Repo layout

```
agent/      main.py (worker), talk.py (terminal caller), tasks.py (Celery), calllog.py (per-call logs),
            logging_setup.py (JSON logs), prompts.py, tools.py, extraction.py, events.py, metrics.py,
            telephony.py, dial.py, check.py, make_token.py
crm/        port.py (CRMClient), audited.py (records every write), sqlite_adapter.py, zoho_adapter.py
services/   docker-compose.yml (server), docker-compose.local.yml (laptop, no GPU), livekit*.yaml, stt_server.py
deploy/     vllm_start.sh (vLLM without Docker)
docs/       architecture.md, call-record.md
evals/      run_extraction.py (CRM-update accuracy), run_speech.py (STT/TTS accuracy)
loadtest/   calls-per-GPU test
tests/      unit and wiring tests (no GPU), real-LLM scenario tests (opt-in)
```

## Tests

```bash
pip install -e ".[dev,talk]"
pytest                                               # no GPU or server needed
RUN_LLM_TESTS=1 pytest tests/test_scenarios_llm.py   # real scenarios against vLLM (LLM_BASE_URL in .env)
python -m evals.run_extraction                       # CRM-update accuracy on 15 sample transcripts
python -m evals.run_speech                           # STT/TTS accuracy (round trip, or --audio-dir real recordings)
```

**Full dry run without a GPU.** Fake STT, LLM and TTS stand in for the models; LiveKit, Redis, Celery, the agent and `agent.talk` are real:
```bash
python -m tests.stub_models --port 9000                    # fake models
docker compose -f services/docker-compose.local.yml up -d livekit redis
# .env: LLM_BASE_URL=STT_BASE_URL=TTS_BASE_URL=http://127.0.0.1:9000/v1  TURN_DETECTION=vad
#       REDIS_URL=redis://127.0.0.1:6379/0  RECORDING_ENABLED=0
celery -A agent.tasks worker -Q postcall -c 2 &
python -m agent.main dev &
python -m agent.talk --lead L003 --audio-file caller.wav --save-audio agent.wav --seconds 25
ls data/calls/                                             # the call record and its log folder
```

**Calls per GPU:**
1. `python -m loadtest.make_audio`
2. `python -m loadtest.run_load --levels 1,5,10,20,30,40`

It prints p50/p95 reply latency for each level, and the GPU count needed for 2,800 calls/day.

## Switching to Zoho

Set `CRM_BACKEND=zoho` and the `ZOHO_*` variables (use a **sandbox** first).

These custom lead fields must exist in the Leads layout: `Budget`, `Preferred_Area`, `Property_Type`, `Timeline`, `Interest_Level`, `Callback_Time`, `Do_Not_Call`, `Project_Interest`. Their API names can be overridden with `ZOHO_FIELD_<NAME>`.

## Not covered yet

- Arabic: needs an Arabic STT and TTS model.
- 3CX trunk setup (the code is ready).
- Failover across several GPU servers.
- Recording consent and retention rules.
