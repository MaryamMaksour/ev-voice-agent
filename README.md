# AI Voice Agent: real-estate lead qualification (self-hosted LiveKit + vLLM)

The agent calls a lead, **checks the lead in the CRM before it speaks**, asks the qualifying questions,
and **updates the CRM** during and after the call. English only. Everything runs on the company GPU server, because that is cheaper than cloud AI APIs at this call volume:

- **Self-hosted LiveKit** for real-time audio. You talk to the agent **by voice from the demo website** in your browser today; a **3CX** phone line plugs in later through LiveKit SIP.
- **Qwen3.5-9B on vLLM** in production, **Ollama** for testing. Switching is two lines in `.env`; the code is the same.
- **faster-whisper** speech-to-text and **Kokoro** text-to-speech.
- **CI/CD with GitHub Actions**: every push is tested; a push to `main` deploys to the GPU server, checks every service, and rolls back if something is broken.
- **Every call fully logged**: recording, transcript, every LLM request and reply, tool calls, CRM changes before and after, time of each STT, LLM and TTS step, tokens, and errors. This is what the [EV Agent Dashboard](https://github.com/MaryamMaksour/ev-agent-dashboard) reads.

**Architecture: [docs/architecture.md](docs/architecture.md)** · **Every logged field: [docs/call-record.md](docs/call-record.md)**

```
 Laptop browser (mic + speakers)                  Company GPU server  (services/docker-compose.yml)
 demo website, lead L003          ──WebRTC──►  LiveKit :7880 ◄──► Redis ◄──► Egress ──► data/recordings/
   (later: 3CX ──SIP──► LiveKit SIP)              │
                                                  ▼
                                       Agent worker (one process per call)
          Silero VAD → STT :8001 → turn detector → vLLM Qwen3.5-9B :8000 (+ CRM tools) → Kokoro TTS :8880
                                                  │ every CRM write ─► AuditedCRM ─► SQLite / Zoho
                                                  │ timeline, LLM requests, CRM changes ─► data/calls/<call>/
                                                  ▼ caller hangs up
                     post-call check (same process): extraction (LLM) → fill CRM → summary note
                                                  → logs + recording → data/calls/<call>.json
```

## What happens on a call

1. The demo website (or, later, 3CX) puts the caller in a LiveKit room named after the lead (`lead-L003-…`).
2. LiveKit sends the room to the agent worker. The agent:
   - loads the lead (name, project of interest, what's already known),
   - skips leads marked do-not-call,
   - before a phone call, checks the **calling rules** (at most 1 call per lead a day and 2 a week, never a lead who said not interested, do-not-call or wrong number) and that the LLM is answering; a blocked dial is logged in `data/blocked/` and not made,
   - starts the recording,
   - greets the lead by name.
3. It confirms who it is talking to, then asks about interest, budget, area, property type and timeline.
4. If the LLM fails mid-call, the agent says a short sorry line, books a callback, adds a CRM note and ends the call, instead of going silent (`agent/fallback.py`).
5. It saves each fact with `update_lead` as soon as it hears it. Other tools: `schedule_callback`, `mark_do_not_call`, `transfer_to_human`, `end_call`.
6. After the caller hangs up, the same process checks the transcript with the LLM:
   - fills fields the agent missed and enforces do-not-call,
   - writes a summary note to the CRM,
   - saves the call record next to its logs and recording.

**Thinking is off** for voice speed. vLLM gets `chat_template_kwargs: {enable_thinking: false}` on every request; Ollama gets `think: false` and `/no_think` in the prompt. The agent also strips any `<think>` text before speech, as a last safety net. `reasoning_tokens` in `llm.jsonl` shows whether the model thought anyway.

**Why there is no Celery.** Each call already runs in its own process, so the post-call check (1–2 s on vLLM) runs in parallel for every call without a queue. If a process dies before it finishes, the call isn't lost: the agent saved `data/calls/<call_id>.raw.json` first, and `python -m agent.postcall` finishes it (the agent container runs this at every start). If the post-call work itself fails (LLM or CRM down), the raw file is kept and retried every 5 minutes, up to `POST_CALL_MAX_ATTEMPTS` (5) times; the call record shows `post_call_status` `failed`, then `needs_review` if it never worked. **Redis stays**, because LiveKit needs it to talk to Egress (call recording) and, later, the SIP service for 3CX.

---

## 0. Test on your laptop with Ollama

Ollama runs the same model (Qwen3.5-9B). It is slower than vLLM, which is fine for checking behaviour.

```bash
ollama pull qwen3.5:9b                     # or qwen3:8b on a small GPU
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,demo]"
cp .env.example .env                       # already set for Ollama: LLM_BASE_URL=http://localhost:11434/v1
python -m crm.sqlite_adapter --seed        # 20 demo leads, L001–L020
python -m agent.check --skip stt,tts,livekit   # is Ollama answering?
```

**Type to the agent** (only Ollama needed):
```bash
python -m agent.main console --text
```

**Talk to it by voice** (LiveKit, STT and TTS run in Docker on the laptop, on the CPU):
```bash
docker compose -f services/docker-compose.local.yml --env-file .env up -d   # add --profile ollama if Ollama isn't installed
python -m agent.main download-files        # once: VAD + turn-detector weights
python -m agent.main dev                   # terminal 1: the agent
python -m demo.server                      # terminal 2: the demo website, then open http://localhost:8090
```
Or `python -m agent.main console` for voice without LiveKit.

**Real-LLM tests and the CRM accuracy eval** work the same against Ollama or vLLM:
```bash
RUN_LLM_TESTS=1 pytest tests/test_scenarios_llm.py -v
python -m evals.run_extraction
```

**Moving to vLLM** means changing these two lines in `.env`. Nothing else changes:
```
LLM_BASE_URL=http://<server-ip>:8000/v1
LLM_MODEL=qwen3.5-9b
```

## 1. Start the server (company GPU server)

Needs:
- Docker with the NVIDIA container toolkit.
- One NVIDIA GPU. 48 GB fits everything. On 24 GB, use FP8 (see `.env.example`).
- Ports 7880/tcp, 7881/tcp and 7882/udp reachable from the laptops that will call.

```bash
git clone https://github.com/MaryamMaksour/ev-voice-agent /opt/ev-voice-agent && cd /opt/ev-voice-agent
cp .env.example .env                  # set LIVEKIT_API_SECRET (32+ chars); HF_TOKEN if needed
bash deploy/deploy.sh origin/main     # builds, starts everything, checks every service
docker compose -f services/docker-compose.yml --env-file .env logs -f vllm   # first start downloads Qwen3.5-9B (~19 GB)
```

By default the server uses its own vLLM. It ignores the laptop's `LLM_*` lines in `.env`, so the same `.env` file works in both places.

**Ollama already installed on the server?** Add `SERVER_LLM=ollama` to `.env` and run `deploy.sh` again. vLLM is then not started (no 19 GB download), and the agent uses `LLM_BASE_URL` / `LLM_MODEL` from `.env` (default: `http://localhost:11434/v1`, `qwen3.5:9b`). Check the model is there with `ollama list`. Fine for tests and a few calls at a time; for many calls at once, go back to vLLM by removing the line. Running `docker compose` by hand in this mode? Add `-f services/docker-compose.ollama.yml` after `-f services/docker-compose.yml`, or vLLM starts too.

**GPU not working yet?** For a test, add `SERVER_GPU=0` as well (it needs `SERVER_LLM=ollama`), and `LLM_TIMEOUT=120`. Speech-to-text then runs Parakeet on the CPU and text-to-speech the CPU build of Kokoro, so the NVIDIA Container Toolkit isn't needed. Replies are slow (several seconds or more): good for checking that everything connects, not for real calls. Remove the line when the GPU works.

The easiest way to run that test is one command, which sets those `.env` lines, starts everything, waits until it's healthy, loads the demo CRM (first time only) and makes one simulated call: `bash deploy/cpu_test.sh`. Afterwards, `bash deploy/stack.sh <command>` runs `docker compose` with the right files for your `.env`, for example `bash deploy/stack.sh ps`, `bash deploy/stack.sh logs --tail 50 agent` or `bash deploy/stack.sh down`.

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
| `redis` | LiveKit ↔ Egress (and later SIP) | 6379 (localhost only) |
| `egress` | Call recording | — |
| `vllm` | Qwen3.5-9B | 8000 |
| `stt` | faster-whisper large-v3-turbo | 8001 |
| `tts` | Kokoro | 8880 |
| `agent` | LiveKit agent worker + post-call check | — |

**GPU memory:**
- `VLLM_GPU_UTIL` (default 0.55) is the share vLLM takes. STT and TTS use the rest.
- On a 24 GB card, set `VLLM_EXTRA_ARGS=--quantization fp8` and `VLLM_GPU_UTIL=0.6`.
- **No Docker on the server?** `bash deploy/vllm_start.sh` starts vLLM with the same flags.

## 2. Talk to the agent: the demo website

A single page: pick one of the 20 demo leads, press **Start call** and talk through your browser. The lead's row refreshes while you talk and highlights what the agent saved. When the call ends the page shows the outcome, the summary and the conversation. **Reset lead** puts a lead back to its starting values so you can repeat the demo.

It uses the demo CRM only (it refuses to start with Zoho) and has no login: it listens on the server's own `127.0.0.1`.

On the company server:

1. In the server's `.env` add `DEMO=1` and make sure `CRM_BACKEND=sqlite`. **Remove `DEMO=1` before real phone calls**: while it is on, LiveKit advertises `127.0.0.1` so the call audio can travel through the SSH tunnel, and phones cannot connect.
2. Deploy as usual (`DEPLOY_DIR=$(pwd) bash deploy/deploy.sh <commit>`). First time only, load the leads: `bash deploy/stack.sh exec agent python -m crm.sqlite_adapter --seed`.
3. On your laptop (on the VPN) open the tunnel and keep it open:
   ```
   ssh -L 8090:localhost:8090 -L 7880:localhost:7880 -L 7881:localhost:7881 <user>@<server>
   ```
4. Open **http://localhost:8090** (use `localhost`: browsers only allow the microphone there), pick a lead and press **Start call**. Use headphones.

If the page does not load: `bash deploy/stack.sh logs --tail 50 demo`. If you connect but nobody answers, the agent is not running: `bash deploy/stack.sh logs --tail 50 agent`.

Every demo call is saved like a normal call (`data/calls/`), so it also shows in the dashboard as a browser call.

## 3. Where everything is saved

On the server, after each call:

```
data/calls/<call_id>.json              the call record (dashboard): transcript, outcome, CRM changes,
                                       latency per turn, tokens, models, versions, settings, errors
data/calls/<call_id>/events.jsonl      timeline: who spoke when, every state change, tool call, CRM change, metric, error
data/calls/<call_id>/llm.jsonl         every LLM request, during the call and the post-call check:
                                       full messages, reply, tool calls, tokens, time to first token
data/calls/<call_id>/crm_changes.jsonl every CRM write during and after the call, before -> after
data/calls/<call_id>/agent.log         this call's log lines
data/calls/<call_id>/recording.ogg     both sides of the call
logs/agent|postcall|stt/<date>.jsonl   all service logs as JSON, each line tagged with call_id
```

| You want to see | Where |
|---|---|
| Time per model (STT, LLM, TTS) for every turn | `metrics.turn_details` in the call record; `metrics.stt/llm/tts` lines in `events.jsonl` |
| Tokens in / out, tokens per second | `metrics.tokens`, and per request in `llm.jsonl` |
| What the agent said, and what it was sent | `transcript`; `llm.jsonl` (`messages`, `text`) |
| Result of the call | `classification`, `extraction`, `filled_after_call` |
| Tool calls | `tool_calls` (name, arguments, output, error) |
| CRM changes | `crm_changes.jsonl`: each field before → after |
| Errors | `errors` (stage: stt / llm / tts / vad / session / post_call) |
| Which model and server answered | `models.llm`, `models.served_by` (vllm / ollama), `settings.llm_backend` |

```bash
python -m agent.check --calls                                  # where the time goes in recent calls
grep '"call_id": "<call_id>"' logs/agent/$(date -u +%F).jsonl  # one call's lines from the service log
docker compose -f services/docker-compose.yml logs vllm        # vLLM, LiveKit, Egress (rotated 10 × 50 MB)
```

Every field: [docs/call-record.md](docs/call-record.md).

## 4. CI/CD

`.github/workflows/ci-cd.yml` runs on every push and pull request:

| Job | What it checks |
|---|---|
| **Tests** | All unit and wiring tests, with a fake LLM (no GPU): tools, CRM, post-call check, logging, the Ollama/vLLM switch, the terminal client |
| **Compose files are valid** | The server and laptop compose files, and the deploy scripts |
| **Agent image builds** | The Docker image of the agent |
| **Deploy to the GPU server** | Only on a push to `main` (or a manual run with "deploy" ticked), and only after the three jobs above pass |

**The deploy** runs `deploy/deploy.sh` on the server:
1. It checks out the new commit.
2. It rebuilds and restarts what changed. Running calls finish first: the old agent gets up to 10 minutes.
3. It runs `agent.check` inside the agent container, which checks the LLM, STT, TTS, LiveKit and Redis.
4. If the check fails, it goes back to the previous commit and the job fails (red in GitHub).

You can run the same script by hand: `bash deploy/deploy.sh <commit or branch>`.

**One-time setup.** The server is on the company network, so GitHub's machines can't reach it. Instead, a GitHub runner on the server fetches the work:
1. On the server, clone the repo to `/opt/ev-voice-agent` and create `.env` there (see section 1). The server needs read access to the repo: add a **deploy key** (GitHub → repo Settings → Deploy keys).
2. GitHub → repo **Settings → Actions → Runners → New self-hosted runner** (Linux). Run the commands it shows on the server, and add the label **`gpu-server`** when asked. Install it as a service (`sudo ./svc.sh install && sudo ./svc.sh start`). The runner's user must be allowed to use Docker (`sudo usermod -aG docker <user>`).
3. GitHub → **Settings → Secrets and variables → Actions → Variables**:
   - `DEPLOY_ENABLED` = `true`
   - `DEPLOY_DIR` = `/opt/ev-voice-agent`
4. GitHub → **Settings → Environments → New environment** `production`. Optional: add yourself as a required reviewer, so every deploy waits for your click.
5. Make `main` the default branch (Settings → General), and protect it so changes come through pull requests.

Until `DEPLOY_ENABLED` is `true`, the deploy job is skipped and CI still runs. Secrets stay in the server's `.env` and never go to GitHub.

## 5. Phone calls through 3CX (later)

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
| `LLM_BASE_URL`, `LLM_MODEL` | Ollama, `qwen3.5:9b` | The LLM the laptop uses. The server uses its own vLLM, unless `SERVER_LLM=ollama`. |
| `SERVER_LLM` | `vllm` | `ollama` = the server uses the Ollama installed on it (and `LLM_BASE_URL` / `LLM_MODEL`) instead of starting vLLM |
| `SERVER_GPU` | `1` | `0` = no GPU on the server (tests only): STT and TTS run on the CPU. Needs `SERVER_LLM=ollama` |
| `LLM_BACKEND` | from the URL | `vllm` or `ollama`; port 11434 means Ollama. Set it only if Ollama is on another port. |
| `VLLM_SERVED_NAME`, `VLLM_MODEL` | `qwen3.5-9b`, `Qwen/Qwen3.5-9B` | vLLM's model name, and the Hugging Face model it serves |
| `LLM_TEMPERATURE`, `LLM_TIMEOUT` | 0.3, 30 | |
| `VLLM_GPU_UTIL`, `VLLM_MAX_MODEL_LEN`, `VLLM_MAX_NUM_SEQS` | 0.55, 16384, 48 | GPU share, context length, calls answered at once |
| `TURN_DETECTION` | `model` | `model` = LiveKit's local turn detector, `vad` = silence only |
| `RECORDING_ENABLED` | 0 (laptop) | The server records anyway; `SERVER_RECORDING_ENABLED=0` turns that off |
| `RECORDING_WAIT_SECONDS` | 10 | How long the post-call check waits for the recording file |
| `CALL_TIMEZONE` | `Asia/Dubai` in `.env.example` | "Today", callback dates and the calling rules |
| `MAX_CALLS_PER_DAY`, `MAX_CALLS_PER_WEEK` | 1, 2 | Phone calls per lead per day (calendar day) and per 7 days. 0 = no limit |
| `BLOCKED_STATUSES` | `not_interested,do_not_call,wrong_number` | Never call leads with these CRM statuses |
| `CALL_HOURS` | empty (24/7) | e.g. `09:00-18:00`: only call in this window |
| `MAX_CALL_SECONDS` | 600 | A call longer than this ends with a short goodbye (`ended_by` = `max_duration`). 0 = no limit |
| `SILENCE_TIMEOUT_SECONDS`, `SILENCE_GRACE_SECONDS` | 20, 15 | A silent caller is asked "are you still there?" after the first, and hung up on after the second (`ended_by` = `silence`). 0 = off |
| `POST_CALL_MAX_ATTEMPTS` | 5 | Failed post-call work is retried every 5 minutes this many times, then marked `needs_review` |
| `LLM_FAILURE_CALLBACK_MINUTES` | 60 | The LLM failed mid-call: the agent says sorry, ends the call and books a callback this far ahead |
| `BLOCKED_DIR` | `data/blocked` | Dials the rules stopped, one line each (see [docs/call-record.md](docs/call-record.md)) |
| `LOGS_DIR`, `LOG_LEVEL`, `LOG_KEEP_DAYS` | `logs`, INFO, 30 | Service logs |
| `LOG_LLM_MESSAGES` | 1 | Save the full messages of each LLM request |
| `CRM_BACKEND` | `sqlite` | `zoho` for Zoho CRM (see below) |
| `SIP_OUTBOUND_TRUNK_ID`, `HUMAN_TRANSFER_TO` | | 3CX, later |

## Repo layout

```
agent/      main.py (worker), postcall.py (finish left-over calls),
            calllog.py (per-call logs), logging_setup.py (JSON logs), prompts.py, tools.py, extraction.py,
            events.py, metrics.py, telephony.py, dial.py, check.py, make_token.py, config.py
crm/        port.py (CRMClient), audited.py (records every write), sqlite_adapter.py, zoho_adapter.py
demo/       the demo website: server.py (small API) and index.html (the page)
services/   docker-compose.yml (server), docker-compose.demo.yml (demo website), docker-compose.local.yml (laptop, no GPU), livekit*.yaml, stt_server.py
deploy/     deploy.sh (deploy + check + roll back), vllm_start.sh (vLLM without Docker)
.github/    workflows/ci-cd.yml
docs/       architecture.md, call-record.md
evals/      run_extraction.py (CRM-update accuracy), run_speech.py (STT/TTS accuracy)
loadtest/   calls-per-GPU test
tests/      unit and wiring tests (no GPU), real-LLM scenario tests (opt-in)
```

## Tests

```bash
pip install -e ".[dev]"
pytest                                               # no GPU or server needed (CI runs this)
RUN_LLM_TESTS=1 pytest tests/test_scenarios_llm.py   # real scenarios against the LLM in .env (Ollama or vLLM)
python -m evals.run_extraction                       # CRM-update accuracy on 15 sample transcripts
python -m evals.run_speech                           # STT/TTS accuracy (round trip, or --audio-dir real recordings)
```

**Calls per GPU:**
1. `python -m loadtest.make_audio`
2. `python -m loadtest.run_load --levels 1,5,10,20,30,40`

It prints p50/p95 reply latency for each level, and the GPU count needed for 2,800 calls/day.

## Switching to Zoho

Set `CRM_BACKEND=zoho` and the `ZOHO_*` variables (use a **sandbox** first).

These custom lead fields must exist in the Leads layout: `Budget`, `Preferred_Area`, `Property_Type`, `Timeline`, `Interest_Level`, `Callback_Time`, `Do_Not_Call`, `Project_Interest`. Their API names can be overridden with `ZOHO_FIELD_<NAME>`.

## Not covered yet

- 3CX trunk setup (the code is ready).
- Failover across several GPU servers.
- Recording consent and retention rules. (Calling limits are in place: see `MAX_CALLS_PER_DAY` and the other calling rules.)
