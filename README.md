# AI Voice Agent: real-estate lead qualification (self-hosted, Ollama)

The agent calls a lead, **checks the lead in the CRM before it speaks**, asks the qualifying questions, and **updates the CRM** during and after the call.

Everything is open source and self-hosted. The LLM (**Qwen3-8B / 14B**) is served by **Ollama** everywhere: laptop, RunPod and EC2. The stack is sized for about **2,800 calls/day** (30–40 calls at the same time at peak). This is the English demo that proves the approach works.

## Architecture

The design uses four standard patterns:

| Pattern | Where |
|---|---|
| **Cascaded voice pipeline on a WebRTC SFU** (LiveKit reference architecture): VAD → STT → turn detector → LLM + tools → TTS | `agent/main.py` |
| **Microservices**: each model is its own service behind an OpenAI-compatible API | `services/` |
| **Hexagonal (ports & adapters)**: the agent only knows `CRMClient`; SQLite or Zoho plug in | `crm/` |
| **Event-driven post-call processing** (producer/consumer) | `agent/events.py` |

```
Caller (browser now / 3CX later)
        │ WebRTC
        ▼
 LiveKit Server ◄──► Agent worker:  Silero VAD → STT → turn detector → LLM (+tools) → TTS
 (self-hosted)                          │ OpenAI API    │ OpenAI API        │ OpenAI API   │ CRM port
                                        ▼               ▼                   ▼              ▼
                                 faster-whisper      Ollama             Kokoro       SQLite / Zoho
                                 (large-v3-turbo)  (qwen3:8b / 14b)
        call ends ─► CallEnded event ─► extraction (JSON schema) ─► fill missing fields + summary note ─► CRM
```

**What happens on a call:**
1. The agent loads the lead from the CRM: name, project of interest, and what's already known.
2. It confirms identity, then asks about interest, budget, area, property type and timeline.
3. It saves each fact with `update_lead` as soon as it hears it. Other tools: `schedule_callback`, `mark_do_not_call`, `end_call`.
4. After the call, a structured-extraction pass double-checks the transcript:
   - It fills fields the agent missed.
   - It logs any disagreements.
   - It enforces do-not-call.
   - It writes a summary note to the CRM.
   - It saves the call record (transcript, CRM actions, latency per turn) to `data/calls/`.
5. Leads already marked do-not-call are never called.

**Qwen3 "thinking" is switched off** for voice speed in three ways:
- Ollama's `think: false` flag on every request.
- `/no_think` at the end of the system prompt.
- Any `<think>…</think>` text is stripped before text-to-speech, so it is never spoken.

## Repo layout

```
agent/      main.py (worker), prompts.py, tools.py, extraction.py, events.py, metrics.py, make_token.py
crm/        port.py (CRMClient), sqlite_adapter.py (demo), zoho_adapter.py
services/   stt_server.py, docker-compose.yml (EC2), docker-compose.local.yml (laptop), livekit*.yaml
deploy/     runpod_start.sh (Ollama + whisper on a RunPod pod)
evals/      run_extraction.py: CRM-update accuracy on 15 sample transcripts
loadtest/   make_audio.py, sim_caller.py, run_load.py: calls-per-GPU test
tests/      unit + wiring tests (no GPU), real-LLM scenario tests (opt-in)
```

## Where to run it

| Where | What it proves | Cost |
|---|---|---|
| **Laptop + Ollama** | The real Qwen3 model writes the right CRM updates (text mode) and the full call flow works. On a 4 GB GPU the model runs partly on CPU, so expect 5–20 s per voice turn. | Free |
| **RunPod L40S pod** | Real voice latency with the real models, while AWS GPU quota is pending | ~$0.9/hr |
| **EC2 g6e.xlarge** | Final latency and **calls per GPU** on the production hardware | ~$1.9/hr from AWS credits |

The code is identical everywhere; only `.env` changes.

---

## Setup (all profiles)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,loadtest]"
cp .env.example .env                                    # then edit
python -m crm.sqlite_adapter --seed                     # 20 demo leads in data/crm.db
python -m agent.main download-files                     # VAD + turn-detector weights (once)
pytest                                                  # unit + wiring tests, no GPU needed
```

## Laptop with Ollama (start today, free)

1. Install [Ollama](https://ollama.com).
   - Set the context size: `OLLAMA_CONTEXT_LENGTH=8192`. On Windows, set it in *System → Environment Variables*, then restart Ollama.
   - Pull the model:
     ```bash
     ollama pull qwen3:8b          # ~5 GB  (or qwen3:14b, ~9 GB)
     ```
2. In `.env`, keep `LLM_BASE_URL=http://localhost:11434/v1` and set `LLM_MODEL=qwen3:8b`.
3. **Text-mode tests with the real model.** Speed doesn't matter here, so these are the main laptop results:
   ```bash
   python -m evals.run_extraction                          # demo test 3: CRM field accuracy (target >= 95%)
   python -m evals.run_extraction --model qwen3:14b        # compare models
   RUN_LLM_TESTS=1 pytest tests/test_scenarios_llm.py -v   # demo test 4: callback / not interested / DNC / wrong person
   ```
4. **Voice call (slow on a 4 GB GPU, just to see the flow):**
   ```bash
   docker compose -f services/docker-compose.local.yml --env-file .env up -d   # LiveKit + STT (CPU) + Kokoro (CPU)
   python -m agent.main dev
   python -m agent.make_token --lead L003
   ```
   Then open https://agents-playground.livekit.io in **Chrome**, choose manual connect, and paste the URL and token. Afterwards, check the result with `python -m crm.sqlite_adapter --show`.

   No native Ollama install? Add `--profile ollama` to the compose command to run Ollama in Docker.

## RunPod (no AWS quota needed)

1. Create a pod with:
   - 1× **L40S 48 GB**
   - The "RunPod PyTorch" template
   - A 100 GB volume
   - HTTP ports `11434,8001` exposed
2. In the pod terminal: `git clone <repo> && cd ev-voice-agent && LLM_MODEL=qwen3:14b bash deploy/runpod_start.sh`
3. On the laptop `.env`, use the RunPod URLs (see `.env.example`). Then:
   ```bash
   docker compose -f services/docker-compose.local.yml --env-file .env up -d livekit tts
   python -m agent.main dev
   ```
   Make calls from the Playground as above.

## EC2 g6e.xlarge (final numbers)

**AWS checklist:**
1. **GPU quota:** Service Quotas → EC2 → *Running On-Demand G and VT instances* → **4 vCPUs**. If AWS refuses, reopen the case with a detailed use case.
2. **Region:** pick one with g6e (`us-east-1`, `us-west-2`, `eu-west-1`).
3. **AMI:** *AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*. It includes the driver, Docker and the nvidia-container-toolkit.
4. **Disk:** 200 GB gp3.
5. **Security group, your IP only:**
   - 22/tcp
   - 7880/tcp
   - 7881/tcp
   - **7882/udp** (WebRTC media)
6. **Budget alert:** Billing → Budgets. **Stop the instance when you're not testing.**

```bash
git clone <repo> && cd ev-voice-agent
cp .env.example .env            # set LIVEKIT_API_SECRET, LLM_MODEL=qwen3:14b (or qwen3:8b)
mkdir -p data && docker run --rm -v $PWD:/app -w /app python:3.11-slim \
  sh -c "pip install -q pydantic && python -m crm.sqlite_adapter --seed"
docker compose -f services/docker-compose.yml --env-file .env up -d --build
docker compose -f services/docker-compose.yml logs -f ollama-pull   # first start downloads the model
```

The compose file runs:
- LiveKit
- **Ollama** (GPU, model kept loaded, `OLLAMA_NUM_PARALLEL=16` requests at once)
- faster-whisper (GPU)
- Kokoro (GPU)
- the agent

**Test calls from your laptop:**
- Tunnel the signalling port: `ssh -L 7880:localhost:7880 ubuntu@<EC2-IP>`.
- In the laptop `.env`, set `LIVEKIT_URL=ws://localhost:7880` with the same key and secret.
- Run `python -m agent.make_token --lead L001` and use the Playground in Chrome.
- Media goes directly to the instance over UDP 7882.

**Calls-per-GPU test** (demo test 6). Run it from the laptop or a small CPU instance in the same region, not on the GPU box:
```bash
python -m loadtest.make_audio                         # caller voice via Kokoro (TTS_BASE_URL must reach the box)
python -m loadtest.run_load --levels 1,5,10,20,30,40
```
It prints p50/p95 reply latency for each level and the GPU count needed for 2,800 calls/day. Repeat it for:
- `LLM_MODEL=qwen3:8b` and `qwen3:14b`
- `OLLAMA_NUM_PARALLEL` set to 8, 16 and 32 (it's the knob for concurrent LLM requests; each slot uses its own context memory)

> **Watch the CPU, not only the GPU.** g6e.xlarge has only 4 vCPUs, and each call runs VAD and the turn-detector model on the CPU. If `load_avg_1m` in the results climbs above ~4 before the GPU is busy, the CPU is the limit. Either use g6e.2xlarge (8 vCPU), or run LiveKit + the agent on a normal CPU instance such as c7i.2xlarge (no GPU quota needed) and keep only the models on the GPU box.
>
> The LLM is reached through the standard OpenAI API. If Ollama's request queueing turns out to be the limit, any other OpenAI-compatible server is a `.env` change, with no code change.

## The demo tests

| # | Test | How | Pass |
|---|---|---|---|
| 1 | Natural English call with interruptions | Playground calls | No talking over the caller, stays on script |
| 2 | Agent checks the lead before speaking | Playground call | Greets by name, knows the project |
| 3 | CRM updates correct, 8B vs 14B | `python -m evals.run_extraction --model qwen3:8b` | ≥ 95% of fields correct |
| 4 | Scenarios: callback / not interested / do-not-call / wrong person | `RUN_LLM_TESTS=1 pytest tests/test_scenarios_llm.py` | All pass |
| 5 | Latency per turn | `data/calls/*.json` → `metrics`, or `run_load --levels 1` | p50 < 1 s, p95 < 1.5 s (GPU) |
| 6 | **Calls per GPU** | `python -m loadtest.run_load` | Highest level with p95 ≤ 1.5 s → GPU count for 2,800/day |

## Dry run without any GPU

`tests/stub_models.py` serves fake STT, LLM and TTS on one port. It lets you exercise LiveKit, the agent, the CRM and the load-test harness on any machine:
```bash
python -m tests.stub_models --port 9000
# .env: LLM_BASE_URL / STT_BASE_URL / TTS_BASE_URL = http://localhost:9000/v1 ; TURN_DETECTION=vad
```

## Switching to Zoho

Set `CRM_BACKEND=zoho` and the `ZOHO_*` variables (use a **sandbox** first). Custom lead fields (`Budget`, `Preferred_Area`, `Property_Type`, `Timeline`, `Interest_Level`, `Callback_Time`, `Do_Not_Call`, `Project_Interest`) must exist in the Leads layout. Their API names can be overridden with `ZOHO_FIELD_<NAME>`.

## Out of scope for this demo

Arabic, 3CX/SIP integration, production monitoring, failover, and compliance (recording consent, retention).
