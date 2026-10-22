# Demo website

A web page for showing the voice agent. You pick a lead, edit the prompt, choose the voice, press **Start call** and talk to the agent through your browser. When you end the call, the page shows what happened: the conversation, every tool the agent used (what it sent and what came back), and the CRM changes before and after.

It uses the **demo CRM only** (20 made-up leads). It never touches real leads. It has no login and listens on your own machine only (`127.0.0.1`).

## What you can change

| Setting | How it works |
|---|---|
| **Lead** | Any of the 20 demo leads. **Reset lead** puts one back to its starting values so you can repeat a demo. |
| **Prompt** | Edit the agent's instructions. Keep the `{placeholders}` and the tool names. A mistake is shown before the call starts. **Restore default** undoes your edit. |
| **Voice** | Male or female, American or British (Kokoro voices). |
| **Speech-to-text** | A list of speech servers. One that is not running shows as "(not running)". See below to add another. |

The text-to-speech engine (Kokoro) is one engine, so only its voice can be chosen. Speech-to-text is fixed per server, so a different engine needs its own server.

## Start it

You need Docker, Ollama and this project installed (`pip install -e ".[demo]"`, plus the agent's own packages from the main README).

1. **LLM**: start Ollama with the model from `.env` (`LLM_BASE_URL`, `LLM_MODEL`).
2. **Speech and LiveKit servers**: `docker compose -f services/docker-compose.local.yml --env-file .env up -d` (LiveKit, speech-to-text, text-to-speech).
3. **Demo CRM**: `python -m crm.sqlite_adapter --db data/demo_crm.db --seed`
4. **Agent worker**, using the same demo CRM and with demo mode on:
   ```
   set CRM_SQLITE_PATH=data/demo_crm.db
   set DEMO_MODE=1
   python -m agent.main dev
   ```
   (On Linux or Mac use `export` instead of `set`.)
5. **Demo website**: `python -m demo.server`, then open **http://localhost:8090**.

Use `localhost`, not the machine's name: browsers only allow the microphone on `localhost` or HTTPS. To open it from your laptop while it runs on the server, forward the ports over SSH (`ssh -L 8090:localhost:8090 -L 7880:localhost:7880 <server>`).

Run `python -m agent.check` first if something does not answer. It says which service is down.

## While the GPU is unavailable

The models then run on the CPU, so each reply can take many seconds and the first call is the slowest while the models load. Set `LLM_TIMEOUT=120` in `.env` (as `deploy/cpu_test.sh` does). The demo shows the flow, not production speed.

## Add another speech-to-text choice

Start another speech server on a free port (for example Whisper on 8002), then add it to `demo/options.json`:

```json
{"id": "whisper", "label": "Whisper", "base_url": "http://localhost:8002/v1", "model": "whisper-1"}
```

## Safety

- The agent only accepts a custom prompt, voice or speech server when **`DEMO_MODE=1`** *and* the call was started by this website. A normal worker ignores them, so real calls can never be changed this way.
- The website refuses to start if `CRM_BACKEND` is anything but the demo SQLite CRM.
- Every demo call is saved like a normal call (`data/calls/`), so it also shows up in the dashboard as a browser call.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `DEMO_MODE` | off | Set to `1` on the agent worker to allow per-call overrides |
| `DEMO_PORT` | 8090 | Port of the demo website |
| `CRM_SQLITE_PATH` | `data/demo_crm.db` for the website | Must be the same on the worker and the website |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | from `.env` | How the website reaches LiveKit |
| `DEMO_LIVEKIT_PUBLIC_URL` | `LIVEKIT_URL` | Where the **browser** reaches LiveKit, if different |
| `DEMO_OPTIONS` | `demo/options.json` | The voice and speech-to-text lists |
