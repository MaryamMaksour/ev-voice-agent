#!/usr/bin/env bash
# One command to test the whole voice agent on a server without a working GPU:
#
#   bash deploy/cpu_test.sh
#
# It sets the .env lines for a CPU test (SERVER_LLM=ollama, SERVER_GPU=0, LLM_TIMEOUT=120), starts
# everything, waits until every service is OK, loads the demo CRM (first time only) and makes one
# simulated phone call through the full pipeline. Safe to run again: it never resets the CRM twice.
# Slow on a CPU by design: this checks that everything connects, not how fast it is.
set -euo pipefail
cd "$(dirname "$0")/.."

set_env() {   # set_env KEY VALUE: replace the line in .env, or add it
  if grep -qE "^$1=" .env; then sed -i "s|^$1=.*|$1=$2|" .env; else echo "$1=$2" >> .env; fi
}

if [ ! -f .env ]; then
  cp .env.example .env
  chmod 600 .env
  echo "==> made .env from .env.example"
fi
if grep -qE '^LIVEKIT_API_SECRET=change-me' .env; then
  set_env LIVEKIT_API_SECRET "$(openssl rand -hex 32)"
  echo "==> made a new LiveKit secret"
fi
set_env SERVER_LLM ollama
set_env SERVER_GPU 0
timeout=$(grep -E '^LLM_TIMEOUT=' .env | tail -n 1 | cut -d= -f2- | cut -d. -f1 || true)
if ! [ "${timeout:-0}" -ge 120 ] 2>/dev/null; then set_env LLM_TIMEOUT 120; fi

echo "==> settings used:"
grep -E '^(SERVER_LLM|SERVER_GPU|LLM_BASE_URL|LLM_MODEL|LLM_TIMEOUT)=' .env | sed 's/^/    /'

source deploy/stack.sh
stack_compose

mkdir -p data/recordings data/calls logs
chmod 777 data/recordings   # the Egress container writes here as its own user

echo "==> building and starting (the first time takes 10-20 minutes: it downloads the speech models)"
GIT_SHA=$(git rev-parse --short HEAD) "${COMPOSE[@]}" up -d --build --remove-orphans

echo "==> waiting until every service is OK (up to 15 minutes)"
ok=0
for i in $(seq 1 60); do
  if "${COMPOSE[@]}" exec -T agent python -m agent.check; then ok=1; break; fi
  echo "    not ready yet ($i/60), trying again in 15 s"
  sleep 15
done
if [ "$ok" != 1 ]; then
  echo "==> something is not OK. Last log lines of each service:" >&2
  for s in stt tts livekit agent; do
    echo "----- $s" >&2
    "${COMPOSE[@]}" logs --tail 20 "$s" >&2 || true
  done
  exit 1
fi

if [ ! -f data/crm.db ]; then
  echo "==> loading the demo CRM (20 leads, L001-L020)"
  "${COMPOSE[@]}" exec -T agent python -m crm.sqlite_adapter --seed
fi

echo "==> making the test caller's voice"
"${COMPOSE[@]}" exec -T agent python -m loadtest.make_audio

echo "==> one test call through the full pipeline (on a CPU each reply can take a while)"
"${COMPOSE[@]}" exec -T agent python -m loadtest.run_load --levels 1 --reply-timeout 120

echo
echo "==> done. The call record is the newest file in data/calls/ (the dashboard reads it from there)."
echo "    Everything is still running. To stop it:  bash deploy/stack.sh down"
