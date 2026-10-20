#!/usr/bin/env bash
# Deploy one commit on the GPU server, check every service, and roll back if the check fails.
#
#   DEPLOY_DIR=/opt/ev-voice-agent bash deploy/deploy.sh <git-sha>
#
# GitHub Actions runs this on the server (self-hosted runner, see README "CI/CD"); you can run it by hand too.
# DEPLOY_DIR is the server's clone of the repo, where .env, data/ and logs/ live. Default: this repo.
set -euo pipefail

TARGET="${1:?usage: deploy.sh <git-sha or branch>}"
DIR="${DEPLOY_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$DIR"

if [ ! -f .env ]; then
  echo "no .env in $DIR: copy .env.example to .env and fill it in first" >&2
  exit 1
fi

COMPOSE=(docker compose -f services/docker-compose.yml)
# SERVER_LLM=ollama in .env: use the server's own Ollama instead of vLLM (services/docker-compose.ollama.yml)
SERVER_LLM=$(grep -E '^SERVER_LLM=' .env | tail -n 1 | cut -d= -f2- | tr -d '[:space:]"' || true)
case "${SERVER_LLM:-vllm}" in
  ollama) COMPOSE+=(-f services/docker-compose.ollama.yml); echo "==> LLM: Ollama on this server" ;;
  vllm)   echo "==> LLM: vLLM" ;;
  *)      echo "SERVER_LLM in .env must be vllm or ollama, not '$SERVER_LLM'" >&2; exit 1 ;;
esac
# SERVER_GPU=0 in .env: no GPU (tests only): STT and TTS on the CPU (services/docker-compose.cpu.yml)
SERVER_GPU=$(grep -E '^SERVER_GPU=' .env | tail -n 1 | cut -d= -f2- | tr -d '[:space:]"' || true)
case "${SERVER_GPU:-1}" in
  0)
    if [ "${SERVER_LLM:-vllm}" != ollama ]; then
      echo "SERVER_GPU=0 needs SERVER_LLM=ollama in .env: vLLM can't run without a GPU" >&2; exit 1
    fi
    COMPOSE+=(-f services/docker-compose.cpu.yml); echo "==> no GPU: STT and TTS on the CPU (slow, for tests)" ;;
  1) ;;
  *) echo "SERVER_GPU in .env must be 1 or 0, not '$SERVER_GPU'" >&2; exit 1 ;;
esac
COMPOSE+=(--env-file .env)
HEALTH_TRIES="${HEALTH_TRIES:-18}"     # x 10 s = 3 minutes for the agent to come up

PREV=$(git rev-parse HEAD)
git fetch --quiet --prune origin
git checkout --quiet --detach "$TARGET"
NEW=$(git rev-parse HEAD)
mkdir -p data/recordings data/calls logs
chmod 777 data/recordings   # the Egress container writes here as its own user

up() {
  local sha
  sha=$(git rev-parse --short HEAD)
  echo "==> starting $sha"
  # running calls finish first: the old agent gets up to 10 minutes (stop_grace_period)
  GIT_SHA="$sha" "${COMPOSE[@]}" up -d --build --remove-orphans
}

healthy() {
  for i in $(seq 1 "$HEALTH_TRIES"); do
    if "${COMPOSE[@]}" exec -T agent python -m agent.check; then
      return 0
    fi
    echo "==> not healthy yet ($i/$HEALTH_TRIES), retrying in 10 s"
    sleep 10
  done
  return 1
}

if [ "$PREV" = "$NEW" ]; then
  echo "==> $NEW is already checked out; rebuilding anyway"
fi

up
if healthy; then
  echo "==> deployed $(git rev-parse --short HEAD)"
  # free disk: untagged leftovers of older builds (tagged versions stay, for a manual rollback)
  docker image prune -f >/dev/null || true
  exit 0
fi

echo "==> health check failed: rolling back to $(git rev-parse --short "$PREV")" >&2
"${COMPOSE[@]}" logs --tail 100 agent >&2 || true
git checkout --quiet --detach "$PREV"
up
healthy || echo "==> the previous version is not healthy either: check the server by hand" >&2
exit 1
