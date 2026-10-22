#!/usr/bin/env bash
# docker compose with the right files for this server's .env (SERVER_LLM, SERVER_GPU), so you never type them.
#
#   bash deploy/stack.sh ps                                    # what is running
#   bash deploy/stack.sh logs --tail 50 agent                  # last log lines of one service
#   bash deploy/stack.sh exec agent python -m agent.check      # is every service OK?
#   bash deploy/stack.sh down                                  # stop everything
#
# deploy.sh sources this file to get the same command in COMPOSE.

_env_value() {
  grep -E "^$1=" .env 2>/dev/null | tail -n 1 | cut -d= -f2- | tr -d '[:space:]"' || true
}

# Sets COMPOSE to the docker compose command for this .env. Run it from the repo folder.
stack_compose() {
  local llm gpu
  llm=$(_env_value SERVER_LLM)
  gpu=$(_env_value SERVER_GPU)
  COMPOSE=(docker compose -f services/docker-compose.yml)
  # SERVER_LLM=ollama: the server's own Ollama instead of vLLM (services/docker-compose.ollama.yml)
  case "${llm:-vllm}" in
    ollama) COMPOSE+=(-f services/docker-compose.ollama.yml); echo "==> LLM: Ollama on this server" >&2 ;;
    vllm)   echo "==> LLM: vLLM" >&2 ;;
    *)      echo "SERVER_LLM in .env must be vllm or ollama, not '$llm'" >&2; return 1 ;;
  esac
  # SERVER_GPU=0: no GPU (tests only): STT and TTS on the CPU (services/docker-compose.cpu.yml)
  case "${gpu:-1}" in
    0)
      if [ "${llm:-vllm}" != ollama ]; then
        echo "SERVER_GPU=0 needs SERVER_LLM=ollama in .env: vLLM can't run without a GPU" >&2; return 1
      fi
      COMPOSE+=(-f services/docker-compose.cpu.yml); echo "==> no GPU: STT and TTS on the CPU (slow, for tests)" >&2 ;;
    1) ;;
    *) echo "SERVER_GPU in .env must be 1 or 0, not '$gpu'" >&2; return 1 ;;
  esac
  COMPOSE+=(--env-file .env)
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -euo pipefail
  cd "$(dirname "$0")/.."
  if [ ! -f .env ]; then
    echo "no .env here: copy .env.example to .env and fill it in first" >&2
    exit 1
  fi
  stack_compose
  exec "${COMPOSE[@]}" "$@"
fi
