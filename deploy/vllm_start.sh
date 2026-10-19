#!/usr/bin/env bash
# Start Qwen3.5-9B on vLLM without Docker (a GPU server where docker compose isn't an option).
#
#   bash deploy/vllm_start.sh            # serves on :8000, logs to logs/vllm/vllm.log
#
# Same flags as the vllm service in services/docker-compose.yml. Then in .env:
#   LLM_BASE_URL=http://<server-ip>:8000/v1   LLM_MODEL=qwen3.5-9b
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-9B}"
SERVED="${VLLM_SERVED_NAME:-qwen3.5-9b}"
LOG_DIR="${LOGS_DIR:-logs}/vllm"
mkdir -p "$LOG_DIR"

# Qwen3.5 needs vLLM 0.17 or newer; pinned to the same version as docker-compose.yml
python3 -c "import vllm" 2>/dev/null || pip install "vllm==0.27.1"

echo "starting vLLM: $MODEL as '$SERVED' on :8000 (log: $LOG_DIR/vllm.log)"
# shellcheck disable=SC2086
nohup vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 --port 8000 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-16384}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-48}" \
  --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.55}" \
  ${VLLM_EXTRA_ARGS:-} \
  >> "$LOG_DIR/vllm.log" 2>&1 &

echo "waiting for the model to load (first start downloads ~19 GB)..."
until curl -sf http://localhost:8000/health >/dev/null; do sleep 5; done
curl -s http://localhost:8000/v1/models
echo
echo "vLLM is up."
