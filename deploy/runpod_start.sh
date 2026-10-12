#!/usr/bin/env bash
# Run the GPU models on a RunPod pod (no AWS quota needed): Ollama + faster-whisper.
#
# Pod: 1x L40S 48 GB, template "RunPod PyTorch" (CUDA 12.x), 100 GB volume,
#      expose HTTP ports 11434,8001.
# Then in the pod terminal:
#   git clone <this repo> && cd ev-voice-agent
#   LLM_MODEL=qwen3:14b bash deploy/runpod_start.sh
#
# On the laptop .env:
#   LLM_BASE_URL=https://<POD_ID>-11434.proxy.runpod.net/v1
#   STT_BASE_URL=https://<POD_ID>-8001.proxy.runpod.net/v1
#   TTS_BASE_URL=http://localhost:8880/v1        (Kokoro on the laptop CPU)
set -euo pipefail

LLM_MODEL="${LLM_MODEL:-qwen3:14b}"
WHISPER_MODEL="${WHISPER_MODEL:-large-v3-turbo}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
# keep weights on the persistent volume
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/ollama-models}"
mkdir -p "$LOG_DIR" "$HF_HOME" "$OLLAMA_MODELS"
cd "$(dirname "$0")/.."

command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
pip install -q faster-whisper==1.2.1 fastapi==0.115.6 "uvicorn[standard]==0.34.0" python-multipart==0.0.20

echo "starting Ollama on :11434"
OLLAMA_HOST=0.0.0.0:11434 \
OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-16}" \
OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}" \
OLLAMA_MAX_QUEUE=512 OLLAMA_KEEP_ALIVE=-1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 \
  nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &

echo "starting faster-whisper ($WHISPER_MODEL) on :8001"
STT_ENGINE=whisper WHISPER_MODEL="$WHISPER_MODEL" WHISPER_DEVICE=cuda WHISPER_COMPUTE_TYPE=float16 \
  nohup python -m uvicorn --app-dir services stt_server:app --host 0.0.0.0 --port 8001 \
  > "$LOG_DIR/stt.log" 2>&1 &

until curl -sf http://localhost:11434/api/version >/dev/null; do sleep 2; done
echo "pulling $LLM_MODEL (first run downloads ~5-9 GB)..."
ollama pull "$LLM_MODEL"
# load into the GPU now, not on the first call
curl -sf http://localhost:11434/api/generate -d "{\"model\": \"$LLM_MODEL\", \"keep_alive\": -1}" >/dev/null
until curl -sf http://localhost:8001/health >/dev/null; do sleep 5; done
echo "ready. logs: $LOG_DIR"
