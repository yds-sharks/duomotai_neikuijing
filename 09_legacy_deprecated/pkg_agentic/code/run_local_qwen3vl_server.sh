#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/data_1/mwx/anaconda3/envs/endo/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data_10/mwx/huggingface_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PORT="${PORT:-8888}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
LOG_DIR="${LOG_DIR:-/mnt/data_1/yds/多模态/agentic/outputs/runtime/logs}"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/qwen3vl_vllm_${STAMP}.log"

echo "[start] model=${MODEL_PATH} gpu=${GPU_IDS} port=${PORT} log=${LOG_PATH}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --trust-remote-code \
  > "${LOG_PATH}" 2>&1
