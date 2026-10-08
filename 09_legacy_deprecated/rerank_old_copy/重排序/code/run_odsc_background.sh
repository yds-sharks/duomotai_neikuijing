#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${PROJECT_DIR}/code/rerank_odsc.py"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/data/tmp}"

mkdir -p "${OUTPUT_DIR}"

GPU_ID="${GPU_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TAG="${TAG:-top3}"

OUTPUT_JSONL="${OUTPUT_DIR}/endobench_odsc_${TAG}.jsonl"
DEBUG_JSONL="${OUTPUT_DIR}/endobench_odsc_${TAG}_debug.jsonl"
ERROR_JSONL="${OUTPUT_DIR}/endobench_odsc_${TAG}_errors.jsonl"
PROGRESS_LOG="${OUTPUT_DIR}/endobench_odsc_${TAG}_progress.log"
STDOUT_LOG="${OUTPUT_DIR}/endobench_odsc_${TAG}_stdout.log"

if [[ "${NUM_SHARDS}" -gt 1 ]]; then
  OUTPUT_JSONL="${OUTPUT_DIR}/endobench_odsc_${TAG}.shard${SHARD_INDEX}.jsonl"
  DEBUG_JSONL="${OUTPUT_DIR}/endobench_odsc_${TAG}_debug.shard${SHARD_INDEX}.jsonl"
  ERROR_JSONL="${OUTPUT_DIR}/endobench_odsc_${TAG}_errors.shard${SHARD_INDEX}.jsonl"
  PROGRESS_LOG="${OUTPUT_DIR}/endobench_odsc_${TAG}_progress.shard${SHARD_INDEX}.log"
  STDOUT_LOG="${OUTPUT_DIR}/endobench_odsc_${TAG}_stdout.shard${SHARD_INDEX}.log"
fi

CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_PATH}"
  --retrieval-export-jsonl "${RETRIEVAL_EXPORT_JSONL:-${PROJECT_DIR}/data/retrieval_export.jsonl}"
  --model-path "${MODEL_PATH:-Qwen/Qwen3-VL-Embedding-2B}"
  --device "cuda:${GPU_ID}"
  --batch-size "${BATCH_SIZE}"
  --num-shards "${NUM_SHARDS}"
  --shard-index "${SHARD_INDEX}"
  --output-jsonl "${OUTPUT_JSONL}"
  --debug-jsonl "${DEBUG_JSONL}"
  --error-jsonl "${ERROR_JSONL}"
  --progress-log "${PROGRESS_LOG}"
  --select-k "${SELECT_K:-3}"
  --lambda-disc "${LAMBDA_DISC:-1.0}"
  --lambda-cover "${LAMBDA_COVER:-0.5}"
  --lambda-cross "${LAMBDA_CROSS:-0.3}"
  --cross-sim-threshold "${CROSS_SIM_THRESHOLD:-0.5}"
)

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  CMD+=(--overwrite)
fi

if [[ -n "${LIMIT:-}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}" > "${STDOUT_LOG}" 2>&1 &
echo $!
