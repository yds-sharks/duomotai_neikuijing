#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/data_1/yds/venvs/qwen3vl-rerank/bin/python}"
SCRIPT_PATH="${SCRIPT_PATH:-/mnt/data_1/yds/多模态/核心代码梳理/rerank_multimodal_ppr.py}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data_1/yds/多模态/agentic/outputs/rerank_batch}"
mkdir -p "${OUTPUT_DIR}"

GPU_ID="${GPU_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TAG="${TAG:-agentic_top5}"
SELECT_K="${SELECT_K:-5}"

OUTPUT_JSONL="${OUTPUT_DIR}/endobench_multimodal_ppr_${TAG}.jsonl"
DEBUG_JSONL="${OUTPUT_DIR}/endobench_multimodal_ppr_${TAG}_debug.jsonl"
ERROR_JSONL="${OUTPUT_DIR}/endobench_multimodal_ppr_${TAG}_errors.jsonl"
PROGRESS_LOG="${OUTPUT_DIR}/endobench_multimodal_ppr_${TAG}_progress.log"
STDOUT_LOG="${OUTPUT_DIR}/endobench_multimodal_ppr_${TAG}_stdout.log"

if [[ "${NUM_SHARDS}" -gt 1 ]]; then
  OUTPUT_JSONL="${OUTPUT_DIR}/endobench_multimodal_ppr_${TAG}.shard${SHARD_INDEX}.jsonl"
  DEBUG_JSONL="${OUTPUT_DIR}/endobench_multimodal_ppr_${TAG}_debug.shard${SHARD_INDEX}.jsonl"
  ERROR_JSONL="${OUTPUT_DIR}/endobench_multimodal_ppr_${TAG}_errors.shard${SHARD_INDEX}.jsonl"
  PROGRESS_LOG="${OUTPUT_DIR}/endobench_multimodal_ppr_${TAG}_progress.shard${SHARD_INDEX}.log"
  STDOUT_LOG="${OUTPUT_DIR}/endobench_multimodal_ppr_${TAG}_stdout.shard${SHARD_INDEX}.log"
fi

CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_PATH}"
  --device "cuda:${GPU_ID}"
  --batch-size "${BATCH_SIZE}"
  --num-shards "${NUM_SHARDS}"
  --shard-index "${SHARD_INDEX}"
  --select-k "${SELECT_K}"
  --output-jsonl "${OUTPUT_JSONL}"
  --debug-jsonl "${DEBUG_JSONL}"
  --error-jsonl "${ERROR_JSONL}"
  --progress-log "${PROGRESS_LOG}"
)

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "[run] ${CMD[*]}"
nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}" > "${STDOUT_LOG}" 2>&1 &
echo "[pid] $!"
echo "[stdout] ${STDOUT_LOG}"
