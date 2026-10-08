#!/usr/bin/env bash
set -euo pipefail

TEST_DIR="${TEST_DIR:-/mnt/data_1/yds/多模态/retrieval/test}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data_1/mwx/anaconda3/envs/endo/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data_10/mwx/huggingface_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8888/v1}"
API_KEY="${API_KEY:-EMPTY}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data_1/yds/多模态/agentic/outputs/endobench_eval}"
RETRIEVAL_REPLAY_JSONL="${RETRIEVAL_REPLAY_JSONL:-}"
LIMIT="${LIMIT:-500}"
TOPK="${TOPK:-3}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-qwen3vl_rag_agentic}"

if [[ -z "${RETRIEVAL_REPLAY_JSONL}" ]]; then
  echo "Missing RETRIEVAL_REPLAY_JSONL. Point it to a retrieval/rerank JSONL." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
RETRIEVAL_REPLAY_JSONL="${RETRIEVAL_REPLAY_JSONL}" \
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
"${PYTHON_BIN}" "${TEST_DIR}/evaluate_unified.py" \
  --backend qwen3-vl \
  --mode rag \
  --model "${MODEL_PATH}" \
  --base-url "${BASE_URL}" \
  --api-key "${API_KEY}" \
  --benchmark Saint-lsy/EndoBench \
  --split test \
  --dataset all \
  --task all \
  --scene all \
  --category all \
  --subtask all \
  --limit "${LIMIT}" \
  --num-runs 1 \
  --temperature 0.2 \
  --top-p 0.9 \
  --max-tokens 512 \
  --retrieval-query-mode question_options \
  --topk "${TOPK}" \
  --retrieval-batch-size 64 \
  --gpu "${GPU_IDS}" \
  --save-response \
  --output-dir "${OUTPUT_DIR}" \
  --output-subdir "${OUTPUT_SUBDIR}"
