#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/data_1/yds/多模态/retrieval/500_0.1-0.4"
TEST_DIR="/mnt/data_1/yds/多模态/retrieval/test"
PYTHON_BIN="/mnt/data_1/mwx/anaconda3/envs/endo/bin/python"
MODEL_PATH="/mnt/data_10/mwx/huggingface_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
BASE_URL="http://127.0.0.1:8888/v1"
API_KEY="EMPTY"
GPU_IDS="0,1,2,3"
PORT="8888"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${ROOT_DIR}/run_qwen3_vl_rag_full500_${STAMP}.log"
VLLM_LOG="${ROOT_DIR}/run_qwen3_vl_rag_full500_${STAMP}_vllm.log"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${RUN_LOG}"
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "Stopping vLLM server pid=${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT

start_server() {
  log "Starting vLLM with local model path: ${MODEL_PATH}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    > "${VLLM_LOG}" 2>&1 &
  SERVER_PID=$!
  log "vLLM pid=${SERVER_PID}"
}

wait_for_server() {
  log "Waiting for vLLM readiness on ${BASE_URL}"
  for _ in $(seq 1 180); do
    if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
from urllib import request
request.urlopen("http://127.0.0.1:8888/v1/models", timeout=5)
PY
    then
      log "vLLM is ready"
      return 0
    fi
    sleep 2
  done
  log "vLLM did not become ready in time"
  return 1
}

run_one() {
  local replay_jsonl="$1"
  local output_subdir="$2"

  log "Starting evaluation: replay=${replay_jsonl} output_subdir=${output_subdir}"
  RETRIEVAL_REPLAY_JSONL="${replay_jsonl}" \
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
      --limit 500 \
      --num-runs 1 \
      --temperature 0.2 \
      --top-p 0.9 \
      --max-tokens 512 \
      --retrieval-query-mode question_options \
      --topk 3 \
      --retrieval-batch-size 64 \
      --gpu "${GPU_IDS}" \
      --save-response \
      --output-dir "${ROOT_DIR}" \
      --output-subdir "${output_subdir}" \
      >> "${RUN_LOG}" 2>&1
  log "Finished evaluation: ${output_subdir}"
}

log "Full 500-sample run started"
start_server
wait_for_server

run_one "${ROOT_DIR}/retrieval_500_nodedup_w090_010.jsonl" "eval_w090_010_full500"
run_one "${ROOT_DIR}/retrieval_500_nodedup_w080_020.jsonl" "eval_w080_020_full500"
run_one "${ROOT_DIR}/retrieval_500_nodedup_w070_030.jsonl" "eval_w070_030_full500"
run_one "${ROOT_DIR}/retrieval_500_nodedup_w060_040.jsonl" "eval_w060_040_full500"

log "All four evaluations completed"
