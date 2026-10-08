#!/usr/bin/env bash
set -euo pipefail

INPUT_JSONL="${INPUT_JSONL:-/mnt/data_1/yds/多模态/agentic/outputs/qa_stage1_prepared/pilot_qa_candidates.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/mnt/data_1/yds/多模态/agentic/outputs/runtime_smoke/pilot_retrieval.jsonl}"
LIMIT="${LIMIT:-2}"
TOPK="${TOPK:-5}"

python3 /mnt/data_1/yds/多模态/agentic/code/agentic_rag_pipeline.py \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --limit "${LIMIT}" \
  --topk "${TOPK}" \
  --skip-generator \
  --continue-on-error
