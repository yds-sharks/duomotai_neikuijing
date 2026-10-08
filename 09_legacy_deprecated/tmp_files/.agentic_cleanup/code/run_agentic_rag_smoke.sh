#!/bin/bash
# v0.4 smoke test: Agent directly controls first-stage evidence

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENTIC_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

INPUT_JSONL="${INPUT_JSONL:-${AGENTIC_ROOT}/outputs/qa_stage1_verified/qa_gold.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${AGENTIC_ROOT}/outputs/runtime/smoke_test.jsonl}"
LIMIT="${LIMIT:-5}"

echo "=== v0.4 Agent Controller Smoke Test ==="
echo "Input: $INPUT_JSONL"
echo "Output: $OUTPUT_JSONL"
echo "Limit: $LIMIT"
echo ""

python3 "${SCRIPT_DIR}/agentic_rag_pipeline.py" \
  --input-jsonl "$INPUT_JSONL" \
  --output-jsonl "$OUTPUT_JSONL" \
  --limit "$LIMIT" \
  --text-k 20 \
  --image-k 20 \
  --max-rounds 2 \
  --continue-on-error

echo ""
echo "=== Done ==="
echo "Check output: $OUTPUT_JSONL"
