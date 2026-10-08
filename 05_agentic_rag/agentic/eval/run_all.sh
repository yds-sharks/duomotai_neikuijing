#!/bin/bash
# EndoBench main-table evaluation (paper Table 1 + behavior analysis).
# Run AFTER GRPO training finishes (needs 3 GPUs: controller / generator / retriever).
# Each mode is resumable: per-sample JSONL append + qid skip.
#
# GPU layout mirrors training: ctrl=cuda:0, generator(scorer)=cuda:1, retr image=cuda:0 text=cuda:1.
set -e
cd "$(dirname "$0")/.."
PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
CKPT=${CKPT:-train/ckpt_grpo_v4}          # final GRPO ckpt (or ckpt_grpo_v4_u150 etc.)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- smoke first (64 samples per mode) ----------
# $PY eval/eval_endobench.py --mode baseline    --limit 64
# $PY eval/eval_endobench.py --mode vanilla_rag --limit 64
# $PY eval/eval_endobench.py --mode agentic --ctrl-model $CKPT --limit 64

# ---------- row 1: frozen generator, no retrieval ----------
$PY eval/eval_endobench.py --mode baseline \
  --gen-device cuda:1

# ---------- row 2: frozen RAG pipeline (img6+txt6, no controller) ----------
$PY eval/eval_endobench.py --mode vanilla_rag \
  --gen-device cuda:1 --retr-image-device cuda:0 --retr-text-device cuda:1

# ---------- row 3: proprietary multimodal controller (same prompt/action space) ----------
$PY eval/eval_endobench.py --mode gpt4o \
  --gen-device cuda:1 --retr-image-device cuda:0 --retr-text-device cuda:1

# ---------- row 4: AgenticRL controller (ours, greedy) ----------
$PY eval/eval_endobench.py --mode agentic \
  --ctrl-model "$CKPT" --ctrl-device cuda:0 \
  --gen-device cuda:1 --retr-image-device cuda:0 --retr-text-device cuda:1

# ---------- aggregate -> main table + behavior + utility ----------
$PY eval/build_paper_table.py --modes baseline,vanilla_rag,gpt4o,agentic \
  --out-dir eval/results --save eval/results/main_table.json
