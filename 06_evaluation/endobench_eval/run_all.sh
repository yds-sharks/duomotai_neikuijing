#!/bin/bash
# EndoBench main-table evaluation (paper Table 1 + behavior analysis).
# Runs from the isolated endobench_eval directory.
# Shared code modules are referenced via absolute paths inside eval_endobench.py.
set -e
cd /mnt/data_1/yds/多模态/endobench_eval

PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
CKPT=${CKPT:-/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/ckpt_grpo_v4_u50}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,3

COMMON="--gen-device cuda:1 --retr-image-device cuda:0 --retr-text-device cuda:1 \
  --out-dir results_v2 --cand-cache results_v2/cand_cache_v2.jsonl \
  --use-organ-filter --use-weight-module"

# ---------- row 1: frozen generator, no retrieval ----------
$PY eval_endobench.py --mode baseline $COMMON

# ---------- row 2: frozen RAG pipeline (img6+txt6, no controller) ----------
$PY eval_endobench.py --mode vanilla_rag $COMMON

# ---------- row 3: proprietary multimodal controller ----------
$PY eval_endobench.py --mode gpt4o $COMMON

# ---------- row 4: AgenticRL controller (ours, greedy) ----------
$PY eval_endobench.py --mode agentic --ctrl-model "$CKPT" --ctrl-device cuda:0 $COMMON

# ---------- aggregate -> main table ----------
$PY build_paper_table.py --modes baseline,vanilla_rag,gpt4o,agentic \
  --out-dir results_v2 --save results_v2/main_table.json
