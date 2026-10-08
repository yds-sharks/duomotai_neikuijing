#!/bin/bash
# run_one_1k.sh <CFG> <GPU0> <GPU1> [CKPT] — stratified_1000 子集, 2 GPUs per run
set -e
cd /mnt/data_1/yds/多模态/endobench_eval
PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
CKPT=${4:-/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/ckpt_grpo_v4_u50}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=$1; G0=$2; G1=$3
EXTRA=""; SUFFIX=""
case $CFG in
  A) EXTRA="" ;;
  B) EXTRA="--no-mplus --no-mminus" ;;
  C) EXTRA="--selection-only" ;;
  D) EXTRA="--rewrite-only" ;;
  E) EXTRA="--no-mminus" ;;
  F) EXTRA="--no-mplus" ;;
  G) EXTRA="--sequential" ;;
  SFT) EXTRA=""; SUFFIX="_sft" ;;
  RFT) EXTRA=""; SUFFIX="_rft" ;;
  *) echo "unknown CFG $CFG"; exit 1 ;;
esac
CUDA_VISIBLE_DEVICES=$G0,$G1 $PY eval_endobench.py --mode agentic --ctrl-model $CKPT \
  --ctrl-device cuda:0 --gen-device cuda:1 \
  --retr-image-device cuda:0 --retr-text-device cuda:1 \
  --weight-device cuda:0 \
  --out-dir results_1k/$CFG$SUFFIX --cand-cache results_v2/cand_cache_v2.jsonl \
  --use-organ-filter --use-weight-module \
  --max-rounds 3 --qid-file stratified_1000.jsonl --latency $EXTRA 2>&1 | tee logs/1k_$CFG$SUFFIX.log
