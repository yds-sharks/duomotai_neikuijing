#!/bin/bash
# run_smoke_one.sh <CFG> <GPU0> <GPU1>  — 10-question smoke for one ablation config
set -e
cd /mnt/data_1/yds/多模态/endobench_eval
PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
CKPT=/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/ckpt_grpo_v4_u50
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=$1; G0=$2; G1=$3
EXTRA=""
case $CFG in
  A) EXTRA="" ;;
  B) EXTRA="--no-mplus --no-mminus" ;;
  C) EXTRA="--selection-only" ;;
  D) EXTRA="--rewrite-only" ;;
  E) EXTRA="--no-mminus" ;;
  F) EXTRA="--no-mplus" ;;
  G) EXTRA="--sequential" ;;
  *) echo "unknown CFG $CFG"; exit 1 ;;
esac
CUDA_VISIBLE_DEVICES=$G0,$G1 $PY eval_endobench.py --mode agentic --ctrl-model $CKPT \
  --ctrl-device cuda:0 --gen-device cuda:1 \
  --retr-image-device cuda:0 --retr-text-device cuda:1 \
  --weight-device cuda:0 \
  --out-dir smoke/$CFG --cand-cache results_v2/cand_cache_v2.jsonl \
  --use-organ-filter --use-weight-module \
  --max-rounds 3 --qid-file smoke10.jsonl --latency $EXTRA 2>&1 | tee logs/smoke_$CFG.log
