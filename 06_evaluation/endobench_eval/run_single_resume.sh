#!/bin/bash
# Resume single-round agentic eval (3645/6832 done, skip existing)
# GPU 0+1
set -e
cd /mnt/data_1/yds/多模态/endobench_eval

PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
CKPT=/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/ckpt_grpo_v4_u50
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

$PY eval_endobench.py --mode agentic --ctrl-model "$CKPT" \
  --ctrl-device cuda:0 --gen-device cuda:1 \
  --retr-image-device cuda:0 --retr-text-device cuda:1 \
  --weight-device cuda:0 \
  --out-dir results_v2 --cand-cache results_v2/cand_cache_v2.jsonl \
  --use-organ-filter --use-weight-module \
  --max-rounds 1 2>&1 | tee logs/single_resume.log
