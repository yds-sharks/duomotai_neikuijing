#!/bin/bash
# Agentic EndoBench evaluation pipeline (v2: translate + organ + weight)
# Runs from the isolated endobench_eval directory.
cd /mnt/data_1/yds/多模态/endobench_eval

CTRL_MODEL="/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/ckpt_grpo_v4_u50"
PYTHON="/mnt/data_1/yds/venvs/qwen35-train/bin/python"
COMMON_FLAGS="--gen-device cuda:1 --retr-image-device cuda:0 --retr-text-device cuda:1 \
  --out-dir results_v2 --cand-cache results_v2/cand_cache_v2.jsonl \
  --use-organ-filter --use-weight-module"

echo "$(date): starting vanilla_rag v2 (translate+organ+weight)" >> logs/auto.log
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,3 \
  "$PYTHON" -u eval_endobench.py --mode vanilla_rag $COMMON_FLAGS \
  >> logs/vanilla_rag_v2_full.log 2>&1

echo "$(date): vanilla_rag v2 done, starting agentic v2" >> logs/auto.log
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,3 \
  "$PYTHON" -u eval_endobench.py --mode agentic --ctrl-model "$CTRL_MODEL" \
  --ctrl-device cuda:0 $COMMON_FLAGS \
  >> logs/agentic_u50_v2_full.log 2>&1

echo "$(date): agentic v2 done" >> logs/auto.log
