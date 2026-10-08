#!/bin/bash
cd /mnt/data_1/yds/多模态/rerank_image_and_text/agentic

echo "$(date): starting vanilla_rag v2 (translate+organ+weight)" >> eval/results_v2/auto.log
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,3 \
  /mnt/data_1/yds/venvs/qwen35-train/bin/python -u eval/eval_endobench.py \
  --mode vanilla_rag \
  --gen-device cuda:1 --retr-image-device cuda:0 --retr-text-device cuda:1 \
  --out-dir eval/results_v2 \
  --cand-cache eval/results_v2/cand_cache_v2.jsonl \
  --use-organ-filter --use-weight-module \
  >> eval/results_v2/vanilla_rag_v2_full.log 2>&1

echo "$(date): vanilla_rag v2 done, starting agentic v2" >> eval/results_v2/auto.log
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,3 \
  /mnt/data_1/yds/venvs/qwen35-train/bin/python -u eval/eval_endobench.py \
  --mode agentic --ctrl-model train/ckpt_grpo_v4_u50 \
  --ctrl-device cuda:0 --gen-device cuda:1 \
  --retr-image-device cuda:0 --retr-text-device cuda:1 \
  --out-dir eval/results_v2 \
  --cand-cache eval/results_v2/cand_cache_v2.jsonl \
  --use-organ-filter --use-weight-module \
  >> eval/results_v2/agentic_u50_v2_full.log 2>&1

echo "$(date): agentic v2 done" >> eval/results_v2/auto.log
