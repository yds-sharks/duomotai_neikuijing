#!/bin/bash
# Multi-round evaluation: M+ (retained) + M- (failed query history)
# Runs 3 ablation groups serially on GPU 3.
set -e
cd /mnt/data_1/yds/多模态/endobench_eval

PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
CKPT=/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/ckpt_grpo_v4_u50
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

# All on GPU 3 (cuda:0 within CUDA_VISIBLE_DEVICES=3)
COMMON="--gen-device cuda:0 --retr-image-device cuda:0 --retr-text-device cuda:0 \
  --ctrl-device cuda:0 --weight-device cuda:0 \
  --cand-cache results_v2/cand_cache_v2.jsonl \
  --use-organ-filter --use-weight-module \
  --max-rounds 2"

echo "============================================"
echo "[1/3] mr_full (M+ + M-)"
echo "============================================"
$PY eval_endobench.py --mode agentic --ctrl-model "$CKPT" \
  --out-dir results_v2/mr_full $COMMON 2>&1 | tee logs/mr_full.log

echo "============================================"
echo "[2/3] mr_no_mplus (no M+, only M-)"
echo "============================================"
$PY eval_endobench.py --mode agentic --ctrl-model "$CKPT" \
  --out-dir results_v2/mr_no_mplus --no-mplus $COMMON 2>&1 | tee logs/mr_no_mplus.log

echo "============================================"
echo "[3/3] mr_no_mminus (no M-, only M+)"
echo "============================================"
$PY eval_endobench.py --mode agentic --ctrl-model "$CKPT" \
  --out-dir results_v2/mr_no_mminus --no-mminus $COMMON 2>&1 | tee logs/mr_no_mminus.log

echo "============================================"
echo "All 3 multi-round eval groups completed!"
echo "============================================"
