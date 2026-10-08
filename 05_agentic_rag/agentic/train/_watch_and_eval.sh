#!/bin/bash
cd /mnt/data_1/yds/多模态/rerank_image_and_text/agentic
echo "[watch] waiting for training [done]... $(date +%H:%M:%S)"
while ! grep -q "\[done\]" train/sft_full_v1.log 2>/dev/null; do sleep 30; done
echo "[watch] training done at $(date +%H:%M:%S); waiting 45s for GPU release"
sleep 45
echo "[watch] launching eval $(date +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=0 /mnt/data_1/yds/venvs/qwen35-train/bin/python train/eval_ctrl_ckpt.py \
  --ckpts train/ckpt_qwen35_ctrl_full_v1_u101,train/ckpt_qwen35_ctrl_full_v1_u151,train/ckpt_qwen35_ctrl_full_v1 \
  --val train/sft_ctrl_val.jsonl --val-len-file train/seq_len_val.jsonl \
  --max-len 4608 --device cuda:0 --out-dir train/eval_ctrl
echo "[watch] eval finished $(date +%H:%M:%S)"
