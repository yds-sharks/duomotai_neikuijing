#!/bin/bash
# 本地 4x A6000 夜间三波:
#   wave1: A(GPU0,1) + B(GPU2,3)
#   wave2: E(GPU0,1) + F(GPU2,3)
#   wave3: SFT(GPU0,1) + RFT(GPU2,3)  — 训练消融, Clean Full 设置
cd /mnt/data_1/yds/多模态/endobench_eval
TRAIN=/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train
echo "[wave1] A+B start $(date '+%m-%d %H:%M')"
bash run_one_1k.sh A 0 1 & P1=$!
bash run_one_1k.sh B 2 3 & P2=$!
wait $P1 $P2
echo "[wave2] E+F start $(date '+%m-%d %H:%M')"
bash run_one_1k.sh E 0 1 & P3=$!
bash run_one_1k.sh F 2 3 & P4=$!
wait $P3 $P4
echo "[wave3] SFT+RFT start $(date '+%m-%d %H:%M')"
bash run_one_1k.sh SFT 0 1 $TRAIN/ckpt_qwen35_ctrl_full_v1 & P5=$!
bash run_one_1k.sh RFT 2 3 $TRAIN/ckpt_ctrl_rft_v3 & P6=$!
wait $P5 $P6
echo "ALL LOCAL RUNS DONE $(date '+%m-%d %H:%M')"
