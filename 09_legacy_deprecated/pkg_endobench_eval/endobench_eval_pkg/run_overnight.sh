#!/bin/bash
# 2x A100-80G 夜间计划（每卡一个任务）:
#   wave1: C (GPU0) + D (GPU1)   ~2.5-3h
#   wave2: G (GPU0)              ~2.5-3h
# 训练消融(SFT/RFT)在本地机器运行，此处不需要对应 ckpt。
cd "$(dirname "$0")"
mkdir -p logs results
echo "[wave1] C+D start $(date '+%m-%d %H:%M')"
bash run_config.sh C 0 & P1=$!
bash run_config.sh D 1 & P2=$!
wait $P1 $P2
echo "[wave2] G start $(date '+%m-%d %H:%M')"
bash run_config.sh G 0 & P3=$!
wait $P3
echo "ALL DONE $(date '+%m-%d %H:%M')"
