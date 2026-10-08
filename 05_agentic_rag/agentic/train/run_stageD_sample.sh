#!/bin/bash
# Stage D(v2.1):统一口径打分(u=Δ logP_options)+组内 z 归一化优势
# 输入:Stage C 4 片 train/rollouts_selected_sample_v1_sh{i}.jsonl(逐片处理)
# 生成器 Qwen3-VL-8B 冻结;qwen3vl-rerank env;4 卡 4 片
# 验证指标:组内有梯度占比(目标 >70%,v1 接近 0)+平均 leakage_rate
set -u
cd /mnt/data_1/yds/多模态/rerank_image_and_text/agentic
PY=/mnt/data_1/yds/venvs/qwen3vl-rerank/bin/python
NSHARD=4
for i in $(seq 0 $((NSHARD-1))); do
  IN="train/rollouts_selected_sample_v1_sh${i}.jsonl"
  if [ ! -f "$IN" ]; then echo "缺少输入 $IN"; exit 1; fi
  CUDA_VISIBLE_DEVICES=$i PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PY" train/score_rewards.py \
      --input "$IN" --out "train/rewards_sample_v1_sh${i}.jsonl" \
      --device cuda:0 --normalize options \
      --rewrite-cost 0.05 --leak-penalty 0.5 \
      --log-every 10 \
      > "train/rewards_sample_v1_sh${i}.log" 2>&1 &
  disown
  echo "shard $i -> GPU$i  PID $!"
  sleep 2
done
echo "全部 $NSHARD 片已启动"
