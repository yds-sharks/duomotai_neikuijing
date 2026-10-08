#!/bin/bash
# Stage C(v2.1):REWRITE 分支的 round-1 keep/drop 贪心挑选
# 输入:Stage B 4 片 train/rollouts_sample_v1_sh{i}.jsonl(逐片处理,不再合并)
# ACCEPT 分支(selected=round0 keep 应用 obs)/anchor 成员在此一并落实 selected_passages
# env: qwen35-train;4 卡 4 片
set -u
cd /mnt/data_1/yds/多模态/rerank_image_and_text/agentic
PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
MODEL=train/ckpt_qwen35_ctrl_full_v1
NSHARD=4
for i in $(seq 0 $((NSHARD-1))); do
  IN="train/rollouts_sample_v1_sh${i}.jsonl"
  if [ ! -f "$IN" ]; then echo "缺少输入 $IN"; exit 1; fi
  CUDA_VISIBLE_DEVICES=$i PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PY" train/gen_policy_select_round2.py \
      --input "$IN" --out "train/rollouts_selected_sample_v1_sh${i}.jsonl" \
      --model "$MODEL" --policy-device cuda:0 \
      --log-every 20 \
      > "train/rollouts_selected_sample_v1_sh${i}.log" 2>&1 &
  disown
  echo "shard $i -> GPU$i  PID $!"
  sleep 2
done
echo "全部 $NSHARD 片已启动"
