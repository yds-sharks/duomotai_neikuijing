#!/bin/bash
# Stage A 全量:策略高温采样动作(强制改写4 + 自然接受2)
# 高并发:每卡 3 进程 = 12 分片(每片 ~267 题),显存 3x10G<48G
# env: qwen35-train (transformers 5.13.1, Qwen3.5 policy)
set -u
cd /mnt/data_1/yds/多模态/rerank_image_and_text/agentic
PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
IN=outputs/stage2_calibration/agent_context_v11_train3200_paired.jsonl
MODEL=train/ckpt_qwen35_ctrl_full_v1
NSHARD=12
SIZE=267
for i in $(seq 0 $((NSHARD-1))); do
  st=$((i*SIZE))
  gpu=$((i%4))
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PY" train/gen_policy_actions.py \
      --input "$IN" --out "train/policy_actions_v2_sh${i}.jsonl" \
      --model "$MODEL" --policy-device cuda:0 \
      --n-natural 2 --n-force-rewrite 4 --temperature 1.4 \
      --start "$st" --limit "$SIZE" --log-every 20 \
      > "train/policy_actions_v2_sh${i}.log" 2>&1 &
  disown
  echo "shard $i -> GPU$gpu  start=$st limit=$SIZE  PID $!"
  sleep 2
done
echo "全部 $NSHARD 片已启动"
