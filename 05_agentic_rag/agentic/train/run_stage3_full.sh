#!/bin/bash
# Stage 3 全量(v2.1):tier-aware 分层采样,输出 group 格式
# 8 分片 x 400 题,每卡 2 进程(4B bf16 ~10G/实例,48G 卡宽松)
# trivial 由 --include-trivial-frac 0.15 按 crc32(qid) 确定性抽样(分片安全)
# env: qwen35-train
set -u
cd /mnt/data_1/yds/多模态/rerank_image_and_text/agentic
PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
OBS=outputs/stage2_calibration/agent_context_v11_train3200_sep.jsonl
TIER=outputs/stage2_calibration/judge_tier_sep.jsonl
MODEL=train/ckpt_qwen35_ctrl_full_v1
NSHARD=8
SIZE=400
for i in $(seq 0 $((NSHARD-1))); do
  st=$((i*SIZE))
  gpu=$((i/2))          # 片 0,1->GPU0; 2,3->GPU1; ...
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PY" train/gen_policy_sample.py \
      --obs "$OBS" --tier "$TIER" \
      --out "train/policy_sample_v1_sh${i}.jsonl" \
      --model "$MODEL" --policy-device cuda:0 \
      --include-trivial-frac 0.15 \
      --start "$st" --limit "$SIZE" --log-every 20 \
      > "train/policy_sample_v1_sh${i}.log" 2>&1 &
  disown
  echo "shard $i -> GPU$gpu  start=$st limit=$SIZE  PID $!"
  sleep 2
done
echo "全部 $NSHARD 片已启动"
