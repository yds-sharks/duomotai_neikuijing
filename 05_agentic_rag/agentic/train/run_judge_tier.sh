#!/usr/bin/env bash
# Stage ②:judge 打分 + 难度分档。8 片(每卡2进程),生成器 Qwen3-VL-8B。
set -u
cd "$(dirname "$0")/.."
PY=/mnt/data_1/yds/venvs/qwen3vl-rerank/bin/python
IN=outputs/stage2_calibration/agent_context_v11_train3200_sep.jsonl
SIZE=400
for i in 0 1 2 3 4 5 6 7; do
  st=$((i * SIZE))
  gpu=$((i % 4))
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" train/judge_and_tier.py \
      --input "$IN" --out "outputs/stage2_calibration/judge_tier_sh${i}.jsonl" \
      --device cuda:0 \
      --start "$st" --limit "$SIZE" --log-every 20 \
      > "outputs/stage2_calibration/judge_tier_sh${i}.log" 2>&1 &
  disown
  echo "shard $i -> GPU$gpu start=$st limit=$SIZE PID $!"
  sleep 2
done
echo "全部 8 片已启动"
