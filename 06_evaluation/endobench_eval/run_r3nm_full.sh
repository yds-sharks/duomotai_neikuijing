#!/bin/bash
# R3-NM: 3-Round Iterative Agent without Memory — 全量 6,832 题
# 规格: CLEAN_MEMORY_EXPERIMENT_SPEC_2026-07-28.md
# 回退: 预算耗尽/ParseFail/空keep → top-5 (绝不 closed-book)
# 分片: shard0 (GPU 0/1, 0-3416) + shard1 (GPU 2/3, 3416-6832), 错开启动避免 Milvus 锁
cd /mnt/data_1/yds/多模态/endobench_eval
HERE=$PWD
PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
CKPT=/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/ckpt_grpo_v4_u50
CACHE=$HERE/results_v2/cand_cache_v2.jsonl
TRANS=$HERE/endobench_translated_queries.jsonl
COMMON="--mode agentic --ctrl-model $CKPT --ctrl-device cuda:0 --gen-device cuda:1 \
  --retr-image-device cuda:0 --retr-text-device cuda:1 --weight-device cuda:0 \
  --cand-cache $CACHE --translate-cache $TRANS \
  --use-organ-filter --use-weight-module --max-rounds 3 --latency \
  --no-mplus --no-mminus"

mkdir -p logs results_full/R3_NM_shard0 results_full/R3_NM_shard1

echo "[R3-NM] shard0 start $(date '+%m-%d %H:%M')"
CUDA_VISIBLE_DEVICES=0,1 $PY eval_endobench.py $COMMON \
  --out-dir results_full/R3_NM_shard0 --offset 0 --limit 3416 \
  2>&1 | tee logs/r3nm_shard0.log &

# 等 shard0 检索上线再启 shard1（避免 Milvus Lite 并发初始化锁冲突）
while ! grep -q "\[retr\] online" logs/r3nm_shard0.log 2>/dev/null; do sleep 15; done
sleep 20

echo "[R3-NM] shard1 start $(date '+%m-%d %H:%M')"
CUDA_VISIBLE_DEVICES=2,3 $PY eval_endobench.py $COMMON \
  --out-dir results_full/R3_NM_shard1 --offset 3416 --limit 3416 \
  2>&1 | tee logs/r3nm_shard1.log &

wait
echo "[R3-NM] ALL DONE $(date '+%m-%d %H:%M')"
