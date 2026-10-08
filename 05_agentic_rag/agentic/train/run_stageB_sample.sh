#!/bin/bash
# Stage B(v2.1):对 group 内 REWRITE 成员的改写 query 走真实检索召回图文对
# 输入:Stage ③ 合并产物 train/policy_sample_v1.jsonl(group 格式)
# 单卡双模型(image+text 同卡)+ 每片独占一份 Milvus 库拷贝,4 卡 4 片并行
# env: qwen3vl-rerank (与建库口径一致)
set -u
cd /mnt/data_1/yds/多模态/rerank_image_and_text/agentic
PY=/mnt/data_1/yds/venvs/qwen3vl-rerank/bin/python
IN=train/policy_sample_v1.jsonl
MILVUS_DIR=/mnt/data_1/yds/多模态/data_house/milvus
NSHARD=4
SIZE=${SIZE:-500}
if [ ! -f "$IN" ]; then echo "缺少输入 $IN(需先合并 Stage ③ 8 片)"; exit 1; fi
for i in $(seq 0 $((NSHARD-1))); do
  st=$((i*SIZE))
  if [ "$i" -eq 0 ]; then DB="$MILVUS_DIR/multimodal_vector_indexes.db"; else DB="$MILVUS_DIR/multimodal_vector_indexes_shard${i}.db"; fi
  CUDA_VISIBLE_DEVICES=$i PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PY" train/gen_retrieval_for_rewrites.py \
      --input "$IN" --out "train/rollouts_sample_v1_sh${i}.jsonl" \
      --image-device cuda:0 --text-device cuda:0 --db-path "$DB" \
      --image-select-k 6 --text-select-k 6 \
      --start "$st" --limit "$SIZE" --log-every 20 \
      > "train/rollouts_sample_v1_sh${i}.log" 2>&1 &
  disown
  echo "shard $i -> GPU$i  start=$st limit=$SIZE  db=$(basename "$DB")  PID $!"
  sleep 3
done
echo "全部 $NSHARD 片已启动"
