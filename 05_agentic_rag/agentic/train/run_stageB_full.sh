#!/bin/bash
# Stage B 全量:对每条改写 query 走真实检索召回文段
# 单卡双模型(image+text 同卡)+ 每片独占一份 Milvus 库拷贝,4 卡 4 片并行
# env: qwen3vl-rerank (transformers 4.57.x, 与建库口径一致)
# 依赖:Stage A 合并输出 train/policy_actions_v2.jsonl(图文对混合 obs)
set -u
cd /mnt/data_1/yds/多模态/rerank_image_and_text/agentic
PY=/mnt/data_1/yds/venvs/qwen3vl-rerank/bin/python
IN=train/policy_actions_v2.jsonl
MILVUS_DIR=/mnt/data_1/yds/多模态/data_house/milvus
NSHARD=4
SIZE=800
if [ ! -f "$IN" ]; then echo "缺少输入 $IN(需先合并 Stage A 12 片)"; exit 1; fi
for i in $(seq 0 $((NSHARD-1))); do
  st=$((i*SIZE))
  if [ "$i" -eq 0 ]; then DB="$MILVUS_DIR/multimodal_vector_indexes.db"; else DB="$MILVUS_DIR/multimodal_vector_indexes_shard${i}.db"; fi
  CUDA_VISIBLE_DEVICES=$i PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PY" train/gen_retrieval_for_rewrites.py \
      --input "$IN" --out "train/rollouts_recalled_v2_sh${i}.jsonl" \
      --image-device cuda:0 --text-device cuda:0 --db-path "$DB" \
      --image-select-k 6 --text-select-k 6 \
      --start "$st" --limit "$SIZE" --log-every 20 \
      > "train/rollouts_recalled_v2_sh${i}.log" 2>&1 &
  disown
  echo "shard $i -> GPU$i  start=$st limit=$SIZE  db=$(basename "$DB")  PID $!"
  sleep 3
done
echo "全部 $NSHARD 片已启动"
