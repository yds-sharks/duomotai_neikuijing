#!/usr/bin/env bash
# 全量重建 obs_candidates:4 片单卡并行,每片独立 Milvus DB 副本,分模态图文对。
set -u
cd "$(dirname "$0")/.."
PY=/mnt/data_1/yds/venvs/qwen3vl-rerank/bin/python
IN=outputs/stage2_calibration/agent_context_v11_train3200.jsonl
SIZE=800
DBS=(multimodal_vector_indexes.db multimodal_vector_indexes_shard1.db multimodal_vector_indexes_shard2.db multimodal_vector_indexes_shard3.db)
for i in 0 1 2 3; do
  st=$((i * SIZE))
  DB="/mnt/data_1/yds/多模态/data_house/milvus/${DBS[$i]}"
  CUDA_VISIBLE_DEVICES=$i nohup "$PY" train/rebuild_obs_candidates.py \
      --input "$IN" --out "outputs/stage2_calibration/obs_sep_sh${i}.jsonl" \
      --image-device cuda:0 --text-device cuda:0 --db-path "$DB" \
      --image-select-k 6 --text-select-k 6 \
      --start "$st" --limit "$SIZE" --log-every 50 \
      > "outputs/stage2_calibration/obs_sep_sh${i}.log" 2>&1 &
  disown
  echo "shard $i -> GPU$i start=$st limit=$SIZE db=${DBS[$i]} PID $!"
  sleep 2
done
echo "全部 4 片已启动"
