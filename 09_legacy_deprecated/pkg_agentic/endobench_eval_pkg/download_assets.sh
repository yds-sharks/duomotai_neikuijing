#!/bin/bash
# 从 GitHub Release 下载运行资产并重组（私有仓库需先 gh auth login）
set -e
cd "$(dirname "$0")"
REL=eval-assets-v1
REPO=yds-sharks/agentic
mkdir -p assets/dl models
cd assets/dl
echo "[dl] gh release download $(date +%H:%M:%S)"
gh release download $REL --repo $REPO --clobber \
  --pattern "u50.tar.part_*" \
  --pattern "milvus.db.part_*" \
  --pattern "cand_cache_v2.jsonl" \
  --pattern "endobench_translated_queries.jsonl" \
  --pattern "multimodal_samples.db" \
  --pattern "weight_ckpts.tar"
cd ..
echo "[dl] reassemble u50 checkpoint $(date +%H:%M:%S)"
cat dl/u50.tar.part_* | tar xf - -C models/
echo "[dl] reassemble milvus db $(date +%H:%M:%S)"
cat dl/milvus.db.part_* > assets/multimodal_vector_indexes.db
mv dl/cand_cache_v2.jsonl dl/endobench_translated_queries.jsonl dl/multimodal_samples.db assets/
mkdir -p weight_module/checkpoints
tar xf dl/weight_ckpts.tar -C weight_module/checkpoints/
rm -rf dl
echo "[dl] ASSETS READY $(date +%H:%M:%S)"
