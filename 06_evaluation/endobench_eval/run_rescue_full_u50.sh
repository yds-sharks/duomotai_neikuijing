#!/bin/bash
# Full 4091-question rescue evaluation (u50, 3-round)
# Runs 2 parallel processes on 4x A6000 GPUs.
#   Process 1: GPU 0+1 (default Milvus DB),     shard0 (~2045 questions)
#   Process 2: GPU 2+3 (shard1 Milvus DB copy), shard1 (~2046 questions)
set -e
cd /mnt/data_1/yds/多模态/endobench_eval

PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
CKPT=/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/ckpt_grpo_v4_u50
MILVUS_DIR=/mnt/data_1/yds/多模态/data_house/milvus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

# ---------- Process 1: GPU 0+1, shard0 ----------
CUDA_VISIBLE_DEVICES=0,1 $PY rescue_3round.py \
  --subset wrong_full_shard0.jsonl \
  --ctrl-model "$CKPT" \
  --ctrl-device cuda:0 \
  --gen-device cuda:1 \
  --retr-text-device cuda:1 \
  --retr-image-device cuda:0 \
  --milvus-db-path "$MILVUS_DIR/multimodal_vector_indexes.db" \
  --max-rounds 3 \
  --use-organ-filter \
  --out rescue_full_u50_shard0.jsonl \
  2>&1 | tee logs/rescue_full_u50_shard0.log &
PID1=$!

# ---------- Process 2: GPU 2+3, shard1 ----------
CUDA_VISIBLE_DEVICES=2,3 $PY rescue_3round.py \
  --subset wrong_full_shard1.jsonl \
  --ctrl-model "$CKPT" \
  --ctrl-device cuda:0 \
  --gen-device cuda:1 \
  --retr-text-device cuda:1 \
  --retr-image-device cuda:0 \
  --milvus-db-path "$MILVUS_DIR/multimodal_vector_indexes_shard1.db" \
  --max-rounds 3 \
  --use-organ-filter \
  --out rescue_full_u50_shard1.jsonl \
  2>&1 | tee logs/rescue_full_u50_shard1.log &
PID2=$!

echo "[run] shard0 PID=$PID1, shard1 PID=$PID2"
echo "[run] waiting for both processes to finish..."

wait $PID1
EXIT1=$?
wait $PID2
EXIT2=$?

echo "[run] shard0 exit=$EXIT1, shard1 exit=$EXIT2"

if [ $EXIT1 -ne 0 ] || [ $EXIT2 -ne 0 ]; then
  echo "[run] WARNING: one or both processes had non-zero exit code"
fi

# ---------- Merge results ----------
echo "[run] merging shard results..."
$PY -c "
import json

results = []
for shard in ['rescue_full_u50_shard0.jsonl', 'rescue_full_u50_shard1.jsonl']:
    try:
        with open(shard) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    except FileNotFoundError:
        print(f'WARNING: {shard} not found')

with open('rescue_full_u50.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

n = len(results)
n_rescued = sum(1 for r in results if r.get('rescued'))
n_error = sum(1 for r in results if r.get('error'))

from collections import Counter
by_action = Counter(r.get('orig_action', 'unknown') for r in results)
rescued_by_action = Counter(r.get('orig_action', 'unknown') for r in results if r.get('rescued'))

print(f'===== FULL RESCUE SUMMARY (u50) =====')
print(f'Total: {n}')
print(f'Rescued (wrong->correct): {n_rescued} ({100*n_rescued/max(n,1):.1f}%)')
print(f'Errors: {n_error}')
for act in sorted(by_action):
    print(f'  {act}: {rescued_by_action[act]}/{by_action[act]} rescued ({100*rescued_by_action[act]/max(by_action[act],1):.1f}%)')
"

echo "[run] done. Results: rescue_full_u50.jsonl"
