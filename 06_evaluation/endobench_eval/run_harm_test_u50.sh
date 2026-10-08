#!/bin/bash
# Harm test: run 3-round mechanism on 2741 originally-correct questions.
# Measures how many correct answers become wrong (harm rate H).
# Uses the SAME rescue_3round.py with deterministic ParseFail fallback.
#
# 2 parallel processes on 4x A6000:
#   Process 1: GPU 0+1 (default Milvus DB),     shard0 (~1370 questions)
#   Process 2: GPU 2+3 (shard1 Milvus DB copy), shard1 (~1371 questions)
set -e
cd /mnt/data_1/yds/多模态/endobench_eval

PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
CKPT=/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/ckpt_grpo_v4_u50
MILVUS_DIR=/mnt/data_1/yds/多模态/data_house/milvus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

# ---------- Process 1: GPU 0+1, shard0 ----------
CUDA_VISIBLE_DEVICES=0,1 $PY rescue_3round.py \
  --subset correct_full_shard0.jsonl \
  --ctrl-model "$CKPT" \
  --ctrl-device cuda:0 \
  --gen-device cuda:1 \
  --retr-text-device cuda:1 \
  --retr-image-device cuda:0 \
  --milvus-db-path "$MILVUS_DIR/multimodal_vector_indexes.db" \
  --max-rounds 3 \
  --use-organ-filter \
  --out rescue_harm_u50_shard0.jsonl \
  2>&1 | tee logs/rescue_harm_u50_shard0.log &
PID1=$!

# ---------- Process 2: GPU 2+3, shard1 ----------
CUDA_VISIBLE_DEVICES=2,3 $PY rescue_3round.py \
  --subset correct_full_shard1.jsonl \
  --ctrl-model "$CKPT" \
  --ctrl-device cuda:0 \
  --gen-device cuda:1 \
  --retr-text-device cuda:1 \
  --retr-image-device cuda:0 \
  --milvus-db-path "$MILVUS_DIR/multimodal_vector_indexes_shard1.db" \
  --max-rounds 3 \
  --use-organ-filter \
  --out rescue_harm_u50_shard1.jsonl \
  2>&1 | tee logs/rescue_harm_u50_shard1.log &
PID2=$!

echo "[harm] shard0 PID=$PID1, shard1 PID=$PID2"
echo "[harm] waiting for both processes to finish..."

wait $PID1
EXIT1=$?
wait $PID2
EXIT2=$?

echo "[harm] shard0 exit=$EXIT1, shard1 exit=$EXIT2"

if [ $EXIT1 -ne 0 ] || [ $EXIT2 -ne 0 ]; then
  echo "[harm] WARNING: one or both processes had non-zero exit code"
fi

# ---------- Merge results ----------
echo "[harm] merging shard results..."
$PY -c "
import json
from collections import Counter

results = []
for shard in ['rescue_harm_u50_shard0.jsonl', 'rescue_harm_u50_shard1.jsonl']:
    try:
        with open(shard) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    except FileNotFoundError:
        print(f'WARNING: {shard} not found')

with open('rescue_harm_u50.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

n = len(results)
n_harmed = sum(1 for r in results if r.get('orig_correct') and not r.get('correct'))
n_maintained = sum(1 for r in results if r.get('orig_correct') and r.get('correct'))
n_fallback = sum(1 for r in results if r.get('fallback_to_orig', False))

by_action = Counter(r.get('orig_action', 'unknown') for r in results)
harmed_by = Counter(r.get('orig_action', 'unknown') for r in results if r.get('orig_correct') and not r.get('correct'))

print(f'===== HARM TEST SUMMARY (u50, 3-round) =====')
print(f'Total originally-correct: {n}')
print(f'Maintained correct: {n_maintained} ({100*n_maintained/n:.1f}%)')
print(f'Harmed (correct->wrong): {n_harmed} ({100*n_harmed/n:.1f}%)')
print(f'Fallback to orig: {n_fallback}')
print()
for act in sorted(by_action):
    print(f'  {act}: {harmed_by[act]}/{by_action[act]} harmed ({100*harmed_by[act]/max(by_action[act],1):.1f}%)')
print()

# Compute true accuracy
total_q = 6832
orig_correct = 2741
rescued = 434  # from patched rescue_full_u50.jsonl
H = n_harmed
true_acc = (orig_correct - H + rescued) / total_q * 100
delta = (rescued - H) / total_q * 100
print(f'===== TRUE ACCURACY =====')
print(f'Original correct: {orig_correct}')
print(f'Rescued (wrong->correct): {rescued}')
print(f'Harmed (correct->wrong): {H}')
print(f'True correct: {orig_correct - H + rescued}')
print(f'True accuracy: {true_acc:.2f}% (vs single-round 40.12%)')
print(f'Delta: {delta:+.2f}pp')
"

echo "[harm] done. Results: rescue_harm_u50.jsonl"
