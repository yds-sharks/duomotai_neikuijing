#!/bin/bash
# 200-question ablation: memory mechanism decomposition.
# Runs on wrong200_subset.jsonl (100 ACCEPT + 100 REWRITE).
#
# Baseline (already have): M+ + M-, 3-round → rescue200_u50.jsonl (18.5%)
#
# Variants (3 batches, 2-way parallel):
#   Batch 1: no-memory  |  M+ only
#   Batch 2: M- only    |  max_rounds=2
#   Batch 3: Top-1 M+   |  Top-5 M+
#
# NOTE: 200-question subset overestimates rescue rate (18.5% vs 10.6% full).
# Results are for mechanism diagnosis ONLY, not paper-final numbers.
# Key variants will be scaled to full 4091 after screening.
set -e
cd /mnt/data_1/yds/多模态/endobench_eval

PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
CKPT=/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/ckpt_grpo_v4_u50
MILVUS_DIR=/mnt/data_1/yds/多模态/data_house/milvus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs ablation_200

SUBSET=wrong200_subset.jsonl
COMMON_ARGS="--subset $SUBSET --ctrl-model $CKPT --max-rounds 3 --use-organ-filter"
GPU_A="CUDA_VISIBLE_DEVICES=0,1"
GPU_B="CUDA_VISIBLE_DEVICES=2,3"
DEV_A="--ctrl-device cuda:0 --gen-device cuda:1 --retr-text-device cuda:1 --retr-image-device cuda:0"
DB_A="$MILVUS_DIR/multimodal_vector_indexes.db"
DB_B="$MILVUS_DIR/multimodal_vector_indexes_shard1.db"

run_pair() {
    local name_a="$1" flags_a="$2" out_a="$3"
    local name_b="$4" flags_b="$5" out_b="$6"

    echo "============================================"
    echo "[ablation] $name_a  ||  $name_b"
    echo "============================================"

    env $GPU_A $PY rescue_3round.py $COMMON_ARGS $DEV_A \
        --milvus-db-path "$DB_A" $flags_a \
        --out "ablation_200/$out_a" \
        2>&1 | tee "logs/ablation_${out_a%.jsonl}.log" &
    PID_A=$!

    env $GPU_B $PY rescue_3round.py $COMMON_ARGS $DEV_A \
        --milvus-db-path "$DB_B" $flags_b \
        --out "ablation_200/$out_b" \
        2>&1 | tee "logs/ablation_${out_b%.jsonl}.log" &
    PID_B=$!

    echo "[ablation] PID_A=$PID_A  PID_B=$PID_B"
    wait $PID_A
    EXIT_A=$?
    wait $PID_B
    EXIT_B=$?
    echo "[ablation] $name_a exit=$EXIT_A  $name_b exit=$EXIT_B"

    if [ $EXIT_A -ne 0 ] || [ $EXIT_B -ne 0 ]; then
        echo "[ablation] WARNING: non-zero exit detected"
    fi
}

# ---------- Batch 1: no-memory | M+ only ----------
run_pair \
    "no-memory" "--no-mplus --no-mminus" "nomem.jsonl" \
    "M+_only"   "--no-mminus"            "mplus_only.jsonl"

# ---------- Batch 2: M- only | max_rounds=2 ----------
run_pair \
    "M-_only"     "--no-mplus"   "mminus_only.jsonl" \
    "2-round"     "--max-rounds 2" "round2.jsonl"

# ---------- Batch 3: Top-1 | Top-5 ----------
run_pair \
    "Top-1-M+" "--mplus-topk 1" "top1.jsonl" \
    "Top-5-M+" "--mplus-topk 5" "top5.jsonl"

# ---------- Summary ----------
echo ""
echo "============================================"
echo "[ablation] All batches complete. Summarizing..."
echo "============================================"
$PY -c "
import json, os
from collections import Counter

variants = [
    ('baseline_M+M-_3round', 'rescue200_u50.jsonl'),
    ('no-memory',            'ablation_200/nomem.jsonl'),
    ('M+_only',              'ablation_200/mplus_only.jsonl'),
    ('M-_only',              'ablation_200/mminus_only.jsonl'),
    ('2-round',              'ablation_200/round2.jsonl'),
    ('Top-1-M+',             'ablation_200/top1.jsonl'),
    ('Top-5-M+',             'ablation_200/top5.jsonl'),
]

print(f'{\"Variant\":<25} {\"N\":>4} {\"Rescued\":>8} {\"Rate\":>7} {\"ACCEPT\":>10} {\"REWRITE\":>10} {\"ParseFail\":>10} {\"Fallback\":>9}')
print('-' * 95)
for name, path in variants:
    if not os.path.exists(path):
        print(f'{name:<25} {\"MISSING\":>4}')
        continue
    results = [json.loads(l) for l in open(path) if l.strip()]
    n = len(results)
    rescued = sum(1 for r in results if r.get('rescued'))
    pf = sum(1 for r in results if any(not rd.get('parse_ok', True) for rd in r.get('rounds', [])))
    fb = sum(1 for r in results if r.get('fallback_to_orig', False))
    by_act = Counter(r.get('orig_action', '?') for r in results)
    res_by = Counter(r.get('orig_action', '?') for r in results if r.get('rescued'))
    acc = f'{res_by[\"ACCEPT\"]}/{by_act[\"ACCEPT\"]}' if 'ACCEPT' in by_act else 'N/A'
    rw = f'{res_by[\"REWRITE\"]}/{by_act[\"REWRITE\"]}' if 'REWRITE' in by_act else 'N/A'
    print(f'{name:<25} {n:>4} {rescued:>8} {100*rescued/max(n,1):>6.1f}% {acc:>10} {rw:>10} {pf:>10} {fb:>9}')
"

echo "[ablation] done."
