#!/bin/bash
# P0.1: Fair Full re-run on 200 wrong questions with updated ParseFail handling.
# Same QIDs/order as wrong200_subset.jsonl (100 ACCEPT + 100 REWRITE).
# Uses rescue_3round.py with M+ + M-, 3 rounds, Top-3, organ filter.
# Output: ablation_200/fair_full.jsonl
set -e
cd /mnt/data_1/yds/多模态/endobench_eval

PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
CKPT=/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/ckpt_grpo_v4_u50
MILVUS_DIR=/mnt/data_1/yds/多模态/data_house/milvus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs ablation_200

echo "============================================"
echo "[fair_full] Starting fair Full re-run (200 questions)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0,1 $PY rescue_3round.py \
    --subset wrong200_subset.jsonl \
    --ctrl-model $CKPT \
    --max-rounds 3 \
    --use-organ-filter \
    --ctrl-device cuda:0 --gen-device cuda:1 \
    --retr-text-device cuda:1 --retr-image-device cuda:0 \
    --milvus-db-path "$MILVUS_DIR/multimodal_vector_indexes.db" \
    --out ablation_200/fair_full.jsonl \
    2>&1 | tee logs/ablation_fair_full.log

echo ""
echo "============================================"
echo "[fair_full] Done. Comparing with old baseline..."
echo "============================================"
$PY -c "
import json, os
from collections import Counter

variants = [
    ('fair_full (NEW)',  'ablation_200/fair_full.jsonl'),
    ('old_baseline',     'rescue200_u50.jsonl'),
    ('no-memory',        'ablation_200/nomem.jsonl'),
    ('M+_only',          'ablation_200/mplus_only.jsonl'),
    ('M-_only',          'ablation_200/mminus_only.jsonl'),
    ('2-round',          'ablation_200/round2.jsonl'),
    ('Top-1-M+',         'ablation_200/top1.jsonl'),
    ('Top-5-M+',         'ablation_200/top5.jsonl'),
]

print(f'{\"Variant\":<22} {\"N\":>4} {\"Rescued\":>8} {\"Rate\":>7} {\"ACCEPT\":>10} {\"REWRITE\":>10} {\"ParseFail\":>10} {\"Fallback\":>9}')
print('-' * 92)
for name, path in variants:
    if not os.path.exists(path):
        print(f'{name:<22} {\"MISSING\":>4}')
        continue
    results = [json.loads(l) for l in open(path) if l.strip()]
    n = len(results)
    rescued = sum(1 for r in results if r.get('rescued'))
    pf = sum(1 for r in results if any(not rd.get('parse_ok', True) for rd in r.get('rounds', [])))
    fb = sum(1 for r in results if r.get('fallback_to_orig', False))
    by_act = Counter(r.get('orig_action', r.get('action', '?')) for r in results)
    res_by = Counter(r.get('orig_action', r.get('action', '?')) for r in results if r.get('rescued'))
    acc = f'{res_by[\"ACCEPT\"]}/{by_act[\"ACCEPT\"]}' if 'ACCEPT' in by_act else 'N/A'
    rw = f'{res_by[\"REWRITE\"]}/{by_act[\"REWRITE\"]}' if 'REWRITE' in by_act else 'N/A'
    print(f'{name:<22} {n:>4} {rescued:>8} {100*rescued/max(n,1):>6.1f}% {acc:>10} {rw:>10} {pf:>10} {fb:>9}')
"
