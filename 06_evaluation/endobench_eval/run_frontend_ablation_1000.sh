#!/bin/bash
# P0.2: Front-end ablation on stratified 1002 questions.
# All configs use 3-round full memory (M+ + M-, Top-3).
# Varies: weight module (modality mixer) + organ filter (anatomical router)
#
# Batch 1: Full front-end (GPU 0+1) | w/o organ (GPU 2+3)
# Batch 2: w/o weight (GPU 0+1)    | w/o both (GPU 2+3)
# Don't use set -e: individual config failures shouldn't stop the entire batch
set +e
cd /mnt/data_1/yds/多模态/endobench_eval

PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
CKPT=/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/ckpt_grpo_v4_u50
MILVUS_DIR=/mnt/data_1/yds/多模态/data_house/milvus
GEN_MODEL=/mnt/data_10/mwx/huggingface_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

QID_FILE=stratified_1000.jsonl
DEV_A="--ctrl-device cuda:0 --gen-device cuda:1 --retr-text-device cuda:1 --retr-image-device cuda:0"
DEV_B="--ctrl-device cuda:0 --gen-device cuda:1 --retr-text-device cuda:1 --retr-image-device cuda:0"
DB_A="$MILVUS_DIR/multimodal_vector_indexes.db"
DB_B="$MILVUS_DIR/multimodal_vector_indexes_shard1.db"

COMMON="--mode agentic --max-rounds 3 --qid-file $QID_FILE --gen-model $GEN_MODEL --ctrl-model $CKPT"

mkdir -p results_ablation logs

# Each config needs its own cand-cache because organ filter and weight module
# produce different retrieval/selection results. The existing cand_cache_v2.jsonl
# was built with organ+weight (matches Config 1 only).

run_pair() {
    local name_a="$1" flags_a="$2" out_a="$3" cache_a="$4"
    local name_b="$5" flags_b="$6" out_b="$7" cache_b="$8"

    echo "============================================"
    echo "[frontend] $name_a  ||  $name_b"
    echo "============================================"

    CUDA_VISIBLE_DEVICES=0,1 $PY eval_endobench.py $COMMON $DEV_A \
        --milvus-db-path "$DB_A" --cand-cache "$cache_a" \
        --out-dir "results_ablation/$out_a" $flags_a \
        2>&1 | tee "logs/frontend_${out_a}.log" &
    PID_A=$!

    CUDA_VISIBLE_DEVICES=2,3 $PY eval_endobench.py $COMMON $DEV_B \
        --milvus-db-path "$DB_B" --cand-cache "$cache_b" \
        --out-dir "results_ablation/$out_b" $flags_b \
        2>&1 | tee "logs/frontend_${out_b}.log" &
    PID_B=$!

    echo "[frontend] PID_A=$PID_A  PID_B=$PID_B"
    wait $PID_A; EXIT_A=$?
    wait $PID_B; EXIT_B=$?
    echo "[frontend] $name_a exit=$EXIT_A  $name_b exit=$EXIT_B"
}

# ---------- Batch 1: Full front-end | w/o organ filter ----------
# Full: reuse existing cand_cache_v2.jsonl (built with organ+weight)
# w/o organ: new cache (no organ filter + weight module)
run_pair \
    "Full_front-end"    "--use-organ-filter --use-weight-module" "frontend_full"     "results_v2/cand_cache_v2.jsonl" \
    "w/o_organ_router"  "--use-weight-module"                    "frontend_no_organ" "cand_cache_noorgan_weight_1000.jsonl"

# ---------- Batch 2: w/o weight module | w/o both ----------
# w/o weight: new cache (organ filter + no weight module)
# w/o both: new cache (no organ filter + no weight module)
run_pair \
    "w/o_weight_module" "--use-organ-filter"                     "frontend_no_weight" "cand_cache_organ_noweight_1000.jsonl" \
    "w/o_both"          ""                                        "frontend_no_both"   "cand_cache_noorgan_noweight_1000.jsonl"

# ---------- Summary ----------
echo ""
echo "============================================"
echo "[frontend] All batches complete. Summarizing..."
echo "============================================"
$PY -c "
import json, os
from collections import defaultdict

configs = [
    ('Full front-end',    'results_ablation/frontend_full/agentic_samples.jsonl'),
    ('w/o weight module', 'results_ablation/frontend_no_weight/agentic_samples.jsonl'),
    ('w/o organ router',  'results_ablation/frontend_no_organ/agentic_samples.jsonl'),
    ('w/o both',          'results_ablation/frontend_no_both/agentic_samples.jsonl'),
]

print(f'{\"Config\":<22} {\"N\":>5} {\"Acc\":>7} {\"Acc+Cmp\":>8} {\"ParseFail\":>10} {\"MeanRounds\":>11} {\"MeanLat\":>8}')
print('-' * 75)
for name, path in configs:
    if not os.path.exists(path):
        print(f'{name:<22} {\"MISSING\":>5}')
        continue
    records = [json.loads(l) for l in open(path) if l.strip()]
    n = len(records)
    correct = sum(1 for r in records if r.get('correct'))
    pf = sum(1 for r in records if not r.get('parse_ok', True))
    mr = sum(r.get('n_rounds', 1) for r in records) / n if n else 0
    ml = sum(r.get('latency_s', 0) for r in records) / n if n else 0
    # Compare with correct subset
    cmp_records = [r for r in records if r.get('correct') is not None]
    print(f'{name:<22} {n:>5} {100*correct/max(n,1):>6.2f}% {pf:>10} {mr:>11.2f} {ml:>7.1f}s')

# Per-scene breakdown
print()
for name, path in configs:
    if not os.path.exists(path):
        continue
    records = [json.loads(l) for l in open(path) if l.strip()]
    scenes = defaultdict(list)
    for r in records:
        scenes[r.get('scene', '?')].append(r)
    print(f'--- {name} ---')
    for scene in sorted(scenes):
        rs = scenes[scene]
        acc = sum(1 for r in rs if r.get('correct')) / len(rs) if rs else 0
        print(f'  {scene:<25} n={len(rs):>4} acc={100*acc:.2f}%')
"

echo "[frontend] done."
