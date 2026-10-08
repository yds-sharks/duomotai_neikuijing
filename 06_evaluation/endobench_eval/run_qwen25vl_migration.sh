#!/bin/bash
# P1.6: Qwen2.5-VL-7B generator migration test on stratified 513 questions.
# Reuses saved retrieval traces (cand_cache_v2.jsonl) — only changes generator.
# Same controller (u50), same retrieval (organ+weight), different generator.
#
# Batch 1: baseline (GPU 0+1)    | vanilla_rag (GPU 2+3)
# Batch 2: single-round (GPU 0+1) | 3-round (GPU 2+3)
#
# Extension threshold: 3-Round vs Closed-book ≥ +2pp; 3-Round vs Vanilla ≥ +1pp
set +e
cd /mnt/data_1/yds/多模态/endobench_eval

PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
CKPT=/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/ckpt_grpo_v4_u50
MILVUS_DIR=/mnt/data_1/yds/多模态/data_house/milvus
GEN25=/mnt/data_10/mwx/huggingface_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

QID_FILE=stratified_512.jsonl
DEV_A="--gen-device cuda:1 --retr-image-device cuda:0 --retr-text-device cuda:1"
DEV_B="--gen-device cuda:1 --retr-image-device cuda:0 --retr-text-device cuda:1"
DB_A="$MILVUS_DIR/multimodal_vector_indexes.db"
DB_B="$MILVUS_DIR/multimodal_vector_indexes_shard1.db"
CACHE="results_v2/cand_cache_v2.jsonl"

COMMON="--qid-file $QID_FILE --gen-model $GEN25 --cand-cache $CACHE --use-organ-filter --use-weight-module"

mkdir -p results_qwen25vl logs

run_pair() {
    local name_a="$1" mode_a="$2" extra_a="$3" out_a="$4"
    local name_b="$5" mode_b="$6" extra_b="$7" out_b="$8"

    echo "============================================"
    echo "[qwen25vl] $name_a  ||  $name_b"
    echo "============================================"

    CUDA_VISIBLE_DEVICES=0,1 $PY eval_endobench.py --mode $mode_a $COMMON $DEV_A \
        --milvus-db-path "$DB_A" --out-dir "results_qwen25vl/$out_a" $extra_a \
        2>&1 | tee "logs/qwen25vl_${out_a}.log" &
    PID_A=$!

    CUDA_VISIBLE_DEVICES=2,3 $PY eval_endobench.py --mode $mode_b $COMMON $DEV_B \
        --milvus-db-path "$DB_B" --out-dir "results_qwen25vl/$out_b" $extra_b \
        2>&1 | tee "logs/qwen25vl_${out_b}.log" &
    PID_B=$!

    echo "[qwen25vl] PID_A=$PID_A  PID_B=$PID_B"
    wait $PID_A; EXIT_A=$?
    wait $PID_B; EXIT_B=$?
    echo "[qwen25vl] $name_a exit=$EXIT_A  $name_b exit=$EXIT_B"
}

# ---------- Batch 1: baseline | vanilla_rag ----------
run_pair \
    "baseline"     "baseline"    ""                           "baseline" \
    "vanilla_rag"  "vanilla_rag" ""                           "vanilla_rag"

# ---------- Batch 2: single-round | 3-round ----------
run_pair \
    "single-round" "agentic"     "--ctrl-model $CKPT --ctrl-device cuda:0 --max-rounds 1" "single_round" \
    "3-round"      "agentic"     "--ctrl-model $CKPT --ctrl-device cuda:0 --max-rounds 3" "three_round"

# ---------- Summary ----------
echo ""
echo "============================================"
echo "[qwen25vl] All batches complete. Summarizing..."
echo "============================================"
$PY -c "
import json, os
from collections import defaultdict

configs = [
    ('Closed-book',  'results_qwen25vl/baseline/agentic_samples.jsonl' if os.path.exists('results_qwen25vl/baseline/agentic_samples.jsonl') else 'results_qwen25vl/baseline/baseline_samples.jsonl'),
    ('Vanilla RAG',  'results_qwen25vl/vanilla_rag/vanilla_rag_samples.jsonl'),
    ('Single-round', 'results_qwen25vl/single_round/agentic_samples.jsonl'),
    ('3-Round',      'results_qwen25vl/three_round/agentic_samples.jsonl'),
]

results = {}
print(f'{\"Config\":<15} {\"N\":>5} {\"Correct\":>8} {\"Acc\":>7} {\"MeanLat\":>8}')
print('-' * 50)
for name, path in configs:
    if not os.path.exists(path):
        print(f'{name:<15} {\"MISSING\":>5}')
        continue
    records = [json.loads(l) for l in open(path) if l.strip()]
    n = len(records)
    correct = sum(1 for r in records if r.get('correct'))
    ml = sum(r.get('latency_s', 0) for r in records) / n if n else 0
    acc = correct / n if n else 0
    results[name] = {'acc': acc, 'n': n, 'correct': correct}
    print(f'{name:<15} {n:>5} {correct:>8} {100*acc:>6.2f}% {ml:>7.1f}s')

# Check extension thresholds
print()
if 'Closed-book' in results and '3-Round' in results:
    diff_cb = results['3-Round']['acc'] - results['Closed-book']['acc']
    print(f'3-Round vs Closed-book: {100*diff_cb:+.2f}pp (threshold: +2.00pp) → {\"PASS\" if diff_cb >= 0.02 else \"FAIL\"} (extend to full)')

if 'Vanilla RAG' in results and '3-Round' in results:
    diff_vr = results['3-Round']['acc'] - results['Vanilla RAG']['acc']
    print(f'3-Round vs Vanilla RAG: {100*diff_vr:+.2f}pp (threshold: +1.00pp) → {\"PASS\" if diff_vr >= 0.01 else \"FAIL\"} (extend to full)')

# Per-scene
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

echo "[qwen25vl] done."
