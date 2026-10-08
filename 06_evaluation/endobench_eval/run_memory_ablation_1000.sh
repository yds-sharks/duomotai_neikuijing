#!/bin/bash
# P1.4 + P1.5: Memory ablation + round count on stratified 1002 questions.
# All configs use full front-end (--use-organ-filter --use-weight-module).
# Varies: memory mechanism (M+/M-) and round count.
#
# Reference (already run in Phase 2): Full memory 3-round = results_ablation/frontend_full
#
# Batch 1: no-memory (GPU 0+1)  | M+ only (GPU 2+3)
# Batch 2: M- only  (GPU 0+1)  | 2-round (GPU 2+3)
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

# All use full front-end + 3-round (except 2-round variant)
COMMON="--mode agentic --max-rounds 3 --qid-file $QID_FILE --gen-model $GEN_MODEL --ctrl-model $CKPT --use-organ-filter --use-weight-module"

mkdir -p results_ablation logs

run_pair() {
    local name_a="$1" flags_a="$2" out_a="$3"
    local name_b="$4" flags_b="$5" out_b="$6"

    echo "============================================"
    echo "[memory] $name_a  ||  $name_b"
    echo "============================================"

    CUDA_VISIBLE_DEVICES=0,1 $PY eval_endobench.py $COMMON $DEV_A \
        --milvus-db-path "$DB_A" --cand-cache "results_v2/cand_cache_v2.jsonl" \
        --out-dir "results_ablation/$out_a" $flags_a \
        2>&1 | tee "logs/memory_${out_a}.log" &
    PID_A=$!

    CUDA_VISIBLE_DEVICES=2,3 $PY eval_endobench.py $COMMON $DEV_B \
        --milvus-db-path "$DB_B" --cand-cache "results_v2/cand_cache_v2.jsonl" \
        --out-dir "results_ablation/$out_b" $flags_b \
        2>&1 | tee "logs/memory_${out_b}.log" &
    PID_B=$!

    echo "[memory] PID_A=$PID_A  PID_B=$PID_B"
    wait $PID_A; EXIT_A=$?
    wait $PID_B; EXIT_B=$?
    echo "[memory] $name_a exit=$EXIT_A  $name_b exit=$EXIT_B"
}

# ---------- Batch 1: no-memory | M+ only ----------
run_pair \
    "no-memory"  "--no-mplus --no-mminus"  "mem_nomem" \
    "M+_only"     "--no-mminus"             "mem_mplus_only"

# ---------- Batch 2: M- only | 2-round ----------
run_pair \
    "M-_only"     "--no-mplus"              "mem_mminus_only" \
    "2-round"     "--max-rounds 2"          "mem_round2"

# ---------- Summary ----------
echo ""
echo "============================================"
echo "[memory] All batches complete. Summarizing..."
echo "============================================"
$PY -c "
import json, os
from collections import defaultdict

configs = [
    ('Full 3-round (ref)', 'results_ablation/frontend_full/agentic_samples.jsonl'),
    ('no-memory',          'results_ablation/mem_nomem/agentic_samples.jsonl'),
    ('M+ only',            'results_ablation/mem_mplus_only/agentic_samples.jsonl'),
    ('M- only',            'results_ablation/mem_mminus_only/agentic_samples.jsonl'),
    ('2-round',            'results_ablation/mem_round2/agentic_samples.jsonl'),
]

print(f'{\"Config\":<22} {\"N\":>5} {\"Correct\":>8} {\"Acc\":>7} {\"Harm\":>5} {\"Rescue\":>7} {\"ParseFail\":>10} {\"MeanRounds\":>11} {\"MeanLat\":>8}')
print('-' * 90)

# Load single-round results for comparison
sr_path = 'results_v2/agentic_samples.jsonl'
sr_map = {}
if os.path.exists(sr_path):
    for line in open(sr_path):
        if line.strip():
            r = json.loads(line)
            sr_map[r.get('qid', '')] = r

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
    # Harm/Rescue vs single-round
    harm = rescue = 0
    for r in records:
        qid = r.get('qid', '')
        if qid in sr_map:
            sr_correct = sr_map[qid].get('correct', False)
            if sr_correct and not r.get('correct'):
                harm += 1
            elif not sr_correct and r.get('correct'):
                rescue += 1
    print(f'{name:<22} {n:>5} {correct:>8} {100*correct/max(n,1):>6.2f}% {harm:>5} {rescue:>7} {pf:>10} {mr:>11.2f} {ml:>7.1f}s')

# Per-scene for each config
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

echo "[memory] done."
