#!/bin/bash
# 10-question verification: confirm --no-mplus/--no-mminus actually control memory read/write
set -e
cd /mnt/data_1/yds/多模态/endobench_eval

PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
CKPT=/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/ckpt_grpo_v4_u50
MILVUS_DIR=/mnt/data_1/yds/多模态/data_house/milvus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p verify_10

COMMON="--mode agentic --ctrl-model $CKPT --max-rounds 3 --offset 360 --limit 10 \
  --use-organ-filter --log-every 1 --seed 42 \
  --cand-cache verify_10/cand_cache.jsonl"

echo "============================================"
echo "[verify] Run 1: Full memory (M+ + M-)"
echo "============================================"
CUDA_VISIBLE_DEVICES=0,1 $PY eval_endobench.py $COMMON \
  --milvus-db-path $MILVUS_DIR/multimodal_vector_indexes.db \
  --ctrl-device cuda:0 --gen-device cuda:1 \
  --retr-text-device cuda:1 --retr-image-device cuda:0 \
  --out-dir verify_10/full \
  2>&1 | tee logs/verify_full.log &

echo "============================================"
echo "[verify] Run 2: No memory (--no-mplus --no-mminus)"
echo "============================================"
CUDA_VISIBLE_DEVICES=2,3 $PY eval_endobench.py $COMMON \
  --no-mplus --no-mminus \
  --milvus-db-path $MILVUS_DIR/multimodal_vector_indexes_shard1.db \
  --ctrl-device cuda:0 --gen-device cuda:1 \
  --retr-text-device cuda:1 --retr-image-device cuda:0 \
  --out-dir verify_10/nomem \
  2>&1 | tee logs/verify_nomem.log &

echo "[verify] Both runs started, waiting..."
wait

echo ""
echo "============================================"
echo "[verify] Comparing memory flags..."
echo "============================================"
$PY -c "
import json

for name, path in [('Full', 'verify_10/full/agentic_samples.jsonl'),
                   ('NoMem', 'verify_10/nomem/agentic_samples.jsonl')]:
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f'\n--- {name} ({len(records)} samples) ---')
    for r in records:
        rounds = r.get('rounds', [])
        multi = len(rounds) > 1
        n_ret = [rd.get('n_retained', 0) for rd in rounds]
        n_bc = [rd.get('n_breadcrumb', 0) for rd in rounds]
        n_cands = [rd.get('n_cands', 0) for rd in rounds]
        print(f'  {r[\"qid\"]}: rounds={len(rounds)}, action={r.get(\"action\")}, '
              f'n_retained={n_ret}, n_breadcrumb={n_bc}, n_cands={n_cands}')

# Compare
print('\n============================================')
print('VERIFICATION RESULT')
print('============================================')
full_recs = [json.loads(l) for l in open('verify_10/full/agentic_samples.jsonl') if l.strip()]
nomem_recs = [json.loads(l) for l in open('verify_10/nomem/agentic_samples.jsonl') if l.strip()]

full_has_retained = any(rd.get('n_retained', 0) > 0 for r in full_recs for rd in r.get('rounds', []))
full_has_breadcrumb = any(rd.get('n_breadcrumb', 0) > 0 for r in full_recs for rd in r.get('rounds', []))
nomem_has_retained = any(rd.get('n_retained', 0) > 0 for r in nomem_recs for rd in r.get('rounds', []))
nomem_has_breadcrumb = any(rd.get('n_breadcrumb', 0) > 0 for r in nomem_recs for rd in r.get('rounds', []))

print(f'Full:  has_retained={full_has_retained}, has_breadcrumb={full_has_breadcrumb}')
print(f'NoMem: has_retained={nomem_has_retained}, has_breadcrumb={nomem_has_breadcrumb}')

ok = (not nomem_has_retained) and (not nomem_has_breadcrumb)
print(f'\\nPASS: {ok}' if ok else f'\\nFAIL: no-mplus/no-mminus did not disable memory!')
"
