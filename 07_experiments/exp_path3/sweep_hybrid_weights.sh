#!/usr/bin/env bash
set -euo pipefail

# Required env:
# PROJECT_ROOT, CONFIG, INPUT_JSON, OUT, QRELS, ELIG

TOPK=100
K_DENSE=400
K_SPARSE=400
BATCH=8
EMBED_BS=64

RUNS_BASE="$OUT/runs"
METRICS_DIR="$OUT/metrics"
mkdir -p "$RUNS_BASE" "$METRICS_DIR"

# Sweep grid (w_sparse); w_dense = 1 - w_sparse
WS_LIST=("0.05" "0.10" "0.20" "0.30" "0.40")

SUMMARY="$METRICS_DIR/top100_k400_weight_sweep.csv"
echo "w_dense,w_sparse,nDCG@10,MRR@10,Recall@10,Recall@50,Recall@100,run_dir" > "$SUMMARY"

cd "$PROJECT_ROOT"

for WS in "${WS_LIST[@]}"; do
  # compute w_dense with python to avoid bc issues
  WD=$(python - <<PY
ws=float("$WS")
print(f"{1.0-ws:.2f}")
PY
)

  TAG="top${TOPK}_k${K_DENSE}_wd${WD}_ws${WS}"
  RUN_DIR="$RUNS_BASE/$TAG"
  rm -rf "$RUN_DIR"
  mkdir -p "$RUN_DIR"

  echo "[RUN] hybrid weights: w_dense=$WD w_sparse=$WS -> $RUN_DIR"

  python insert/exp_path3/run_retrieval_baselines.py \
    --config "$CONFIG" \
    --input "$INPUT_JSON" \
    --out_dir "$RUN_DIR" \
    --topk "$TOPK" \
    --k_dense "$K_DENSE" \
    --k_sparse "$K_SPARSE" \
    --batch_size "$BATCH" \
    --embed_bs "$EMBED_BS" \
    --qids_file "$ELIG" \
    --modes hybrid \
    --disable_lang_adapt \
    --w_dense "$WD" \
    --w_sparse "$WS"

  METRICS_OUT="$METRICS_DIR/${TAG}_metrics.csv"
  PERQ_OUT="$METRICS_DIR/${TAG}_per_query.json"
  LATEX_OUT="$METRICS_DIR/${TAG}_metrics.tex"

  python insert/exp_path3/eval_runs.py \
    --qrels "$QRELS" \
    --runs "$RUN_DIR/hybrid.trec" \
    --metrics_out "$METRICS_OUT" \
    --per_query_out "$PERQ_OUT" \
    --latex_out "$LATEX_OUT"

  # Parse the 2nd line of metrics csv
  LINE=$(tail -n 1 "$METRICS_OUT")
  # LINE format: run_file,nDCG@10,MRR@10,Recall@10,Recall@50,Recall@100,num_qids
  ND=$(echo "$LINE" | cut -d, -f2)
  MR=$(echo "$LINE" | cut -d, -f3)
  R10=$(echo "$LINE" | cut -d, -f4)
  R50=$(echo "$LINE" | cut -d, -f5)
  R100=$(echo "$LINE" | cut -d, -f6)

  echo "${WD},${WS},${ND},${MR},${R10},${R50},${R100},${RUN_DIR}" >> "$SUMMARY"
done

echo "[DONE] Summary -> $SUMMARY"
