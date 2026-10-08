#!/usr/bin/env bash
set -euo pipefail

# =========================
# 0) 你需要确认/填好的路径
# =========================
PROJECT_ROOT="/mnt/data_1/yds/RAG/Hybrid_milvus/总版"
OUT="/mnt/data_1/yds/微调/实验/SIGIR补充/path3_p1_full_run"

# Path3 的配置、输入、qids、qrels（沿用你之前 top100 的那套）
CONFIG="${CONFIG:-$PROJECT_ROOT/insert/exp_path3/config_path3_p1.yaml}"
INPUT_JSON="${INPUT_JSON:-/mnt/data_1/yds/微调/实验/实验二/实验二_医学TOP300_带参考答案.json}"   # 你自己如果已有就保持一致
ELIG="${ELIG:-$OUT/qids_eligible.txt}"
QRELS="${QRELS:-$OUT/qrels_path3_p1.tsv}"

# 你今晚要用的 top1000 的 dense/sparse run（TREC 格式）
# 一般在 $OUT/runs/top1000/ 下；如果你的实际位置不同，改这里即可
DENSE_TOP1000_RUN="${DENSE_TOP1000_RUN:-$OUT/runs/top1000/dense.trec}"
SPARSE_TOP1000_RUN="${SPARSE_TOP1000_RUN:-$OUT/runs/top1000/sparse.trec}"

# Hybrid 权重（今晚固定）
WD=0.80
WS=0.20
TOPK=1000

# 输出目录（单独开一个，避免和之前 top100 混）
RUN_DIR="$OUT/runs/top1000_hybrid_wd0.80_ws0.20"
MET_DIR="$OUT/metrics/top1000_hybrid_wd0.80_ws0.20_sev_cache"
LOG_DIR="$OUT/logs"
mkdir -p "$RUN_DIR" "$MET_DIR" "$LOG_DIR"

HYBRID_RUN="$RUN_DIR/hybrid_top1000_wd0.80_ws0.20.trec"
CACHE_OUT="$MET_DIR/sev_cache_top1000.jsonl"
LOG_FILE="$LOG_DIR/top1000_hybrid08_02_sev_cache_$(date +%Y%m%d_%H%M%S).log"

cd "$PROJECT_ROOT"

echo "[1/4] Sanity check dense/sparse run files..."
ls -lh "$DENSE_TOP1000_RUN" "$SPARSE_TOP1000_RUN"

echo "[2/4] Build hybrid run (wd=${WD}, ws=${WS}, topk=${TOPK}) ..."
python insert/exp_path3/build_hybrid_run_from_dense_sparse.py \
  --dense_run "$DENSE_TOP1000_RUN" \
  --sparse_run "$SPARSE_TOP1000_RUN" \
  --out_run   "$HYBRID_RUN" \
  --w_dense   "$WD" \
  --w_sparse  "$WS" \
  --top_k     "$TOPK" \
  --tag       "hybrid_top1000_wd0.80_ws0.20"

echo "Hybrid run saved: $HYBRID_RUN"
echo "Hybrid run quick stats:"
wc -l "$HYBRID_RUN"
head -n 3 "$HYBRID_RUN"

echo "[3/4] Run SEV scoring + cache for top1000 candidates (this is the heavy part) ..."
# 说明：
# - 这里直接复用你现成的 run_sev_cache_and_sweep.py
# - 目的：把 top1000 的 (qid, docid) 全部打分并写入 cache_out
# - sweep 参数今晚可以给一个很粗的/或者中心附近的小 sweep；明天你再基于 cache 做细扫/融合
python insert/exp_path3/run_sev_cache_and_sweep.py \
  --config "$CONFIG" \
  --input_json "$INPUT_JSON" \
  --qids_file "$ELIG" \
  --run_in "$HYBRID_RUN" \
  --qrels "$QRELS" \
  --eval_py "insert/exp_path3/eval_runs.py" \
  --out_dir "$MET_DIR" \
  --cache_out "$CACHE_OUT" \
  --th_center 0.5 \
  --th_span 0.0 \
  --th_step 0.05 \
  --dtype bf16 \
  --prompt_template $'文段是否支持回答问题：{question}\n文段如下：{passage}' \
  2>&1 | tee "$LOG_FILE"

echo "[4/4] Done."
echo "Cache: $CACHE_OUT"
echo "Log:   $LOG_FILE"
