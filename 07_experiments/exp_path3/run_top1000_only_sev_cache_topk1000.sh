#!/usr/bin/env bash
set -euo pipefail

# =========================
# 0) 固定路径（如需改，改这里）
# =========================
ROOT="/mnt/data_1/yds/RAG/Hybrid_milvus/总版"
OUT="/mnt/data_1/yds/微调/实验/SIGIR补充/path3_p1_full_run"

CONFIG="${CONFIG:-$ROOT/insert/exp_path3/config_path3_sev_merged.yaml}"

RUN_IN="${RUN_IN:-$OUT/runs/top1000_hybrid_wd0.80_ws0.20/hybrid_top1000_wd0.80_ws0.20.trec}"
INPUT_JSON="$OUT/queries_path3_2349_sev_input.json"

QIDS_FILE="${QIDS_FILE:-$OUT/qids_eligible_all.txt}"
QRELS="${QRELS:-$OUT/align/qrels_all_p1.tsv}"

MET_DIR="${MET_DIR:-$OUT/metrics/top1000_only_sev_cache_full2349_topk1000}"
LOG_DIR="${LOG_DIR:-$OUT/logs}"

mkdir -p "$MET_DIR" "$LOG_DIR"

CACHE_OUT="${CACHE_OUT:-$MET_DIR/sev_cache_top1000_2349q_topk1000.jsonl}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/top1000_only_sev_cache_full2349_topk1000_$(date +%Y%m%d_%H%M%S).log}"

TOPK_IN="${TOPK_IN:-1000}"
TOPK_OUT="${TOPK_OUT:-1000}"

TH_CENTER="${TH_CENTER:-0.5}"
TH_SPAN="${TH_SPAN:-0.0}"
TH_STEP="${TH_STEP:-0.05}"

DTYPE="${DTYPE:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-512}"

# prompt 单独变量，避免与 tee 串行
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-$'文段是否支持回答问题：{question}\n文段如下：{passage}'}

cd "$ROOT"

# =========================
# 1) 打印配置（你确认）
# =========================
echo "================= Path3 SEV Top1000 Cache (TOPK_IN=$TOPK_IN TOPK_OUT=$TOPK_OUT) ================="
echo "[ROOT]      $ROOT"
echo "[OUT]       $OUT"
echo "[CONFIG]    $CONFIG"
echo "[RUN_IN]    $RUN_IN"
echo "[INPUT_JSON]$INPUT_JSON"
echo "[QIDS_FILE] $QIDS_FILE"
echo "[QRELS]     $QRELS"
echo "[MET_DIR]   $MET_DIR"
echo "[CACHE_OUT] $CACHE_OUT"
echo "[LOG_FILE]  $LOG_FILE"
echo "[DTYPE]     $DTYPE   [BATCH_SIZE] $BATCH_SIZE   [MAX_LENGTH] $MAX_LENGTH"
echo "[TH]        center=$TH_CENTER span=$TH_SPAN step=$TH_STEP"
echo "----------------- PROMPT_TEMPLATE -----------------"
printf "%s\n" "$PROMPT_TEMPLATE"
echo "---------------------------------------------------"

# =========================
# 2) 强校验：文件存在
# =========================
echo "[CHK] existence..."
ls -lh "$CONFIG" "$RUN_IN" "$INPUT_JSON" "$QIDS_FILE" "$QRELS" >/dev/null

# =========================
# 3) 强校验：run_in 必须 2349*1000
# =========================
echo "[CHK] run_in shape..."
RUN_LINES=$(wc -l < "$RUN_IN" | tr -d ' ')
echo "  run_in lines = $RUN_LINES (expect 2349000)"
if [[ "$RUN_LINES" != "2349000" ]]; then
  echo "[FATAL] RUN_IN lines mismatch, stop."
  exit 2
fi

SAMPLE_QID=$(head -n 1 "$RUN_IN" | awk '{print $1}')
SAMPLE_CNT=$(awk -v q="$SAMPLE_QID" '$1==q{c++} END{print c+0}' "$RUN_IN")
echo "  sample qid=$SAMPLE_QID has lines=$SAMPLE_CNT (expect 1000)"
if [[ "$SAMPLE_CNT" != "1000" ]]; then
  echo "[FATAL] RUN_IN per-qid not 1000, stop."
  exit 3
fi

# =========================
# 4) 强校验：qids=2349，qrels=2349，queries=2349 且含 qid
# =========================
echo "[CHK] qids/qrels/queries..."
QIDS_N=$(wc -l < "$QIDS_FILE" | tr -d ' ')
QRELS_N=$(wc -l < "$QRELS" | tr -d ' ')
echo "  qids lines  = $QIDS_N (expect 2349)"
echo "  qrels lines = $QRELS_N (expect 2349)"
if [[ "$QIDS_N" != "2349" ]]; then
  echo "[FATAL] QIDS_FILE not 2349, stop."
  exit 4
fi
if [[ "$QRELS_N" != "2349" ]]; then
  echo "[FATAL] QRELS not 2349, stop."
  exit 5
fi

python - <<PY
import json
p="$INPUT_JSON"
obj=json.load(open(p,'r',encoding='utf-8'))
assert isinstance(obj,list), ("queries json must be list", type(obj))
assert len(obj)==2349, ("queries len must be 2349", len(obj))
k=obj[0].keys()
assert "qid" in k and "question" in k, ("missing keys", k)
print("  queries ok: len=",len(obj)," sample_keys=",list(obj[0].keys()))
PY

# =========================
# 5) 打印 config 结构（避免 paths/milvus 缺失）
# =========================
python - <<PY
import yaml, pprint
cfg=yaml.safe_load(open("$CONFIG","r",encoding="utf-8"))
print("[CFG] top keys:", sorted(list(cfg.keys())) if isinstance(cfg,dict) else type(cfg))
for k in ["paths","milvus","retrieval","sev","model","reranker"]:
    if isinstance(cfg,dict) and k in cfg:
        print(f"[CFG] has '{k}'")
PY

echo "[RUN] start SEV cache + (optional) tiny sweep ..."

# =========================
# 6) 正式运行（关键：明确 topk_in/out=1000）
# =========================
python insert/exp_path3/run_sev_cache_and_sweep.py \
  --config "$CONFIG" \
  --input_json "$INPUT_JSON" \
  --qids_file "$QIDS_FILE" \
  --run_in "$RUN_IN" \
  --qrels "$QRELS" \
  --eval_py "insert/exp_path3/eval_runs.py" \
  --out_dir "$MET_DIR" \
  --cache_out "$CACHE_OUT" \
  --topk_in "$TOPK_IN" \
  --topk_out "$TOPK_OUT" \
  --batch_size "$BATCH_SIZE" \
  --max_length "$MAX_LENGTH" \
  --th_center "$TH_CENTER" \
  --th_span "$TH_SPAN" \
  --th_step "$TH_STEP" \
  --dtype "$DTYPE" \
  --prompt_template "$PROMPT_TEMPLATE" \
  2>&1 | tee "$LOG_FILE"

echo "[DONE] cache -> $CACHE_OUT"
echo "[DONE] log   -> $LOG_FILE"

# 结果快速验收：行数必须 2349000（=2349*1000）
echo "[CHK] cache lines (expect 2349000):"
wc -l "$CACHE_OUT"
