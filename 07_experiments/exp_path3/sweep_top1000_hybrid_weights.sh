#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data_1/yds/RAG/Hybrid_milvus/总版"
OUT="/mnt/data_1/yds/微调/实验/SIGIR补充/path3_p1_full_run"

QRELS="$OUT/align/qrels_all_p1.tsv"
EVAL="$ROOT/insert/exp_path3/eval_runs.py"

DENSE_RUN="$OUT/runs/top1000/dense.trec"
SPARSE_RUN="$OUT/runs/top1000/sparse.trec"

WORK="$OUT/metrics/top1000_hybrid_weight_sweep"
RUN_DIR="$WORK/runs"
MET_DIR="$WORK/metrics"
PERQ_DIR="$WORK/per_query"
mkdir -p "$RUN_DIR" "$MET_DIR" "$PERQ_DIR"

SUMMARY="$WORK/sweep_summary.csv"
echo "wd,ws,nDCG@10,MRR@10,Recall@10,Recall@50,Recall@100,run_path,metrics_csv" > "$SUMMARY"

# ---------- 生成一个融合脚本（min-max per qid 归一化后线性融合） ----------
FUSER="$WORK/fuse_dense_sparse_topk1000.py"
cat > "$FUSER" <<'PY'
import argparse, math
from collections import defaultdict

def read_run(path, topk=1000):
    # TREC: qid Q0 docid rank score tag
    by_q = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6: 
                continue
            qid, _, docid, rank, score, tag = parts[:6]
            by_q[qid].append((docid, float(score)))
    # 确保最多 topk
    for qid in list(by_q.keys()):
        by_q[qid] = by_q[qid][:topk]
    return by_q

def minmax_norm(scores_dict):
    # scores_dict: {docid: score}
    if not scores_dict:
        return {}
    vals = list(scores_dict.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-12:
        return {k: 0.0 for k in scores_dict.keys()}
    return {k: (v - mn) / (mx - mn) for k, v in scores_dict.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wd", type=float, required=True)
    ap.add_argument("--ws", type=float, required=True)
    ap.add_argument("--topk_in", type=int, default=1000)
    ap.add_argument("--topk_out", type=int, default=1000)
    ap.add_argument("--tag", default="hybrid")
    args = ap.parse_args()

    dense_byq = read_run(args.dense, topk=args.topk_in)
    sparse_byq = read_run(args.sparse, topk=args.topk_in)

    # 以 qid 并集为准
    qids = sorted(set(dense_byq.keys()) | set(sparse_byq.keys()))
    with open(args.out, "w", encoding="utf-8") as w:
        for qid in qids:
            d = {doc: s for doc, s in dense_byq.get(qid, [])}
            s = {doc: sc for doc, sc in sparse_byq.get(qid, [])}
            # union docs
            docs = set(d.keys()) | set(s.keys())
            d_norm = minmax_norm(d)
            s_norm = minmax_norm(s)
            fused = []
            for doc in docs:
                fd = d_norm.get(doc, 0.0)
                fs = s_norm.get(doc, 0.0)
                fused_score = args.wd * fd + args.ws * fs
                fused.append((doc, fused_score))
            fused.sort(key=lambda x: x[1], reverse=True)
            fused = fused[:args.topk_out]
            for i, (doc, sc) in enumerate(fused, start=1):
                # qid Q0 docid rank score tag
                w.write(f"{qid} Q0 {doc} {i} {sc:.8f} {args.tag}\n")

if __name__ == "__main__":
    main()
PY

# ---------- sweep ----------
cd "$ROOT" || exit 1

for wd in $(python - <<'PY'
import numpy as np
vals = np.round(np.arange(0.20, 0.95 + 1e-9, 0.05), 2)
for v in vals: print(f"{v:.2f}")
PY
); do
  ws=$(python - <<PY
wd=float("$wd")
print(f"{1.0-wd:.2f}")
PY
)

  NAME="hybrid_top1000_wd${wd}_ws${ws}"
  RUN_OUT="$RUN_DIR/${NAME}.trec"
  MET_OUT="$MET_DIR/${NAME}.csv"
  PERQ_OUT="$PERQ_DIR/${NAME}.json"

  echo "[RUN] fuse wd=$wd ws=$ws -> $RUN_OUT"
  python "$FUSER" \
    --dense "$DENSE_RUN" \
    --sparse "$SPARSE_RUN" \
    --out "$RUN_OUT" \
    --wd "$wd" --ws "$ws" \
    --topk_in 1000 --topk_out 1000 \
    --tag "$NAME"

  echo "[EVAL] $RUN_OUT"
  python "$EVAL" \
    --qrels "$QRELS" \
    --runs "$RUN_OUT" \
    --metrics_out "$MET_OUT" \
    --per_query_out "$PERQ_OUT"

  # 从 metrics csv 里抽一行写汇总（假设第2行是数值）
  line=$(tail -n +2 "$MET_OUT" | head -n 1)
  # 兼容字段顺序：用 python 解析表头找列
  python - <<PY >> "$SUMMARY"
import csv
met="$MET_OUT"
with open(met,'r',encoding='utf-8') as f:
    r=csv.DictReader(f)
    row=next(r)
wd="$wd"; ws="$ws"
def g(k):
    return row.get(k, row.get(k.replace("@",""), ""))
print(",".join([wd,ws,
               row.get("nDCG@10",""),
               row.get("MRR@10",""),
               row.get("Recall@10",""),
               row.get("Recall@50",""),
               row.get("Recall@100",""),
               "$RUN_OUT",
               met]))
PY

done

echo "[DONE] summary -> $SUMMARY"
echo "Top 10 by nDCG@10:"
python - <<'PY'
import csv
p="/mnt/data_1/yds/微调/实验/SIGIR补充/path3_p1_full_run/metrics/top1000_hybrid_weight_sweep/sweep_summary.csv"
rows=[]
with open(p,'r',encoding='utf-8') as f:
    r=csv.DictReader(f)
    for row in r:
        try:
            row["_ndcg"]=float(row["nDCG@10"])
        except:
            row["_ndcg"]=-1.0
        rows.append(row)
rows.sort(key=lambda x:x["_ndcg"], reverse=True)
for i,row in enumerate(rows[:10],1):
    print(i, "wd",row["wd"],"ws",row["ws"],"nDCG@10",row["nDCG@10"],"MRR@10",row["MRR@10"])
PY
