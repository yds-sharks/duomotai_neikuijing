#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: tune_sev_fusion_from_cache.py
Purpose:
  Offline weight sweep for fusing retr_score + score_prob + cls_prob from SEV cache JSONL.
  Generates TREC runs and evaluates with eval_runs.py.

No model inference. Only reads existing SEV cache.
"""

import os, json, argparse, itertools, subprocess
from collections import defaultdict

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def get(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

def minmax_norm(arr):
    mn = min(arr)
    mx = max(arr)
    if mx <= mn:
        return [0.0] * len(arr)
    inv = 1.0 / (mx - mn)
    return [(x - mn) * inv for x in arr]

def rrf_from_scores(scores, k=60):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    rank = [0] * len(scores)
    for r, i in enumerate(order, start=1):
        rank[i] = r
    return [1.0 / (k + rank[i]) for i in range(len(scores))]

def eval_run(eval_py, qrels, run_path, out_csv, per_query_out):
    cmd = ["python", eval_py, "--qrels", qrels, "--runs", run_path, "--metrics_out", out_csv, "--per_query_out", per_query_out]
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--eval_py", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--mode", choices=["minmax", "rrf"], default="minmax")
    ap.add_argument("--w_retr", default="0.5,1.0,1.5")
    ap.add_argument("--w_score", default="1.5,2.0,2.5,3.0")
    ap.add_argument("--w_cls", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--rrf_k", type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    W_retr = [float(x) for x in args.w_retr.split(",") if x.strip() != ""]
    W_score = [float(x) for x in args.w_score.split(",") if x.strip() != ""]
    W_cls  = [float(x) for x in args.w_cls.split(",") if x.strip() != ""]

    # 1) load cache once
    buckets = defaultdict(list)
    with open(args.cache, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            d=json.loads(line)
            qid=str(get(d, ["qid","query_id","q"]))
            doc=str(get(d, ["pk","docid","doc_id","id"]))
            retr=safe_float(get(d, ["retr_score","retrieval_score","score"], 0.0))
            sprob=safe_float(get(d, ["score_prob"], 0.0))
            cprob=safe_float(get(d, ["cls_prob"], 0.0))
            buckets[qid].append((doc, retr, sprob, cprob))

    qids=list(buckets.keys())
    total=sum(len(v) for v in buckets.values())
    print(f"[OK] loaded cache qids={len(qids)} total_pairs={total}")

    # 2) precompute normalized vectors per qid
    pre={}
    for qid, arr in buckets.items():
        docs=[x[0] for x in arr]
        retr=[x[1] for x in arr]
        sprob=[x[2] for x in arr]
        cprob=[x[3] for x in arr]
        if args.mode=="minmax":
            r=minmax_norm(retr)
            s=sprob  # keep raw prob
            c=cprob  # keep raw prob
        else:
            r=rrf_from_scores(retr, k=args.rrf_k)
            s=rrf_from_scores(sprob, k=args.rrf_k)
            c=rrf_from_scores(cprob, k=args.rrf_k)
        pre[qid]=(docs,r,s,c)

    # 3) sweep
    summary=os.path.join(args.out_dir, f"sweep_fusion_{args.mode}.csv")
    with open(summary, "w", encoding="utf-8") as out:
        out.write("mode,w_retr,w_score,w_cls,run_path,nDCG@10,MRR@10,Recall@10,Recall@50,Recall@100\n")
        for wR,wS,wC in itertools.product(W_retr,W_score,W_cls):
            tag=f"fuse_{args.mode}_wr{wR}_ws{wS}_wc{wC}"
            run_path=os.path.join(args.out_dir, tag+".trec")
            csv_path=os.path.join(args.out_dir, tag+".csv")

            with open(run_path, "w", encoding="utf-8") as rf:
                for qid in qids:
                    docs,r,s,c=pre[qid]
                    fused=[(docs[i], wR*r[i]+wS*s[i]+wC*c[i]) for i in range(len(docs))]
                    fused.sort(key=lambda x:x[1], reverse=True)
                    for rank,(doc,sc) in enumerate(fused[:args.topk], start=1):
                        rf.write(f"{qid} Q0 {doc} {rank} {sc:.6f} {tag}\n")

            perq_path = os.path.join(args.out_dir, tag + "_perq.json")
            eval_run(args.eval_py, args.qrels, run_path, csv_path, perq_path)

            import pandas as pd
            df=pd.read_csv(csv_path)
            row=df.iloc[0].to_dict()
            out.write(f"{args.mode},{wR},{wS},{wC},{run_path},{row['nDCG@10']},{row['MRR@10']},{row['Recall@10']},{row['Recall@50']},{row['Recall@100']}\n")
            out.flush()

    print("[OK] sweep done ->", summary)

if __name__=="__main__":
    main()
