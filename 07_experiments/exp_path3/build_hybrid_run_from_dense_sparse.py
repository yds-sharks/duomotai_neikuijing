#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: build_hybrid_run_from_dense_sparse.py
Purpose:
  Fuse dense/sparse TREC run files into a hybrid run using per-qid min-max normalization.

Input (TREC run format):
  qid Q0 docid rank score tag

Output:
  hybrid run file (same format), top_k per qid.

Example:
  python insert/exp_path3/build_hybrid_run_from_dense_sparse.py \
    --dense_run /path/to/dense_top1000.trec \
    --sparse_run /path/to/sparse_top1000.trec \
    --out_run   /path/to/hybrid_wd0.80_ws0.20_top1000.trec \
    --w_dense 0.8 --w_sparse 0.2 --top_k 1000
"""

import argparse
from collections import defaultdict

def read_trec_run(path: str):
    by_qid = defaultdict(dict)  # qid -> {docid: score}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Expect: qid Q0 docid rank score tag  (6 columns)
            if len(parts) < 6:
                raise ValueError(f"Bad TREC line (need >=6 cols): {line}")
            qid, docid, score = parts[0], parts[2], float(parts[4])
            by_qid[qid][docid] = score
    return by_qid

def minmax_norm(scores: dict):
    # scores: docid -> score
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx <= mn:
        return {d: 0.0 for d in scores}
    denom = (mx - mn)
    return {d: (s - mn) / denom for d, s in scores.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense_run", required=True)
    ap.add_argument("--sparse_run", required=True)
    ap.add_argument("--out_run", required=True)
    ap.add_argument("--w_dense", type=float, required=True)
    ap.add_argument("--w_sparse", type=float, required=True)
    ap.add_argument("--top_k", type=int, default=1000)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    dense = read_trec_run(args.dense_run)
    sparse = read_trec_run(args.sparse_run)

    # union qids
    all_qids = sorted(set(dense.keys()) | set(sparse.keys()))
    tag = args.tag or f"hybrid_wd{args.w_dense:.2f}_ws{args.w_sparse:.2f}"

    with open(args.out_run, "w", encoding="utf-8") as out:
        for qid in all_qids:
            d_scores = dense.get(qid, {})
            s_scores = sparse.get(qid, {})
            all_docs = set(d_scores.keys()) | set(s_scores.keys())

            d_norm = minmax_norm(d_scores)
            s_norm = minmax_norm(s_scores)

            fused = []
            for docid in all_docs:
                dn = d_norm.get(docid, 0.0)
                sn = s_norm.get(docid, 0.0)
                fs = args.w_dense * dn + args.w_sparse * sn
                fused.append((docid, fs))

            fused.sort(key=lambda x: x[1], reverse=True)
            fused = fused[: args.top_k]

            for rank, (docid, score) in enumerate(fused, start=1):
                out.write(f"{qid} Q0 {docid} {rank} {score:.8f} {tag}\n")

if __name__ == "__main__":
    main()
