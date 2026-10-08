#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: fuse_runs_linear_top1000.py
Purpose: Offline linear fusion of two TREC runs (dense + sparse) into a fused top1000 run.
Key: per-query min-max normalization (optional) to reduce score scale mismatch.
Output: standard TREC run: qid Q0 docid rank score tag
"""

import argparse
from collections import defaultdict

def read_run(path: str, topk: int):
    # returns dict[qid] -> list[(docid, score)] in rank order, truncated to topk
    run = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, score, _tag = parts[:6]
            run[qid].append((docid, float(score)))
    # truncate by rank order already in file
    for qid in list(run.keys()):
        run[qid] = run[qid][:topk]
    return run

def minmax_norm(scores_by_doc: dict):
    # scores_by_doc: docid -> score
    if not scores_by_doc:
        return {}
    vals = list(scores_by_doc.values())
    mn, mx = min(vals), max(vals)
    if mx == mn:
        # all same => map to 1.0
        return {d: 1.0 for d in scores_by_doc.keys()}
    return {d: (s - mn) / (mx - mn) for d, s in scores_by_doc.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense_run", required=True)
    ap.add_argument("--sparse_run", required=True)
    ap.add_argument("--out_run", required=True)
    ap.add_argument("--wd", type=float, required=True, help="dense weight")
    ap.add_argument("--ws", type=float, required=True, help="sparse weight")
    ap.add_argument("--topk_in", type=int, default=1000, help="read topk from each run")
    ap.add_argument("--topk_out", type=int, default=1000, help="output topk per query")
    ap.add_argument("--normalize", choices=["none", "minmax"], default="minmax")
    ap.add_argument("--tag", default="linfuse")
    args = ap.parse_args()

    dense = read_run(args.dense_run, args.topk_in)
    sparse = read_run(args.sparse_run, args.topk_in)

    qids = sorted(set(dense.keys()) | set(sparse.keys()))

    with open(args.out_run, "w", encoding="utf-8") as out:
        for qid in qids:
            d_list = dense.get(qid, [])
            s_list = sparse.get(qid, [])

            d_map = {doc: sc for doc, sc in d_list}
            s_map = {doc: sc for doc, sc in s_list}

            if args.normalize == "minmax":
                d_map_n = minmax_norm(d_map)
                s_map_n = minmax_norm(s_map)
            else:
                d_map_n, s_map_n = d_map, s_map

            # union candidates
            cand = set(d_map_n.keys()) | set(s_map_n.keys())
            fused = []
            for doc in cand:
                ds = d_map_n.get(doc, 0.0)
                ss = s_map_n.get(doc, 0.0)
                score = args.wd * ds + args.ws * ss
                fused.append((doc, score))

            fused.sort(key=lambda x: x[1], reverse=True)
            fused = fused[: args.topk_out]

            for r, (docid, sc) in enumerate(fused, start=1):
                out.write(f"{qid} Q0 {docid} {r} {sc:.8f} {args.tag}_wd{args.wd:.2f}_ws{args.ws:.2f}\n")

if __name__ == "__main__":
    main()
