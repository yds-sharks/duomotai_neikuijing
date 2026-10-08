#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: fuse_runs_rrf_top1000.py
Purpose: Offline RRF fusion of two TREC runs (dense + sparse) into a fused top1000 run.
RRF score = sum(1 / (k + rank)), robust to score scale mismatch.
"""

import argparse
from collections import defaultdict

def read_rank(path: str, topk: int):
    # returns dict[qid] -> dict[docid] = rank (1-based), truncated
    ranks = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, _score, _tag = parts[:6]
            rank = int(rank)
            if rank <= topk and docid not in ranks[qid]:
                ranks[qid][docid] = rank
    return ranks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense_run", required=True)
    ap.add_argument("--sparse_run", required=True)
    ap.add_argument("--out_run", required=True)
    ap.add_argument("--k", type=int, default=60, help="RRF constant (typical 60)")
    ap.add_argument("--topk_in", type=int, default=1000)
    ap.add_argument("--topk_out", type=int, default=1000)
    ap.add_argument("--tag", default="rrf")
    args = ap.parse_args()

    d = read_rank(args.dense_run, args.topk_in)
    s = read_rank(args.sparse_run, args.topk_in)

    qids = sorted(set(d.keys()) | set(s.keys()))
    with open(args.out_run, "w", encoding="utf-8") as out:
        for qid in qids:
            d_rank = d.get(qid, {})
            s_rank = s.get(qid, {})
            cand = set(d_rank.keys()) | set(s_rank.keys())
            fused = []
            for doc in cand:
                sc = 0.0
                if doc in d_rank:
                    sc += 1.0 / (args.k + d_rank[doc])
                if doc in s_rank:
                    sc += 1.0 / (args.k + s_rank[doc])
                fused.append((doc, sc))
            fused.sort(key=lambda x: x[1], reverse=True)
            fused = fused[: args.topk_out]
            for r, (docid, sc) in enumerate(fused, start=1):
                out.write(f"{qid} Q0 {docid} {r} {sc:.10f} {args.tag}_k{args.k}\n")

if __name__ == "__main__":
    main()
