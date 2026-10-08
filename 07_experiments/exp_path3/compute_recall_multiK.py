#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: compute_recall_multiK.py
Purpose: Compute Recall@K for single-anchor qrels from a TREC run.
Auto-detect qrels formats:
  - 2 cols: qid docid
  - 3 cols: qid docid rel
  - 4 cols (TREC qrels): qid 0 docid rel
"""

import argparse
from collections import defaultdict

def load_qrels(path: str):
    gold = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            # Auto-detect
            if len(parts) == 2:
                qid, docid = parts[0], parts[1]
            elif len(parts) == 3:
                # assume: qid docid rel
                qid, docid = parts[0], parts[1]
            else:
                # assume TREC qrels: qid 0 docid rel  (or more cols)
                qid, docid = parts[0], parts[2]

            gold[qid] = docid
    return gold

def recall_at_ks(run_path: str, gold: dict, ks):
    cand = {k: defaultdict(set) for k in ks}
    with open(run_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, docid, rank = parts[0], parts[2], int(parts[3])
            if qid not in gold:
                continue
            for k in ks:
                if rank <= k:
                    cand[k][qid].add(docid)

    out = {}
    total = len(gold)
    for k in ks:
        hit = 0
        for qid, g in gold.items():
            if g in cand[k].get(qid, set()):
                hit += 1
        out[k] = hit / total if total else 0.0
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ks", default="10,50,100,200,500,1000")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    gold = load_qrels(args.qrels)
    rec = recall_at_ks(args.run, gold, ks)
    print(" ".join([f"R@{k}={rec[k]:.6f}" for k in ks]))

if __name__ == "__main__":
    main()
