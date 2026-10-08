#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: analyze_rank_shifts.py
Purpose:
  Compare baseline vs new run, find queries where the GOLD doc moved up the most.
"""

import argparse, csv, json
from collections import defaultdict

def read_qrels(path):
    """
    Robust qrels reader.
    Supports:
      - 3 cols: qid docid rel
      - 4 cols: qid 0 docid rel   (TREC)
      - >4 cols: qid ... docid ... rel (use first as qid, last as rel, and a best-effort docid position)
    Strategy:
      - If 4 cols: docid=col[2], rel=col[3]
      - Else if 3 cols: docid=col[1], rel=col[2]
      - Else (>=5): try docid=col[2] if looks like TREC; otherwise docid=col[1]; rel=last
    """
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) == 1:
                cols = line.split()  # fallback to whitespace

            if len(cols) == 3:
                qid, docid, rel = cols[0], cols[1], cols[2]
            elif len(cols) == 4:
                qid, docid, rel = cols[0], cols[2], cols[3]
            else:
                qid = cols[0]
                rel = cols[-1]
                # best-effort docid pick
                docid = cols[2] if len(cols) >= 4 else cols[1]
                # if cols[2] is purely numeric and cols[1] looks like docid, fallback
                if docid.isdigit() and len(cols) >= 2:
                    docid = cols[1]

            # 只保留“正相关”的 gold（通常 rel>0）
            try:
                if float(rel) > 0:
                    qrels[qid] = docid
            except ValueError:
                # rel 不是数字就直接跳过或当作 1（你也可以改成 continue）
                qrels[qid] = docid

    return qrels


def read_trec(path, topk=None):
    # TREC: qid Q0 docid rank score run
    runs = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts=line.strip().split()
            if len(parts) < 6: 
                continue
            qid, _, docid, rank, score, run = parts[:6]
            runs[qid].append((int(rank), docid, float(score)))
    # ensure sorted by rank
    for qid in runs:
        runs[qid].sort(key=lambda x: x[0])
        if topk:
            runs[qid] = runs[qid][:topk]
    return runs

def rank_of(docid, ranked_list):
    for r, d, s in ranked_list:
        if d == docid:
            return r
    return None

def top_docids(ranked_list, k=10):
    return [d for _, d, _ in ranked_list[:k]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--run_a", required=True, help="baseline trec")
    ap.add_argument("--run_b", required=True, help="new trec")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--min_before", type=int, default=30, help="only keep cases with gold_rank_before >= this")
    ap.add_argument("--max_after", type=int, default=20, help="only keep cases with gold_rank_after <= this")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    qrels = read_qrels(args.qrels)
    A = read_trec(args.run_a, topk=args.topk)
    B = read_trec(args.run_b, topk=args.topk)

    rows = []
    for qid, gold in qrels.items():
        if qid not in A or qid not in B:
            continue
        ra = rank_of(gold, A[qid])
        rb = rank_of(gold, B[qid])
        if ra is None or rb is None:
            continue
        delta = ra - rb  # positive means improved
        if ra < args.min_before:
            continue
        if rb > args.max_after:
            continue
        rows.append((delta, qid, ra, rb))

    rows.sort(reverse=True)

    # write csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["delta", "qid", "gold_rank_before", "gold_rank_after"])
        for delta, qid, ra, rb in rows[:args.limit]:
            w.writerow([delta, qid, ra, rb])

    # write jsonl cases
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for delta, qid, ra, rb in rows[:args.limit]:
            gold = qrels[qid]
            item = {
                "qid": qid,
                "gold_docid": gold,
                "gold_rank_before": ra,
                "gold_rank_after": rb,
                "delta": delta,
                "top10_before": top_docids(A[qid], 10),
                "top10_after": top_docids(B[qid], 10),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] cases={len(rows)} wrote {args.out_csv} and {args.out_jsonl}")

if __name__ == "__main__":
    main()
