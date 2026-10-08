#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: eval_runs.py
Purpose:
  Evaluate runs with qrels: nDCG@10 (graded), MRR@10, Recall@10/50/100.
Inputs:
  --qrels: TREC qrels (qid 0 docid rel)
  --runs:  one or more TREC run files
Outputs:
  --metrics_out: CSV summary per run
  --per_query_out: JSON with per-query metrics per run
  --latex_out: LaTeX table (numbers are computed, not fabricated)
"""
import argparse, csv, json, math, os
from typing import Dict, List, Tuple

def read_qrels(path: str) -> Dict[str, Dict[str, int]]:
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, _, docid, rel = line.split()
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels

def read_run(path: str) -> Dict[str, List[Tuple[str, float]]]:
    run = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, _, docid, rank, score, _ = line.split()
            run.setdefault(qid, []).append((docid, float(score)))
    # ensure stable order: by provided rank already, but keep insertion order
    return run

def dcg(rels: List[int]) -> float:
    s = 0.0
    for i, rel in enumerate(rels, start=1):
        gain = (2**rel - 1)
        s += gain / math.log2(i + 1)
    return s

def ndcg_at_k(qrels_q: Dict[str, int], ranked: List[str], k: int) -> float:
    rels = [qrels_q.get(docid, 0) for docid in ranked[:k]]
    ideal = sorted(qrels_q.values(), reverse=True)[:k]
    denom = dcg(ideal)
    if denom <= 0:
        return 0.0
    return dcg(rels) / denom

def mrr_at_k(qrels_q: Dict[str, int], ranked: List[str], k: int) -> float:
    for i, docid in enumerate(ranked[:k], start=1):
        if qrels_q.get(docid, 0) > 0:
            return 1.0 / i
    return 0.0

def recall_at_k(qrels_q: Dict[str, int], ranked: List[str], k: int) -> float:
    rel_docs = {d for d, r in qrels_q.items() if r > 0}
    if not rel_docs:
        return 0.0
    got = sum(1 for d in ranked[:k] if d in rel_docs)
    return got / len(rel_docs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--metrics_out", required=True)
    ap.add_argument("--per_query_out", required=True)
    ap.add_argument("--latex_out", default="")
    args = ap.parse_args()

    qrels = read_qrels(args.qrels)
    qids = sorted(qrels.keys())

    per_query = {}
    summary_rows = []

    for run_path in args.runs:
        run = read_run(run_path)
        run_name = os.path.basename(run_path)

        pq = {}
        nds, mrrs, r10s, r50s, r100s = [], [], [], [], []

        for qid in qids:
            ranked = [docid for docid, _ in run.get(qid, [])]
            nd = ndcg_at_k(qrels[qid], ranked, 10)
            mr = mrr_at_k(qrels[qid], ranked, 10)
            r10 = recall_at_k(qrels[qid], ranked, 10)
            r50 = recall_at_k(qrels[qid], ranked, 50)
            r100 = recall_at_k(qrels[qid], ranked, 100)

            pq[qid] = {"ndcg@10": nd, "mrr@10": mr, "recall@10": r10, "recall@50": r50, "recall@100": r100}
            nds.append(nd); mrrs.append(mr); r10s.append(r10); r50s.append(r50); r100s.append(r100)

        per_query[run_name] = pq
        summary_rows.append({
            "run_file": run_name,
            "nDCG@10": sum(nds)/max(1,len(nds)),
            "MRR@10": sum(mrrs)/max(1,len(mrrs)),
            "Recall@10": sum(r10s)/max(1,len(r10s)),
            "Recall@50": sum(r50s)/max(1,len(r50s)),
            "Recall@100": sum(r100s)/max(1,len(r100s)),
            "num_qids": len(qids),
        })

    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    os.makedirs(os.path.dirname(args.per_query_out), exist_ok=True)
    with open(args.per_query_out, "w", encoding="utf-8") as f:
        json.dump(per_query, f, ensure_ascii=False, indent=2)

    if args.latex_out:
        os.makedirs(os.path.dirname(args.latex_out), exist_ok=True)
        with open(args.latex_out, "w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
            f.write("Run & nDCG@10 & MRR@10 & R@10 & R@50 & R@100 \\\\\n\\midrule\n")
            for r in summary_rows:
                f.write(f"{r['run_file']} & {r['nDCG@10']:.4f} & {r['MRR@10']:.4f} & {r['Recall@10']:.4f} & {r['Recall@50']:.4f} & {r['Recall@100']:.4f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")

    print(json.dumps({"num_runs": len(summary_rows), "num_qids": len(qids), "metrics_out": args.metrics_out}, ensure_ascii=False))

if __name__ == "__main__":
    main()
