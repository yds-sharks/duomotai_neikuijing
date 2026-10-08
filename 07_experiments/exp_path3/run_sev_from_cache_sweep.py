#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: run_sev_from_cache_sweep.py
Purpose:
  Re-rank TREC run using cached SEV signals (no model inference), and sweep params.

Supports 2 strategies:
  1) pipeline: stable partition by cls_prob >= th_cls, then sort by score_prob within partitions.
  2) fusion: final = norm(retrieval_score) + lam*score_prob + mu*cls_prob

Inputs:
  - run_in: TREC run file (qid Q0 docid rank score runname)
  - cache_jsonl: cached SEV signals for the same (qid, docid) pairs
  - qrels + eval_py: to evaluate each generated run

Outputs:
  - For each setting: a .trec run file
  - A sweep_summary.csv with metrics

Notes:
  - Assumes run_in provides topk candidates per qid (e.g., 100).
  - If a candidate is missing in cache, it falls back to cls_prob=0.0, score_prob=0.0.
"""

import argparse, csv, json, os, subprocess, sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

@dataclass
class Cand:
    doc: str
    rank: int
    retr_score: float
    cls_prob: float = 0.0
    score_prob: float = 0.0

def parse_run_trec(path: str) -> Dict[str, List[Cand]]:
    q2 = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # qid Q0 doc rank score run
            if len(parts) < 6:
                raise ValueError(f"Bad TREC line: {line}")
            qid, _, doc, rnk, sc, _ = parts[:6]
            q2[qid].append(Cand(doc=str(doc), rank=int(rnk), retr_score=float(sc)))
    # ensure sorted by rank
    for qid in q2:
        q2[qid].sort(key=lambda x: x.rank)
    return q2

def load_cache(cache_jsonl: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    returns cache[qid][doc] = {"cls_prob":..., "score_prob":...}
    supports:
      A) per-qid line: {"qid":..., "results":[{"pk"/"doc"/"docid":..., "cls_prob":..., "score_prob":...}, ...]}
      B) per-pair line: {"qid":..., "pk"/"doc"/"docid":..., "cls_prob":..., "score_prob":...}
    """
    cache = defaultdict(dict)
    with open(cache_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # detect per-qid
            if isinstance(obj, dict) and ("results" in obj) and isinstance(obj["results"], list):
                qid = str(obj.get("qid") or obj.get("query_id") or obj.get("q_id") or obj.get("id"))
                if not qid:
                    # if not provided, skip (shouldn't happen)
                    continue
                for r in obj["results"]:
                    doc = r.get("pk") or r.get("doc") or r.get("docid") or r.get("id")
                    if doc is None:
                        continue
                    cache[qid][str(doc)] = {
                        "cls_prob": float(r.get("cls_prob", 0.0)),
                        "score_prob": float(r.get("score_prob", r.get("score", 0.0))),
                    }
                continue

            # per-pair
            qid = str(obj.get("qid") or obj.get("query_id") or obj.get("q_id") or obj.get("id") or "")
            doc = obj.get("pk") or obj.get("doc") or obj.get("docid")
            if qid and doc is not None:
                cache[qid][str(doc)] = {
                    "cls_prob": float(obj.get("cls_prob", 0.0)),
                    "score_prob": float(obj.get("score_prob", obj.get("score", 0.0))),
                }

    return cache

def norm_scores(cands: List[Cand], method: str) -> List[float]:
    xs = [c.retr_score for c in cands]
    if method == "none":
        return xs
    if not xs:
        return xs
    if method == "minmax":
        mn, mx = min(xs), max(xs)
        if mx - mn < 1e-12:
            return [0.0 for _ in xs]
        return [(x - mn) / (mx - mn) for x in xs]
    if method == "zscore":
        mean = sum(xs) / len(xs)
        var = sum((x-mean)*(x-mean) for x in xs) / max(len(xs), 1)
        std = var ** 0.5
        if std < 1e-12:
            return [0.0 for _ in xs]
        return [(x - mean) / std for x in xs]
    raise ValueError(f"Unknown norm: {method}")

def write_trec(out_path: str, q2cands: Dict[str, List[Cand]], run_name: str, topk_out: int):
    with open(out_path, "w", encoding="utf-8") as f:
        for qid, cands in q2cands.items():
            cands = cands[:topk_out]
            for i, c in enumerate(cands, start=1):
                f.write(f"{qid}\tQ0\t{c.doc}\t{i}\t{float(c.retr_score)}\t{run_name}\n")

def eval_run(eval_py: str, qrels: str, run_path: str, tmp_dir: str) -> Dict[str, float]:
    os.makedirs(tmp_dir, exist_ok=True)
    metrics_csv = os.path.join(tmp_dir, "_metrics.csv")
    perq_json = os.path.join(tmp_dir, "_perq.json")
    latex_tex = os.path.join(tmp_dir, "_m.tex")
    cmd = [
        sys.executable, eval_py,
        "--qrels", qrels,
        "--runs", run_path,
        "--metrics_out", metrics_csv,
        "--per_query_out", perq_json,
        "--latex_out", latex_tex
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # parse first data line
    with open(metrics_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Empty metrics csv")
    row = rows[0]
    return {
        "nDCG@10": float(row["nDCG@10"]),
        "MRR@10": float(row["MRR@10"]),
        "Recall@10": float(row["Recall@10"]),
        "Recall@50": float(row["Recall@50"]),
        "Recall@100": float(row["Recall@100"]),
        "num_qids": float(row["num_qids"]),
    }

def parse_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_in", required=True)
    ap.add_argument("--cache_jsonl", required=True)
    ap.add_argument("--eval_py", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--topk_out", type=int, default=1000)

    ap.add_argument("--strategy", choices=["pipeline", "fusion"], required=True)

    # pipeline params
    ap.add_argument("--cls_th_list", default="0.30,0.35,0.40")
    ap.add_argument("--within_sort", choices=["score_prob", "retr_score"], default="score_prob")
    ap.add_argument("--irrelevant_order", choices=["keep", "score_prob", "retr_score"], default="keep")

    # fusion params
    ap.add_argument("--lam_list", default="0.0,0.25,0.5,1.0,2.0,4.0")
    ap.add_argument("--mu_list", default="0.0,0.25,0.5,1.0")
    ap.add_argument("--norm", choices=["none", "minmax", "zscore"], default="minmax")

    ap.add_argument("--run_prefix", default="sev_sweep")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_eval_dir = os.path.join(args.out_dir, "_tmp_eval")

    q2 = parse_run_trec(args.run_in)
    cache = load_cache(args.cache_jsonl)

    summary_path = os.path.join(args.out_dir, "sweep_summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as fsum:
        fieldnames = ["strategy","param","run_path","nDCG@10","MRR@10","Recall@10","Recall@50","Recall@100","num_qids"]
        w = csv.DictWriter(fsum, fieldnames=fieldnames)
        w.writeheader()

        if args.strategy == "pipeline":
            cls_ths = parse_list(args.cls_th_list)
            for th in cls_ths:
                out_run = os.path.join(args.out_dir, f"{args.run_prefix}_pipeline_cls{th:.2f}.trec")

                q2new = {}
                for qid, cands in q2.items():
                    # attach cache signals
                    for c in cands:
                        sig = cache.get(qid, {}).get(c.doc, {})
                        c.cls_prob = float(sig.get("cls_prob", 0.0))
                        c.score_prob = float(sig.get("score_prob", 0.0))

                    rel = [c for c in cands if c.cls_prob >= th]
                    irr = [c for c in cands if c.cls_prob < th]

                    # sort inside relevant
                    if args.within_sort == "score_prob":
                        rel.sort(key=lambda x: (x.score_prob, -x.rank), reverse=True)
                    else:
                        rel.sort(key=lambda x: (x.retr_score, -x.rank), reverse=True)

                    # irrelevant order
                    if args.irrelevant_order == "keep":
                        irr.sort(key=lambda x: x.rank)
                    elif args.irrelevant_order == "score_prob":
                        irr.sort(key=lambda x: (x.score_prob, -x.rank), reverse=True)
                    else:
                        irr.sort(key=lambda x: (x.retr_score, -x.rank), reverse=True)

                    q2new[qid] = rel + irr

                # IMPORTANT: write with a "score" column. Here we keep retr_score in the score slot.
                write_trec(out_run, q2new, run_name=f"{args.run_prefix}_pipeline", topk_out=args.topk_out)
                m = eval_run(args.eval_py, args.qrels, out_run, tmp_eval_dir)
                w.writerow({
                    "strategy": "pipeline",
                    "param": f"cls_th={th:.2f},within={args.within_sort},irr={args.irrelevant_order}",
                    "run_path": out_run,
                    **m
                })
                fsum.flush()

        else:
            lams = parse_list(args.lam_list)
            mus  = parse_list(args.mu_list)

            for lam in lams:
                for mu in mus:
                    out_run = os.path.join(args.out_dir, f"{args.run_prefix}_fusion_l{lam:.2f}_m{mu:.2f}_{args.norm}.trec")
                    q2new = {}

                    for qid, cands in q2.items():
                        # attach signals
                        for c in cands:
                            sig = cache.get(qid, {}).get(c.doc, {})
                            c.cls_prob = float(sig.get("cls_prob", 0.0))
                            c.score_prob = float(sig.get("score_prob", 0.0))

                        base = norm_scores(cands, args.norm)
                        finals = []
                        for c, b in zip(cands, base):
                            finals.append((b + lam*c.score_prob + mu*c.cls_prob, c))
                        finals.sort(key=lambda x: (x[0], -x[1].rank), reverse=True)

                        # write back: we still must put something in TREC score column.
                        # Put the final score to make the run self-contained.
                        new_list = []
                        for sc, c in finals:
                            nc = Cand(doc=c.doc, rank=c.rank, retr_score=float(sc), cls_prob=c.cls_prob, score_prob=c.score_prob)
                            new_list.append(nc)
                        q2new[qid] = new_list

                    write_trec(out_run, q2new, run_name=f"{args.run_prefix}_fusion", topk_out=args.topk_out)
                    m = eval_run(args.eval_py, args.qrels, out_run, tmp_eval_dir)
                    w.writerow({
                        "strategy": "fusion",
                        "param": f"lam={lam:.2f},mu={mu:.2f},norm={args.norm}",
                        "run_path": out_run,
                        **m
                    })
                    fsum.flush()

    print(f"[OK] sweep summary -> {summary_path}")

if __name__ == "__main__":
    main()
