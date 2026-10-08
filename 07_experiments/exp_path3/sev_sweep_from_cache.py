#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: sev_sweep_from_cache.py
Purpose:
  Sweep SEV thresholds using an EXISTING cache JSONL (no model inference).
  For each threshold, build a TREC run by stable partition:
    relevant(prob>=th) first, then irrelevant, preserving relative order.
  Then call eval_runs.py to compute metrics.

Works with multiple cache formats (robust parsing):
  - per-qid lines: {"qid":..., "results":[{pk, cls_prob, score_prob, ...}, ...]}
  - per-qid lines: {"qid":..., "cands":[...]} / {"docs":[...]}
  - per-pair lines: {"qid":..., "pk":..., "rank":..., "cls_prob":..., ...}
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


# ------------------------- utils -------------------------

def _as_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def _get_first(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _read_metrics_csv(metrics_csv: Path) -> Dict[str, float]:
    # eval_runs.py outputs CSV with header:
    # run_file,nDCG@10,MRR@10,Recall@10,Recall@50,Recall@100,num_qids
    with metrics_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty metrics csv: {metrics_csv}")
    r = rows[0]
    out = {}
    for k, v in r.items():
        if k == "run_file":
            continue
        out[k] = float(v)
    return out


@dataclass
class Cand:
    pk: str
    base_rank: int
    base_score: float
    signals: Dict[str, float]   # e.g. {"cls_prob":0.53, "score_prob":0.61}


def load_cache_grouped(cache_path: Path, topk_in: int, text_field: str = "text") -> Tuple[Dict[str, List[Cand]], List[str]]:
    """
    Return:
      grouped[qid] = list of Cand sorted by base_rank (stable)
      signals_found = list of signal keys present (subset of ["cls_prob","score_prob"] etc)
    """
    grouped: Dict[str, List[Cand]] = defaultdict(list)
    signals_found = set()

    def add_cand(qid: str, raw: Dict[str, Any], fallback_rank: int):
        pk = _get_first(raw, ["pk", "docid", "id", "pid"])
        if pk is None:
            return
        rank = _get_first(raw, ["rank", "orig_rank", "base_rank", "rnk"], default=fallback_rank)
        try:
            rank = int(rank)
        except Exception:
            rank = int(fallback_rank)

        base_score = _as_float(_get_first(raw, ["score", "base_score", "bm25", "dense", "hybrid_score"]), default=0.0) or 0.0

        sigs: Dict[str, float] = {}
        for key in ["cls_prob", "score_prob", "cls", "score"]:
            if key in raw and raw[key] is not None:
                sigs[key] = float(raw[key])
                signals_found.add(key)

        # 兼容你旧处理里叫 cls_prob / score_prob；这里做一个轻微归一
        if "cls" in sigs and "cls_prob" not in sigs:
            sigs["cls_prob"] = sigs["cls"]
            signals_found.add("cls_prob")
        if "score" in sigs and "score_prob" not in sigs:
            sigs["score_prob"] = sigs["score"]
            signals_found.add("score_prob")

        grouped[str(qid)].append(Cand(pk=str(pk), base_rank=rank, base_score=base_score, signals=sigs))

    with cache_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Case A: per-qid record with list
            qid = _get_first(obj, ["qid", "query_id", "q_id", "id"])
            lst = _get_first(obj, ["results", "cands", "docs", "candidates"], default=None)
            if qid is not None and isinstance(lst, list):
                # keep list order as fallback rank
                for j, r in enumerate(lst[:topk_in], start=1):
                    if isinstance(r, dict):
                        add_cand(str(qid), r, fallback_rank=j)
                continue

            # Case B: per-pair record
            # Expect keys like {"qid":..., "pk":..., "rank":..., "cls_prob":..., "score_prob":...}
            qid2 = _get_first(obj, ["qid", "query_id", "q_id"])
            pk2 = _get_first(obj, ["pk", "docid", "id", "pid"])
            if qid2 is not None and pk2 is not None:
                add_cand(str(qid2), obj, fallback_rank=len(grouped[str(qid2)]) + 1)
                continue

            # Unknown line format: ignore (but let user know if too many)
            # We keep silent to be robust.

    # sort per qid by base_rank and cut to topk_in
    for qid, cands in grouped.items():
        cands.sort(key=lambda x: x.base_rank)
        grouped[qid] = cands[:topk_in]

    # prioritize canonical names
    canon = []
    if "cls_prob" in signals_found:
        canon.append("cls_prob")
    if "score_prob" in signals_found:
        canon.append("score_prob")
    # include any other signals just in case
    for k in sorted(signals_found):
        if k not in canon:
            canon.append(k)

    return grouped, canon


def write_trec_run(out_path: Path, run_name: str, grouped: Dict[str, List[Cand]], signal: str, threshold: float, topk_out: int):
    """
    Stable partition based on cand.signals[signal] >= threshold.
    If signal missing -> treat as 0.0.
    """
    with out_path.open("w", encoding="utf-8") as fp:
        for qid, cands in grouped.items():
            rel, irr = [], []
            for c in cands:
                p = c.signals.get(signal, 0.0)
                if p >= threshold:
                    rel.append(c)
                else:
                    irr.append(c)
            merged = (rel + irr)[:topk_out]

            for rnk, c in enumerate(merged, start=1):
                # TREC: qid Q0 docid rank score runname
                # score here can be the signal prob; keep it to reflect SEV ordering
                sc = c.signals.get(signal, 0.0)
                fp.write(f"{qid}\tQ0\t{c.pk}\t{rnk}\t{sc:.8f}\t{run_name}\n")


def run_eval(eval_py: Path, qrels: Path, run_path: Path, metrics_out: Path, per_query_out: Path):
    cmd = [
        sys.executable, str(eval_py),
        "--qrels", str(qrels),
        "--runs", str(run_path),
        "--metrics_out", str(metrics_out),
        "--per_query_out", str(per_query_out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_in", required=True, help="Existing cache jsonl (no inference).")
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--eval_py", required=True, help="Path to eval_runs.py")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--signals", default="cls_prob,score_prob", help="Comma-separated signal keys to sweep.")
    ap.add_argument("--topk_in", type=int, default=100)
    ap.add_argument("--topk_out", type=int, default=100)

    ap.add_argument("--th_min", type=float, default=0.05)
    ap.add_argument("--th_max", type=float, default=0.95)
    ap.add_argument("--th_step", type=float, default=0.05)

    ap.add_argument("--run_prefix", default="sev_sweep")
    ap.add_argument("--keep_runs", action="store_true", help="Keep per-threshold trec files (default deletes).")

    args = ap.parse_args()

    cache_in = Path(args.cache_in)
    qrels = Path(args.qrels)
    eval_py = Path(args.eval_py)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    grouped, signals_found = load_cache_grouped(cache_in, topk_in=args.topk_in)
    if not grouped:
        raise RuntimeError(f"No records loaded from cache: {cache_in}")

    requested = [s.strip() for s in args.signals.split(",") if s.strip()]
    # only keep those present, but also allow if user insists (missing treated as 0)
    signals = requested

    # build thresholds
    ths = []
    t = args.th_min
    # avoid float drift
    while t <= args.th_max + 1e-9:
        ths.append(round(t, 6))
        t += args.th_step

    summary_rows = []
    best = {}  # signal -> (nDCG@10, th, metrics)
    total_qids = len(grouped)

    for signal in signals:
        for th in ths:
            run_name = f"{args.run_prefix}_{signal}_th{th:g}"
            run_path = out_dir / f"{run_name}.trec"
            metrics_csv = out_dir / f"{run_name}.metrics.csv"
            perq_json = out_dir / f"{run_name}.per_query.json"

            write_trec_run(run_path, run_name, grouped, signal=signal, threshold=th, topk_out=args.topk_out)
            # eval
            try:
                run_eval(eval_py, qrels, run_path, metrics_csv, perq_json)
            except subprocess.CalledProcessError as e:
                # print stderr for debugging
                sys.stderr.write(e.stderr.decode("utf-8", errors="ignore"))
                raise

            mets = _read_metrics_csv(metrics_csv)
            row = {
                "signal": signal,
                "threshold": th,
                "num_qids": total_qids,
                "nDCG@10": mets["nDCG@10"],
                "MRR@10": mets["MRR@10"],
                "Recall@10": mets["Recall@10"],
                "Recall@50": mets["Recall@50"],
                "Recall@100": mets["Recall@100"],
                "run_path": str(run_path),
            }
            summary_rows.append(row)

            cur = mets["nDCG@10"]
            if signal not in best or cur > best[signal][0]:
                best[signal] = (cur, th, mets)

            if not args.keep_runs:
                # keep only summary csv; remove heavy artifacts
                try:
                    run_path.unlink(missing_ok=True)
                    metrics_csv.unlink(missing_ok=True)
                    perq_json.unlink(missing_ok=True)
                except Exception:
                    pass

        b = best[signal]
        print(json.dumps({
            "signal": signal,
            "best_threshold": b[1],
            "best_nDCG@10": b[0],
            "best_MRR@10": b[2]["MRR@10"],
            "best_Recall@10": b[2]["Recall@10"],
            "best_Recall@50": b[2]["Recall@50"],
            "best_Recall@100": b[2]["Recall@100"],
        }, ensure_ascii=False))

    # write summary csv
    summary_csv = out_dir / "sev_threshold_sweep_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "signal","threshold","num_qids","nDCG@10","MRR@10","Recall@10","Recall@50","Recall@100","run_path"
        ])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"[OK] summary -> {summary_csv}")


if __name__ == "__main__":
    main()
