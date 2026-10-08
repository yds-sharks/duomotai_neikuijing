#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: run_retrieval_baselines.py (FIXED v3)
Purpose:
  Run dense/sparse/hybrid baselines and export TREC run files.

Fixes:
  - Correct batching: retr.search_batch(q_batch) not full queries
  - Fail-fast only on real invariants (batch alignment, pk/score presence)
  - Allow short results (<topk) but log statistics (short/empty counts)
  - Atomic write tmp -> rename
New:
  - Support hybrid weight override via --w_dense/--w_sparse (passed to weights_override)
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, Iterable, List, Set, Optional, Tuple
import yaml


def iter_items(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt:
        return
    if path.endswith(".jsonl"):
        for line in txt.splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)
    else:
        obj = json.loads(txt)
        if isinstance(obj, list):
            for it in obj:
                yield it
        else:
            yield obj


def load_qids(path: str) -> Set[str]:
    if not path:
        return set()
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                s.add(t)
    return s


def chunks(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield i, lst[i:i + n]


def write_run_stream(fp, qid: str, results: List[Dict[str, Any]], run_name: str):
    # results: [{"pk":..., "score":...}, ...]
    for rnk, hit in enumerate(results, start=1):
        fp.write(f"{qid}\tQ0\t{str(hit['pk'])}\t{rnk}\t{float(hit['score'])}\t{run_name}\n")


def _parse_weight_override(args) -> Optional[Tuple[float, float]]:
    # If user provides either weight, require both.
    if args.w_dense is None and args.w_sparse is None:
        return None
    if args.w_dense is None or args.w_sparse is None:
        raise ValueError("Please provide BOTH --w_dense and --w_sparse (or provide neither).")
    w_d = float(args.w_dense)
    w_s = float(args.w_sparse)
    if w_d < 0 or w_s < 0:
        raise ValueError("Weights must be non-negative.")
    s = w_d + w_s
    if s <= 0:
        raise ValueError("Sum of weights must be > 0.")
    # Normalize to sum=1 for stability
    w_d /= s
    w_s /= s
    return (w_d, w_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--qid_field", default="")
    ap.add_argument("--qids_file", default="")
    ap.add_argument("--disable_lang_adapt", action="store_true")
    ap.add_argument("--k_dense", type=int, default=0)
    ap.add_argument("--k_sparse", type=int, default=0)
    ap.add_argument("--embed_bs", type=int, default=64)
    ap.add_argument("--modes", default="dense,sparse,hybrid")
    ap.add_argument("--log_every", type=int, default=50)

    # NEW: hybrid weights override
    ap.add_argument("--w_dense", type=float, default=None, help="Override fusion weight for dense (hybrid only).")
    ap.add_argument("--w_sparse", type=float, default=None, help="Override fusion weight for sparse (hybrid only).")

    # true fail-fast invariants
    ap.add_argument("--fail_fast_batch_align", action="store_true", default=True)
    ap.add_argument("--fail_fast_missing_fields", action="store_true", default=True)

    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from insert.bm25_bge_vectorstore_v2.retrieval_service import HybridRetriever
    retr = HybridRetriever(cfg)

    os.makedirs(args.out_dir, exist_ok=True)

    eligible = load_qids(args.qids_file) if args.qids_file else None

    qids: List[str] = []
    queries: List[str] = []
    for i, item in enumerate(iter_items(args.input)):
        if args.head and i >= args.head:
            break
        qid = str(item.get(args.qid_field)) if args.qid_field else str(i)
        if eligible is not None and qid not in eligible:
            continue
        q = item.get("question", "") or item.get("query", "")
        qids.append(qid)
        queries.append(q)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    run_files = {
        "dense":  ("dense.trec",  "dense_bge_m3"),
        "sparse": ("sparse.trec", "sparse_bm25lex"),
        "hybrid": ("hybrid.trec", "hybrid_default"),
    }
    for m in modes:
        if m not in run_files:
            raise ValueError(f"Unknown mode: {m}. Allowed: {list(run_files.keys())}")

    k_dense = int(args.k_dense or args.topk)
    k_sparse = int(args.k_sparse or args.topk)

    weights_override = _parse_weight_override(args)

    # If overriding weights, strongly recommend running hybrid only (but not required).
    if weights_override is not None:
        w_d, w_s = weights_override
        print(json.dumps({
            "weights_override": {"w_dense": w_d, "w_sparse": w_s},
            "note": "weights_override will be passed to HybridRetriever.search_batch(weights_override=...). "
                    "For clean ablation, use --modes hybrid and set --disable_lang_adapt."
        }, ensure_ascii=False))

    for mode in modes:
        fname, default_run_name = run_files[mode]

        # Dynamic run tag for hybrid weight sweep
        if mode == "hybrid" and weights_override is not None:
            w_d, w_s = weights_override
            run_name = f"hybrid_w{w_d:.2f}_s{w_s:.2f}"
        else:
            run_name = default_run_name

        out_path = os.path.join(args.out_dir, fname)
        tmp_path = out_path + ".tmp"

        total_written = 0
        num_batches = 0
        short_qids = 0
        empty_qids = 0
        min_len = 10**9
        max_len = 0

        with open(tmp_path, "w", encoding="utf-8") as fp:
            for start, q_batch in chunks(queries, args.batch_size):
                qid_batch = qids[start:start + len(q_batch)]

                outs = retr.search_batch(
                    q_batch,
                    topk=args.topk,
                    mode=mode,
                    weights_override=(weights_override if mode == "hybrid" else None),
                    disable_lang_adapt=args.disable_lang_adapt,
                    k_dense=k_dense,
                    k_sparse=k_sparse,
                    embed_batch_size=args.embed_bs,
                )

                if args.fail_fast_batch_align and len(outs) != len(q_batch):
                    raise RuntimeError(
                        f"[FAIL-FAST] mode={mode} start={start}: outs={len(outs)} != batch={len(q_batch)}"
                    )

                for qid, out in zip(qid_batch, outs):
                    rs = out.get("results", []) or []

                    # stats (do not crash)
                    if len(rs) == 0:
                        empty_qids += 1
                    if len(rs) < args.topk:
                        short_qids += 1
                    min_len = min(min_len, len(rs))
                    max_len = max(max_len, len(rs))

                    if args.fail_fast_missing_fields:
                        for h in rs:
                            if "pk" not in h or h["pk"] is None:
                                raise RuntimeError(f"[FAIL-FAST] mode={mode} qid={qid}: missing pk")
                            if "score" not in h:
                                raise RuntimeError(f"[FAIL-FAST] mode={mode} qid={qid}: missing score")

                    write_run_stream(fp, qid, rs, run_name)
                    total_written += len(rs)

                num_batches += 1
                if args.log_every > 0 and (num_batches % args.log_every == 0):
                    print(json.dumps({
                        "mode": mode,
                        "batch": num_batches,
                        "start": start,
                        "batch_size": len(q_batch),
                        "written_lines_so_far": total_written,
                        "short_qids_so_far": short_qids,
                        "empty_qids_so_far": empty_qids,
                        "min_len_so_far": (min_len if min_len < 10**9 else None),
                        "max_len_so_far": max_len,
                        "run_name": run_name
                    }, ensure_ascii=False))

        os.replace(tmp_path, out_path)

        print(json.dumps({
            "mode": mode,
            "num_queries_run": len(qids),
            "topk": args.topk,
            "batch_size": args.batch_size,
            "k_dense": k_dense,
            "k_sparse": k_sparse,
            "embed_bs": args.embed_bs,
            "weights_override": ({"w_dense": weights_override[0], "w_sparse": weights_override[1]}
                                 if (mode == "hybrid" and weights_override is not None) else None),
            "run_name": run_name,
            "out_path": out_path,
            "total_written_lines": total_written,
            "short_qids": short_qids,
            "empty_qids": empty_qids,
            "min_results_len": (min_len if min_len < 10**9 else 0),
            "max_results_len": max_len,
        }, ensure_ascii=False, indent=2))

    print(json.dumps({
        "num_queries_run": len(qids),
        "modes": modes,
        "out_dir": args.out_dir,
        "filtered_by_qids_file": bool(args.qids_file),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
