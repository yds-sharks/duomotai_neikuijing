#!/usr/bin/env python3
"""Stage B -- live retrieval for policy rewrite queries (qwen3vl-rerank env).

Reads Stage A output (policy_actions_*.jsonl). For each question, runs the SAME
first-stage retrieval that built the dataset (BGE-M3 text + Qwen3-VL image + Milvus)
on every unique REWRITE query, and attaches the recalled top-k passages to each
rewrite action. The query image is fixed per question -> image hits computed once.

ACCEPT actions need no retrieval: their evidence is obs_candidates[keep] (already in
Stage A output). NO generator scoring here (that is the reusable reward step).

Output: the Stage A records, augmented with per-rewrite ``recalled_passages``.

Run (retrieval env; image-enc + text-enc on the given cards):
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /mnt/data_1/yds/venvs/qwen3vl-rerank/bin/python train/gen_retrieval_for_rewrites.py \
    --input train/policy_actions_v1.jsonl --out train/rollouts_recalled_v1.jsonl \
    --image-device cuda:1 --text-device cuda:2
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))

from retrieval_adapter import FirstStageRetriever, load_config  # noqa: E402
from evidence_selection import select_top_evidence  # noqa: E402
from trajectory_runtime import normalize_candidate  # noqa: E402


def load_done_qids(out_path: str) -> set:
    done: set = set()
    p = Path(out_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line).get("qid"))
                except Exception:
                    pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Stage A output (policy_actions_*.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--db-path", default="", help="override Milvus Lite db path (per-shard copy to avoid contention)")
    ap.add_argument("--image-device", default="cuda:1")
    ap.add_argument("--text-device", default="cuda:2")
    ap.add_argument("--text-k", type=int, default=20)
    ap.add_argument("--image-k", type=int, default=20)
    ap.add_argument("--select-k", type=int, default=12, help="(deprecated: merged sort) kept for compat")
    ap.add_argument("--image-select-k", type=int, default=6, help="top-K image passages (fixed across rewrites, background)")
    ap.add_argument("--text-select-k", type=int, default=6, help="top-K text passages (rewrite-driven -> provides the gradient)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    cfg = copy.deepcopy(load_config(args.config) if args.config else load_config())
    cfg.setdefault("retrieval", {})
    cfg["retrieval"]["text_device"] = args.text_device
    cfg["retrieval"]["image_device"] = args.image_device
    if args.db_path:
        cfg["retrieval"]["milvus_db_path"] = args.db_path
    cfg["retrieval"]["first_stage_text_k"] = args.text_k
    cfg["retrieval"]["first_stage_image_k"] = args.image_k
    retriever = FirstStageRetriever(cfg)
    print(f"[检索器] 就绪:image={args.image_device} text={args.text_device} "
          f"text_k={args.text_k} image_k={args.image_k} select_k={args.select_k}", flush=True)

    rows: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.start:
        rows = rows[args.start:]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[数据] 待检索 {len(rows)} 题(start={args.start} limit={args.limit})", flush=True)

    done = load_done_qids(args.out)
    print(f"[断点] 已完成 {len(done)} 题,将跳过", flush=True)

    out_f = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    n_done = 0
    tot_rw = 0
    for qi, o in enumerate(rows):
        qid = o.get("qid", "")
        if qid in done:
            continue
        image_path = o.get("query_image_path", "")
        img_hits = retriever.search_image(image_path, k=args.image_k) if image_path else []

        cache: Dict[str, List[Dict[str, Any]]] = {}
        n_rw = 0
        for r in o.get("group", []):
            if r.get("action") == "REWRITE" and (r.get("rewrite_query") or "").strip():
                q = r["rewrite_query"].strip()
                if q not in cache:
                    txt = retriever.search_text(q, k=args.text_k)
                    # 分模态各取 top-K:图像(固定/背景) + 文本(改写驱动/梯度来源)
                    top = (select_top_evidence(img_hits, select_k=args.image_select_k)
                           + select_top_evidence(txt, select_k=args.text_select_k))
                    cache[q] = [normalize_candidate(h) for h in top]
                    n_rw += 1
                r["recalled_passages"] = cache[q]
        tot_rw += n_rw

        out_f.write(json.dumps(o, ensure_ascii=False) + "\n")
        out_f.flush()
        n_done += 1
        if args.log_every and qi % args.log_every == 0:
            dt = time.time() - t0
            print(f"[题{qi} {qid}] 去重改写检索={n_rw} 累计检索={tot_rw} "
                  f"| {dt / max(n_done, 1):.2f}s/题", flush=True)

    out_f.close()
    print(f"[完成] 写出 {n_done} 题 -> {args.out}(累计改写检索 {tot_rw} 次)", flush=True)


if __name__ == "__main__":
    main()
