#!/usr/bin/env python3
"""重建 round-0 obs_candidates:分模态(图 top-Ki + 文 top-Kt)的图文对混合。

修正两个老 bug:
  1) 老逻辑把图文倒一个池子按分数排 -> 图像通吃、obs 全是图像,策略无从挑文本;
  2) 文本命中丢了配对图 -> 现在 search_text 已回查补全 image_path,每条证据都是图文对。

输入:agent_context_v11_train3200.jsonl(含 3200 题元信息 + query 图 + original_query)
输出:同结构,但把 rounds[0].candidates 换成分模态图文对候选(供 Stage A 重跑)。
可分片(--start/--limit)+断点续(按 qid)。
"""
from __future__ import annotations

import argparse
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
                if line:
                    try:
                        done.add(json.loads(line).get("qid"))
                    except Exception:
                        pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="outputs/stage2_calibration/agent_context_v11_train3200.jsonl")
    ap.add_argument("--out", default="outputs/stage2_calibration/agent_context_v11_train3200_paired.jsonl")
    ap.add_argument("--config", default="")
    ap.add_argument("--db-path", default="", help="override Milvus Lite db path (per-shard copy)")
    ap.add_argument("--image-device", default="cuda:0")
    ap.add_argument("--text-device", default="cuda:0")
    ap.add_argument("--text-k", type=int, default=20)
    ap.add_argument("--image-k", type=int, default=20)
    ap.add_argument("--image-select-k", type=int, default=6)
    ap.add_argument("--text-select-k", type=int, default=6)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    cfg["retrieval"]["image_device"] = args.image_device
    cfg["retrieval"]["text_device"] = args.text_device
    if args.db_path:
        cfg["retrieval"]["milvus_db_path"] = args.db_path

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

    done = load_done_qids(args.out)
    print(f"[数据] 待处理 {len(rows)} 题(start={args.start} limit={args.limit}) 已完成 {len(done)}", flush=True)

    retriever = FirstStageRetriever(cfg)
    print(f"[检索器] 就绪 image={args.image_device} text={args.text_device} "
          f"图top-{args.image_select_k}+文top-{args.text_select_k}", flush=True)

    out_f = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    n = 0
    for row in rows:
        qid = row.get("qid")
        if qid in done:
            continue
        qimg = row.get("query_image_path", "")
        oq = row.get("original_query", "") or row.get("question", "")
        img_hits = retriever.search_image(qimg, k=args.image_k) if qimg else []
        txt_hits = retriever.search_text(oq, k=args.text_k) if oq else []
        cands = (select_top_evidence(img_hits, select_k=args.image_select_k)
                 + select_top_evidence(txt_hits, select_k=args.text_select_k))
        cands = [normalize_candidate(h) for h in cands]

        rounds = row.get("rounds") or [{}]
        r0 = rounds[0] if rounds else {}
        r0["round_idx"] = 0
        r0["query"] = oq
        r0["candidates"] = cands
        row["rounds"] = [r0]
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()
        n += 1
        if n % args.log_every == 0:
            dt = time.time() - t0
            print(f"[{n}/{len(rows)}] {dt / n:.2f}s/题 图{sum(1 for c in cands if c.get('origin') == 'image')}"
                  f"/文{sum(1 for c in cands if c.get('origin') == 'text')}", flush=True)

    out_f.close()
    retriever.close()
    print(f"[完成] 写出 {n} 题 -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
