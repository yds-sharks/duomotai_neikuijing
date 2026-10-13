#!/usr/bin/env python3
"""P2 放量前：把 qa_stage1_prepared/source_pool.jsonl 转成出题脚本统一格式。

输出行 = train.jsonl 风格：顶层 query_image_path + source{...}（含排除键 sample_id/doc_id
与 oracle_context 用的 pdf_context_text）。排除 P2 200 题小样已用源样本（小样 57 道合格
题将直接并入最终题库，避免同源重复出题）。

用法：
  /mnt/data_1/yds/venvs/qwen35-train/bin/python p2_build_source_pool.py \
      --pool /mnt/data_1/yds/多模态/rerank_image_and_text/agentic/outputs/qa_stage1_prepared/source_pool.jsonl \
      --exclude-sample-ids-from ./p2_mcq_200/p2_mcq_samples.jsonl \
      --out ./p2_source_pool.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--exclude-sample-ids-from", default="")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="./p2_source_pool.jsonl")
    args = ap.parse_args()

    exclude: set = set()
    if args.exclude_sample_ids_from and Path(args.exclude_sample_ids_from).exists():
        for line in open(args.exclude_sample_ids_from, encoding="utf-8"):
            try:
                r = json.loads(line)
                if "src_qid" in r:  # 小样记录
                    # 小样行没存 sample_id，用 source 行的 query_image_path 对不上——直接跳过将由 shuffle 差异覆盖
                    pass
            except (ValueError, KeyError):
                continue
    # 小样用过的源：从 train.jsonl 顺序抽取（seed=0 shuffle 前池顺序），直接复算
    # ——p2_generate_mcq.py 的抽样逻辑：load_questions(pool)+Random(0) 洗牌取前 200。
    # 这里复算该洗牌，取前 200 个 sample_id。

    rows: List[Dict[str, Any]] = [json.loads(l) for l in open(args.pool, encoding="utf-8")]
    # 复算小样已用（与小样脚本同 seed 同序）
    train = [json.loads(l) for l in open(
        "/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/outputs/mcq_image_v2_4000/train.jsonl",
        encoding="utf-8")]
    rng0 = random.Random(0)
    idx0 = list(range(len(train)))
    rng0.shuffle(idx0)
    used = {train[i]["source"].get("sample_id") for i in idx0[:200]}

    out_rows = []
    skipped = 0
    for r in rows:
        sid = str(r.get("sample_id") or "")
        img = str(r.get("query_image_path") or "")
        ctx = str(r.get("pdf_context_text") or "")
        if sid in used or not img or len(ctx) < 80:
            skipped += 1
            continue
        out_rows.append({
            "qid": f"p2src_{sid}",
            "query_image_path": img,
            "source": {
                "sample_id": sid,
                "group_id": str(r.get("group_id") or ""),
                "doc_id": str(r.get("doc_id") or ""),
                "doc_name": str(r.get("doc_name") or ""),
                "source_path": str(r.get("source_path") or ""),
                "origin_pdf_path": str(r.get("origin_pdf") or ""),
                "page_idx": r.get("page_idx"),
                "pdf_context_text": ctx,
                "image_path": img,
                "image_id": str(r.get("image_id") or ""),
                "level1": str(r.get("primary_knowledge_type") or ""),
                "level2": "",
                "text_role": str(r.get("text_role") or ""),
                "evidence_text": str(r.get("caption_or_pair_text") or "")[:1200],
            },
        })

    rng = random.Random(args.seed)
    rng.shuffle(out_rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"源池转换完成: {len(out_rows)} 可用源样本（跳过 {skipped}：小样已用/无图/上下文过短） -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
