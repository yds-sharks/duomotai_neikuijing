#!/usr/bin/env python3
"""Export real EndoBench questions (HF cache) into harness queries jsonl.

Fields emitted per line:
  qid, question(EN原文), options{A..}, answer, answer_text(gt), query_image_path,
  original_query, retrieval_hint_zh(中文检索提示, 来自翻译缓存), scene/task/category.

    /mnt/data_1/yds/venvs/qwen35-train/bin/python make_queries.py --limit 10 --out runs/queries_smoke10.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent / "05_agentic_rag" / "agentic" / "eval"
TRANSLATE_CACHE = EVAL_DIR / "endobench_translated_queries.jsonl"
IMG_DIR = EVAL_DIR / "endobench_images"
OPTION_LETTERS = "ABCDEF"


def is_null_option(v) -> bool:
    s = str(v).strip().lower()
    return v is None or s in ("", "none", "null", "nan", "n/a", "-")


def load_translate_map() -> dict:
    m = {}
    if TRANSLATE_CACHE.exists():
        with open(TRANSLATE_CACHE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    m[int(d["index"])] = str(d.get("query_zh") or "")
                except (ValueError, KeyError):
                    continue
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="Saint-lsy/EndoBench")
    ap.add_argument("--split", default="test")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", default=str(HERE / "runs" / "queries_smoke10.jsonl"))
    args = ap.parse_args()

    from datasets import load_dataset  # venv with datasets + HF cache

    ds = load_dataset(args.benchmark)[args.split]
    tmap = load_translate_map()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for i in range(args.offset, min(len(ds), args.offset + args.limit)):
            ex = ds[i]
            idx = int(ex.get("index", i))
            options = {
                L: str(ex[L]).strip()
                for L in OPTION_LETTERS
                if ex.get(L) is not None and not is_null_option(ex.get(L))
            }
            img = IMG_DIR / f"eb_{idx}.jpg"
            row = {
                "qid": f"eb_{idx}",
                "question": str(ex.get("question") or "").strip(),
                "options": options,
                "answer": str(ex.get("answer") or "").strip(),
                "answer_text": str(ex.get("gt") or "").strip(),
                "query_image_path": str(img) if img.exists() else "",
                "original_query": str(ex.get("question") or "").strip(),
                "retrieval_hint_zh": tmap.get(idx, ""),
                "gold_source": {"doc_id": "", "page_idx": None},
                "scene": str(ex.get("scene") or ""),
                "task": str(ex.get("task") or ""),
                "category": str(ex.get("category") or ""),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"wrote {n_written} questions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
