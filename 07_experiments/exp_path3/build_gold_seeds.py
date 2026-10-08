#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: build_gold_seeds.py
Purpose:
  From each query's candidate window, select 1–K gold seed passages (P1-only by default).
Inputs:
  --input: JSON/JSONL. Each item contains:
           question(str), p_text(list[{text,support_level,rank_score,...}]), n_text(...)
  --head:  take first N items for trial
Outputs:
  --output: seeds.jsonl, each line:
    {"qid":"0","query":"...","seeds":[{"text":..., "level":"P1","rank_score":0.95,"src":"p_text","idx":0}, ...]}
Logs:
  prints summary to stdout
"""
import argparse, json, os
from typing import Any, Dict, List, Iterable

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--head", type=int, default=0, help="take first N queries (0 means all)")
    ap.add_argument("--max_seeds", type=int, default=2)
    ap.add_argument("--level", default="P1", choices=["P1","P2","P3"], help="seed support_level to select")
    ap.add_argument("--qid_field", default="", help="if empty, use running index as qid")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    total = 0
    wrote = 0
    empty_seed = 0

    with open(args.output, "w", encoding="utf-8") as w:
        for i, item in enumerate(iter_items(args.input)):
            if args.head and i >= args.head:
                break
            total += 1

            qid = str(item.get(args.qid_field)) if args.qid_field else str(i)
            query = item.get("question", "") or item.get("query", "")
            p_list: List[Dict[str, Any]] = item.get("p_text", []) or []

            cand = []
            for j, p in enumerate(p_list):
                if p.get("support_level") != args.level:
                    continue
                cand.append({
                    "text": p.get("text",""),
                    "level": args.level,
                    "rank_score": float(p.get("rank_score", 0.0) or 0.0),
                    "src": "p_text",
                    "idx": j,
                })

            cand.sort(key=lambda x: -x["rank_score"])
            seeds = cand[:args.max_seeds]

            if not seeds:
                empty_seed += 1

            out = {"qid": qid, "query": query, "seeds": seeds}
            w.write(json.dumps(out, ensure_ascii=False) + "\n")
            wrote += 1

    stats = {
        "num_queries_read": total,
        "num_lines_written": wrote,
        "level": args.level,
        "max_seeds": args.max_seeds,
        "empty_seed_queries": empty_seed,
        "empty_seed_ratio": empty_seed / max(1, total),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
