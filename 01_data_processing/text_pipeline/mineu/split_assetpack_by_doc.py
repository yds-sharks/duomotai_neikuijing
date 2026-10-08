#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Split AssetPack JSONL into train/test by document.")
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in iter_jsonl(input_jsonl):
        grouped[row["doc_out_dir"]].append(row)

    doc_ids = sorted(grouped.keys())
    rng = random.Random(args.seed)
    rng.shuffle(doc_ids)

    total_docs = len(doc_ids)
    test_docs_n = max(1, round(total_docs * args.test_ratio))
    test_doc_set = set(doc_ids[:test_docs_n])

    train_rows: List[Dict] = []
    test_rows: List[Dict] = []
    train_docs: List[Dict] = []
    test_docs: List[Dict] = []

    for doc_out_dir in doc_ids:
        rows = grouped[doc_out_dir]
        split = "test" if doc_out_dir in test_doc_set else "train"
        target_rows = test_rows if split == "test" else train_rows
        target_docs = test_docs if split == "test" else train_docs

        for row in rows:
            row = dict(row)
            row["dataset_split"] = split
            target_rows.append(row)

        type_counter = Counter(row.get("type", "unknown") for row in rows)
        target_docs.append(
            {
                "doc_out_dir": doc_out_dir,
                "blocks": len(rows),
                "type_counts": dict(type_counter),
            }
        )

    train_jsonl = out_dir / "train.assetpack.jsonl"
    test_jsonl = out_dir / "test.assetpack.jsonl"
    train_docs_json = out_dir / "train.docs.json"
    test_docs_json = out_dir / "test.docs.json"
    manifest_json = out_dir / "split_manifest.json"

    write_jsonl(train_jsonl, train_rows)
    write_jsonl(test_jsonl, test_rows)
    train_docs_json.write_text(json.dumps(train_docs, ensure_ascii=False, indent=2), encoding="utf-8")
    test_docs_json.write_text(json.dumps(test_docs, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "input_jsonl": str(input_jsonl),
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "total_docs": total_docs,
        "train_docs": len(train_docs),
        "test_docs": len(test_docs),
        "train_blocks": len(train_rows),
        "test_blocks": len(test_rows),
        "train_jsonl": str(train_jsonl),
        "test_jsonl": str(test_jsonl),
        "train_docs_json": str(train_docs_json),
        "test_docs_json": str(test_docs_json),
    }
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
