#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: sample_assetpack_jsonl.py
Purpose:
  Randomly sample records from a large AssetPack JSONL with stratified sampling by modality/type.
  - Avoid loading the whole file into memory.
  - Ensure all modalities/types are represented (if present in data).
  - Optionally copy sampled images and export table HTML for quick inspection.

Sampling:
  Two-pass:
    Pass1: count records per type (text/image/table/equation/other + any extra types)
    Pass2: reservoir sampling per type using allocated k per type
"""

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any


DEFAULT_TYPES = ["text", "image", "table", "equation", "other"]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ln, json.loads(line)
            except Exception:
                # Skip malformed lines but keep going
                continue


def pass1_count_types(jsonl_path: Path, type_key: str = "type") -> Counter:
    c = Counter()
    for _, obj in iter_jsonl(jsonl_path):
        t = str(obj.get(type_key, "other") or "other").strip()
        c[t] += 1
    return c


def allocate_k(total_n: int, counts: Counter, prefer_types: List[str], min_each: int = 1) -> Dict[str, int]:
    """
    Allocate total_n across types.
    - Guarantee min_each for each prefer_type that exists in data (count>0), if possible.
    - Remaining budget allocated proportional to counts among prefer_types (and then others).
    """
    present_types = [t for t in prefer_types if counts.get(t, 0) > 0]
    if not present_types:
        # fall back: allocate everything to most common type(s)
        most_common = [t for t, _ in counts.most_common()]
        present_types = most_common[:1] if most_common else ["other"]

    k = {t: 0 for t in counts.keys()}

    # reserve minimum
    base_need = min_each * len(present_types)
    if base_need > total_n:
        # not enough budget to give everyone min_each; give 1 to as many as possible
        for t in present_types[:total_n]:
            k[t] = 1
        return k

    for t in present_types:
        k[t] = min_each

    remaining = total_n - base_need
    if remaining <= 0:
        return k

    # allocate remaining proportionally to counts among present_types
    denom = sum(counts[t] for t in present_types)
    if denom <= 0:
        # evenly
        for i in range(remaining):
            k[present_types[i % len(present_types)]] += 1
        return k

    # initial proportional assignment
    fractional: List[Tuple[float, str]] = []
    assigned = 0
    for t in present_types:
        add = remaining * (counts[t] / denom)
        add_int = int(add)
        k[t] += add_int
        assigned += add_int
        fractional.append((add - add_int, t))

    # distribute leftover by largest fractional parts
    leftover = remaining - assigned
    fractional.sort(reverse=True)
    for i in range(leftover):
        k[fractional[i % len(fractional)][1]] += 1

    return k


def reservoir_sample_by_type(
    jsonl_path: Path,
    k_by_type: Dict[str, int],
    seed: int,
    type_key: str = "type",
) -> Dict[str, List[Tuple[int, Dict[str, Any]]]]:
    rng = random.Random(seed)
    seen = defaultdict(int)
    reservoirs: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {t: [] for t in k_by_type.keys()}

    for ln, obj in iter_jsonl(jsonl_path):
        t = str(obj.get(type_key, "other") or "other").strip()
        if t not in k_by_type:
            # include unexpected types if appear; treat as "other"
            t = "other"
            if t not in k_by_type:
                continue

        k = int(k_by_type.get(t, 0))
        if k <= 0:
            continue

        seen[t] += 1
        r = reservoirs[t]

        if len(r) < k:
            r.append((ln, obj))
        else:
            j = rng.randint(1, seen[t])
            if j <= k:
                r[j - 1] = (ln, obj)

    return reservoirs


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", default="/mnt/data_1/yds/多模态/data/output/pdf/assetpack.jsonl")
    ap.add_argument("--out_dir", default="/mnt/data_1/yds/多模态/data/output/pdf/_sample_check")
    ap.add_argument("--num", type=int, default=60, help="Total number of samples to draw.")
    ap.add_argument("--seed", type=int, default=20251229)
    ap.add_argument("--min_each_type", type=int, default=3, help="Minimum samples per main type if present.")
    ap.add_argument("--types", nargs="*", default=DEFAULT_TYPES, help="Types to prioritize for stratified sampling.")
    ap.add_argument("--copy_images", action="store_true", default=True, help="Copy sampled image files for inspection.")
    ap.add_argument("--export_table_html", action="store_true", default=True, help="Export table_html as .html files.")
    ap.add_argument("--export_pretty_json", action="store_true", default=True, help="Export each sample as pretty .json.")
    args = ap.parse_args()

    input_jsonl = Path(args.input_jsonl).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"input_jsonl not found: {input_jsonl}")

    # Pass 1: counts
    counts = pass1_count_types(input_jsonl)
    (out_dir / "type_counts.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")

    # Allocate k
    k_by_type = allocate_k(
        total_n=args.num,
        counts=counts,
        prefer_types=args.types,
        min_each=args.min_each_type
    )

    # Ensure "other" exists in allocation if unexpected types exist
    if "other" not in k_by_type:
        k_by_type["other"] = 0

    (out_dir / "k_by_type.json").write_text(json.dumps(k_by_type, ensure_ascii=False, indent=2), encoding="utf-8")

    # Pass 2: reservoir sampling
    reservoirs = reservoir_sample_by_type(input_jsonl, k_by_type=k_by_type, seed=args.seed)

    # Flatten + shuffle (still deterministic by seed)
    rng = random.Random(args.seed)
    samples: List[Tuple[int, Dict[str, Any]]] = []
    for t, lst in reservoirs.items():
        samples.extend(lst)
    rng.shuffle(samples)

    # Write sampled jsonl
    sampled_jsonl = out_dir / f"sampled_{args.num}_seed{args.seed}.jsonl"
    with sampled_jsonl.open("w", encoding="utf-8") as fw:
        for ln, obj in samples:
            obj2 = dict(obj)
            obj2["_source_line"] = ln
            fw.write(json.dumps(obj2, ensure_ascii=False) + "\n")

    # Optional exports
    images_dir = out_dir / "images"
    tables_dir = out_dir / "tables_html"
    pretty_dir = out_dir / "pretty_json"

    missing_images = 0
    exported_tables = 0

    for idx, (ln, obj) in enumerate(samples):
        t = str(obj.get("type", "other") or "other").strip()
        # pretty json
        if args.export_pretty_json:
            pretty_dir.mkdir(parents=True, exist_ok=True)
            (pretty_dir / f"{idx:05d}_{t}_L{ln}.json").write_text(
                json.dumps(obj, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

        # copy image
        if args.copy_images and obj.get("img_abs_path"):
            src = Path(str(obj["img_abs_path"]))
            if src.exists() and src.is_file():
                dst = images_dir / src.name
                safe_copy(src, dst)
            else:
                missing_images += 1

        # export table html
        if args.export_table_html and t == "table" and obj.get("table_html"):
            tables_dir.mkdir(parents=True, exist_ok=True)
            html_path = tables_dir / f"{idx:05d}_table_L{ln}.html"
            html_path.write_text(str(obj["table_html"]), encoding="utf-8")
            exported_tables += 1

    # Summary
    sampled_type_counts = Counter()
    for _, obj in samples:
        sampled_type_counts[str(obj.get("type", "other") or "other").strip()] += 1

    summary = {
        "input_jsonl": str(input_jsonl),
        "out_dir": str(out_dir),
        "num_requested": args.num,
        "seed": args.seed,
        "min_each_type": args.min_each_type,
        "types_prioritized": args.types,
        "type_counts_all": dict(counts),
        "k_by_type_allocated": dict(k_by_type),
        "type_counts_sampled": dict(sampled_type_counts),
        "sampled_jsonl": str(sampled_jsonl),
        "copied_images_dir": str(images_dir) if args.copy_images else None,
        "tables_html_dir": str(tables_dir) if args.export_table_html else None,
        "pretty_json_dir": str(pretty_dir) if args.export_pretty_json else None,
        "missing_images_in_sample": missing_images,
        "exported_tables_in_sample": exported_tables,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("========== SAMPLE DONE ==========")
    print("Sample JSONL :", sampled_jsonl)
    print("Sample types :", dict(sampled_type_counts))
    if args.copy_images:
        print("Images dir   :", images_dir, f"(missing in sample: {missing_images})")
    if args.export_table_html:
        print("Tables dir   :", tables_dir, f"(exported: {exported_tables})")
    print("Summary      :", out_dir / "summary.json")


if __name__ == "__main__":
    main()
