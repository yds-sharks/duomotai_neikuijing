#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import yaml


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stratified pilot anchors.")
    parser.add_argument(
        "--config",
        default="/mnt/data_1/yds/多模态/权重模块/pipeline_config.yaml",
    )
    parser.add_argument("--target-total", type=int, default=None, help="Override pilot target count.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    pcfg = cfg["pilot"]
    source = Path(pcfg["source_split_file"])
    out_file = Path(pcfg["output_pilot_file"])
    out_stats = Path(pcfg["output_pilot_stats"])
    out_file.parent.mkdir(parents=True, exist_ok=True)

    target_total = int(args.target_total or pcfg["target_total"])
    train_only = bool(pcfg.get("train_only", True))
    ratios = pcfg["class_ratio"]
    seed = int(pcfg.get("random_seed", 20260423))
    random.seed(seed)

    rows = read_jsonl(source)
    if train_only:
        rows = [r for r in rows if r.get("split") == "train"]

    buckets = {"A": [], "B": [], "C": []}
    for r in rows:
        c = r.get("anchor_class", "")
        if c in buckets:
            buckets[c].append(r)

    targets = {
        "A": int(round(target_total * float(ratios.get("A", 0.7)))),
        "B": int(round(target_total * float(ratios.get("B", 0.2)))),
        "C": int(round(target_total * float(ratios.get("C", 0.1)))),
    }
    # Correct rounding drift.
    diff = target_total - sum(targets.values())
    if diff != 0:
        targets["A"] += diff

    sampled = []
    for c in ["A", "B", "C"]:
        random.shuffle(buckets[c])
        sampled.extend(buckets[c][: min(targets[c], len(buckets[c]))])

    # Backfill if any class shortage.
    if len(sampled) < target_total:
        used_ids = {r.get("candidate_id") for r in sampled}
        remain = [r for r in rows if r.get("candidate_id") not in used_ids]
        random.shuffle(remain)
        sampled.extend(remain[: target_total - len(sampled)])

    sampled = sampled[:target_total]
    random.shuffle(sampled)

    # Keep generation fields only.
    keep_keys = {
        "candidate_id",
        "split",
        "anchor_class",
        "anchor_score",
        "anchor_text",
        "anchor_text_norm",
        "semantic_core",
        "semantic_core_hash",
        "slots",
        "counts",
        "flags",
        "score_components",
    }
    cleaned = [{k: v for k, v in r.items() if k in keep_keys} for r in sampled]
    write_jsonl(out_file, cleaned)

    dist = Counter(r.get("anchor_class", "UNK") for r in cleaned)
    split_dist = Counter(r.get("split", "UNK") for r in cleaned)
    stats = {
        "source_rows": len(rows),
        "target_total": target_total,
        "actual_total": len(cleaned),
        "class_distribution": dict(dist),
        "split_distribution": dict(split_dist),
        "seed": seed,
    }
    out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out_file}")
    print(f"[done] {out_stats}")
    print(stats)


if __name__ == "__main__":
    main()

