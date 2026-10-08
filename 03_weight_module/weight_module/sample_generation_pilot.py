#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pilot generation input by class quota.")
    parser.add_argument(
        "--config",
        default="/mnt/data_1/yds/多模态/权重模块/pipeline_config.yaml",
    )
    parser.add_argument(
        "--input",
        default="/mnt/data_1/yds/多模态/权重模块/outputs/anchor_pool.mainset_14000.split.jsonl",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    pilot_cfg = cfg["pilot"]
    out_path = Path(cfg["stage3_outputs"]["pilot_anchor_input"])
    stats_path = Path(cfg["stage3_outputs"]["pilot_anchor_stats"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.input))
    if pilot_cfg.get("train_split_only", True):
        rows = [r for r in rows if r.get("split") == "train"]

    by_class = defaultdict(list)
    for row in rows:
        by_class[row.get("anchor_class", "UNK")].append(row)

    random.seed(int(pilot_cfg["seed"]))
    sampled = []
    quota = pilot_cfg["class_quota"]
    for cls in ["A", "B", "C"]:
        pool = by_class.get(cls, [])
        need = int(quota.get(cls, 0))
        if len(pool) < need:
            raise RuntimeError(f"class {cls} not enough rows: have={len(pool)} need={need}")
        sampled.extend(random.sample(pool, need))

    # Stable order for downstream reproducibility.
    sampled.sort(key=lambda o: (o.get("anchor_class", ""), -float(o.get("anchor_score", 0.0)), o.get("candidate_id", "")))

    keep_keys = [
        "candidate_id",
        "split",
        "anchor_class",
        "anchor_score",
        "anchor_text",
        "anchor_text_norm",
        "slots",
        "counts",
        "flags",
        "semantic_core",
        "semantic_core_hash",
    ]
    with out_path.open("w", encoding="utf-8") as f:
        for row in sampled:
            out = {k: row.get(k) for k in keep_keys}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    stats = {
        "source_rows": len(rows),
        "pilot_rows": len(sampled),
        "class_distribution": dict(Counter(r["anchor_class"] for r in sampled)),
        "split_distribution": dict(Counter(r["split"] for r in sampled)),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done]", out_path)
    print("[done]", stats_path)
    print(stats)


if __name__ == "__main__":
    main()

