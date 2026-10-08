#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def core_to_split(core_hash: str) -> str:
    if not core_hash:
        return "train"
    v = int(core_hash[:8], 16) % 10
    if v < 8:
        return "train"
    if v == 8:
        return "val"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare anchor mainset and semantic-core split.")
    parser.add_argument(
        "--input",
        default="/mnt/data_1/yds/多模态/权重模块/outputs/anchor_pool.scored.v1_ready.jsonl",
    )
    parser.add_argument(
        "--output-mainset",
        default="/mnt/data_1/yds/多模态/权重模块/outputs/anchor_pool.mainset_14000.jsonl",
    )
    parser.add_argument(
        "--output-split",
        default="/mnt/data_1/yds/多模态/权重模块/outputs/anchor_pool.mainset_14000.split.jsonl",
    )
    parser.add_argument(
        "--output-stats",
        default="/mnt/data_1/yds/多模态/权重模块/outputs/anchor_pool.mainset_14000.stats.json",
    )
    parser.add_argument("--target", type=int, default=14000)
    args = parser.parse_args()

    inp = Path(args.input)
    out_main = Path(args.output_mainset)
    out_split = Path(args.output_split)
    out_stats = Path(args.output_stats)

    rows = []
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    rows.sort(
        key=lambda o: (
            float(o.get("anchor_score", 0.0)),
            -int(o.get("char_len", 0)),
            o.get("candidate_id", ""),
        ),
        reverse=True,
    )
    mainset = rows[: args.target]

    class_dist = Counter()
    split_dist = Counter()
    split_class = {"train": Counter(), "val": Counter(), "test": Counter()}
    unique_core = set()

    with out_main.open("w", encoding="utf-8") as fm, out_split.open("w", encoding="utf-8") as fs:
        for o in mainset:
            fm.write(json.dumps(o, ensure_ascii=False) + "\n")

            core_hash = o.get("semantic_core_hash", "")
            split = core_to_split(core_hash)
            o2 = dict(o)
            o2["split"] = split
            fs.write(json.dumps(o2, ensure_ascii=False) + "\n")

            cls = o.get("anchor_class", "UNK")
            class_dist[cls] += 1
            split_dist[split] += 1
            split_class[split][cls] += 1
            if core_hash:
                unique_core.add(core_hash)

    stats = {
        "input_rows": len(rows),
        "target_mainset": args.target,
        "mainset_rows": len(mainset),
        "unique_semantic_core_hash": len(unique_core),
        "class_distribution": dict(class_dist),
        "split_distribution": dict(split_dist),
        "split_class_distribution": {
            k: dict(v) for k, v in split_class.items()
        },
    }

    out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done]", out_main)
    print("[done]", out_split)
    print("[done]", out_stats)
    print(stats)


if __name__ == "__main__":
    main()

