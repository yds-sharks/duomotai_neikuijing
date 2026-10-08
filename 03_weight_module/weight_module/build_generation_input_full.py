#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path


SOURCE = Path("/mnt/data_1/yds/多模态/权重模块/outputs/anchor_pool.mainset_14000.split.jsonl")
OUT = Path("/mnt/data_1/yds/多模态/权重模块/outputs/generation_input.mainset_14000.jsonl")
STATS = Path("/mnt/data_1/yds/多模态/权重模块/outputs/generation_input.mainset_14000.stats.json")


KEEP_KEYS = [
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


def main() -> None:
    total = 0
    class_counter = Counter()
    split_counter = Counter()
    with SOURCE.open("r", encoding="utf-8") as fin, OUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out = {key: row.get(key) for key in KEEP_KEYS}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            total += 1
            class_counter[row.get("anchor_class", "UNK")] += 1
            split_counter[row.get("split", "UNK")] += 1

    stats = {
        "source": str(SOURCE),
        "output": str(OUT),
        "rows": total,
        "class_distribution": dict(class_counter),
        "split_distribution": dict(split_counter),
    }
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done]", OUT)
    print("[done]", STATS)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
