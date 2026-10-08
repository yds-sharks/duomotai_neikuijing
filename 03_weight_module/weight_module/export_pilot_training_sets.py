#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pilot training datasets from reviewed chains.")
    parser.add_argument(
        "--config",
        default="/mnt/data_1/yds/多模态/权重模块/pipeline_config.yaml",
    )
    args = parser.parse_args()
    cfg = load_config(Path(args.config))

    reviewed_path = Path(cfg["stage3_outputs"]["reviewed_chain_pilot"])
    density_cls_path = Path(cfg["stage3_outputs"]["density_cls_pilot"])
    dep_cls_path = Path(cfg["stage3_outputs"]["dependency_cls_pilot"])
    rank_path = Path(cfg["stage3_outputs"]["density_rank_pilot"])
    report_path = Path(cfg["stage3_outputs"]["quality_report_pilot"])

    level_dist = Counter()
    dep_dist = Counter()
    rank_pairs = 0
    rows = 0

    with density_cls_path.open("w", encoding="utf-8") as fcls, dep_cls_path.open("w", encoding="utf-8") as fdep, rank_path.open("w", encoding="utf-8") as frank:
        for obj in iter_jsonl(reviewed_path):
            rows += 1
            gen = obj.get("generated_chain", {})
            candidate_id = obj.get("candidate_id")
            split = obj.get("split")
            anchor_class = obj.get("anchor_class")
            core = obj.get("semantic_core")
            core_hash = obj.get("semantic_core_hash")

            chain = gen.get("main_chain", [])
            for item in chain:
                level = item.get("level")
                dep = item.get("image_dependency")
                query_id = f"{candidate_id}_{level}"
                base = {
                    "query_id": query_id,
                    "candidate_id": candidate_id,
                    "split": split,
                    "anchor_class": anchor_class,
                    "semantic_core": core,
                    "semantic_core_hash": core_hash,
                    "query_text": item.get("query_text", ""),
                    "density_level": level,
                    "density_score": item.get("density_score"),
                }
                fcls.write(json.dumps(base, ensure_ascii=False) + "\n")
                fdep.write(json.dumps({**base, "image_dependency": dep}, ensure_ascii=False) + "\n")
                level_dist[level] += 1
                dep_dist[str(dep)] += 1

            # pairwise ranking on main chain
            for i in range(len(chain)):
                for j in range(i + 1, len(chain)):
                    a = chain[i]
                    b = chain[j]
                    frank.write(
                        json.dumps(
                            {
                                "candidate_id": candidate_id,
                                "split": split,
                                "semantic_core_hash": core_hash,
                                "query_id_a": f"{candidate_id}_{a.get('level')}",
                                "query_id_b": f"{candidate_id}_{b.get('level')}",
                                "label": -1 if i < j else 1,
                                "pair_type": "main_chain",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    rank_pairs += 1

    existing_report = {}
    if report_path.exists():
        existing_report = json.loads(report_path.read_text(encoding="utf-8"))
    existing_report["export_summary"] = {
        "reviewed_rows": rows,
        "density_cls_rows": sum(level_dist.values()),
        "dependency_cls_rows": sum(dep_dist.values()),
        "density_rank_rows": rank_pairs,
        "level_distribution": dict(level_dist),
        "dependency_distribution": dict(dep_dist),
    }
    report_path.write_text(json.dumps(existing_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done]", density_cls_path)
    print("[done]", dep_cls_path)
    print("[done]", rank_path)
    print(existing_report["export_summary"])


if __name__ == "__main__":
    main()

