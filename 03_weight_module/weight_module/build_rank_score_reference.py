#!/usr/bin/env python3
"""Build a reference distribution for pairwise-ranker raw scores."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from semantic_density_service import SemanticDensityService, load_yaml


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def percentile(sorted_values, pct):
    if not sorted_values:
        return None
    idx = int(round((len(sorted_values) - 1) * pct))
    return float(sorted_values[idx])


def main() -> None:
    root_dir = Path("/mnt/data_1/yds/多模态/权重模块")
    rank_cfg = load_yaml(root_dir / "checkpoints/density_rank_v10_mengzi/config.resolved.yaml")
    density_cfg = load_yaml(root_dir / "checkpoints/density_cls_v10_mengzi/config.resolved.yaml")
    source_path = Path(density_cfg["data"]["path"])
    output_path = root_dir / "checkpoints/density_rank_v10_mengzi/rank_score_reference.json"

    service = SemanticDensityService(root_dir=root_dir, device="auto", rank_reference_path="/tmp/nonexistent_rank_ref.json")

    rows = list(iter_jsonl(source_path))
    query_texts = [str(row["query_text"]).strip() for row in rows]
    scores = service.rank_model.score_queries(query_texts, batch_size=128)

    level_buckets = defaultdict(list)
    for row, score in zip(rows, scores):
        level_buckets[str(row["density_level"])].append(float(score))

    sorted_scores = sorted(float(v) for v in scores)
    level_stats = {}
    for level_name, values in sorted(level_buckets.items()):
        values = sorted(values)
        level_stats[level_name] = {
            "count": len(values),
            "mean": round(sum(values) / len(values), 6),
            "p10": round(percentile(values, 0.10), 6),
            "p50": round(percentile(values, 0.50), 6),
            "p90": round(percentile(values, 0.90), 6),
        }

    payload = {
        "source_path": str(source_path),
        "n_queries": len(sorted_scores),
        "sorted_scores": [round(v, 6) for v in sorted_scores],
        "overall": {
            "min": round(sorted_scores[0], 6),
            "p10": round(percentile(sorted_scores, 0.10), 6),
            "p50": round(percentile(sorted_scores, 0.50), 6),
            "p90": round(percentile(sorted_scores, 0.90), 6),
            "max": round(sorted_scores[-1], 6),
        },
        "level_stats": level_stats,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "n_queries": len(sorted_scores), "level_stats": level_stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
