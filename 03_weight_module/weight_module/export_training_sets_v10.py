#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


SOURCE = Path("/mnt/data_1/yds/多模态/权重模块/outputs/generated_chain.mainset_14000.v10_balanced_forms.jsonl")
OUT_DIR = Path("/mnt/data_1/yds/多模态/权重模块/outputs")

DENSITY_CLS = OUT_DIR / "density_cls.mainset_14000.v10.jsonl"
DEPENDENCY_CLS = OUT_DIR / "dependency_cls.mainset_14000.v10.jsonl"
DENSITY_RANK = OUT_DIR / "density_rank.mainset_14000.v10.jsonl"
REPORT = OUT_DIR / "training_sets.mainset_14000.v10.report.json"


LEVEL_TO_INT = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_sample_base(chain_obj: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    candidate_id = chain_obj.get("candidate_id")
    level = item.get("level")
    return {
        "query_id": f"{candidate_id}_{level}",
        "candidate_id": candidate_id,
        "split": chain_obj.get("split"),
        "anchor_class": chain_obj.get("anchor_class"),
        "semantic_core": chain_obj.get("semantic_core"),
        "semantic_core_hash": chain_obj.get("semantic_core_hash"),
        "anchor_text": chain_obj.get("anchor_text"),
        "query_text": item.get("query_text", ""),
        "density_level": level,
        "density_label": LEVEL_TO_INT.get(level),
        "density_score": item.get("density_score"),
        "rule_density_raw": item.get("rule_density_raw"),
        "image_dependency": item.get("image_dependency"),
        "style_tag": item.get("style_tag"),
        "kept_slots": item.get("kept_slots", []),
        "dropped_slots": item.get("dropped_slots", []),
        "source_type": "main_chain",
        "llm_model": chain_obj.get("llm_model"),
    }


def main() -> None:
    density_rows: List[Dict[str, Any]] = []
    dependency_rows: List[Dict[str, Any]] = []
    rank_rows: List[Dict[str, Any]] = []

    chain_count = 0
    level_dist = Counter()
    dep_dist = Counter()
    split_dist = Counter()
    class_dist = Counter()

    for obj in iter_jsonl(SOURCE):
        chain_count += 1
        candidate_id = obj.get("candidate_id")
        split = obj.get("split")
        core_hash = obj.get("semantic_core_hash")
        chain = obj.get("main_chain", [])
        split_dist[split] += 1
        class_dist[obj.get("anchor_class")] += 1

        samples = []
        for item in chain:
            base = build_sample_base(obj, item)
            density_rows.append(base)
            dependency_rows.append(base)
            samples.append(base)
            level_dist[base["density_level"]] += 1
            dep_dist[str(base["image_dependency"])] += 1

        # One anchor yields 10 ordered ranking pairs from L0<L1<...<L4.
        samples.sort(key=lambda x: x["density_label"])
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                low = samples[i]
                high = samples[j]
                rank_rows.append(
                    {
                        "pair_id": f"{candidate_id}_{low['density_level']}_vs_{high['density_level']}",
                        "candidate_id": candidate_id,
                        "split": split,
                        "semantic_core_hash": core_hash,
                        "query_id_low": low["query_id"],
                        "query_text_low": low["query_text"],
                        "density_level_low": low["density_level"],
                        "density_label_low": low["density_label"],
                        "density_score_low": low["density_score"],
                        "query_id_high": high["query_id"],
                        "query_text_high": high["query_text"],
                        "density_level_high": high["density_level"],
                        "density_label_high": high["density_label"],
                        "density_score_high": high["density_score"],
                        "label": 1,
                        "label_meaning": "query_high_is_denser_than_query_low",
                        "source_type": "main_chain_pair",
                    }
                )

    density_count = write_jsonl(DENSITY_CLS, density_rows)
    dependency_count = write_jsonl(DEPENDENCY_CLS, dependency_rows)
    rank_count = write_jsonl(DENSITY_RANK, rank_rows)

    report = {
        "source": str(SOURCE),
        "chain_rows": chain_count,
        "density_cls": {
            "path": str(DENSITY_CLS),
            "rows": density_count,
            "level_distribution": dict(level_dist),
            "split_distribution": dict(Counter(row["split"] for row in density_rows)),
        },
        "dependency_cls": {
            "path": str(DEPENDENCY_CLS),
            "rows": dependency_count,
            "dependency_distribution": dict(dep_dist),
            "split_distribution": dict(Counter(row["split"] for row in dependency_rows)),
        },
        "density_rank": {
            "path": str(DENSITY_RANK),
            "rows": rank_count,
            "split_distribution": dict(Counter(row["split"] for row in rank_rows)),
        },
        "chain_split_distribution": dict(split_dist),
        "chain_class_distribution": dict(class_dist),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
