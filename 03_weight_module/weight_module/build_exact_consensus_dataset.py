#!/usr/bin/env python3
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BASE = Path("/mnt/data_1/yds/多模态/权重模块")
RUNS = [
    (
        "v1",
        BASE / "最终结果/run5000_high_anchor_v1/output/generated_candidates.v2.jsonl",
        BASE / "最终结果/run5000_high_anchor_v1/output/reviewed_candidates.v2.jsonl",
    ),
    (
        "v2",
        BASE / "最终结果/run5000_high_anchor_v2/output/generated_candidates.v2.jsonl",
        BASE / "最终结果/run5000_high_anchor_v2/output/reviewed_candidates.v2.jsonl",
    ),
]
OUT_DIR = BASE / "最终结果/exact_consensus_lr"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def build_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source_name, gen_path, rev_path in RUNS:
        rev_map = {
            (((row.get("source") or {}).get("job") or {}).get("job_id")): row
            for row in iter_jsonl(rev_path)
        }
        for row in iter_jsonl(gen_path):
            job = row.get("job") or {}
            gen_item = ((row.get("response") or {}).get("items") or [{}])[0]
            rev_row = rev_map.get(job.get("job_id"))
            if not rev_row:
                continue
            rev_item = ((rev_row.get("review") or {}).get("items") or [{}])[0]
            gen_density = gen_item.get("density_level")
            gen_routing = gen_item.get("routing_preference")
            rev_density = rev_item.get("density_level")
            rev_routing = rev_item.get("routing_preference")
            if not gen_density or not gen_routing:
                continue
            if gen_density != rev_density or gen_routing != rev_routing:
                continue

            query_text = str(gen_item.get("query_text") or "").strip()
            rows.append(
                {
                    "source_run": source_name,
                    "job_id": job.get("job_id"),
                    "query_text": query_text,
                    "query_text_norm": normalize_text(query_text),
                    "source_text": ((job.get("legacy_anchor") or {}).get("text")) or "",
                    "density_level": gen_density,
                    "routing_label": gen_routing,
                    "target_density_level": job.get("target_density_level"),
                    "target_routing_preference": job.get("target_routing_preference"),
                    "rewrite_confidence": gen_item.get("rewrite_confidence"),
                    "label_confidence": gen_item.get("label_confidence"),
                    "review_confidence": rev_item.get("confidence"),
                }
            )
    return rows


def group_split(rows: List[Dict[str, Any]], test_ratio: float = 0.1, seed: int = 20260428) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["query_text_norm"]].append(row)

    group_items = list(grouped.items())
    rng = random.Random(seed)
    rng.shuffle(group_items)

    label_by_group: Dict[str, Tuple[str, str]] = {}
    by_label: Dict[Tuple[str, str], List[Tuple[str, List[Dict[str, Any]]]]] = defaultdict(list)
    for key, group_rows in group_items:
        label = (group_rows[0]["density_level"], group_rows[0]["routing_label"])
        label_by_group[key] = label
        by_label[label].append((key, group_rows))

    out: List[Dict[str, Any]] = []
    for label, items in sorted(by_label.items()):
        rng.shuffle(items)
        total_rows = sum(len(group_rows) for _, group_rows in items)
        target_test_rows = round(total_rows * test_ratio)
        chosen_test = 0
        for idx, (key, group_rows) in enumerate(items):
            remaining_groups = len(items) - idx
            assign_test = False
            if target_test_rows > 0:
                if chosen_test < target_test_rows:
                    assign_test = True
                if remaining_groups == 1 and chosen_test == 0 and target_test_rows > 0:
                    assign_test = True
            split = "test" if assign_test else "train"
            if assign_test:
                chosen_test += len(group_rows)
            for row in group_rows:
                row = dict(row)
                row["split"] = split
                out.append(row)
    return out


def main() -> None:
    rows = build_rows()
    split_rows = group_split(rows)

    full_path = OUT_DIR / "exact_lr_consensus.full.jsonl"
    train_path = OUT_DIR / "exact_lr_consensus.train.jsonl"
    test_path = OUT_DIR / "exact_lr_consensus.test.jsonl"
    report_path = OUT_DIR / "exact_lr_consensus.report.json"

    write_jsonl(full_path, split_rows)
    write_jsonl(train_path, (row for row in split_rows if row["split"] == "train"))
    write_jsonl(test_path, (row for row in split_rows if row["split"] == "test"))

    split_dist = Counter(row["split"] for row in split_rows)
    density_dist = Counter(row["density_level"] for row in split_rows)
    routing_dist = Counter(row["routing_label"] for row in split_rows)
    joint_dist = Counter((row["density_level"], row["routing_label"]) for row in split_rows)

    dup_group_count = 0
    dup_extra_rows = 0
    by_key = defaultdict(list)
    for row in split_rows:
        by_key[(row["query_text_norm"], row["density_level"], row["routing_label"])].append(row)
    for group in by_key.values():
        if len(group) > 1:
            dup_group_count += 1
            dup_extra_rows += len(group) - 1

    report = {
        "rule": "keep rows only when generated density/routing exactly match reviewed density/routing",
        "split_rule": "grouped 9/1 by normalized query_text to avoid identical query leakage across train/test",
        "total_rows": len(split_rows),
        "split_distribution": dict(split_dist),
        "density_distribution": dict(density_dist),
        "routing_distribution": dict(routing_dist),
        "joint_distribution": {f"{k[0]}_{k[1]}": v for k, v in sorted(joint_dist.items())},
        "duplicate_groups_same_text_same_label_kept": dup_group_count,
        "duplicate_extra_rows_same_text_same_label_kept": dup_extra_rows,
        "outputs": {
            "full": str(full_path),
            "train": str(train_path),
            "test": str(test_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
