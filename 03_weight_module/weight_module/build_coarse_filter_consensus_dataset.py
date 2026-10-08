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
OUT_DIR = BASE / "最终结果/coarse_filter_exact_routing"

LEVEL_TO_ID = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}
COARSE_TO_ID = {"L01": 0, "L234": 1}


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


def coarse_level(level: str) -> str:
    if level in {"L0", "L1"}:
        return "L01"
    if level in {"L2", "L3", "L4"}:
        return "L234"
    raise ValueError(f"Unsupported level: {level}")


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
            if not gen_density or not gen_routing or not rev_density or not rev_routing:
                continue
            if coarse_level(gen_density) != coarse_level(rev_density):
                continue
            if gen_routing != rev_routing:
                continue

            query_text = str(gen_item.get("query_text") or "").strip()
            rows.append(
                {
                    "source_run": source_name,
                    "job_id": job.get("job_id"),
                    "query_text": query_text,
                    "query_text_norm": normalize_text(query_text),
                    "source_text": ((job.get("legacy_anchor") or {}).get("text")) or "",
                    "generated_density_level": gen_density,
                    "reviewed_density_level": rev_density,
                    "final_density_level": rev_density,
                    "final_density_id": LEVEL_TO_ID[rev_density],
                    "final_density_coarse": coarse_level(rev_density),
                    "final_density_coarse_id": COARSE_TO_ID[coarse_level(rev_density)],
                    "routing_label": rev_routing,
                    "target_density_level": job.get("target_density_level"),
                    "target_routing_preference": job.get("target_routing_preference"),
                    "rewrite_confidence": gen_item.get("rewrite_confidence"),
                    "label_confidence": gen_item.get("label_confidence"),
                    "review_confidence": rev_item.get("confidence"),
                }
            )
    return rows


def dedupe_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["query_text_norm"], row["final_density_level"], row["routing_label"])].append(row)

    deduped = []
    extra_rows = 0
    for key, group in grouped.items():
        deduped.append(group[0])
        extra_rows += len(group) - 1

    by_text_labels: Dict[str, set[Tuple[str, str]]] = defaultdict(set)
    for row in deduped:
        by_text_labels[row["query_text_norm"]].add((row["final_density_level"], row["routing_label"]))
    conflict_groups = {k: v for k, v in by_text_labels.items() if len(v) > 1}

    report = {
        "exact_duplicate_groups": sum(1 for group in grouped.values() if len(group) > 1),
        "exact_duplicate_extra_rows_removed": extra_rows,
        "conflict_text_groups_after_dedup": len(conflict_groups),
    }
    return deduped, report


def split_rows(rows: List[Dict[str, Any]], test_ratio: float = 0.1, seed: int = 20260428) -> List[Dict[str, Any]]:
    by_joint: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_joint[(row["final_density_level"], row["routing_label"])].append(row)

    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for joint, group_rows in sorted(by_joint.items()):
        group_rows = group_rows[:]
        rng.shuffle(group_rows)
        n = len(group_rows)
        test_n = max(1, round(n * test_ratio)) if n > 1 else 0
        train_n = n - test_n
        for idx, row in enumerate(group_rows):
            row = dict(row)
            row["split"] = "train" if idx < train_n else "test"
            out.append(row)
    return out


def main() -> None:
    raw_rows = build_rows()
    deduped_rows, dedup_report = dedupe_rows(raw_rows)
    split_data = split_rows(deduped_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full_path = OUT_DIR / "coarse_filter_exact_routing.full.jsonl"
    train_path = OUT_DIR / "coarse_filter_exact_routing.train.jsonl"
    test_path = OUT_DIR / "coarse_filter_exact_routing.test.jsonl"
    density5_path = OUT_DIR / "density5_cls.coarse_filter_exact_routing.jsonl"
    density2_path = OUT_DIR / "density2_cls.coarse_filter_exact_routing.jsonl"
    report_path = OUT_DIR / "coarse_filter_exact_routing.report.json"

    write_jsonl(full_path, split_data)
    write_jsonl(train_path, (row for row in split_data if row["split"] == "train"))
    write_jsonl(test_path, (row for row in split_data if row["split"] == "test"))

    density5_rows = []
    density2_rows = []
    for row in split_data:
        base = {
            "query_id": row["job_id"],
            "candidate_id": row["job_id"],
            "split": row["split"],
            "query_text": row["query_text"],
            "source_run": row["source_run"],
            "routing_label": row["routing_label"],
        }
        density5_rows.append(
            {
                **base,
                "density_level": row["final_density_level"],
                "density_label": row["final_density_id"],
            }
        )
        density2_rows.append(
            {
                **base,
                "density_coarse_level": row["final_density_coarse"],
                "density_label": row["final_density_coarse_id"],
            }
        )

    write_jsonl(density5_path, density5_rows)
    write_jsonl(density2_path, density2_rows)

    split_dist = Counter(row["split"] for row in split_data)
    density_dist = Counter(row["final_density_level"] for row in split_data)
    coarse_dist = Counter(row["final_density_coarse"] for row in split_data)
    routing_dist = Counter(row["routing_label"] for row in split_data)
    joint_dist = Counter((row["final_density_level"], row["routing_label"]) for row in split_data)

    report = {
        "rule": "keep rows when coarse density matches between generated/reviewed and routing matches exactly; use reviewed fine density as final label",
        "raw_kept_rows_before_dedup": len(raw_rows),
        "rows_after_exact_dedup": len(split_data),
        "split_rule": "stratified 9/1 split by final density+routing after exact dedup",
        "split_distribution": dict(split_dist),
        "density_distribution": dict(density_dist),
        "coarse_density_distribution": dict(coarse_dist),
        "routing_distribution": dict(routing_dist),
        "joint_distribution": {f"{k[0]}_{k[1]}": v for k, v in sorted(joint_dist.items())},
        **dedup_report,
        "outputs": {
            "full": str(full_path),
            "train": str(train_path),
            "test": str(test_path),
            "density5": str(density5_path),
            "density2": str(density2_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
