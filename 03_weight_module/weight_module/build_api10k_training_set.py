#!/usr/bin/env python3
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


BASE = Path("/mnt/data_1/yds/多模态/权重模块")
V1_FULL = BASE / "最终结果/run5000_high_anchor_v1/final_dataset/coarse_lr_consensus3465.full.jsonl"
V2_GEN = BASE / "最终结果/run5000_high_anchor_v2/output/generated_candidates.v2.jsonl"
V2_REV = BASE / "最终结果/run5000_high_anchor_v2/output/reviewed_candidates.v2.jsonl"

V2_FINAL_DIR = BASE / "最终结果/run5000_high_anchor_v2/final_dataset"
MERGED_DIR = BASE / "最终结果/run5000_merged_trainset"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def coarse_density(level: Optional[str]) -> Optional[str]:
    if level in {"L0", "L1"}:
        return "L01"
    if level in {"L2", "L3", "L4"}:
        return "L24"
    return None


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def stable_id(prefix: str, text: str, label_a: str, label_b: str) -> str:
    raw = f"{prefix}\t{text}\t{label_a}\t{label_b}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def load_v1_rows() -> List[Dict[str, Any]]:
    rows = []
    for row in iter_jsonl(V1_FULL):
        row = dict(row)
        row["dataset_source"] = "run5000_high_anchor_v1"
        row["query_text_norm"] = normalize_text(row.get("query_text", ""))
        rows.append(row)
    return rows


def build_v2_final_dataset() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    gen_map: Dict[str, Dict[str, Any]] = {}
    rev_map: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(V2_GEN):
        job = row.get("job") or {}
        job_id = str(job.get("job_id") or "").strip()
        if job_id:
            gen_map[job_id] = row
    for row in iter_jsonl(V2_REV):
        src = row.get("source") or {}
        job = src.get("job") or {}
        job_id = str(job.get("job_id") or "").strip()
        if job_id:
            rev_map[job_id] = row

    matched = []
    combo_counter = Counter()
    gen_combo_counter = Counter()
    rev_combo_counter = Counter()
    removed_l01_r1 = 0

    for job_id, gen_row in gen_map.items():
        rev_row = rev_map.get(job_id)
        if not rev_row:
            continue
        job = gen_row.get("job") or {}
        gen_item = ((gen_row.get("response") or {}).get("items") or [{}])[0]
        rev_item = ((rev_row.get("review") or {}).get("items") or [{}])[0]

        gen_density = gen_item.get("density_level")
        gen_routing = gen_item.get("routing_preference")
        rev_density = rev_item.get("density_level")
        rev_routing = rev_item.get("routing_preference")

        gen_coarse = coarse_density(gen_density)
        rev_coarse = coarse_density(rev_density)
        if not gen_coarse or not rev_coarse or not gen_routing or not rev_routing:
            continue

        gen_combo_counter[(gen_density, gen_routing)] += 1
        rev_combo_counter[(rev_density, rev_routing)] += 1

        if gen_coarse != rev_coarse or gen_routing != rev_routing:
            continue

        if gen_coarse == "L01" and gen_routing == "R1":
            removed_l01_r1 += 1
            continue

        source_text = ((job.get("legacy_anchor") or {}).get("text")) or ""
        query_text = gen_item.get("query_text") or ""
        row = {
            "job_id": job_id,
            "query_text": query_text,
            "density_coarse_label": gen_coarse,
            "routing_label": gen_routing,
            "source_text": source_text,
            "target_density_level": job.get("target_density_level"),
            "target_routing_preference": job.get("target_routing_preference"),
            "generated_density": gen_density,
            "generated_routing": gen_routing,
            "reviewed_density": rev_density,
            "reviewed_routing": rev_routing,
            "rewrite_confidence": gen_item.get("rewrite_confidence"),
            "label_confidence": gen_item.get("label_confidence"),
            "review_confidence": rev_item.get("confidence"),
            "dataset_source": "run5000_high_anchor_v2",
            "query_text_norm": normalize_text(query_text),
        }
        matched.append(row)
        combo_counter[(gen_coarse, gen_routing)] += 1

    matched.sort(key=lambda x: x["job_id"])
    V2_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    full_path = V2_FINAL_DIR / f"coarse_lr_consensus{len(matched)}.full.jsonl"
    min_path = V2_FINAL_DIR / f"coarse_lr_consensus{len(matched)}.min.jsonl"
    summary_path = V2_FINAL_DIR / f"coarse_lr_consensus{len(matched)}.summary.json"
    audit_path = V2_FINAL_DIR / f"coarse_lr_consensus{len(matched)}.audit.md"
    stats_path = V2_FINAL_DIR / "coarse_LR_match_stats.json"

    write_jsonl(full_path, matched)
    write_jsonl(
        min_path,
        (
            {
                "query_text": row["query_text"],
                "density_coarse_label": row["density_coarse_label"],
                "routing_label": row["routing_label"],
            }
            for row in matched
        ),
    )

    combo_distribution = []
    total = len(matched)
    for (density_label, routing_label), count in sorted(combo_counter.items(), key=lambda x: (-x[1], x[0])):
        combo_distribution.append(
            {
                "density_coarse_label": density_label,
                "routing_label": routing_label,
                "count": count,
                "ratio": round(count / total, 6) if total else 0.0,
            }
        )

    match_stats = {
        "matched_total": total,
        "matched_rate_over_successful_pairs": round(total / max(len(rev_map), 1), 6),
        "coarse_combo_distribution": [
            {
                "coarse_density": density_label,
                "routing": routing_label,
                "count": count,
                "ratio": round(count / total, 6) if total else 0.0,
            }
            for (density_label, routing_label), count in sorted(combo_counter.items(), key=lambda x: (-x[1], x[0]))
        ],
        "generated_original_distribution_within_matched": [
            {
                "density": density,
                "routing": routing,
                "count": count,
                "ratio": round(count / total, 6) if total else 0.0,
            }
            for (density, routing), count in sorted(gen_combo_counter.items(), key=lambda x: (-x[1], x[0]))
        ],
        "reviewed_original_distribution_within_matched": [
            {
                "density": density,
                "routing": routing,
                "count": count,
                "ratio": round(count / total, 6) if total else 0.0,
            }
            for (density, routing), count in sorted(rev_combo_counter.items(), key=lambda x: (-x[1], x[0]))
        ],
    }
    stats_path.write_text(json.dumps(match_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "final_count": total,
        "removed_l01_r1": removed_l01_r1,
        "combo_distribution": combo_distribution,
        "output_full": str(full_path),
        "output_min": str(min_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_lines = [
        f"# Coarse LR Consensus {total} Audit",
        "",
        f"- final_count: {total}",
        "- rule: coarse L+R consensus between generated and reviewed, then drop L01/R1",
    ]
    for row in matched[:30]:
        audit_lines.extend(
            [
                "",
                f"## {row['job_id']}",
                f"- query_text: {row['query_text']}",
                f"- label: {row['density_coarse_label']} / {row['routing_label']}",
                f"- source_text: {row['source_text']}",
            ]
        )
    audit_path.write_text("\n".join(audit_lines), encoding="utf-8")
    return matched, summary


def stratified_split(rows: List[Dict[str, Any]], seed: int = 20260428) -> List[Dict[str, Any]]:
    by_combo: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_combo[(row["density_coarse_label"], row["routing_label"])].append(row)

    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for combo, combo_rows in sorted(by_combo.items()):
        combo_rows = combo_rows[:]
        rng.shuffle(combo_rows)
        n = len(combo_rows)
        test_n = max(1, round(n * 0.10))
        val_n = max(1, round(n * 0.10))
        if test_n + val_n >= n:
            if n >= 3:
                test_n = 1
                val_n = 1
            elif n == 2:
                test_n = 1
                val_n = 0
            else:
                test_n = 0
                val_n = 0
        train_n = n - test_n - val_n
        for idx, row in enumerate(combo_rows):
            row = dict(row)
            if idx < train_n:
                row["split"] = "train"
            elif idx < train_n + val_n:
                row["split"] = "val"
            else:
                row["split"] = "test"
            out.append(row)
    return out


def merge_and_export(v1_rows: List[Dict[str, Any]], v2_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    seen: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    conflicts = []
    duplicate_same_label = 0
    for row in v1_rows + v2_rows:
        key = row["query_text_norm"]
        pair = (row["density_coarse_label"], row["routing_label"])
        if not key:
            continue
        if key in seen:
            prev = seen[key]
            prev_pair = (prev["density_coarse_label"], prev["routing_label"])
            if prev_pair == pair:
                duplicate_same_label += 1
            else:
                conflicts.append(
                    {
                        "query_text": row["query_text"],
                        "existing_pair": prev_pair,
                        "new_pair": pair,
                        "existing_source": prev.get("dataset_source"),
                        "new_source": row.get("dataset_source"),
                    }
                )
            continue
        seen[key] = dict(row)

    merged_rows = sorted(seen.values(), key=lambda x: (x["density_coarse_label"], x["routing_label"], x["query_text_norm"]))
    merged_rows = stratified_split(merged_rows)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    full_path = MERGED_DIR / "coarse_lr_merged_dedup.full.jsonl"
    min_path = MERGED_DIR / "coarse_lr_merged_dedup.min.jsonl"
    density_path = MERGED_DIR / "density_coarse_cls.api10k_merged.jsonl"
    routing_path = MERGED_DIR / "routing_cls.api10k_merged.jsonl"
    report_path = MERGED_DIR / "merge_report.json"
    conflict_path = MERGED_DIR / "label_conflicts.json"

    write_jsonl(full_path, merged_rows)
    write_jsonl(
        min_path,
        (
            {
                "query_text": row["query_text"],
                "density_coarse_label": row["density_coarse_label"],
                "routing_label": row["routing_label"],
                "split": row["split"],
            }
            for row in merged_rows
        ),
    )

    density_label_map = {"L01": 0, "L24": 1}
    routing_label_map = {"R1": 0, "R2": 1, "R3": 2}
    density_rows = []
    routing_rows = []
    for row in merged_rows:
        base_id = stable_id("api10k", row["query_text_norm"], row["density_coarse_label"], row["routing_label"])
        density_rows.append(
            {
                "query_id": f"density_{base_id}",
                "candidate_id": row.get("job_id") or base_id,
                "split": row["split"],
                "query_text": row["query_text"],
                "density_coarse_label": row["density_coarse_label"],
                "density_label": density_label_map[row["density_coarse_label"]],
                "routing_label": row["routing_label"],
                "dataset_source": row.get("dataset_source"),
            }
        )
        routing_rows.append(
            {
                "query_id": f"routing_{base_id}",
                "candidate_id": row.get("job_id") or base_id,
                "split": row["split"],
                "query_text": row["query_text"],
                "routing_label": row["routing_label"],
                "routing_id": routing_label_map[row["routing_label"]],
                "density_coarse_label": row["density_coarse_label"],
                "dataset_source": row.get("dataset_source"),
            }
        )
    write_jsonl(density_path, density_rows)
    write_jsonl(routing_path, routing_rows)

    split_dist = Counter(row["split"] for row in merged_rows)
    combo_dist = Counter((row["density_coarse_label"], row["routing_label"]) for row in merged_rows)
    source_dist = Counter(row.get("dataset_source") for row in merged_rows)
    report = {
        "v1_rows": len(v1_rows),
        "v2_rows": len(v2_rows),
        "merged_rows": len(merged_rows),
        "duplicate_same_label_removed": duplicate_same_label,
        "conflict_rows_removed": len(conflicts),
        "split_distribution": dict(split_dist),
        "source_distribution": dict(source_dist),
        "combo_distribution": [
            {
                "density_coarse_label": density,
                "routing_label": routing,
                "count": count,
                "ratio": round(count / len(merged_rows), 6) if merged_rows else 0.0,
            }
            for (density, routing), count in sorted(combo_dist.items(), key=lambda x: (-x[1], x[0]))
        ],
        "outputs": {
            "full": str(full_path),
            "min": str(min_path),
            "density_cls": str(density_path),
            "routing_cls": str(routing_path),
            "conflicts": str(conflict_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    conflict_path.write_text(json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    v1_rows = load_v1_rows()
    v2_rows, v2_summary = build_v2_final_dataset()
    report = merge_and_export(v1_rows, v2_rows)
    print(json.dumps({"v2_summary": v2_summary, "merge_report": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
