#!/usr/bin/env python3
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


SOURCE = Path("/mnt/data_1/yds/多模态/权重模块/outputs/generated_chain.mainset_14000.v10_balanced_forms.jsonl")
ERRORS = Path("/mnt/data_1/yds/多模态/权重模块/outputs/generated_chain.mainset_14000.v10_balanced_forms.errors.jsonl")
REPORT = Path("/mnt/data_1/yds/多模态/权重模块/outputs/generated_chain.mainset_14000.v10_quality_report.json")


EXPECTED_LEVELS = ["L0", "L1", "L2", "L3", "L4"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def qtype(text: str) -> str:
    checks = [
        ("disease", r"病变"),
        ("screening", r"排查|排除"),
        ("consider", r"考虑|优先想到"),
        ("warning", r"警惕|重视|注意"),
        ("meaning", r"说明|提示|意味着|代表|指向"),
        ("understand", r"理解|怎么判断|方向判断|判断"),
        ("normal", r"正常吗|有问题吗|要紧吗"),
        ("common", r"常见吗"),
        ("what", r"是什么情况|怎么回事|是什么问题"),
        ("relation", r"有关吗|关系吗|附近|部位"),
    ]
    for name, pattern in checks:
        if re.search(pattern, text):
            return name
    return "other"


def main() -> None:
    rows = load_jsonl(SOURCE)
    errors = load_jsonl(ERRORS)

    class_counter = Counter()
    split_counter = Counter()
    model_counter = Counter()
    chain_len_counter = Counter()
    issue_counter = Counter()
    level_qtypes = defaultdict(Counter)
    level_lengths = defaultdict(list)
    repeated_texts = Counter()

    duplicate_ids = Counter()
    for row in rows:
        cid = row.get("candidate_id")
        duplicate_ids[cid] += 1
        class_counter[row.get("anchor_class", "UNK")] += 1
        split_counter[row.get("split", "UNK")] += 1
        model_counter[row.get("llm_model", "UNK")] += 1

        chain = row.get("main_chain", [])
        chain_len_counter[len(chain)] += 1
        if len(chain) != 5:
            issue_counter["main_chain_len_not_5"] += 1
            continue

        levels = [item.get("level") for item in chain]
        if levels != EXPECTED_LEVELS:
            issue_counter["wrong_level_order"] += 1

        prev_score = -1.0
        seen_texts = set()
        for item in chain:
            level = item.get("level", "")
            text = str(item.get("query_text", "")).strip()
            score = item.get("density_score")
            if not text.endswith("？"):
                issue_counter[f"{level}_not_question"] += 1
            if len(text) > 90:
                issue_counter[f"{level}_too_long"] += 1
            if text in seen_texts:
                issue_counter[f"{level}_duplicate_within_chain"] += 1
            seen_texts.add(text)
            repeated_texts[(level, text)] += 1
            level_qtypes[level][qtype(text)] += 1
            level_lengths[level].append(len(text))
            if not isinstance(score, (int, float)):
                issue_counter[f"{level}_missing_score"] += 1
            elif float(score) <= prev_score:
                issue_counter[f"{level}_non_monotonic_score"] += 1
            if isinstance(score, (int, float)):
                prev_score = float(score)

    top_repeated = [
        {"level": level, "query_text": text, "count": count}
        for (level, text), count in repeated_texts.most_common(30)
        if count > 1
    ]

    length_stats = {}
    for level, vals in level_lengths.items():
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        if n:
            length_stats[level] = {
                "min": vals_sorted[0],
                "p50": vals_sorted[n // 2],
                "p90": vals_sorted[int(n * 0.9)],
                "max": vals_sorted[-1],
            }

    duplicate_bad = {k: v for k, v in duplicate_ids.items() if k and v > 1}
    report = {
        "source": str(SOURCE),
        "errors_file": str(ERRORS),
        "success_rows": len(rows),
        "error_rows": len(errors),
        "class_distribution": dict(class_counter),
        "split_distribution": dict(split_counter),
        "model_distribution": dict(model_counter),
        "chain_len_distribution": dict(chain_len_counter),
        "issue_counts": dict(issue_counter),
        "duplicate_candidate_ids": duplicate_bad,
        "question_type_distribution_by_level": {
            level: dict(counter) for level, counter in sorted(level_qtypes.items())
        },
        "length_stats_by_level": length_stats,
        "top_repeated_queries": top_repeated,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
