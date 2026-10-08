#!/usr/bin/env python3
"""Analyze benchmark question type distribution without using samples for training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable


DEFAULT_BENCHMARK_JSONL = (
    "/mnt/data_10/mwx/workspace/multi_modal_rag/evaluate_benchmark/"
    "rag_retrieval_export/output/20260511_151007/retrieval_export.jsonl"
)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def counter_to_rows(counter: Counter, total: int) -> list[dict[str, Any]]:
    rows = []
    for key, count in counter.most_common():
        if not isinstance(key, tuple):
            key = (key,)
        rows.append({"key": list(key), "count": count, "ratio": count / total if total else 0.0})
    return rows


def analyze(path: Path) -> Dict[str, Any]:
    dataset = Counter()
    category = Counter()
    task = Counter()
    subtask = Counter()
    n = 0

    for row in iter_jsonl(path):
        sample = row.get("sample") or row
        n += 1
        ds = str(sample.get("dataset") or "")
        cat = str(sample.get("category") or "")
        task_name = str(sample.get("task") or "")
        subtask_name = str(sample.get("subtask") or "")
        dataset[ds] += 1
        category[cat] += 1
        task[(cat, task_name)] += 1
        subtask[(cat, task_name, subtask_name)] += 1

    return {
        "benchmark_jsonl": str(path),
        "num_samples": n,
        "dataset_distribution": counter_to_rows(dataset, n),
        "category_distribution": counter_to_rows(category, n),
        "task_distribution": counter_to_rows(task, n),
        "subtask_distribution": counter_to_rows(subtask, n),
        "training_use_allowed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-jsonl", default=DEFAULT_BENCHMARK_JSONL)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(Path(args.benchmark_jsonl))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
