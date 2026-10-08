#!/usr/bin/env python3
"""Run the local weight module on a benchmark split and export per-sample labels."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from datasets import load_dataset

from multimodal_weight_service import MultimodalWeightService


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "weight_module_runtime.yaml"
DEFAULT_OUTPUT_JSONL = Path(__file__).resolve().parent / "benchmark_weight_outputs.jsonl"
DEFAULT_SUMMARY_JSON = Path(__file__).resolve().parent / "benchmark_weight_summary.json"
FILTER_FIELDS = ("dataset", "task", "scene", "category", "subtask")


def parse_csv_values(raw: str) -> Optional[set[str]]:
    text = str(raw or "").strip()
    if not text or text.lower() == "all":
        return None
    return {item.strip() for item in text.split(",") if item.strip()}


def load_filtered_dataset(args: argparse.Namespace):
    dataset = load_dataset(args.benchmark)[args.split]
    for field in FILTER_FIELDS:
        values = parse_csv_values(getattr(args, field))
        if values is None:
            continue
        dataset = dataset.filter(lambda row, field=field, values=values: str(row.get(field, "")) in values)

    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)

    start = max(int(args.offset), 0)
    stop = len(dataset) if args.limit is None else min(len(dataset), start + max(int(args.limit), 0))
    return dataset.select(range(start, stop))


def build_output_record(example: Dict[str, Any], index: int, analysis: Dict[str, Any]) -> Dict[str, Any]:
    text_analysis = analysis["text_analysis"]
    return {
        "index": example.get("index", index),
        "dataset": example.get("dataset", ""),
        "task": example.get("task", ""),
        "scene": example.get("scene", ""),
        "category": example.get("category", ""),
        "subtask": example.get("subtask", ""),
        "question": str(example.get("question", "")).strip(),
        "density_level": text_analysis["density_level"],
        "image_dependency_level": text_analysis["image_dependency_level"],
        "density_confidence": text_analysis["model_outputs"]["density_confidence"],
        "image_dependency_confidence": text_analysis["model_outputs"]["image_dependency_confidence"],
    }


def summarize(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    density_counter: Counter[str] = Counter()
    dependency_counter: Counter[str] = Counter()
    combo_counter: Counter[str] = Counter()
    category_density: Dict[str, Counter[str]] = {}
    category_dependency: Dict[str, Counter[str]] = {}
    total = 0

    for row in records:
        total += 1
        density = str(row["density_level"])
        dependency = str(row["image_dependency_level"])
        category = str(row.get("category", ""))

        density_counter[density] += 1
        dependency_counter[dependency] += 1
        combo_counter[f"{density}|{dependency}"] += 1
        category_density.setdefault(category, Counter())[density] += 1
        category_dependency.setdefault(category, Counter())[dependency] += 1

    return {
        "total_samples": total,
        "density_counts": dict(sorted(density_counter.items())),
        "image_dependency_counts": dict(sorted(dependency_counter.items())),
        "combo_counts": dict(sorted(combo_counter.items())),
        "category_density_counts": {k: dict(sorted(v.items())) for k, v in sorted(category_density.items())},
        "category_dependency_counts": {k: dict(sorted(v.items())) for k, v in sorted(category_dependency.items())},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run image_text_weight_module on a benchmark split.")
    parser.add_argument("--benchmark", default="Saint-lsy/EndoBench", help="Hugging Face dataset id.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument("--dataset", default="all", help="Dataset filter: all or comma-separated values.")
    parser.add_argument("--task", default="all", help="Task filter: all or comma-separated values.")
    parser.add_argument("--scene", default="all", help="Scene filter: all or comma-separated values.")
    parser.add_argument("--category", default="all", help="Category filter: all or comma-separated values.")
    parser.add_argument("--subtask", default="all", help="Subtask filter: all or comma-separated values.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of filtered samples to process.")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N filtered samples.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle filtered samples before slicing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when --shuffle is enabled.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Weight module runtime config.")
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_JSONL), help="Per-sample output JSONL.")
    parser.add_argument("--output-summary-json", default=str(DEFAULT_SUMMARY_JSON), help="Summary counts JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)

    dataset = load_filtered_dataset(args)
    output_path = Path(args.output_jsonl).resolve()
    summary_path = Path(args.output_summary_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with MultimodalWeightService(config_path=args.config) as service, output_path.open("w", encoding="utf-8") as f:
        for idx, example in enumerate(dataset):
            question = str(example.get("question", "")).strip()
            if not question:
                continue
            analysis = service.analyze(query_text=question)
            row = build_output_record(example, idx, analysis)
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "benchmark": args.benchmark,
        "split": args.split,
        "filters": {
            "dataset": args.dataset,
            "task": args.task,
            "scene": args.scene,
            "category": args.category,
            "subtask": args.subtask,
            "limit": args.limit,
            "offset": args.offset,
            "shuffle": bool(args.shuffle),
            "seed": int(args.seed),
        },
        "output_jsonl": str(output_path),
        "summary": summarize(rows),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"processed_samples={len(rows)}")
    print(f"saved_jsonl={output_path}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
