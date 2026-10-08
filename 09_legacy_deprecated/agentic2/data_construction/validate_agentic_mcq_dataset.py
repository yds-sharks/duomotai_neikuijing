#!/usr/bin/env python3
"""Validate MedAlign-RAG MCQ JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable


LEAKAGE_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"答案[是为]",
        r"正确答案",
        r"选[A-D]",
        r"选择[A-D]",
        r"\b[A-D]\s*是正确",
    )
]


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            row["_line_no"] = line_no
            yield row


def validate(path: Path) -> Dict[str, Any]:
    issues = Counter()
    split_counts = Counter()
    query_type_counts = Counter()
    qids = set()
    doc_splits: Dict[str, set[str]] = defaultdict(set)
    examples: Dict[str, list[dict[str, Any]]] = defaultdict(list)
    total = 0

    for row in iter_jsonl(path):
        total += 1
        line_no = row.pop("_line_no")
        qid = str(row.get("qid") or "")
        split = str(row.get("split") or "")
        answer = row.get("answer")
        options = row.get("options")
        source = row.get("source")
        provenance = row.get("provenance") or {}

        def add(issue: str) -> None:
            issues[issue] += 1
            if len(examples[issue]) < 5:
                examples[issue].append({"line_no": line_no, "qid": qid})

        if not qid:
            add("missing_qid")
        elif qid in qids:
            add("duplicate_qid")
        qids.add(qid)

        if split not in {"train", "dev", "internal_test"}:
            add("invalid_split")
        else:
            split_counts[split] += 1

        query_type = str(row.get("query_type") or "")
        if not query_type:
            add("missing_query_type")
        else:
            query_type_counts[query_type] += 1

        if not isinstance(options, dict) or set(options) != {"A", "B", "C", "D"}:
            add("invalid_options")
        else:
            if len(set(str(v) for v in options.values())) != 4:
                add("duplicate_option_values")
            if answer not in {"A", "B", "C", "D"}:
                add("invalid_answer")
            elif row.get("answer_text") != options.get(answer):
                add("answer_text_mismatch")

        question = str(row.get("question") or "")
        if not question:
            add("missing_question")
        if any(pattern.search(question) for pattern in LEAKAGE_PATTERNS):
            add("question_answer_leakage")

        if provenance.get("benchmark_used_for_training") is not False:
            add("benchmark_flag_not_false")

        if not isinstance(source, dict):
            add("missing_source")
        else:
            if not source.get("doc_id") and not source.get("doc_name"):
                add("missing_source_doc")
            if not source.get("evidence_text"):
                add("missing_evidence_text")
            doc_id = str(source.get("doc_id") or "")
            if doc_id and split:
                doc_splits[doc_id].add(split)

    leaked_docs = {doc_id: sorted(splits) for doc_id, splits in doc_splits.items() if len(splits) > 1}
    if leaked_docs:
        issues["doc_id_split_leakage"] = len(leaked_docs)
        examples["doc_id_split_leakage"] = [
            {"doc_id": doc_id, "splits": splits}
            for doc_id, splits in list(leaked_docs.items())[:5]
        ]

    return {
        "input_jsonl": str(path),
        "total": total,
        "issues": dict(issues),
        "examples": dict(examples),
        "split_counts": dict(split_counts),
        "query_type_counts": dict(query_type_counts),
        "valid": not issues,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--report-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(Path(args.input_jsonl))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(text + "\n", encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
