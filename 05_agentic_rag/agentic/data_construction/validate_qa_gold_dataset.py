#!/usr/bin/env python3
"""Validate the gold QA JSONL assembled by the Stage-1 pipeline.

Covers: required fields, query_type allowed set, file existence, option schema,
answer leakage, evidence presence (at least one of caption_or_pair_text /
image_local_context_text / pdf_context_text), pdf_context_pages length, and
doc_id split-leakage.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

ALLOWED_QUERY_TYPES = {
    "anatomical_site_recognition",
    "lesion_or_finding_identification",
    "procedure_or_operation_recognition",
    "spatial_region_understanding",
}

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

REQUIRED_TOP_FIELDS = [
    "qid", "split", "query_type", "question", "options",
    "answer", "answer_text", "low_information_query_seed", "query_image_path",
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

        def add(issue: str) -> None:
            issues[issue] += 1
            if len(examples[issue]) < 5:
                examples[issue].append({"line_no": line_no, "qid": qid})

        # --- top-level required fields ---
        for field in REQUIRED_TOP_FIELDS:
            val = row.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                add(f"missing_{field}")

        # --- qid uniqueness ---
        if not qid:
            pass  # already flagged above
        elif qid in qids:
            add("duplicate_qid")
        qids.add(qid)

        # --- split ---
        if split not in {"train", "dev", "internal_test"}:
            add("invalid_split")
        else:
            split_counts[split] += 1

        # --- query_type ---
        query_type = str(row.get("query_type") or "")
        if not query_type:
            pass  # already flagged
        elif query_type not in ALLOWED_QUERY_TYPES:
            add(f"invalid_query_type:{query_type}")
        else:
            query_type_counts[query_type] += 1

        # --- options + answer ---
        if isinstance(options, dict) and set(options) == {"A", "B", "C", "D"}:
            if len(set(str(v) for v in options.values())) != 4:
                add("duplicate_option_values")
            if answer not in {"A", "B", "C", "D"}:
                add("invalid_answer")
            elif row.get("answer_text") != options.get(answer):
                add("answer_text_mismatch")
        else:
            add("invalid_options")

        # --- question ---
        question = str(row.get("question") or "")
        if not question:
            pass  # already flagged
        if any(pattern.search(question) for pattern in LEAKAGE_PATTERNS):
            add("question_answer_leakage")
        # answer text should not appear verbatim in the question
        answer_text = str(row.get("answer_text") or "")
        if answer_text and len(answer_text) >= 4 and answer_text in question:
            add("answer_text_in_question")

        # --- source ---
        if not isinstance(source, dict):
            add("missing_source")
        else:
            if not source.get("doc_id") and not source.get("doc_name"):
                add("missing_source_doc")
            # evidence: at least one of the three text fields must be non-empty
            has_evidence = any([
                str(source.get("caption_or_pair_text") or "").strip(),
                str(source.get("image_local_context_text") or "").strip(),
                str(source.get("pdf_context_text") or "").strip(),
            ])
            if not has_evidence:
                add("missing_all_evidence_text")

            doc_id = str(source.get("doc_id") or "")
            if doc_id and split:
                doc_splits[doc_id].add(split)

            # pdf_context_pages should be 1-2 pages
            pdf_pages = source.get("pdf_context_pages")
            if isinstance(pdf_pages, list) and not (1 <= len(pdf_pages) <= 2):
                add("pdf_context_pages_out_of_range")

        # --- file existence checks (lightweight: just check Path.exists) ---
        qip = str(row.get("query_image_path") or "")
        if qip and not Path(qip).exists():
            add("missing_query_image_file")

        origin_pdf = (source or {}).get("origin_pdf", "") if isinstance(source, dict) else ""
        if origin_pdf and not Path(origin_pdf).exists():
            add("missing_origin_pdf_file")

    # --- doc_id split-leakage ---
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
