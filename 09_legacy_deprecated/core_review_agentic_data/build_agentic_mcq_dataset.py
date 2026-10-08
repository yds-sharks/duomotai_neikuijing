#!/usr/bin/env python3
"""Build source-grounded four-choice MCQs from multimodal_samples.db."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_DB_PATH = "/mnt/data_1/yds/多模态/data_house/multimodal_samples.db"
DEFAULT_OUTPUT_DIR = "/mnt/data_1/yds/多模态/agentic/outputs/mcq_template_4000"
DEFAULT_TOTAL = 4000
DEFAULT_SPLIT_COUNTS = {"train": 3200, "dev": 400, "internal_test": 400}

ORGAN_DISTRACTOR_POOL = [
    "食管",
    "胃",
    "小肠",
    "结直肠",
    "十二指肠",
    "回肠",
    "空肠",
    "盲肠",
    "胆道",
    "胰腺",
]

KNOWLEDGE_TYPE_POOL = [
    "病变特征",
    "检查操作",
    "治疗操作",
    "诊断评估",
    "解剖特征",
    "基础概念",
]

QUERY_TYPE_TEMPLATES = {
    "organ_identification": [
        "根据该样本关联的医学图像与证据描述，最主要涉及的解剖部位是？",
        "结合给定图像或证据文本，该样本最符合下列哪个部位？",
        "从证据内容判断，该医学样本主要对应哪个消化系统部位？",
    ],
    "knowledge_type_identification": [
        "根据该样本的医学证据，其主要知识类型属于哪一类？",
        "结合图像描述或邻近正文，该样本主要在描述哪类医学信息？",
        "从证据内容判断，该样本最主要对应以下哪种医学知识类型？",
    ],
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


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def doc_split(doc_id: str, seed: int) -> str:
    value = int(stable_hash(f"{seed}:{doc_id}")[:8], 16) / 0xFFFFFFFF
    if value < 0.80:
        return "train"
    if value < 0.90:
        return "dev"
    return "internal_test"


def parse_json(value: Any, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return fallback


def clean_text(text: str, max_len: int = 1200) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) > max_len:
        return text[:max_len].rstrip() + "..."
    return text


def first_image(images_value: Any) -> Tuple[str, str]:
    images = parse_json(images_value, [])
    if not isinstance(images, list):
        return "", ""
    for item in images:
        if not isinstance(item, dict):
            continue
        image_id = str(item.get("image_id") or "").strip()
        image_path = str(item.get("image_path") or "").strip()
        if image_id or image_path:
            return image_id, image_path
    return "", ""


@lru_cache(maxsize=10000)
def locate_origin_pdf(source_path: str) -> str:
    if not source_path:
        return ""
    root = Path(source_path)
    if not root.exists():
        return ""
    matches = sorted(root.glob("*/*/*_origin.pdf"))
    if matches:
        return str(matches[0])
    matches = sorted(root.rglob("*_origin.pdf"))
    return str(matches[0]) if matches else ""


def option_map(answer_text: str, distractors: Sequence[str], rng: random.Random) -> Tuple[Dict[str, str], str]:
    labels = ["A", "B", "C", "D"]
    values = [answer_text] + list(distractors)
    rng.shuffle(values)
    options = dict(zip(labels, values))
    answer = next(label for label, value in options.items() if value == answer_text)
    return options, answer


def choose_distractors(answer_text: str, pool: Sequence[str], rng: random.Random) -> Optional[List[str]]:
    candidates = [item for item in pool if item and item != answer_text]
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) < 3:
        return None
    return rng.sample(candidates, 3)


def question_for(query_type: str, rng: random.Random) -> str:
    return rng.choice(QUERY_TYPE_TEMPLATES[query_type])


def split_organ_tags(value: Any) -> List[str]:
    tags = parse_json(value, [])
    if isinstance(tags, list):
        return [str(tag).strip() for tag in tags if str(tag).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def row_to_source(row: sqlite3.Row, level1: str, level2: str) -> Dict[str, Any]:
    image_id, image_path = first_image(row["images"])
    return {
        "source_type": "vector_db",
        "doc_id": str(row["doc_id"] or ""),
        "doc_name": str(row["doc_name"] or ""),
        "source_path": str(row["source_path"] or ""),
        "origin_pdf_path": locate_origin_pdf(str(row["source_path"] or "")),
        "page_idx": row["page_idx"] if row["page_idx"] is not None else None,
        "block_id": row["block_id"] if row["block_id"] is not None else None,
        "sample_id": str(row["sample_id"] or ""),
        "group_id": str(row["group_id"] or ""),
        "image_id": image_id,
        "image_path": image_path,
        "evidence_text": clean_text(str(row["text"] or "")),
        "level1": level1,
        "level2": level2,
        "record_source_type": str(row["source_type"] or ""),
        "text_role": str(row["text_role"] or ""),
    }


def build_candidate(
    row: sqlite3.Row,
    query_type: str,
    rng: random.Random,
    qid_seed: str,
) -> Optional[Dict[str, Any]]:
    organ_tags = split_organ_tags(row["organ_tags"])
    level1 = next((tag for tag in organ_tags if tag and tag != "通用"), "")
    level2 = str(row["primary_knowledge_type"] or "").strip()
    if not level2 or level2 == "其他":
        return None

    if query_type == "organ_identification":
        if not level1:
            return None
        answer_text = level1
        distractors = choose_distractors(answer_text, ORGAN_DISTRACTOR_POOL, rng)
    elif query_type == "knowledge_type_identification":
        answer_text = level2
        distractors = choose_distractors(answer_text, KNOWLEDGE_TYPE_POOL, rng)
    else:
        raise ValueError(f"Unsupported query_type: {query_type}")

    if not distractors:
        return None

    options, answer = option_map(answer_text, distractors, rng)
    image_id, image_path = first_image(row["images"])
    qid_hash = stable_hash(f"{qid_seed}:{row['sample_id']}:{query_type}")[:12]
    question = question_for(query_type, rng)

    return {
        "qid": f"mcq_{qid_hash}",
        "split": "",
        "query_type": query_type,
        "question": question,
        "options": options,
        "answer": answer,
        "answer_text": answer_text,
        "query_image_path": image_path if row["have_image"] else "",
        "source": row_to_source(row, level1=level1 or "通用", level2=level2),
        "provenance": {
            "generated_by": "manual_template",
            "benchmark_used_for_training": False,
            "template_version": "v1",
            "api_verified": False,
        },
        "verification": {
            "status": "not_run",
            "verifier_model": "",
        },
    }


def iter_rows(db_path: Path, min_chars: int, max_chars: int) -> Iterable[sqlite3.Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        query = """
            SELECT sample_id, group_id, source_type, have_image, doc_id, doc_name,
                   source_path, page_idx, block_id, text, text_role, organ_tags,
                   primary_knowledge_type, secondary_knowledge_types, labels,
                   retrieval_meta, images, extra
            FROM multimodal_samples
            WHERE length(text) BETWEEN ? AND ?
              AND primary_knowledge_type IS NOT NULL
              AND primary_knowledge_type != ''
              AND primary_knowledge_type != '其他'
              AND doc_id IS NOT NULL
              AND doc_id != ''
        """
        for row in conn.execute(query, (min_chars, max_chars)):
            yield row
    finally:
        conn.close()


def build_candidates(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rng = random.Random(args.seed)
    candidates: List[Dict[str, Any]] = []
    seen_qids = set()
    per_type_limit = args.total * args.overgenerate_factor
    type_counts = Counter()

    rows = list(iter_rows(Path(args.db_path), args.min_chars, args.max_chars))
    rng.shuffle(rows)
    query_types = ["organ_identification", "knowledge_type_identification"]

    for row in rows:
        for query_type in query_types:
            if type_counts[query_type] >= per_type_limit:
                continue
            candidate = build_candidate(row, query_type, rng, qid_seed=str(args.seed))
            if candidate is None:
                continue
            if candidate["qid"] in seen_qids:
                continue
            seen_qids.add(candidate["qid"])
            type_counts[query_type] += 1
            candidates.append(candidate)
        if all(type_counts[qtype] >= per_type_limit for qtype in query_types):
            break

    rng.shuffle(candidates)
    return candidates


def split_and_sample(
    candidates: Sequence[Dict[str, Any]],
    split_counts: Dict[str, int],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        doc_id = str(item.get("source", {}).get("doc_id") or "")
        split = doc_split(doc_id, seed)
        pools[split].append(item)

    selected: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {"pool_counts": {}, "selected_counts": {}}
    for split, needed in split_counts.items():
        pool = pools[split]
        report["pool_counts"][split] = len(pool)
        if len(pool) < needed:
            raise ValueError(f"Insufficient {split} candidates: need {needed}, have {len(pool)}")
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in pool:
            by_type[item["query_type"]].append(item)
        for items in by_type.values():
            rng.shuffle(items)

        split_items: List[Dict[str, Any]] = []
        query_types = sorted(by_type)
        per_type_base = needed // len(query_types)
        remainder = needed % len(query_types)
        for index, query_type in enumerate(query_types):
            take = per_type_base + (1 if index < remainder else 0)
            split_items.extend(by_type[query_type][:take])
        if len(split_items) < needed:
            leftovers = [item for item in pool if item not in split_items]
            rng.shuffle(leftovers)
            split_items.extend(leftovers[: needed - len(split_items)])

        rng.shuffle(split_items)
        for idx, item in enumerate(split_items[:needed], start=1):
            item = dict(item)
            item["split"] = split
            item["qid"] = f"medalign_{split}_{idx:05d}"
            selected.append(item)
        report["selected_counts"][split] = len(split_items[:needed])

    return selected, report


def validate_basic(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    issues = Counter()
    qids = set()
    doc_splits: Dict[str, set[str]] = defaultdict(set)
    for item in items:
        qid = item.get("qid")
        if qid in qids:
            issues["duplicate_qid"] += 1
        qids.add(qid)
        answer = item.get("answer")
        options = item.get("options") or {}
        if answer not in {"A", "B", "C", "D"}:
            issues["invalid_answer"] += 1
        elif item.get("answer_text") != options.get(answer):
            issues["answer_text_mismatch"] += 1
        if len(set(options.values())) != 4:
            issues["duplicate_options"] += 1
        if item.get("provenance", {}).get("benchmark_used_for_training") is not False:
            issues["benchmark_flag_not_false"] += 1
        question = str(item.get("question") or "")
        if any(pattern.search(question) for pattern in LEAKAGE_PATTERNS):
            issues["question_leakage"] += 1
        doc_id = str(item.get("source", {}).get("doc_id") or "")
        if doc_id:
            doc_splits[doc_id].add(str(item.get("split") or ""))

    leaked_docs = {doc_id: sorted(splits) for doc_id, splits in doc_splits.items() if len(splits) > 1}
    if leaked_docs:
        issues["doc_id_split_leakage"] = len(leaked_docs)
    return {"issues": dict(issues), "doc_id_split_leakage": leaked_docs}


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: Sequence[Dict[str, Any]], split_report: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "db_path": args.db_path,
        "total_requested": args.total,
        "total_written": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "query_type_counts": dict(Counter(row["query_type"] for row in rows)),
        "source_record_type_counts": dict(Counter(row["source"]["record_source_type"] for row in rows)),
        "answer_text_counts_top30": Counter(row["answer_text"] for row in rows).most_common(30),
        "split_report": split_report,
        "basic_validation": validate_basic(rows),
        "benchmark_used_for_training": False,
        "notes": [
            "Questions are template-generated; answers are derived from structured source metadata.",
            "API verification is not run by this script.",
            "EndoBench samples are not used for training data construction.",
        ],
    }


def parse_split_counts(text: str) -> Dict[str, int]:
    if not text:
        return dict(DEFAULT_SPLIT_COUNTS)
    loaded = json.loads(text)
    return {str(key): int(value) for key, value in loaded.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--split-counts-json", default="")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--min-chars", type=int, default=30)
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--overgenerate-factor", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_counts = parse_split_counts(args.split_counts_json)
    if sum(split_counts.values()) != args.total:
        raise ValueError(f"Split counts sum {sum(split_counts.values())} != total {args.total}")

    output_dir = Path(args.output_dir)
    candidates = build_candidates(args)
    selected, split_report = split_and_sample(candidates, split_counts, args.seed)
    selected = sorted(selected, key=lambda item: (item["split"], item["qid"]))

    write_jsonl(output_dir / f"agentic_mcq_{args.total}.jsonl", selected)
    for split in split_counts:
        write_jsonl(output_dir / f"{split}.jsonl", [item for item in selected if item["split"] == split])

    report = summarize(selected, split_report, args)
    report_path = output_dir / "construction_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
