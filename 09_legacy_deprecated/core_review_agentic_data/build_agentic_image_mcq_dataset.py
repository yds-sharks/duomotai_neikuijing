#!/usr/bin/env python3
"""Build image-query style MCQs from image-text pairs.

The question text intentionally does not expose source evidence. Evidence text,
PDF paths, and two-page context are kept under `source` for verification and RAG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_DB_PATH = "/mnt/data_1/yds/多模态/data_house/multimodal_samples.db"
DEFAULT_OUTPUT_DIR = "/mnt/data_1/yds/多模态/agentic/outputs/mcq_image_v2_4000"
DEFAULT_TOTAL = 4000
DEFAULT_SPLIT_COUNTS = {"train": 3200, "dev": 400, "internal_test": 400}

ORGAN_POOL = ["食管", "胃", "十二指肠", "小肠", "结直肠", "胆胰"]
KNOWLEDGE_TYPE_POOL = ["病变特征", "检查操作", "治疗操作", "诊断评估", "解剖特征", "基础概念"]

QUESTION_TEMPLATES = {
    "image_organ_identification": [
        "这张内镜图像主要显示哪个解剖部位？",
        "图中所示区域最可能属于哪个消化道部位？",
        "请判断这张图像主要对应下列哪个部位？",
        "该内镜图像主要观察到的是哪个部位？",
    ],
    "image_content_type_identification": [
        "这张医学图像主要对应下列哪类内容？",
        "图中信息最主要属于哪一类医学内容？",
        "该图像最主要展示的是下列哪类医学信息？",
    ],
}


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


def clean_text(text: str, max_len: int = 1600) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) > max_len:
        return text[:max_len].rstrip() + "..."
    return text


def split_organ_tags(value: Any) -> List[str]:
    tags = parse_json(value, [])
    if isinstance(tags, list):
        return [str(tag).strip() for tag in tags if str(tag).strip()]
    text = str(value or "").strip()
    return [text] if text else []


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
    matches = sorted(root.glob("auto/*_origin.pdf"))
    if matches:
        return str(matches[0])
    matches = sorted(root.rglob("*_origin.pdf"))
    return str(matches[0]) if matches else ""


@lru_cache(maxsize=10000)
def locate_content_list(source_path: str) -> str:
    if not source_path:
        return ""
    root = Path(source_path)
    if not root.exists():
        return ""
    matches = sorted(root.glob("auto/*_content_list.json"))
    if matches:
        return str(matches[0])
    matches = sorted(root.rglob("*_content_list.json"))
    return str(matches[0]) if matches else ""


@lru_cache(maxsize=10000)
def load_content_items(content_list_path: str) -> Tuple[Dict[str, Any], ...]:
    if not content_list_path:
        return tuple()
    path = Path(content_list_path)
    if not path.exists():
        return tuple()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return tuple()
    if not isinstance(data, list):
        return tuple()
    return tuple(item for item in data if isinstance(item, dict))


def item_image_name(item: Dict[str, Any]) -> str:
    for key in ("img_path", "image_path", "path"):
        value = str(item.get(key) or "")
        if value:
            return Path(value).name
    return ""


def infer_image_page_idx(source_path: str, image_path: str) -> Optional[int]:
    content_path = locate_content_list(source_path)
    items = load_content_items(content_path)
    if not items or not image_path:
        return None
    image_name = Path(image_path).name
    for item in items:
        if item.get("type") not in {"image", "figure"}:
            continue
        if item_image_name(item) == image_name and item.get("page_idx") is not None:
            return int(item["page_idx"])
    return None


def page_context(source_path: str, center_page: Optional[int], window_pages: int) -> Dict[str, Any]:
    content_path = locate_content_list(source_path)
    items = load_content_items(content_path)
    if not items or center_page is None:
        return {"content_list_path": content_path, "page_indices": [], "text": ""}
    start = max(0, center_page)
    end = center_page + max(1, window_pages) - 1
    page_indices = list(range(start, end + 1))
    chunks: List[str] = []
    for item in items:
        if item.get("page_idx") not in page_indices:
            continue
        text = str(item.get("text") or item.get("caption") or "").strip()
        if text:
            chunks.append(text)
    return {
        "content_list_path": content_path,
        "page_indices": page_indices,
        "text": clean_text("\n".join(chunks), max_len=3000),
    }


def choose_distractors(answer_text: str, pool: Sequence[str], rng: random.Random) -> Optional[List[str]]:
    candidates = [item for item in pool if item and item != answer_text]
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) < 3:
        return None
    return rng.sample(candidates, 3)


def option_map(answer_text: str, distractors: Sequence[str], rng: random.Random) -> Tuple[Dict[str, str], str]:
    labels = ["A", "B", "C", "D"]
    values = [answer_text] + list(distractors)
    rng.shuffle(values)
    options = dict(zip(labels, values))
    answer = next(label for label, value in options.items() if value == answer_text)
    return options, answer


def make_source(row: sqlite3.Row, level1: str, level2: str, window_pages: int) -> Dict[str, Any]:
    image_id, image_path = first_image(row["images"])
    source_path = str(row["source_path"] or "")
    image_page_idx = infer_image_page_idx(source_path, image_path)
    context = page_context(source_path, image_page_idx, window_pages)
    return {
        "source_type": "vector_db",
        "doc_id": str(row["doc_id"] or ""),
        "doc_name": str(row["doc_name"] or ""),
        "source_path": source_path,
        "origin_pdf_path": locate_origin_pdf(source_path),
        "page_idx": image_page_idx,
        "context_page_indices": context["page_indices"],
        "content_list_path": context["content_list_path"],
        "pdf_context_text": context["text"],
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


def build_candidate(row: sqlite3.Row, query_type: str, rng: random.Random, seed: int, window_pages: int) -> Optional[Dict[str, Any]]:
    image_id, image_path = first_image(row["images"])
    if not image_path or not Path(image_path).exists():
        return None
    organ_tags = split_organ_tags(row["organ_tags"])
    level1 = next((tag for tag in organ_tags if tag and tag != "通用"), "")
    level2 = str(row["primary_knowledge_type"] or "").strip()
    if not level2 or level2 == "其他":
        return None

    if query_type == "image_organ_identification":
        if level1 not in ORGAN_POOL:
            return None
        answer_text = level1
        distractors = choose_distractors(answer_text, ORGAN_POOL, rng)
    elif query_type == "image_content_type_identification":
        if level2 not in KNOWLEDGE_TYPE_POOL:
            return None
        answer_text = level2
        distractors = choose_distractors(answer_text, KNOWLEDGE_TYPE_POOL, rng)
    else:
        raise ValueError(query_type)
    if not distractors:
        return None

    options, answer = option_map(answer_text, distractors, rng)
    qid_hash = stable_hash(f"{seed}:{row['sample_id']}:{query_type}")[:12]
    return {
        "qid": f"mcqimg_{qid_hash}",
        "split": "",
        "query_type": query_type,
        "question": rng.choice(QUESTION_TEMPLATES[query_type]),
        "options": options,
        "answer": answer,
        "answer_text": answer_text,
        "query_image_path": image_path,
        "source": make_source(row, level1, level2, window_pages),
        "provenance": {
            "generated_by": "manual_template_image_query",
            "benchmark_used_for_training": False,
            "template_version": "image_v2",
            "api_verified": False,
        },
        "verification": {"status": "not_run", "verifier_model": ""},
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
            WHERE source_type = 'image_text_pair'
              AND have_image = 1
              AND length(text) BETWEEN ? AND ?
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
    rows = list(iter_rows(Path(args.db_path), args.min_chars, args.max_chars))
    rng.shuffle(rows)
    query_types = ["image_organ_identification", "image_content_type_identification"]
    candidates: List[Dict[str, Any]] = []
    seen = set()
    per_type_limit = args.total * args.overgenerate_factor
    type_counts = Counter()
    for row in rows:
        for query_type in query_types:
            if type_counts[query_type] >= per_type_limit:
                continue
            item = build_candidate(row, query_type, rng, args.seed, args.window_pages)
            if item is None or item["qid"] in seen:
                continue
            seen.add(item["qid"])
            candidates.append(item)
            type_counts[query_type] += 1
        if all(type_counts[q] >= per_type_limit for q in query_types):
            break
    rng.shuffle(candidates)
    return candidates


def split_and_sample(candidates: Sequence[Dict[str, Any]], split_counts: Dict[str, int], seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        pools[doc_split(item["source"]["doc_id"], seed)].append(item)

    selected: List[Dict[str, Any]] = []
    report = {"pool_counts": {}, "selected_counts": {}, "selected_query_type_counts": {}}
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
        query_types = sorted(by_type)
        split_items: List[Dict[str, Any]] = []
        base = needed // len(query_types)
        rem = needed % len(query_types)
        for i, query_type in enumerate(query_types):
            take = base + (1 if i < rem else 0)
            split_items.extend(by_type[query_type][:take])
        if len(split_items) < needed:
            selected_ids = {id(item) for item in split_items}
            leftovers = [item for item in pool if id(item) not in selected_ids]
            rng.shuffle(leftovers)
            split_items.extend(leftovers[: needed - len(split_items)])
        rng.shuffle(split_items)
        split_items = split_items[:needed]
        report["selected_counts"][split] = len(split_items)
        report["selected_query_type_counts"][split] = dict(Counter(item["query_type"] for item in split_items))
        for idx, item in enumerate(split_items, start=1):
            item = dict(item)
            item["split"] = split
            item["qid"] = f"medalign_img_{split}_{idx:05d}"
            selected.append(item)
    return selected, report


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_split_counts(text: str) -> Dict[str, int]:
    if not text:
        return dict(DEFAULT_SPLIT_COUNTS)
    loaded = json.loads(text)
    return {str(k): int(v) for k, v in loaded.items()}


def summarize(rows: Sequence[Dict[str, Any]], split_report: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    doc_splits: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        doc_splits[row["source"]["doc_id"]].add(row["split"])
    leaked = {doc: sorted(splits) for doc, splits in doc_splits.items() if len(splits) > 1}
    return {
        "db_path": args.db_path,
        "total_requested": args.total,
        "total_written": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "query_type_counts": dict(Counter(row["query_type"] for row in rows)),
        "answer_text_counts_top30": Counter(row["answer_text"] for row in rows).most_common(30),
        "with_query_image": sum(bool(row.get("query_image_path")) for row in rows),
        "with_origin_pdf": sum(bool(row["source"].get("origin_pdf_path")) for row in rows),
        "with_page_idx": sum(row["source"].get("page_idx") is not None for row in rows),
        "with_pdf_context_text": sum(bool(row["source"].get("pdf_context_text")) for row in rows),
        "split_report": split_report,
        "doc_id_split_leakage": leaked,
        "benchmark_used_for_training": False,
        "notes": [
            "Questions are image-query style and do not expose evidence text.",
            "Evidence text and two-page PDF context are kept only under source for verification/RAG.",
            "API verification is not run by this script.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--split-counts-json", default="")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument("--overgenerate-factor", type=int, default=3)
    parser.add_argument("--window-pages", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_counts = parse_split_counts(args.split_counts_json)
    if sum(split_counts.values()) != args.total:
        raise ValueError(f"Split counts sum {sum(split_counts.values())} != total {args.total}")
    output_dir = Path(args.output_dir)
    candidates = build_candidates(args)
    selected, split_report = split_and_sample(candidates, split_counts, args.seed)
    selected = sorted(selected, key=lambda row: (row["split"], row["qid"]))
    write_jsonl(output_dir / f"agentic_image_mcq_{args.total}.jsonl", selected)
    for split in split_counts:
        write_jsonl(output_dir / f"{split}.jsonl", [row for row in selected if row["split"] == split])
    report = summarize(selected, split_report, args)
    (output_dir / "construction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
