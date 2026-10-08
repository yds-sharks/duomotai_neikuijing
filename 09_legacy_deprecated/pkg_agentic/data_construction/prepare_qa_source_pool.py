#!/usr/bin/env python3
"""Prepare Stage-1 source pool before any API calls.

This script only extracts deterministic source information: text/image/PDF/page
context/metadata. It does not generate questions and does not call external APIs.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from qa_stage1_common import (
    clean_text,
    context_for_pages,
    context_near_image,
    doc_split,
    first_image,
    infer_image_page_idx,
    locate_content_list,
    locate_origin_pdf,
    normalize_tags,
    parse_json,
    source_id_from_sample,
    stable_hash,
)

DEFAULT_DB_PATH = "/mnt/data_1/yds/多模态/data_house/multimodal_samples.db"
DEFAULT_OUTPUT_DIR = "/mnt/data_1/yds/多模态/agentic/outputs/qa_stage1_prepared"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-sources", type=int, default=6000)
    parser.add_argument("--limit-rows", type=int, default=0, help="Debug only: limit DB rows before filtering.")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--max-context-chars", type=int, default=5000)
    parser.add_argument("--max-local-context-chars", type=int, default=2500)
    parser.add_argument("--local-before-items", type=int, default=8)
    parser.add_argument("--local-after-items", type=int, default=8)
    parser.add_argument("--max-caption-chars", type=int, default=1800)
    parser.add_argument("--min-context-chars", type=int, default=20)
    return parser.parse_args()


def fetch_rows(db_path: str, limit_rows: int, seed: int) -> List[sqlite3.Row]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    sql = """
        SELECT sample_id, group_id, source_type, have_image, doc_id, doc_name,
               source_path, page_idx, block_id, content_hash, text, text_role,
               organ_tags, primary_knowledge_type, secondary_knowledge_types,
               labels, retrieval_flags, retrieval_meta, images, bbox, coord_sys, extra
        FROM multimodal_samples
        WHERE source_type = 'image_text_pair' AND have_image = 1
    """
    if limit_rows > 0:
        sql += f" LIMIT {int(limit_rows)}"
    rows = list(con.execute(sql))
    con.close()
    rows.sort(key=lambda row: stable_hash(f"{seed}:{row['sample_id']}"))
    return rows


def build_source(row: sqlite3.Row, args: argparse.Namespace, rejects: Counter) -> Dict[str, Any] | None:
    sample_id = str(row["sample_id"] or "")
    image_id, image_path, image_meta = first_image(row["images"])
    if not image_path:
        rejects["no_image_path"] += 1
        return None
    if not Path(image_path).exists():
        rejects["missing_image_file"] += 1
        return None

    source_path = str(row["source_path"] or "")
    if not source_path or not Path(source_path).exists():
        rejects["missing_source_path"] += 1
        return None

    origin_pdf = locate_origin_pdf(source_path)
    if not origin_pdf or not Path(origin_pdf).exists():
        rejects["missing_origin_pdf"] += 1
        return None

    content_list = locate_content_list(source_path)
    if not content_list or not Path(content_list).exists():
        rejects["missing_content_list"] += 1
        return None

    page_idx = infer_image_page_idx(source_path, image_path, row["page_idx"])
    if page_idx is None:
        rejects["cannot_infer_page_idx"] += 1
        return None

    context = context_for_pages(source_path, page_idx, max_chars=args.max_context_chars)
    local_context = context_near_image(
        source_path,
        image_path,
        before_items=args.local_before_items,
        after_items=args.local_after_items,
        max_chars=args.max_local_context_chars,
    )
    caption_or_pair_text = clean_text(row["text"], max_len=args.max_caption_chars)
    if len(context["text"]) < args.min_context_chars and len(local_context["text"]) < args.min_context_chars and not caption_or_pair_text:
        rejects["empty_text_and_context"] += 1
        return None

    doc_id = str(row["doc_id"] or "") or stable_hash(source_path)[:8]
    organ_tags = normalize_tags(row["organ_tags"])
    secondary_types = normalize_tags(row["secondary_knowledge_types"])

    source_id = source_id_from_sample(sample_id)
    return {
        "source_id": source_id,
        "sample_id": sample_id,
        "group_id": str(row["group_id"] or ""),
        "doc_id": doc_id,
        "doc_name": str(row["doc_name"] or ""),
        "split": doc_split(doc_id, args.seed),
        "query_image_path": image_path,
        "image_id": image_id,
        "image_meta": image_meta,
        "origin_pdf": origin_pdf,
        "source_path": source_path,
        "content_list_path": content_list,
        "page_idx": page_idx,
        "pdf_context_pages": context["page_indices"],
        "image_item_index": local_context["image_item_index"],
        "image_local_context_text": local_context["text"],
        "caption_or_pair_text": caption_or_pair_text,
        "pdf_context_text": context["text"],
        "organ_tags": organ_tags,
        "primary_knowledge_type": clean_text(row["primary_knowledge_type"]),
        "secondary_knowledge_types": secondary_types,
        "labels": parse_json(row["labels"], {}),
        "retrieval_flags": parse_json(row["retrieval_flags"], {}),
        "retrieval_meta": parse_json(row["retrieval_meta"], {}),
        "bbox": parse_json(row["bbox"], None),
        "coord_sys": clean_text(row["coord_sys"]),
        "extra": parse_json(row["extra"], {}),
        "preparation": {
            "prepared_by": "prepare_qa_source_pool.py",
            "api_called": False,
            "context_policy": "prefer content-list text near matched image item; keep image page plus adjacent page as fallback",
        },
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rejects: Counter = Counter()
    rows = fetch_rows(args.db, args.limit_rows, args.seed)
    accepted: List[Dict[str, Any]] = []
    for row in rows:
        item = build_source(row, args, rejects)
        if item is None:
            continue
        accepted.append(item)
        if args.max_sources > 0 and len(accepted) >= args.max_sources:
            break

    source_path = out_dir / "source_pool.jsonl"
    with source_path.open("w", encoding="utf-8") as handle:
        for item in accepted:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "db": args.db,
        "output": str(source_path),
        "source_pool_count": len(accepted),
        "rows_seen": len(rows),
        "max_sources": args.max_sources,
        "reject_counts": dict(rejects),
        "split_counts": dict(Counter(item["split"] for item in accepted)),
        "primary_knowledge_type_counts": dict(Counter(item["primary_knowledge_type"] for item in accepted).most_common()),
        "doc_id_count": len({item["doc_id"] for item in accepted}),
        "api_called": False,
    }
    report_path = out_dir / "source_pool_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
