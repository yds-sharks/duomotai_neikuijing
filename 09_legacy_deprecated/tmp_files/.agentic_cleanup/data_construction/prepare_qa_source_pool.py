#!/usr/bin/env python3
"""Prepare Stage-1 source pool before any API calls.

This script only extracts deterministic source information: text/image/PDF/page
context/metadata. It does not generate questions and does not call external APIs.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

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
    row_get,
    source_id_from_sample,
    stable_hash,
    write_json_atomic,
    write_jsonl_atomic,
)

# Columns we would like to read. Only those present in the DB are selected;
# missing ones are tolerated (schema drift across DB versions).
DESIRED_COLUMNS = [
    "sample_id", "group_id", "source_type", "have_image", "doc_id", "doc_name",
    "source_path", "page_idx", "block_id", "content_hash", "text", "text_role",
    "organ_tags", "primary_knowledge_type", "secondary_knowledge_types",
    "labels", "retrieval_flags", "retrieval_meta", "images", "bbox", "coord_sys", "extra",
]
REQUIRED_COLUMNS = ["sample_id", "source_type", "have_image", "source_path", "images"]

DEFAULT_DB_PATH = "/mnt/data_1/yds/多模态/data_house/multimodal_samples.db"
DATA_DIR = Path(__file__).resolve().parent
AGENTIC_ROOT = DATA_DIR.parent
DEFAULT_OUTPUT_DIR = str(AGENTIC_ROOT / "outputs" / "qa_stage1_prepared")


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
    available = {r[1] for r in con.execute("PRAGMA table_info(multimodal_samples)")}
    if not available:
        con.close()
        raise RuntimeError("Table 'multimodal_samples' not found or has no columns.")
    missing_required = [c for c in REQUIRED_COLUMNS if c not in available]
    if missing_required:
        con.close()
        raise RuntimeError(f"DB missing required columns: {missing_required}. Available: {sorted(available)}")
    select_cols = [c for c in DESIRED_COLUMNS if c in available]
    missing_optional = [c for c in DESIRED_COLUMNS if c not in available]
    if missing_optional:
        print(f"[warn] optional columns absent, defaulting to None: {missing_optional}")
    sql = (
        f"SELECT {', '.join(select_cols)} FROM multimodal_samples "
        "WHERE source_type = 'image_text_pair' AND have_image = 1"
    )
    if limit_rows > 0:
        sql += f" LIMIT {int(limit_rows)}"
    rows = list(con.execute(sql))
    con.close()
    rows.sort(key=lambda row: stable_hash(f"{seed}:{row['sample_id']}"))
    return rows


def build_source(row: sqlite3.Row, args: argparse.Namespace, rejects: Counter) -> Dict[str, Any] | None:
    sample_id = str(row_get(row, "sample_id", "") or "")
    image_id, image_path, image_meta = first_image(row_get(row, "images"))
    if not image_path:
        rejects["no_image_path"] += 1
        return None
    if not Path(image_path).exists():
        rejects["missing_image_file"] += 1
        return None

    source_path = str(row_get(row, "source_path", "") or "")
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

    page_idx = infer_image_page_idx(source_path, image_path, row_get(row, "page_idx"))
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
    caption_or_pair_text = clean_text(row_get(row, "text"), max_len=args.max_caption_chars)
    if len(context["text"]) < args.min_context_chars and len(local_context["text"]) < args.min_context_chars and not caption_or_pair_text:
        rejects["empty_text_and_context"] += 1
        return None

    doc_id = str(row_get(row, "doc_id", "") or "") or stable_hash(source_path)[:8]
    organ_tags = normalize_tags(row_get(row, "organ_tags"))
    secondary_types = normalize_tags(row_get(row, "secondary_knowledge_types"))

    source_id = source_id_from_sample(sample_id)
    return {
        "source_id": source_id,
        "sample_id": sample_id,
        "group_id": str(row_get(row, "group_id", "") or ""),
        "doc_id": doc_id,
        "doc_name": str(row_get(row, "doc_name", "") or ""),
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
        "primary_knowledge_type": clean_text(row_get(row, "primary_knowledge_type")),
        "secondary_knowledge_types": secondary_types,
        "labels": parse_json(row_get(row, "labels"), {}),
        "retrieval_flags": parse_json(row_get(row, "retrieval_flags"), {}),
        "retrieval_meta": parse_json(row_get(row, "retrieval_meta"), {}),
        "bbox": parse_json(row_get(row, "bbox"), None),
        "coord_sys": clean_text(row_get(row, "coord_sys")),
        "extra": parse_json(row_get(row, "extra"), {}),
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

    t0 = time.time()
    rejects: Counter = Counter()
    rows = fetch_rows(args.db, args.limit_rows, args.seed)
    print(f"[time] fetched {len(rows)} rows in {time.time() - t0:.1f}s")

    accepted: List[Dict[str, Any]] = []
    t_scan = time.time()
    for i, row in enumerate(rows, start=1):
        item = build_source(row, args, rejects)
        if item is not None:
            accepted.append(item)
        if i % 2000 == 0 or i == len(rows):
            elapsed = time.time() - t_scan
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (len(rows) - i) / rate if rate > 0 else 0.0
            print(f"[progress] scanned {i}/{len(rows)} accepted={len(accepted)} "
                  f"rate={rate:.0f}/s eta={eta:.0f}s")
        if args.max_sources > 0 and len(accepted) >= args.max_sources:
            print(f"[info] reached max_sources={args.max_sources}, stop scanning")
            break

    source_path = out_dir / "source_pool.jsonl"
    written = write_jsonl_atomic(source_path, accepted)

    report = {
        "db": args.db,
        "output": str(source_path),
        "source_pool_count": written,
        "rows_seen": len(rows),
        "max_sources": args.max_sources,
        "elapsed_s": round(time.time() - t0, 1),
        "reject_counts": dict(rejects),
        "split_counts": dict(Counter(item["split"] for item in accepted)),
        "primary_knowledge_type_counts": dict(Counter(item["primary_knowledge_type"] for item in accepted).most_common()),
        "doc_id_count": len({item["doc_id"] for item in accepted}),
        "api_called": False,
    }
    write_json_atomic(out_dir / "source_pool_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
