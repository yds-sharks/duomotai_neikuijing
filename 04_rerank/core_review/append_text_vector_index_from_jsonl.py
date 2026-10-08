#!/usr/bin/env python3
"""Validate and append text records into text_vector_index from a JSONL file.

This script is designed for the existing multimodal retrieval pipeline:
    JSONL -> normalized text records -> BGE-M3 embeddings -> Milvus Lite text_vector_index

Safety model:
1. Default mode is validate-only.
2. Append mode must be enabled explicitly with --append.
3. Existing ids in Milvus are skipped by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from pymilvus import Collection, connections, utility

try:
    from retrieval.多模态.common import DEFAULT_BGE_MODEL_PATH
    from retrieval.多模态.text_dense_module import embed_texts, load_bge_m3_encoder
except Exception:
    sys.path.insert(0, "/mnt/data_1/yds/多模态/retrieval/多模态")
    from common import DEFAULT_BGE_MODEL_PATH  # type: ignore
    from text_dense_module import embed_texts, load_bge_m3_encoder  # type: ignore


DEFAULT_MILVUS_DB_PATH = "/mnt/data_1/yds/多模态/data_house/milvus/multimodal_vector_indexes.db"
TEXT_COLLECTION_NAME = "text_vector_index"
TEXT_VECTOR_DIM = 1024


def text_row_to_milvus(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": record["id"],
        "sample_id": record["sample_id"],
        "group_id": record["group_id"],
        "image_ids": record["image_ids"],
        "text": record["text"],
        "source_type": record["source_type"],
        "text_role": record["text_role"],
        "have_image": bool(record["have_image"]),
        "body_site_main": record["body_site_main"],
        "body_site_all": record["body_site_all"],
        "knowledge_type_main": record["knowledge_type_main"],
        "organ_tags": record["organ_tags"],
        "primary_knowledge_type": record["primary_knowledge_type"],
        "secondary_knowledge_types": record["secondary_knowledge_types"],
        "is_general": bool(record["is_general"]),
        "is_mixed": bool(record["is_mixed"]),
        "is_key_knowledge": bool(record["is_key_knowledge"]),
        "doc_id": record["doc_id"] or "",
        "doc_name": record["doc_name"] or "",
        "page_idx": int(record["page_idx"] if record["page_idx"] is not None else -1),
        "block_id": int(record["block_id"] if record["block_id"] is not None else -1),
        "dense_vector": record["embedding"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate/append text records into text_vector_index")
    parser.add_argument(
        "--input-jsonl",
        default="/mnt/data_10/mwx/workspace/multi_modal_rag/data_processing/stage3_label/output/labeled_samples.jsonl",
        help="Input JSONL path",
    )
    parser.add_argument("--milvus-db-path", default=DEFAULT_MILVUS_DB_PATH, help="Milvus Lite DB path")
    parser.add_argument("--collection-name", default=TEXT_COLLECTION_NAME, help="Target collection")
    parser.add_argument("--model-path", default=DEFAULT_BGE_MODEL_PATH, help="BGE-M3 model path")
    parser.add_argument("--device", default="cuda", help="Embedding device; use auto/cpu/cuda")
    parser.add_argument("--embed-batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--insert-batch-size", type=int, default=256, help="Milvus insert batch size")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N records")
    parser.add_argument("--append", action="store_true", help="Actually append new vectors into Milvus")
    parser.add_argument("--skip-dedup", action="store_true", help="Skip querying existing ids in Milvus")
    parser.add_argument("--show-samples", type=int, default=3, help="Show this many normalized samples")
    parser.add_argument("--report-json", default="", help="Optional path to dump the validation report")
    return parser.parse_args()


def canonical_text(value: Any) -> str:
    return str(value or "").strip()


def ensure_list_of_str(value: Any) -> List[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                return [str(x).strip() for x in loaded if str(x).strip()]
        except Exception:
            pass
        return [text]
    return [str(value).strip()]


def ensure_json_list(value: Any) -> List[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            if isinstance(loaded, list):
                return loaded
        except Exception:
            pass
    return []


def ensure_json_dict(value: Any) -> Dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
    return {}


def build_retrieval_flags(organ_tags: List[str]) -> Dict[str, Any]:
    concrete = [tag for tag in organ_tags if tag != "通用"]
    return {
        "is_general": "通用" in organ_tags,
        "is_mixed": False if "通用" in organ_tags else len(concrete) >= 2,
        "is_key_knowledge": False,
    }


def build_retrieval_meta(organ_tags: List[str], primary_knowledge_type: str) -> Dict[str, Any]:
    return {
        "body_site_main": "通用" if "通用" in organ_tags else (organ_tags[0] if organ_tags else ""),
        "body_site_all": organ_tags,
        "knowledge_type_main": primary_knowledge_type,
    }


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def stable_int64(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def derive_doc_fields(raw: Dict[str, Any]) -> Tuple[str, str]:
    doc_id = canonical_text(pick_first(raw, ("doc_id",), ""))
    doc_name = canonical_text(pick_first(raw, ("doc_name",), ""))
    if doc_id and doc_name:
        return doc_id, doc_name

    doc_out_dir = canonical_text(raw.get("doc_out_dir"))
    doc_parse_dir = canonical_text(raw.get("doc_parse_dir"))

    path_source = doc_out_dir or doc_parse_dir
    if path_source:
        base = os.path.basename(path_source.rstrip("/"))
        if base == "auto":
            base = os.path.basename(os.path.dirname(path_source.rstrip("/")))
        match = re.search(r"__([0-9a-f]{8})$", base)
        if not doc_id and match:
            doc_id = match.group(1)

        if not doc_name:
            if doc_parse_dir:
                parts = doc_parse_dir.rstrip("/").split("/")
                if len(parts) >= 2 and parts[-1] == "auto":
                    doc_name = parts[-2]
            if not doc_name and match:
                doc_name = base[: match.start()].replace("_", " ").strip()
            elif not doc_name:
                doc_name = base.replace("_", " ").strip()

    return doc_id, doc_name


def pick_first(record: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def normalize_record(raw: Dict[str, Any], line_no: int) -> Tuple[Dict[str, Any] | None, List[str]]:
    errors: List[str] = []

    labels = ensure_json_dict(raw.get("labels"))
    retrieval_flags = ensure_json_dict(raw.get("retrieval_flags"))
    retrieval_meta = ensure_json_dict(raw.get("retrieval_meta"))
    images = ensure_json_list(raw.get("images"))
    content_hash = canonical_text(raw.get("content_hash"))

    text = canonical_text(pick_first(raw, ("text", "content", "query_text"), ""))
    if not text:
        errors.append("missing_text")

    organ_tags = ensure_list_of_str(
        pick_first(raw, ("organ_tags", "level1"), labels.get("level1", []))
    )
    primary_knowledge_type = canonical_text(
        pick_first(raw, ("primary_knowledge_type", "level2_main"), labels.get("level2_main", ""))
    )
    secondary_knowledge_types = ensure_list_of_str(
        pick_first(raw, ("secondary_knowledge_types", "level2_aux"), labels.get("level2_aux", []))
    )

    if not retrieval_flags:
        retrieval_flags = build_retrieval_flags(organ_tags)
    if not retrieval_meta:
        retrieval_meta = build_retrieval_meta(organ_tags, primary_knowledge_type)

    sample_id = canonical_text(pick_first(raw, ("sample_id", "id"), ""))
    doc_id, doc_name = derive_doc_fields(raw)
    block_id_raw = pick_first(raw, ("block_id",), None)
    page_idx_raw = pick_first(raw, ("page_idx",), None)
    source_type = canonical_text(pick_first(raw, ("source_type",), "text_only")) or "text_only"
    text_role = canonical_text(pick_first(raw, ("text_role",), "knowledge_block")) or "knowledge_block"

    try:
        block_id = int(block_id_raw) if block_id_raw not in (None, "") else -1
    except Exception:
        block_id = -1
        errors.append("invalid_block_id")

    try:
        page_idx = int(page_idx_raw) if page_idx_raw not in (None, "") else -1
    except Exception:
        page_idx = -1
        errors.append("invalid_page_idx")

    if block_id_raw in (None, ""):
        block_seed = content_hash or text
        block_id = stable_int64(block_seed) if block_seed else stable_int64(f"line_{line_no}")

    if not sample_id:
        if doc_id and block_id >= 0:
            sample_id = f"txt_{doc_id}_{block_id}"
        elif content_hash:
            sample_id = f"txt_append_{content_hash[:16]}"
        else:
            sample_id = f"txt_append_{stable_hash(text)}"

    group_id = canonical_text(pick_first(raw, ("group_id",), f"grp_{sample_id}")) or f"grp_{sample_id}"
    have_image_raw = pick_first(raw, ("have_image",), None)
    if have_image_raw is None:
        have_image = bool(images)
    else:
        have_image = bool(have_image_raw)

    normalized = {
        "id": sample_id,
        "sample_id": sample_id,
        "group_id": group_id,
        "image_ids": [str(item.get("image_id", "")).strip() for item in images if str(item.get("image_id", "")).strip()],
        "text": text,
        "source_type": source_type,
        "text_role": text_role,
        "have_image": have_image,
        "body_site_main": canonical_text(retrieval_meta.get("body_site_main", "")),
        "body_site_all": ensure_json_list(retrieval_meta.get("body_site_all", organ_tags)),
        "knowledge_type_main": canonical_text(retrieval_meta.get("knowledge_type_main", primary_knowledge_type)),
        "organ_tags": organ_tags,
        "primary_knowledge_type": primary_knowledge_type,
        "secondary_knowledge_types": secondary_knowledge_types,
        "is_general": bool(retrieval_flags.get("is_general", False)),
        "is_mixed": bool(retrieval_flags.get("is_mixed", False)),
        "is_key_knowledge": bool(retrieval_flags.get("is_key_knowledge", False)),
        "doc_id": doc_id,
        "doc_name": doc_name,
        "page_idx": page_idx,
        "block_id": block_id,
        "_line_no": line_no,
        "_raw_keys": sorted(raw.keys()),
    }

    if len(text) > 32768:
        errors.append("text_too_long")
    if len(sample_id) > 128:
        errors.append("sample_id_too_long")
    if len(group_id) > 128:
        errors.append("group_id_too_long")
    if len(source_type) > 32:
        errors.append("source_type_too_long")
    if len(text_role) > 32:
        errors.append("text_role_too_long")
    if len(normalized["body_site_main"]) > 128:
        errors.append("body_site_main_too_long")
    if len(normalized["knowledge_type_main"]) > 128:
        errors.append("knowledge_type_main_too_long")
    if len(primary_knowledge_type) > 128:
        errors.append("primary_knowledge_type_too_long")
    if len(doc_id) > 128:
        errors.append("doc_id_too_long")
    if len(doc_name) > 1024:
        errors.append("doc_name_too_long")

    return (normalized if not errors else normalized), errors


def iter_jsonl(path: Path, limit: int | None = None) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)
            if limit is not None and line_no >= limit:
                break


def validate_jsonl(path: Path, limit: int | None = None, show_samples: int = 3) -> Dict[str, Any]:
    issues = Counter()
    source_keys = Counter()
    normalized_rows: List[Dict[str, Any]] = []
    total = 0

    for line_no, raw in iter_jsonl(path, limit=limit):
        total += 1
        source_keys.update(raw.keys())
        normalized, errors = normalize_record(raw, line_no)
        for err in errors:
            issues[err] += 1
        if normalized is not None and len(normalized_rows) < show_samples:
            normalized_rows.append(normalized)

    return {
        "input_jsonl": str(path),
        "checked_records": total,
        "issues": dict(issues),
        "source_keys": dict(source_keys),
        "normalized_samples": normalized_rows,
        "validation_ok": not issues,
    }


def _expr_list(values: Sequence[str]) -> str:
    escaped = [json.dumps(str(v), ensure_ascii=False) for v in values]
    return "[" + ",".join(escaped) + "]"


def fetch_existing_ids(collection: Collection, ids: Sequence[str], chunk_size: int = 200) -> set[str]:
    existing: set[str] = set()
    uniq_ids = [str(x) for x in dict.fromkeys(ids) if str(x)]
    for start in range(0, len(uniq_ids), chunk_size):
        chunk = uniq_ids[start : start + chunk_size]
        if not chunk:
            continue
        rows = collection.query(expr=f"id in {_expr_list(chunk)}", output_fields=["id"])
        for row in rows:
            existing.add(str(row.get("id", "")))
    return existing


def append_records(args: argparse.Namespace) -> Dict[str, Any]:
    path = Path(args.input_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    normalized_rows: List[Dict[str, Any]] = []
    issues = Counter()

    for line_no, raw in iter_jsonl(path, limit=args.limit):
        normalized, errors = normalize_record(raw, line_no)
        for err in errors:
            issues[err] += 1
        if normalized is not None and not errors:
            normalized_rows.append(normalized)
        if line_no % 20000 == 0:
            print(f"[normalize] processed={line_no} valid={len(normalized_rows)}", flush=True)

    if issues:
        raise ValueError(f"Validation failed, please fix input first: {dict(issues)}")

    connections.connect(alias="default", uri=args.milvus_db_path)
    try:
        if not utility.has_collection(args.collection_name):
            raise ValueError(f"Missing collection: {args.collection_name}")
        collection = Collection(args.collection_name)
        collection.load()

        if args.skip_dedup:
            existing_ids = set()
            new_rows = normalized_rows
            print(f"[dedup] skipped existing-id query, new={len(new_rows)}", flush=True)
        else:
            existing_ids = fetch_existing_ids(collection, [row["id"] for row in normalized_rows])
            new_rows = [row for row in normalized_rows if row["id"] not in existing_ids]
        print(
            f"[dedup] validated={len(normalized_rows)} existing={len(existing_ids)} new={len(new_rows)}",
            flush=True,
        )

        if not new_rows:
            return {
                "validated_records": len(normalized_rows),
                "existing_records": len(existing_ids),
                "inserted_records": 0,
                "message": "No new rows to append",
            }

        encoder = load_bge_m3_encoder(model_path=args.model_path, device=args.device)
        inserted = 0
        for start in range(0, len(new_rows), args.insert_batch_size):
            batch = new_rows[start : start + args.insert_batch_size]
            vectors = embed_texts(
                [row["text"] for row in batch],
                encoder=encoder,
                batch_size=min(args.embed_batch_size, len(batch)),
            )
            if vectors and len(vectors[0]) != TEXT_VECTOR_DIM:
                raise ValueError(
                    f"Unexpected embedding dim: {len(vectors[0])} != {TEXT_VECTOR_DIM}"
                )
            insert_payload = []
            for row, vector in zip(batch, vectors):
                row = dict(row)
                row["embedding"] = vector
                row.pop("_line_no", None)
                row.pop("_raw_keys", None)
                insert_payload.append(text_row_to_milvus(row))
            collection.insert(insert_payload)
            inserted += len(insert_payload)
            if inserted % 2048 == 0 or inserted == len(new_rows):
                print(f"[insert] inserted={inserted}/{len(new_rows)}", flush=True)

        collection.flush()
        collection.load()
        return {
            "validated_records": len(normalized_rows),
            "existing_records": len(existing_ids),
            "inserted_records": inserted,
            "message": "Append completed",
        }
    finally:
        connections.disconnect("default")


def main() -> None:
    args = parse_args()
    path = Path(args.input_jsonl)

    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    report = validate_jsonl(path, limit=args.limit, show_samples=args.show_samples)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if not args.append:
        print("\nValidation finished. Re-run with --append after manual confirmation.")
        return

    if not report["validation_ok"]:
        raise SystemExit("Validation failed; append aborted.")

    result = append_records(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
