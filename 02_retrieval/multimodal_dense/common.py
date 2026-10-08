#!/usr/bin/env python3
"""Shared helpers for multimodal vector-record extraction."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Iterable, List


DEFAULT_DB_PATH = "/mnt/data_1/yds/多模态/data_house/multimodal_samples.db"
DEFAULT_BGE_MODEL_PATH = "/mnt/data_1/yds/RAG/Hybrid_milvus/总版/pretrained_models/BAAI/bge-m3"
DEFAULT_QWEN3_VL_MODEL_PATH = (
    "/mnt/data_10/mwx/huggingface_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
    "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)


def parse_json_field(raw: Any, default: Any) -> Any:
    if raw in (None, ""):
        return default
    if isinstance(raw, (dict, list)):
        return raw
    return json.loads(raw)


def load_row_json_fields(row: sqlite3.Row) -> Dict[str, Any]:
    record = dict(row)
    for field, default in (
        ("organ_tags", []),
        ("secondary_knowledge_types", []),
        ("labels", {}),
        ("retrieval_flags", {}),
        ("retrieval_meta", {}),
        ("images", []),
    ):
        record[field] = parse_json_field(record.get(field), default)
    return record


def load_db_rows(
    db_path: str,
    query: str,
    params: Iterable[Any] | None = None,
) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(query, tuple(params or ()))
        return [load_row_json_fields(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def image_ids_from_images(images: List[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for image in images or []:
        image_id = str(image.get("image_id") or "").strip()
        if image_id:
            ids.append(image_id)
    return ids
