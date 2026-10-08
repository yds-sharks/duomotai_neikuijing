#!/usr/bin/env python3
"""Shared helpers for multimodal vector-record extraction."""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, Iterable, List


DEFAULT_DB_PATH = os.environ.get(
    "SAMPLES_DB",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "assets", "multimodal_samples.db"),
)
DEFAULT_BGE_MODEL_PATH = os.environ.get("BGE_M3_PATH", "BAAI/bge-m3")
DEFAULT_QWEN3_VL_MODEL_PATH = os.environ.get(
    "QWEN3_VL_PATH", "Qwen/Qwen3-VL-8B-Instruct"
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
