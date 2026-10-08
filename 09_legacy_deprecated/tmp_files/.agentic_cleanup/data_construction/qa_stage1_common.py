#!/usr/bin/env python3
"""Common helpers for Stage-1 reliable QA data construction."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import re


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def row_get(row: Any, key: str, default: Any = None) -> Any:
    """Safe getter for sqlite3.Row / dict that tolerates missing columns."""
    try:
        keys = row.keys()
    except AttributeError:
        keys = row
    if key in keys:
        value = row[key]
        return default if value is None else value
    return default


def write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Write rows to a temp file in the same dir, then os.replace for atomicity.

    Prevents downstream steps from ever reading a half-written file if this
    process is interrupted mid-write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return count


def write_json_atomic(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def doc_split(doc_id: str, seed: int = 20260709) -> str:
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


def ensure_list(value: Any) -> List[Any]:
    parsed = parse_json(value, [])
    if isinstance(parsed, list):
        return parsed
    if parsed in (None, ""):
        return []
    return [parsed]


def clean_text(text: Any, max_len: int | None = None) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if max_len and len(cleaned) > max_len:
        return cleaned[: max_len - 3].rstrip() + "..."
    return cleaned


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def first_image(images_value: Any) -> Tuple[str, str, Dict[str, Any]]:
    images = parse_json(images_value, [])
    if not isinstance(images, list):
        return "", "", {}
    primary: Dict[str, Any] | None = None
    fallback: Dict[str, Any] | None = None
    for item in images:
        if not isinstance(item, dict):
            continue
        if fallback is None:
            fallback = item
        if item.get("is_primary") is True:
            primary = item
            break
    item = primary or fallback or {}
    return str(item.get("image_id") or ""), str(item.get("image_path") or ""), item


@lru_cache(maxsize=20000)
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


@lru_cache(maxsize=20000)
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


@lru_cache(maxsize=20000)
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


def item_image_names(item: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for key in ("img_path", "image_path", "path"):
        value = str(item.get(key) or "")
        if value:
            names.append(Path(value).name)
    return names


def infer_image_page_idx(source_path: str, image_path: str, db_page_idx: Any = None) -> Optional[int]:
    content_path = locate_content_list(source_path)
    items = load_content_items(content_path)
    image_name = Path(str(image_path or "")).name
    if items and image_name:
        for item in items:
            if item.get("type") not in {"image", "figure"}:
                continue
            if image_name in item_image_names(item) and item.get("page_idx") is not None:
                try:
                    return int(item["page_idx"])
                except Exception:
                    return None
    try:
        if db_page_idx is not None and str(db_page_idx) != "":
            return int(db_page_idx)
    except Exception:
        return None
    return None


def _append_text(chunks: List[str], value: Any) -> None:
    if value in (None, ""):
        return
    if isinstance(value, list):
        for item in value:
            _append_text(chunks, item)
        return
    if isinstance(value, dict):
        for key in ("text", "caption", "html"):
            _append_text(chunks, value.get(key))
        return
    text = clean_text(value)
    if text:
        chunks.append(text)


def item_text(item: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for key in (
        "text",
        "caption",
        "image_caption",
        "image_footnote",
        "table_caption",
        "table_footnote",
        "table_body",
    ):
        _append_text(chunks, item.get(key))
    return " ".join(chunks)


def context_for_pages(source_path: str, center_page: Optional[int], max_chars: int = 5000) -> Dict[str, Any]:
    content_path = locate_content_list(source_path)
    items = load_content_items(content_path)
    if center_page is None or not items:
        return {"content_list_path": content_path, "page_indices": [], "text": ""}

    available_pages = sorted({int(item["page_idx"]) for item in items if item.get("page_idx") is not None})
    if not available_pages:
        return {"content_list_path": content_path, "page_indices": [], "text": ""}

    pages = [center_page]
    if center_page + 1 in available_pages:
        pages.append(center_page + 1)
    elif center_page - 1 in available_pages:
        pages.insert(0, center_page - 1)
    pages = sorted(dict.fromkeys(p for p in pages if p in available_pages))

    chunks: List[str] = []
    for item in items:
        try:
            page_idx = int(item.get("page_idx"))
        except Exception:
            continue
        if page_idx not in pages:
            continue
        text = item_text(item)
        if text:
            chunks.append(f"[page {page_idx}] {text}")

    return {
        "content_list_path": content_path,
        "page_indices": pages,
        "text": clean_text("\n".join(chunks), max_len=max_chars),
    }


def find_image_item_index(source_path: str, image_path: str) -> Optional[int]:
    content_path = locate_content_list(source_path)
    items = load_content_items(content_path)
    image_name = Path(str(image_path or "")).name
    if not items or not image_name:
        return None
    for idx, item in enumerate(items):
        if item.get("type") not in {"image", "figure"}:
            continue
        if image_name in item_image_names(item):
            return idx
    return None


def context_near_image(
    source_path: str,
    image_path: str,
    before_items: int = 8,
    after_items: int = 8,
    max_chars: int = 2500,
) -> Dict[str, Any]:
    """Return local content-list text around the matched image item.

    This is preferred for API prompts because figure captions and immediate
    explanatory paragraphs are often more precise than whole-page context.
    """
    content_path = locate_content_list(source_path)
    items = load_content_items(content_path)
    idx = find_image_item_index(source_path, image_path)
    if idx is None or not items:
        return {"content_list_path": content_path, "image_item_index": None, "page_idx": None, "text": ""}

    image_item = items[idx]
    page_idx = image_item.get("page_idx")
    start = max(0, idx - max(0, before_items))
    end = min(len(items), idx + max(0, after_items) + 1)
    chunks: List[str] = []
    for pos in range(start, end):
        item = items[pos]
        text = item_text(item)
        if not text:
            continue
        item_page = item.get("page_idx", "")
        prefix = f"[item {pos} page {item_page}]"
        chunks.append(f"{prefix} {text}")

    return {
        "content_list_path": content_path,
        "image_item_index": idx,
        "page_idx": page_idx,
        "text": clean_text("\n".join(chunks), max_len=max_chars),
    }


def source_id_from_sample(sample_id: str) -> str:
    return f"src_{stable_hash(sample_id)[:12]}"


def candidate_id(source_id: str, query_type: str) -> str:
    return f"cand_{stable_hash(source_id + ':' + query_type)[:12]}"


def normalize_tags(value: Any) -> List[str]:
    tags = ensure_list(value)
    out: List[str] = []
    for tag in tags:
        text = clean_text(tag)
        if text and text not in out:
            out.append(text)
    return out


def non_generic_organs(tags: Sequence[str]) -> List[str]:
    generic = {"通用", "其他", "未知", "全身", ""}
    return [tag for tag in tags if tag not in generic]
