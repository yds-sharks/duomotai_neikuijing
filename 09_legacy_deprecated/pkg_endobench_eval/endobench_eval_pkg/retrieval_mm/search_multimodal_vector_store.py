#!/usr/bin/env python3
"""Search text_vector_index or image_vector_index with aligned embedding logic.

Design goals
------------
- Reuse the exact embedding path from build scripts:
  - text: BGE-m3 via `text_dense_module.embed_texts(...)`
  - image: Qwen3-VL image encoder via `image_dense_module.embed_images(...)`
- Support unified retrieval request fields:
  - Q (query text or image path), k, level1, level2, retrieval_db(text|image)
- Support two run modes:
  - online search (single / batch)
  - offline jsonl -> jsonl replay generation
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pymilvus import Collection, connections

try:
    from .build_multimodal_milvus import (
        DEFAULT_MILVUS_DB_PATH,
        IMAGE_COLLECTION_NAME,
        TEXT_COLLECTION_NAME,
    )
    from .common import DEFAULT_BGE_MODEL_PATH, DEFAULT_DB_PATH, DEFAULT_QWEN3_VL_MODEL_PATH
    from .image_dense_module import embed_images, load_qwen3vl_image_encoder
    from .text_dense_module import embed_texts, load_bge_m3_encoder
except ImportError:
    from build_multimodal_milvus import (
        DEFAULT_MILVUS_DB_PATH,
        IMAGE_COLLECTION_NAME,
        TEXT_COLLECTION_NAME,
    )
    from common import DEFAULT_BGE_MODEL_PATH, DEFAULT_DB_PATH, DEFAULT_QWEN3_VL_MODEL_PATH
    from image_dense_module import embed_images, load_qwen3vl_image_encoder
    from text_dense_module import embed_texts, load_bge_m3_encoder


LEVEL1_FIELD = "body_site_main"
LEVEL2_FIELD = "knowledge_type_main"


@dataclass
class SearchConfig:
    milvus_db_path: str = DEFAULT_MILVUS_DB_PATH
    main_db_path: str = DEFAULT_DB_PATH
    text_collection_name: str = TEXT_COLLECTION_NAME
    image_collection_name: str = IMAGE_COLLECTION_NAME
    text_model_path: str = DEFAULT_BGE_MODEL_PATH
    image_model_path: str = DEFAULT_QWEN3_VL_MODEL_PATH
    text_device: str = "auto"
    image_device: str = "cuda"
    image_torch_dtype: str = "auto"
    nprobe: int = 64


@dataclass
class RetrievalRequest:
    q: str
    k: int = 5
    level1: str = ""
    level2: str = ""
    retrieval_db: str = "text"  # text | image


def _pick_query(payload: Dict[str, Any]) -> str:
    for key in ("Q", "q", "query", "question"):
        value = payload.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def _pick_int(payload: Dict[str, Any], keys: Sequence[str], default: int) -> int:
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def _pick_text(payload: Dict[str, Any], keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def request_from_payload(payload: Dict[str, Any]) -> RetrievalRequest:
    request = RetrievalRequest(
        q=_pick_query(payload),
        k=_pick_int(payload, ("k", "topk"), 5),
        level1=_pick_text(payload, ("level1", "view_level1", "body_site_main", "视图level1"), ""),
        level2=_pick_text(payload, ("level2", "class_level2", "knowledge_type_main", "分类level2"), ""),
        retrieval_db=_pick_text(payload, ("retrieval_db", "db", "检索库"), "text").lower(),
    )
    return request


def _safe_expr_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _build_filter_expr(level1: str, level2: str) -> str:
    clauses: List[str] = []
    if level1:
        clauses.append(f'{LEVEL1_FIELD} == "{_safe_expr_value(level1)}"')
    if level2:
        clauses.append(f'{LEVEL2_FIELD} == "{_safe_expr_value(level2)}"')
    return " && ".join(clauses)


def _resolve_text_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class MultimodalVectorSearchEngine:
    """Search text/image vector indexes with aligned embedding logic."""

    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self.text_device = _resolve_text_device(self.config.text_device)
        self.image_device = self.config.image_device

        self._collections: Dict[str, Collection] = {}
        self._text_encoder = None
        self._image_processor = None
        self._image_model = None

        try:
            connections.connect(alias="default", uri=self.config.milvus_db_path)
        except Exception as exc:
            message = str(exc)
            if "milvus-lite is required" in message or "pkg_resources" in message:
                raise RuntimeError(
                    "Failed to connect Milvus Lite. "
                    "Please ensure the runtime has both milvus_lite and setuptools(pkg_resources). "
                    "Typical fix: pip install pymilvus[milvus_lite] setuptools"
                ) from exc
            raise

    def close(self) -> None:
        connections.disconnect("default")

    def __enter__(self) -> "MultimodalVectorSearchEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _get_collection(self, retrieval_db: str) -> Collection:
        if retrieval_db not in self._collections:
            if retrieval_db == "text":
                name = self.config.text_collection_name
            elif retrieval_db == "image":
                name = self.config.image_collection_name
            else:
                raise ValueError(f"Unsupported retrieval_db: {retrieval_db}")
            collection = Collection(name)
            collection.load()
            self._collections[retrieval_db] = collection
        return self._collections[retrieval_db]

    def _encode_text_query(self, query: str) -> List[float]:
        if self._text_encoder is None:
            self._text_encoder = load_bge_m3_encoder(
                model_path=self.config.text_model_path,
                device=self.text_device,
            )
        return embed_texts([query], encoder=self._text_encoder, batch_size=1)[0]

    def _encode_image_query(self, image_path: str) -> List[float]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image query path does not exist: {image_path}")
        if self._image_processor is None or self._image_model is None:
            self._image_processor, self._image_model = load_qwen3vl_image_encoder(
                model_path=self.config.image_model_path,
                device=self.image_device,
                torch_dtype=self.config.image_torch_dtype,
            )
        return embed_images(
            [image_path],
            processor=self._image_processor,
            model=self._image_model,
            device=self.image_device,
            pooling="mean",
        )[0]

    def _search_raw_hits(self, request: RetrievalRequest):
        if request.retrieval_db == "text":
            vector = self._encode_text_query(request.q)
            output_fields = [
                "id",
                "sample_id",
                "group_id",
                "image_ids",
                "text",
                "source_type",
                "text_role",
                "have_image",
                "body_site_main",
                "body_site_all",
                "knowledge_type_main",
                "organ_tags",
                "primary_knowledge_type",
                "secondary_knowledge_types",
                "is_general",
                "is_mixed",
                "is_key_knowledge",
                "doc_id",
                "doc_name",
                "page_idx",
                "block_id",
            ]
        elif request.retrieval_db == "image":
            vector = self._encode_image_query(request.q)
            output_fields = [
                "id",
                "image_id",
                "parent_sample_id",
                "group_id",
                "image_path",
                "image_type",
                "is_primary",
                "body_site_main",
                "body_site_all",
                "knowledge_type_main",
                "organ_tags",
                "primary_knowledge_type",
                "is_general",
                "is_mixed",
                "doc_id",
            ]
        else:
            raise ValueError(f"Unsupported retrieval_db: {request.retrieval_db}")

        expr = _build_filter_expr(request.level1, request.level2)
        search_limit = max(int(request.k), 1) * 5
        search_limit = max(search_limit, int(request.k))
        search_limit = min(search_limit, 16384)

        collection = self._get_collection(request.retrieval_db)
        res = collection.search(
            data=[vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": int(self.config.nprobe)}},
            limit=search_limit,
            expr=expr or None,
            output_fields=output_fields,
        )
        return res[0], expr

    def _fetch_parent_sample_meta(self, sample_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        uniq = [str(x) for x in dict.fromkeys(sample_ids) if str(x)]
        if not uniq:
            return {}

        conn = sqlite3.connect(self.config.main_db_path)
        conn.row_factory = sqlite3.Row
        try:
            placeholders = ",".join(["?"] * len(uniq))
            sql = (
                "SELECT sample_id, group_id, doc_id, doc_name, page_idx, text "
                f"FROM multimodal_samples WHERE sample_id IN ({placeholders})"
            )
            rows = conn.execute(sql, uniq).fetchall()
            return {str(row["sample_id"]): dict(row) for row in rows}
        finally:
            conn.close()

    @staticmethod
    def _hit_score(hit: Any) -> float:
        if hasattr(hit, "score"):
            return float(hit.score)
        if hasattr(hit, "distance"):
            return float(hit.distance)
        return 0.0

    def _format_text_hits(self, hits: Iterable[Any], *, k: int) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        seen_sample_ids = set()

        for hit in hits:
            entity = hit.entity
            sample_id = str(entity.get("sample_id") or "")
            if not sample_id or sample_id in seen_sample_ids:
                continue
            seen_sample_ids.add(sample_id)

            formatted.append(
                {
                    "rank": 0,
                    "score": self._hit_score(hit),
                    "retrieval_db": "text",
                    "sample_id": sample_id,
                    "group_id": entity.get("group_id"),
                    "doc_id": entity.get("doc_id"),
                    "doc_name": entity.get("doc_name"),
                    "page_idx": entity.get("page_idx"),
                    "block_id": entity.get("block_id"),
                    "level1": entity.get(LEVEL1_FIELD),
                    "level2": entity.get(LEVEL2_FIELD),
                    "content": entity.get("text"),
                    "text_role": entity.get("text_role"),
                    "image_ids": entity.get("image_ids"),
                    "source_type": entity.get("source_type"),
                }
            )
            if len(formatted) >= k:
                break

        for idx, item in enumerate(formatted, start=1):
            item["rank"] = idx
        return formatted

    def _format_image_hits(self, hits: Iterable[Any], *, k: int) -> List[Dict[str, Any]]:
        staged: List[Dict[str, Any]] = []
        seen_sample_ids = set()
        parent_sample_ids: List[str] = []

        for hit in hits:
            entity = hit.entity
            sample_id = str(entity.get("parent_sample_id") or "")
            if not sample_id or sample_id in seen_sample_ids:
                continue
            seen_sample_ids.add(sample_id)
            parent_sample_ids.append(sample_id)

            staged.append(
                {
                    "rank": 0,
                    "score": self._hit_score(hit),
                    "retrieval_db": "image",
                    "sample_id": sample_id,
                    "group_id": entity.get("group_id"),
                    "doc_id": entity.get("doc_id"),
                    "doc_name": None,
                    "page_idx": None,
                    "level1": entity.get(LEVEL1_FIELD),
                    "level2": entity.get(LEVEL2_FIELD),
                    "content": "",
                    "image_id": entity.get("image_id"),
                    "image_path": entity.get("image_path"),
                    "image_type": entity.get("image_type"),
                    "is_primary": entity.get("is_primary"),
                }
            )
            if len(staged) >= k:
                break

        parent_map = self._fetch_parent_sample_meta(parent_sample_ids)
        for item in staged:
            parent = parent_map.get(item["sample_id"], {})
            item["group_id"] = item.get("group_id") or parent.get("group_id")
            item["doc_id"] = item.get("doc_id") or parent.get("doc_id")
            item["doc_name"] = parent.get("doc_name")
            item["page_idx"] = parent.get("page_idx")
            item["content"] = parent.get("text") or ""

        for idx, item in enumerate(staged, start=1):
            item["rank"] = idx
        return staged

    def search(self, request: RetrievalRequest) -> Dict[str, Any]:
        if request.k <= 0:
            raise ValueError(f"k must be > 0, got {request.k}")
        if not request.q or not str(request.q).strip():
            raise ValueError("Q must be non-empty")

        request = RetrievalRequest(
            q=str(request.q).strip(),
            k=int(request.k),
            level1=str(request.level1 or "").strip(),
            level2=str(request.level2 or "").strip(),
            retrieval_db=str(request.retrieval_db or "").strip().lower(),
        )

        if request.retrieval_db not in ("text", "image"):
            raise ValueError("retrieval_db must be one of: text, image")

        hits, expr = self._search_raw_hits(request)
        if request.retrieval_db == "text":
            results = self._format_text_hits(hits, k=request.k)
        else:
            results = self._format_image_hits(hits, k=request.k)

        return {
            "request": asdict(request),
            "applied_conditions": {
                "level1_field": LEVEL1_FIELD,
                "level2_field": LEVEL2_FIELD,
                "level1": request.level1,
                "level2": request.level2,
                "expr": expr,
                "dedup_key": "sample_id",
                "nprobe": int(self.config.nprobe),
            },
            "results": results,
            "hit_count": len(results),
        }

    def batch_search(self, requests: Sequence[RetrievalRequest]) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for idx, req in enumerate(requests):
            try:
                outputs.append(self.search(req))
            except Exception as exc:
                outputs.append(
                    {
                        "request": asdict(req),
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "index": idx,
                        },
                        "results": [],
                        "hit_count": 0,
                    }
                )
        return outputs

    def run_offline_jsonl(self, input_jsonl: str, output_jsonl: str) -> int:
        requests: List[RetrievalRequest] = []
        raw_records: List[Dict[str, Any]] = []

        with open(input_jsonl, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid json at line {line_no}: {input_jsonl}") from exc
                raw_records.append(payload)
                requests.append(request_from_payload(payload))

        outputs = self.batch_search(requests)

        out_dir = os.path.dirname(output_jsonl)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_jsonl, "w", encoding="utf-8") as handle:
            for idx, (raw, out) in enumerate(zip(raw_records, outputs)):
                row = {
                    "query_idx": idx,
                    "input": raw,
                    "request": out.get("request", {}),
                    "applied_conditions": out.get("applied_conditions", {}),
                    "results": out.get("results", []),
                    "hit_count": out.get("hit_count", 0),
                }
                if "error" in out:
                    row["error"] = out["error"]
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search multimodal Milvus indexes")
    parser.add_argument("--milvus-db-path", default=DEFAULT_MILVUS_DB_PATH)
    parser.add_argument("--main-db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--text-model-path", default=DEFAULT_BGE_MODEL_PATH)
    parser.add_argument("--image-model-path", default=DEFAULT_QWEN3_VL_MODEL_PATH)
    parser.add_argument("--text-device", default="auto")
    parser.add_argument("--image-device", default="cuda")
    parser.add_argument("--image-torch-dtype", default="auto")
    parser.add_argument("--nprobe", type=int, default=64)

    parser.add_argument("--q", default="", help="Query text (for text db) or image path (for image db)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--level1", default="")
    parser.add_argument("--level2", default="")
    parser.add_argument("--retrieval-db", default="text", choices=("text", "image"))

    parser.add_argument("--input-jsonl", default="", help="Batch input jsonl with request fields")
    parser.add_argument("--output-jsonl", default="", help="Batch output jsonl path")
    parser.add_argument("--pretty", action="store_true", help="Pretty print single-query output")
    return parser.parse_args()


def _build_engine_from_args(args: argparse.Namespace) -> MultimodalVectorSearchEngine:
    config = SearchConfig(
        milvus_db_path=args.milvus_db_path,
        main_db_path=args.main_db_path,
        text_model_path=args.text_model_path,
        image_model_path=args.image_model_path,
        text_device=args.text_device,
        image_device=args.image_device,
        image_torch_dtype=args.image_torch_dtype,
        nprobe=int(args.nprobe),
    )
    return MultimodalVectorSearchEngine(config=config)


def main() -> None:
    args = parse_args()
    with _build_engine_from_args(args) as engine:
        if args.input_jsonl:
            if not args.output_jsonl:
                raise ValueError("--output-jsonl is required when --input-jsonl is set")
            total = engine.run_offline_jsonl(args.input_jsonl, args.output_jsonl)
            print(f"[done] offline retrieval rows={total} output={args.output_jsonl}")
            return

        request = RetrievalRequest(
            q=args.q,
            k=args.k,
            level1=args.level1,
            level2=args.level2,
            retrieval_db=args.retrieval_db,
        )
        out = engine.search(request)
        if args.pretty:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
