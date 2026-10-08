#!/usr/bin/env python3
"""First-stage retrieval adapter for the agentic RAG runtime.

This module wraps the existing multimodal Milvus search engine without copying
its model/index logic. It exposes one stable call:

    retrieve(query_text, query_image_path) -> {text, image, combined}

v0.3 新增：exclude_ids 参数，用于在**召回层**抑制已见/dropped 的证据 ID，
使多轮检索单调探索新证据。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

CODE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CODE_DIR / "agentic_runtime_config.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = Path(config.get("project_root") or config_path.parent.parent)
    if not project_root.is_absolute():
        project_root = (config_path.parent / project_root).resolve()
    config["project_root"] = str(project_root)

    for section, keys in {
        "paths": ("qa_gold_path", "output_dir"),
        "routing": ("router_checkpoint",),
    }.items():
        values = config.get(section, {})
        for key in keys:
            value = values.get(key)
            if value and not Path(value).is_absolute():
                values[key] = str((project_root / value).resolve())
    return config


def load_search_module(search_dir: str):
    search_dir_path = Path(search_dir)
    module_path = search_dir_path / "search_multimodal_vector_store.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing search module: {module_path}")
    if str(search_dir_path) not in sys.path:
        sys.path.insert(0, str(search_dir_path))
    spec = importlib.util.spec_from_file_location("agentic_multimodal_search", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load search module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FirstStageRetriever:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        paths = self.config.get("paths", {})
        retrieval_cfg = self.config.get("retrieval", {})
        required = {
            "paths.multimodal_search_dir": paths.get("multimodal_search_dir"),
            "paths.main_db_path": paths.get("main_db_path"),
            "retrieval.milvus_db_path": retrieval_cfg.get("milvus_db_path"),
            "retrieval.text_model_path": retrieval_cfg.get("text_model_path"),
            "retrieval.image_model_path": retrieval_cfg.get("image_model_path"),
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing retrieval config keys: {', '.join(missing)}")
        module = load_search_module(paths["multimodal_search_dir"])
        self._module = module
        search_config = module.SearchConfig(
            milvus_db_path=retrieval_cfg.get("milvus_db_path"),
            main_db_path=paths.get("main_db_path"),
            text_model_path=retrieval_cfg.get("text_model_path"),
            image_model_path=retrieval_cfg.get("image_model_path"),
            text_device=retrieval_cfg.get("text_device", "auto"),
            image_device=retrieval_cfg.get("image_device", "cuda"),
            image_torch_dtype=retrieval_cfg.get("image_torch_dtype", "auto"),
            nprobe=int(retrieval_cfg.get("nprobe", 64)),
        )
        self.engine = module.MultimodalVectorSearchEngine(config=search_config)
        self.text_k = int(retrieval_cfg.get("first_stage_text_k", 20))
        self.image_k = int(retrieval_cfg.get("first_stage_image_k", 20))

    def close(self) -> None:
        close = getattr(self.engine, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "FirstStageRetriever":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _normalize_hit(hit: Dict[str, Any], source: str) -> Dict[str, Any]:
        return {
            "source": source,
            "rank": int(hit.get("rank") or 0),
            "score": float(hit.get("score") or 0.0),
            "text": str(hit.get("content") or hit.get("text") or ""),
            "content": str(hit.get("content") or hit.get("text") or ""),
            "doc_id": str(hit.get("doc_id") or ""),
            "doc_name": str(hit.get("doc_name") or ""),
            "page_idx": hit.get("page_idx"),
            "block_id": hit.get("block_id"),
            "sample_id": str(hit.get("sample_id") or ""),
            "group_id": str(hit.get("group_id") or ""),
            "image_path": str(hit.get("image_path") or ""),
            "image_id": str(hit.get("image_id") or ""),
            "raw": hit,
        }

    @staticmethod
    def _get_hit_id(hit: Dict[str, Any]) -> str:
        """提取证据 ID 用于 suppression 比较。"""
        return str(hit.get("sample_id") or hit.get("doc_id") or "")

    def _filter_excluded(self, hits: List[Dict[str, Any]], exclude_ids: Optional[Set[str]]) -> List[Dict[str, Any]]:
        """在召回层过滤掉已被抑制的证据 ID。"""
        if not exclude_ids:
            return hits
        return [h for h in hits if self._get_hit_id(h) not in exclude_ids]

    def search_text(self, query_text: str, *, k: Optional[int] = None, level1: str = "", level2: str = "", exclude_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        if not query_text or not str(query_text).strip():
            return []
        # 多取一些，过滤掉 exclude 后仍可能有足够数量
        fetch_k = int(k or self.text_k)
        if exclude_ids:
            fetch_k += len(exclude_ids)
        req = self._module.RetrievalRequest(
            q=str(query_text).strip(),
            k=fetch_k,
            level1=level1,
            level2=level2,
            retrieval_db="text",
        )
        out = self.engine.search(req)
        hits = [self._normalize_hit(hit, "text") for hit in out.get("results", [])]
        return self._filter_excluded(hits, exclude_ids)[:int(k or self.text_k)]

    def search_image(self, image_path: str, *, k: Optional[int] = None, level1: str = "", level2: str = "", exclude_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        if not image_path or not Path(image_path).exists():
            return []
        fetch_k = int(k or self.image_k)
        if exclude_ids:
            fetch_k += len(exclude_ids)
        req = self._module.RetrievalRequest(
            q=str(image_path),
            k=fetch_k,
            level1=level1,
            level2=level2,
            retrieval_db="image",
        )
        out = self.engine.search(req)
        hits = [self._normalize_hit(hit, "image") for hit in out.get("results", [])]
        return self._filter_excluded(hits, exclude_ids)[:int(k or self.image_k)]

    def retrieve(
        self,
        query_text: str,
        query_image_path: str = "",
        *,
        text_k: Optional[int] = None,
        image_k: Optional[int] = None,
        level1: str = "",
        level2: str = "",
        exclude_ids: Optional[Set[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        text_hits = []
        image_hits = []
        if text_k is None or int(text_k) > 0:
            text_hits = self.search_text(query_text, k=text_k, level1=level1, level2=level2, exclude_ids=exclude_ids)
        if query_image_path and (image_k is None or int(image_k) > 0):
            image_hits = self.search_image(query_image_path, k=image_k, level1=level1, level2=level2, exclude_ids=exclude_ids)
        combined = text_hits + image_hits
        return {"text": text_hits, "image": image_hits, "combined": combined}
