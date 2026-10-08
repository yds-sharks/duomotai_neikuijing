#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无标签测试向量库查询接口（query-only）。

对齐 /mnt/data_1/yds/多模态/code/search_vector_store.py 的调用风格，
默认连接本次测试库：
/mnt/data_1/yds/多模态/data_house/test_ingest_188w_20260508_bgem3/vector_store_test_unlabeled_bgem3.db
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer


@dataclass
class SearchConfig:
    """查询配置"""

    milvus_db_path: str = (
        "/mnt/data_1/yds/多模态/data_house/"
        "test_ingest_188w_20260508_bgem3/vector_store_test_unlabeled_bgem3.db"
    )
    collection_name: str = "text_blocks"
    model_path: str = "/mnt/data_1/yds/RAG/Hybrid_milvus/总版/pretrained_models/BAAI/bge-m3"
    device: Optional[str] = None
    nprobe: int = 64


class VectorSearchEngine:
    """向量搜索引擎（无标签库）"""

    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        if self.config.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        connections.connect(alias="default", uri=self.config.milvus_db_path)
        self.collection = Collection(self.config.collection_name)
        self.collection.load()
        self.model = SentenceTransformer(self.config.model_path, device=self.device)

    def close(self) -> None:
        try:
            connections.disconnect("default")
        except Exception:
            pass

    def search(self, query: str, topk: int = 5) -> List[Dict[str, Any]]:
        """单查询"""
        query = str(query).strip()
        if not query:
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        results = self.collection.search(
            data=query_embedding.tolist(),
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": self.config.nprobe}},
            limit=max(1, int(topk)),
            output_fields=[
                "doc_name",
                "doc_id",
                "page_idx",
                "text",
                "summary",
                "block_id",
                "metadata",
                "lang",
            ],
        )

        formatted: List[Dict[str, Any]] = []
        for hits in results:
            for i, hit in enumerate(hits, 1):
                formatted.append(
                    {
                        "rank": i,
                        "score": float(hit.score),
                        "doc_name": hit.entity.get("doc_name"),
                        "doc_id": hit.entity.get("doc_id"),
                        "page_idx": hit.entity.get("page_idx"),
                        "text": hit.entity.get("text"),
                        "summary": hit.entity.get("summary"),
                        "block_id": hit.entity.get("block_id"),
                        "metadata": hit.entity.get("metadata"),
                        "lang": hit.entity.get("lang"),
                    }
                )
        return formatted

    def batch_search(self, queries: List[str], topk: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """批量查询"""
        out: Dict[str, List[Dict[str, Any]]] = {}
        for q in queries:
            try:
                out[q] = self.search(q, topk=topk)
            except Exception:
                out[q] = []
        return out


_global_engine: Optional[VectorSearchEngine] = None


def _get_global_engine() -> VectorSearchEngine:
    global _global_engine
    if _global_engine is None:
        _global_engine = VectorSearchEngine()
    return _global_engine


def search(query: str, topk: int = 5) -> List[Dict[str, Any]]:
    """便捷单查询"""
    engine = _get_global_engine()
    return engine.search(query, topk=topk)


def batch_search(queries: List[str], topk: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """便捷批量查询"""
    engine = _get_global_engine()
    return engine.batch_search(queries, topk=topk)


if __name__ == "__main__":
    demo_query = "胃ESD术后穿孔如何判断"
    rows = search(demo_query, topk=3)
    print(f"query: {demo_query}")
    for row in rows:
        print(
            f"[{row['rank']}] score={row['score']:.4f} "
            f"{row['doc_name']} (p.{row['page_idx']})"
        )
