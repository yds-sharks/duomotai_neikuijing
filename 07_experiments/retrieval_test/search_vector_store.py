#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量库 / 检索结果回放批量查询接口

默认行为：
1. 若设置环境变量 `RETRIEVAL_REPLAY_JSONL`，优先从指定 JSONL 回放检索结果
2. 否则回退到原来的 Milvus 实时检索

这样可以在不改 prompt 和评测主流程的前提下，直接复用离线检索结果跑 RAG。
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

sys.path.insert(0, "/mnt/data_1/yds/多模态/insert")


def _normalize_query(text: str) -> str:
    return " ".join(str(text or "").split())


def _clone_doc(doc: Dict[str, Any], rank: int) -> Dict[str, Any]:
    cloned = dict(doc)
    cloned["rank"] = rank
    return cloned


@dataclass
class SearchConfig:
    """查询配置"""

    milvus_db_path: str = "/mnt/data_1/yds/多模态/data/vector_store.db"
    collection_name: str = "text_blocks"
    model_path: str = "/mnt/data_1/yds/RAG/Hybrid_milvus/总版/pretrained_models/BAAI/bge-m3"
    device: Optional[str] = None
    nprobe: int = 64
    replay_jsonl: Optional[str] = None


class ReplaySearchEngine:
    """从离线 JSONL 回放检索结果。"""

    def __init__(self, replay_jsonl: str):
        self.replay_jsonl = replay_jsonl
        self._query_to_docs: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    def _register(self, query: str, docs: List[Dict[str, Any]]) -> None:
        normalized = _normalize_query(query)
        if not normalized or normalized in self._query_to_docs:
            return
        self._query_to_docs[normalized] = [dict(doc) for doc in docs]

    def _load(self) -> None:
        with open(self.replay_jsonl, "r", encoding="utf-8") as file_obj:
            for line_no, line in enumerate(file_obj, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Failed to parse retrieval replay JSONL at line {line_no}: {self.replay_jsonl}"
                    ) from exc

                docs = record.get("retrieval_docs") or []
                for key in ("query_used", "retrieval_query_rewritten", "retrieval_query"):
                    value = record.get(key)
                    if value:
                        self._register(str(value), docs)

        if not self._query_to_docs:
            raise ValueError(f"No replayable retrieval entries found in {self.replay_jsonl}")

    def search(self, query: str, topk: int = 5) -> List[Dict[str, Any]]:
        docs = self._query_to_docs.get(_normalize_query(query), [])
        return [_clone_doc(doc, rank=i) for i, doc in enumerate(docs[:topk], start=1)]

    def batch_search(self, queries: List[str], topk: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        return {query: self.search(query, topk=topk) for query in queries}


class VectorSearchEngine:
    """Milvus 向量搜索引擎。"""

    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()

        import torch
        from pymilvus import Collection, connections
        from sentence_transformers import SentenceTransformer

        if self.config.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        connections.connect(alias="default", uri=self.config.milvus_db_path)
        self.collection = Collection(self.config.collection_name)
        self.collection.load()
        self.model = SentenceTransformer(self.config.model_path, device=self.device)

    def search(self, query: str, topk: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        results = self.collection.search(
            data=query_embedding.tolist(),
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": self.config.nprobe}},
            limit=topk,
            output_fields=[
                "doc_name",
                "doc_id",
                "page_idx",
                "text",
                "summary",
                "block_id",
                "metadata",
            ],
        )

        formatted_results = []
        for hits in results:
            for i, hit in enumerate(hits, 1):
                formatted_results.append(
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
                    }
                )
        return formatted_results

    def batch_search(self, queries: List[str], topk: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        results = {}
        for query in queries:
            try:
                results[query] = self.search(query, topk)
            except Exception as exc:
                print(f"查询失败 '{query[:30]}...': {exc}")
                results[query] = []
        return results


_global_engine: Optional[object] = None


def _build_engine() -> object:
    replay_jsonl = os.environ.get("RETRIEVAL_REPLAY_JSONL")
    if replay_jsonl:
        return ReplaySearchEngine(replay_jsonl)
    return VectorSearchEngine()


def _get_global_engine():
    global _global_engine
    if _global_engine is None:
        _global_engine = _build_engine()
    return _global_engine


def search(query: str, topk: int = 5) -> List[Dict[str, Any]]:
    engine = _get_global_engine()
    return engine.search(query, topk)


def batch_search(queries: List[str], topk: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    engine = _get_global_engine()
    return engine.batch_search(queries, topk)


if __name__ == "__main__":
    replay_path = os.environ.get("RETRIEVAL_REPLAY_JSONL")
    if replay_path:
        print(f"Replay mode enabled: {replay_path}")
    results = search("胃ESD治疗方法", topk=3)
    for result in results:
        print(f"[{result['rank']}] {result.get('doc_name')} (p.{result.get('page_idx')})")
