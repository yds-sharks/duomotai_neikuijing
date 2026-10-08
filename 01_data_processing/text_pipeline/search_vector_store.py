#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量库批量查询接口

配置好的向量库查询工具，支持批量查询

用法:
    from search_vector_store import batch_search, SearchConfig
    
    # 批量查询
    queries = [
        "胃ESD治疗方法",
        "什么是早期胃癌",
        "结肠镜检查注意事项"
    ]
    results = batch_search(queries, topk=5)
    
    # 遍历每个查询的结果
    for query, docs in results.items():
        print(f"\n查询: {query}")
        for doc in docs:
            print(f"  [{doc['rank']}] {doc['doc_name']} (p.{doc['page_idx']}) - {doc['score']:.4f}")
"""

import sys
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass

sys.path.insert(0, "/mnt/data_1/yds/多模态/insert")

from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import torch


# ==================== 配置 ====================

@dataclass
class SearchConfig:
    """查询配置"""
    milvus_db_path: str = "/mnt/data_1/yds/多模态/data/vector_store.db"
    collection_name: str = "text_blocks"
    model_path: str = "/mnt/data_1/yds/RAG/Hybrid_milvus/总版/pretrained_models/BAAI/bge-m3"
    device: Optional[str] = None
    nprobe: int = 64


# ==================== 核心类 ====================

class VectorSearchEngine:
    """向量搜索引擎"""
    
    def __init__(self, config: Optional[SearchConfig] = None):
        """
        初始化搜索引擎
        
        Args:
            config: 查询配置，默认使用内置配置
        """
        self.config = config or SearchConfig()
        
        # 确定设备
        if self.config.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
        
        # 连接 Milvus
        connections.connect(alias="default", uri=self.config.milvus_db_path)
        self.collection = Collection(self.config.collection_name)
        self.collection.load()
        
        # 加载模型
        self.model = SentenceTransformer(self.config.model_path, device=self.device)
    
    def search(self, query: str, topk: int = 5) -> List[Dict[str, Any]]:
        """
        单查询
        
        Args:
            query: 查询文本
            topk: 返回结果数量
            
        Returns:
            结果列表，按相似度降序排列
            每个元素: {
                'rank': int,
                'score': float,
                'doc_name': str,
                'doc_id': str,
                'page_idx': int,
                'text': str,
                'summary': str,
                'block_id': int,
                'metadata': dict
            }
        """
        # 编码查询
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # 执行搜索
        results = self.collection.search(
            data=query_embedding.tolist(),
            anns_field="dense_vector",
            param={
                "metric_type": "IP",
                "params": {"nprobe": self.config.nprobe}
            },
            limit=topk,
            output_fields=[
                "doc_name", "doc_id", "page_idx", 
                "text", "summary", "block_id", "metadata"
            ],
        )
        
        # 格式化结果
        formatted_results = []
        for hits in results:
            for i, hit in enumerate(hits, 1):
                formatted_results.append({
                    "rank": i,
                    "score": float(hit.score),
                    "doc_name": hit.entity.get("doc_name"),
                    "doc_id": hit.entity.get("doc_id"),
                    "page_idx": hit.entity.get("page_idx"),
                    "text": hit.entity.get("text"),
                    "summary": hit.entity.get("summary"),
                    "block_id": hit.entity.get("block_id"),
                    "metadata": hit.entity.get("metadata"),
                })
        
        return formatted_results
    
    def batch_search(
        self, 
        queries: List[str], 
        topk: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量查询
        
        Args:
            queries: 查询列表
            topk: 每个查询返回结果数
            
        Returns:
            字典: {query: [结果列表]}
        """
        results = {}
        for query in queries:
            try:
                results[query] = self.search(query, topk)
            except Exception as e:
                print(f"查询失败 '{query[:30]}...': {e}")
                results[query] = []
        return results


# ==================== 便捷函数 ====================

_global_engine: Optional[VectorSearchEngine] = None

def _get_global_engine() -> VectorSearchEngine:
    """获取全局引擎实例"""
    global _global_engine
    if _global_engine is None:
        _global_engine = VectorSearchEngine()
    return _global_engine


def search(query: str, topk: int = 5) -> List[Dict[str, Any]]:
    """便捷单查询"""
    engine = _get_global_engine()
    return engine.search(query, topk)


def batch_search(queries: List[str], topk: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """便捷批量查询"""
    engine = _get_global_engine()
    return engine.batch_search(queries, topk)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 示例1: 单查询
    print("示例1 - 单查询:")
    results = search("胃ESD治疗方法", topk=3)
    for r in results:
        print(f"[{r['rank']}] {r['doc_name']} (p.{r['page_idx']}) - {r['score']:.4f}")
    
    # 示例2: 批量查询
    print("\n示例2 - 批量查询:")
    queries = ["什么是早期胃癌", "结肠镜检查"]
    batch_results = batch_search(queries, topk=2)
    for query, docs in batch_results.items():
        print(f"\n查询: {query}")
        for doc in docs:
            print(f"  [{doc['rank']}] {doc['doc_name']}: {doc['summary'][:50]}...")
