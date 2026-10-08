#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码二：build_vector_store.py

从中央文本数据库读取纯文本数据，构建 Milvus 向量库
使用 BM25 + BGE-M3 混合检索架构（复用 insert/bm25_bge_vectorstore_v2）

用法:
    # 1. 构建语料统计（仅需一次）
    python build_vector_store.py \
        --db /mnt/data_1/yds/多模态/data/text_database.db \
        --milvus-db /mnt/data_1/yds/多模态/data/vector_store.db \
        --mode stats \
        --stats-dir ./corpus_stats

    # 2. 构建 Milvus 向量库（全量）
    python build_vector_store.py \
        --db /mnt/data_1/yds/多模态/data/text_database.db \
        --milvus-db /mnt/data_1/yds/多模态/data/vector_store.db \
        --mode build \
        --stats-dir ./corpus_stats \
        --collection text_blocks

    # 3. 从特定文档构建（增量）
    python build_vector_store.py \
        --db /mnt/data_1/yds/多模态/data/text_database.db \
        --milvus-db /mnt/data_1/yds/多模态/data/vector_store.db \
        --mode build \
        --doc-id a0281c14 \
        --stats-dir ./corpus_stats

    # 4. 查询测试
    python build_vector_store.py \
        --milvus-db /mnt/data_1/yds/多模态/data/vector_store.db \
        --mode query \
        --collection text_blocks \
        --query "胃ESD治疗方法"
"""

import os
import sys
import json
import sqlite3
import argparse
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
from tqdm import tqdm
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from sentence_transformers import SentenceTransformer

# 复用现有的成熟组件
sys.path.insert(0, "/mnt/data_1/yds/多模态/insert")
from bm25_bge_vectorstore_v2.tokenizers import Tokenizer
from bm25_bge_vectorstore_v2.dense_embedder import BGEDense
from bm25_bge_vectorstore_v2.milvus_client import MilvusClient


# ============ 中央数据库访问层 ============

class TextDBReader:
    """读取中央文本数据库"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def get_text_blocks(
        self,
        doc_id: Optional[str] = None,
        min_length: int = 10,
        max_length: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """获取文本块迭代器"""
        with self._connect() as conn:
            sql = """
                SELECT b.*, d.doc_name
                FROM text_blocks b
                JOIN documents d ON b.doc_id = d.doc_id
                WHERE b.char_length >= ?
            """
            params = [min_length]

            if max_length:
                sql += " AND b.char_length <= ?"
                params.append(max_length)

            if doc_id:
                sql += " AND b.doc_id = ?"
                params.append(doc_id)

            sql += " ORDER BY b.block_id"

            if limit:
                sql += f" LIMIT {int(limit)}"

            cursor = conn.execute(sql, params)
            for row in cursor:
                yield {
                    "block_id": row["block_id"],
                    "doc_id": row["doc_id"],
                    "doc_name": row["doc_name"],
                    "page_idx": row["page_idx"],
                    "content_hash": row["content_hash"],
                    "text": row["text"],
                    "bbox": json.loads(row["bbox"]) if row["bbox"] else [],
                    "char_length": row["char_length"],
                }

    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计"""
        with self._connect() as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            block_count = conn.execute("SELECT COUNT(*) FROM text_blocks").fetchone()[0]
            return {
                "documents": doc_count,
                "text_blocks": block_count,
            }

    def get_doc_ids(self) -> List[str]:
        """获取所有文档ID"""
        with self._connect() as conn:
            cursor = conn.execute("SELECT doc_id FROM documents")
            return [row[0] for row in cursor]


# ============ BM25 统计构建 ============

class BM25StatsBuilder:
    """从中央库构建 BM25 语料统计"""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.df = defaultdict(int)
        self.doc_lens = []
        self.token2id = {}  # 动态构建词表
        self.next_id = 0

    def _get_token_id(self, token: str) -> int:
        """获取或创建 token ID"""
        if token not in self.token2id:
            self.token2id[token] = self.next_id
            self.next_id += 1
        return self.token2id[token]

    def add_document(self, text: str):
        """添加文档文本，计算词频统计"""
        tokens = self.tokenizer.tokenize_mixed(text)
        self.doc_lens.append(len(tokens))

        # 统计文档频率（基于 token 字符串）
        unique_tokens = set(tokens)
        for tok in unique_tokens:
            self.df[tok] += 1  # 直接用 token 字符串作为 key
            if tok not in self.token2id:
                self.token2id[tok] = self.next_id
                self.next_id += 1

    def build(self) -> Dict:
        """生成 BM25 统计"""
        N = len(self.doc_lens)
        avgdl = sum(self.doc_lens) / max(N, 1)

        return {
            "N": N,
            "avgdl": avgdl,
            "df": dict(self.df),  # 保持字典格式
            "token2id": self.token2id,
            "vocab_size": len(self.token2id),
        }


# ============ Milvus 向量库构建 ============

def create_collection(
    milvus_db: str,
    collection_name: str,
    dim: int,
    overwrite: bool = False
) -> Collection:
    """创建 Milvus 集合"""
    from pymilvus import utility
    connections.connect(alias="default", uri=milvus_db)

    if overwrite and collection_name in utility.list_collections():
        print(f"删除已存在集合: {collection_name}")
        Collection(collection_name).drop()

    if collection_name in utility.list_collections():
        print(f"使用已存在集合: {collection_name}")
        return Collection(collection_name)

    # 定义 Schema
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="block_id", dtype=DataType.INT64),  # 关联中央库
        FieldSchema(name="text_hash", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="page_idx", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="lang", dtype=DataType.VARCHAR, max_length=8),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]

    schema = CollectionSchema(fields, description="Text blocks from central DB")
    collection = Collection(name=collection_name, schema=schema)

    # 创建稀疏索引
    try:
        collection.create_index(
            field_name="sparse_vector",
            index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
            },
        )
        print("稀疏索引创建成功")
    except Exception as e:
        print(f"稀疏索引: {e}")

    # 创建稠密索引
    nlist = min(4096, max(1024, dim))
    collection.create_index(
        field_name="dense_vector",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": nlist},
        },
    )
    print(f"稠密索引创建成功 (IVF_FLAT, nlist={nlist})")

    collection.load()
    return collection


def build_vectors_from_db(
    text_db_path: str,
    milvus_db_path: str,
    collection_name: str,
    stats_dir: str,
    doc_id: Optional[str] = None,
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 64,
    min_length: int = 10,
):
    """从中央库构建向量库"""

    # 初始化读取器
    reader = TextDBReader(text_db_path)
    db_stats = reader.get_stats()
    print(f"中央库统计: 文档 {db_stats['documents']}, 文本块 {db_stats['text_blocks']}")

    # 加载或构建 BM25 统计
    stats_path = os.path.join(stats_dir, "corpus_stats.json")
    if os.path.exists(stats_path):
        print(f"加载已有统计: {stats_path}")
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        tokenizer = Tokenizer(
            bm25_vocab_path=None,
            jieba_userdict_path=None,
            stopwords_zh_path=None,
            stopwords_en_path=None,
        )
        tokenizer.token2id = stats["token2id"]
    else:
        print("构建 BM25 语料统计...")
        os.makedirs(stats_dir, exist_ok=True)
        tokenizer = Tokenizer(
            bm25_vocab_path=None,
            jieba_userdict_path=None,
            stopwords_zh_path=None,
            stopwords_en_path=None,
        )
        builder = BM25StatsBuilder(tokenizer)

        for block in tqdm(reader.get_text_blocks(min_length=min_length), desc="Tokenizing"):
            builder.add_document(block["text"])

        stats = builder.build()
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"统计已保存: {stats_path}")

    print(f"语料统计: N={stats['N']}, avgdl={stats['avgdl']:.2f}, vocab={stats['vocab_size']}")

    # 初始化编码器
    print(f"\n加载模型: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = BGEDense(model_name, device=device)
    dim = embedder.model.get_sentence_embedding_dimension()
    print(f"设备: {device}, 维度: {dim}")

    # 创建集合
    collection = create_collection(milvus_db_path, collection_name, dim, overwrite=True)

    # 加载 BM25 向量化器
    from bm25_bge_vectorstore_v2.bm25_vectorizer import BM25Vectorizer
    vectorizer = BM25Vectorizer(
        df=stats["df"],
        N=stats["N"],
        avgdl=stats["avgdl"],
        token2id=stats["token2id"],
        k1=1.2,
        b=0.75,
    )

    # 批量插入
    print(f"\n开始插入数据...")
    inserted = 0
    failed = 0

    batch_rows = []
    batch_texts = []

    for block in tqdm(
        reader.get_text_blocks(doc_id=doc_id, min_length=min_length),
        desc="Building vectors"
    ):
        text = block["text"]

        # BM25 稀疏向量
        tokens = tokenizer.tokenize_mixed(text)
        sparse_raw = vectorizer.vectorize_doc(tokens)
        sparse_map = {int(i): float(v) for i, v in zip(sparse_raw["indices"], sparse_raw["values"]) if v}

        if not sparse_map:
            failed += 1
            continue

        row = {
            "block_id": block["block_id"],
            "text_hash": block["content_hash"][:64],
            "doc_id": block["doc_id"],
            "doc_name": block["doc_name"],
            "page_idx": block["page_idx"],
            "text": text[:8192],
            "summary": text[:512],
            "metadata": {
                "bbox": block["bbox"],
                "char_length": block["char_length"],
            },
            "lang": "zh" if any('\u4e00' <= c <= '\u9fff' for c in text[:100]) else "en",
            "sparse_vector": sparse_map,
        }

        batch_rows.append(row)
        batch_texts.append(text)

        # 批量处理
        if len(batch_rows) >= batch_size:
            try:
                # 稠密编码
                embeddings = embedder.encode_batch(batch_texts)
                for r, emb in zip(batch_rows, embeddings):
                    r["dense_vector"] = emb.tolist()

                # 插入
                collection.insert(batch_rows)
                inserted += len(batch_rows)

                # 每 10 批 flush
                if (inserted // batch_size) % 10 == 0:
                    collection.flush()

            except Exception as e:
                print(f"批次插入失败: {e}")
                failed += len(batch_rows)

            batch_rows = []
            batch_texts = []

    # 处理剩余批次
    if batch_rows:
        try:
            embeddings = embedder.encode_batch(batch_texts)
            for r, emb in zip(batch_rows, embeddings):
                r["dense_vector"] = emb.tolist()
            collection.insert(batch_rows)
            inserted += len(batch_rows)
        except Exception as e:
            print(f"尾部插入失败: {e}")
            failed += len(batch_rows)

    collection.flush()

    print(f"\n{'='*60}")
    print("构建完成!")
    print(f"数据库: {milvus_db_path}")
    print(f"集合: {collection_name}")
    print(f"插入成功: {inserted}")
    print(f"插入失败: {failed}")
    print(f"集合总数: {collection.num_entities}")
    print(f"{'='*60}")


def query_vector_store(
    milvus_db: str,
    collection_name: str,
    query: str,
    model_name: str = "BAAI/bge-m3",
    topk: int = 5,
    alpha: float = 0.6,  # 稠密权重
):
    """查询向量库"""
    print(f"\n查询: {query}")

    # 连接 Milvus
    connections.connect(alias="default", uri=milvus_db)
    collection = Collection(collection_name)
    collection.load()

    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = BGEDense(model_name, device=device)

    # 加载 tokenizer（简化版）
    from bm25_bge_vectorstore_v2.tokenizers import Tokenizer
    tokenizer = Tokenizer(
        bm25_vocab_path=None,
        jieba_userdict_path=None,
        stopwords_zh_path=None,
        stopwords_en_path=None,
    )

    # 编码查询
    query_dense = embedder.encode_batch([query])

    # 这里简化处理，只做稠密检索
    # 完整版应该同时做稀疏检索并融合
    results = collection.search(
        data=query_dense.tolist(),
        anns_field="dense_vector",
        param={"metric_type": "IP", "params": {"nprobe": 32}},
        limit=topk,
        output_fields=["doc_name", "page_idx", "text", "summary", "doc_id"],
    )

    print(f"\nTop-{topk} 结果:")
    for i, hits in enumerate(results):
        for j, hit in enumerate(hits):
            print(f"\n[{j+1}] 相似度: {hit.score:.4f}")
            print(f"    文档: {hit.entity.get('doc_name')}")
            print(f"    页码: {hit.entity.get('page_idx')}")
            summary = hit.entity.get('summary') or hit.entity.get('text') or ""
            print(f"    内容: {summary[:200]}...")


def main():
    parser = argparse.ArgumentParser(description="从中央库构建 Milvus 向量库")
    parser.add_argument("--db", help="中央文本数据库路径 (SQLite)")
    parser.add_argument("--milvus-db", required=True, help="Milvus 数据库路径")
    parser.add_argument("--collection", default="text_blocks", help="集合名称")
    parser.add_argument("--mode", choices=["stats", "build", "query"], required=True, help="运行模式")
    parser.add_argument("--stats-dir", default="./corpus_stats", help="统计文件目录")
    parser.add_argument("--doc-id", help="指定文档ID（增量构建）")
    parser.add_argument("--model", default="BAAI/bge-m3", help="编码模型")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--min-length", type=int, default=10, help="最小文本长度")
    parser.add_argument("--query", type=str, help="查询文本（mode=query时使用）")
    parser.add_argument("--topk", type=int, default=5, help="返回结果数")

    args = parser.parse_args()

    if args.mode == "query":
        query_vector_store(
            args.milvus_db,
            args.collection,
            args.query,
            args.model,
            args.topk,
        )
    else:
        if not args.db:
            print("错误: --db 必须指定（stats/build 模式）")
            return

        if args.mode == "stats":
            # 仅构建统计
            reader = TextDBReader(args.db)
            tokenizer = Tokenizer(
                bm25_vocab_path=None,
                jieba_userdict_path=None,
                stopwords_zh_path=None,
                stopwords_en_path=None,
            )
            builder = BM25StatsBuilder(tokenizer)

            for block in tqdm(reader.get_text_blocks(min_length=args.min_length), desc="Building stats"):
                builder.add_document(block["text"])

            stats = builder.build()
            os.makedirs(args.stats_dir, exist_ok=True)
            stats_path = os.path.join(args.stats_dir, "corpus_stats.json")
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"统计已保存: {stats_path}")
            print(f"N={stats['N']}, avgdl={stats['avgdl']:.2f}, vocab={stats['vocab_size']}")

        elif args.mode == "build":
            build_vectors_from_db(
                args.db,
                args.milvus_db,
                args.collection,
                args.stats_dir,
                args.doc_id,
                args.model,
                args.batch_size,
                args.min_length,
            )


if __name__ == "__main__":
    main()
