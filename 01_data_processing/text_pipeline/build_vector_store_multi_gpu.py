#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码二（多GPU版）：build_vector_store_multi_gpu.py

从中央文本数据库读取纯文本数据，使用多GPU并行构建 Milvus 向量库
支持 4x A6000 并行处理

用法:
    # 1. 构建语料统计（仅需一次）
    python build_vector_store_multi_gpu.py \
        --db /mnt/data_1/yds/多模态/data/text_database.db \
        --milvus-db /mnt/data_1/yds/多模态/data/vector_store.db \
        --mode stats \
        --stats-dir ./corpus_stats

    # 2. 构建 Milvus 向量库（多GPU并行）
    python build_vector_store_multi_gpu.py \
        --db /mnt/data_1/yds/多模态/data/text_database.db \
        --milvus-db /mnt/data_1/yds/多模态/data/vector_store.db \
        --mode build \
        --stats-dir ./corpus_stats \
        --collection text_blocks \
        --gpus 0,1,2,3 \
        --batch-size 16

    # 3. 查询测试
    python build_vector_store_multi_gpu.py \
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
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from collections import defaultdict
from contextlib import contextmanager
import time

import numpy as np
import torch
from tqdm import tqdm
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from sentence_transformers import SentenceTransformer

# 复用现有的成熟组件
sys.path.insert(0, "/mnt/data_1/yds/多模态/insert")
from bm25_bge_vectorstore_v2.tokenizers import Tokenizer
from bm25_bge_vectorstore_v2.dense_embedder import BGEDense
from bm25_bge_vectorstore_v2.bm25_vectorizer import BM25Vectorizer


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
        limit: Optional[int] = None,
        offset: int = 0
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
                sql += f" LIMIT {int(limit)} OFFSET {int(offset)}"

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
        self.token2id = {}
        self.next_id = 0

    def add_document(self, text: str):
        """添加文档文本，计算词频统计"""
        tokens = self.tokenizer.tokenize_mixed(text)
        self.doc_lens.append(len(tokens))

        unique_tokens = set(tokens)
        for tok in unique_tokens:
            self.df[tok] += 1
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
            "df": dict(self.df),
            "token2id": self.token2id,
            "vocab_size": len(self.token2id),
        }


# ============ 多GPU编码器 ============

class MultiGPUEncoder:
    """多GPU并行编码器"""

    def __init__(self, model_path: str, gpu_ids: List[int], batch_size: int = 16):
        self.model_path = model_path
        self.gpu_ids = gpu_ids
        self.batch_size = batch_size
        self.num_gpus = len(gpu_ids)

        # 在每个GPU上加载模型
        self.models = {}
        self.devices = {}
        for gpu_id in gpu_ids:
            device = f"cuda:{gpu_id}"
            print(f"  加载模型到 GPU {gpu_id}...")
            model = SentenceTransformer(model_path, device=device)
            self.models[gpu_id] = model
            self.devices[gpu_id] = device

        print(f"  多GPU编码器就绪: {gpu_ids}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """并行编码文本"""
        if not texts:
            return np.array([])

        # 分割到各个GPU
        chunk_size = math.ceil(len(texts) / self.num_gpus)
        chunks = []
        for i, gpu_id in enumerate(self.gpu_ids):
            start = i * chunk_size
            end = min(start + chunk_size, len(texts))
            if start < len(texts):
                chunks.append((gpu_id, texts[start:end]))

        # 并行编码
        results = {}
        import threading

        def encode_chunk(gpu_id, chunk_texts):
            model = self.models[gpu_id]
            embeddings = model.encode(
                chunk_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            results[gpu_id] = embeddings

        threads = []
        for gpu_id, chunk_texts in chunks:
            t = threading.Thread(target=encode_chunk, args=(gpu_id, chunk_texts))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # 合并结果
        all_embeddings = []
        for gpu_id, _ in chunks:
            all_embeddings.append(results[gpu_id])

        return np.vstack(all_embeddings)


# ============ Milvus 向量库构建 ============

def create_collection(
    milvus_db: str,
    collection_name: str,
    dim: int,
    overwrite: bool = False
) -> Collection:
    """创建 Milvus 集合"""
    connections.connect(alias="default", uri=milvus_db)

    if overwrite and collection_name in utility.list_collections():
        print(f"删除已存在集合: {collection_name}")
        Collection(collection_name).drop()

    if collection_name in utility.list_collections():
        print(f"使用已存在集合: {collection_name}")
        return Collection(collection_name)

    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="block_id", dtype=DataType.INT64),
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

    try:
        collection.create_index(
            field_name="sparse_vector",
            index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"},
        )
        print("稀疏索引创建成功")
    except Exception as e:
        print(f"稀疏索引: {e}")

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


def build_vectors_from_db_multi_gpu(
    text_db_path: str,
    milvus_db_path: str,
    collection_name: str,
    stats_dir: str,
    gpu_ids: List[int] = [0, 1, 2, 3],
    model_path: str = "/mnt/data_1/yds/RAG/Hybrid_milvus/总版/pretrained_models/BAAI/bge-m3",
    batch_size: int = 16,
    min_length: int = 10,
    chunk_size: int = 10000,
):
    """多GPU版本：从中央库构建向量库"""

    reader = TextDBReader(text_db_path)
    db_stats = reader.get_stats()
    print(f"中央库统计: 文档 {db_stats['documents']}, 文本块 {db_stats['text_blocks']}")

    # 加载 BM25 统计
    stats_path = os.path.join(stats_dir, "corpus_stats.json")
    print(f"加载统计: {stats_path}")
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    tokenizer = Tokenizer(
        bm25_vocab_path=None,
        jieba_userdict_path=None,
        stopwords_zh_path=None,
        stopwords_en_path=None,
    )
    tokenizer.token2id = stats["token2id"]

    print(f"语料统计: N={stats['N']}, avgdl={stats['avgdl']:.2f}, vocab={stats['vocab_size']}")

    # 初始化多GPU编码器
    print(f"\n初始化多GPU编码器: GPUs {gpu_ids}")
    encoder = MultiGPUEncoder(model_path, gpu_ids, batch_size)
    dim = encoder.models[gpu_ids[0]].get_sentence_embedding_dimension()
    print(f"维度: {dim}")

    # 创建集合
    collection = create_collection(milvus_db_path, collection_name, dim, overwrite=True)

    # 加载 BM25 向量化器
    vectorizer = BM25Vectorizer(
        df=stats["df"],
        N=stats["N"],
        avgdl=stats["avgdl"],
        token2id=stats["token2id"],
        k1=1.2,
        b=0.75,
    )

    # 分批处理
    print(f"\n开始插入数据 (chunk_size={chunk_size}, batch_size={batch_size})...")
    inserted = 0
    failed = 0
    total_blocks = db_stats['text_blocks']

    # 使用chunks处理避免内存问题
    offset = 0
    pbar = tqdm(total=total_blocks, desc="Processing blocks")

    while offset < total_blocks:
        # 获取一个chunk的文本块
        chunk_blocks = list(reader.get_text_blocks(
            min_length=min_length,
            limit=chunk_size,
            offset=offset
        ))

        if not chunk_blocks:
            break

        # 处理这个chunk
        batch_rows = []
        batch_texts = []

        for block in chunk_blocks:
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

        # 多GPU编码
        if batch_texts:
            try:
                embeddings = encoder.encode(batch_texts)
                for r, emb in zip(batch_rows, embeddings):
                    r["dense_vector"] = emb.tolist()

                # 分批插入Milvus（每批100条）
                insert_batch_size = 100
                for i in range(0, len(batch_rows), insert_batch_size):
                    sub_batch = batch_rows[i:i + insert_batch_size]
                    try:
                        collection.insert(sub_batch)
                        inserted += len(sub_batch)
                    except Exception as e:
                        print(f"插入失败: {e}")
                        failed += len(sub_batch)

                # 定期flush
                if inserted % 1000 == 0:
                    collection.flush()

            except Exception as e:
                print(f"编码失败: {e}")
                failed += len(batch_rows)

        offset += len(chunk_blocks)
        pbar.update(len(chunk_blocks))

    pbar.close()
    collection.flush()

    print(f"\n{'='*60}")
    print("构建完成!")
    print(f"数据库: {milvus_db_path}")
    print(f"集合: {collection_name}")
    print(f"插入成功: {inserted}")
    print(f"插入失败: {failed}")
    print(f"集合总数: {collection.num_entities}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="多GPU版本：从中央库构建 Milvus 向量库")
    parser.add_argument("--db", help="中央文本数据库路径 (SQLite)")
    parser.add_argument("--milvus-db", required=True, help="Milvus 数据库路径")
    parser.add_argument("--collection", default="text_blocks", help="集合名称")
    parser.add_argument("--mode", choices=["stats", "build", "query"], required=True, help="运行模式")
    parser.add_argument("--stats-dir", default="./corpus_stats", help="统计文件目录")
    parser.add_argument("--doc-id", help="指定文档ID（增量构建）")
    parser.add_argument("--model", default="/mnt/data_1/yds/RAG/Hybrid_milvus/总版/pretrained_models/BAAI/bge-m3", help="编码模型路径")
    parser.add_argument("--gpus", default="0,1,2,3", help="使用的GPU ID，如 0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=16, help="每GPU批大小")
    parser.add_argument("--chunk-size", type=int, default=10000, help="每轮处理的块数")
    parser.add_argument("--min-length", type=int, default=10, help="最小文本长度")
    parser.add_argument("--query", type=str, help="查询文本（mode=query时使用）")
    parser.add_argument("--topk", type=int, default=5, help="返回结果数")

    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(",")]

    if args.mode == "query":
        from build_vector_store import query_vector_store
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
            # 构建统计
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
            build_vectors_from_db_multi_gpu(
                args.db,
                args.milvus_db,
                args.collection,
                args.stats_dir,
                gpu_ids,
                args.model,
                args.batch_size,
                args.min_length,
                args.chunk_size,
            )


if __name__ == "__main__":
    main()
