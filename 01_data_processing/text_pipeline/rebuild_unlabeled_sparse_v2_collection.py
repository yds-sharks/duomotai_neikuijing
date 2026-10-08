#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为现有无标签 Milvus-Lite 库构建新集合：
- 读取源集合 text_blocks
- 保留 dense_vector
- 新增 sparse_vector_v2（BM25，包含长度归一化）

注意：Milvus-Lite 不能直接给已有集合追加向量字段，因此采用“新集合”方案。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Any

from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)

import sys
sys.path.insert(0, "/mnt/data_1/yds/多模态/insert")
from bm25_bge_vectorstore_v2.tokenizers import Tokenizer
from bm25_bge_vectorstore_v2.bm25_vectorizer import BM25Vectorizer


def to_sparse_map(vec: Dict[str, List[Any]]) -> Dict[int, float]:
    idx = [int(i) for i in vec.get("indices", [])]
    val = [float(v) for v in vec.get("values", [])]
    out: Dict[int, float] = {}
    for i, v in zip(idx, val):
        if i >= 0 and v != 0.0:
            out[i] = out.get(i, 0.0) + v
    return out


def build_dst_schema(dense_dim: int) -> CollectionSchema:
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="block_id", dtype=DataType.INT64),
        FieldSchema(name="text_hash", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="page_idx", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="lang", dtype=DataType.VARCHAR, max_length=8),
        # 原稀疏向量（从源集合复制）
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        # 新稀疏向量（重算）
        FieldSchema(name="sparse_vector_v2", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
    ]
    return CollectionSchema(fields=fields, description="unlabeled text blocks with sparse v2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--src", default="text_blocks")
    ap.add_argument("--dst", default="text_blocks_sparse_v2")
    ap.add_argument("--stats-json", required=True)
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--drop-dst", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0, help="仅用于调试，0 表示全量")
    args = ap.parse_args()

    # 避免上次异常留下的 lock 影响本次启动
    lock_path = os.path.join(os.path.dirname(args.db), f".{os.path.basename(args.db)}.lock")
    if os.path.exists(lock_path):
        try:
            os.remove(lock_path)
            print(f"[info] removed stale lock: {lock_path}")
        except Exception:
            pass

    connections.connect(alias="default", uri=args.db)

    src = Collection(args.src)
    src.load()

    dense_dim = None
    for f in src.schema.fields:
        if f.name == "dense_vector":
            dense_dim = int(f.params.get("dim"))
            break
    if dense_dim is None:
        raise RuntimeError("source collection has no dense_vector")

    if utility.has_collection(args.dst):
        if args.drop_dst:
            Collection(args.dst).drop()
        else:
            raise RuntimeError(f"destination collection exists: {args.dst} (use --drop-dst)")

    dst_schema = build_dst_schema(dense_dim=dense_dim)
    dst = Collection(name=args.dst, schema=dst_schema)

    # 先建索引，Lite 下可用；如果报错不影响插入
    try:
        dst.create_index(
            field_name="dense_vector",
            index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}},
        )
        print("[index] dense ok")
    except Exception as e:
        print(f"[index] dense warn: {e}")
    for f_name in ("sparse_vector", "sparse_vector_v2"):
        try:
            dst.create_index(
                field_name=f_name,
                index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP", "params": {}},
            )
            print(f"[index] {f_name} ok")
        except Exception as e:
            print(f"[index] {f_name} warn: {e}")

    # BM25（长度归一化）
    stats = json.load(open(args.stats_json, "r", encoding="utf-8"))
    tok = Tokenizer(
        bm25_vocab_path=None,
        jieba_userdict_path=None,
        stopwords_zh_path=None,
        stopwords_en_path=None,
        forbid_single_char=True,
        max_terms_per_doc=4000,
    )
    tok.token2id = {k: int(v) for k, v in stats["token2id"].items()}
    vec = BM25Vectorizer(
        df=stats["df"],
        N=int(stats["N"]),
        avgdl=float(stats["avgdl"]),
        token2id=tok.token2id,
        k1=1.2,
        b=0.75,
        idf_clip_min=0.0,
        min_df=1,
    )
    print(f"[bm25] N={stats['N']} avgdl={stats['avgdl']:.4f} vocab={len(tok.token2id)}")

    output_fields = [
        "pk",
        "block_id",
        "text_hash",
        "doc_id",
        "doc_name",
        "page_idx",
        "text",
        "summary",
        "metadata",
        "lang",
        "dense_vector",
        "sparse_vector",
    ]

    it = src.query_iterator(batch_size=int(args.batch_size), output_fields=output_fields)

    written = 0
    batch_no = 0
    while True:
        rows = it.next()
        if not rows:
            break

        entities: List[Dict[str, Any]] = []
        for r in rows:
            text = (r.get("text") or "").strip()
            toks = tok.tokenize_mixed(text) if text else []
            sp_v2 = to_sparse_map(vec.vectorize_doc(toks))
            if not sp_v2:
                # 稀疏为空时跳过，避免无效点
                continue

            entities.append(
                {
                    "pk": int(r["pk"]),
                    "block_id": int(r.get("block_id") or 0),
                    "text_hash": r.get("text_hash") or "",
                    "doc_id": r.get("doc_id") or "",
                    "doc_name": r.get("doc_name") or "",
                    "page_idx": int(r.get("page_idx") or 0),
                    "text": text[:8192],
                    "summary": (r.get("summary") or "")[:512],
                    "metadata": r.get("metadata") or {},
                    "lang": (r.get("lang") or "")[:8],
                    "sparse_vector": r.get("sparse_vector") or {},
                    "sparse_vector_v2": sp_v2,
                    "dense_vector": r.get("dense_vector") or [],
                }
            )

        n = len(entities)
        if n > 0:
            dst.insert(entities)
            written += n

        batch_no += 1
        if batch_no % 20 == 0:
            dst.flush()
            print(f"[progress] batch={batch_no} written={written}")

        if args.max_rows > 0 and written >= args.max_rows:
            print(f"[stop] max_rows={args.max_rows}")
            break

    it.close()
    dst.flush()
    dst.load()

    print("=" * 60)
    print("[done]")
    print(f"src={args.src} src_entities={src.num_entities}")
    print(f"dst={args.dst} dst_entities={dst.num_entities}")
    print("sparse_vector_v2 built with BM25(k1=1.2,b=0.75) => length normalization included")
    print("=" * 60)

    connections.disconnect("default")


if __name__ == "__main__":
    main()
