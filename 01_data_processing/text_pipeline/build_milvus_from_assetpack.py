#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_milvus_from_assetpack.py

从 assetpack.jsonl 构建 Milvus 本地向量库
- 仅处理 type="text" 的文本块
- 使用 SentenceTransformer 生成向量
- 使用 Milvus Lite 本地存储

用法:
    python build_milvus_from_assetpack.py \
        --jsonl_path /mnt/data_1/yds/多模态/data/output/消化系统与内镜/assetpack.jsonl \
        --db_path /mnt/data_1/yds/多模态/data/output/消化系统与内镜/milvus.db \
        --collection_name text_blocks
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection
)
from sentence_transformers import SentenceTransformer


def load_text_blocks(jsonl_path: str) -> List[Dict[str, Any]]:
    """从JSONL加载文本块（仅type=text且有内容）"""
    text_blocks = []
    skipped = 0
    total = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            # 只保留文本类型且有实际内容的
            if item.get("type") != "text":
                skipped += 1
                continue

            text = item.get("text", "") or ""
            text = text.strip()
            if not text:
                skipped += 1
                continue

            # 从 doc_out_dir 提取 doc_id
            doc_out_dir = item.get("doc_out_dir", "")
            doc_id = Path(doc_out_dir).name.split("__")[-1] if "__" in Path(doc_out_dir).name else "unknown"

            block = {
                "doc_id": doc_id,
                "doc_name": Path(doc_out_dir).name,
                "page_idx": item.get("page_idx", -1),
                "content_hash": item.get("content_hash", ""),
                "text": text,
                "bbox": item.get("bbox_norm1000", []),
                "coord_sys": item.get("coord_sys", ""),
            }
            text_blocks.append(block)

    print(f"总计记录: {total}")
    print(f"文本块数量: {len(text_blocks)}")
    print(f"跳过记录: {skipped}")
    return text_blocks


def build_milvus_collection(
    db_path: str,
    collection_name: str,
    text_blocks: List[Dict[str, Any]],
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 128
):
    """构建Milvus向量库"""

    # 连接Milvus Lite
    connections.connect(alias="default", uri=db_path)
    print(f"\n已连接到: {db_path}")

    # 加载embedding模型
    print(f"加载模型: {model_name}")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"向量维度: {dim}")

    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="page_idx", dtype=DataType.INT64),
        FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]

    schema = CollectionSchema(fields, description="Text blocks from AssetPack")

    # 删除已存在的集合
    if collection_name in Collection.list():
        print(f"删除已存在的集合: {collection_name}")
        Collection(collection_name).drop()

    # 创建集合
    collection = Collection(name=collection_name, schema=schema)
    print(f"创建集合: {collection_name}")

    # 分批插入数据
    total = len(text_blocks)
    print(f"\n开始插入 {total} 条记录...")

    for i in range(0, total, batch_size):
        batch = text_blocks[i:i + batch_size]
        texts = [x["text"] for x in batch]

        # 生成向量
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # 准备数据
        data = [
            [x["doc_id"] for x in batch],
            [x["doc_name"] for x in batch],
            [x["page_idx"] for x in batch],
            [x["content_hash"] for x in batch],
            texts,
            embeddings
        ]

        collection.insert(data)
        print(f"  已插入: {min(i + len(batch), total)} / {total}")

    collection.flush()
    print(f"\n数据刷新完成，共 {collection.num_entities} 条")

    # 创建索引
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 1024}
    }
    collection.create_index("embedding", index_params)
    print("索引创建完成")

    collection.load()
    print("集合加载完成，可以开始查询")

    return collection


def search_similar(
    db_path: str,
    collection_name: str,
    query: str,
    model_name: str = "BAAI/bge-m3",
    top_k: int = 5
):
    """示例：相似度搜索"""
    connections.connect(alias="default", uri=db_path)
    collection = Collection(collection_name)
    collection.load()

    model = SentenceTransformer(model_name)
    query_vec = model.encode([query]).tolist()

    results = collection.search(
        data=query_vec,
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 32}},
        limit=top_k,
        output_fields=["doc_name", "page_idx", "text"]
    )

    print(f"\n查询: {query}")
    print(f"Top-{top_k} 结果:")
    for i, hits in enumerate(results):
        for hit in hits:
            print(f"  [{hit.score:.4f}] {hit.entity.get('doc_name')} p{hit.entity.get('page_idx')}: {hit.entity.get('text')[:100]}...")


def main():
    parser = argparse.ArgumentParser(description="从AssetPack JSONL构建Milvus向量库")
    parser.add_argument("--jsonl_path", required=True, help="AssetPack JSONL 文件路径")
    parser.add_argument("--db_path", required=True, help="Milvus 数据库文件路径（如 ./milvus.db）")
    parser.add_argument("--collection_name", default="text_blocks", help="集合名称")
    parser.add_argument("--model_name", default="BAAI/bge-m3", help="SentenceTransformer模型")
    parser.add_argument("--batch_size", type=int, default=128, help="批处理大小")
    parser.add_argument("--query", type=str, default=None, help="建库后执行查询测试")

    args = parser.parse_args()

    # 加载文本块
    print("=" * 50)
    print("步骤1: 加载文本块")
    print("=" * 50)
    text_blocks = load_text_blocks(args.jsonl_path)

    if not text_blocks:
        print("没有找到有效的文本块，退出")
        return

    # 构建向量库
    print("\n" + "=" * 50)
    print("步骤2: 构建向量库")
    print("=" * 50)
    build_milvus_collection(
        db_path=args.db_path,
        collection_name=args.collection_name,
        text_blocks=text_blocks,
        model_name=args.model_name,
        batch_size=args.batch_size
    )

    # 可选查询测试
    if args.query:
        print("\n" + "=" * 50)
        print("步骤3: 查询测试")
        print("=" * 50)
        search_similar(
            db_path=args.db_path,
            collection_name=args.collection_name,
            query=args.query,
            model_name=args.model_name
        )

    print("\n" + "=" * 50)
    print("完成！")
    print(f"数据库: {args.db_path}")
    print(f"集合: {args.collection_name}")
    print("=" * 50)


if __name__ == "__main__":
    main()
