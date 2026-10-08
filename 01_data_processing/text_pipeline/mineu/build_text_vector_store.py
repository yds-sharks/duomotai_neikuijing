import os
import json
import glob
import argparse
from pathlib import Path

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection
)

from sentence_transformers import SentenceTransformer


# ========= 读取所有文本 =========
def load_all_text_blocks(total_store_path):
    text_blocks = []

    parse_paths = glob.glob(
        os.path.join(total_store_path, "documents", "*", "parses", "*", "blocks.jsonl")
    )

    for path in parse_paths:
        doc_id = Path(path).parts[-4]

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                if item["type"] == "text" and item.get("text"):
                    text_blocks.append({
                        "doc_id": doc_id,
                        "page_idx": item["page_idx"],
                        "content_hash": item["content_hash"],
                        "text": item["text"]
                    })

    return text_blocks


# ========= 建 Milvus Lite =========
def build_milvus(db_path, collection_name, texts):

    connections.connect(alias="default", uri=db_path)

    dim = 384

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="page_idx", dtype=DataType.INT64),
        FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]

    schema = CollectionSchema(fields, description="Text Vector Store")

    if collection_name in Collection.list():
        Collection(collection_name).drop()

    collection = Collection(name=collection_name, schema=schema)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    batch_size = 256

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        embeddings = model.encode([x["text"] for x in batch]).tolist()

        data = [
            [x["doc_id"] for x in batch],
            [x["page_idx"] for x in batch],
            [x["content_hash"] for x in batch],
            [x["text"] for x in batch],
            embeddings
        ]

        collection.insert(data)

        print(f"已写入 {i + len(batch)} / {len(texts)}")

    collection.flush()

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 1024}
    }

    collection.create_index("embedding", index_params)
    collection.load()

    print("向量库构建完成")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--total_store_path", required=True)
    parser.add_argument("--db_path", required=True)
    parser.add_argument("--collection_name", default="text_collection")

    args = parser.parse_args()

    texts = load_all_text_blocks(args.total_store_path)
    print(f"发现文本块数量: {len(texts)}")

    build_milvus(args.db_path, args.collection_name, texts)
