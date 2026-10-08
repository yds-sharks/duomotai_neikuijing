import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_doc_id(doc_out_dir):
    # 取 __ 后面的部分
    return doc_out_dir.split("__")[-1]


def ingest(jsonl_path, total_store_path):

    data = load_jsonl(jsonl_path)

    # === 按 doc_out_dir 分组 ===
    grouped = defaultdict(list)
    for item in data:
        grouped[item["doc_out_dir"]].append(item)

    total_store = Path(total_store_path)
    total_store.mkdir(parents=True, exist_ok=True)

    print(f"发现文档数量: {len(grouped)}")

    for doc_out_dir, items in grouped.items():

        doc_id = extract_doc_id(doc_out_dir)
        mineru_version = items[0].get("mineru_version", "unknown")

        doc_dir = total_store / "documents" / doc_id
        parse_dir = doc_dir / "parses" / f"mineru_v{mineru_version}"

        parse_dir.mkdir(parents=True, exist_ok=True)

        blocks = []

        for idx, item in enumerate(items):
            block = {
                "block_id": f"{doc_id}_p{item['page_idx']}_{idx}",
                "doc_id": doc_id,
                "page_idx": item["page_idx"],
                "type": item["type"],
                "bbox": item["bbox_norm1000"],
                "text": item.get("text"),
                "img_abs_path": item.get("img_abs_path"),
                "table_html": item.get("table_html"),
                "content_hash": item["content_hash"],
                "coord_sys": item.get("coord_sys"),
                "is_noise": item.get("is_noise"),
                "extra": item.get("extra"),
            }
            blocks.append(block)

        save_jsonl(blocks, parse_dir / "blocks.jsonl")

        meta = {
            "doc_id": doc_id,
            "doc_out_dir": doc_out_dir,
            "created_at": datetime.now().isoformat(),
            "parse_versions": [f"mineru_v{mineru_version}"],
            "derived_versions": []
        }

        with open(doc_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"入库完成: {doc_id}")

    print("全部入库完成")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", required=True)
    parser.add_argument("--total_store_path", required=True)

    args = parser.parse_args()

    ingest(args.jsonl_path, args.total_store_path)