#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块名：batch_retreve_top5.py

功能：
- 从 JSONL 文件中逐行读取样本，每行包含：
    {
      "image": "...",
      "description": "...",
      "usage": { ... }
    }
- 使用 description 作为检索 query，通过 HybridRetriever 在现有向量库中检索 topK 文段；
- 将检索结果写回到每条样本的 `top5` 字段中，输出为新的 JSONL 文件。

特性：
- 支持分批调用 Milvus（--query-batch），避免 gRPC 报文超过 512MB 触发 RESOURCE_EXHAUSTED。
"""

import argparse
import json
import pathlib
import yaml
from typing import List, Dict, Any

from bm25_bge_vectorstore_v2.retrieval_service import HybridRetriever


def load_items_jsonl(p: pathlib.Path) -> List[Dict[str, Any]]:
    """逐行读取 JSONL，返回列表。"""
    items: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            items.append(json.loads(ln))
    return items


def main():
    ap = argparse.ArgumentParser()
    # 输入文件：默认用你现在这份 JSONL
    ap.add_argument(
        "--input",
        default="/mnt/data_1/yds/RAG/Hybrid_milvus/总版/insert/数据临时站/results_merged-8B.jsonl",
        help="输入 JSONL 文件路径（每行包含 image / description / usage 等）",
    )
    # 输出文件：默认同目录加后缀
    ap.add_argument(
        "--output",
        default="",
        help="输出 JSONL 文件路径（默认为 input 同目录加 _with_top5.jsonl）",
    )
    # 检索配置 YAML
    ap.add_argument(
        "--config",
        required=True,
        help="HybridRetriever 的 YAML 配置文件路径",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=5,
        help="每条 description 检索的文段数量（默认 5）",
    )
    ap.add_argument(
        "--embed-batch",
        type=int,
        default=64,
        help="embedding 批大小（默认 64）",
    )
    ap.add_argument(
        "--k-dense",
        type=int,
        default=120,
        help="dense 候选池大小（默认 120，需 >= topk）",
    )
    ap.add_argument(
        "--k-sparse",
        type=int,
        default=800,
        help="sparse 候选池大小（默认 800）",
    )
    ap.add_argument(
        "--query-batch",
        type=int,
        default=64,
        help="一次发给 Milvus 搜索的 query 数量，避免 gRPC 请求过大（默认 64）",
    )

    args = ap.parse_args()

    in_path = pathlib.Path(args.input).resolve()
    if args.output:
        out_path = pathlib.Path(args.output).resolve()
    else:
        out_path = in_path.with_name(in_path.stem + "_with_top5.jsonl")

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text(encoding="utf-8"))

    print(f"[INFO] 读取输入文件: {in_path}")
    items = load_items_jsonl(in_path)
    print(f"[INFO] 样本数量: {len(items)}")

    # 准备 query 列表（与 items 保持一一对应顺序）
    queries: List[str] = []
    valid_indices: List[int] = []  # 记录哪些 index 有有效 description
    for idx, it in enumerate(items):
        desc = (it.get("description") or "").strip()
        if not desc:
            # 没有 description 的样本，后面给一个空 top5
            continue
        queries.append(desc)
        valid_indices.append(idx)

    print(f"[INFO] 有效 description 数量: {len(queries)}")

    # 构建 HybridRetriever
    retriever = HybridRetriever(cfg)

    # 分批调用 search_batch，避免单次请求过大
    print("[INFO] 开始批量检索 (mode=hybrid)...")
    res_list: List[Dict[str, Any]] = []

    total_q = len(queries)
    qb = max(1, args.query_batch)

    for start in range(0, total_q, qb):
        end = min(start + qb, total_q)
        batch_qs = queries[start:end]
        print(f"[INFO] 检索子批次 {start}-{end-1} / {total_q-1} (size={len(batch_qs)})")

        batch_res: List[Dict[str, Any]] = retriever.search_batch(
            batch_qs,
            topk=args.topk,
            mode="hybrid",          # 如果你只想 BGE-only，可以改成 "dense" 并 k_sparse=0
            disable_lang_adapt=False,
            k_dense=args.k_dense,
            k_sparse=args.k_sparse,
            embed_batch_size=args.embed_batch,
        )
        res_list.extend(batch_res)

    print("[INFO] 检索完成")

    if len(res_list) != len(queries):
        raise RuntimeError(
            f"检索结果数量 ({len(res_list)}) 与查询数量 ({len(queries)}) 不一致，请检查 HybridRetriever.search_batch 返回。"
        )

    # 将检索结果写回原 items 对应行
    for i, idx in enumerate(valid_indices):
        item = items[idx]
        res = res_list[i]

        # 兼容不同返回格式：尝试从 "results" / "hits" / "docs" 里取
        hits = (
            res.get("results")
            or res.get("hits")
            or res.get("docs")
            or []
        )

        top_hits = []
        for h in hits[: args.topk]:
            top_hits.append(
                {
                    "text": h.get("text", ""),
                    "score": float(h.get("score", 0.0)),
                    # 如果你希望把源文件名 / rank 等也带上，可以解除注释：
                    # "file_name": h.get("file_name"),
                    # "rank": h.get("rank"),
                }
            )

        item["top5"] = top_hits

    # 对于没有 description 的样本，补一个空列表
    for it in items:
        if "top5" not in it:
            it["top5"] = []

    # 写出新的 JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"[OK] 已写出带 top5 字段的新文件：{out_path}")


if __name__ == "__main__":
    main()
