#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import yaml

from .milvus_client import MilvusClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mil = cfg.get("milvus", {})

    # 初始化客户端：索引配置一并传入，保证与全链路一致
    m = MilvusClient(
        uri=mil.get("uri", "/tmp/milvus_lite.db"),
        user=mil.get("user", ""),
        password=mil.get("password", ""),
        db_name=mil.get("db_name", "default"),
        index_dense_cfg=mil.get("index_dense", {}),
        index_sparse_cfg=mil.get("index_sparse", {}),
    )

    # 创建/获取集合（内部会按 Lite 自动降级索引并跳过分区）
    collection_name = mil.get("collection", "cervix_v2")
    dims_dense = int(mil.get("dims_dense", 1024))
    max_text_len = int(mil.get("max_text_len", 4096))
    max_summary_len = int(mil.get("max_summary_len", 1024))
    partitions = mil.get("partitions", [])

    coll = m.ensure_collection(
        collection=collection_name,
        dims_dense=dims_dense,
        max_text_len=max_text_len,
        max_summary_len=max_summary_len,
        partitions=partitions,
    )

    # 加载集合到内存（Lite 已加载时可能告警，忽略即可）
    try:
        coll.load()
    except Exception:
        pass

    print(f"✅ Collection ready & loaded: {collection_name}")


if __name__ == "__main__":
    main()
