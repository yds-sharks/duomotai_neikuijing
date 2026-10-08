#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码一：build_text_database.py

将 assetpack.jsonl 中的纯文本数据提取出来，构建中央文本数据库（SQLite）
作为总库，只存储文本数据，不涉及向量化

用法:
    # 从单个 jsonl 文件构建数据库
    python build_text_database.py \
        --input /mnt/data_1/yds/多模态/data/output/消化系统与内镜/assetpack.jsonl \
        --output /mnt/data_1/yds/多模态/data/text_database.db

    # 从包含多个 jsonl 的目录构建（自动递归查找）
    python build_text_database.py \
        --input /mnt/data_1/yds/多模态/data/output/ \
        --output /mnt/data_1/yds/多模态/data/text_database.db
"""

import os
import sys
import json
import sqlite3
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime
from tqdm import tqdm
from contextlib import contextmanager


# ============ 数据库 Schema ============

SCHEMA = """
-- 文档表：存储每个文档的元信息
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,           -- 文档唯一ID (从 doc_out_dir 提取)
    doc_name TEXT NOT NULL,            -- 文档名称
    source_path TEXT,                  -- 原始路径
    page_count INTEGER DEFAULT 0,      -- 页数
    block_count INTEGER DEFAULT 0,    -- 文本块数量
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    extra_json TEXT                     -- 额外元数据（JSON格式）
);

-- 文本块表：存储每个文本块
CREATE TABLE IF NOT EXISTS text_blocks (
    block_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,              -- 关联文档ID
    page_idx INTEGER NOT NULL,         -- 页码
    content_hash TEXT NOT NULL,        -- 内容哈希（去重用）
    text TEXT NOT NULL,                -- 文本内容
    bbox TEXT,                         -- 边界框 JSON [x1, y1, x2, y2]
    coord_sys TEXT,                    -- 坐标系统
    mineru_version TEXT,               -- MinerU版本
    is_noise BOOLEAN DEFAULT 0,       -- 是否噪声
    char_length INTEGER,               -- 文本长度
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_blocks_doc_id ON text_blocks(doc_id);
CREATE INDEX IF NOT EXISTS idx_blocks_page ON text_blocks(page_idx);
CREATE INDEX IF NOT EXISTS idx_blocks_hash ON text_blocks(content_hash);
CREATE INDEX IF NOT EXISTS idx_blocks_length ON text_blocks(char_length);
"""


class TextDatabase:
    """中央文本数据库管理"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _connect(self):
        """上下文管理器：数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """初始化数据库表结构"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            conn.commit()
        print(f"数据库初始化完成: {self.db_path}")

    def upsert_document(self, doc: Dict[str, Any]):
        """插入或更新文档"""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO documents (doc_id, doc_name, source_path, extra_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    doc_name = excluded.doc_name,
                    source_path = excluded.source_path,
                    extra_json = excluded.extra_json,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                doc["doc_id"],
                doc["doc_name"],
                doc.get("source_path", ""),
                json.dumps(doc.get("extra", {}), ensure_ascii=False)
            ))
            conn.commit()

    def insert_text_block(self, block: Dict[str, Any]) -> int:
        """插入文本块，返回 block_id"""
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO text_blocks
                (doc_id, page_idx, content_hash, text, bbox, coord_sys, mineru_version, is_noise, char_length)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                block["doc_id"],
                block["page_idx"],
                block["content_hash"],
                block["text"],
                json.dumps(block.get("bbox", [])) if block.get("bbox") else None,
                block.get("coord_sys", ""),
                block.get("mineru_version", ""),
                block.get("is_noise", False),
                len(block["text"])
            ))
            conn.commit()
            return cursor.lastrowid

    def update_document_stats(self, doc_id: str):
        """更新文档统计信息（页数、块数）"""
        with self._connect() as conn:
            conn.execute("""
                UPDATE documents SET
                    page_count = (SELECT MAX(page_idx) + 1 FROM text_blocks WHERE doc_id = ?),
                    block_count = (SELECT COUNT(*) FROM text_blocks WHERE doc_id = ?),
                    updated_at = CURRENT_TIMESTAMP
                WHERE doc_id = ?
            """, (doc_id, doc_id, doc_id))
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        with self._connect() as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            block_count = conn.execute("SELECT COUNT(*) FROM text_blocks").fetchone()[0]
            total_chars = conn.execute("SELECT SUM(char_length) FROM text_blocks").fetchone()[0] or 0
            avg_length = conn.execute(
                "SELECT AVG(char_length) FROM text_blocks"
            ).fetchone()[0] or 0

            return {
                "documents": doc_count,
                "text_blocks": block_count,
                "total_chars": total_chars,
                "avg_block_length": round(avg_length, 2),
            }

    def query_text_blocks(
        self,
        doc_id: Optional[str] = None,
        min_length: int = 0,
        limit: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """查询文本块（生成器）"""
        with self._connect() as conn:
            sql = """
                SELECT b.*, d.doc_name
                FROM text_blocks b
                JOIN documents d ON b.doc_id = d.doc_id
                WHERE b.char_length >= ?
            """
            params = [min_length]

            if doc_id:
                sql += " AND b.doc_id = ?"
                params.append(doc_id)

            sql += " ORDER BY b.doc_id, b.page_idx, b.block_id"

            if limit:
                sql += f" LIMIT {int(limit)}"

            cursor = conn.execute(sql, params)
            for row in cursor:
                yield dict(row)

    def get_all_doc_ids(self) -> List[str]:
        """获取所有文档ID"""
        with self._connect() as conn:
            cursor = conn.execute("SELECT doc_id FROM documents")
            return [row[0] for row in cursor]


def extract_doc_id(doc_out_dir: str) -> str:
    """从 doc_out_dir 路径提取 doc_id"""
    name = Path(doc_out_dir).name
    if "__" in name:
        return name.split("__")[-1][:16]  # 限制长度
    return name[:16]


def find_jsonl_files(input_path: str) -> List[str]:
    """查找所有 jsonl 文件"""
    path = Path(input_path)

    if path.is_file() and path.suffix == ".jsonl":
        return [str(path)]

    if path.is_dir():
        return [str(f) for f in path.rglob("*.jsonl")]

    return []


def parse_jsonl_file(jsonl_path: str, db: TextDatabase):
    """解析单个 jsonl 文件，提取纯文本数据"""
    print(f"处理: {jsonl_path}")

    stats = {"total": 0, "text_kept": 0, "skipped": 0, "docs_added": set()}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        # 先计数总行数
        total_lines = sum(1 for _ in f)
        f.seek(0)

        # 第一遍：收集文档信息
        doc_info_map = {}
        for line in tqdm(f, total=total_lines, desc="  Scanning", leave=False):
            try:
                item = json.loads(line.strip())
                doc_out_dir = item.get("doc_out_dir", "")
                if not doc_out_dir:
                    continue

                doc_id = extract_doc_id(doc_out_dir)
                if doc_id not in doc_info_map:
                    doc_info_map[doc_id] = {
                        "doc_id": doc_id,
                        "doc_name": Path(doc_out_dir).name[:100],
                        "source_path": doc_out_dir,
                        "extra": {"jsonl_source": jsonl_path},
                    }
            except:
                continue

        # 插入文档记录
        for doc in doc_info_map.values():
            db.upsert_document(doc)
            stats["docs_added"].add(doc["doc_id"])

        # 第二遍：插入文本块
        f.seek(0)
        for line in tqdm(f, total=total_lines, desc="  Inserting text", leave=False):
            stats["total"] += 1

            try:
                item = json.loads(line.strip())
            except:
                stats["skipped"] += 1
                continue

            # 只保留纯文本
            if item.get("type") != "text":
                stats["skipped"] += 1
                continue

            text = str(item.get("text", "")).strip()
            if len(text) < 3:  # 过滤太短的无意义文本
                stats["skipped"] += 1
                continue

            doc_out_dir = item.get("doc_out_dir", "")
            doc_id = extract_doc_id(doc_out_dir)

            block = {
                "doc_id": doc_id,
                "page_idx": item.get("page_idx", -1),
                "content_hash": item.get("content_hash", "")[:64],
                "text": text,
                "bbox": item.get("bbox_norm1000", []),
                "coord_sys": item.get("coord_sys", ""),
                "mineru_version": item.get("mineru_version", ""),
                "is_noise": item.get("is_noise", False),
            }

            db.insert_text_block(block)
            stats["text_kept"] += 1

    # 更新每个文档的统计
    for doc_id in stats["docs_added"]:
        db.update_document_stats(doc_id)

    print(f"  完成: 总记录 {stats['total']}, 保留文本 {stats['text_kept']}, 跳过 {stats['skipped']}, 文档 {len(stats['docs_added'])}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="构建中央文本数据库")
    parser.add_argument("--input", required=True, help="输入 jsonl 文件或包含 jsonl 的目录")
    parser.add_argument("--output", required=True, help="输出 SQLite 数据库路径")
    parser.add_argument("--min-length", type=int, default=3, help="最小文本长度过滤")

    args = parser.parse_args()

    # 查找所有 jsonl 文件
    jsonl_files = find_jsonl_files(args.input)
    if not jsonl_files:
        print(f"错误: 未找到 jsonl 文件: {args.input}")
        return

    print(f"找到 {len(jsonl_files)} 个 jsonl 文件")

    # 初始化数据库
    db = TextDatabase(args.output)

    # 处理所有文件
    total_stats = {"total": 0, "text_kept": 0, "skipped": 0, "docs": set()}

    for jsonl_path in jsonl_files:
        stats = parse_jsonl_file(jsonl_path, db)
        total_stats["total"] += stats["total"]
        total_stats["text_kept"] += stats["text_kept"]
        total_stats["skipped"] += stats["skipped"]
        total_stats["docs"].update(stats["docs_added"])

    # 输出统计
    print(f"\n{'='*60}")
    print("构建完成!")
    print(f"数据库: {args.output}")
    print(f"\n数据统计:")
    print(f"  处理记录: {total_stats['total']}")
    print(f"  保留文本块: {total_stats['text_kept']}")
    print(f"  跳过记录: {total_stats['skipped']}")
    print(f"  文档数量: {len(total_stats['docs'])}")

    db_stats = db.get_stats()
    print(f"\n数据库内容:")
    print(f"  文档: {db_stats['documents']}")
    print(f"  文本块: {db_stats['text_blocks']}")
    print(f"  总字符: {db_stats['total_chars']:,}")
    print(f"  平均块长度: {db_stats['avg_block_length']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
