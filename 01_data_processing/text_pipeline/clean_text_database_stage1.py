#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-1 规则清洗：清洗 text_database.db 中低质量文本块

特点：
1) 默认 dry-run，仅统计不会改库
2) apply 模式会复制一份新数据库并在新库中删除低质量块
3) 记录删除原因，便于人工复查

示例：
  # 仅统计
  python3 clean_text_database_stage1.py \
      --input /mnt/data_1/yds/多模态/data/text_database.db \
      --mode dry-run

  # 生成清洗后新库
  python3 clean_text_database_stage1.py \
      --input /mnt/data_1/yds/多模态/data/text_database.db \
      --output /mnt/data_1/yds/多模态/data/text_database_stage1_clean.db \
      --mode apply \
      --min-length 5 \
      --dedup-hash
"""

import argparse
import re
import shutil
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


SHORT_LABEL_RE = re.compile(r"^(病例|视频|步骤|附件|目录)\s*\d{1,3}$")
PURE_NUM_RE = re.compile(r"^\d{1,4}$")
CHAPTER_ONLY_RE = re.compile(r"^第[0-9一二三四五六七八九十百千万]+[章节篇节部卷册](\s*)$")
CATALOG_LIKE_RE = re.compile(r"^.{1,40}/\d{1,4}$")
FIG_SHORT_RE = re.compile(r"^(图|表|FIG|Figure|Table)\s*[0-9A-Za-z\-.]*\s*[:：]?\s*$", re.IGNORECASE)
PUBLISH_META_RE = re.compile(r"(版权所有|ISBN|责任编辑|主编|开本|印张|出版|Copyright)", re.IGNORECASE)


@dataclass
class CleanConfig:
    min_length: int = 5
    dedup_hash: bool = False
    keep_medical_short: bool = True
    medical_keywords: Tuple[str, ...] = (
        "内镜", "胃", "食管", "肠", "结肠", "直肠", "胆", "胰", "肝",
        "癌", "肿瘤", "病变", "诊断", "治疗", "切除", "出血", "穿孔",
        "息肉", "溃疡", "炎", "ERCP", "EUS", "ESD", "EMR", "NBI",
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def is_symbol_or_punct(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def text_signal_stats(text: str) -> Dict[str, float]:
    s = normalize_text(text)
    n = len(s)
    if n == 0:
        return {"len": 0, "cjk_alpha_num_ratio": 0.0, "symbol_ratio": 1.0}

    cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
    alpha = sum(1 for c in s if c.isalpha())
    digit = sum(1 for c in s if c.isdigit())
    sym = sum(1 for c in s if is_symbol_or_punct(c))

    return {
        "len": n,
        "cjk_alpha_num_ratio": (cjk + alpha + digit) / n,
        "symbol_ratio": sym / n,
    }


def should_drop(
    text: str,
    char_length: int,
    content_hash: str,
    cfg: CleanConfig,
    seen_hash: Optional[set] = None,
) -> Optional[str]:
    s = normalize_text(text)
    sig = text_signal_stats(s)

    if not s:
        return "empty_text"

    if char_length < cfg.min_length:
        return f"too_short_lt_{cfg.min_length}"

    if all((c.isspace() or is_symbol_or_punct(c)) for c in s):
        return "only_symbols"

    if PURE_NUM_RE.fullmatch(s):
        return "pure_number"

    if SHORT_LABEL_RE.fullmatch(s):
        return "short_label_with_number"

    if CHAPTER_ONLY_RE.fullmatch(s):
        return "chapter_only_heading"

    if CATALOG_LIKE_RE.fullmatch(s):
        return "catalog_slash_page"

    if FIG_SHORT_RE.fullmatch(s):
        return "figure_table_short_caption"

    if PUBLISH_META_RE.search(s) and len(s) <= 120:
        return "publication_meta"

    if sig["symbol_ratio"] >= 0.6 and len(s) <= 40:
        return "high_symbol_ratio_short"

    if len(s) <= 12 and sig["cjk_alpha_num_ratio"] < 0.5:
        return "low_info_density_short"

    if cfg.keep_medical_short and len(s) <= 12:
        if any(k.lower() in s.lower() for k in cfg.medical_keywords):
            return None

    if cfg.dedup_hash and seen_hash is not None:
        if content_hash and content_hash in seen_hash:
            return "duplicate_content_hash"
        if content_hash:
            seen_hash.add(content_hash)

    return None


def scan_blocks(
    conn: sqlite3.Connection,
    cfg: CleanConfig,
) -> Tuple[int, Counter, List[Tuple[int, str]]]:
    cur = conn.cursor()
    total = cur.execute("SELECT COUNT(*) FROM text_blocks").fetchone()[0]
    reason_counter: Counter = Counter()
    rejected: List[Tuple[int, str]] = []
    seen_hash = set() if cfg.dedup_hash else None

    rows = cur.execute(
        "SELECT block_id, text, char_length, content_hash FROM text_blocks ORDER BY block_id"
    )
    iterable: Iterable[Tuple[int, str, int, str]]
    if tqdm is not None:
        iterable = tqdm(rows, total=total, desc="Scanning")
    else:
        iterable = rows

    for block_id, text, char_len, content_hash in iterable:
        reason = should_drop(text or "", int(char_len or 0), content_hash or "", cfg, seen_hash)
        if reason:
            reason_counter[reason] += 1
            rejected.append((int(block_id), reason))

    return total, reason_counter, rejected


def update_doc_stats(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE documents
        SET block_count = (
                SELECT COUNT(*) FROM text_blocks tb WHERE tb.doc_id = documents.doc_id
            ),
            page_count = COALESCE((
                SELECT MAX(page_idx) + 1 FROM text_blocks tb WHERE tb.doc_id = documents.doc_id
            ), 0),
            updated_at = CURRENT_TIMESTAMP
        """
    )
    conn.commit()


def apply_cleaning(
    input_db: Path,
    output_db: Path,
    cfg: CleanConfig,
) -> None:
    if output_db.exists():
        raise FileExistsError(f"输出文件已存在，请先删除或换路径: {output_db}")

    output_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(input_db), str(output_db))

    conn = sqlite3.connect(str(output_db))
    try:
        total, reason_counter, rejected = scan_blocks(conn, cfg)
        dropped = len(rejected)
        keep = total - dropped

        conn.execute("DROP TABLE IF EXISTS cleaning_rejections_stage1")
        conn.execute(
            """
            CREATE TABLE cleaning_rejections_stage1 (
                block_id INTEGER PRIMARY KEY,
                reason TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO cleaning_rejections_stage1 (block_id, reason) VALUES (?, ?)",
            rejected,
        )
        conn.execute(
            "DELETE FROM text_blocks WHERE block_id IN (SELECT block_id FROM cleaning_rejections_stage1)"
        )
        update_doc_stats(conn)
        conn.commit()

        print("\n=== Stage-1 清洗完成 ===")
        print(f"输入库: {input_db}")
        print(f"输出库: {output_db}")
        print(f"总块数: {total}")
        print(f"删除块数: {dropped} ({dropped / max(total, 1):.2%})")
        print(f"保留块数: {keep} ({keep / max(total, 1):.2%})")
        print("\n删除原因分布:")
        for reason, c in reason_counter.most_common():
            print(f"- {reason}: {c} ({c / max(total, 1):.2%})")
    finally:
        conn.close()


def dry_run(input_db: Path, cfg: CleanConfig, topn: int = 20) -> None:
    conn = sqlite3.connect(str(input_db))
    try:
        total, reason_counter, rejected = scan_blocks(conn, cfg)
        dropped = len(rejected)
        keep = total - dropped

        print("\n=== Dry Run 统计 ===")
        print(f"数据库: {input_db}")
        print(f"总块数: {total}")
        print(f"预计删除: {dropped} ({dropped / max(total, 1):.2%})")
        print(f"预计保留: {keep} ({keep / max(total, 1):.2%})")

        print("\n删除原因分布:")
        for reason, c in reason_counter.most_common():
            print(f"- {reason}: {c} ({c / max(total, 1):.2%})")

        by_reason: Dict[str, List[int]] = defaultdict(list)
        for bid, reason in rejected:
            if len(by_reason[reason]) < topn:
                by_reason[reason].append(bid)

        print("\n示例（每类最多 5 条）:")
        cur = conn.cursor()
        for reason, _ in reason_counter.most_common():
            ids = by_reason[reason][:5]
            if not ids:
                continue
            placeholders = ",".join(["?"] * len(ids))
            rows = cur.execute(
                f"SELECT block_id, char_length, text FROM text_blocks WHERE block_id IN ({placeholders}) ORDER BY block_id",
                ids,
            ).fetchall()
            print(f"\n[{reason}]")
            for bid, ln, text in rows:
                t = normalize_text(text or "")
                print(f"  - {bid} | len={ln} | {t[:120]}")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 规则清洗 text_database.db")
    parser.add_argument("--input", required=True, help="输入 SQLite 库路径")
    parser.add_argument("--output", help="输出 SQLite 库路径（mode=apply 必填）")
    parser.add_argument("--mode", choices=["dry-run", "apply"], default="dry-run")
    parser.add_argument("--min-length", type=int, default=5, help="最小长度阈值")
    parser.add_argument("--dedup-hash", action="store_true", help="按 content_hash 去重（仅保留第一次出现）")
    parser.add_argument("--no-keep-medical-short", action="store_true", help="不保护短医学关键词文本")
    args = parser.parse_args()

    input_db = Path(args.input)
    if not input_db.exists():
        raise FileNotFoundError(f"输入数据库不存在: {input_db}")

    cfg = CleanConfig(
        min_length=args.min_length,
        dedup_hash=args.dedup_hash,
        keep_medical_short=not args.no_keep_medical_short,
    )

    if args.mode == "dry-run":
        dry_run(input_db, cfg)
        return

    if not args.output:
        raise ValueError("mode=apply 时必须提供 --output")
    apply_cleaning(input_db, Path(args.output), cfg)


if __name__ == "__main__":
    main()
