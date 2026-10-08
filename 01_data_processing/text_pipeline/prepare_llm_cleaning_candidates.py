#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为第二阶段（LLM 清洗）准备候选样本：
从 SQLite text_blocks 中筛选疑似低质量文本，导出 JSONL 供 LLM 标注。
"""

import argparse
import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Candidate:
    block_id: int
    doc_id: str
    page_idx: int
    char_length: int
    text: str
    signals: List[str]
    score: int


CHAPTER_RE = re.compile(r"第[0-9一二三四五六七八九十百千万]+[章节篇节部卷册]")
SLASH_PAGE_RE = re.compile(r"/\d{1,4}$")
FIG_RE = re.compile(r"^(图|表|FIG|Figure|Table)\s*[0-9A-Za-z\-.]*", re.IGNORECASE)
PUB_META_RE = re.compile(r"(主编|ISBN|责任编辑|版权所有|出版|Copyright)", re.IGNORECASE)
REF_RE = re.compile(r"^\s*\[?\d{1,3}\]?\s*[\.、)]\s*")


def symbol_ratio(text: str) -> float:
    s = (text or "").strip()
    if not s:
        return 1.0
    sym = 0
    for ch in s:
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            sym += 1
    return sym / len(s)


def detect_signals(text: str, ln: int) -> Tuple[List[str], int]:
    s = re.sub(r"\s+", " ", (text or "").strip())
    signals: List[str] = []
    score = 0

    if ln <= 8:
        signals.append("very_short")
        score += 3
    elif ln <= 15:
        signals.append("short")
        score += 1

    if CHAPTER_RE.search(s) and ln <= 40:
        signals.append("chapter_like")
        score += 2

    if SLASH_PAGE_RE.search(s) and ln <= 80:
        signals.append("catalog_page_like")
        score += 2

    if FIG_RE.search(s):
        signals.append("figure_table_like")
        score += 2

    if PUB_META_RE.search(s):
        signals.append("publication_meta_like")
        score += 2

    if REF_RE.search(s) and ln <= 40:
        signals.append("enumeration_like")
        score += 1

    sr = symbol_ratio(s)
    if sr >= 0.45 and ln <= 80:
        signals.append("high_symbol_ratio")
        score += 2

    latin = sum(1 for c in s if c.isalpha() and ("a" <= c.lower() <= "z"))
    cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
    if latin >= 6 and cjk == 0 and ln <= 30 and s.upper() == s:
        signals.append("upper_english_fragment")
        score += 1

    return signals, score


def export_candidates(db_path: Path, output_path: Path, min_score: int, limit: int) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows = cur.execute(
        "SELECT block_id, doc_id, page_idx, char_length, text FROM text_blocks ORDER BY block_id"
    )

    picked: List[Candidate] = []
    for row in rows:
        text = row["text"] or ""
        ln = int(row["char_length"] or len(text))
        signals, score = detect_signals(text, ln)
        if score >= min_score:
            picked.append(
                Candidate(
                    block_id=int(row["block_id"]),
                    doc_id=str(row["doc_id"]),
                    page_idx=int(row["page_idx"]),
                    char_length=ln,
                    text=text,
                    signals=signals,
                    score=score,
                )
            )

    picked.sort(key=lambda x: (-x.score, x.char_length, x.block_id))
    if limit > 0:
        picked = picked[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in picked:
            rec: Dict = {
                "block_id": item.block_id,
                "doc_id": item.doc_id,
                "page_idx": item.page_idx,
                "char_length": item.char_length,
                "text": item.text,
                "signals": item.signals,
                "score": item.score,
                "llm_action": "",
                "llm_reason": "",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"db={db_path}")
    print(f"output={output_path}")
    print(f"min_score={min_score}")
    print(f"limit={limit}")
    print(f"exported={len(picked)}")

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 LLM 二阶段清洗候选数据")
    parser.add_argument("--db", required=True, help="输入 SQLite 数据库")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件")
    parser.add_argument("--min-score", type=int, default=3, help="疑似噪声最小分值")
    parser.add_argument("--limit", type=int, default=50000, help="最多导出条数，<=0 表示全部")
    args = parser.parse_args()

    export_candidates(
        db_path=Path(args.db),
        output_path=Path(args.output),
        min_score=args.min_score,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

