#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2 Step1: 检索价值优先清洗（drop / keep / rewrite）

目标：
1) 删除无语义噪声（drop）
2) 保留高检索价值文段（keep）
3) 对可修复块做重写（rewrite）

关键能力：
- 规则引擎：先做高置信 keep/drop
- LLM 判定：基于前后文输出 action + quality_score + reason + rewritten_text
- 审计落库：记录每条块的来源、动作、分数、原因
- 输出新库：删除 drop，写回 rewrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Block:
    block_id: int
    doc_id: str
    page_idx: int
    char_length: int
    text: str
    prev_text: str
    next_text: str


CHAPTER_RE = re.compile(r"第[0-9一二三四五六七八九十百千万]+[章节篇节部卷册]")
SLASH_PAGE_RE = re.compile(r"/\d{1,4}$")
FIG_RE = re.compile(r"^(图|表|FIG|Figure|Table)\s*[0-9A-Za-z\-.]*", re.IGNORECASE)
PUB_META_RE = re.compile(r"(主编|ISBN|责任编辑|版权所有|出版|Copyright)", re.IGNORECASE)
REF_RE = re.compile(r"^\s*\[?\d{1,3}\]?\s*[\.、)]\s*")
ENUM_LIST_RE = re.compile(
    r"^\s*(?:[A-Za-zＡ-Ｚa-zａ-ｚ][、.:：]\s*[\u4e00-\u9fffA-Za-z]{1,10}\s*){2,}\s*$"
)
LATEX_OCR_RE = re.compile(r"(\\text|\\mathrm|\\mathbb|\\spadesuit|\\textcircled|\$\s*\\)")
PURE_NUM_RE = re.compile(r"^\d{1,4}$")
MEDICAL_HINT_RE = re.compile(
    r"(内镜|胃|食管|小肠|结肠|直肠|胰|胆|肝|癌|肿瘤|病变|溃疡|息肉|出血|穿孔|炎|ERCP|EUS|ESD|EMR|NBI)"
)
ACTION_RE = re.compile(r"(诊断|治疗|切除|检查|评估|手术|操作|处理|预防|随访|并发症|适应证|禁忌证|止血|穿刺)")
OUTCOME_RE = re.compile(r"(提示|考虑|符合|显示|可见|证实|阳性|阴性|改善|恶化|风险)")
LOW_VALUE_PATTERN = re.compile(
    r"(大小的病变|位于.*部|见.*病变|难易度|病例|视频|章节|目录|页)"
)
HARD_DROP_PATTERN = re.compile(
    r"(CIP|图书馆CIP|主编|责任编辑|译者|版权所有|ISBN|视频\s*\d+|p\.\d+|第\d+章|难易度|病例\d+|目录)"
)


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def symbol_ratio(s: str) -> float:
    t = normalize_text(s)
    if not t:
        return 1.0
    sym = 0
    for ch in t:
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            sym += 1
    return sym / len(t)


def char_stats(s: str) -> Dict[str, int]:
    t = normalize_text(s)
    return {
        "len": len(t),
        "cjk": sum(1 for c in t if "\u4e00" <= c <= "\u9fff"),
        "latin": sum(1 for c in t if c.isalpha() and ("a" <= c.lower() <= "z")),
        "digit": sum(1 for c in t if c.isdigit()),
        "bad": sum(1 for c in t if c in "�□◻◼◇◆※#@$%^&*_+=~`|"),
    }


def candidate_signals(text: str, ln: int) -> Tuple[List[str], int]:
    s = normalize_text(text)
    sig: List[str] = []
    score = 0

    if ln <= 8:
        sig.append("very_short")
        score += 3
    elif ln <= 15:
        sig.append("short")
        score += 1

    if CHAPTER_RE.search(s) and ln <= 50:
        sig.append("chapter_like")
        score += 2
    if SLASH_PAGE_RE.search(s) and ln <= 100:
        sig.append("catalog_page_like")
        score += 2
    if FIG_RE.search(s):
        sig.append("figure_table_like")
        score += 2
    if PUB_META_RE.search(s):
        sig.append("publication_meta_like")
        score += 2
    if REF_RE.search(s) and ln <= 60:
        sig.append("enumeration_like")
        score += 1
    if ENUM_LIST_RE.search(s):
        sig.append("enum_fragment_like")
        score += 3
    if LATEX_OCR_RE.search(s) and ln <= 120:
        sig.append("latex_ocr_like")
        score += 2
    if symbol_ratio(s) >= 0.45 and ln <= 120:
        sig.append("high_symbol_ratio")
        score += 2

    st = char_stats(s)
    if st["bad"] >= 2:
        sig.append("bad_char_like")
        score += 2
    if st["latin"] >= 6 and st["cjk"] == 0 and ln <= 24 and s.upper() == s:
        sig.append("upper_latin_fragment")
        score += 1

    return sig, score


def hash64(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:64]


def rule_decision(block: Block) -> Tuple[Optional[str], str, int]:
    """
    返回: (action, reason, score)
    action: keep | drop | rewrite | None(交给LLM)
    """
    s = normalize_text(block.text)
    ln = block.char_length
    _, score = candidate_signals(s, ln)
    st = char_stats(s)
    sr = symbol_ratio(s)

    if not s:
        return "drop", "empty_text", score
    if ln <= 3:
        return "drop", "very_short", score
    if HARD_DROP_PATTERN.search(s):
        return "drop", "hard_drop_non_content_meta_or_catalog", score
    if PURE_NUM_RE.fullmatch(s):
        return "drop", "pure_number", score
    if all((c.isspace() or unicodedata.category(c).startswith(("P", "S"))) for c in s):
        return "drop", "only_symbols", score
    if ENUM_LIST_RE.search(s) and ln <= 50 and not MEDICAL_HINT_RE.search(s):
        return "drop", "enum_fragment_without_medical_semantics", score
    if LATEX_OCR_RE.search(s) and (st["cjk"] + st["latin"]) <= 5:
        return "drop", "latex_ocr_noise", score
    if sr >= 0.62 and ln <= 60:
        return "drop", "high_symbol_ratio_short", score
    if st["bad"] >= 3 and ln <= 80:
        return "drop", "bad_chars_noise", score
    if ln <= 20 and not ACTION_RE.search(s) and not OUTCOME_RE.search(s):
        return "drop", "short_without_retrieval_signal", score

    # 明显低检索价值：位置/尺寸类片段，缺少诊疗结论或行为信息
    if ln <= 80 and MEDICAL_HINT_RE.search(s) and not ACTION_RE.search(s) and not OUTCOME_RE.search(s):
        if LOW_VALUE_PATTERN.search(s) or s.startswith(("1.", "2.", "3.", "4.", "5.")):
            return "drop", "low_retrieval_value_fragment", score

    # 高置信保留（仅保留长文段语义充分）
    if ln >= 80 and st["cjk"] >= 20 and sr < 0.2:
        return "keep", "long_semantic_sentence", score
    # 含医学行为/结论信号的中短文段，不再规则直保，统一交给 LLM 做质量复核
    if MEDICAL_HINT_RE.search(s) and (ACTION_RE.search(s) or OUTCOME_RE.search(s)) and ln >= 10 and sr < 0.35 and st["bad"] == 0:
        return None, "medical_statement_needs_llm_validation", score

    # 明显可修复：有语义但被 LaTeX/OCR 污染
    if LATEX_OCR_RE.search(s) and (st["cjk"] + st["latin"]) >= 8:
        return "rewrite", "has_semantics_but_latex_ocr_polluted", score

    return None, "to_llm", score


def call_chat_completions(
    api_key: str,
    base_url: str,
    model: str,
    items: List[Block],
    timeout: int,
    max_retries: int,
) -> Dict[int, Dict[str, str]]:
    url = base_url.rstrip("/") + "/chat/completions"

    system_prompt = (
        "你是医学RAG数据清洗助手。请根据当前文本及前后文，输出 action 和质量分。"
        "动作定义："
        "1) drop: 无检索价值（目录残片/标题碎片/仅位置尺寸描述且无诊疗意义）或噪声乱码；"
        "2) keep: 具备独立检索价值，可用于回答医学问题；"
        "3) rewrite: 有检索价值但被 OCR/符号污染，需修复后保留。"
        "rewrite 约束："
        "A. 绝对不能省略原文已有医学语义；"
        "B. 可以去噪、重排语序、适度补充衔接信息；"
        "C. 补充内容必须由当前文本或前后文可支持，禁止凭空新增事实。"
        "必须严格输出 JSON，不能输出额外文字。"
    )

    user_items = []
    for it in items:
        user_items.append(
            {
                "block_id": it.block_id,
                "prev_text": normalize_text(it.prev_text)[:220],
                "text": normalize_text(it.text)[:420],
                "next_text": normalize_text(it.next_text)[:220],
                "char_length": it.char_length,
            }
        )

    user_prompt = (
        "请逐条输出 JSON: "
        "{\"results\":[{\"block_id\":1,\"action\":\"drop|keep|rewrite\",\"quality_score\":0-100,"
        "\"reason\":\"简短理由\",\"rewritten_text\":\"仅 rewrite 时填写，其余留空\"}]}\n"
        "判定标准："
        "高检索价值文段应能独立支持问答，至少包含诊断/处理/检查发现及其意义/结论中的一种。"
        "仅位置大小描述（如“某部位见X cm病变”）且无诊疗意义，判定为 drop。"
        "仅标题/目录/图号/编号/残缺词组应 drop。"
        "若能通过轻度修复得到高检索价值文段，选 rewrite。"
        "rewrite 时必须完整覆盖原语义，不得删减关键信息。"
        "\n数据如下:\n"
        + json.dumps(user_items, ensure_ascii=False)
    )

    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            content = obj["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            out: Dict[int, Dict[str, str]] = {}
            for row in parsed.get("results", []):
                bid = int(row["block_id"])
                action = str(row.get("action", "keep")).strip().lower()
                if action not in {"drop", "keep", "rewrite"}:
                    action = "keep"
                quality = int(row.get("quality_score", 50))
                quality = max(0, min(100, quality))
                reason = str(row.get("reason", ""))[:160]
                rewritten = normalize_text(str(row.get("rewritten_text", "")))
                out[bid] = {
                    "action": action,
                    "quality_score": quality,
                    "reason": reason,
                    "rewritten_text": rewritten,
                }
            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(12, 2 ** attempt))
    raise RuntimeError(f"调用模型失败: {last_err}")


def iter_blocks_with_context(conn: sqlite3.Connection) -> Iterable[Block]:
    cur = conn.cursor()
    query = """
    SELECT
      block_id, doc_id, page_idx, char_length, text,
      COALESCE(LAG(text) OVER (ORDER BY block_id), '') AS prev_text,
      COALESCE(LEAD(text) OVER (ORDER BY block_id), '') AS next_text
    FROM text_blocks
    ORDER BY block_id
    """
    for row in cur.execute(query):
        yield Block(
            block_id=int(row[0]),
            doc_id=str(row[1]),
            page_idx=int(row[2]),
            char_length=int(row[3] or 0),
            text=str(row[4] or ""),
            prev_text=str(row[5] or ""),
            next_text=str(row[6] or ""),
        )


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


def run(args: argparse.Namespace) -> None:
    input_db = Path(args.input_db)
    output_db = Path(args.output_db)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = work_dir / "stage2_step1_quality_decisions.jsonl"

    if not input_db.exists():
        raise FileNotFoundError(f"输入库不存在: {input_db}")
    if output_db.exists() and not args.overwrite_output:
        raise FileExistsError(f"输出库已存在: {output_db}（如需覆盖加 --overwrite-output）")

    api_key = (args.api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("缺少 OpenAI API Key。请设置 OPENAI_API_KEY 或传入 --api-key。")
    base_url = (args.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).strip()

    conn = sqlite3.connect(str(input_db))
    conn.row_factory = sqlite3.Row
    try:
        total = conn.execute("SELECT COUNT(*) FROM text_blocks").fetchone()[0]
        print(f"总文本块: {total}")

        to_llm: List[Block] = []
        records: List[Dict] = []
        cnt_keep_rule = 0
        cnt_drop_rule = 0
        cnt_rewrite_rule = 0
        seen = 0

        for b in iter_blocks_with_context(conn):
            if args.max_blocks > 0 and seen >= args.max_blocks:
                break
            seen += 1
            action, reason, score = rule_decision(b)

            if action == "keep":
                cnt_keep_rule += 1
                records.append(
                    {
                        "block_id": b.block_id,
                        "source": "rule",
                        "action": "keep",
                        "quality_score": 85 if reason == "long_semantic_sentence" else 72,
                        "reason": reason,
                        "rewritten_text": "",
                        "text_preview": normalize_text(b.text)[:220],
                    }
                )
                continue

            if action == "drop":
                cnt_drop_rule += 1
                records.append(
                    {
                        "block_id": b.block_id,
                        "source": "rule",
                        "action": "drop",
                        "quality_score": 5,
                        "reason": reason,
                        "rewritten_text": "",
                        "text_preview": normalize_text(b.text)[:220],
                    }
                )
                continue

            if action == "rewrite":
                cnt_rewrite_rule += 1
                to_llm.append(b)
                continue

            # None -> 候选打分过滤
            _, cand_score = candidate_signals(b.text, b.char_length)
            if cand_score >= args.candidate_score:
                to_llm.append(b)
            else:
                cnt_keep_rule += 1
                records.append(
                    {
                        "block_id": b.block_id,
                        "source": "rule",
                        "action": "keep",
                        "quality_score": 70,
                        "reason": "low_risk_after_candidate_filter",
                        "rewritten_text": "",
                        "text_preview": normalize_text(b.text)[:220],
                    }
                )

        print(f"规则保留: {cnt_keep_rule}")
        print(f"规则删除: {cnt_drop_rule}")
        print(f"规则待修复(交LLM): {cnt_rewrite_rule}")
        print(f"LLM候选总数: {len(to_llm)}")

        batch_size = max(1, args.batch_size)
        for i in range(0, len(to_llm), batch_size):
            chunk = to_llm[i : i + batch_size]
            result = call_chat_completions(
                api_key=api_key,
                base_url=base_url,
                model=args.model,
                items=chunk,
                timeout=args.timeout_sec,
                max_retries=args.max_retries,
            )
            for b in chunk:
                row = result.get(
                    b.block_id,
                    {
                        "action": "keep",
                        "quality_score": 60,
                        "reason": "llm_no_output_default_keep",
                        "rewritten_text": "",
                    },
                )
                action = row["action"]
                rewritten = normalize_text(row.get("rewritten_text", ""))

                if action == "rewrite" and not rewritten:
                    action = "keep"
                    row["reason"] = "rewrite_without_text_fallback_keep"
                # 语义保守保护：rewrite 过短可能丢失语义，回退 keep
                if action == "rewrite":
                    src = normalize_text(b.text)
                    if len(src) >= 20 and len(rewritten) < int(0.6 * len(src)):
                        action = "keep"
                        row["reason"] = "rewrite_too_short_possible_semantic_loss_fallback_keep"

                records.append(
                    {
                        "block_id": b.block_id,
                        "source": "llm",
                        "action": action,
                        "quality_score": int(row.get("quality_score", 60)),
                        "reason": str(row.get("reason", ""))[:160],
                        "rewritten_text": rewritten[:8192],
                        "text_preview": normalize_text(b.text)[:220],
                    }
                )

            done = min(i + batch_size, len(to_llm))
            print(f"LLM 进度: {done}/{len(to_llm)}")
            if args.max_llm_calls > 0 and (done // batch_size) >= args.max_llm_calls:
                print("达到 --max-llm-calls，提前停止。")
                break

        with decisions_path.open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"决策文件: {decisions_path}")
    finally:
        conn.close()

    if args.dry_run:
        print("dry-run 模式，不生成输出库。")
        return

    if output_db.exists() and args.overwrite_output:
        output_db.unlink()
    output_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(input_db), str(output_db))

    out_conn = sqlite3.connect(str(output_db))
    try:
        out_conn.execute("DROP TABLE IF EXISTS cleaning_audit_stage2_step1")
        out_conn.execute(
            """
            CREATE TABLE cleaning_audit_stage2_step1 (
                block_id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                action TEXT NOT NULL,
                quality_score INTEGER DEFAULT 0,
                reason TEXT,
                text_preview TEXT,
                rewritten_text TEXT
            )
            """
        )

        drop_ids: List[Tuple[int]] = []
        rewrite_rows: List[Tuple[str, int, str, int]] = []

        with decisions_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                bid = int(row["block_id"])
                action = str(row["action"])
                rewritten = normalize_text(str(row.get("rewritten_text", "")))

                out_conn.execute(
                    """
                    INSERT OR REPLACE INTO cleaning_audit_stage2_step1
                    (block_id, source, action, quality_score, reason, text_preview, rewritten_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        bid,
                        str(row.get("source", "")),
                        action,
                        int(row.get("quality_score", 0)),
                        str(row.get("reason", "")),
                        str(row.get("text_preview", "")),
                        rewritten,
                    ),
                )

                if action == "drop":
                    drop_ids.append((bid,))
                elif action == "rewrite" and rewritten:
                    rewrite_rows.append((rewritten, len(rewritten), hash64(rewritten), bid))

        if drop_ids:
            out_conn.executemany("DELETE FROM text_blocks WHERE block_id = ?", drop_ids)
        if rewrite_rows:
            out_conn.executemany(
                """
                UPDATE text_blocks
                SET text = ?, char_length = ?, content_hash = ?
                WHERE block_id = ?
                """,
                rewrite_rows,
            )

        update_doc_stats(out_conn)
        out_conn.commit()

        left = out_conn.execute("SELECT COUNT(*) FROM text_blocks").fetchone()[0]
        audit = out_conn.execute("SELECT COUNT(*) FROM cleaning_audit_stage2_step1").fetchone()[0]
        dcnt = out_conn.execute(
            "SELECT COUNT(*) FROM cleaning_audit_stage2_step1 WHERE action='drop'"
        ).fetchone()[0]
        rcnt = out_conn.execute(
            "SELECT COUNT(*) FROM cleaning_audit_stage2_step1 WHERE action='rewrite'"
        ).fetchone()[0]
        print("\n=== Stage2-Step1 完成 ===")
        print(f"输出库: {output_db}")
        print(f"审计记录: {audit}")
        print(f"删除数(drop): {dcnt}")
        print(f"修复数(rewrite): {rcnt}")
        print(f"剩余 text_blocks: {left}")
    finally:
        out_conn.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage2 Step1 质量优先清洗")
    p.add_argument("--input-db", required=True, help="输入数据库")
    p.add_argument("--output-db", required=True, help="输出数据库（新库）")
    p.add_argument("--work-dir", default="/mnt/data_1/yds/多模态/data/cleaning_runs/stage2_step1_quality", help="中间目录")
    p.add_argument("--model", default="gpt-4o", help="模型名")
    p.add_argument("--api-key", default="", help="API Key")
    p.add_argument("--base-url", default="", help="API Base URL")
    p.add_argument("--batch-size", type=int, default=40, help="LLM每批条数")
    p.add_argument("--candidate-score", type=int, default=3, help="候选最小可疑分")
    p.add_argument("--timeout-sec", type=int, default=90, help="超时秒数")
    p.add_argument("--max-retries", type=int, default=3, help="API重试次数")
    p.add_argument("--max-llm-calls", type=int, default=0, help="限制批次数，0=不限")
    p.add_argument("--max-blocks", type=int, default=0, help="仅处理前N条，0=全量")
    p.add_argument("--dry-run", action="store_true", help="仅输出决策，不写新库")
    p.add_argument("--overwrite-output", action="store_true", help="覆盖已存在输出库")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
