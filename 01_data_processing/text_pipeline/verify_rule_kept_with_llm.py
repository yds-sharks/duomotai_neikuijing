#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅对上一阶段“规则保留”的数据继续做 LLM 复核。

特点：
1) 只处理 cleaning_audit_stage2_step1 里 source='rule' AND action='keep'
2) 支持断点续跑（基于 decisions 文件）
3) 边跑边写库（drop 直接删，rewrite 直接改）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Set, Tuple


def norm(s: str) -> str:
    return " ".join((s or "").strip().split())


def hash64(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:64]


def call_llm(
    api_key: str,
    base_url: str,
    model: str,
    items: List[Dict],
    timeout: int,
    max_retries: int,
) -> Dict[int, Dict]:
    url = base_url.rstrip("/") + "/chat/completions"
    system_prompt = (
        "你是医学RAG数据清洗助手。任务：复核原先规则保留的文本块是否真的有检索价值。"
        "动作：drop/keep/rewrite。"
        "若仅标题、目录、无独立问答价值片段，drop。"
        "rewrite 可去噪和重排，可适度补充衔接，但不得遗漏原语义，不得凭空造事实。"
        "只输出 JSON。"
    )
    user_prompt = (
        "输出 JSON: "
        "{\"results\":[{\"block_id\":1,\"action\":\"drop|keep|rewrite\",\"quality_score\":0-100,"
        "\"reason\":\"简短理由\",\"rewritten_text\":\"仅rewrite填写\"}]}\n"
        + json.dumps(items, ensure_ascii=False)
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
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_err = None
    for i in range(max_retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            content = obj["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            out: Dict[int, Dict] = {}
            for row in parsed.get("results", []):
                bid = int(row["block_id"])
                act = str(row.get("action", "keep")).lower().strip()
                if act not in {"drop", "keep", "rewrite"}:
                    act = "keep"
                qs = int(row.get("quality_score", 60))
                qs = max(0, min(100, qs))
                rs = str(row.get("reason", ""))[:180]
                rt = norm(str(row.get("rewritten_text", "")))
                out[bid] = {"action": act, "quality_score": qs, "reason": rs, "rewritten_text": rt}
            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(15, 2 ** i))
    raise RuntimeError(f"LLM调用失败: {last_err}")


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


def main() -> None:
    p = argparse.ArgumentParser(description="仅复核规则保留数据（LLM）")
    p.add_argument("--input-db", required=True, help="上一阶段输出库")
    p.add_argument("--output-db", required=True, help="复核后输出库")
    p.add_argument("--work-dir", required=True, help="工作目录（保存断点与决策）")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--api-key", default="")
    p.add_argument("--base-url", default="")
    p.add_argument("--batch-size", type=int, default=120)
    p.add_argument("--timeout-sec", type=int, default=180)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--max-llm-calls", type=int, default=0, help="限制批次数，0=不限")
    p.add_argument("--overwrite-output", action="store_true")
    args = p.parse_args()

    api_key = (args.api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    base_url = (args.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).strip()
    if not api_key:
        raise RuntimeError("缺少 API key")

    in_db = Path(args.input_db)
    out_db = Path(args.output_db)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    decisions = work / "verify_rule_keep_decisions.jsonl"

    if not in_db.exists():
        raise FileNotFoundError(in_db)
    if out_db.exists() and not args.overwrite_output:
        print(f"输出库已存在，进入续跑模式: {out_db}")
    elif not out_db.exists():
        shutil.copy2(str(in_db), str(out_db))

    conn = sqlite3.connect(str(out_db))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS cleaning_audit_stage2_rule_keep_llm (
            block_id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            action TEXT NOT NULL,
            quality_score INTEGER DEFAULT 0,
            reason TEXT,
            rewritten_text TEXT
        )
        """
    )
    conn.commit()

    processed: Set[int] = set()
    if decisions.exists():
        with decisions.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                processed.add(int(row["block_id"]))

    ids = [
        r[0]
        for r in cur.execute(
            """
            SELECT block_id
            FROM cleaning_audit_stage2_step1
            WHERE source='rule' AND action='keep'
            ORDER BY block_id
            """
        ).fetchall()
    ]
    ids = [x for x in ids if x not in processed]
    total = len(ids)
    print(f"待复核（原规则保留）: {total}")

    calls = 0
    with decisions.open("a", encoding="utf-8") as fout:
        for i in range(0, total, args.batch_size):
            batch_ids = ids[i : i + args.batch_size]
            if not batch_ids:
                break

            qmarks = ",".join(["?"] * len(batch_ids))
            rows = cur.execute(
                f"""
                SELECT t.block_id, t.char_length, t.text,
                       COALESCE((SELECT text FROM text_blocks WHERE block_id=t.block_id-1),'') AS prev_text,
                       COALESCE((SELECT text FROM text_blocks WHERE block_id=t.block_id+1),'') AS next_text
                FROM text_blocks t
                WHERE t.block_id IN ({qmarks})
                ORDER BY t.block_id
                """,
                batch_ids,
            ).fetchall()

            items = []
            for r in rows:
                items.append(
                    {
                        "block_id": int(r["block_id"]),
                        "char_length": int(r["char_length"] or 0),
                        "prev_text": norm(str(r["prev_text"]))[:220],
                        "text": norm(str(r["text"]))[:420],
                        "next_text": norm(str(r["next_text"]))[:220],
                    }
                )

            result = call_llm(
                api_key=api_key,
                base_url=base_url,
                model=args.model,
                items=items,
                timeout=args.timeout_sec,
                max_retries=args.max_retries,
            )

            for r in rows:
                bid = int(r["block_id"])
                src_text = norm(str(r["text"]))
                out = result.get(
                    bid,
                    {"action": "keep", "quality_score": 60, "reason": "llm_no_output_default_keep", "rewritten_text": ""},
                )
                action = out["action"]
                rewritten = norm(str(out.get("rewritten_text", "")))
                if action == "rewrite":
                    if not rewritten:
                        action = "keep"
                    elif len(src_text) >= 20 and len(rewritten) < int(0.6 * len(src_text)):
                        action = "keep"
                        out["reason"] = "rewrite_too_short_fallback_keep"

                if action == "drop":
                    cur.execute("DELETE FROM text_blocks WHERE block_id=?", (bid,))
                elif action == "rewrite":
                    cur.execute(
                        "UPDATE text_blocks SET text=?, char_length=?, content_hash=? WHERE block_id=?",
                        (rewritten[:8192], len(rewritten), hash64(rewritten), bid),
                    )

                cur.execute(
                    """
                    INSERT OR REPLACE INTO cleaning_audit_stage2_rule_keep_llm
                    (block_id, source, action, quality_score, reason, rewritten_text)
                    VALUES (?, 'llm', ?, ?, ?, ?)
                    """,
                    (
                        bid,
                        action,
                        int(out.get("quality_score", 60)),
                        str(out.get("reason", "")),
                        rewritten[:8192],
                    ),
                )

                fout.write(
                    json.dumps(
                        {
                            "block_id": bid,
                            "action": action,
                            "quality_score": int(out.get("quality_score", 60)),
                            "reason": str(out.get("reason", "")),
                            "rewritten_text": rewritten[:8192],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            conn.commit()
            calls += 1
            done = min(i + args.batch_size, total)
            print(f"复核进度: {done}/{total}")
            if args.max_llm_calls > 0 and calls >= args.max_llm_calls:
                print("达到 --max-llm-calls，暂停。")
                break

    update_doc_stats(conn)
    left = cur.execute("SELECT COUNT(*) FROM text_blocks").fetchone()[0]
    chk = cur.execute("SELECT COUNT(*) FROM cleaning_audit_stage2_rule_keep_llm").fetchone()[0]
    drop = cur.execute("SELECT COUNT(*) FROM cleaning_audit_stage2_rule_keep_llm WHERE action='drop'").fetchone()[0]
    rewrite = cur.execute("SELECT COUNT(*) FROM cleaning_audit_stage2_rule_keep_llm WHERE action='rewrite'").fetchone()[0]
    conn.close()

    print("\n=== 规则保留复核完成/暂停 ===")
    print(f"输出库: {out_db}")
    print(f"复核记录: {chk}")
    print(f"drop: {drop}, rewrite: {rewrite}")
    print(f"当前 text_blocks: {left}")


if __name__ == "__main__":
    main()

