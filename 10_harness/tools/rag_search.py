#!/usr/bin/env python3
"""rag_search: THE single RAG tool exposed to the central brain.

Input: a natural-language query (the question's image is used automatically).
Output: final passages selected by the FULL RAG pipeline — dual-path retrieval +
internal evidence selection (the former v0.5 evidence-selection agent now lives
INSIDE the RAG tool). The brain never sees raw first-stage candidates.
"""

from __future__ import annotations

from typing import Any, Dict

from tools.base import Tool, ToolResult


class RagSearchTool(Tool):
    name = "rag_search"
    description = (
        "对一条 query 运行完整 RAG 管道：双路检索（文本 + 题目图像）+ 内部证据筛选。"
        "返回最终证据文段，并已并入会话证据集。证据不足时请改写 query 再次调用。"
    )
    args_schema = '{"query": str, "use_image": bool?}  # 题目有图时 use_image 默认为 true'

    def run(self, session, args: Dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, message="rag_search requires non-empty args.query")
        use_image = bool(args.get("use_image", bool(session.query_image_path)))
        kept = session.rag.search(
            query=query,
            image_path=session.query_image_path if use_image else "",
            question=session.question,
            options=session.options,
        )
        added, dup = session.add_evidence(kept)
        repeat = "（警告：该 query 与之前重复）" if query in session.search_history else ""
        session.log_search(query)
        return ToolResult(
            ok=True,
            candidates=kept,  # this round's final passages (what the brain reviews)
            n_new=added,
            message=(
                f"rag_search 完成：返回 {len(kept)} 条文段，新增 {added} 条证据"
                + (f"，跳过 {dup} 条重复" if dup else "")
                + ("" if use_image or not session.query_image_path else "；未使用图像")
                + repeat
            ),
        )
