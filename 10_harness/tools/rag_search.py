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
        "Run the complete RAG pipeline on a query: dual-path retrieval (text + question "
        "image) followed by internal evidence selection. Returns the final evidence "
        "passages, already merged into the session's evidence set. Rephrase the query "
        "and call again if the evidence is insufficient."
    )
    args_schema = '{"query": str, "use_image": bool?}  # use_image defaults to true when the question has an image'

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
        repeat = " (WARNING: query already used before)" if query in session.search_history else ""
        session.log_search(query)
        return ToolResult(
            ok=True,
            candidates=kept,  # this round's final passages (what the brain reviews)
            n_new=added,
            message=(
                f"rag_search ok: {len(kept)} passages returned, +{added} new evidence"
                + (f", {dup} duplicates skipped" if dup else "")
                + ("" if use_image or not session.query_image_path else "; image NOT used")
                + repeat
            ),
        )
