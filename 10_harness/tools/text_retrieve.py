#!/usr/bin/env python3
"""text_retrieve: text-side retrieval tool (BGE-M3 dense+sparse via backend)."""

from __future__ import annotations

from typing import Any, Dict

from tools.base import Tool, ToolResult


class TextRetrieveTool(Tool):
    name = "text_retrieve"
    description = (
        "Search the text knowledge base with a natural-language query. "
        "Returns a numbered candidate list this round. Reuse a query already in the "
        "search history is discouraged; rephrase instead."
    )
    args_schema = '{"query": str, "k": int?}  # k optional, default from config'

    def run(self, session, args: Dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, message="text_retrieve requires non-empty args.query")
        k = int(args.get("k") or session.config["retrieval"].get("text_k", 20))
        hits = session.retrieval.search_text(query, k=k)
        added = session.register_candidates(hits, origin="text", query=query)
        session.log_search(query)
        return ToolResult(
            ok=True,
            candidates=session.last_candidates,
            n_new=added,
            message=f"text_retrieve ok: {len(hits)} hits, {added} new vs collected",
        )
