#!/usr/bin/env python3
"""image_retrieve: image-side retrieval tool (vision-tower embedding via backend)."""

from __future__ import annotations

from typing import Any, Dict

from tools.base import Tool, ToolResult


class ImageRetrieveTool(Tool):
    name = "image_retrieve"
    description = (
        "Search the image knowledge base with an endoscopy image. By default uses the "
        "question's own image; pass args.image_path to search with another image. "
        "Returns a numbered candidate list this round."
    )
    args_schema = '{"k": int?, "image_path": str?}'

    def run(self, session, args: Dict[str, Any]) -> ToolResult:
        image_path = str(args.get("image_path") or session.query_image_path).strip()
        if not image_path:
            return ToolResult(ok=False, message="image_retrieve: no image available for this question")
        k = int(args.get("k") or session.config["retrieval"].get("image_k", 20))
        hits = session.retrieval.search_image(image_path, k=k)
        added = session.register_candidates(hits, origin="image", query=image_path)
        session.log_search(f"[image] {image_path}")
        return ToolResult(
            ok=True,
            candidates=session.last_candidates,
            n_new=added,
            message=f"image_retrieve ok: {len(hits)} hits, {added} new vs collected",
        )
