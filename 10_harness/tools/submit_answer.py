#!/usr/bin/env python3
"""submit_answer: ACCEPT the current evidence set; the generator produces the answer."""

from __future__ import annotations

from typing import Any, Dict

from tools.base import Tool, ToolResult


class SubmitAnswerTool(Tool):
    name = "submit_answer"
    description = (
        "Finish the episode: the accumulated evidence set is sent to the generator, "
        "which produces the final answer. Call this when the evidence is sufficient "
        "or the budget is nearly exhausted."
    )
    args_schema = "{}  # no arguments"

    def run(self, session, args: Dict[str, Any]) -> ToolResult:
        session.mark_submit()
        return ToolResult(
            ok=True,
            candidates=session.last_passages,
            message=(
                f"submit_answer accepted with {len(session.collected)} evidence items; "
                "generator will answer now"
            ),
            finish=True,
        )
