#!/usr/bin/env python3
"""submit_answer: ACCEPT the current evidence set; the generator produces the answer."""

from __future__ import annotations

from typing import Any, Dict

from tools.base import Tool, ToolResult


class SubmitAnswerTool(Tool):
    name = "submit_answer"
    description = (
        "结束本回合：已积累的证据集将交给生成器产生最终答案。"
        "证据足够或预算将尽时调用。"
    )
    args_schema = "{}  # 无参数"

    def run(self, session, args: Dict[str, Any]) -> ToolResult:
        session.mark_submit()
        return ToolResult(
            ok=True,
            candidates=session.last_passages,
            message=f"submit_answer 已接受，共 {len(session.collected)} 条证据；生成器即将作答",
            finish=True,
        )
