#!/usr/bin/env python3
"""Harness tool set (the RAG capability surface exposed to the brain)."""

from __future__ import annotations

from tools.base import Tool, ToolRegistry, ToolResult
from tools.text_retrieve import TextRetrieveTool
from tools.image_retrieve import ImageRetrieveTool
from tools.keep_evidence import KeepEvidenceTool
from tools.submit_answer import SubmitAnswerTool

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "TextRetrieveTool",
    "ImageRetrieveTool",
    "KeepEvidenceTool",
    "SubmitAnswerTool",
]


def default_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            TextRetrieveTool(),
            ImageRetrieveTool(),
            KeepEvidenceTool(),
            SubmitAnswerTool(),
        ]
    )
