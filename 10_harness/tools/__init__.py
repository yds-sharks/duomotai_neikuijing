#!/usr/bin/env python3
"""Harness tool set: the brain's ENTIRE action space is just two tools.

rag_search     — the complete RAG pipeline (retrieval + internal evidence selection)
submit_answer  — finish the episode; the generator answers from the evidence set
"""

from __future__ import annotations

from tools.base import Tool, ToolRegistry, ToolResult
from tools.rag_search import RagSearchTool
from tools.submit_answer import SubmitAnswerTool

__all__ = ["Tool", "ToolRegistry", "ToolResult", "RagSearchTool", "SubmitAnswerTool"]


def default_registry() -> ToolRegistry:
    return ToolRegistry([RagSearchTool(), SubmitAnswerTool()])
