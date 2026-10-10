#!/usr/bin/env python3
"""AgentSession: per-question state of the agent episode.

Simplified single-tool contract: the brain calls rag_search (full RAG pipeline)
and submit_answer; the session only accumulates final passages, the search
breadcrumb, and the budget.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from backends.rag_backend import RagPipelineBackend
from prompts.brain_prompt import candidate_identity


class AgentSession:
    def __init__(
        self,
        config: Dict[str, Any],
        rag: RagPipelineBackend,
        qid: str = "",
        question: str = "",
        options: Optional[Dict[str, Any]] = None,
        query_image_path: str = "",
        retrieval_hint: str = "",
        exclude_sample_ids: Tuple[str, ...] = (),
        exclude_doc_ids: Tuple[str, ...] = (),
    ):
        self.config = config
        self.rag = rag
        self.qid = qid
        self.question = question
        self.options = options or {}
        self.query_image_path = query_image_path
        self.retrieval_hint = retrieval_hint  # zh hint for the zh corpus (e.g. EndoBench translation)
        # v0.3 P0 去泄露：rag_search 时排除题图自身样本与同书 doc_id（题目为库内来源时）
        self.exclude_sample_ids = tuple(str(s) for s in exclude_sample_ids if s)
        self.exclude_doc_ids = tuple(str(d) for d in exclude_doc_ids if d)

        self.collected: List[Dict[str, Any]] = []  # accumulated final passages across rag_search calls
        self._collected_ids: set = set()
        self.last_passages: List[Dict[str, Any]] = []  # passages returned by the latest rag_search
        self.search_history: List[str] = []

        budget = config.get("budget", {})
        self.max_rounds = int(budget.get("max_rounds", 3))
        self.max_tool_calls = int(budget.get("max_tool_calls", 6))
        self.max_collected = int(budget.get("max_collected", 12))

        self.rounds_used = 0
        self.tool_calls_used = 0
        self.submitted = False

    # ------------------------------------------------------------------ tools API
    def add_evidence(self, passages: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Merge rag_search output into the evidence set. Returns (n_added, n_duplicates)."""
        added = dup = 0
        for p in passages:
            item = dict(p)
            cid = candidate_identity(item)
            if cid in self._collected_ids:
                dup += 1
                continue
            if len(self.collected) >= self.max_collected:
                break
            item["kept_at_round"] = self.rounds_used
            self.collected.append(item)
            self._collected_ids.add(cid)
            added += 1
        self.last_passages = [dict(p) for p in passages]
        return added, dup

    def log_search(self, query: str) -> None:
        self.search_history.append(query)

    def mark_submit(self) -> None:
        self.submitted = True

    # ------------------------------------------------------------------ views
    def state_summary(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "collected": len(self.collected),
            "rounds_used": self.rounds_used,
            "tool_calls_used": self.tool_calls_used,
            "budget": {"max_rounds": self.max_rounds, "max_tool_calls": self.max_tool_calls},
            "submitted": self.submitted,
        }

    def budget_exhausted(self) -> bool:
        return self.rounds_used >= self.max_rounds or self.tool_calls_used >= self.max_tool_calls

    def advance(self) -> None:
        self.rounds_used += 1
        self.tool_calls_used += 1
