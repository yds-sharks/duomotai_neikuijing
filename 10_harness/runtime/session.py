#!/usr/bin/env python3
"""AgentSession: per-question state of the agent episode (candidates, evidence, budget)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from backends.retrieval_backend import RetrievalBackend
from prompts.brain_prompt import candidate_identity


class AgentSession:
    """Holds everything the tools and the brain need for ONE question.

    Numbering contract: `last_candidates` is re-numbered 1..N each retrieval round
    (fields harness_no / harness_round); keep_evidence refers to these numbers.
    Cross-round dedup uses candidate_identity().
    """

    def __init__(
        self,
        config: Dict[str, Any],
        retrieval: RetrievalBackend,
        qid: str = "",
        question: str = "",
        options: Optional[Dict[str, Any]] = None,
        query_image_path: str = "",
    ):
        self.config = config
        self.retrieval = retrieval
        self.qid = qid
        self.question = question
        self.options = options or {}
        self.query_image_path = query_image_path

        self.collected: List[Dict[str, Any]] = []
        self._collected_ids: set = set()
        self.last_candidates: List[Dict[str, Any]] = []
        self.search_history: List[str] = []

        budget = config.get("budget", {})
        self.max_rounds = int(budget.get("max_rounds", 3))
        self.max_tool_calls = int(budget.get("max_tool_calls", 6))
        self.max_collected = int(budget.get("max_collected", 12))

        self.rounds_used = 0
        self.tool_calls_used = 0
        self.submitted = False

    # ------------------------------------------------------------------ tools API
    def register_candidates(self, hits: List[Dict[str, Any]], origin: str, query: str) -> int:
        """Set this round's candidate list (renumbered), skipping already-collected items."""
        fresh: List[Dict[str, Any]] = []
        for h in hits:
            item = dict(h)
            item.setdefault("origin", origin)
            item.setdefault("source", item.get("origin", origin))
            cid = candidate_identity(item)
            if cid in self._collected_ids:
                continue
            fresh.append(item)
        numbered: List[Dict[str, Any]] = []
        for i, item in enumerate(fresh, 1):
            item["harness_no"] = i
            item["harness_round"] = self.rounds_used
            item["harness_query"] = query
            numbered.append(item)
        self.last_candidates = numbered
        return len(numbered)

    def keep_from_candidates(self, numbers: List[int]) -> Tuple[int, List[int]]:
        """Move selected candidates into the collected set. Returns (n_added, unknown_numbers)."""
        by_no = {c.get("harness_no"): c for c in self.last_candidates}
        added, unknown = 0, []
        for n in numbers:
            item = by_no.get(n)
            if item is None:
                unknown.append(n)
                continue
            cid = candidate_identity(item)
            if cid in self._collected_ids:
                continue
            if len(self.collected) >= self.max_collected:
                break
            item = dict(item)
            item["kept_at_round"] = self.rounds_used
            self.collected.append(item)
            self._collected_ids.add(cid)
            added += 1
        return added, unknown

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
