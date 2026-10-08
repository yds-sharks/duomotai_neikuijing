#!/usr/bin/env python3
"""Evidence selection utilities for the agentic RAG runtime.

This module replaces the previous PPR/Qwen3-VL reranker. The new framework does
NOT use any standalone reranker model. After first-stage coarse retrieval, we
simply sort text+image candidates by their first-stage retrieval score and keep
the top-k (default 5) as the observation window handed to the agent.

The agent (not a reranker) is responsible for selecting which of these top-k
pieces of evidence are actually useful, and for deciding ACCEPT vs REWRITE.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

DROP_KEYS = ("raw",)


def _score(item: Dict[str, Any]) -> float:
    for key in ("score", "first_stage_score"):
        value = item.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def select_top_evidence(
    candidates: List[Dict[str, Any]],
    *,
    select_k: int = 5,
    drop_raw: bool = True,
) -> List[Dict[str, Any]]:
    """Sort text+image candidates by first-stage score and keep the top-k.

    No reranking model is involved: the order is purely the coarse-retrieval
    score, with text and image candidates mixed into a single pool.
    """
    rows = sorted(candidates or [], key=_score, reverse=True)
    out: List[Dict[str, Any]] = []
    for rank, item in enumerate(rows[: max(int(select_k), 0)], start=1):
        copied = dict(item)
        if drop_raw:
            for key in DROP_KEYS:
                copied.pop(key, None)
        copied["rank"] = rank
        copied["first_stage_score"] = _score(item)
        out.append(copied)
    return out


def apply_selection(
    top_evidence: List[Dict[str, Any]],
    selected_indices: Optional[List[int]],
) -> List[Dict[str, Any]]:
    """Return the agent-selected subset of the top-k evidence.

    ``selected_indices`` are 1-based positions into ``top_evidence`` (matching the
    ``[1]..[N]`` numbering shown to the agent). If empty/invalid, the full
    top_evidence list is returned so downstream generation still has context.
    """
    if not selected_indices:
        return list(top_evidence)
    picked: List[Dict[str, Any]] = []
    seen = set()
    for idx in selected_indices:
        try:
            pos = int(idx)
        except (TypeError, ValueError):
            continue
        if 1 <= pos <= len(top_evidence) and pos not in seen:
            seen.add(pos)
            picked.append(top_evidence[pos - 1])
    return picked or list(top_evidence)


class TopKEvidenceSelector:
    """Thin callable wrapper around :func:`select_top_evidence`."""

    def __init__(self, select_k: int = 5):
        self.select_k = int(select_k)

    def select(
        self,
        candidates: List[Dict[str, Any]],
        *,
        select_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return select_top_evidence(candidates, select_k=int(select_k or self.select_k))
