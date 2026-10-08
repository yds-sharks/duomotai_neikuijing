#!/usr/bin/env python3
"""Reward and GRPO-style group advantage utilities for rewrite rollout."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def _as_int(value: Any) -> Optional[int]:
    try:
        if value is None or str(value) == "":
            return None
        return int(value)
    except Exception:
        return None


def evidence_hit(evidence: List[Dict[str, Any]], gold_doc_id: str = "", gold_page_idx: Any = None, page_tolerance: int = 1) -> bool:
    gold_page = _as_int(gold_page_idx)
    for item in evidence or []:
        doc_ok = bool(gold_doc_id) and str(item.get("doc_id") or "") == str(gold_doc_id)
        page = _as_int(item.get("page_idx"))
        page_ok = gold_page is not None and page is not None and abs(page - gold_page) <= page_tolerance
        if gold_doc_id and gold_page is not None:
            if doc_ok and page_ok:
                return True
        elif gold_doc_id and doc_ok:
            return True
        elif gold_page is not None and page_ok:
            return True
    return False


def leakage_penalty(query: str, answer: str = "", answer_text: str = "") -> float:
    q = str(query or "")
    penalty = 0.0
    if answer and f"答案{answer}" in q:
        penalty += 1.0
    if answer and f"最终答案" in q:
        penalty += 1.0
    # Direct answer text is a soft penalty because option-aware retrieval may include option contents.
    if answer_text and len(str(answer_text)) >= 2 and str(answer_text) in q:
        penalty += 0.2
    return penalty


def score_candidate(
    *,
    original_hit: bool,
    candidate_hit: bool,
    query: str,
    answer: str = "",
    answer_text: str = "",
    is_original: bool = False,
    agent_action: str = "",
) -> Dict[str, Any]:
    evidence_delta = float(candidate_hit) - float(original_hit)
    leak = leakage_penalty(query, answer=answer, answer_text=answer_text)
    unnecessary = 0.0
    if not is_original and original_hit and candidate_hit and str(agent_action).upper() == "REWRITE":
        unnecessary = 0.1
    length_penalty = max(0.0, (len(query) - 120) / 400.0)
    reward = evidence_delta + 0.5 * float(candidate_hit) - 0.5 * leak - unnecessary - 0.05 * length_penalty
    return {
        "reward": float(reward),
        "components": {
            "evidence_delta": evidence_delta,
            "candidate_evidence_hit": bool(candidate_hit),
            "original_evidence_hit": bool(original_hit),
            "leakage_penalty": float(leak),
            "unnecessary_rewrite_penalty": float(unnecessary),
            "length_penalty": float(length_penalty),
        },
    }


def attach_group_advantages(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    rewards = [float(c.get("reward", 0.0)) for c in candidates]
    if not rewards:
        return {"mean_reward": 0.0, "std_reward": 0.0, "candidates": candidates}
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(max(var, 0.0))
    denom = std if std > 1e-6 else 1.0
    for c, r in zip(candidates, rewards):
        adv = (r - mean) / denom
        c["advantage"] = float(adv)
        # During policy training: loss = - advantage * sum_token_logprob.
        # Here logprobs are not available yet, so expose the multiplier only.
        c["policy_loss_weight"] = float(-adv)
    best = max(candidates, key=lambda x: float(x.get("reward", 0.0))) if candidates else None
    return {
        "mean_reward": float(mean),
        "std_reward": float(std),
        "best_candidate_id": best.get("candidate_id") if best else "",
        "candidates": candidates,
    }
