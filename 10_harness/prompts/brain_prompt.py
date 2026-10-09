#!/usr/bin/env python3
"""Prompt rendering for the central brain (system prompt + per-round state message)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

MAX_TEXT_CHARS = 220  # per-candidate text preview length
MAX_CAND_SHOWN = 10  # cap candidates rendered per round
MAX_HISTORY_SHOWN = 6


def build_system_prompt(tool_specs: List[Dict[str, str]]) -> str:
    tools_block = "\n".join(
        f"- {s['name']}: {s['description']} | args: {s['args']}" for s in tool_specs
    )
    return (
        "You are the central brain of a medical multimodal RAG system (endoscopy QA).\n"
        "You do NOT answer from memory: you gather EVIDENCE with tools, keep what helps, "
        "and submit when ready. The generator (same model family) answers ONLY from your "
        "kept evidence set.\n\n"
        "TOOLS\n" + tools_block + "\n\n"
        "PROTOCOL\n"
        "Each turn reply with EXACTLY ONE JSON object and nothing else:\n"
        '{"thought": "<brief reasoning>", "tool": "<tool name>", "args": {...}}\n\n'
        "POLICY\n"
        "1. Typically start with image_retrieve (the question shows an endoscopy image), "
        "then keep_evidence for useful candidates.\n"
        "2. If evidence is insufficient, call text_retrieve with a REPHRASED, more specific "
        "query (never repeat a query already in the search history).\n"
        "3. Keep only candidates that could change the answer; quality beats quantity.\n"
        "4. Submit (submit_answer) when evidence is sufficient or the remaining budget is low.\n"
        "5. Candidate numbers in keep_evidence refer to the LATEST numbered candidate list only.\n"
    )


def _preview(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def render_candidates(candidates: List[Dict[str, Any]]) -> str:
    if not candidates:
        return "(empty candidate list this round)"
    lines = []
    for c in candidates[:MAX_CAND_SHOWN]:
        num = c.get("harness_no", "?")
        origin = c.get("origin", c.get("source", "?"))
        score = c.get("score")
        score_s = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
        lines.append(f"[{num}] ({origin}, score={score_s}) {_preview(c.get('text', ''))}")
    if len(candidates) > MAX_CAND_SHOWN:
        lines.append(f"... (+{len(candidates) - MAX_CAND_SHOWN} more, not shown)")
    return "\n".join(lines)


def render_state_message(
    last_tool: str,
    last_message: str,
    candidates: List[Dict[str, Any]],
    collected: List[Dict[str, Any]],
    search_history: List[str],
    rounds_used: int,
    tool_calls_used: int,
    max_rounds: int,
    max_tool_calls: int,
) -> str:
    history = search_history[-MAX_HISTORY_SHOWN:]
    history_block = "\n".join(f"  - {h}" for h in history) if history else "  (none yet)"
    msg = (
        f"TOOL FEEDBACK: {last_tool or '(start)'} -> {last_message or 'new episode, choose your first tool'}\n\n"
        f"CANDIDATES THIS ROUND (1-based, for keep_evidence):\n{render_candidates(candidates)}\n\n"
        f"EVIDENCE SET: {len(collected)} item(s) collected so far\n"
        f"SEARCH HISTORY (do not repeat):\n{history_block}\n\n"
        f"BUDGET: round {rounds_used + 1}/{max_rounds}, tool calls {tool_calls_used}/{max_tool_calls}.\n"
        "Reply with ONE JSON object: {\"thought\": ..., \"tool\": ..., \"args\": {...}}"
    )
    if candidates and collected and len(collected) >= 2:
        msg += "\nEvidence is non-trivial: consider keep_evidence then submit_answer if sufficient."
    return msg


def candidate_identity(item: Dict[str, Any]) -> str:
    """Stable identity key for cross-round dedup (same idea as trajectory_runtime.candidate_id)."""
    return "|".join(
        str(item.get(k, "") or "")
        for k in ("doc_id", "page_idx", "block_id", "sample_id", "group_id", "image_path", "image_id")
    )


def observation_json(obs_dict: Dict[str, Any]) -> str:
    """Compact machine-readable echo of an observation (kept short for context budget)."""
    slim = {
        "tool": obs_dict.get("tool"),
        "ok": obs_dict.get("ok"),
        "message": obs_dict.get("message"),
        "n_candidates": len(obs_dict.get("candidates", [])),
    }
    return json.dumps(slim, ensure_ascii=False)
