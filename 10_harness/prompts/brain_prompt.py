#!/usr/bin/env python3
"""Prompt rendering for the central brain (system prompt + per-round state message).

Single-tool protocol: the brain only decides WHEN to call rag_search (and with which
query), and WHEN to submit. All retrieval/evidence-selection intelligence lives
inside the RAG tool.
"""

from __future__ import annotations

from typing import Any, Dict, List

MAX_TEXT_CHARS = 260  # per-passage preview length
MAX_PASSAGE_SHOWN = 8
MAX_HISTORY_SHOWN = 6


def build_system_prompt(tool_specs: List[Dict[str, str]]) -> str:
    tools_block = "\n".join(f"- {s['name']}: {s['description']} | args: {s['args']}" for s in tool_specs)
    return (
        "You are the central brain of a medical multimodal RAG system (endoscopy QA).\n"
        "The query endoscopy image is attached to each message when present — read it "
        "directly; it is often decisive for visual questions (organ identification, "
        "lesion description, finding localization).\n"
        "You do NOT answer from memory and you do NOT manage evidence yourself: a single "
        "tool, rag_search, runs the whole RAG pipeline and returns curated passages. Your "
        "job is query planning: craft the best search query, decide whether to search "
        "again (rephrased), and decide when the evidence suffices. The generator answers "
        "ONLY from the accumulated evidence set.\n\n"
        "TOOLS\n" + tools_block + "\n\n"
        "PROTOCOL\n"
        "Each turn reply with EXACTLY ONE JSON object and nothing else:\n"
        '{"thought": "<brief reasoning>", "tool": "<tool name>", "args": {...}}\n\n'
        "POLICY\n"
        "1. Typically call rag_search first with a specific, information-dense query; the "
        "question image is searched automatically when present.\n"
        "2. Review the returned passages: if they likely change the answer, or the evidence "
        "already covers the question, submit. Otherwise call rag_search again with a "
        "REPHRASED, more specific query (never repeat a previous query verbatim).\n"
        "3. Budget is tight: with few calls left, prefer submitting over exploring.\n"
    )


def _preview(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def render_passages(passages: List[Dict[str, Any]]) -> str:
    if not passages:
        return "(no passages returned this call)"
    lines = []
    for i, p in enumerate(passages[:MAX_PASSAGE_SHOWN], 1):
        origin = p.get("origin", p.get("source", "?"))
        score = p.get("score")
        score_s = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
        lines.append(f"[{i}] ({origin}, score={score_s}) {_preview(p.get('text', ''))}")
    if len(passages) > MAX_PASSAGE_SHOWN:
        lines.append(f"... (+{len(passages) - MAX_PASSAGE_SHOWN} more)")
    return "\n".join(lines)


def render_state_message(
    last_tool: str,
    last_message: str,
    passages: List[Dict[str, Any]],
    collected: List[Dict[str, Any]],
    search_history: List[str],
    rounds_used: int,
    tool_calls_used: int,
    max_rounds: int,
    max_tool_calls: int,
    retrieval_hint: str = "",
) -> str:
    history = search_history[-MAX_HISTORY_SHOWN:]
    history_block = "\n".join(f"  - {h}" for h in history) if history else "  (none yet)"
    hint = (
        f"RETRIEVAL HINT (zh translation of the question; the corpus is Chinese — base your "
        f"search query on it): {retrieval_hint}\n\n"
        if retrieval_hint
        else ""
    )
    msg = (
        f"TOOL FEEDBACK: {last_tool or '(start)'} -> {last_message or 'FIRST ROUND: no search has run yet — plan your first rag_search query.'}\n\n"
        f"{hint}"
        f"PASSAGES RETURNED BY LAST rag_search:\n{render_passages(passages)}\n\n"
        f"EVIDENCE SET: {len(collected)} passage(s) accumulated across all searches\n"
        f"SEARCH HISTORY (do not repeat):\n{history_block}\n\n"
        f"BUDGET: round {rounds_used + 1}/{max_rounds}, tool calls {tool_calls_used}/{max_tool_calls}.\n"
        "Reply with ONE JSON object: {\"thought\": ..., \"tool\": ..., \"args\": {...}}"
    )
    if collected:
        msg += "\nEvidence is available: submit_answer if it covers the question, else search again with a rephrased query."
    return msg


def candidate_identity(item: Dict[str, Any]) -> str:
    """Stable identity key for cross-call dedup (same idea as trajectory_runtime.candidate_id)."""
    return "|".join(
        str(item.get(k, "") or "")
        for k in ("doc_id", "page_idx", "block_id", "sample_id", "group_id", "image_path", "image_id")
    )
