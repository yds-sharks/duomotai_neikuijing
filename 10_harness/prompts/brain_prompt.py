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
        "你是一个多模态医学 RAG 系统（内镜问答）的中央大脑。\n"
        "每条消息会附带题目的内镜图像（若有）——请直接读图；对视觉类问题（器官识别、"
        "病灶描述、病变定位）图像往往是决定性的。\n"
        "你不要凭记忆作答，也不亲自管理证据：唯一工具 rag_search 会运行完整 RAG 管道"
        "并返回筛选后的文段。你的职责是查询规划：构造最佳检索 query、决定是否换一种"
        "问法再检索、判断证据是否足够。生成器只依据已积累的证据集作答。\n\n"
        "可用工具\n" + tools_block + "\n\n"
        "协议\n"
        "每轮只回复一个 JSON 对象，不要输出其他任何内容：\n"
        '{"thought": "<简要推理>", "tool": "<工具名>", "args": {...}}\n\n'
        "策略\n"
        "1. 通常先调用 rag_search。检索语料是中文医学文献——query 必须用中文写"
        "（依据检索提示和图像所见来构造）。题目图像（若有）会自动参与检索。\n"
        "2. 审阅返回的文段：若很可能改变答案、或证据已覆盖问题，就调用 submit_answer；"
        "否则用更具体的中文改写 query 再次 rag_search（不要与之前的 query 逐字重复）。\n"
        "3. 预算紧张：剩余调用很少时，优先提交而不是继续探索。\n"
    )


def _preview(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def render_passages(passages: List[Dict[str, Any]]) -> str:
    if not passages:
        return "（本次调用未返回文段）"
    lines = []
    for i, p in enumerate(passages[:MAX_PASSAGE_SHOWN], 1):
        origin = p.get("origin", p.get("source", "?"))
        score = p.get("score")
        score_s = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
        lines.append(f"[{i}] ({origin}, score={score_s}) {_preview(p.get('text', ''))}")
    if len(passages) > MAX_PASSAGE_SHOWN:
        lines.append(f"...（另有 {len(passages) - MAX_PASSAGE_SHOWN} 条未展示）")
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
    final_round: bool = False,
) -> str:
    history = search_history[-MAX_HISTORY_SHOWN:]
    history_block = "\n".join(f"  - {h}" for h in history) if history else "  （尚无检索）"
    hint = (
        f"检索提示（题目的中文翻译——请用中文写 rag_search 的 query，可结合本提示与图像所见改写）：{retrieval_hint}\n\n"
        if retrieval_hint
        else ""
    )
    msg = (
        f"上轮工具反馈: {last_tool or '（开始）'} -> {last_message or '第一轮：尚未检索——请规划你的第一次 rag_search 查询。'}\n\n"
        f"{hint}"
        f"上次 rag_search 返回的文段:\n{render_passages(passages)}\n\n"
        f"证据集: 已累计 {len(collected)} 条文段\n"
        f"检索历史（不要重复）:\n{history_block}\n\n"
        f"预算: 第 {rounds_used + 1}/{max_rounds} 轮，工具调用 {tool_calls_used}/{max_tool_calls} 次。\n"
        '只回复一个 JSON 对象: {"thought": ..., "tool": ..., "args": {...}}'
    )
    if collected:
        msg += "\n已有证据：若足以覆盖问题请 submit_answer，否则用中文改写 query 再检索。"
    if final_round:
        msg += "\n最后一轮：这是你最后一次行动——立即 submit_answer，用现有证据作答。继续检索没有意义。"
    return msg


def candidate_identity(item: Dict[str, Any]) -> str:
    """Stable identity key for cross-call dedup (same idea as trajectory_runtime.candidate_id)."""
    return "|".join(
        str(item.get(k, "") or "")
        for k in ("doc_id", "page_idx", "block_id", "sample_id", "group_id", "image_path", "image_id")
    )
