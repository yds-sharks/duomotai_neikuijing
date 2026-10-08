#!/usr/bin/env python3
"""Prompt construction and answer parsing for local RAG generation."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

CHOICES = tuple("ABCDEF")
ANSWER_REGEX = r"最终答案[:：]\s*([A-F])"
RAG_INSTRUCTION = "以下是从知识库中检索到的参考资料，仅供辅助判断。请结合图像、题目和选项综合作答，不要机械照搬参考资料。"
PROMPT_SUFFIX = (
    "请先给出简短分析过程，然后严格按照下面要求作答：\n"
    "1. 最后一行必须单独输出：最终答案：X\n"
    "2. X 必须为 {valid_choices} 之一。\n"
    "3. 如果某选项内容为 null 或 None，则该选项无效，不要选择。\n"
    "4. 最后一行只能写“最终答案：X”，不要跟具体选项内容。"
)


def valid_options(options: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key in CHOICES:
        if key not in options:
            continue
        value = options.get(key)
        if value is None or str(value).strip().lower() in {"", "null", "none"}:
            continue
        out.append(key)
    return out


def format_options(options: Dict[str, Any]) -> str:
    return "\n".join(f"{key}: {options.get(key)}" for key in valid_options(options))


def evidence_text(item: Dict[str, Any]) -> str:
    for key in ("text", "content", "summary", "chunk", "page_content", "passage", "snippet"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def format_evidence(items: Iterable[Dict[str, Any]], max_chars_per_doc: int = 900) -> str:
    blocks: List[str] = []
    for idx, item in enumerate(items, start=1):
        text = evidence_text(item).strip()
        if max_chars_per_doc and len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc].rstrip() + "..."
        score = item.get("score", item.get("ppr_score", 0.0))
        try:
            score_text = f"{float(score):.4f}"
        except Exception:
            score_text = str(score)
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] {item.get('doc_name') or 'unknown'} (p.{item.get('page_idx', '?')}, score={score_text})",
                    text or "[empty evidence]",
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "未检索到可用参考资料。"


def build_rag_prompt(question: str, options: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    valid = valid_options(options)
    valid_choices = "/".join(valid or list(CHOICES[:4]))
    base = f"{question}\n{format_options(options)}".strip()
    suffix = PROMPT_SUFFIX.format(valid_choices=valid_choices)
    prompt = f"{RAG_INSTRUCTION}\n\n参考资料：\n{format_evidence(evidence)}\n\n{base}\n\n{suffix}"
    return prompt, valid


def parse_prediction(text: Any, valid: List[str]) -> str:
    if text is None:
        return ""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return ""
    match = re.search(ANSWER_REGEX, lines[-1])
    if not match:
        return ""
    pred = match.group(1)
    return pred if not valid or pred in set(valid) else ""
