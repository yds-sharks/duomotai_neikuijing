#!/usr/bin/env python3
"""v0.3 answer-utility reward + GRPO group advantage.

核心奖励：
    r(a) = P_G(a* | q, I_q, E_a) − P_G(a* | q, I_q, ∅)

- P_G 来自冻结 generator（Qwen3-VL-8B, vLLM）的 logprobs
- a* 为正确选项 token（A/B/C/D）
- E_a 为动作 a 产出的证据集：ACCEPT 用 kept 子集；REWRITE 用重检索+再 keep 后的证据集
- baseline P_G(...|∅) = 不给证据时的正确选项概率（整组共享）

⚠️ 护城河：不要引入 evidence_delta / support_delta（gold doc/page 命中 top-k）作为
主项。那是相关性代理信号，会侵蚀 answer-utility 方法的独立性。gold doc/page 命中率
只作为离线分析指标，不进 reward。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from generator_adapter import OpenAICompatibleGenerator
from rag_prompting import build_rag_prompt, valid_options


def _as_int(value: Any) -> Optional[int]:
    try:
        if value is None or str(value) == "":
            return None
        return int(value)
    except Exception:
        return None


def leakage_penalty(query: str, answer: str = "", answer_text: str = "") -> float:
    """轻量约束：rewrite query 含答案字母或直接泄露 gold answer 扣分。

    这是工程约束，论文里不作为主贡献。
    """
    q = str(query or "")
    penalty = 0.0
    if answer and f"答案{answer}" in q:
        penalty += 1.0
    if answer and "最终答案" in q:
        penalty += 1.0
    if answer_text and len(str(answer_text)) >= 2 and str(answer_text) in q:
        penalty += 0.2
    return penalty


def score_answer_utility(
    *,
    p_correct_with_evidence: float,
    p_correct_no_evidence: float,
    query: str = "",
    answer: str = "",
    answer_text: str = "",
    is_original: bool = False,
    agent_action: str = "",
    original_utility: float = 0.0,
) -> Dict[str, Any]:
    """answer-utility 主奖励。

    Args:
        p_correct_with_evidence: 当前动作 E_a 下正确选项概率（从 logprobs 取）。
        p_correct_no_evidence: baseline P_G(a*|∅)，整组共享。
        query: 当前 query（用于 leakage/length 约束）。
        answer / answer_text: 用于 leakage 检查。
        is_original: 是否是 baseline 动作（ACCEPT 原始 top-k）。
        agent_action: ACCEPT / REWRITE。
        original_utility: baseline 动作的 utility（用于 unnecessary_rewrite 检测）。

    Returns:
        {
          "reward": float,
          "components": {
            "answer_utility": float,      # 主项
            "leakage_penalty": float,
            "unnecessary_rewrite_penalty": float,
            "length_penalty": float,
            "p_correct_with": float,
            "p_correct_baseline": float,
          }
        }
    """
    answer_utility = float(p_correct_with_evidence) - float(p_correct_no_evidence)
    leak = leakage_penalty(query, answer=answer, answer_text=answer_text)
    # 不必要改写：原 evidence 已把 utility 抬到较高水平（>=0.15），仍 REWRITE 扣分
    unnecessary = 0.0
    if not is_original and str(agent_action).upper() == "REWRITE" and original_utility >= 0.15:
        unnecessary = 0.1
    length_penalty = max(0.0, (len(query) - 120) / 400.0)
    reward = answer_utility - 0.5 * leak - unnecessary - 0.05 * length_penalty
    return {
        "reward": float(reward),
        "components": {
            "answer_utility": float(answer_utility),
            "leakage_penalty": float(leak),
            "unnecessary_rewrite_penalty": float(unnecessary),
            "length_penalty": float(length_penalty),
            "p_correct_with": float(p_correct_with_evidence),
            "p_correct_baseline": float(p_correct_no_evidence),
        },
    }


def measure_p_correct(
    generator: OpenAICompatibleGenerator,
    *,
    question: str,
    options: Dict[str, Any],
    answer: str,
    query_image_path: str = "",
    evidence: Optional[List[Dict[str, Any]]] = None,
    top_logprobs: int = 5,
) -> Tuple[float, Optional[float]]:
    """用冻结 generator 测量给定证据集 E 下正确选项概率 P_G(a*|q,I_q,E)。

    - evidence 为 None / 空列表 → 测 baseline P_G(a*|∅)
    - evidence 非空 → prompt 里注入证据文本
    返回 (p_correct, logprob)。若答案字母不在 ABCDEF 内，返回 (0.0, None)。
    """
    if not question or not options or not answer:
        return 0.0, None
    valid = valid_options(options)
    if not valid:
        return 0.0, None
    prompt, _ = build_rag_prompt(question, options, evidence or [])
    out = generator.generate_with_logprobs(
        prompt,
        query_image_path,
        top_logprobs=top_logprobs,
        max_tokens=8,
    )
    from generator_adapter import extract_option_probability
    p, lp = extract_option_probability(out["logprobs"], answer)
    return p, lp


def attach_group_advantages(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """GRPO 组内相对优势 A=(r-mean)/std。

    每个 candidate 必须有 "reward" 字段。
    """
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
        c["policy_loss_weight"] = float(-adv)
    best = max(candidates, key=lambda x: float(x.get("reward", 0.0))) if candidates else None
    return {
        "mean_reward": float(mean),
        "std_reward": float(std),
        "best_candidate_id": best.get("candidate_id") if best else "",
        "candidates": candidates,
    }


# ---- 旧接口兼容（保留函数签名，避免调用方立即崩溃，但内部已废弃）----

def evidence_hit(evidence: List[Dict[str, Any]], gold_doc_id: str = "", gold_page_idx: Any = None, page_tolerance: int = 1) -> bool:
    """离线分析用：判断 top-k 是否命中 gold doc/page。

    ⚠️ 不进入 reward，仅作离线分析指标。
    """
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


def score_candidate(**kwargs: Any) -> Dict[str, Any]:
    """旧版 evidence-hit 奖励（已废弃）。

    保留签名是为了让 v0.2 的 smoke 脚本在导入时不报错；
    实际 v0.3 应改用 score_answer_utility。
    """
    return score_answer_utility(
        p_correct_with_evidence=float(kwargs.get("p_correct_with_evidence") or 0.0),
        p_correct_no_evidence=float(kwargs.get("p_correct_no_evidence") or 0.0),
        query=str(kwargs.get("query") or ""),
        answer=str(kwargs.get("answer") or ""),
        answer_text=str(kwargs.get("answer_text") or ""),
        is_original=bool(kwargs.get("is_original") or False),
        agent_action=str(kwargs.get("agent_action") or ""),
        original_utility=float(kwargs.get("original_utility") or 0.0),
    )
