#!/usr/bin/env python3
"""Prompt templates for Stage-1 QA generation and verification.

These templates are intentionally separate from the API caller so they can be
reviewed and optimized before expensive API runs.
"""

from __future__ import annotations

import json
from typing import Any, Dict

QUERY_TYPE_DESCRIPTIONS = {
    "anatomical_site_recognition": "识别图像主要解剖部位、器官、亚部位或可见结构。包含粗部位、亚部位和结构识别三个难度层次。",
    "lesion_or_finding_identification": "识别病变、异常表现、内镜下发现或诊断相关视觉特征。优先构造同部位不同病理/表现的 hard negatives。",
    "procedure_or_operation_recognition": "识别检查步骤、治疗操作、器械行为或正在处理的结构。必须有可见器械/操作场景，或 PDF 上下文明说该图对应某个操作步骤。",
    "spatial_region_understanding": "识别病灶、器械或操作区域的空间位置。只有存在明确方位/区域证据时才构造，不为凑数强行生成。",
}

QA_GENERATION_SYSTEM_PROMPT = """你是医学多模态 QA 数据构造专家。
你的任务是基于给定 query image、图文对描述、图片邻近文本、PDF 相邻两页上下文和元信息，判断该来源是否适合构造一道真实、自然、图像依赖的医学四选一问题。

必须遵守：
1. 你可以提出候选 question、options 和 candidate_answer，但最终正确答案不是由你单独决定，必须由来源证据和独立 verifier 严格约束。
2. 你必须给出 evidence_basis，说明候选答案如何被图像、图文对和上下文支持；证据不足必须 reject。
3. 不要把 organ_tags、labels 或 primary_knowledge_type 机械改写成问题；这些字段只能作为弱提示。
4. 如果来源证据不足、图像依赖低、多个答案都可能正确、选项粒度难以统一，必须 reject。
5. 题目应像真实用户面对医学图像时会问的问题，避免内部 taxonomy 分类题。
6. low_information_query_seed 以执行层给出的规则默认值为准；除非默认值明显不适合，否则不要改得更具体。
7. 只输出 JSON，不要输出解释性正文。"""

QA_GENERATION_USER_TEMPLATE = """请基于下面的来源信息生成一道医学多模态四选一候选题，或判断该来源不适合生成问题。

候选题型：{query_type}
题型定义：{query_type_description}
候选子类型提示：{candidate_subtype_hint}
规则默认 low_information_query_seed：{default_low_information_query_seed}

来源信息：
- source_id: {source_id}
- sample_id: {sample_id}
- doc_id: {doc_id}
- doc_name: {doc_name}
- page_idx: {page_idx}
- pdf_context_pages: {pdf_context_pages}
- organ_tags: {organ_tags}
- primary_knowledge_type: {primary_knowledge_type}
- secondary_knowledge_types: {secondary_knowledge_types}
- labels: {labels}

图文对描述或图注候选：
{caption_or_pair_text}

图片邻近文本块（优先证据）：
{image_local_context_text}

PDF 相邻两页上下文（fallback/补充证据）：
{pdf_context_text}

问题设计要求：
- question 必须简短、自然、低信息，不要写“根据图注/根据上下文/根据证据”。
- question 不能包含正确答案文本，不能复述图文对描述或 PDF 句子。
- options 必须是 A/B/C/D 四个选项，且同类型、同粒度、互斥。
- candidate_answer 必须是 A/B/C/D，candidate_answer_text 必须严格等于 options[candidate_answer]。
- 候选答案必须能被 query image 与来源证据共同支持；你必须写 evidence_basis。
- lesion/finding 类题优先使用同部位不同病理或相似表现作为 hard negatives，避免过于容易的跨器官干扰项。
- anatomical_site 类题不要只做粗部位，可在证据支持时覆盖亚部位或结构识别。
- procedure/operation 类题必须有可见器械/操作场景，或上下文明说该图对应某个操作步骤；否则 reject。
- spatial/region 类题必须有明确方位、区域或箭头所指等证据；否则 reject。
- image_dependency 必须为 high、medium 或 low。low 应 reject。
- low_information_query_seed 默认使用规则值；API 可给 alternative_low_information_query_seed，但不能让原始 query 过于具体。

不允许生成的问题类型：
- “这张医学图像主要对应下列哪类内容？”
- “该图像属于病变特征/检查操作/治疗操作中的哪一类？”
- 任何只是在问 primary_knowledge_type 或内部标签的问题。

请严格输出 JSON：
{{
  "accept_source": true/false,
  "reject_reason": "如果 accept_source=false，简短说明原因；否则为空字符串",
  "query_type": "{query_type}",
  "question": "",
  "options": {{"A": "", "B": "", "C": "", "D": ""}},
  "candidate_answer": "A/B/C/D",
  "candidate_answer_text": "",
  "evidence_basis": "不超过120字，说明候选答案如何由图像和来源证据支持；不要复制长原文",
  "image_dependency": "high/medium/low",
  "ambiguity": true/false,
  "alternative_low_information_query_seed": "可为空；只有规则默认值明显不合适时才给出"
}}"""

QA_VERIFICATION_SYSTEM_PROMPT = """你是医学多模态 QA 数据质检员。
你会看到 query image、题目、选项、图文对描述、图片邻近文本和 PDF 两页上下文，但**看不到生成器提出的候选答案**。

必须遵守：
1. 不要重新创造新题目，不要润色题目，只做审核。
2. 你要**独立作答**：结合 query image 与来源证据，从 A/B/C/D 中选出你认为唯一正确的答案 verifier_answer。这是本项目答案治理的核心——gold answer 只有在你的独立作答与生成器候选一致、且证据支持时才成立。
3. 如果答案不能由图像和来源证据唯一确定，或你无法自信作答，answer_supported 置 false。
4. 如果题目只靠文本标签可答、图像依赖低，reject。
5. 如果 organ_tags 多标签导致主答案不唯一，reject。
6. 如果题面泄露 caption/PDF 证据，reject。
7. 如果多个选项都可能正确或选项粒度不一致，reject。
8. 只输出 JSON，不要输出解释性正文。"""

QA_VERIFICATION_USER_TEMPLATE = """请对下面这道医学多模态四选一候选题**独立作答并审核**是否可以进入训练集。
注意：这里不提供候选答案，你必须自己根据 query image 和来源证据判断唯一正确选项。

候选题：
- query_type: {query_type}
- question: {question}
- options: {options}
- rule_low_information_query_seed: {low_information_query_seed}

来源信息：
- source_id: {source_id}
- sample_id: {sample_id}
- doc_id: {doc_id}
- doc_name: {doc_name}
- page_idx: {page_idx}
- pdf_context_pages: {pdf_context_pages}
- organ_tags: {organ_tags}
- primary_knowledge_type: {primary_knowledge_type}
- secondary_knowledge_types: {secondary_knowledge_types}
- labels: {labels}

图文对描述或图注候选：
{caption_or_pair_text}

图片邻近文本块（优先证据）：
{image_local_context_text}

PDF 相邻两页上下文（fallback/补充证据）：
{pdf_context_text}

审核标准：
- verifier_answer: 你独立选出的唯一正确选项（A/B/C/D）。
- verifier_confidence: high/medium/low，你对该独立作答的把握。
- answer_supported: verifier_answer 是否能被图像和来源证据唯一且充分支持。
- image_dependency: high/medium/low，低图像依赖不能进入训练集。
- question_naturalness: high/medium/low，题目是否像真实用户面对图像时会问的问题。
- benchmark_style_match: 是否贴近解剖部位、病变/发现、操作/流程、空间区域这类真实题型。
- multi_organ_ambiguity: 多器官标签或图像/文本是否导致主答案不唯一。
- option_quality: good/bad，选项是否同类型、同粒度、互斥。
- text_leakage: 题面是否泄露图文对或 PDF 证据。
- all_options_same_granularity: 四个选项粒度是否一致。
- multiple_correct_options: 是否存在多个可能正确选项。

请严格输出 JSON：
{{
  "verifier_answer": "A/B/C/D",
  "verifier_confidence": "high/medium/low",
  "answer_supported": true/false,
  "image_dependency": "high/medium/low",
  "question_naturalness": "high/medium/low",
  "benchmark_style_match": true/false,
  "multi_organ_ambiguity": true/false,
  "option_quality": "good/bad",
  "text_leakage": true/false,
  "all_options_same_granularity": true/false,
  "multiple_correct_options": true/false,
  "reason": "不超过120字"
}}"""


def render_generation_user_prompt(source: Dict[str, Any], query_type: str) -> str:
    return QA_GENERATION_USER_TEMPLATE.format(
        query_type=query_type,
        query_type_description=QUERY_TYPE_DESCRIPTIONS.get(query_type, ""),
        candidate_subtype_hint=source.get("candidate_subtype_hint", ""),
        default_low_information_query_seed=source.get("low_information_query_seed", ""),
        source_id=source.get("source_id", ""),
        sample_id=source.get("sample_id", ""),
        doc_id=source.get("doc_id", ""),
        doc_name=source.get("doc_name", ""),
        page_idx=source.get("page_idx", ""),
        pdf_context_pages=json.dumps(source.get("pdf_context_pages", []), ensure_ascii=False),
        organ_tags=json.dumps(source.get("organ_tags", []), ensure_ascii=False),
        primary_knowledge_type=source.get("primary_knowledge_type", ""),
        secondary_knowledge_types=json.dumps(source.get("secondary_knowledge_types", []), ensure_ascii=False),
        labels=json.dumps(source.get("labels", {}), ensure_ascii=False),
        caption_or_pair_text=source.get("caption_or_pair_text", ""),
        image_local_context_text=source.get("image_local_context_text", ""),
        pdf_context_text=source.get("pdf_context_text", ""),
    )


def render_verification_user_prompt(candidate: Dict[str, Any]) -> str:
    """Render the blind-answer verification prompt.

    Deliberately omits candidate_answer / candidate_answer_text / evidence_basis so
    the verifier answers independently; the caller compares verifier_answer against
    the generator's candidate_answer to decide gold.
    """
    source = candidate.get("source") or candidate
    return QA_VERIFICATION_USER_TEMPLATE.format(
        query_type=candidate.get("query_type") or candidate.get("candidate_query_type", ""),
        question=candidate.get("question", ""),
        options=json.dumps(candidate.get("options", {}), ensure_ascii=False),
        low_information_query_seed=candidate.get("low_information_query_seed") or source.get("low_information_query_seed", ""),
        source_id=source.get("source_id", ""),
        sample_id=source.get("sample_id", ""),
        doc_id=source.get("doc_id", ""),
        doc_name=source.get("doc_name", ""),
        page_idx=source.get("page_idx", ""),
        pdf_context_pages=json.dumps(source.get("pdf_context_pages", []), ensure_ascii=False),
        organ_tags=json.dumps(source.get("organ_tags", []), ensure_ascii=False),
        primary_knowledge_type=source.get("primary_knowledge_type", ""),
        secondary_knowledge_types=json.dumps(source.get("secondary_knowledge_types", []), ensure_ascii=False),
        labels=json.dumps(source.get("labels", {}), ensure_ascii=False),
        caption_or_pair_text=source.get("caption_or_pair_text", ""),
        image_local_context_text=source.get("image_local_context_text", ""),
        pdf_context_text=source.get("pdf_context_text", ""),
    )
