#!/usr/bin/env python3
"""v0.3 agent policy adapter (GPT rollout / inference).

单 policy 一次输出 keep/drop + ACCEPT/REWRITE(+rewrite_query)。
没有中间 reranker：检索结果直接喂给 policy，policy 直接看像素做逐条价值判断。

输出 schema（与设计文档 §3.2 对齐）：
  {"keep": [0, 2, 4], "drop": [1, 3], "action": "ACCEPT"}
  {"keep": [0], "drop": [1,2,3,4], "action": "REWRITE", "rewrite_query": "..."}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from generator_adapter import candidate_chat_urls, image_to_data_uri
from rag_prompting import format_evidence

CODE_DIR = Path(__file__).resolve().parent
AGENTIC_ROOT = CODE_DIR.parent
DEFAULT_API_CONFIG = str(AGENTIC_ROOT / "data_construction" / "api_config.local.json")

SYSTEM_PROMPT = """你是医学多模态 RAG 的 agentic 证据控制器。

给定问题、选项、原始低信息 query、query image 和当前候选证据（每条带 modality/文本/图片/初检分数），
你需要做两件事，**在一次 JSON 输出里完成**：

1. **keep/drop 逐条价值判断**：对每条检索证据，判断它对回答当前问题是有价值（keep）还是误导/无用（drop）。
   关键：医学图像里"视觉相似 ≠ 临床相同"，同部位不同病理经常高度相似。你要能 drop 掉那些"看起来像、
   但临床逻辑/证据文本指向另一个诊断"的高置信误导证据。

2. **ACCEPT/REWRITE 决策**：
   - 如果 kept 证据已经足够唯一确定正确答案 → action="ACCEPT"，不需要改写。
   - 如果 kept 证据不足、歧义、或明显缺少关键判别信息 → action="REWRITE"，并给出更精准的
     rewrite_query 用于下一轮重检索。

约束：
- rewrite_query 不能包含答案字母（A/B/C/D）。
- rewrite_query 不能直接说"正确答案是..."或泄露 gold answer。
- rewrite_query 应注入判别性临床词、视觉特征、部位或选项区分信息。
- 不要生成最终答案，只做证据取舍和检索策略决策。
- 证据索引从 0 开始，必须严格使用题面中显示的索引。
- 只输出 JSON，不输出解释性正文。
"""

USER_TEMPLATE = """请基于 query image 和当前检索证据，做 keep/drop + ACCEPT/REWRITE 决策。

qid: {qid}
query_type: {query_type}
原始 query: {original_query}
问题: {question}
选项: {options}

当前候选证据（{num_evidence} 条，按检索顺序从 0 编号）：
{evidence_block}

若某条证据包含图像，其像素会在下方以“候选证据图像 [idx]”标记后附加。

请严格输出 JSON：
{{
  "keep": [idx_1, idx_2, ...],
  "drop": [idx_1, idx_2, ...],
  "action": "ACCEPT" 或 "REWRITE",
  "rewrite_query": "仅当 action=REWRITE 时填写，否则空字符串",
  "reason": "不超过 100 字，说明 kept 证据是否足够作答，或为什么需要改写"
}}

注意：keep 与 drop 索引必须覆盖全部 {num_evidence} 条证据，且不能重叠。
"""


def load_api_config(path: str = DEFAULT_API_CONFIG) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def _parse_keep_drop(data: Dict[str, Any], num_evidence: int) -> Dict[str, Any]:
    """校验并规整 keep/drop 字段。"""
    keep = data.get("keep") or []
    drop = data.get("drop") or []
    if not isinstance(keep, list):
        keep = []
    if not isinstance(drop, list):
        drop = []
    keep = [int(x) for x in keep if str(x).isdigit()]
    drop = [int(x) for x in drop if str(x).isdigit()]
    keep = sorted(set(i for i in keep if 0 <= i < num_evidence))
    drop = sorted(set(i for i in drop if 0 <= i < num_evidence))
    keep_set = set(keep)
    drop_set = set(drop)
    # 重叠索引归到 keep
    overlap = keep_set & drop_set
    if overlap:
        drop = [i for i in drop if i not in overlap]
        drop_set = set(drop)
    # 未出现的索引默认归到 drop
    for i in range(num_evidence):
        if i not in keep_set and i not in drop_set:
            drop.append(i)
    drop = sorted(set(drop))
    return {"keep": keep, "drop": drop}


class GPTAgent:
    """v0.3 GPT agent：一次输出 keep/drop + ACCEPT/REWRITE。"""

    def __init__(
        self,
        *,
        api_config_path: str = DEFAULT_API_CONFIG,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_tokens: int = 800,
        max_evidence_images: int = 20,
    ):
        cfg = load_api_config(api_config_path)
        self.model = model or cfg.get("model")
        self.base_url = base_url or cfg.get("base_url")
        self.api_key = api_key or cfg.get("api_key")
        self.timeout = int(timeout or cfg.get("timeout", 120))
        self.max_tokens = int(max_tokens)
        self.max_evidence_images = int(max_evidence_images)
        if not self.api_key:
            raise ValueError("Missing API key for GPTAgent")
        if not self.base_url:
            raise ValueError("Missing base_url for GPTAgent")
        if not self.model:
            raise ValueError("Missing model for GPTAgent")
        self.session = requests.Session()
        self.session.trust_env = False

    def _post_json(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        errors: List[str] = []
        for url in candidate_chat_urls(self.base_url):
            try:
                resp = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code in {400, 404, 405} and "response_format" in payload:
                    payload2 = dict(payload)
                    payload2.pop("response_format", None)
                    resp = self.session.post(url, headers=headers, json=payload2, timeout=self.timeout)
                if resp.status_code in {404, 405}:
                    errors.append(f"{url}: {resp.status_code} {resp.text[:200]}")
                    continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"].get("content") or "{}"
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    start = content.find("{")
                    end = content.rfind("}")
                    if start >= 0 and end > start:
                        return json.loads(content[start : end + 1])
                    raise
            except Exception as exc:
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
        raise RuntimeError("GPT agent request failed: " + " | ".join(errors[-3:]))

    def decide(
        self,
        *,
        qid: str,
        query_type: str,
        original_query: str,
        question: str,
        options: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        image_path: str = "",
    ) -> Dict[str, Any]:
        num_evidence = len(evidence)
        user_text = USER_TEMPLATE.format(
            qid=qid,
            query_type=query_type,
            original_query=original_query,
            question=question,
            options=json.dumps(options, ensure_ascii=False),
            num_evidence=num_evidence,
            evidence_block=format_evidence(evidence, max_chars_per_doc=700),
        )
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        image_uri = image_to_data_uri(image_path, max_edge=1024, quality=85, max_bytes=4_000_000)
        if image_uri:
            content.append({"type": "text", "text": "查询图像："})
            content.append({"type": "image_url", "image_url": {"url": image_uri}})
        attached = 0
        for idx, item in enumerate(evidence):
            if attached >= self.max_evidence_images:
                break
            evidence_image = str(item.get("image_path") or "")
            evidence_uri = image_to_data_uri(
                evidence_image,
                max_edge=768,
                quality=80,
                max_bytes=2_000_000,
            )
            if not evidence_uri:
                continue
            content.append({"type": "text", "text": f"候选证据图像 [{idx}]："})
            content.append({"type": "image_url", "image_url": {"url": evidence_uri}})
            attached += 1
        data = self._post_json([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ])

        kd = _parse_keep_drop(data, num_evidence)
        action = str(data.get("action") or "").upper()
        if action not in {"ACCEPT", "REWRITE"}:
            action = "REWRITE"
        rewrite_query = str(data.get("rewrite_query") or "").strip()
        if action == "ACCEPT":
            rewrite_query = ""
        return {
            "keep": kd["keep"],
            "drop": kd["drop"],
            "action": action,
            "rewrite_query": rewrite_query,
            "reason": str(data.get("reason") or ""),
            "raw": data,
        }


# 兼容旧名（smoke 脚本引用）
GPTRewriteAgent = GPTAgent
