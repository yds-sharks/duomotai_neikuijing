#!/usr/bin/env python3
"""GPT rewrite-agent adapter for Stage-2 rollout smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from generator_adapter import candidate_chat_urls, image_to_data_uri
from rag_prompting import format_evidence

DEFAULT_API_CONFIG = "/mnt/data_1/yds/多模态/agentic/data_construction/api_config.local.json"

SYSTEM_PROMPT = """你是医学多模态 RAG 的 query rewrite 控制器。
给定问题、选项、原始低信息 query、query image 和当前 top-k 证据，判断当前证据是否足以支持回答。

你只能输出 JSON。
如果证据足够，输出 action=ACCEPT，rewrite_candidates=[]。
如果证据不足或偏题，输出 action=REWRITE，并给出 2-4 条更适合检索的 rewrite_candidates。

约束：
- rewrite query 不能包含答案字母。
- rewrite query 不能直接说“正确答案是...”。
- rewrite query 可以包含题目中的选项内容，因为真实检索时可利用选项区分意图。
- rewrite query 应强调关键视觉特征、部位、病变/操作类型或选项区分信息。
- 不要生成最终答案，只做 ACCEPT/REWRITE 决策。
"""

USER_TEMPLATE = """请判断是否需要改写检索 query。

qid: {qid}
query_type: {query_type}
原始 query: {original_query}
问题: {question}
选项: {options}

当前 top-k 证据：
{evidence_block}

请严格输出 JSON：
{{
  "action": "ACCEPT" 或 "REWRITE",
  "rewrite_candidates": ["...", "..."],
  "failure_type": "none/missing_anatomical_site/missing_lesion_or_finding/missing_procedure_or_operation/missing_spatial_region/off_topic/ambiguous_evidence",
  "reason": "不超过80字"
}}
"""


def load_api_config(path: str = DEFAULT_API_CONFIG) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


class GPTRewriteAgent:
    def __init__(
        self,
        *,
        api_config_path: str = DEFAULT_API_CONFIG,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_tokens: int = 800,
    ):
        cfg = load_api_config(api_config_path)
        self.model = model or cfg.get("model")
        self.base_url = base_url or cfg.get("base_url")
        self.api_key = api_key or cfg.get("api_key")
        self.timeout = int(timeout or cfg.get("timeout", 120))
        self.max_tokens = int(max_tokens)
        if not self.api_key:
            raise ValueError("Missing API key for GPTRewriteAgent")
        if not self.base_url:
            raise ValueError("Missing base_url for GPTRewriteAgent")
        if not self.model:
            raise ValueError("Missing model for GPTRewriteAgent")
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
                    # Some OpenAI-compatible gateways do not accept response_format.
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
        user_text = USER_TEMPLATE.format(
            qid=qid,
            query_type=query_type,
            original_query=original_query,
            question=question,
            options=json.dumps(options, ensure_ascii=False),
            evidence_block=format_evidence(evidence, max_chars_per_doc=700),
        )
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        image_uri = image_to_data_uri(image_path, max_edge=1024, quality=85, max_bytes=4_000_000)
        if image_uri:
            content.append({"type": "image_url", "image_url": {"url": image_uri}})
        data = self._post_json([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ])
        action = str(data.get("action") or "").upper()
        if action not in {"ACCEPT", "REWRITE"}:
            action = "REWRITE"
        rewrites = data.get("rewrite_candidates") or []
        if isinstance(rewrites, str):
            rewrites = [rewrites]
        rewrites = [str(x).strip() for x in rewrites if str(x).strip()]
        if action == "ACCEPT":
            rewrites = []
        return {
            "action": action,
            "rewrite_candidates": rewrites[:4],
            "failure_type": str(data.get("failure_type") or "none"),
            "reason": str(data.get("reason") or ""),
            "raw": data,
        }
