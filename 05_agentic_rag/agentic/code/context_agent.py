#!/usr/bin/env python3
"""v0.3 keep/drop evidence-controller agent (GPT teacher).

This is the reconstructed Stage-2 controller used to generate per-round
trajectories for SFT cold-start. Unlike the legacy v0.2 ``GPTRewriteAgent``
(which only emitted ACCEPT/REWRITE + rewrite_candidates), this agent performs a
per-candidate *value judgement*:

    - keep / drop : which of the shown top-k candidates actually help answer THIS
      question (0-based indices into the displayed candidate list)
    - action      : ACCEPT (kept evidence is enough) or REWRITE (need better
      retrieval, provide a single ``rewrite_query``)

The agent sees the query image and, faithfully, the candidate evidence images
(up to ``max_evidence_images``), mirroring what the local policy will later be
trained on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from generator_adapter import candidate_chat_urls, image_to_data_uri

DEFAULT_API_CONFIG = "/mnt/data_1/yds/多模态/agentic/data_construction/api_config.local.json"

SYSTEM_PROMPT_V11 = """你是医学多模态 RAG 的证据控制器（evidence controller）。
给定问题、选项、query image、当前检索 query，以及本轮检索到的候选证据（按 [0]、[1]... 编号），
你要对每一条候选做“价值判断”，再决定是否需要改写检索。

你只能输出 JSON。判断分两步：
1) keep / drop：逐条判断候选证据是否对“回答本题”有帮助。keep 放入有用证据的编号，drop 放入无用/偏题/冗余证据的编号。编号是候选列表里的 0-based 下标。
2) action：
   - 若保留的证据已足以支持作答 → action="ACCEPT"，rewrite_query=""。
   - 若证据不足、偏题或缺少关键信息 → action="REWRITE"，并给出一条更适合检索的 rewrite_query。

约束：
- keep 与 drop 合起来应覆盖所有候选编号，且互不重叠；越界/重复的编号会被忽略。
- keep 最多 3 条：从候选中挑出对回答本题最有帮助的证据（通常 1~3 条；仅当全部无用时才为空），其余放入 drop。
- 若某条候选被标为“检索历史”（search_history），它不是真实证据，而是上一轮的检索线索，请把它 keep 下来作为改写方向参考。
- 若某条候选被标为“retained”，它是上一轮已保留的证据（M+），可继续 keep 或 drop。
- rewrite_query 不能包含答案字母，不能直接说“正确答案是…”。
- rewrite_query 可以包含题目选项内容（真实检索时可用选项区分意图），应强调关键视觉特征、部位、病变/操作类型或选项区分信息。
- 不要生成最终答案，只做 keep/drop + ACCEPT/REWRITE 决策。
- reason 不超过 80 字。
"""

USER_TEMPLATE = """请对本轮候选证据做 keep/drop 价值判断，并决定 ACCEPT/REWRITE。

qid: {qid}
query_type: {query_type}
原始 query: {original_query}
当前检索 query: {current_query}
问题: {question}
选项: {options}

本轮候选证据（0-based 编号）：
{candidate_block}

请严格输出 JSON（keep 最多 3 条，挑最有帮助的）：
{{
  "keep": [0, 2],
  "drop": [1, 3],
  "action": "ACCEPT" 或 "REWRITE",
  "rewrite_query": "若 ACCEPT 则为空字符串",
  "reason": "不超过80字"
}}
"""


def load_api_config(path: str = DEFAULT_API_CONFIG) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_candidates(candidates: List[Dict[str, Any]], *, max_chars_per_doc: int = 700) -> str:
    """0-based numbered candidate block shown to the agent."""
    if not candidates:
        return "（本轮没有候选证据）"
    blocks: List[str] = []
    for idx, item in enumerate(candidates):
        origin = str(item.get("origin") or item.get("source") or "retrieval")
        text = str(item.get("text") or item.get("content") or "").strip()
        if max_chars_per_doc and len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc].rstrip() + "..."
        score = item.get("score", 0.0)
        try:
            score_text = f"{float(score):.4f}"
        except Exception:
            score_text = str(score)
        head = f"[{idx}] ({origin}) {item.get('doc_name') or 'unknown'} (p.{item.get('page_idx', '?')}, score={score_text})"
        has_img = "  <含证据图像>" if str(item.get("image_path") or "") else ""
        blocks.append(f"{head}{has_img}\n{text or '[empty evidence]'}")
    return "\n\n".join(blocks)


class GPTContextAgent:
    def __init__(
        self,
        *,
        api_config_path: str = DEFAULT_API_CONFIG,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 180,
        max_tokens: int = 900,
        max_evidence_images: int = 8,
    ):
        cfg = load_api_config(api_config_path)
        self.model = model or cfg.get("model")
        self.base_url = base_url or cfg.get("base_url")
        self.api_key = api_key or cfg.get("api_key")
        self.timeout = int(timeout or cfg.get("timeout", 180))
        self.max_tokens = int(max_tokens)
        self.max_evidence_images = int(max_evidence_images)
        if not self.api_key or self.api_key == "PASTE_YOUR_KEY_HERE":
            raise ValueError("Missing/placeholder API key for GPTContextAgent")
        if not self.base_url:
            raise ValueError("Missing base_url for GPTContextAgent")
        if not self.model:
            raise ValueError("Missing model for GPTContextAgent")
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
        raise RuntimeError("GPT context agent request failed: " + " | ".join(errors[-3:]))

    @staticmethod
    def _sanitize_keep_drop(keep: Any, drop: Any, n: int) -> Dict[str, List[int]]:
        def _clean(seq: Any) -> List[int]:
            out: List[int] = []
            seen = set()
            if isinstance(seq, (list, tuple)):
                for x in seq:
                    try:
                        i = int(x)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= i < n and i not in seen:
                        seen.add(i)
                        out.append(i)
            return out

        keep_l = _clean(keep)
        drop_l = [i for i in _clean(drop) if i not in set(keep_l)]
        assigned = set(keep_l) | set(drop_l)
        # unassigned candidates default to drop (conservative)
        for i in range(n):
            if i not in assigned:
                drop_l.append(i)
        return {"keep": sorted(keep_l), "drop": sorted(drop_l)}

    def decide(
        self,
        *,
        qid: str,
        query_type: str,
        original_query: str,
        current_query: str,
        question: str,
        options: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        image_path: str = "",
    ) -> Dict[str, Any]:
        n = len(candidates)
        user_text = USER_TEMPLATE.format(
            qid=qid,
            query_type=query_type,
            original_query=original_query,
            current_query=current_query,
            question=question,
            options=json.dumps(options, ensure_ascii=False),
            candidate_block=format_candidates(candidates),
        )
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        q_uri = image_to_data_uri(image_path, max_edge=1024, quality=85, max_bytes=4_000_000)
        if q_uri:
            content.append({"type": "text", "text": "查询图像："})
            content.append({"type": "image_url", "image_url": {"url": q_uri}})
        attached = 0
        for idx, item in enumerate(candidates):
            if attached >= self.max_evidence_images:
                break
            ip = str(item.get("image_path") or "")
            if not ip:
                continue
            uri = image_to_data_uri(ip, max_edge=768, quality=85, max_bytes=4_000_000)
            if not uri:
                continue
            content.append({"type": "text", "text": f"候选证据图像 [{idx}]："})
            content.append({"type": "image_url", "image_url": {"url": uri}})
            attached += 1
        data = self._post_json([
            {"role": "system", "content": SYSTEM_PROMPT_V11},
            {"role": "user", "content": content},
        ])
        action = str(data.get("action") or "").upper()
        if action not in {"ACCEPT", "REWRITE"}:
            action = "REWRITE"
        kd = self._sanitize_keep_drop(data.get("keep"), data.get("drop"), n)
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
