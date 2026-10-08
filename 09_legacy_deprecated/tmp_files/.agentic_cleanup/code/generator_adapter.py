#!/usr/bin/env python3
"""OpenAI-compatible multimodal generator adapter.

v0.3 新增：logprob 接口。冻结 generator（Qwen3-VL-8B，vLLM 部署）除了出文本答案，
还要能返回每个 token 的 logprob，用于计算 answer-utility 奖励
（P_G(a*|E) − P_G(a*|∅)）。

vLLM 的 /v1/chat/completions 支持 logprobs 参数，开启后 choices[0].logprobs.content
返回每个生成 token 的 logprob 和 top_logprobs。
"""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image

CODE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CODE_DIR / "agentic_runtime_config.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def image_to_data_uri(path: str, max_edge: int = 1536, quality: int = 90, max_bytes: int = 8_000_000) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return ""
    try:
        with Image.open(file_path) as img:
            rgb = img.convert("RGB")
            if max(rgb.size) > max_edge:
                rgb.thumbnail((max_edge, max_edge), Image.LANCZOS)
            buf = BytesIO()
            rgb.save(buf, format="JPEG", quality=quality, optimize=True)
            data = buf.getvalue()
    except Exception:
        data = file_path.read_bytes()
    if len(data) > max_bytes:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def candidate_chat_urls(base_url: str) -> List[str]:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return []
    if normalized.endswith("/chat/completions"):
        return [normalized]
    parsed = urlparse(normalized)
    urls = [f"{normalized}/chat/completions"]
    if not parsed.path.rstrip("/").endswith("/v1"):
        urls.append(f"{normalized}/v1/chat/completions")
    deduped: List[str] = []
    seen = set()
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


class OpenAICompatibleGenerator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        gen = self.config.get("generator", {})
        self.base_url = gen.get("base_url", "http://127.0.0.1:8888/v1")
        self.api_key = gen.get("api_key", "EMPTY")
        self.model = gen.get("model")
        self.temperature = float(gen.get("temperature", 0.2))
        self.top_p = float(gen.get("top_p", 0.9))
        self.max_tokens = int(gen.get("max_tokens", 512))
        self.timeout = int(gen.get("timeout", 120))
        self.session = requests.Session()
        self.session.trust_env = False

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        errors: List[str] = []
        for url in candidate_chat_urls(self.base_url):
            try:
                resp = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code in {404, 405}:
                    errors.append(f"{url}: {resp.status_code} {resp.text[:200]}")
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
        raise RuntimeError("Generator request failed: " + " | ".join(errors[-3:]))

    def _build_messages(self, prompt: str, image_path: str = "", *, system_prompt: str = "") -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        image_uri = image_to_data_uri(image_path)
        if image_uri:
            content.append({"type": "image_url", "image_url": {"url": image_uri}})
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def generate(self, prompt: str, image_path: str = "", *, system_prompt: str = "") -> str:
        """只取文本答案（向后兼容）。"""
        messages = self._build_messages(prompt, image_path, system_prompt=system_prompt)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        data = self._post(payload)
        return data["choices"][0]["message"].get("content") or ""

    def generate_with_logprobs(
        self,
        prompt: str,
        image_path: str = "",
        *,
        system_prompt: str = "",
        top_logprobs: int = 5,
        max_tokens: int = 8,
    ) -> Dict[str, Any]:
        """生成答案并返回完整 logprobs，用于 answer-utility 奖励计算。

        vLLM OpenAI-compatible API 接受 logprobs=true 和 top_logprobs=N。
        返回结构：
        {
          "content": "...",  # 生成文本
          "logprobs": {      # choices[0].logprobs
            "content": [     # 每个生成 token 的详情
              {"token": "A", "logprob": -0.3, "top_logprobs": [...]},
              ...
            ]
          },
          "raw": {...}       # 原始响应
        }
        """
        messages = self._build_messages(prompt, image_path, system_prompt=system_prompt)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": int(max_tokens),
            "logprobs": True,
            "top_logprobs": int(top_logprobs),
        }
        data = self._post(payload)
        choice = data["choices"][0]
        return {
            "content": choice["message"].get("content") or "",
            "logprobs": choice.get("logprobs") or {},
            "raw": data,
        }


def extract_option_probability(
    logprobs: Dict[str, Any],
    option_letter: str,
) -> Tuple[float, Optional[float]]:
    """从 logprobs 中提取某个选项字母的概率。

    规则：找到生成内容中**第一个**与选项字母匹配的 token，取其 logprob 转概率。
    如果生成序列里没有该字母（例如模型输出了非 ABCD 字符），返回 (0.0, None)。

    Returns:
        (p_correct, logprob)：
        p_correct 是概率（float），logprob 是原始对数概率（用于调试）。
        若未匹配到任何有效字母，logprob 为 None。
    """
    letter = str(option_letter).strip().upper()
    if letter not in {"A", "B", "C", "D", "E", "F"}:
        return 0.0, None
    token_list = (logprobs or {}).get("content") or []
    if not token_list:
        return 0.0, None
    import math

    for tok in token_list:
        token_text = str(tok.get("token") or "").strip()
        if not token_text:
            continue
        if token_text.upper() == letter:
            lp = float(tok.get("logprob", 0.0))
            return float(math.exp(lp)), lp
    return 0.0, None
