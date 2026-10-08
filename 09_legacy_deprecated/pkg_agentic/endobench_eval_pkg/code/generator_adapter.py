#!/usr/bin/env python3
"""OpenAI-compatible multimodal generator adapter."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
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

    def generate(self, prompt: str, image_path: str = "", *, system_prompt: str = "") -> str:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        image_uri = image_to_data_uri(image_path)
        if image_uri:
            content.append({"type": "image_url", "image_url": {"url": image_uri}})
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        errors: List[str] = []
        for url in candidate_chat_urls(self.base_url):
            try:
                resp = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code in {404, 405}:
                    errors.append(f"{url}: {resp.status_code} {resp.text[:200]}")
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"].get("content") or ""
            except Exception as exc:
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
        raise RuntimeError("Generator request failed: " + " | ".join(errors[-3:]))
