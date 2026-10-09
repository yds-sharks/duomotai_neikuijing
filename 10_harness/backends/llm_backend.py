#!/usr/bin/env python3
"""Brain LLM backends: OpenAI-compatible (vLLM serving Qwen3.5-4B / GPT API) + Mock.

The brain must emit ONE JSON object per step:
    {"thought": "...", "tool": "<tool name>", "args": {...}}
Robust JSON extraction tolerates markdown fences and surrounding prose.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Protocol


class BrainBackend(Protocol):
    def decide(self, messages: List[Dict[str, str]]) -> Dict[str, Any]: ...
    def close(self) -> None: ...


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> Dict[str, Any]:
    """Pull the first JSON object out of a model reply."""
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"no JSON object found in brain reply: {text[:200]!r}")
    return json.loads(m.group(0))


class OpenAIChatBrain:
    """Talks to any OpenAI-compatible chat endpoint (vLLM, or GPT for teacher runs)."""

    def __init__(self, cfg: Dict[str, Any]):
        from openai import OpenAI  # lazy import: smoke tests need no client

        self._client = OpenAI(
            base_url=cfg["base_url"], api_key=cfg.get("api_key", "EMPTY"), timeout=cfg.get("timeout", 120)
        )
        self._model = cfg["model"]
        self._temperature = float(cfg.get("temperature", 0.2))
        self._top_p = float(cfg.get("top_p", 0.9))
        self._max_tokens = int(cfg.get("max_tokens", 512))

    def decide(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self._temperature,
            top_p=self._top_p,
            max_tokens=self._max_tokens,
        )
        text = resp.choices[0].message.content or ""
        return extract_json(text)

    def close(self) -> None: ...


class ScriptedBrain:
    """Plays a fixed decision script — deterministic smoke test of the whole loop."""

    def __init__(self, script: List[Dict[str, Any]]):
        self._script = list(script)
        self._i = 0

    def decide(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        if self._i >= len(self._script):
            return {"thought": "script exhausted", "tool": "submit_answer", "args": {}}
        step = self._script[self._i]
        self._i += 1
        return dict(step)

    def close(self) -> None: ...


def build_brain(cfg: Dict[str, Any]) -> BrainBackend:
    kind = cfg.get("backend", "openai")
    if kind == "openai":
        return OpenAIChatBrain(cfg)
    raise ValueError(f"unknown brain backend: {kind}")
