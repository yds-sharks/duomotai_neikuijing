#!/usr/bin/env python3
"""Reward backend: answer-utility of an evidence set (aligned with reward_model.score_candidate).

u = 1.0 if the frozen generator answers correctly with this evidence set, else 0.0.
An optional logprob-weighted variant is provided for finer-grained GRPO signals.
Generation itself goes through the same OpenAI-compatible endpoint as the brain
(the unified Qwen3.5-4B in Round 2).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Protocol


class RewardBackend(Protocol):
    def answer_utility(
        self,
        question: str,
        options: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        gold_answer: str = "",
        image_path: str = "",
    ) -> float: ...
    def generate(self, question: str, options: Dict[str, Any], evidence: List[Dict[str, Any]], image_path: str = "") -> Dict[str, Any]: ...
    def utility_from_generation(self, out: Dict[str, Any], gold_answer: str) -> float: ...
    def close(self) -> None: ...


_LETTER_RE = re.compile(r"\b([A-J])\b")


def extract_option_letter(text: str, options: Dict[str, Any]) -> str:
    """Pull the predicted option letter from a free-form generator reply."""
    up = text.upper()
    for letter in sorted(options, key=len, reverse=True):
        if letter.upper() in up:
            return letter
    m = _LETTER_RE.search(up)
    return m.group(1) if m else ""


def render_generator_prompt(
    question: str, options: Dict[str, Any], evidence: List[Dict[str, Any]], image_attached: bool = False
) -> str:
    """Standalone renderer mirroring rag_prompting.build_rag_prompt's contract."""
    opt_lines = "\n".join(f"{k}. {v}" for k, v in options.items())
    ev_lines = []
    for i, ev in enumerate(evidence, 1):
        src = ev.get("source") or ev.get("origin") or "text"
        body = ev.get("text", "")
        ev_lines.append(f"[{i}] ({src}) {body}")
    ev_block = "\n".join(ev_lines) if ev_lines else "（未收集到证据）"
    img_note = (
        "题目的内镜图像已作为附图输入——请直接读图，并与下面的证据结合判断。\n\n"
        if image_attached
        else ""
    )
    return (
        "你是医学问答助手。请依据下方证据回答这道多选题，只回复正确选项的字母。\n\n"
        f"{img_note}"
        f"问题: {question}\n选项:\n{opt_lines}\n\n证据:\n{ev_block}\n\n答案:"
    )


class OpenAIGeneratorReward:
    """Generates via an OpenAI-compatible endpoint and scores answer-utility."""

    def __init__(self, cfg: Dict[str, Any]):
        from openai import OpenAI  # lazy import

        self._client = OpenAI(
            base_url=cfg["base_url"], api_key=cfg.get("api_key", "EMPTY"), timeout=cfg.get("timeout", 120)
        )
        self._model = cfg["model"]
        self._temperature = float(cfg.get("temperature", 0.2))
        self._max_tokens = int(cfg.get("max_tokens", 512))

    def generate(self, question: str, options: Dict[str, Any], evidence: List[Dict[str, Any]], image_path: str = "") -> Dict[str, Any]:
        prompt = render_generator_prompt(question, options, evidence, image_attached=bool(image_path))
        content: Any = prompt
        if image_path:  # OpenAI vision format (base64 data URI)
            import base64

            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": content}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        return {"response": text, "prediction": extract_option_letter(text, options)}

    def utility_from_generation(self, out: Dict[str, Any], gold_answer: str) -> float:
        return 1.0 if gold_answer and out.get("prediction") == gold_answer else 0.0

    def answer_utility(
        self,
        question: str,
        options: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        gold_answer: str = "",
        image_path: str = "",
    ) -> float:
        out = self.generate(question, options, evidence, image_path=image_path)
        return self.utility_from_generation(out, gold_answer)

    def close(self) -> None: ...


class MockGeneratorReward:
    """Deterministic mock: answers 'A' unless evidence count >= 2, then gold answer."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.last_prediction = ""

    def generate(self, question: str, options: Dict[str, Any], evidence: List[Dict[str, Any]], image_path: str = "") -> Dict[str, Any]:
        letters = sorted(options)
        pred = letters[1] if len(evidence) >= 2 and len(letters) > 1 else letters[0]
        self.last_prediction = pred
        return {"response": f"[mock] {pred}", "prediction": pred}

    def utility_from_generation(self, out: Dict[str, Any], gold_answer: str) -> float:
        return 1.0 if gold_answer and out.get("prediction") == gold_answer else 0.0

    def answer_utility(
        self,
        question: str,
        options: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        gold_answer: str = "",
        image_path: str = "",
    ) -> float:
        return self.utility_from_generation(self.generate(question, options, evidence, image_path=image_path), gold_answer)

    def close(self) -> None: ...


def build_reward(cfg: Dict[str, Any]) -> RewardBackend:
    kind = cfg.get("backend", "openai")
    if kind == "openai":
        return OpenAIGeneratorReward(cfg)
    if kind == "transformers":
        from backends.transformers_backend import TransformersChat

        return TransformersChat(cfg)
    if kind == "mock":
        return MockGeneratorReward(cfg)
    raise ValueError(f"unknown reward backend: {kind}")
