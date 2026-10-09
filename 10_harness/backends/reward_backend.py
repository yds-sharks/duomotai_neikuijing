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
    ) -> float: ...
    def generate(self, question: str, options: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]: ...
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


def render_generator_prompt(question: str, options: Dict[str, Any], evidence: List[Dict[str, Any]]) -> str:
    """Standalone renderer mirroring rag_prompting.build_rag_prompt's contract."""
    opt_lines = "\n".join(f"{k}. {v}" for k, v in options.items())
    ev_lines = []
    for i, ev in enumerate(evidence, 1):
        src = ev.get("source") or ev.get("origin") or "text"
        body = ev.get("text", "")
        ev_lines.append(f"[{i}] ({src}) {body}")
    ev_block = "\n".join(ev_lines) if ev_lines else "(no evidence collected)"
    return (
        "You are a medical QA assistant. Answer the multiple-choice question using the "
        "evidence below. Reply with the option letter only.\n\n"
        f"Question: {question}\nOptions:\n{opt_lines}\n\nEvidence:\n{ev_block}\n\nAnswer:"
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

    def generate(self, question: str, options: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = render_generator_prompt(question, options, evidence)
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
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
    ) -> float:
        out = self.generate(question, options, evidence)
        return self.utility_from_generation(out, gold_answer)

    def close(self) -> None: ...


class MockGeneratorReward:
    """Deterministic mock: answers 'A' unless evidence count >= 2, then gold answer."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.last_prediction = ""

    def generate(self, question: str, options: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    ) -> float:
        return self.utility_from_generation(self.generate(question, options, evidence), gold_answer)

    def close(self) -> None: ...


def build_reward(cfg: Dict[str, Any]) -> RewardBackend:
    kind = cfg.get("backend", "openai")
    if kind == "openai":
        return OpenAIGeneratorReward(cfg)
    if kind == "mock":
        return MockGeneratorReward(cfg)
    raise ValueError(f"unknown reward backend: {kind}")
