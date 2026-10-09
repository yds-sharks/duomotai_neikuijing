#!/usr/bin/env python3
"""Harness shared dataclasses: tool calls, observations, rounds, trajectories.

Field names follow the v0.5 training chain (05_agentic_rag/agentic/code/schemas.py
and train/rollouts_*.jsonl) so downstream builders (build_sft_from_trajectory.py,
build_dpo_pairs.py, ...) can consume harness trajectories with minimal adapters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """One decision of the central brain."""

    thought: str = ""
    tool: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""  # verbatim brain output (for training / debugging)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Observation:
    """What the brain sees after executing a tool."""

    tool: str
    ok: bool
    candidates: List[Dict[str, Any]] = field(default_factory=list)  # numbered 1..N this round
    n_new: int = 0  # candidates newly added to collected evidence this step
    message: str = ""  # human-readable feedback (errors, dedup notes, budget)
    state: Dict[str, Any] = field(default_factory=dict)  # budget/collected summary

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Round:
    """One brain step: call + observation (the trainable unit)."""

    index: int
    call: ToolCall
    observation: Observation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "call": self.call.to_dict(),
            "observation": self.observation.to_dict(),
        }


@dataclass
class Trajectory:
    """End-to-end record of one question handled by the brain."""

    qid: str
    question: str = ""
    options: Dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    answer_text: str = ""
    query_image_path: str = ""
    original_query: str = ""
    gold_source: Dict[str, Any] = field(default_factory=dict)
    rounds: List[Round] = field(default_factory=list)
    final_evidence: List[Dict[str, Any]] = field(default_factory=list)
    generator_response: str = ""
    prediction: str = ""
    correct: Optional[bool] = None
    u_set: Optional[float] = None  # answer-utility of the final evidence set
    finish_reason: str = ""  # submitted | budget_exhausted | error
    n_rounds: int = 0
    n_tool_calls: int = 0
    error: Optional[Dict[str, str]] = None

    # --- training-chain compatibility aliases (same keys as v0.5 rollouts) ---
    def compat_dict(self) -> Dict[str, Any]:
        """Dict with v0.5-compatible top-level keys plus harness-specific extras."""
        last_candidates: List[Dict[str, Any]] = []
        for r in reversed(self.rounds):
            if r.observation.candidates:
                last_candidates = r.observation.candidates
                break
        d: Dict[str, Any] = {
            "qid": self.qid,
            "question": self.question,
            "options": self.options,
            "answer": self.answer,
            "answer_text": self.answer_text,
            "query_image_path": self.query_image_path,
            "original_query": self.original_query,
            "gold_source": self.gold_source,
            "obs_candidates": last_candidates,
            "final_evidence": self.final_evidence,
            "generator_response": self.generator_response,
            "prediction": self.prediction,
            "correct": self.correct,
            "u_set": self.u_set,
            "finish_reason": self.finish_reason,
            "n_rounds": self.n_rounds,
            "n_tool_calls": self.n_tool_calls,
            "rounds": [r.to_dict() for r in self.rounds],
        }
        if self.error is not None:
            d["error"] = self.error
        return d
