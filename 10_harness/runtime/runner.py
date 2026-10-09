#!/usr/bin/env python3
"""HarnessRunner: assemble backends from config and run batches to a trajectory jsonl."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from backends.rag_backend import RagPipelineBackend, build_rag_pipeline
from backends.llm_backend import build_brain
from backends.reward_backend import build_reward
from brain.agent_brain import AgentBrain


class HarnessRunner:
    def __init__(self, config: Dict[str, Any], mock: bool = False):
        self.config = config
        self.mock = mock
        self.rag = build_rag_pipeline(config, mock=mock)
        brain_cfg = dict(config.get("brain", {}))
        if mock:
            brain_cfg = dict(brain_cfg, backend="mock")  # runner callers inject scripted brains directly
        self.brain = build_brain(brain_cfg)
        reward_cfg = dict(config.get("generator", {}))
        if mock:
            reward_cfg = dict(reward_cfg, backend="mock")
        self.reward = build_reward(reward_cfg)

    def make_agent(self, brain_backend=None) -> AgentBrain:
        return AgentBrain(
            config=self.config,
            rag=self.rag,
            brain=brain_backend or self.brain,
            reward=self.reward,
        )

    def run_batch(
        self,
        agent: AgentBrain,
        items: Iterable[Dict[str, Any]],
        out_path: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        out_fp = open(out_path, "a", encoding="utf-8") if out_path else None
        t0 = time.time()
        try:
            for i, item in enumerate(items):
                if limit is not None and i >= limit:
                    break
                try:
                    traj = agent.solve(item)
                except Exception:
                    traj = {
                        "qid": str(item.get("qid", f"item{i}")),
                        "error": {"stage": "solve", "detail": traceback_fmt()},
                        "finish_reason": "error",
                    }
                results.append(traj)
                if out_fp:
                    out_fp.write(json.dumps(traj, ensure_ascii=False) + "\n")
                    out_fp.flush()
        finally:
            if out_fp:
                out_fp.close()
        self.last_wall_seconds = time.time() - t0
        return results

    def close(self) -> None:
        self.rag.close()
        self.brain.close()
        self.reward.close()


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate accuracy / behaviour stats over a batch of trajectories."""
    n = len(results)
    scored = [r for r in results if r.get("correct") is not None]
    submitted = [r for r in results if r.get("finish_reason") == "submitted"]
    return {
        "n": n,
        "accuracy": sum(1 for r in scored if r["correct"]) / len(scored) if scored else None,
        "n_scored": len(scored),
        "submit_rate": len(submitted) / n if n else None,
        "avg_rounds": sum(r.get("n_rounds", 0) for r in results) / n if n else None,
        "avg_tool_calls": sum(r.get("n_tool_calls", 0) for r in results) / n if n else None,
        "avg_collected": sum(len(r.get("final_evidence", [])) for r in results) / n if n else None,
        "errors": sum(1 for r in results if r.get("error")),
    }


def write_summary(path: str, summary: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    payload = dict(summary)
    if extra:
        payload.update(extra)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def traceback_fmt() -> str:
    import traceback

    return traceback.format_exc(limit=5)
