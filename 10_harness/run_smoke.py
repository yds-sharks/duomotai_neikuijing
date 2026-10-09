#!/usr/bin/env python3
"""Smoke test of the full agent loop with Mock backends (no GPU / DB / network).

Verifies: tool dispatch, budget accounting, cross-round dedup, invalid-tool recovery,
generator answer + answer-utility, and trajectory field compatibility.

    python run_smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backends.llm_backend import ScriptedBrain  # noqa: E402
from backends.reward_backend import MockGeneratorReward  # noqa: E402
from backends.retrieval_backend import MockRetrievalBackend  # noqa: E402
from brain.agent_brain import AgentBrain  # noqa: E402
from runtime.runner import summarize  # noqa: E402

SMOKE_CONFIG = {
    "retrieval": {"text_k": 20, "image_k": 20},
    "budget": {"max_rounds": 6, "max_tool_calls": 8, "max_keep_per_round": 5, "max_collected": 12},
}


def make_item() -> dict:
    return {
        "qid": "smoke_0001",
        "query_type": "image_organ_identification",
        "question": "该内镜图像主要观察到的是哪个部位？",
        "options": {"A": "食管", "B": "胃", "C": "十二指肠", "D": "胆胰"},
        "answer": "B",
        "answer_text": "胃",
        "query_image_path": "mock://query.jpg",
        "original_query": "这张图是什么部位？",
        "gold_source": {"doc_id": "mockdocimage", "page_idx": 0},
    }


# Script exercises: image retrieve -> keep -> INVALID tool -> text retrieve -> keep -> submit
SCRIPT = [
    {"thought": "先看图像本身", "tool": "image_retrieve", "args": {"k": 5}},
    {"thought": "前两条文本与部位相关", "tool": "keep_evidence", "args": {"keep": [1, 2]}},
    {"thought": "试着调一个不存在的工具", "tool": "no_such_tool", "args": {}},
    {"thought": "换更具体的文本查询", "tool": "text_retrieve", "args": {"query": "胃镜下胃窦黏膜表现"}},
    {"thought": "补一条新证据", "tool": "keep_evidence", "args": {"keep": [1, 999]}},
    {"thought": "证据足够，提交", "tool": "submit_answer", "args": {}},
]


def main() -> int:
    reward = MockGeneratorReward()
    agent = AgentBrain(config=SMOKE_CONFIG, retrieval=MockRetrievalBackend(), brain=ScriptedBrain(SCRIPT), reward=reward)
    item = make_item()
    traj = agent.solve(item)

    # ---- assertions ----
    assert traj["finish_reason"] == "submitted", traj["finish_reason"]
    assert traj["n_tool_calls"] == len(SCRIPT), traj["n_tool_calls"]
    assert traj["n_rounds"] == len(SCRIPT)
    assert len(traj["final_evidence"]) == 3, len(traj["final_evidence"])  # 2 + 1 new (999 out of range)
    assert traj["prediction"] == "B" and traj["correct"] is True and traj["u_set"] == 1.0
    assert traj["obs_candidates"], "compat field obs_candidates must be non-empty"
    r4 = traj["rounds"][4]["observation"]
    assert r4["n_new"] == 1 and "out-of-range" in r4["message"], r4["message"]  # 999 ignored
    assert traj["rounds"][2]["observation"]["ok"] is False  # invalid tool recovered
    # cross-round dedup: round-4 text_retrieve excludes already-kept image hits
    kept_ids = {(c["doc_id"], c["page_idx"]) for c in traj["final_evidence"]}
    assert len(kept_ids) == 3

    print("== smoke trajectory ==")
    print(json.dumps({k: traj[k] for k in ("qid", "finish_reason", "n_rounds", "n_tool_calls", "prediction", "correct", "u_set")}, ensure_ascii=False, indent=2))
    print("== round trace ==")
    for r in traj["rounds"]:
        obs = r["observation"]
        print(f"  r{r['index']}: {r['call']['tool']:>14} ok={obs['ok']} +{obs['n_new']}  {obs['message'][:80]}")
    print("== batch summarize ==")
    print(json.dumps(summarize([traj]), ensure_ascii=False, indent=2))
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
