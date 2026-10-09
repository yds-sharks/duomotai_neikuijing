#!/usr/bin/env python3
"""Smoke test of the single-tool agent loop with Mock backends (no GPU / DB / network).

The brain's action space is exactly two tools: rag_search (full RAG pipeline,
internally selects evidence) and submit_answer.

    python run_smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backends.llm_backend import ScriptedBrain  # noqa: E402
from backends.rag_backend import MockRagPipeline  # noqa: E402
from backends.reward_backend import MockGeneratorReward  # noqa: E402
from brain.agent_brain import AgentBrain  # noqa: E402
from runtime.runner import summarize  # noqa: E402

SMOKE_CONFIG = {
    "budget": {"max_rounds": 6, "max_tool_calls": 8, "max_collected": 12},
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
        "gold_source": {"doc_id": "mockdocimg", "page_idx": 1},
    }


# Script exercises: search (image auto) -> search with rephrased query -> INVALID tool -> submit
SCRIPT = [
    {"thought": "图像题，先用图像+文本查一次", "tool": "rag_search", "args": {"query": "内镜图像 部位识别 胃"}},
    {"thought": "证据不够，换个更具体的问法", "tool": "rag_search", "args": {"query": "胃镜下胃窦黏膜表现"}},
    {"thought": "试着调一个不存在的工具", "tool": "no_such_tool", "args": {}},
    {"thought": "证据足够，提交", "tool": "submit_answer", "args": {}},
]


def main() -> int:
    rag = MockRagPipeline()
    agent = AgentBrain(
        config=SMOKE_CONFIG,
        rag=rag,
        brain=ScriptedBrain(SCRIPT),
        reward=MockGeneratorReward(),
    )
    traj = agent.solve(make_item())

    # ---- assertions ----
    assert traj["finish_reason"] == "submitted", traj["finish_reason"]
    assert traj["n_tool_calls"] == len(SCRIPT) and traj["n_rounds"] == len(SCRIPT)
    assert len(traj["final_evidence"]) == 3, len(traj["final_evidence"])  # 2 + 1 fresh passages
    assert traj["prediction"] == "B" and traj["correct"] is True and traj["u_set"] == 1.0
    assert traj["obs_candidates"], "compat field obs_candidates must be non-empty"
    assert traj["rounds"][2]["observation"]["ok"] is False  # invalid tool recovered
    # brain never saw raw first-stage hits: observation carries final passages only
    r0 = traj["rounds"][0]["observation"]
    assert r0["n_new"] == 2 and len(r0["candidates"]) == 2, r0
    # cross-call dedup: 2nd call returned 1 fresh passage (mock), nothing duplicated
    assert traj["rounds"][1]["observation"]["n_new"] == 1
    kept_ids = {(c["doc_id"], c["page_idx"]) for c in traj["final_evidence"]}
    assert len(kept_ids) == 3

    print("== smoke trajectory ==")
    print(json.dumps({k: traj[k] for k in ("qid", "finish_reason", "n_rounds", "n_tool_calls", "prediction", "correct", "u_set")}, ensure_ascii=False, indent=2))
    print("== round trace ==")
    for r in traj["rounds"]:
        obs = r["observation"]
        print(f"  r{r['index']}: {r['call']['tool']:>13} ok={obs['ok']} +{obs['n_new']}  {obs['message'][:80]}")
    print("== batch summarize ==")
    print(json.dumps(summarize([traj]), ensure_ascii=False, indent=2))
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
