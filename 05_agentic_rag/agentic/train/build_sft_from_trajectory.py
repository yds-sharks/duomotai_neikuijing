#!/usr/bin/env python3
"""Build per-round SFT samples from Stage-2 trajectories.

Each retrieval round of each trajectory becomes ONE behaviour-cloning sample:
the model observes (system v11 prompt, user_text with the numbered candidate
block, query image, up to N evidence images) and must reproduce the teacher's
JSON decision (keep/drop + ACCEPT/REWRITE + rewrite_query + reason).

Because the recorded round already contains the exact candidate list shown to
the teacher (including any breadcrumb / search_history item at index 0), the
keep/drop indices align 1:1 with the sample's candidate order -- no remapping
is needed. ``has_search_history`` flags rounds that carried a breadcrumb.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import sys

CODE_DIR = Path(__file__).resolve().parent.parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from context_agent import SYSTEM_PROMPT_V11, USER_TEMPLATE, format_candidates  # noqa: E402

DEFAULT_INPUT = "/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/outputs/stage2_calibration/agent_context_v11_train3200.jsonl"
DEFAULT_TRAIN = "/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/sft_ctrl_train.jsonl"
DEFAULT_VAL = "/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/sft_ctrl_val.jsonl"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_user_text(traj: Dict[str, Any], rd: Dict[str, Any]) -> str:
    return USER_TEMPLATE.format(
        qid=traj.get("qid", ""),
        query_type=traj.get("query_type", ""),
        original_query=traj.get("original_query", ""),
        current_query=rd.get("query", ""),
        question=traj.get("question", ""),
        options=json.dumps(traj.get("options", {}), ensure_ascii=False),
        candidate_block=format_candidates(rd.get("candidates", [])),
    )


def round_to_sample(traj: Dict[str, Any], rd: Dict[str, Any]) -> Dict[str, Any]:
    agent = rd.get("agent", {})
    candidates = rd.get("candidates", [])
    target = {
        "keep": agent.get("keep", []),
        "drop": agent.get("drop", []),
        "action": agent.get("action", "ACCEPT"),
        "rewrite_query": agent.get("rewrite_query", ""),
        "reason": agent.get("reason", ""),
    }
    return {
        "qid": traj.get("qid", ""),
        "round_idx": rd.get("round_idx", 0),
        "query_type": traj.get("query_type", ""),
        "system": SYSTEM_PROMPT_V11,
        "user_text": build_user_text(traj, rd),
        "query_image_path": traj.get("query_image_path", ""),
        "evidence_image_paths": [str(c.get("image_path") or "") for c in candidates],
        "target": json.dumps(target, ensure_ascii=False),
        "has_search_history": int(rd.get("n_breadcrumb", 0)) > 0,
        "n_candidates": len(candidates),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-jsonl", default=DEFAULT_INPUT)
    p.add_argument("--train-out", default=DEFAULT_TRAIN)
    p.add_argument("--val-out", default=DEFAULT_VAL)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-error", action="store_true", help="also emit rounds from non-ok trajectories")
    args = p.parse_args()

    trajs = list(iter_jsonl(Path(args.input_jsonl)))
    qids = sorted({str(t.get("qid")) for t in trajs})
    rng = random.Random(args.seed)
    rng.shuffle(qids)
    n_val = max(int(len(qids) * args.val_ratio), 1) if qids else 0
    val_qids = set(qids[:n_val])

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    stats = {"traj": 0, "traj_skipped": 0, "rounds": 0, "accept": 0, "rewrite": 0, "breadcrumb": 0, "two_round": 0}

    for t in trajs:
        if str(t.get("status")) != "ok" and not args.keep_error:
            stats["traj_skipped"] += 1
            continue
        stats["traj"] += 1
        rounds = t.get("rounds", [])
        if len(rounds) >= 2:
            stats["two_round"] += 1
        bucket = val if str(t.get("qid")) in val_qids else train
        for rd in rounds:
            sample = round_to_sample(t, rd)
            bucket.append(sample)
            stats["rounds"] += 1
            if sample["target"] and '"REWRITE"' in sample["target"]:
                stats["rewrite"] += 1
            else:
                stats["accept"] += 1
            if sample["has_search_history"]:
                stats["breadcrumb"] += 1

    for path_str, rows in ((args.train_out, train), (args.val_out, val)):
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "input_trajectories": len(trajs),
        "train_samples": len(train),
        "val_samples": len(val),
        "val_qids": len(val_qids),
        **stats,
        "train_out": args.train_out,
        "val_out": args.val_out,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
