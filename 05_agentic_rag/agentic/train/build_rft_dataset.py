#!/usr/bin/env python3
"""Build the RFT (rejection-sampling fine-tune) dataset from Stage-D rewards (v2.1).

Three sample sources (one target per source, per question):
  1. rewrite-win  : rewrite tier where the best REWRITE member's u_set beats the
                    anchor_original u_set -> teach REWRITE with that member's query.
  2. selection    : selection tier, best non-empty select member with u_set > 0
                    -> teach ACCEPT with that keep-set.
  3. trivial      : trivial tier, best ACCEPT member (keep-1 anchor or self select,
                    highest u_set, keep non-empty) -> teach "good enough -> ACCEPT".

Target JSON prefers the member's own raw_text (natural phrasing from sampling);
falls back to a constructed JSON with a generic reason when raw_text is missing
or unparseable (constructed anchors).

Output rows mirror the SFT schema (train_ctrl_sft_full.py / ctrl_data_common):
  {qid, round_idx, query_type, system, user_text, query_image_path,
   evidence_image_paths, target, has_search_history, n_candidates}

Usage:
  python train/build_rft_dataset.py \
    --rewards "train/rewards_sample_v1_sh*.jsonl" \
    --train-out train/rft_ctrl_train.jsonl --val-out train/rft_ctrl_val.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

from context_agent import SYSTEM_PROMPT_V11, USER_TEMPLATE, format_candidates  # noqa: E402
from gen_policy_actions import _first_json  # noqa: E402
from grpo_reward import parse_action  # noqa: E402


def iter_rewards(pattern: str):
    for fp in sorted(glob.glob(pattern)):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def member_target(m: Dict[str, Any], n_cand: int) -> Optional[str]:
    """Prefer the member's own sampled JSON; else construct a clean one.

    The target action/keep must stay consistent with the member's recorded
    decision (the unit Stage D scored); raw_text that drifted (e.g. a select
    rollout whose JSON says REWRITE) falls back to the constructed form.
    """
    want_action = m.get("action", "ACCEPT")
    want_keep = sorted({int(x) for x in (m.get("keep") or []) if 0 <= int(x) < n_cand})
    raw = str(m.get("raw_text") or "")
    if raw:
        a = parse_action(_first_json(raw), n_cand)
        if a["parsed"]:
            keep = sorted({int(x) for x in a["keep"] if 0 <= int(x) < n_cand})
            if a["action"] == want_action and keep == want_keep:
                drop = [i for i in range(n_cand) if i not in keep]
                tgt = {
                    "keep": keep,
                    "drop": drop,
                    "action": a["action"],
                    "rewrite_query": a["rewrite_query"] if a["action"] == "REWRITE" else "",
                    "reason": (a.get("reason") or "").strip()[:120] or "基于候选证据价值判断",
                }
                return json.dumps(tgt, ensure_ascii=False)
    keep = sorted({int(x) for x in (m.get("keep") or []) if 0 <= int(x) < n_cand})
    drop = [i for i in range(n_cand) if i not in keep]
    action = m.get("action", "ACCEPT")
    tgt = {
        "keep": keep if action == "ACCEPT" else [],
        "drop": drop if action == "ACCEPT" else list(range(n_cand)),
        "action": action,
        "rewrite_query": (m.get("rewrite_query") or "") if action == "REWRITE" else "",
        "reason": "现有证据足以作答" if action == "ACCEPT" else "候选证据不足或偏题，需改写检索",
    }
    if action == "REWRITE" and not tgt["rewrite_query"]:
        return None
    return json.dumps(tgt, ensure_ascii=False)


def build_user_text(d: Dict[str, Any]) -> str:
    obs = d.get("obs_candidates", [])
    return USER_TEMPLATE.format(
        qid=d.get("qid", ""),
        query_type=d.get("query_type", ""),
        original_query=d.get("original_query", ""),
        current_query=d.get("original_query", ""),
        question=d.get("question", ""),
        options=json.dumps(d.get("options", {}), ensure_ascii=False),
        candidate_block=format_candidates(obs),
    )


def to_row(d: Dict[str, Any], m: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    obs = d.get("obs_candidates", [])
    n_cand = len(obs)
    if not n_cand:
        return None
    target = member_target(m, n_cand)
    if target is None:
        return None
    return {
        "qid": d.get("qid", ""),
        "round_idx": 0,
        "query_type": d.get("query_type", ""),
        "system": SYSTEM_PROMPT_V11,
        "user_text": build_user_text(d),
        "query_image_path": d.get("query_image_path", ""),
        "evidence_image_paths": [str(c.get("image_path") or "") for c in obs],
        "target": target,
        "has_search_history": 0,
        "n_candidates": n_cand,
        "rft_source": source,
    }


def pick_rows(d: Dict[str, Any], rewrite_topk: int = 2) -> List[Dict[str, Any]]:
    tier = d.get("tier")
    g = d.get("group", [])
    rows: List[Dict[str, Any]] = []
    if tier == "rewrite":
        anchor = [m for m in g if m.get("kind") == "anchor_original"]
        if not anchor:
            return rows
        u_anchor = float(anchor[0].get("u_set", 0.0))
        wins = [m for m in g if m.get("kind") == "rewrite" and float(m.get("u_set", 0.0)) > u_anchor]
        if not wins:
            return rows
        # top-k distinct winning rewrites lift the REWRITE share into the 15-25% band
        wins.sort(key=lambda m: float(m.get("u_set", 0.0)), reverse=True)
        seen_q = set()
        for m in wins:
            if len(rows) >= rewrite_topk:
                break
            rq = (m.get("rewrite_query") or "").strip()
            if not rq or rq in seen_q:
                continue
            seen_q.add(rq)
            r = to_row(d, m, "rewrite_win")
            if r:
                rows.append(r)
    elif tier == "selection":
        sels = [m for m in g if m.get("kind") == "select" and m.get("keep")
                and float(m.get("u_set", 0.0)) > 0.0]
        if not sels:
            return rows
        best = max(sels, key=lambda m: float(m.get("u_set", 0.0)))
        r = to_row(d, best, "selection_best")
        if r:
            rows.append(r)
    elif tier == "trivial":
        cands = [m for m in g if m.get("action") == "ACCEPT" and m.get("keep")]
        if not cands:
            return rows
        best = max(cands, key=lambda m: float(m.get("u_set", 0.0)))
        r = to_row(d, best, "trivial_accept")
        if r:
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rewards", default="train/rewards_sample_v1_sh*.jsonl")
    ap.add_argument("--train-out", default="train/rft_ctrl_train.jsonl")
    ap.add_argument("--val-out", default="train/rft_ctrl_val.jsonl")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--rewrite-topk", type=int, default=2,
                    help="max distinct winning rewrites per rewrite-tier question")
    ap.add_argument("--rewrite-repeat", type=int, default=1,
                    help="repeat rewrite_win rows N times (oversample to lift REWRITE share)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    stats = {"rewrite_win": 0, "selection_best": 0, "trivial_accept": 0}
    all_rows: List[Dict[str, Any]] = []
    n_q = 0
    for d in iter_rewards(args.rewards):
        n_q += 1
        for r in pick_rows(d, rewrite_topk=args.rewrite_topk):
            stats[r["rft_source"]] += 1
            all_rows.append(r)

    qids = sorted({r["qid"] for r in all_rows})
    rng = random.Random(args.seed)
    rng.shuffle(qids)
    n_val = max(int(len(qids) * args.val_ratio), 1) if qids else 0
    val_qids = set(qids[:n_val])
    train = [r for r in all_rows if r["qid"] not in val_qids]
    val = [r for r in all_rows if r["qid"] in val_qids]
    # oversample rewrite rows in TRAIN only (val keeps the natural distribution)
    if args.rewrite_repeat > 1:
        train = train + [r for r in train if r["rft_source"] == "rewrite_win"] * (args.rewrite_repeat - 1)

    for path_str, rows in ((args.train_out, train), (args.val_out, val)):
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_rw = sum(1 for r in all_rows if '"REWRITE"' in r["target"])
    print(json.dumps({
        "questions": n_q,
        "rows": len(all_rows),
        **stats,
        "rewrite_rows": n_rw,
        "rewrite_frac": round(n_rw / max(len(all_rows), 1), 4),
        "train": len(train),
        "val": len(val),
        "train_out": args.train_out,
        "val_out": args.val_out,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
