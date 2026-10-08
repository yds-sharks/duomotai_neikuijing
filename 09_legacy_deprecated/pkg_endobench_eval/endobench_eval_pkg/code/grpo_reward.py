#!/usr/bin/env python3
"""GRPO reward wiring for the evidence controller (reuses reward_model v0.2).

A rollout = one sampled controller decision (keep/drop + ACCEPT/REWRITE) over a
recorded round's candidate set. Reward is the *current* evidence-utility reward:
keeping the gold-bearing evidence is rewarded, dropping present gold is penal(the
`reward_model.score_candidate` scheme), plus leakage / length penalties. This is
computed purely from the recorded candidates' doc_id/page_idx vs gold_source, so
NO generator needs to be online during rollout.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

CODE_DIR = Path(__file__).resolve().parent.parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from reward_model import attach_group_advantages, evidence_hit, score_candidate  # noqa: E402

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(text: str, n: int) -> Dict[str, Any]:
    """Robustly parse a controller JSON decision; sanitize to 0-based indices."""
    keep: List[int] = []
    action = "ACCEPT"
    rewrite_query = ""
    reason = ""
    obj: Dict[str, Any] = {}
    if text:
        m = _JSON_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = {}
    if isinstance(obj, dict):
        raw_keep = obj.get("keep", [])
        if isinstance(raw_keep, list):
            for v in raw_keep:
                try:
                    iv = int(v)
                except Exception:
                    continue
                if 0 <= iv < n and iv not in keep:
                    keep.append(iv)
        action = str(obj.get("action", "ACCEPT")).upper()
        if action not in ("ACCEPT", "REWRITE"):
            action = "ACCEPT"
        rewrite_query = str(obj.get("rewrite_query", "") or "")
        reason = str(obj.get("reason", "") or "")
    drop = [i for i in range(n) if i not in set(keep)]
    return {"keep": sorted(keep), "drop": drop, "action": action,
            "rewrite_query": rewrite_query, "reason": reason, "parsed": bool(obj)}


def action_reward(rd: Dict[str, Any], action: Dict[str, Any], *,
                  gold_doc: str, gold_page: Any, answer: str, answer_text: str) -> Tuple[float, Dict[str, Any]]:
    cands = rd.get("candidates", [])
    n = len(cands)
    n_bc = int(rd.get("n_breadcrumb", 0))
    kept = [cands[i] for i in action["keep"] if n_bc <= i < n]
    pool = cands[n_bc:]
    cand_hit = evidence_hit(kept, gold_doc, gold_page)
    orig_hit = evidence_hit(pool, gold_doc, gold_page)
    query = action["rewrite_query"] if (action["action"] == "REWRITE" and action["rewrite_query"]) else rd.get("query", "")
    out = score_candidate(
        original_hit=orig_hit, candidate_hit=cand_hit, query=query,
        answer=answer, answer_text=answer_text,
        is_original=(int(rd.get("round_idx", 0)) == 0), agent_action=action["action"],
    )
    comp = out["components"]
    comp["parsed"] = action.get("parsed", True)
    if not action.get("parsed", True):
        # unparseable output: floor the reward so malformed JSON is discouraged
        return float(out["reward"]) - 0.5, comp
    return float(out["reward"]), comp


def group_advantages(rewards: List[float], std_floor: float = 0.0) -> List[float]:
    """Group z-norm with an optional std floor.

    Plain (r-mean)/std inflates tiny within-group differences (pure scorer noise)
    to full +-1 advantages, giving noise-dominated groups the same gradient
    magnitude as signal-rich ones (the Dr.GRPO critique). With std_floor>0,
    small-spread groups get proportionally small advantages instead.
    """
    if not rewards:
        return []
    if std_floor > 0.0:
        mean = sum(rewards) / len(rewards)
        var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        denom = max(math.sqrt(max(var, 0.0)), std_floor)
        return [(r - mean) / denom for r in rewards]
    cands = [{"reward": r} for r in rewards]
    attach_group_advantages(cands)
    return [float(c["advantage"]) for c in cands]
