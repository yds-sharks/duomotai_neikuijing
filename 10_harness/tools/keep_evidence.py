#!/usr/bin/env python3
"""keep_evidence: the brain's keep/drop decision over this round's numbered candidates.

Selecting nothing (empty keep list) is a legal "drop all" signal: the observation
tells the brain the round contributed no evidence, nudging it to re-retrieve or submit.
"""

from __future__ import annotations

from typing import Any, Dict, List

from tools.base import Tool, ToolResult


class KeepEvidenceTool(Tool):
    name = "keep_evidence"
    description = (
        "Select evidence from THIS round's numbered candidate list into the persistent "
        "evidence set. Unselected candidates are dropped. Numbers must refer to the "
        "current candidate list shown in the last observation."
    )
    args_schema = '{"keep": [int, ...]}  # 1-based candidate numbers, max per round from budget'

    def run(self, session, args: Dict[str, Any]) -> ToolResult:
        raw = args.get("keep", [])
        if not isinstance(raw, list):
            return ToolResult(ok=False, message="keep_evidence requires args.keep as a list of ints")
        max_keep = int(session.config["budget"].get("max_keep_per_round", 5))
        keep: List[int] = []
        for v in raw[:max_keep]:
            try:
                keep.append(int(v))
            except (TypeError, ValueError):
                return ToolResult(ok=False, message=f"invalid keep index: {v!r}")
        added, unknown = session.keep_from_candidates(keep)
        msg = f"keep_evidence ok: +{added} new evidence"
        if unknown:
            msg += f"; ignored out-of-range numbers {unknown}"
        if added == 0:
            msg += "; nothing new kept — consider a different query or submit_answer"
        return ToolResult(ok=True, candidates=session.last_candidates, n_new=added, message=msg)
