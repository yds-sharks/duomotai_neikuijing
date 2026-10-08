#!/usr/bin/env python3
"""Build DPO preference pairs from Stage-2 trajectories (answer-utility ranked).

For each recorded round-state we form candidate keep-actions and score each by
ANSWER-UTILITY with the frozen generator (the moat: prefer evidence that makes
the generator answer correctly, not merely "relevant" evidence):

    u(keep) = 1[ pred(question | kept_evidence, image) == gold ]

Candidates compared for the same state:
    - teacher     : the recorded teacher keep set                (usually good)
    - keep_none   : keep = []                                     (no evidence)
    - keep_all    : keep = every real candidate                  (noisy)
    - flip        : keep the teacher-dropped, drop teacher-kept   (adversarial)

A (chosen, rejected) pair is emitted only when utilities differ by >= --margin;
chosen = highest-utility action, rejected = lowest. chosen/rejected are the full
JSON controller decisions (keep/drop/action/rewrite_query/reason), matching what
train_ctrl_dpo.py expects.

Requires the frozen generator (vLLM) at generator.base_url in the runtime config.
Run AFTER trajectory generation; typically the last stage of the pipeline.

Usage:
  python build_dpo_pairs.py --input outputs/stage2_calibration/agent_context_v11_train3200.jsonl \
    --output train/dpo_pairs.jsonl --margin 1 --max-real 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

CODE_DIR = Path(__file__).resolve().parent.parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from context_agent import SYSTEM_PROMPT_V11, USER_TEMPLATE, format_candidates  # noqa: E402
from generator_adapter import OpenAICompatibleGenerator, load_config  # noqa: E402
from rag_prompting import build_rag_prompt, parse_prediction  # noqa: E402


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if line:
                yield json.loads(line)


def action_json(keep: List[int], n: int, *, action: str, rewrite_query: str = "", reason: str = "") -> str:
    drop = [i for i in range(n) if i not in set(keep)]
    return json.dumps({
        "keep": sorted(keep), "drop": drop, "action": action,
        "rewrite_query": rewrite_query, "reason": reason,
    }, ensure_ascii=False)


def measure_correct(gen: OpenAICompatibleGenerator, question: str, options: Dict[str, Any],
                    evidence: List[Dict[str, Any]], image_path: str, gold: str) -> int:
    prompt, valid = build_rag_prompt(question, options, evidence)
    try:
        resp = gen.generate(prompt, image_path)
    except Exception:
        return 0
    pred = parse_prediction(resp, valid)
    return int(bool(pred) and pred == gold)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(CODE_DIR / "agentic_runtime_config.json"))
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--margin", type=int, default=1, help="min utility gap to emit a pair")
    ap.add_argument("--max-real", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    config = load_config(args.config)
    gen = OpenAICompatibleGenerator(config)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_pairs = 0
    n_states = 0
    with out.open("w", encoding="utf-8") as fout:
        for traj in iter_jsonl(Path(args.input)):
            if args.limit and n_states >= args.limit:
                break
            gold = str(traj.get("answer") or "")
            if not gold:
                continue
            question = str(traj.get("question") or "")
            options = traj.get("options", {})
            image_path = str(traj.get("query_image_path") or "")
            for rd in traj.get("rounds", []):
                cands = rd.get("candidates", [])
                n = len(cands)
                if n == 0:
                    continue
                n_states += 1
                n_bc = int(rd.get("n_breadcrumb", 0))
                real_idx = list(range(n_bc, n))[: args.max_real]
                teacher_keep = list(rd.get("agent", {}).get("keep", []))

                def ev(keep):
                    return [cands[i] for i in keep if 0 <= i < n]

                actions = {
                    "teacher": teacher_keep,
                    "keep_none": [],
                    "keep_all": (([0] if n_bc else []) + real_idx),
                    "flip": ([0] if n_bc else []) + [i for i in real_idx if i not in set(teacher_keep)],
                }
                util = {name: measure_correct(gen, question, options, ev(keep), image_path, gold)
                        for name, keep in actions.items()}
                best = max(util, key=lambda k: util[k])
                worst = min(util, key=lambda k: util[k])
                if util[best] - util[worst] < args.margin:
                    continue
                sample = {
                    "qid": traj.get("qid", ""),
                    "round_idx": rd.get("round_idx", 0),
                    "system": SYSTEM_PROMPT_V11,
                    "user_text": USER_TEMPLATE.format(
                        qid=traj.get("qid", ""), query_type=traj.get("query_type", ""),
                        original_query=traj.get("original_query", ""), current_query=rd.get("query", ""),
                        question=question, options=json.dumps(options, ensure_ascii=False),
                        candidate_block=format_candidates(cands),
                    ),
                    "query_image_path": image_path,
                    "evidence_image_paths": [str(c.get("image_path") or "") for c in cands],
                    "chosen": action_json(actions[best], n, action="ACCEPT" if actions[best] else "REWRITE"),
                    "rejected": action_json(actions[worst], n, action="ACCEPT" if actions[worst] else "REWRITE"),
                    "utility": {"chosen": util[best], "rejected": util[worst], "best": best, "worst": worst},
                }
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                fout.flush()
                n_pairs += 1
                if n_pairs % 25 == 0:
                    print(f"[pairs={n_pairs}] states={n_states}", flush=True)

    print(json.dumps({"pairs": n_pairs, "states": n_states, "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
