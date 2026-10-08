#!/usr/bin/env python3
"""Stage 2: unified judge scoring + difficulty tiering (offline records).

Per question:
  - native baseline: judge(no evidence) -> cor_base, logp_gold_base (query image kept, no passages)
  - per initial candidate single-passage augmentation: judge([cand_i]) -> cor_i, u_i = logp_gold_i - logp_gold_base
    (image candidate = image+caption pair; text candidate = pure text. Both are single-passage augmentation.)

Difficulty tier (hard argmax==gold indicator):
  - trivial   : cor_base == True                     -> drop (evidence doesn't matter, no gradient)
  - selection : cor_base == False and any cor_i True  -> train selection, K_rewrite ~= 1
  - rewrite   : cor_base == False and all cor_i False -> train rewrite,  K_rewrite = 8

Also records flip info (cor_base False -> some evidence cor True) to calibrate the bonus.
Reward convention: u_i = logp_gold(evidence) - logp_gold(none); retrieval-level R = 0.3*mean(u) + 0.7*mean(top3 u).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gen_scorer import AnswerScorer  # noqa: E402

DEFAULT_GEN = ("/mnt/data_10/mwx/huggingface_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
               "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")


def load_done_qids(out_path: str) -> set:
    done: set = set()
    p = Path(out_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line).get("qid"))
                except Exception:
                    pass
    return done


def _round0_candidates(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    rounds = row.get("rounds") or []
    if not rounds:
        return []
    return rounds[0].get("candidates") or []


def _top3_mean(vals: List[float]) -> float:
    if not vals:
        return 0.0
    top = sorted(vals, reverse=True)[:3]
    return sum(top) / len(top)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="outputs/stage2_calibration/agent_context_v11_train3200_sep.jsonl")
    ap.add_argument("--out", default="outputs/stage2_calibration/judge_tier_sep.jsonl")
    ap.add_argument("--gen-model", default=DEFAULT_GEN)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--query-edge", type=int, default=768)
    ap.add_argument("--ev-edge", type=int, default=384)
    ap.add_argument("--max-images", type=int, default=8)
    ap.add_argument("--text-preview", type=int, default=160, help="candidate text preview truncation in records")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.start:
        rows = rows[args.start:]
    if args.limit:
        rows = rows[:args.limit]

    done = load_done_qids(args.out)
    print(f"[data] pending {len(rows)} (start={args.start} limit={args.limit}) done {len(done)}", flush=True)

    scorer = AnswerScorer(args.gen_model, device=args.device,
                          query_edge=args.query_edge, ev_edge=args.ev_edge,
                          max_images=args.max_images)
    print(f"[generator] ready {args.device}", flush=True)

    out_f = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    n = 0
    tier_count = {"trivial": 0, "selection": 0, "rewrite": 0, "bad": 0}
    for row in rows:
        qid = row.get("qid")
        if qid in done:
            continue
        question = row.get("question", "")
        options = row.get("options") or {}
        gold = row.get("answer", "")
        qimg = row.get("query_image_path", "")
        cands = _round0_candidates(row)

        if not options or gold not in options:
            tier_count["bad"] += 1
            out_f.write(json.dumps({"qid": qid, "tier": "bad", "reason": "no_options_or_gold"},
                                    ensure_ascii=False) + "\n")
            out_f.flush()
            n += 1
            continue

        base = scorer.judge(question, options, gold, qimg, evidence=[])
        logp_base = base["logp_gold"]
        cor_base = base["correct"]

        cand_recs: List[Dict[str, Any]] = []
        any_flip = False
        us: List[float] = []
        for i, c in enumerate(cands):
            ev = [{"text": c.get("text", "") or "", "image_path": c.get("image_path", "") or ""}]
            j = scorer.judge(question, options, gold, qimg, evidence=ev)
            u = j["logp_gold"] - logp_base
            flip = (not cor_base) and j["correct"]
            any_flip = any_flip or flip
            us.append(u)
            cand_recs.append({
                "idx": i,
                "origin": c.get("origin", ""),
                "sample_id": c.get("sample_id", ""),
                "score": c.get("score"),
                "text_preview": (c.get("text", "") or "")[:args.text_preview],
                "u": u,
                "logp_gold": j["logp_gold"],
                "p_gold": j["p_gold"],
                "pred": j["pred"],
                "cor": j["correct"],
                "flip": flip,
            })

        if cor_base:
            tier = "trivial"
        elif any_flip:
            tier = "selection"
        else:
            tier = "rewrite"
        tier_count[tier] += 1

        rec = {
            "qid": qid,
            "tier": tier,
            "gold": gold,
            "cor_base": cor_base,
            "p_base": base["p_gold"],
            "logp_gold_base": logp_base,
            "pred_base": base["pred"],
            "n_cand": len(cands),
            "n_flip": sum(1 for c in cand_recs if c["flip"]),
            "n_img": sum(1 for c in cand_recs if c["origin"] == "image"),
            "n_txt": sum(1 for c in cand_recs if c["origin"] == "text"),
            "max_u": max(us) if us else 0.0,
            "mean_u": (sum(us) / len(us)) if us else 0.0,
            "top3_mean_u": _top3_mean(us),
            "candidates": cand_recs,
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        n += 1
        if n % args.log_every == 0:
            dt = time.time() - t0
            tot = max(sum(tier_count.values()), 1)
            print(f"[{n}/{len(rows)}] {dt / n:.2f}s/q "
                  f"trivial{tier_count['trivial']} selection{tier_count['selection']} "
                  f"rewrite{tier_count['rewrite']} bad{tier_count['bad']} "
                  f"| rewrite_frac{tier_count['rewrite'] / tot:.2f}", flush=True)

    out_f.close()
    tot = max(sum(tier_count.values()), 1)
    print(f"[done] {n} -> {args.out}", flush=True)
    print(f"[tier] trivial{tier_count['trivial']}({tier_count['trivial']/tot:.1%}) "
          f"selection{tier_count['selection']}({tier_count['selection']/tot:.1%}) "
          f"rewrite{tier_count['rewrite']}({tier_count['rewrite']/tot:.1%}) "
          f"bad{tier_count['bad']}", flush=True)


if __name__ == "__main__":
    main()
