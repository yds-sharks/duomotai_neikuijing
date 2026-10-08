#!/usr/bin/env python3
"""Diagnose answer-utility reward headroom on the recorded states.

For each state we ask: does retrieved evidence actually change the frozen
generator's confidence in the correct option?  If the query image alone already
saturates P(correct)=1, then p(kept)-p(no-evidence) is ~0 everywhere and GRPO
gets no signal.  We measure, per state:

  p0     : no evidence           (baseline; query image kept)
  p_gold : only gold-hit evidence kept
  p_all  : all candidates kept
  *_niq  : same but WITHOUT the query image (evidence-only setting)

reported for both normalize=options (softmax over A/B/C/D) and normalize=vocab
(full-vocab prob of gold letter).  Headroom = p_gold - p0.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from gen_scorer import AnswerScorer  # noqa: E402
from reward_model import evidence_hit  # noqa: E402

DEFAULT_GEN = "/mnt/data_10/mwx/huggingface_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def read_states(path: str, limit: int) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traj = json.loads(line)
            if traj.get("status") not in (None, "ok"):
                continue
            for rd in traj.get("rounds", []):
                if rd.get("candidates"):
                    out.append({"traj": traj, "rd": rd})
            if limit and len(out) >= limit:
                break
    return out[:limit] if limit else out


def frac(xs, pred):
    return sum(1 for x in xs if pred(x)) / max(1, len(xs))


def summ(name, xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        print(f"  {name}: (empty)")
        return
    xs2 = sorted(xs)
    p = lambda q: xs2[min(len(xs2) - 1, int(q * len(xs2)))]
    print(f"  {name}: mean={st.mean(xs):+.3f} p10={p(.1):+.3f} p50={p(.5):+.3f} p90={p(.9):+.3f} min={min(xs):+.3f} max={max(xs):+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--gen-model", default=DEFAULT_GEN)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--max-images", type=int, default=8)
    args = ap.parse_args()

    scorer = AnswerScorer(args.gen_model, device=args.device, max_images=args.max_images)
    states = read_states(args.input, args.limit)
    print(f"[diag] states={len(states)} gen={Path(args.gen_model).name} device={args.device}", flush=True)

    rows = []
    for si, s in enumerate(states):
        traj, rd = s["traj"], s["rd"]
        options = traj.get("options", {}) or {}
        gold_letter = str(traj.get("answer", "") or "")
        if not options or gold_letter not in options:
            continue
        q = traj.get("question", "")
        qimg = traj.get("query_image_path", "")
        gold = traj.get("gold_source", {}) or {}
        gdoc, gpage = str(gold.get("doc_id") or ""), gold.get("page_idx")
        cands = rd.get("candidates", [])
        gold_ev = [{"text": c.get("text", ""), "image_path": c.get("image_path", "")}
                   for c in cands if evidence_hit([c], gdoc, gpage)]
        all_ev = [{"text": c.get("text", ""), "image_path": c.get("image_path", "")} for c in cands]

        def sc(ev, norm, uq=True):
            return scorer.answer_prob(q, options, gold_letter, qimg, ev, normalize=norm, use_query_image=uq)

        # spread of P(correct) across plausible keep-subsets == the signal GRPO sees
        subsets = {
            "none": [], "top1": all_ev[:1], "top3": all_ev[:3],
            "top5": all_ev[:5], "all": all_ev,
        }
        ps = {k: sc(v, "options") for k, v in subsets.items()}
        vals = list(ps.values())
        r = {
            "p0": ps["none"], "p_top1": ps["top1"], "p_top3": ps["top3"],
            "p_top5": ps["top5"], "p_all": ps["all"],
            "spread": max(vals) - min(vals),
            "std": st.pstdev(vals) if len(vals) > 1 else 0.0,
            "d_all": ps["all"] - ps["none"], "d_top3": ps["top3"] - ps["none"],
            "n_gold": len(gold_ev), "n_cand": len(cands),
        }
        rows.append(r)
        if si < 14:
            print(f"  s{si}: p[none/t1/t3/t5/all]="
                  f"{ps['none']:.2f}/{ps['top1']:.2f}/{ps['top3']:.2f}/{ps['top5']:.2f}/{ps['all']:.2f} "
                  f"spread={r['spread']:.2f} n_cand={r['n_cand']}", flush=True)

    print(f"\n[summary] n={len(rows)}")
    summ("p0 (no evidence)", [r["p0"] for r in rows])
    summ("spread = max-min P over subsets", [r["spread"] for r in rows])
    summ("std of P over subsets", [r["std"] for r in rows])
    summ("d_all = p_all - p0", [r["d_all"] for r in rows])
    summ("d_top3 = p_top3 - p0", [r["d_top3"] for r in rows])
    print(f"  frac(spread>0.05)={frac([r['spread'] for r in rows], lambda x: x>0.05):.2f}  "
          f"frac(spread>0.15)={frac([r['spread'] for r in rows], lambda x: x>0.15):.2f}  "
          f"frac(spread>0.30)={frac([r['spread'] for r in rows], lambda x: x>0.30):.2f}")
    print(f"  frac(p0>=0.95)={frac([r['p0'] for r in rows], lambda x: x>=0.95):.2f}  "
          f"frac(p0<=0.05)={frac([r['p0'] for r in rows], lambda x: x<=0.05):.2f}")


if __name__ == "__main__":
    main()
