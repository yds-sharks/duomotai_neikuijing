#!/usr/bin/env python3
"""Compare multi-round ablation groups: mr_full vs mr_no_mplus vs mr_no_mminus vs single.

Loads agentic_samples.jsonl from each subdirectory and produces:
  1. Overall accuracy comparison
  2. Per-scene accuracy breakdown
  3. Controller behavior (REWRITE rate, accept/rewrite acc, mean keep)
  4. Multi-round stats (multi-round rate, mean rounds, mean retained, mean breadcrumb)
  5. Paired bootstrap 95% CI for full vs each ablation
  6. McNemar test for full vs each ablation

Usage:
  python compare_multiround.py --base-dir results_v2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCENE_ORDER = ["Gastroscopy", "Colonoscopy", "Capsule Endoscopy", "Surgical Endoscopy"]

GROUP_LABEL = {
    "mr_full": "M+ + M- (full)",
    "mr_no_mplus": "no M+ (only M-)",
    "mr_no_mminus": "no M- (only M+)",
    "agentic_samples": "single-round (baseline)",
}


def load_records(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def acc(rows: List[Dict[str, Any]]) -> float:
    return sum(r["correct"] for r in rows) / len(rows) if rows else 0.0


def mean_logp(rows: List[Dict[str, Any]]) -> float:
    return sum(r.get("logp_gold", 0.0) for r in rows) / len(rows) if rows else 0.0


def behavior(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    rw = [r for r in rows if r.get("action") == "REWRITE"]
    ac = [r for r in rows if r.get("action") == "ACCEPT"]
    n_multi = sum(1 for r in rows if r.get("n_rounds", 1) > 1)
    # Last-round retained/breadcrumb
    last_ret = [r.get("rounds", [{}])[-1].get("n_retained", 0) for r in rows if r.get("rounds")]
    last_bc = [r.get("rounds", [{}])[-1].get("n_breadcrumb", 0) for r in rows if r.get("rounds")]
    return {
        "n": len(rows),
        "rewrite_rate": len(rw) / len(rows),
        "accept_acc": acc(ac) if ac else 0.0,
        "rewrite_acc": acc(rw) if rw else 0.0,
        "mean_keep_accept": (sum(len(r.get("keep", [])) for r in ac) / len(ac)) if ac else 0.0,
        "mean_n_evidence": sum(r.get("n_evidence", 0) for r in rows) / len(rows),
        "parse_fail_rate": sum(1 for r in rows if not r.get("parse_ok", True)) / len(rows),
        "multi_round_rate": n_multi / len(rows),
        "mean_rounds": sum(r.get("n_rounds", 1) for r in rows) / len(rows),
        "mean_retained_last": sum(last_ret) / len(rows) if last_ret else 0.0,
        "mean_breadcrumb_last": sum(last_bc) / len(rows) if last_bc else 0.0,
    }


def paired_bootstrap_ci(
    rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]],
    n_boot: int = 10000, seed: int = 42,
) -> Dict[str, Any]:
    """Paired bootstrap 95% CI for accuracy difference (A - B)."""
    import random
    idx_map = {r["index"]: r for r in rows_a}
    pairs = []
    for rb in rows_b:
        ra = idx_map.get(rb["index"])
        if ra is not None:
            pairs.append((ra["correct"], rb["correct"]))
    if len(pairs) < 10:
        return {"n_paired": len(pairs), "ci_lo": None, "ci_hi": None, "delta": None}
    rng = random.Random(seed)
    deltas = []
    n = len(pairs)
    for _ in range(n_boot):
        sample = [pairs[rng.randint(0, n - 1)] for _ in range(n)]
        a = sum(p[0] for p in sample) / n
        b = sum(p[1] for p in sample) / n
        deltas.append(a - b)
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot)]
    actual_delta = sum(p[0] for p in pairs) / n - sum(p[1] for p in pairs) / n
    return {"n_paired": n, "delta": actual_delta, "ci_lo": lo, "ci_hi": hi}


def mcnemar_test(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """McNemar's test: b = A correct B wrong, c = A wrong B correct."""
    idx_map = {r["index"]: r for r in rows_a}
    b = c = 0
    for rb in rows_b:
        ra = idx_map.get(rb["index"])
        if ra is None:
            continue
        ca = ra["correct"]
        cb = rb["correct"]
        if ca and not cb:
            b += 1
        elif not ca and cb:
            c += 1
    # Exact binomial p-value (two-sided)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "chi2_cc": 0.0, "p_value_exact": 1.0}
    from math import comb
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    p = min(p, 1.0)
    # Chi-square approximation with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    return {"b": b, "c": c, "chi2_cc": chi2, "p_value_exact": p}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", default=str(Path(__file__).resolve().parent / "results_v2"))
    ap.add_argument("--groups", default="mr_full,mr_no_mplus,mr_no_mminus")
    ap.add_argument("--baseline", default="",
                    help="path to single-round baseline jsonl (default: base-dir/agentic_samples.jsonl)")
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    base = Path(args.base_dir)

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    data: Dict[str, List[Dict[str, Any]]] = {}
    for g in groups:
        path = base / g / "agentic_samples.jsonl"
        data[g] = load_records(path)
        print(f"[load] {g}: {len(data[g])} samples from {path}")

    # Single-round baseline
    bl_path = Path(args.baseline) if args.baseline else base / "agentic_samples.jsonl"
    bl_key = "single"
    data[bl_key] = load_records(bl_path)
    if data[bl_key]:
        print(f"[load] {bl_key}: {len(data[bl_key])} samples from {bl_path}")
        all_keys = groups + [bl_key]
    else:
        print(f"[load] {bl_key}: not found at {bl_path}")
        all_keys = groups

    if not any(data.values()):
        print("no records found")
        return

    # ---------- accuracy table ----------
    print("\n" + "=" * 80)
    print("===== Accuracy Comparison =====")
    print("=" * 80)
    scenes = SCENE_ORDER + sorted({r.get("scene", "?") for rows in data.values() for r in rows} - set(SCENE_ORDER))
    header = f"{'Group':<28}{'N':>6}{'Overall':>9}" + "".join(f"{s[:12]:>13}" for s in scenes)
    print(header)
    print("-" * len(header))
    for g in all_keys:
        rows = data.get(g, [])
        if not rows:
            continue
        overall = acc(rows)
        per = {s: acc([r for r in rows if r.get("scene") == s]) for s in scenes}
        label = GROUP_LABEL.get(g, g)
        print(f"{label:<28}{len(rows):>6}{overall:>9.4f}" + "".join(f"{per[s]:>13.4f}" for s in scenes))

    # ---------- behavior table ----------
    print("\n" + "=" * 80)
    print("===== Controller Behavior =====")
    print("=" * 80)
    print(f"{'Group':<28}{'REWRITE%':>10}{'Acc(ACC)':>10}{'Acc(RW)':>10}"
          f"{'MeanKeep':>10}{'MeanEv':>8}{'ParseFail%':>12}")
    print("-" * 88)
    for g in all_keys:
        rows = data.get(g, [])
        if not rows:
            continue
        b = behavior(rows)
        label = GROUP_LABEL.get(g, g)
        print(f"{label:<28}{b['rewrite_rate']*100:>9.1f}%{b['accept_acc']:>10.4f}{b['rewrite_acc']:>10.4f}"
              f"{b['mean_keep_accept']:>10.2f}{b['mean_n_evidence']:>8.2f}{b['parse_fail_rate']*100:>11.1f}%")

    # ---------- multi-round stats ----------
    print("\n" + "=" * 80)
    print("===== Multi-round Stats =====")
    print("=" * 80)
    print(f"{'Group':<28}{'MultiRd%':>10}{'MeanRounds':>11}{'MeanRet':>9}{'MeanBC':>9}")
    print("-" * 67)
    for g in all_keys:
        rows = data.get(g, [])
        if not rows:
            continue
        b = behavior(rows)
        label = GROUP_LABEL.get(g, g)
        print(f"{label:<28}{b['multi_round_rate']*100:>9.1f}%{b['mean_rounds']:>11.3f}"
              f"{b['mean_retained_last']:>9.3f}{b['mean_breadcrumb_last']:>9.3f}")

    # ---------- mean logP ----------
    print("\n" + "=" * 80)
    print("===== Mean logP(gold) =====")
    print("=" * 80)
    for g in all_keys:
        rows = data.get(g, [])
        if not rows:
            continue
        label = GROUP_LABEL.get(g, g)
        print(f"{label:<28}{mean_logp(rows):>10.3f}")

    # ---------- paired bootstrap + McNemar (full vs each ablation) ----------
    if "mr_full" in data and data["mr_full"]:
        full_rows = data["mr_full"]
        others = [g for g in all_keys if g != "mr_full"]
        print("\n" + "=" * 80)
        print("===== Significance Tests (full vs others) =====")
        print("=" * 80)
        for g in others:
            rows = data.get(g, [])
            if not rows:
                continue
            bs = paired_bootstrap_ci(full_rows, rows)
            mc = mcnemar_test(full_rows, rows)
            label = GROUP_LABEL.get(g, g)
            print(f"\n  full vs {label}:")
            print(f"    paired bootstrap: n={bs['n_paired']}, delta={bs['delta']:.4f}"
                  f"  CI95=[{bs['ci_lo']:.4f}, {bs['ci_hi']:.4f}]"
                  f"  {'*' if bs['ci_lo'] > 0 or bs['ci_hi'] < 0 else 'ns'}")
            print(f"    McNemar: b={mc['b']} (full correct, other wrong)"
                  f"  c={mc['c']} (full wrong, other correct)"
                  f"  chi2_cc={mc.get('chi2_cc', 0.0):.2f}  p={mc.get('p_value_exact', 1.0):.4f}")

    # ---------- save JSON ----------
    out: Dict[str, Any] = {}
    for g in all_keys:
        rows = data.get(g, [])
        if rows:
            out[g] = {"overall": acc(rows), "n": len(rows), "behavior": behavior(rows)}
    if "mr_full" in data and data["mr_full"]:
        out["significance"] = {}
        for g in all_keys:
            if g == "mr_full":
                continue
            rows = data.get(g, [])
            if not rows:
                continue
            out["significance"][g] = {
                "bootstrap": paired_bootstrap_ci(data["mr_full"], rows),
                "mcnemar": mcnemar_test(data["mr_full"], rows),
            }
    if args.save:
        Path(args.save).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[save] {args.save}")


if __name__ == "__main__":
    main()
