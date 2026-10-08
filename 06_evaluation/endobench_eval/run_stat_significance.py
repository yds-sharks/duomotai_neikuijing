#!/usr/bin/env python3
"""P0.3: Statistical significance for 3-Round vs all baselines on full 6832 questions.

Pairs:
  1. 3-Round vs Closed-book (baseline)
  2. 3-Round vs Vanilla RAG
  3. 3-Round vs Single-round (agentic, 1 round)

Output:
  - Accuracy difference
  - Paired bootstrap 95% CI (10000 resamples)
  - Two-sided McNemar p-value
  - Four-quadrant counts
"""
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent

# ---- Load all result sets ----

def load_jsonl(path: Path) -> Dict[str, dict]:
    """Load JSONL and return dict keyed by qid."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = r.get("qid") or f"eb_{r.get('index', '')}"
            out[qid] = r
    return out


def load_3round() -> Dict[str, dict]:
    """Merge harm test (2741 correct) + rescue test (4091 wrong)."""
    out = {}
    for fname in ["rescue_harm_u50_shard0.jsonl", "rescue_harm_u50_shard1.jsonl"]:
        p = HERE / fname
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        qid = r.get("qid") or f"eb_{r.get('index', '')}"
                        out[qid] = r
    p = HERE / "rescue_full_u50.jsonl"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    qid = r.get("qid") or f"eb_{r.get('index', '')}"
                    out[qid] = r
    return out


# ---- Statistical tests ----

def mcnemar_test(a_correct: List[bool], b_correct: List[bool]) -> Tuple[float, float]:
    """Two-sided McNemar test.
    Returns (chi2, p_value).
    """
    # b = 3-Round, a = baseline
    # n01 = a wrong, b right (baseline wrong, 3round right)
    # n10 = a right, b wrong (baseline right, 3round wrong)
    n01 = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)
    n10 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    n = n01 + n10
    if n == 0:
        return 0.0, 1.0
    # Chi-squared with continuity correction
    chi2 = (abs(n01 - n10) - 1) ** 2 / n if n > 0 else 0.0
    # p-value from chi2 distribution (df=1)
    # Using simple approximation: p = 2 * (1 - Phi(sqrt(chi2)))
    # For exact, use the fact that chi2(1) = Z^2
    from math import exp, sqrt, erf
    z = sqrt(chi2)
    p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
    return chi2, p


def paired_bootstrap_ci(
    a_correct: List[bool],
    b_correct: List[bool],
    n_boot: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Paired bootstrap 95% CI for accuracy difference.
    Returns (mean_diff, ci_low, ci_high).
    """
    n = len(a_correct)
    acc_a = sum(a_correct) / n
    acc_b = sum(b_correct) / n
    obs_diff = acc_b - acc_a

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        a_acc_boot = sum(a_correct[i] for i in idx) / n
        b_acc_boot = sum(b_correct[i] for i in idx) / n
        diffs.append(b_acc_boot - a_acc_boot)

    diffs.sort()
    ci_low = diffs[int(0.025 * n_boot)]
    ci_high = diffs[int(0.975 * n_boot)]
    return obs_diff, ci_low, ci_high


def four_quadrants(a_correct: List[bool], b_correct: List[bool]) -> Dict[str, int]:
    """Four-quadrant counts.
    a = baseline, b = 3-Round
    """
    both_right = sum(1 for a, b in zip(a_correct, b_correct) if a and b)
    a_right_b_wrong = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    a_wrong_b_right = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)
    both_wrong = sum(1 for a, b in zip(a_correct, b_correct) if not a and not b)
    return {
        "both_right": both_right,
        "harmed (right→wrong)": a_right_b_wrong,
        "rescued (wrong→right)": a_wrong_b_right,
        "both_wrong": both_wrong,
    }


def main():
    print("=" * 80)
    print("P0.3: Statistical Significance Analysis (Full 6832 questions)")
    print("=" * 80)

    # Load all results
    baseline = load_jsonl(HERE / "results" / "baseline_samples.jsonl")
    vanilla = load_jsonl(HERE / "results_v2" / "vanilla_rag_samples.jsonl")
    single = load_jsonl(HERE / "results_v2" / "agentic_samples.jsonl")
    three_round = load_3round()

    print(f"\nLoaded:")
    print(f"  Baseline (closed-book): {len(baseline)}")
    print(f"  Vanilla RAG:            {len(vanilla)}")
    print(f"  Single-round agentic:   {len(single)}")
    print(f"  3-Round rescue:         {len(three_round)}")

    # Find common qids across all sets
    common = set(baseline.keys()) & set(vanilla.keys()) & set(single.keys()) & set(three_round.keys())
    print(f"  Common qids:            {len(common)}")

    if len(common) < 6000:
        print(f"WARNING: only {len(common)} common qids, expected 6832")
        # Check what's missing
        missing_from_3r = set(baseline.keys()) - set(three_round.keys())
        if missing_from_3r:
            print(f"  Missing from 3-Round: {len(missing_from_3r)} (e.g. {list(missing_from_3r)[:5]})")

    # Build paired lists
    qids = sorted(common)
    bl_correct = [baseline[q].get("correct", False) for q in qids]
    vr_correct = [vanilla[q].get("correct", False) for q in qids]
    sr_correct = [single[q].get("correct", False) for q in qids]
    tr_correct = [three_round[q].get("correct", False) for q in qids]

    pairs = [
        ("3-Round vs Closed-book", bl_correct, tr_correct),
        ("3-Round vs Vanilla RAG", vr_correct, tr_correct),
        ("3-Round vs Single-round", sr_correct, tr_correct),
    ]

    print("\n" + "=" * 80)
    print(f"{'Comparison':<28} {'Acc A':>8} {'Acc B':>8} {'Diff':>8} {'95% CI':>20} {'McNemar χ²':>12} {'p-value':>12} {'Sig':>5}")
    print("-" * 100)

    for name, a_correct, b_correct in pairs:
        n = len(a_correct)
        acc_a = sum(a_correct) / n
        acc_b = sum(b_correct) / n
        diff = acc_b - acc_a

        # McNemar
        chi2, p_val = mcnemar_test(a_correct, b_correct)

        # Bootstrap CI
        mean_diff, ci_low, ci_high = paired_bootstrap_ci(a_correct, b_correct)

        # Four quadrants
        fq = four_quadrants(a_correct, b_correct)

        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        ci_str = f"[{ci_low*100:+.2f}, {ci_high*100:+.2f}]"

        print(f"{name:<28} {acc_a*100:>7.2f}% {acc_b*100:>7.2f}% {diff*100:>+7.2f}pp {ci_str:>20} {chi2:>11.2f} {p_val:>12.2e} {sig:>5}")

        print(f"  Four quadrants: both_right={fq['both_right']}, "
              f"harmed={fq['harmed (right→wrong)']}, "
              f"rescued={fq['rescued (wrong→right)']}, "
              f"both_wrong={fq['both_wrong']}")

    # Per-category breakdown for 3-Round vs Single-round
    print("\n" + "=" * 80)
    print("Per-category: 3-Round vs Single-round")
    print("=" * 80)

    from collections import defaultdict
    cat_data = defaultdict(lambda: {"sr": [], "tr": []})
    for qid in qids:
        cat = single[qid].get("category", "unknown")
        cat_data[cat]["sr"].append(single[qid].get("correct", False))
        cat_data[cat]["tr"].append(three_round[qid].get("correct", False))

    print(f"\n{'Category':<55} {'N':>5} {'Single':>8} {'3-Round':>8} {'Diff':>8} {'McNemar p':>12}")
    print("-" * 100)
    for cat in sorted(cat_data.keys()):
        sr = cat_data[cat]["sr"]
        tr = cat_data[cat]["tr"]
        n = len(sr)
        acc_sr = sum(sr) / n
        acc_tr = sum(tr) / n
        diff = acc_tr - acc_sr
        _, p_val = mcnemar_test(sr, tr)
        print(f"{cat:<55} {n:>5} {acc_sr*100:>7.2f}% {acc_tr*100:>7.2f}% {diff*100:>+7.2f}pp {p_val:>12.2e}")

    # Per-scene breakdown
    print("\n" + "=" * 80)
    print("Per-scene: 3-Round vs Single-round")
    print("=" * 80)

    scene_data = defaultdict(lambda: {"sr": [], "tr": []})
    for qid in qids:
        scene = single[qid].get("scene", "unknown")
        scene_data[scene]["sr"].append(single[qid].get("correct", False))
        scene_data[scene]["tr"].append(three_round[qid].get("correct", False))

    print(f"\n{'Scene':<30} {'N':>5} {'Single':>8} {'3-Round':>8} {'Diff':>8} {'McNemar p':>12}")
    print("-" * 75)
    for scene in sorted(scene_data.keys()):
        sr = scene_data[scene]["sr"]
        tr = scene_data[scene]["tr"]
        n = len(sr)
        acc_sr = sum(sr) / n
        acc_tr = sum(tr) / n
        diff = acc_tr - acc_sr
        _, p_val = mcnemar_test(sr, tr)
        print(f"{scene:<30} {n:>5} {acc_sr*100:>7.2f}% {acc_tr*100:>7.2f}% {diff*100:>+7.2f}pp {p_val:>12.2e}")

    print("\n[done] Statistical significance analysis complete.")


if __name__ == "__main__":
    main()
