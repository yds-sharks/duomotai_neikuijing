#!/usr/bin/env python3
"""P0-5 matched 消融配对统计 + P1 检索成本/稳定性.

输出:
  p0_5_paired_stats.json / .csv
  p1_cost_stability.json / .csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
PKG = Path("/mnt/data_1/yds/多模态/_pkg/agentic/endobench_eval_pkg/results")
B = 10000
SEED = 20260729


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def paired_boot(pairs, rng):
    arr = np.array(pairs, dtype=float)
    n = len(arr)
    out = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        s = arr[idx]
        out[i] = (s[:, 1].mean() - s[:, 0].mean()) * 100
    return out


# ---------- load 1002-question configurations ----------
def load_1002():
    qids = [r["qid"] for r in load_jsonl(HERE / "stratified_1000.jsonl")]
    qset = set(qids)

    single = {r["qid"]: bool(r["correct"])
              for r in load_jsonl(HERE / "results_v2" / "agentic_samples.jsonl")
              if r["qid"] in qset}

    c4mem = {}
    for name in ["rescue_full_u50_shard0", "rescue_full_u50_shard1",
                 "rescue_harm_u50_shard0", "rescue_harm_u50_shard1"]:
        for r in load_jsonl(HERE / f"{name}.jsonl"):
            if r["qid"] in qset:
                c4mem[r["qid"]] = bool(r.get("correct", False))

    c4nm = {}
    for name in ["c4nm_shard0", "c4nm_shard1a", "c4nm_shard1b"]:
        for r in load_jsonl(HERE / f"{name}.jsonl"):
            c4nm[r["qid"]] = bool(r.get("correct", False))

    arms = {}
    for arm in ["B", "E", "F", "A"]:
        arms[arm] = {r["qid"]: bool(r["correct"])
                     for r in load_jsonl(PKG / arm / "agentic_samples.jsonl")}

    return qids, {
        "Single": single,
        "Config4-NoMemory": c4nm,
        "Config4-Memory": c4mem,
        "Standard No Memory": arms["B"],
        "M+ Only": arms["E"],
        "M- Only": arms["F"],
        "Full Memory": arms["A"],
    }


def p0_5():
    rng = np.random.default_rng(SEED)
    qids, cfg = load_1002()

    print("=" * 62)
    print("P0-5 counts verification")
    print("=" * 62)
    expect = {"Single": 403, "Config4-NoMemory": 414, "Config4-Memory": 427,
              "Standard No Memory": 419, "M+ Only": 415, "M- Only": 418,
              "Full Memory": 407}
    for k, v in expect.items():
        got = sum(cfg[k].get(q, False) for q in qids)
        n = sum(1 for q in qids if q in cfg[k])
        flag = "OK " if got == v else "FAIL"
        print(f"  [{flag}] {k:<20} expected={v:<5} got={got}  (n={n})")

    comparisons = [
        ("fixed-budget+conservative-fallback", "Single", "Config4-NoMemory"),
        ("fixed-budget+conservative-fallback", "Config4-NoMemory", "Config4-Memory"),
        ("ACCEPT-stops", "Standard No Memory", "M+ Only"),
        ("ACCEPT-stops", "Standard No Memory", "M- Only"),
        ("ACCEPT-stops", "Standard No Memory", "Full Memory"),
    ]

    rows = []
    print("\n" + "=" * 62)
    print("P0-5 paired statistics")
    print("=" * 62)
    for protocol, left, right in comparisons:
        L, R = cfg[left], cfg[right]
        common = [q for q in qids if q in L and q in R]
        pairs = [(L[q], R[q]) for q in common]
        both = sum(1 for a, b in pairs if a and b)
        lonly = sum(1 for a, b in pairs if a and not b)
        ronly = sum(1 for a, b in pairs if b and not a)
        neither = sum(1 for a, b in pairs if not a and not b)
        lc = both + lonly
        rc = both + ronly
        n = len(pairs)
        d = paired_boot(pairs, rng)
        lo, hi = np.percentile(d, [2.5, 97.5])
        rec = {
            "protocol": protocol, "left": left, "right": right, "N": n,
            "left_correct": lc, "right_correct": rc, "both_correct": both,
            "left_only_correct": lonly, "right_only_correct": ronly,
            "both_wrong": neither,
            "delta_pp": (rc - lc) / n * 100,
            "exact_mcnemar_p": exact_mcnemar(lonly, ronly),
            "paired_bootstrap_ci_low": float(lo),
            "paired_bootstrap_ci_high": float(hi),
        }
        rows.append(rec)
        print(f"\n  [{protocol}] {left}  vs  {right}")
        print(f"    N={n}  left={lc}  right={rc}  both={both}  "
              f"L_only={lonly}  R_only={ronly}  neither={neither}")
        print(f"    delta={rec['delta_pp']:+.2f}pp  "
              f"CI=[{lo:+.2f}, {hi:+.2f}]  exact_p={rec['exact_mcnemar_p']:.4f}")

    (HERE / "p0_5_paired_stats.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(HERE / "p0_5_paired_stats.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n  -> p0_5_paired_stats.json / .csv")


# ---------------- P1: cost & stability ----------------
def p1():
    rows = []

    def add(name, recs, *, mode):
        n = len(recs)
        cor = sum(1 for r in recs if r.get("correct"))
        if mode == "closed_book":
            rec = dict(configuration=name, accuracy=cor / n, correct=cor, n=n,
                       mean_rounds=0.0, mean_retrieval_calls=0.0,
                       multi_round_rate=0.0, parse_fail_count=0, parse_fail_rate=0.0,
                       fallback_count=0, fallback_rate=0.0,
                       mean_final_evidence=0.0, ACCEPT_rate=None, REWRITE_rate=None)
        elif mode == "oneshot":
            rec = dict(configuration=name, accuracy=cor / n, correct=cor, n=n,
                       mean_rounds=1.0, mean_retrieval_calls=1.0,
                       multi_round_rate=0.0, parse_fail_count=0, parse_fail_rate=0.0,
                       fallback_count=0, fallback_rate=0.0,
                       mean_final_evidence=sum(r.get("n_evidence", 0) for r in recs) / n,
                       ACCEPT_rate=None, REWRITE_rate=None)
        else:
            rounds = [r.get("n_rounds", 1) or 1 for r in recs]
            # retrieval calls: one per round (round-0 initial + one per rewrite round)
            calls = rounds
            pf = sum(1 for r in recs
                     if any(not rd.get("parse_ok", True) for rd in r.get("rounds", []))
                     or not r.get("parse_ok", True))
            fb = sum(1 for r in recs if r.get("fallback_to_orig"))
            acc_n = sum(1 for r in recs if r.get("action") == "ACCEPT"
                        or r.get("orig_action") == "ACCEPT")
            rw_n = sum(1 for r in recs if r.get("action") == "REWRITE"
                       or r.get("orig_action") == "REWRITE")
            rec = dict(configuration=name, accuracy=cor / n, correct=cor, n=n,
                       mean_rounds=sum(rounds) / n,
                       mean_retrieval_calls=sum(calls) / n,
                       multi_round_rate=sum(1 for x in rounds if x > 1) / n,
                       parse_fail_count=pf, parse_fail_rate=pf / n,
                       fallback_count=fb, fallback_rate=fb / n,
                       mean_final_evidence=sum(r.get("n_evidence", 0) for r in recs) / n,
                       ACCEPT_rate=acc_n / n if (acc_n + rw_n) else None,
                       REWRITE_rate=rw_n / n if (acc_n + rw_n) else None)
        rows.append(rec)

    add("Closed-book", list(load_jsonl(HERE / "results" / "baseline_samples.jsonl")),
        mode="closed_book")
    add("One-shot RAG", list(load_jsonl(HERE / "results_v2" / "vanilla_rag_samples.jsonl")),
        mode="oneshot")
    add("Single-round Agent",
        list(load_jsonl(HERE / "results_v2" / "agentic_samples.jsonl")), mode="agent")

    r3 = {}
    for sh in ["R3_NM_shard0", "R3_NM_shard1", "R3_NM_shard1b"]:
        p = HERE / "results_full" / sh / "agentic_samples.jsonl"
        if p.exists():
            for r in load_jsonl(p):
                q = r["qid"]
                if q not in r3 or (r3[q].get("reused_from") and not r.get("reused_from")):
                    r3[q] = r
    add("R3-NM", list(r3.values()), mode="agent")

    full = {}
    for name in ["rescue_full_u50_shard0", "rescue_full_u50_shard1",
                 "rescue_harm_u50_shard0", "rescue_harm_u50_shard1"]:
        for r in load_jsonl(HERE / f"{name}.jsonl"):
            full[r["qid"]] = r
    add("AgenticRL Full", list(full.values()), mode="agent")

    print("\n" + "=" * 62)
    print("P1 cost & stability")
    print("=" * 62)
    print(f"{'configuration':<20}{'acc':>8}{'correct':>9}{'rounds':>8}{'calls':>7}"
          f"{'multi%':>8}{'PF%':>7}{'FB%':>7}{'evid':>6}")
    for r in rows:
        print(f"{r['configuration']:<20}{r['accuracy']*100:>7.2f}%{r['correct']:>9}"
              f"{r['mean_rounds']:>8.2f}{r['mean_retrieval_calls']:>7.2f}"
              f"{r['multi_round_rate']*100:>7.1f}%{r['parse_fail_rate']*100:>6.2f}%"
              f"{r['fallback_rate']*100:>6.1f}%{r['mean_final_evidence']:>6.2f}")

    (HERE / "p1_cost_stability.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(HERE / "p1_cost_stability.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("  -> p1_cost_stability.json / .csv")


if __name__ == "__main__":
    p0_5()
    p1()
