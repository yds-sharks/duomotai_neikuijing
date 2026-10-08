#!/usr/bin/env python3
"""P0-1/2/3: 统一全量逐题结果 + Figure3 动作分解 + Figure4 临床分解.

输出:
  full_6832_per_question.jsonl      P0-1 统一逐题
  figure3_action_breakdown.json     P0-2 动作分解 + bootstrap CI
  clinical_breakdown_final.csv/json P0-3 临床分层 + 配对bootstrap + 精确McNemar
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
B = 10000
SEED = 20260729


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------- P0-1: unified per-question ----------------
def build_unified():
    single = {r["qid"]: r for r in load_jsonl(HERE / "results_v2" / "agentic_samples.jsonl")}

    full = {}
    for name in ["rescue_full_u50_shard0", "rescue_full_u50_shard1",
                 "rescue_harm_u50_shard0", "rescue_harm_u50_shard1"]:
        for r in load_jsonl(HERE / f"{name}.jsonl"):
            full[r["qid"]] = r

    rows = []
    for qid, s in single.items():
        f = full.get(qid)
        if f is None:
            raise SystemExit(f"missing qid in Config4: {qid}")
        rounds = f.get("rounds", [])
        n_rounds = f.get("n_rounds", len(rounds))
        # retrieval calls: round-0 initial retrieval (image+text) counts as 1,
        # each additional round issues one rewrite retrieval
        retrieval_calls = n_rounds
        parse_fail = any(not rd.get("parse_ok", True) for rd in rounds)
        rows.append({
            "qid": qid,
            "index": s.get("index"),
            "scene": s.get("scene", ""),
            "task": s.get("task", ""),
            "category": s.get("category", ""),
            "gold": s.get("gold", ""),
            "single_pred": s.get("pred", ""),
            "single_correct": bool(s.get("correct", False)),
            "single_initial_action": s.get("action", ""),
            "full_pred": f.get("pred", ""),
            "full_correct": bool(f.get("correct", False)),
            "rounds": n_rounds,
            "retrieval_calls": retrieval_calls,
            "parse_fail": parse_fail,
            "fallback_used": bool(f.get("fallback_to_orig", False)),
            "n_final_evidence": f.get("n_evidence", 0),
        })
    rows.sort(key=lambda r: r["index"])

    out = HERE / "full_6832_per_question.jsonl"
    with open(out, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    # verification
    n = len(rows)
    sc = sum(r["single_correct"] for r in rows)
    fc = sum(r["full_correct"] for r in rows)
    cc = sum(1 for r in rows if r["single_correct"] and r["full_correct"])
    cw = sum(1 for r in rows if r["single_correct"] and not r["full_correct"])
    wc = sum(1 for r in rows if not r["single_correct"] and r["full_correct"])
    ww = sum(1 for r in rows if not r["single_correct"] and not r["full_correct"])
    expect = {"n": 6832, "single": 2741, "full": 2927,
              "cc": 2491, "cw": 250, "wc": 436, "ww": 3655}
    got = {"n": n, "single": sc, "full": fc, "cc": cc, "cw": cw, "wc": wc, "ww": ww}
    print("=" * 62)
    print("P0-1 verification")
    print("=" * 62)
    for k in expect:
        flag = "OK " if expect[k] == got[k] else "FAIL"
        print(f"  [{flag}] {k:<7} expected={expect[k]:<6} got={got[k]}")
    print(f"  -> {out.name}")
    return rows


# ---------------- bootstrap helpers ----------------
def strat_bootstrap_ratio(items, strata, num_fn, den_fn, rng):
    """Stratified bootstrap for a ratio; items grouped by strata labels."""
    by = defaultdict(list)
    for it, st in zip(items, strata):
        by[st].append(it)
    keys = sorted(by)
    arrs = {k: np.array([[num_fn(it), den_fn(it)] for it in by[k]], dtype=float)
            for k in keys}
    est = []
    for _ in range(B):
        num = den = 0.0
        for k in keys:
            a = arrs[k]
            idx = rng.integers(0, len(a), len(a))
            s = a[idx].sum(axis=0)
            num += s[0]
            den += s[1]
        est.append(num / den if den else np.nan)
    return np.array(est)


def paired_bootstrap_delta(pairs, rng, strata=None):
    """Paired bootstrap for accuracy difference (right - left), in pp."""
    arr = np.array(pairs, dtype=float)  # columns: left_correct, right_correct
    if strata is None:
        n = len(arr)
        out = np.empty(B)
        for i in range(B):
            idx = rng.integers(0, n, n)
            s = arr[idx]
            out[i] = (s[:, 1].mean() - s[:, 0].mean()) * 100
        return out
    by = defaultdict(list)
    for row, st in zip(arr, strata):
        by[st].append(row)
    keys = sorted(by)
    sub = {k: np.array(by[k]) for k in keys}
    out = np.empty(B)
    for i in range(B):
        tot_l = tot_r = cnt = 0.0
        for k in keys:
            a = sub[k]
            idx = rng.integers(0, len(a), len(a))
            s = a[idx]
            tot_l += s[:, 0].sum()
            tot_r += s[:, 1].sum()
            cnt += len(a)
        out[i] = (tot_r - tot_l) / cnt * 100
    return out


def exact_mcnemar(b, c):
    """Exact (binomial) McNemar two-sided p-value."""
    nn = b + c
    if nn == 0:
        return 1.0
    return float(binomtest(min(b, c), nn, 0.5, alternative="two-sided").pvalue)


# ---------------- P0-2: Figure 3 action breakdown ----------------
def figure3(rows):
    rng = np.random.default_rng(SEED)
    result = {"B": B, "seed": SEED, "groups": {}, "rates": {}}
    print("\n" + "=" * 62)
    print("P0-2 Figure 3 action breakdown")
    print("=" * 62)
    hdr = f"{'action':<9}{'n':>6}{'S_cor':>7}{'S_wrong':>8}{'C->C':>7}{'C->W':>7}{'W->C':>7}{'W->W':>7}"
    print(hdr)
    for act in ["ACCEPT", "REWRITE"]:
        g = [r for r in rows if r["single_initial_action"] == act]
        s_cor = sum(1 for r in g if r["single_correct"])
        s_wrong = len(g) - s_cor
        cc = sum(1 for r in g if r["single_correct"] and r["full_correct"])
        cw = sum(1 for r in g if r["single_correct"] and not r["full_correct"])
        wc = sum(1 for r in g if not r["single_correct"] and r["full_correct"])
        ww = sum(1 for r in g if not r["single_correct"] and not r["full_correct"])
        print(f"{act:<9}{len(g):>6}{s_cor:>7}{s_wrong:>8}{cc:>7}{cw:>7}{wc:>7}{ww:>7}")
        result["groups"][act] = {"n": len(g), "single_correct": s_cor,
                                "single_wrong": s_wrong, "correct_to_correct": cc,
                                "correct_to_wrong": cw, "wrong_to_correct": wc,
                                "wrong_to_wrong": ww}

    # four rates with stratified bootstrap (strata = scene x task)
    print(f"\n{'rate':<28}{'estimate':>10}{'ci_low':>9}{'ci_high':>9}")
    for act in ["ACCEPT", "REWRITE"]:
        g = [r for r in rows if r["single_initial_action"] == act]
        # correction rate: W->C / single wrong
        wrong = [r for r in g if not r["single_correct"]]
        st = [f"{r['scene']}|{r['task']}" for r in wrong]
        d = strat_bootstrap_ratio(wrong, st, lambda r: 1.0 if r["full_correct"] else 0.0,
                                  lambda r: 1.0, rng)
        est = sum(1 for r in wrong if r["full_correct"]) / len(wrong)
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        result["rates"][f"{act}_correction_rate"] = {
            "estimate": est, "ci_low": float(lo), "ci_high": float(hi),
            "numerator": sum(1 for r in wrong if r["full_correct"]),
            "denominator": len(wrong)}
        print(f"{act+'_correction_rate':<28}{est:>10.4f}{lo:>9.4f}{hi:>9.4f}")

        # regression rate: C->W / single correct
        cor = [r for r in g if r["single_correct"]]
        st = [f"{r['scene']}|{r['task']}" for r in cor]
        d = strat_bootstrap_ratio(cor, st, lambda r: 0.0 if r["full_correct"] else 1.0,
                                  lambda r: 1.0, rng)
        est = sum(1 for r in cor if not r["full_correct"]) / len(cor)
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        result["rates"][f"{act}_regression_rate"] = {
            "estimate": est, "ci_low": float(lo), "ci_high": float(hi),
            "numerator": sum(1 for r in cor if not r["full_correct"]),
            "denominator": len(cor)}
        print(f"{act+'_regression_rate':<28}{est:>10.4f}{lo:>9.4f}{hi:>9.4f}")

    (HERE / "figure3_action_breakdown.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  -> figure3_action_breakdown.json")
    return result


# ---------------- P0-3: Figure 4 clinical breakdown ----------------
def figure4(rows):
    rng = np.random.default_rng(SEED)
    out_rows = []
    for level, key in [("category", "category"), ("scene", "scene"), ("task", "task")]:
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        for gname in sorted(groups):
            g = groups[gname]
            sc = sum(1 for r in g if r["single_correct"])
            fc = sum(1 for r in g if r["full_correct"])
            cc = sum(1 for r in g if r["single_correct"] and r["full_correct"])
            cw = sum(1 for r in g if r["single_correct"] and not r["full_correct"])
            wc = sum(1 for r in g if not r["single_correct"] and r["full_correct"])
            ww = sum(1 for r in g if not r["single_correct"] and not r["full_correct"])
            pairs = [(r["single_correct"], r["full_correct"]) for r in g]
            d = paired_bootstrap_delta(pairs, rng)
            lo, hi = np.percentile(d, [2.5, 97.5])
            out_rows.append({
                "level": level, "group_name": gname, "n": len(g),
                "single_correct": sc, "full_correct": fc,
                "single_accuracy": sc / len(g), "full_accuracy": fc / len(g),
                "delta_pp": (fc - sc) / len(g) * 100,
                "correct_to_correct": cc, "correct_to_wrong": cw,
                "wrong_to_correct": wc, "wrong_to_wrong": ww,
                "bootstrap_ci_low": float(lo), "bootstrap_ci_high": float(hi),
                "mcnemar_p": exact_mcnemar(cw, wc),
            })

    # overall row
    sc = sum(1 for r in rows if r["single_correct"])
    fc = sum(1 for r in rows if r["full_correct"])
    cc = sum(1 for r in rows if r["single_correct"] and r["full_correct"])
    cw = sum(1 for r in rows if r["single_correct"] and not r["full_correct"])
    wc = sum(1 for r in rows if not r["single_correct"] and r["full_correct"])
    ww = sum(1 for r in rows if not r["single_correct"] and not r["full_correct"])
    pairs = [(r["single_correct"], r["full_correct"]) for r in rows]
    d = paired_bootstrap_delta(pairs, rng)
    lo, hi = np.percentile(d, [2.5, 97.5])
    out_rows.insert(0, {
        "level": "overall", "group_name": "ALL", "n": len(rows),
        "single_correct": sc, "full_correct": fc,
        "single_accuracy": sc / len(rows), "full_accuracy": fc / len(rows),
        "delta_pp": (fc - sc) / len(rows) * 100,
        "correct_to_correct": cc, "correct_to_wrong": cw,
        "wrong_to_correct": wc, "wrong_to_wrong": ww,
        "bootstrap_ci_low": float(lo), "bootstrap_ci_high": float(hi),
        "mcnemar_p": exact_mcnemar(cw, wc),
    })

    fields = list(out_rows[0].keys())
    with open(HERE / "clinical_breakdown_final.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    (HERE / "clinical_breakdown_final.json").write_text(
        json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print("P0-3 Figure 4 clinical breakdown")
    print("=" * 62)
    print(f"{'level':<9}{'group':<48}{'n':>5}{'single':>8}{'full':>8}{'delta':>8}{'p':>10}")
    for r in out_rows:
        print(f"{r['level']:<9}{r['group_name'][:46]:<48}{r['n']:>5}"
              f"{r['single_accuracy']*100:>7.2f}%{r['full_accuracy']*100:>7.2f}%"
              f"{r['delta_pp']:>+7.2f}{r['mcnemar_p']:>10.2e}")
    print("  -> clinical_breakdown_final.csv / .json")
    return out_rows


if __name__ == "__main__":
    rows = build_unified()
    figure3(rows)
    figure4(rows)
