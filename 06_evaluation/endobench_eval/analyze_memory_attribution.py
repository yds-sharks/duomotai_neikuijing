#!/usr/bin/env python3
"""Memory-attribution statistics for the AgenticRL paper revision.

Computes:
  (A) Full-benchmark paired comparison: AgenticRL Full (memory-aware) vs R3-NM
      (memoryless iterative), all 6,832 questions.
  (B) Matched 1,002-subset three-way chain: Single -> C4-NoMemory -> C4-Memory.
  (C) Memory-exposed subgroup analysis (questions where M+/M- actually carried
      content into the controller context).
Outputs JSON to memory_attribution_stats.json.
"""
import json
import math
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
random.seed(20260729)
B = 10000


def load_jsonl(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def exact_mcnemar(b, c):
    """Two-sided exact binomial test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # log-space accumulation to avoid overflow for large n
    log_half_n = n * math.log(0.5)
    terms = []
    for i in range(0, k + 1):
        log_c = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        terms.append(log_c + log_half_n)
    m = max(terms)
    tail = math.exp(m) * sum(math.exp(t - m) for t in terms)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(pairs, B=B, seed=20260729):
    """pairs: list of (left_correct, right_correct). Returns CI of (right-left) in pp."""
    rng = random.Random(seed)
    n = len(pairs)
    deltas = []
    idx = range(n)
    for _ in range(B):
        s = [pairs[rng.randrange(n)] for _ in idx]
        l = sum(1 for a, _ in s if a)
        r = sum(1 for _, b in s if b)
        deltas.append((r - l) / n * 100.0)
    deltas.sort()
    lo = deltas[int(0.025 * B)]
    hi = deltas[int(0.975 * B)]
    return lo, hi


def paired_stats(pairs, name):
    n = len(pairs)
    both = sum(1 for a, b in pairs if a and b)
    l_only = sum(1 for a, b in pairs if a and not b)
    r_only = sum(1 for a, b in pairs if b and not a)
    neither = n - both - l_only - r_only
    left = both + l_only
    right = both + r_only
    p = exact_mcnemar(l_only, r_only)
    lo, hi = paired_bootstrap_ci(pairs)
    return {
        "comparison": name,
        "n": n,
        "left_correct": left,
        "right_correct": right,
        "left_acc": round(left / n * 100, 2),
        "right_acc": round(right / n * 100, 2),
        "delta_pp": round((right - left) / n * 100, 2),
        "both": both,
        "left_only": l_only,
        "right_only": r_only,
        "neither": neither,
        "exact_mcnemar_p": p,
        "ci_low": round(lo, 2),
        "ci_high": round(hi, 2),
    }


# ---------------------------------------------------------------- load data
full = {r["qid"]: r for r in load_jsonl(os.path.join(HERE, "full_6832_per_question.jsonl"))}

r3nm = {}
for shard in ("R3_NM_shard0", "R3_NM_shard1", "R3_NM_shard1b"):
    p = os.path.join(HERE, "results_full", shard, "agentic_samples.jsonl")
    if os.path.exists(p):
        for r in load_jsonl(p):
            r3nm[r["qid"]] = r

baseline = {r["qid"]: r for r in load_jsonl(os.path.join(HERE, "results", "baseline_samples.jsonl"))}
oneshot = {r["qid"]: r for r in load_jsonl(os.path.join(HERE, "results_v2", "vanilla_rag_samples.jsonl"))}

c4mem_rounds = {}
for f in ("rescue_full_u50_shard0.jsonl", "rescue_full_u50_shard1.jsonl",
          "rescue_harm_u50_shard0.jsonl", "rescue_harm_u50_shard1.jsonl"):
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        for r in load_jsonl(p):
            c4mem_rounds[r["qid"]] = r

c4nm = {r["qid"]: r for r in load_jsonl(os.path.join(HERE, "c4nm_1002_per_question.jsonl"))}

report = {"n_full": len(full), "n_r3nm": len(r3nm), "n_c4nm": len(c4nm),
          "n_c4mem_rounds": len(c4mem_rounds)}

# ------------------------------------------- (A) full benchmark: Full vs R3NM
qids = [q for q in full if q in r3nm]
pairs = [(as_bool(r3nm[q]["correct"]), as_bool(full[q]["full_correct"])) for q in qids]
report["A_full_vs_r3nm"] = paired_stats(pairs, "R3-NM (memoryless iterative) vs AgenticRL Full (memory-aware)")

pairs_sb = [(as_bool(baseline[q]["correct"]), as_bool(full[q]["full_correct"])) for q in qids if q in baseline]
report["A_full_vs_closedbook"] = paired_stats(pairs_sb, "closed-book vs AgenticRL Full")
pairs_os = [(as_bool(oneshot[q]["correct"]), as_bool(full[q]["full_correct"])) for q in qids if q in oneshot]
report["A_full_vs_oneshot"] = paired_stats(pairs_os, "one-shot RAG vs AgenticRL Full")
pairs_sr = [(as_bool(full[q]["single_correct"]), as_bool(full[q]["full_correct"])) for q in full]
report["A_full_vs_single"] = paired_stats(pairs_sr, "single-round vs AgenticRL Full")

# ------------------------------------------- (C) memory-exposed subgroup, full
def memory_exposed(rec):
    rounds = rec.get("rounds") or []
    if isinstance(rounds, str):
        return None
    for rd in rounds:
        if not isinstance(rd, dict):
            continue
        if int(rd.get("n_retained") or 0) > 0 or int(rd.get("n_breadcrumb") or 0) > 0:
            return True
    return False


exposed_full = {}
for q, rec in c4mem_rounds.items():
    e = memory_exposed(rec)
    if e is not None:
        exposed_full[q] = e
report["exposure_full_counts"] = {
    "with_round_traces": len(exposed_full),
    "exposed": sum(1 for v in exposed_full.values() if v),
    "not_exposed": sum(1 for v in exposed_full.values() if not v),
}

ex_q = [q for q in qids if exposed_full.get(q)]
nex_q = [q for q in qids if q in exposed_full and not exposed_full[q]]
if ex_q:
    report["C_exposed_full_vs_r3nm"] = paired_stats(
        [(as_bool(r3nm[q]["correct"]), as_bool(full[q]["full_correct"])) for q in ex_q],
        "memory-exposed subgroup: R3-NM vs AgenticRL Full")
    report["C_exposed_full_vs_single"] = paired_stats(
        [(as_bool(full[q]["single_correct"]), as_bool(full[q]["full_correct"])) for q in ex_q],
        "memory-exposed subgroup: single-round vs AgenticRL Full")
if nex_q:
    report["C_unexposed_full_vs_single"] = paired_stats(
        [(as_bool(full[q]["single_correct"]), as_bool(full[q]["full_correct"])) for q in nex_q],
        "memory-inactive subgroup: single-round vs AgenticRL Full")

# ------------------------------------------- (B) matched 1002 chain
sub = [q for q in c4nm if q in full]
report["B_n_subset"] = len(sub)
report["B_single_vs_c4nm"] = paired_stats(
    [(as_bool(full[q]["single_correct"]), as_bool(c4nm[q]["correct"])) for q in sub],
    "1002 subset: single-round vs C4-NoMemory")
report["B_c4nm_vs_c4mem"] = paired_stats(
    [(as_bool(c4nm[q]["correct"]), as_bool(full[q]["full_correct"])) for q in sub],
    "1002 subset: C4-NoMemory vs C4-Memory")
report["B_single_vs_c4mem"] = paired_stats(
    [(as_bool(full[q]["single_correct"]), as_bool(full[q]["full_correct"])) for q in sub],
    "1002 subset: single-round vs C4-Memory")

# matched subset restricted to memory-exposed questions
sub_ex = [q for q in sub if exposed_full.get(q)]
report["B_n_subset_exposed"] = len(sub_ex)
if sub_ex:
    report["B_exposed_c4nm_vs_c4mem"] = paired_stats(
        [(as_bool(c4nm[q]["correct"]), as_bool(full[q]["full_correct"])) for q in sub_ex],
        "1002 subset, memory-exposed: C4-NoMemory vs C4-Memory")

# ------------------------------------------- action-conditioned on full
by_action = defaultdict(list)
for q, rec in full.items():
    by_action[rec.get("single_initial_action")].append(rec)
act = {}
for a, recs in by_action.items():
    wrong = [r for r in recs if not as_bool(r["single_correct"])]
    right = [r for r in recs if as_bool(r["single_correct"])]
    wc = sum(1 for r in wrong if as_bool(r["full_correct"]))
    cw = sum(1 for r in right if not as_bool(r["full_correct"]))
    act[a] = {
        "n": len(recs), "single_correct": len(right), "single_wrong": len(wrong),
        "W2C": wc, "C2W": cw,
        "correction_rate": round(wc / len(wrong) * 100, 2) if wrong else None,
        "regression_rate": round(cw / len(right) * 100, 2) if right else None,
        "net": wc - cw,
    }
report["action_conditioned"] = act

# ------------------------------------------- M+ / M- activity statistics
n_rewrite_rounds = 0
n_rounds_total = 0
retained_sizes = []
breadcrumb_rounds = 0
for q, rec in c4mem_rounds.items():
    rounds = rec.get("rounds") or []
    if isinstance(rounds, str):
        continue
    for rd in rounds:
        if not isinstance(rd, dict):
            continue
        n_rounds_total += 1
        if rd.get("action") == "REWRITE":
            n_rewrite_rounds += 1
        nr = int(rd.get("n_retained") or 0)
        if nr > 0:
            retained_sizes.append(nr)
        if int(rd.get("n_breadcrumb") or 0) > 0:
            breadcrumb_rounds += 1
report["memory_activity"] = {
    "rounds_total": n_rounds_total,
    "rewrite_rounds": n_rewrite_rounds,
    "rounds_with_nonempty_Mplus": len(retained_sizes),
    "mean_Mplus_size": round(sum(retained_sizes) / len(retained_sizes), 2) if retained_sizes else 0,
    "rounds_with_nonempty_Mminus": breadcrumb_rounds,
}

out = os.path.join(HERE, "memory_attribution_stats.json")
with open(out, "w") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
print(json.dumps(report, indent=2, ensure_ascii=False))
print("\nwritten:", out)
