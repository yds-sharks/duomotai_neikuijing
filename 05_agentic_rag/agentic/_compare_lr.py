#!/usr/bin/env python
"""Compare lr sensitivity: lr=1e-6 baseline vs lr=3e-6 test (first 150 states).

Usage: python _compare_lr.py
Overlays R_mean moving-average of both runs (same first-150 data subset) and
prints slope / kl comparison to judge whether higher lr lifts reward.
"""
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIM = 150
BASE = ("lr=1e-6 (baseline)", "train/grpo_v3.log", "#08519c")
TEST = ("lr=3e-6 (test)", "train/grpo_lr3e6_test.log", "#d94801")


def load(log, lim=LIM):
    rows = []
    for ln in open(log, encoding="utf-8", errors="ignore").read().splitlines():
        m = re.match(r"\[e0 s(\d+)\]", ln)
        if not m or "flat-skip" in ln:
            continue
        si = int(m.group(1))
        if si >= lim:
            continue
        d = {"si": si}
        for k in ("R_mean", "kl"):
            mm = re.search(rf"{k}=([+\-\d.]+)", ln)
            d[k] = float(mm.group(1)) if mm else None
        rows.append(d)
    return rows


def slope_kl(rows):
    v = [r for r in rows if r["R_mean"] is not None]
    if len(v) < 3:
        return None, None, len(v)
    xs = [r["si"] for r in v]; ys = [r["R_mean"] for r in v]; n = len(v)
    mx = sum(xs) / n; my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    kl = statistics.mean(r["kl"] for r in v if r["kl"] is not None)
    return slope, kl, n


def moving_avg(rows, w=7):
    v = [r for r in rows if r["R_mean"] is not None]
    xs = [r["si"] for r in v]; ys = [r["R_mean"] for r in v]; n = len(ys)
    ox, oy, h = [], [], w // 2
    for i in range(n):
        seg = ys[max(0, i - h):min(n, i + h + 1)]
        if seg:
            ox.append(xs[i]); oy.append(sum(seg) / len(seg))
    return ox, oy


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LR sensitivity test  (first 150 states, same data)",
                 fontsize=14, fontweight="bold")
    print(f"{'run':<22}{'n':>5}{'R_mean slope':>16}{'kl avg':>10}")
    for label, log, color in (BASE, TEST):
        rows = load(log)
        sl, kl, n = slope_kl(rows)
        ox, oy = moving_avg(rows)
        axes[0].plot(ox, oy, color=color, lw=2, label=f"{label} (mov-avg)")
        if sl is not None:
            print(f"{label:<22}{n:>5}{sl:>+16.6f}{kl:>10.4f}")
        else:
            print(f"{label:<22}{n:>5}{'(too few)':>16}{'-':>10}")
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_title("R_mean moving-average (effect)")
    axes[0].set_xlabel("state index"); axes[0].set_ylabel("R_mean")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # kl growth comparison
    for label, log, color in (BASE, TEST):
        rows = [r for r in load(log) if r["kl"] is not None]
        if rows:
            axes[1].plot([r["si"] for r in rows], [r["kl"] for r in rows],
                         color=color, lw=1.4, alpha=0.8, label=label)
    axes[1].set_title("KL growth (policy movement)")
    axes[1].set_xlabel("state index"); axes[1].set_ylabel("kl")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("train/lr_compare.png", dpi=120)
    print("saved -> train/lr_compare.png")


if __name__ == "__main__":
    main()
