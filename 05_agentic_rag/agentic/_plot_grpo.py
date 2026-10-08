#!/usr/bin/env python
"""Plot GRPO training curves from a run log.

Usage:
    python _plot_grpo.py [logfile] [outfile.png]

Defaults: logfile=train/grpo_v4.log  outfile=train/grpo_v4_curves.png

Panels:
  1. R_mean  (raw per-state + moving average)  -> 效果主指标
  2. probe mean_u (greedy held-out)            -> 真实进度指标
  3. kl (per-update-window avg)                -> 探索预算/约束
  4. gnorm (pre-clip)                          -> 训练稳定性
"""
import re
import sys
import collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse(log: str):
    """Return (per-state rows, probe points); flat/spread-skip lines excluded."""
    rows = []
    probes = []  # (update, mean_u)
    pat = re.compile(r"\[e0 s(\d+)\]")
    ppat = re.compile(r"\[probe (?:final_)?u(\d+)\] mean_u=([+\-\d.]+)")
    for ln in open(log, encoding="utf-8", errors="ignore").read().splitlines():
        pm = ppat.match(ln)
        if pm:
            probes.append((int(pm.group(1)), float(pm.group(2))))
            continue
        m = pat.match(ln)
        if not m or "flat-skip" in ln or "spread-skip" in ln:
            continue
        d = {"si": int(m.group(1))}
        for key in ("R_mean", "R_max", "pg", "kl"):
            mm = re.search(rf"{key}=([+\-\d.]+)", ln)
            d[key] = float(mm.group(1)) if mm else None
        mu = re.search(r"u=(\d+)", ln)
        d["u"] = int(mu.group(1)) if mu else None
        mg = re.search(r"gnorm=([\d.]+)", ln)
        d["gnorm"] = float(mg.group(1)) if mg else None
        rows.append(d)
    return rows, probes


def moving_avg(xs, ys, w):
    """Centered moving average over window w (by index)."""
    out_x, out_y = [], []
    n = len(ys)
    h = w // 2
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        seg = [v for v in ys[lo:hi] if v is not None]
        if seg:
            out_x.append(xs[i])
            out_y.append(sum(seg) / len(seg))
    return out_x, out_y


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else "train/grpo_v4.log"
    out = sys.argv[2] if len(sys.argv) > 2 else "train/grpo_v4_curves.png"
    rows, probes = parse(log)
    if not rows:
        print("no data parsed"); return
    si = [r["si"] for r in rows]
    rmean = [r["R_mean"] for r in rows]
    last_u = rows[-1]["u"]
    last_si = rows[-1]["si"]

    # per-update-window aggregates for kl / gnorm
    by_u_kl = collections.defaultdict(list)
    by_u_gn = collections.defaultdict(list)
    for r in rows:
        if r["u"] is None:
            continue
        if r["kl"] is not None:
            by_u_kl[r["u"]].append(r["kl"])
        if r["gnorm"] is not None:
            by_u_gn[r["u"]].append(r["gnorm"])
    us = sorted(by_u_kl)
    kl_u = [sum(by_u_kl[u]) / len(by_u_kl[u]) for u in us]
    gn_u = [sum(by_u_gn[u]) / len(by_u_gn[u]) for u in us if u in by_u_gn]
    gn_us = [u for u in us if u in by_u_gn]

    w = max(5, len(rows) // 40)  # moving-average window ~2.5% of run
    ma_x, ma_y = moving_avg(si, rmean, w)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"GRPO v4 training curves  (last s{last_si}, u={last_u})",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.scatter(si, rmean, s=5, alpha=0.25, color="#9ecae1", label="R_mean per-state")
    ax.plot(ma_x, ma_y, color="#08519c", lw=2, label=f"moving avg (w={w})")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_title("(1) R_mean  [reward = main effect metric]")
    ax.set_xlabel("state index"); ax.set_ylabel("R_mean (DlogP_options)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if probes:
        px = [p[0] for p in probes]
        py = [p[1] for p in probes]
        ax.plot(px, py, color="#238b45", lw=2, marker="o", ms=5)
        ax.axhline(py[0], color="gray", ls="--", lw=0.8, label="u0 baseline")
        ax.legend()
    ax.set_title("(2) probe mean_u  [greedy held-out = TRUE progress]")
    ax.set_xlabel("update u"); ax.set_ylabel("mean_u (greedy)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(us, kl_u, color="#d94801", lw=1.6, marker=".", ms=3)
    ax.set_title("(3) KL  [exploration budget, keep controlled]")
    ax.set_xlabel("update u"); ax.set_ylabel("kl (per-update avg)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(gn_us, gn_u, color="#6a51a3", lw=1.6, marker=".", ms=3)
    ax.set_title("(4) gnorm  [pre-clip, stability]")
    ax.set_xlabel("update u"); ax.set_ylabel("grad norm")
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=120)
    print(f"saved -> {out}")
    # quick text summary
    tail = ma_y[-5:] if len(ma_y) >= 5 else ma_y
    print(f"R_mean moving-avg last5: {[round(v,3) for v in tail]}")
    if probes:
        print(f"probe mean_u: {[(u, round(v,3)) for u, v in probes]}")
    print(f"kl last: {kl_u[-1]:.4f}   gnorm last: {gn_u[-1]:.1f}")


if __name__ == "__main__":
    main()
