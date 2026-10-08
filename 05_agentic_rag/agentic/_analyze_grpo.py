import re
import sys

log = sys.argv[1] if len(sys.argv) > 1 else "train/grpo_v3.log"
TOTAL_STATES = 1696

nonflat = []   # dicts of metrics
flat = 0
order = []     # (s, type)

re_nf = re.compile(
    r"\[e0 s(\d+)\] R_mean=([+-]?[\d.]+) R_max=([+-]?[\d.]+) "
    r"pg=([+-]?[\d.]+) kl=([\d.]+) lp_base=([+-]?[\d.]+) "
    r"u=\[([+-]?[\d.]+),([+-]?[\d.]+)\] keep0=\[([^\]]*)\] "
    r"t\[pre=(\d+) gen=(\d+) score=(\d+) bwd=(\d+) tot=(\d+)\]"
)
re_flat = re.compile(r"\[e0 s(\d+)\] flat-skip")

with open(log) as f:
    for line in f:
        m = re_nf.search(line)
        if m:
            s = int(m.group(1))
            nonflat.append({
                "s": s,
                "R_mean": float(m.group(2)), "R_max": float(m.group(3)),
                "pg": float(m.group(4)), "kl": float(m.group(5)),
                "lp_base": float(m.group(6)),
                "u_lo": float(m.group(7)), "u_hi": float(m.group(8)),
                "keep0": m.group(9).strip(),
                "pre": int(m.group(10)), "gen": int(m.group(11)),
                "score": int(m.group(12)), "bwd": int(m.group(13)),
                "tot": int(m.group(14)),
            })
            order.append((s, "nf"))
            continue
        mf = re_flat.search(line)
        if mf:
            flat += 1
            order.append((int(mf.group(1)), "flat"))

n_nf = len(nonflat)
n_tot = n_nf + flat
if n_tot == 0:
    print("no states parsed"); sys.exit(0)

last_s = max(s for s, _ in order)
flat_frac = flat / n_tot

# timing
avg_tot_nf = sum(x["tot"] for x in nonflat) / n_nf
avg_bwd = sum(x["bwd"] for x in nonflat) / n_nf
avg_gen = sum(x["gen"] for x in nonflat) / n_nf
avg_score = sum(x["score"] for x in nonflat) / n_nf
FLAT_SECS = 2.0
eff_avg = (1 - flat_frac) * avg_tot_nf + flat_frac * FLAT_SECS
elapsed = sum(x["tot"] for x in nonflat) + flat * FLAT_SECS
remaining = TOTAL_STATES - (last_s + 1)
eta_sec = remaining * eff_avg

def hm(sec):
    return f"{sec/3600:.1f}h"

print("=" * 60)
print(f"进度: s{last_s}/{TOTAL_STATES}  ({100*(last_s+1)/TOTAL_STATES:.1f}%)")
print(f"非flat {n_nf} | flat-skip {flat} ({100*flat_frac:.0f}%)")
print("-" * 60)
print("时序 (非flat状态均值):")
print(f"  gen={avg_gen:.0f}s  score={avg_score:.1f}s  bwd={avg_bwd:.0f}s  tot={avg_tot_nf:.0f}s")
print(f"  flat状态约 {FLAT_SECS:.0f}s")
print(f"  有效均速 = {eff_avg:.1f}s/状态")
print("-" * 60)
print("时长预估:")
print(f"  已用时 ≈ {hm(elapsed)}")
print(f"  剩余 {remaining} 状态 × {eff_avg:.1f}s ≈ {hm(eta_sec)}")
print(f"  全程 ≈ {hm(elapsed + eta_sec)}")
print("=" * 60)

# health: split into early vs recent halves
half = n_nf // 2
early = nonflat[:half] if half else nonflat
recent = nonflat[half:] if half else nonflat

def avg(lst, k):
    return sum(x[k] for x in lst) / len(lst) if lst else 0.0

print("训练健康指标 (前半 vs 后半):")
print(f"  {'指标':<10}{'前半':>12}{'后半':>12}")
for k in ["R_mean", "R_max", "pg", "kl", "lp_base"]:
    print(f"  {k:<10}{avg(early,k):>12.4f}{avg(recent,k):>12.4f}")

# reward positive fraction
pos_rmax = sum(1 for x in nonflat if x["R_max"] > 0)
print("-" * 60)
print(f"  R_max>0 占比: {100*pos_rmax/n_nf:.0f}%  (有正reward信号的状态)")
kl_max = max(x["kl"] for x in nonflat)
kl_recent = avg(recent, "kl")
print(f"  kl: max={kl_max:.4f}  后半均值={kl_recent:.4f}")
print(f"  pg: 后半均值={avg(recent,'pg'):.4f}")

# keep0 empty fraction (controller choosing no evidence)
empty_keep = sum(1 for x in nonflat if x["keep0"] == "")
print(f"  keep0为空占比: {100*empty_keep/n_nf:.0f}%")

# health verdict
print("=" * 60)
issues = []
if kl_max > 0.5:
    issues.append(f"kl 过大({kl_max:.3f})，policy 偏离 ref 过快")
if avg(recent, "R_max") < 0:
    issues.append("后半 R_max 均值为负，reward 信号可能塌缩")
if pos_rmax / n_nf < 0.3:
    issues.append("正reward状态占比过低(<30%)")
if flat_frac > 0.6:
    issues.append(f"flat-skip 率过高({flat_frac:.0%})，多数状态无信号")
if not issues:
    print("健康判断: 良性 ✓  (kl 受控、reward 有正信号、flat-skip 合理)")
else:
    print("健康判断: 需注意")
    for it in issues:
        print("  - " + it)
