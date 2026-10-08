#!/usr/bin/env python3
"""Stage D:统一口径 reward(Δ logP_options) + 组内优势(GRPO)。

统一口径(v2):对每条 rollout,用**冻结生成器**(Qwen3-VL-8B)读正确答案在
{A,B,C,D} 上的归一化对数概率,与"仅查询图无证据"基线相比:

    u = logP_options(a* | 查询图, 证据) - logP_options(a* | 查询图)

log 域不饱和,与 judge 分档 / 在线 GRPO 同口径(--normalize 默认 options)。

按 group 成员 kind 打分:
  - select          : keep 集合整体一次前向,u_set 即 reward(集合级)
  - rewrite         : selected 每条 passage 单独算 u_i,R = 0.7*top3均值 + 0.3*全部均值
  - anchor_original : 原始检索窗(obs 全集)同 rewrite 公式,作"不改写"对比基准
  - anchor_none     : 无证据,u = 0(不占前向)

rewrite 成员另减 --rewrite-cost;仅"答案是X/最终答案"直白泄漏扣分(--leak-penalty),
answer_text 命中只监控不进 reward。输出 member 级 leak 字段与 row 级 leakage_rate。
组内 z 归一化得 advantage(attach_group_advantages)。

env: qwen3vl-rerank(生成器 = Qwen3-VL-8B)。可分片(--start/--limit)。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

from gen_scorer import AnswerScorer  # noqa: E402
from reward_model import attach_group_advantages, hard_leakage, leakage_penalty  # noqa: E402

DEFAULT_GEN = "/mnt/data_10/mwx/huggingface_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
EPS = 1e-6


def to_evidence(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the fields rag_prompting.format_evidence needs (production-aligned prompt)."""
    return [{
        "text": p.get("text", "") or "",
        "image_path": p.get("image_path", "") or "",
        "doc_name": p.get("doc_name", "") or "",
        "page_idx": p.get("page_idx", ""),
        "score": p.get("score", 0.0),
    } for p in (passages or [])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Stage C output (rollouts_selected_v2*.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-model", default=DEFAULT_GEN)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--normalize", default="options", choices=["vocab", "options"])
    ap.add_argument("--rewrite-cost", type=float, default=0.05,
                    help="subtracted from every REWRITE member's reward")
    ap.add_argument("--leak-penalty", type=float, default=0.5,
                    help="weight on HARD leakage only ('答案是X'/'最终答案'); answer_text hit is monitor-only")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    sc = AnswerScorer(args.gen_model, device=args.device)
    print(f"[生成器] 就绪 {Path(args.gen_model).name} on {args.device} normalize={args.normalize}", flush=True)

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
    print(f"[数据] 待处理 {len(rows)} 题(start={args.start} limit={args.limit})", flush=True)

    out_f = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    n = 0
    nonzero_var = 0
    leak_sum = 0.0
    for row in rows:
        opts = row.get("options", {}) or {}
        gold = str(row.get("answer", "") or "")
        qimg = row.get("query_image_path", "")
        q = row.get("question", "")
        if not opts or not gold:
            row["reward_skipped"] = True
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            continue

        p_base = sc.answer_prob(q, opts, gold, qimg, evidence=[], normalize=args.normalize)
        lp_base = math.log(max(p_base, EPS))
        row["p_base"] = p_base

        members = row.get("group", [])
        n_rw = 0
        n_leak = 0
        for r in members:
            kind = r.get("kind") or ("rewrite" if r.get("action") == "REWRITE" else "select")
            sel = r.get("selected_passages") or []
            if kind == "anchor_none":
                r["u_set"] = 0.0
                r["reward"] = 0.0
            elif kind in ("rewrite", "anchor_original"):
                # per-passage utility, aggregated: R = 0.7*mean(top3) + 0.3*mean(all)
                us: List[float] = []
                for psg in sel:
                    p_i = sc.answer_prob(q, opts, gold, qimg, evidence=to_evidence([psg]),
                                         normalize=args.normalize)
                    us.append(math.log(max(p_i, EPS)) - lp_base)
                top = sorted(us, reverse=True)[:3]
                agg = ((0.7 * sum(top) / len(top)) if top else 0.0) + \
                      ((0.3 * sum(us) / len(us)) if us else 0.0)
                r["u_passages"] = [round(x, 6) for x in us]
                r["u_set"] = float(agg)
                r["reward"] = float(agg)
            else:  # select: the keep-set scored as a whole
                p_sel = sc.answer_prob(q, opts, gold, qimg, evidence=to_evidence(sel),
                                       normalize=args.normalize)
                u = math.log(max(p_sel, EPS)) - lp_base
                r["p_sel"] = p_sel
                r["u_set"] = float(u)
                r["reward"] = float(u)
            if r.get("action") == "REWRITE":
                n_rw += 1
                r["reward"] = float(r["reward"]) - args.rewrite_cost
                query = str(r.get("rewrite_query", "") or "")
                leak = leakage_penalty(query, answer=gold, answer_text=row.get("answer_text", ""))
                r["leak"] = float(leak)
                r["reward"] -= args.leak_penalty * hard_leakage(query, answer=gold)
                if leak > 0:
                    n_leak += 1
        row["leakage_rate"] = (n_leak / n_rw) if n_rw else 0.0
        leak_sum += row["leakage_rate"]
        attach_group_advantages(members)

        rewards = [r.get("reward", 0.0) for r in members]
        if members and max(rewards) - min(rewards) > 1e-6:
            nonzero_var += 1
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()
        n += 1
        if n % args.log_every == 0:
            dt = time.time() - t0
            print(f"[{n}/{len(rows)}] {dt / n:.2f}s/题 组内有梯度占比={nonzero_var}/{n} "
                  f"平均leakage_rate={leak_sum / n:.3f}", flush=True)

    out_f.close()
    print(f"[完成] 写出 {n} 题 -> {args.out}(组内有梯度 {nonzero_var}/{n}, "
          f"平均leakage_rate={leak_sum / max(n, 1):.3f})", flush=True)


if __name__ == "__main__":
    main()
