#!/usr/bin/env python3
"""P2 前置：EndoBench 分题型盲答基线（4B logprob，无检索）。

目的：量化 EndoBench 各 task/category 上 Qwen3.5-4B 的看图盲答正确率，
与旧模板题（mcq_image_v2_4000，P1 中 base_correct=41%）对比，判断
P2 出题的方向与难度是否需要调整（担心：出题过简 → 训练后 benchmark 无显著提升）。

方法：
- 从 runs/queries_endobench_full.jsonl（make_queries.py 导出的 6832 题）按 task 分层抽样
- 复用 P1 的 OptionLogprobScorer（冻结 4B，一次前向出选项分布），看图盲答、无证据
- correct = argmax(选项 logit) == gold；与 v0.2.x harness 生成器同款中文 prompt

用法：
  /mnt/data_1/yds/venvs/qwen35-train/bin/python p2a_endobench_baseline.py \
      --config ../../10_harness/harness_config.json \
      --queries ../../10_harness/runs/queries_endobench_full.jsonl \
      --per-task 40 --outdir ./p2a_baseline_out
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from p1_support_sampling import OptionLogprobScorer  # noqa: E402


def stratified_by_task(rows: List[Dict[str, Any]], per_task: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_task: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_task[r.get("task", "unknown")].append(i)
    picked: List[int] = []
    for t in sorted(by_task):
        idx = by_task[t][:]
        rng.shuffle(idx)
        picked.extend(idx[:per_task])
    return [rows[i] for i in picked]


def summarize(records: List[Dict[str, Any]], per_task: int, seconds: float, out_cfg: Dict[str, Any]) -> Dict[str, Any]:
    n = len(records)
    correct = sum(1 for r in records if r["correct"])

    def agg(key: str) -> Dict[str, Dict[str, Any]]:
        buf: Dict[str, List[bool]] = defaultdict(list)
        lps: Dict[str, List[float]] = defaultdict(list)
        for r in records:
            k = str(r.get(key) or "unknown")
            buf[k].append(r["correct"])
            lps[k].append(r["logp_gold"])
        out: Dict[str, Dict[str, Any]] = {}
        for k in sorted(buf):
            v = buf[k]
            lp = lps[k]
            out[k] = {
                "n": len(v),
                "correct": sum(v),
                "acc": round(sum(v) / len(v), 4),
                "logp_gold_median": round(statistics.median(lp), 3),
            }
        return out

    return {
        "overall": {"n": n, "correct": correct, "acc": round(correct / n, 4) if n else None},
        "by_task": agg("task"),
        "by_category": agg("category"),
        "by_scene": agg("scene"),
        "by_n_options": agg("n_options"),
        "config": dict(out_cfg, per_task=per_task, elapsed_seconds=round(seconds, 1)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--per-task", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generator-device", default="cuda:0")
    ap.add_argument("--outdir", default="./p2a_baseline_out")
    ap.add_argument("--limit", type=int, default=0, help="冒烟：>0 时只取前 limit 题")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.queries, encoding="utf-8")]
    if args.limit > 0:
        picked = rows[: args.limit]
    else:
        picked = stratified_by_task(rows, args.per_task, args.seed)
    print(f"[p2a] 全量 {len(rows)}，抽样 {len(picked)}，task 分布 {dict(Counter(r['task'] for r in picked))}", flush=True)

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    gen_cfg = dict(cfg["generator"])
    gen_cfg["device"] = args.generator_device

    scorer = OptionLogprobScorer(gen_cfg)
    t0 = time.time()
    records: List[Dict[str, Any]] = []
    fp = open(outdir / "p2a_baseline_samples.jsonl", "w", encoding="utf-8")
    try:
        for i, q in enumerate(picked, 1):
            opts = q["options"]
            try:
                sc = scorer.judge(
                    q["question"], opts, str(q.get("answer", "")), image_path=str(q.get("query_image_path") or "")
                )
                rec = {
                    "qid": q["qid"],
                    "task": q.get("task", ""),
                    "category": q.get("category", ""),
                    "scene": q.get("scene", ""),
                    "n_options": len(opts),
                    "pred": sc["pred"],
                    "gold": q.get("answer", ""),
                    "correct": sc["correct"],
                    "logp_gold": round(sc["logp_gold"], 4),
                }
            except Exception as e:
                import traceback

                rec = {"qid": q["qid"], "task": q.get("task", ""), "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc(limit=3)}
            records.append(rec)
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fp.flush()
            if i % 50 == 0 or i == len(picked):
                ok = [r for r in records if "correct" in r]
                print(f"[p2a] {i}/{len(picked)} elapsed={time.time()-t0:.0f}s acc={sum(r['correct'] for r in ok)}/{len(ok)}", flush=True)
    finally:
        fp.close()

    ok_records = [r for r in records if "correct" in r]
    report = summarize(ok_records, args.per_task, time.time() - t0, {"queries": args.queries, "seed": args.seed})
    report["n_errors"] = len(records) - len(ok_records)
    (outdir / "p2a_baseline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "by_task": report["by_task"], "n_errors": report["n_errors"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
