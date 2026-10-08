#!/usr/bin/env python3
"""Aggregate per-sample eval JSONLs into the paper's main table + behavior analysis.

Reads <out-dir>/<mode>_samples.jsonl for the requested modes and produces:
  1. Main accuracy table (overall + per-scene), markdown + LaTeX rows
  2. Controller behavior table (rewrite rate, acc-by-action, mean keep, parse-fail)
  3. Evidence-utility stats: mean logP(gold) per mode and uplift vs baseline
     (the same delta-logP metric as the training reward, now on EndoBench)

Usage:
  python eval/build_paper_table.py --modes baseline,vanilla_rag,agentic,gpt4o \
      --out-dir eval/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

SCENE_ORDER = ["Gastroscopy", "Colonoscopy", "Capsule Endoscopy", "Surgical Endoscopy"]
MODE_LABEL = {
    "baseline": "Frozen generator (no retrieval)",
    "vanilla_rag": "Frozen RAG pipeline (top-k, no controller)",
    "gpt4o": "Proprietary multimodal controller",
    "agentic": "AgenticRL controller (ours)",
}


def load_records(out_dir: Path, mode: str) -> List[Dict[str, Any]]:
    path = out_dir / f"{mode}_samples.jsonl"
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes", default="baseline,vanilla_rag,gpt4o,agentic")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results_v2"))
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    data: Dict[str, List[Dict[str, Any]]] = {}
    for m in modes:
        data[m] = load_records(out_dir, m)
        print(f"[load] {m}: {len(data[m])} samples", flush=True)
    if not any(data.values()):
        print("no records found; run eval_endobench.py first")
        return

    # ---------- main accuracy table ----------
    scenes = SCENE_ORDER + sorted({r.get("scene", "?") for rows in data.values() for r in rows}
                                  - set(SCENE_ORDER))
    print("\n===== Main table: accuracy =====")
    header = ["method", "overall"] + scenes
    print(("{:<38}" + "{:>9}" * (len(scenes) + 1)).format(*header))
    latex_rows = []
    table: Dict[str, Any] = {}
    for m in modes:
        rows = data[m]
        if not rows:
            continue
        overall = acc(rows)
        per_scene = {}
        for s in scenes:
            per_scene[s] = acc([r for r in rows if r.get("scene") == s])
        table[m] = {"overall": overall, "per_scene": per_scene, "n": len(rows)}
        print(("{:<38}" + "{:>9.4f}" * (len(scenes) + 1)).format(
            MODE_LABEL.get(m, m), overall, *[per_scene[s] for s in scenes]))
        latex_rows.append(
            f"{MODE_LABEL.get(m, m)} & " +
            " & ".join([f"{overall:.3f}"] + [f"{per_scene[s]:.3f}" for s in scenes]) + r" \\")

    print("\n----- LaTeX rows -----")
    for r in latex_rows:
        print(r)

    # ---------- utility (mean logP gold + uplift vs baseline) ----------
    if data.get("baseline"):
        print("\n===== Evidence utility (options-normalized logP(gold)) =====")
        base_by_idx = {r["index"]: r for r in data["baseline"]}
        print(f"{'method':<38}{'mean_logp':>10}{'uplift_vs_base':>15}")
        for m in modes:
            rows = data.get(m)
            if not rows:
                continue
            ml = mean_logp(rows)
            deltas = [r.get("logp_gold", 0.0) - base_by_idx[r["index"]].get("logp_gold", 0.0)
                      for r in rows if r.get("index") in base_by_idx]
            up = sum(deltas) / len(deltas) if deltas else 0.0
            print(f"{MODE_LABEL.get(m, m):<38}{ml:>10.3f}{up:>15.3f}")

    # ---------- controller behavior ----------
    print("\n===== Controller behavior =====")
    beh: Dict[str, Any] = {}
    for m in ("agentic", "gpt4o"):
        rows = data.get(m)
        if not rows:
            continue
        rw = [r for r in rows if r.get("action") == "REWRITE"]
        ac = [r for r in rows if r.get("action") == "ACCEPT"]
        b = {
            "rewrite_rate": len(rw) / len(rows),
            "accept_acc": acc(ac), "rewrite_acc": acc(rw),
            "mean_keep_accept": (sum(len(r.get("keep", [])) for r in ac) / len(ac)) if ac else 0.0,
            "mean_n_evidence": sum(r.get("n_evidence", 0) for r in rows) / len(rows),
            "parse_fail_rate": sum(1 for r in rows if not r.get("parse_ok", True)) / len(rows),
        }
        beh[m] = b
        print(f"[{MODE_LABEL.get(m, m)}] rewrite_rate={b['rewrite_rate']:.3f} "
              f"accept_acc={b['accept_acc']:.4f} rewrite_acc={b['rewrite_acc']:.4f} "
              f"mean_keep={b['mean_keep_accept']:.2f} parse_fail={b['parse_fail_rate']:.3f}")

    # ---------- per-task table for agentic vs vanilla ----------
    if data.get("agentic") and data.get("vanilla_rag"):
        print("\n===== Per-task accuracy: vanilla_rag vs agentic =====")
        tasks = sorted({r.get("task", "?") for r in data["agentic"]})
        print(f"{'task':<34}{'vanilla':>9}{'agentic':>9}{'delta':>8}{'n':>6}")
        for t in tasks:
            rv = [r for r in data["vanilla_rag"] if r.get("task") == t]
            ra = [r for r in data["agentic"] if r.get("task") == t]
            if not rv or not ra:
                continue
            av, aa = acc(rv), acc(ra)
            print(f"{t:<34}{av:>9.4f}{aa:>9.4f}{aa - av:>+8.4f}{len(ra):>6}")

    out = {"table": table, "behavior": beh}
    if args.save:
        Path(args.save).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[save] {args.save}")


if __name__ == "__main__":
    main()
