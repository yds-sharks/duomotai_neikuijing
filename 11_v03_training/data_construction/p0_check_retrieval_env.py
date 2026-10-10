#!/usr/bin/env python3
"""P0 验收：100 题抽查去泄露检索环境。

验收标准（v0.3 方案第 8 节 P0）：
- 自命中 = 0：配额合并后的候选里不得出现题图自身样本
- 同书命中 = 0：不得出现与题目同书 doc_id 的候选
- 文本候选占比 > 30%：确认文本路不再被图像分数挤掉

对照指标：stats 里记录的过滤前泄露（overfetch 阶段的自命中/同书数）即
旧环境（无排除）的泄露水平，用于 before/after 对照。

用法：
  cd 11_v03_training/data_construction
  /mnt/data_1/yds/venvs/qwen35-train/bin/python p0_check_retrieval_env.py \
      --config ../../10_harness/harness_config.json \
      --questions /mnt/data_1/yds/多模态/rerank_image_and_text/agentic/outputs/mcq_image_v2_4000/train.jsonl \
      --n 100 --outdir ./p0_check_out
先冒烟：--n 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leave_one_out_retriever import LeaveOneOutRetriever  # noqa: E402


def load_questions(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def stratified_sample(items: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    """按 query_type 分层抽样，尽量均衡。"""
    rng = random.Random(seed)
    by_type: Dict[str, List[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_type[str(it.get("query_type", "unknown"))].append(i)
    types = sorted(by_type)
    per_type = max(1, n // len(types)) if types else n
    picked: List[int] = []
    for t in types:
        idx = by_type[t][:]
        rng.shuffle(idx)
        picked.extend(idx[:per_type])
    if len(picked) < n:  # 不足则从剩余里随机补齐
        rest = [i for i in range(len(items)) if i not in set(picked)]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return [items[i] for i in sorted(picked[:n])]


def build_text_query(item: Dict[str, Any]) -> str:
    """朴素下界检索式：题干 + 选项全文（不使用任何改写模型）。"""
    opts = item.get("options") or {}
    opt_s = "；".join(f"{k}. {v}" for k, v in opts.items())
    return f"{item.get('question', '')} 选项：{opt_s}".strip()


def gold_keys(item: Dict[str, Any]) -> Dict[str, List[str]]:
    """从题目的 source/gold_source 取排除键。"""
    src = item.get("source") or item.get("gold_source") or {}
    sample_ids = [str(src.get("sample_id") or "")]
    doc_ids = [str(src.get("doc_id") or "")]
    image_paths = [str(item.get("query_image_path") or "")]
    return {
        "exclude_sample_ids": [s for s in sample_ids if s],
        "exclude_doc_ids": [d for d in doc_ids if d],
        "exclude_image_paths": [p for p in image_paths if p],
    }


def check_one(retriever: LeaveOneOutRetriever, item: Dict[str, Any]) -> Dict[str, Any]:
    q = build_text_query(item)
    keys = gold_keys(item)
    out = retriever.retrieve(q, str(item.get("query_image_path") or ""), **keys)
    combined = out["combined"]
    stats = out["stats"]

    # 过滤后仍残留的泄露（必须为 0）
    sid = set(keys["exclude_sample_ids"])
    img = set(keys["exclude_image_paths"])
    doc = set(keys["exclude_doc_ids"])
    self_after = sum(
        1 for h in combined
        if (str(h.get("sample_id") or "") and str(h.get("sample_id")) in sid)
        or (str(h.get("image_path") or "") and str(h.get("image_path")) in img)
    )
    book_after = sum(1 for h in combined if str(h.get("doc_id") or "") and str(h.get("doc_id")) in doc)

    text_n = len(out["text"])
    image_n = len(out["image"])
    total = text_n + image_n
    rec: Dict[str, Any] = {
        "qid": item.get("qid", ""),
        "query_type": item.get("query_type", ""),
        "text_query": q,
        "gold": keys,
        "self_hit_after": self_after,
        "same_book_after": book_after,
        "leak_before": {  # 过滤掉的泄露 = 旧环境会看到的泄露（结构性对照）
            "self": stats["text"]["drop_self"] + stats["image"]["drop_self"],
            "same_book": stats["text"]["drop_same_book"] + stats["image"]["drop_same_book"],
        },
        "text_n": text_n,
        "image_n": image_n,
        "total_n": total,
        "text_ratio": round(text_n / total, 4) if total else 0.0,
        "text_overfetch": stats["text"]["overfetch"],
        "image_overfetch": stats["image"]["overfetch"],
        "text_scores": [round(float(h.get("score") or 0.0), 4) for h in out["text"][:3]],
        "image_scores": [round(float(h.get("score") or 0.0), 4) for h in out["image"][:3]],
    }
    return rec


def summarize(records: List[Dict[str, Any]], args: argparse.Namespace, seconds: float) -> Dict[str, Any]:
    n = len(records)
    self_before_q = sum(1 for r in records if r["leak_before"]["self"] > 0)
    book_before_q = sum(1 for r in records if r["leak_before"]["same_book"] > 0)
    self_after = sum(r["self_hit_after"] for r in records)
    book_after = sum(r["same_book_after"] for r in records)
    ratios = [r["text_ratio"] for r in records if r["total_n"] > 0]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    text_zero = sum(1 for r in records if r["text_n"] == 0)
    by_type = defaultdict(lambda: {"n": 0, "text_zero": 0, "avg_ratio": 0.0})
    buf = defaultdict(list)
    for r in records:
        b = by_type[r["query_type"]]
        b["n"] += 1
        buf[r["query_type"]].append(r["text_ratio"])
        if r["text_n"] == 0:
            b["text_zero"] += 1
    for t, b in by_type.items():
        b["avg_ratio"] = round(sum(buf[t]) / len(buf[t]), 4) if buf[t] else 0.0
    passed = (self_after == 0) and (book_after == 0) and (avg_ratio > 0.30)
    return {
        "verdict": "PASS" if passed else "FAIL",
        "acceptance": {
            "self_hit_after": self_after,
            "same_book_after": book_after,
            "avg_text_ratio": round(avg_ratio, 4),
            "thresholds": {"self_hit_after": 0, "same_book_after": 0, "avg_text_ratio_min": 0.30},
        },
        "baseline_before_filter": {  # 旧环境（无排除）的泄露水平对照
            "questions_with_self_hit": self_before_q,
            "questions_with_same_book": book_before_q,
        },
        "n_questions": n,
        "avg_total_candidates": round(sum(r["total_n"] for r in records) / n, 2) if n else 0.0,
        "questions_with_zero_text": text_zero,
        "text_n_dist": dict(Counter(r["text_n"] for r in records)),
        "image_n_dist": dict(Counter(r["image_n"] for r in records)),
        "by_query_type": dict(by_type),
        "config": {
            "questions": args.questions,
            "sample_seed": args.seed,
            "text_k": args.text_k,
            "image_k": args.image_k,
            "overfetch_k": args.overfetch_k,
        },
        "elapsed_seconds": round(seconds, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="harness_config.json 路径（取 retrieval 段）")
    ap.add_argument("--questions", required=True, help="题目 jsonl（含 source.doc_id/sample_id）")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text-k", type=int, default=8)
    ap.add_argument("--image-k", type=int, default=4)
    ap.add_argument("--overfetch-k", type=int, default=40)
    ap.add_argument("--outdir", default="./p0_check_out")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    items = load_questions(args.questions)
    picked = stratified_sample(items, args.n, args.seed)
    type_count = Counter(p.get("query_type") for p in picked)
    print(f"[p0] 题目总数 {len(items)}，抽样 {len(picked)}，题型分布 {dict(type_count)}", flush=True)

    with open(args.config, encoding="utf-8") as f:
        retrieval_cfg = json.load(f)["retrieval"]
    retrieval_cfg["text_k"] = args.overfetch_k
    retrieval_cfg["image_k"] = args.overfetch_k

    t0 = time.time()
    with LeaveOneOutRetriever(retrieval_cfg, text_k=args.text_k, image_k=args.image_k, overfetch_k=args.overfetch_k) as retriever:
        records: List[Dict[str, Any]] = []
        samples_fp = open(outdir / "p0_check_samples.jsonl", "w", encoding="utf-8")
        try:
            for i, item in enumerate(picked, 1):
                try:
                    rec = check_one(retriever, item)
                except Exception as e:  # 单题失败不中断，记录错误
                    rec = {"qid": item.get("qid", ""), "error": f"{type(e).__name__}: {e}"}
                records.append(rec)
                samples_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                samples_fp.flush()
                if i % 10 == 0 or i == len(picked):
                    print(f"[p0] {i}/{len(picked)} elapsed={time.time()-t0:.0f}s", flush=True)
        finally:
            samples_fp.close()

    report = summarize(records, args, time.time() - t0)
    (outdir / "p0_check_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["verdict"] and {"verdict": report["verdict"], "acceptance": report["acceptance"], "baseline_before_filter": report["baseline_before_filter"], "questions_with_zero_text": report["questions_with_zero_text"]}, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
