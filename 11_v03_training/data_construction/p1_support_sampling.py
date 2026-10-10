#!/usr/bin/env python3
"""P1 支持度抽样：留一检索环境下，统计"有支持题"占比。

问题（v0.3 方案第 8 节 P1）：排除题图自身与同书 doc_id 之后，每道题还能不能
从**别的书**里搜到支持答案的文段？比例够高 → P2 题库重建按原计划；偏低 →
放宽为只排除自身样本，或扩大源池。

方法：
1. 与 P0 同批 100 题（分层抽样，seed=0）
2. 每题两种 oracle query 各检索一轮（留一环境，文本 8 / 图像 4）：
   - oracle_answer：题干 + gold 答案文本（答案词强信号）
   - oracle_context：题目源样本的 pdf_context_text 前 200 字（源上下文最强信号）
   两轮候选合并去重后为每题候选池（≤24 条）
3. 冻结 Qwen3.5-4B 生成器逐文段打 answer-utility（logprob 版，与 harness 生成器
   同款 prompt 渲染）：
       u_i = logP(gold|题,图,[p_i]) − logP(gold|题,图,∅)
   一次前向取选项字母位置的 logsumexp → log_softmax → logP(gold)
4. 题级判定：有支持 = max_i u_i > τ+（默认 0.2）；有害文段 = u_i < τ−（默认 −0.2）

用法：
  cd 11_v03_training/data_construction
  /mnt/data_1/yds/venvs/qwen35-train/bin/python p1_support_sampling.py \
      --config ../../10_harness/harness_config.json \
      --questions /mnt/data_1/yds/多模态/rerank_image_and_text/agentic/outputs/mcq_image_v2_4000/train.jsonl \
      --n 100 --outdir ./p1_support_out
先冒烟：--n 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_HARNESS = _HERE.parent.parent / "10_harness"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HARNESS))

from leave_one_out_retriever import LeaveOneOutRetriever  # noqa: E402
from p0_check_retrieval_env import gold_keys, load_questions, stratified_sample  # noqa: E402
from backends.reward_backend import render_generator_prompt  # noqa: E402
from backends.transformers_backend import _load  # noqa: E402

TAU_POS = 0.2
TAU_NEG = -0.2
EVIDENCE_MAX_CHARS = 400


def candidate_key(h: Dict[str, Any]) -> str:
    """跨 oracle query 轮的去重键（与 harness candidate_identity 同维度）。"""
    raw = "|".join(
        str(h.get(k) or "")
        for k in ("doc_id", "page_idx", "block_id", "sample_id", "group_id", "image_path", "image_id")
    )
    return hashlib.md5(raw.encode()).hexdigest()


def build_oracle_queries(item: Dict[str, Any]) -> Dict[str, str]:
    opts = item.get("options") or {}
    gold_letter = str(item.get("answer", ""))
    gold_text = str(opts.get(gold_letter, "") or item.get("answer_text", ""))
    src = item.get("source") or {}
    ctx = str(src.get("pdf_context_text") or "")[:200]
    return {
        "oracle_answer": f"{item.get('question', '')} {gold_text}".strip(),
        "oracle_context": ctx.strip(),
    }


class OptionLogprobScorer:
    """logprob 版 answer-utility 打分器（冻结生成器，进程内一次加载）。

    prompt 渲染复用 harness 的 render_generator_prompt（与后续训练 reward 同格式）；
    选项字母打分复用 v0.5 gen_scorer 的单 token logsumexp 方案。
    """

    def __init__(self, generator_cfg: Dict[str, Any]):
        import torch

        self.torch = torch
        self.device = generator_cfg.get("device", "cuda:0")
        self.model, self.processor = _load(
            generator_cfg.get("model_path") or generator_cfg.get("hf_id"), self.device
        )
        self.tok = self.processor.tokenizer
        self.enable_thinking = bool(generator_cfg.get("chat_template_kwargs", {}).get("enable_thinking", False))
        self._letter_cache: Dict[str, List[int]] = {}

    def _letter_ids(self, letter: str) -> List[int]:
        if letter in self._letter_cache:
            return self._letter_cache[letter]
        ids = set()
        for s in (letter, " " + letter):
            enc = self.tok.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(int(enc[0]))
        out = sorted(ids)
        self._letter_cache[letter] = out
        return out

    def judge(
        self,
        question: str,
        options: Dict[str, Any],
        gold_answer: str,
        image_path: str = "",
        evidence: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """一次前向 -> 四选项分布 + logP(gold)。evidence 只用 text（与训练一致，不喂证据图像）。"""
        torch = self.torch
        evidence = evidence or []
        letters = [str(k) for k in options.keys()]
        ev_short = [dict(e, text=str(e.get("text", ""))[:EVIDENCE_MAX_CHARS]) for e in evidence]
        prompt = render_generator_prompt(question, options, ev_short, image_attached=bool(image_path))
        messages = [{"role": "user", "content": prompt}]
        pil_imgs = []
        if image_path:
            from PIL import Image

            pil_imgs.append(Image.open(image_path).convert("RGB"))
            messages[-1] = {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking
        )
        inputs = self.processor(text=[chat], images=pil_imgs or None, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1, :].float()
        opt_logps = []
        for L in letters:
            ids = self._letter_ids(L)
            if ids:
                opt_logps.append(torch.logsumexp(logits[ids], dim=0))
            else:
                opt_logps.append(torch.tensor(-1e9, device=logits.device))
        stacked = torch.stack(opt_logps)
        logprobs = torch.log_softmax(stacked, dim=0)
        pred = letters[int(torch.argmax(stacked).item())]
        gold_lp = float(logprobs[letters.index(gold_answer)].item()) if gold_answer in letters else float("nan")
        return {"logp_gold": gold_lp, "pred": pred, "correct": pred == gold_answer}


def dedup_candidates(per_query_hits: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """两轮 oracle 候选合并去重（保留首次出现即最高分位置）。"""
    seen: Dict[str, Dict[str, Any]] = {}
    for _, hits in per_query_hits.items():
        for h in hits:
            k = candidate_key(h)
            if k and k not in seen:
                seen[k] = h
    return list(seen.values())


def check_one(retriever: LeaveOneOutRetriever, scorer: OptionLogprobScorer, item: Dict[str, Any]) -> Dict[str, Any]:
    q = str(item.get("question", ""))
    opts = item.get("options") or {}
    gold = str(item.get("answer", ""))
    img = str(item.get("query_image_path") or "")
    keys = gold_keys(item)

    per_query: Dict[str, List[Dict[str, Any]]] = {}
    for name, oq in build_oracle_queries(item).items():
        if not oq:
            per_query[name] = []
            continue
        out = retriever.retrieve(oq, img, **keys)
        per_query[name] = out["combined"]
    pool = dedup_candidates(per_query)

    base = scorer.judge(q, opts, gold, image_path=img, evidence=[])  # 空证据基线，每题一次
    rows = []
    for h in pool:
        sc = scorer.judge(q, opts, gold, image_path=img, evidence=[h])
        u = sc["logp_gold"] - base["logp_gold"]
        rows.append(
            {
                "u_i": round(u, 4),
                "label": "pos" if u > TAU_POS else ("neg" if u < TAU_NEG else "neutral"),
                "origin": h.get("source") or h.get("origin") or "",
                "score": round(float(h.get("score") or 0.0), 4),
                "sample_id": h.get("sample_id", ""),
                "doc_id": h.get("doc_id", ""),
                "from_oracle": "oracle_context" if any(h is x or candidate_key(h) == candidate_key(x) for x in per_query.get("oracle_context", [])) else "oracle_answer",
                "text": str(h.get("text", ""))[:120],
            }
        )
    pos_n = sum(1 for r in rows if r["label"] == "pos")
    neg_n = sum(1 for r in rows if r["label"] == "neg")
    return {
        "qid": item.get("qid", ""),
        "query_type": item.get("query_type", ""),
        "gold": gold,
        "base_logp_gold": round(base["logp_gold"], 4),
        "base_correct": base["correct"],
        "pool_n": len(pool),
        "pos_n": pos_n,
        "neg_n": neg_n,
        "supported": pos_n > 0,
        "harmful_present": neg_n > 0,
        "max_u": round(max((r["u_i"] for r in rows), default=float("nan")), 4),
        "pool_rows": rows,
        "n_oracle_answer": len(per_query.get("oracle_answer", [])),
        "n_oracle_context": len(per_query.get("oracle_context", [])),
    }


def summarize(records: List[Dict[str, Any]], args: argparse.Namespace, seconds: float) -> Dict[str, Any]:
    n = len(records)
    sup = sum(1 for r in records if r["supported"])
    harm = sum(1 for r in records if r["harmful_present"])
    by_type: Dict[str, Dict[str, Any]] = defaultdict(dict)
    buf: Dict[str, List[bool]] = defaultdict(list)
    for r in records:
        buf[r["query_type"]].append(r["supported"])
    for t, v in buf.items():
        by_type[t] = {"n": len(v), "supported": sum(v), "rate": round(sum(v) / len(v), 4)}
    pos_dist = Counter(r["pos_n"] for r in records)
    max_us = [r["max_u"] for r in records if r["pool_n"]]
    import statistics

    summary = {
        "n_questions": n,
        "supported_questions": sup,
        "support_rate": round(sup / n, 4) if n else None,
        "questions_with_harmful": harm,
        "thresholds": {"tau_pos": TAU_POS, "tau_neg": TAU_NEG},
        "by_query_type": dict(by_type),
        "pos_n_distribution": {str(k): v for k, v in sorted(pos_dist.items())},
        "max_u_stats": {
            "mean": round(statistics.mean(max_us), 4) if max_us else None,
            "median": round(statistics.median(max_us), 4) if max_us else None,
        },
        "avg_pool_n": round(sum(r["pool_n"] for r in records) / n, 2) if n else None,
        "base_correct_rate": round(sum(1 for r in records if r["base_correct"]) / n, 4) if n else None,
        "config": {
            "questions": args.questions,
            "sample_seed": args.seed,
            "text_k": args.text_k,
            "image_k": args.image_k,
            "overfetch_k": args.overfetch_k,
            "evidence_max_chars": EVIDENCE_MAX_CHARS,
            "generator_device": args.generator_device,
        },
        "elapsed_seconds": round(seconds, 1),
    }
    # 决策参考（方案第 8 节 P1 验收：比例决定源池是否够用）
    if n:
        rate = sup / n
        summary["decision_hint"] = (
            "proceed_p2（比例健康，按原计划进入 P2 题库重建）"
            if rate >= 0.6
            else ("relax_or_expand（比例偏低：考虑只排除自身样本或扩大源池）" if rate < 0.4 else "borderline（介于两者之间，需人工抽查正类样例后决定）")
        )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text-k", type=int, default=8)
    ap.add_argument("--image-k", type=int, default=4)
    ap.add_argument("--overfetch-k", type=int, default=40)
    ap.add_argument("--generator-device", default="cuda:0")
    ap.add_argument("--outdir", default="./p1_support_out")
    ap.add_argument("--limit-pool", type=int, default=20, help="每题最多打分的候选数（两轮 oracle 合并后截断）")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    items = load_questions(args.questions)
    picked = stratified_sample(items, args.n, args.seed)
    print(f"[p1] 抽样 {len(picked)} 题：{dict(Counter(p.get('query_type') for p in picked))}", flush=True)

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    retrieval_cfg = dict(cfg["retrieval"])
    retrieval_cfg["text_k"] = args.overfetch_k
    retrieval_cfg["image_k"] = args.overfetch_k
    gen_cfg = dict(cfg["generator"])
    gen_cfg["device"] = args.generator_device

    t0 = time.time()
    scorer = OptionLogprobScorer(gen_cfg)
    with LeaveOneOutRetriever(retrieval_cfg, text_k=args.text_k, image_k=args.image_k, overfetch_k=args.overfetch_k) as retriever:
        records: List[Dict[str, Any]] = []
        fp = open(outdir / "p1_support_samples.jsonl", "w", encoding="utf-8")
        try:
            for i, item in enumerate(picked, 1):
                try:
                    rec = check_one(retriever, scorer, item)
                    rec["pool_rows"] = rec["pool_rows"][: args.limit_pool]
                except Exception as e:
                    import traceback

                    rec = {"qid": item.get("qid", ""), "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc(limit=3)}
                records.append(rec)
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fp.flush()
                ok = [r for r in records if "supported" in r]
                print(
                    f"[p1] {i}/{len(picked)} elapsed={time.time()-t0:.0f}s "
                    f"支持率={sum(r['supported'] for r in ok)}/{len(ok)}",
                    flush=True,
                )
        finally:
            fp.close()

    ok_records = [r for r in records if "supported" in r]
    report = summarize(ok_records, args, time.time() - t0)
    report["n_errors"] = len(records) - len(ok_records)
    (outdir / "p1_support_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("support_rate", "supported_questions", "by_query_type", "decision_hint", "n_errors")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
