#!/usr/bin/env python3
"""诊断:检索到的图像证据是否会稀释/饱和掉文本证据的信号。

对每题的 4 个改写,分别用两种证据喂给冻结生成器,读正确答案概率:
  - 纯文本:  查询图 + 6 条文本
  - 图文混合:查询图 + 6 图 + 6 文
再看"组内 4 个改写之间的概率跨度(max-min)":
  若 图文混合 的跨度 << 纯文本 的跨度  => 图像把文本差异压平了(稀释/饱和)。
基线 p_base = 查询图(无检索证据)。
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_scorer import AnswerScorer  # noqa: E402

DEFAULT_GEN = "/mnt/data_10/mwx/huggingface_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def ev_text(passages):
    return [{"text": p.get("text", ""), "image_path": ""} for p in passages if p.get("origin") == "text"]


def ev_all(passages):
    img = [{"text": p.get("text", ""), "image_path": p.get("image_path", "")}
           for p in passages if p.get("origin") == "image"]
    return img + ev_text(passages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="train/rollouts_recalled_v1.jsonl")
    ap.add_argument("--gen-model", default=DEFAULT_GEN)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--normalize", default="options", choices=["options", "vocab"])
    args = ap.parse_args()

    sc = AnswerScorer(args.gen_model, device=args.device)
    print(f"[打分器] 就绪 {Path(args.gen_model).name} on {args.device} normalize={args.normalize}", flush=True)

    spread_txt, spread_all = [], []
    pall_level, ptxt_level, pbase_level = [], [], []
    n = 0
    for line in open(args.input):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        opts = o.get("options", {}) or {}
        gold = str(o.get("answer", "") or "")
        if not opts or not gold:
            continue
        qimg = o.get("query_image_path", "")
        rw = [r for r in o["group"] if r["action"] == "REWRITE" and r.get("recalled_passages")]
        if len(rw) < 2:
            continue
        q = o.get("question", "")
        p_base = sc.answer_prob(q, opts, gold, qimg, evidence=[], normalize=args.normalize)
        pt = [sc.answer_prob(q, opts, gold, qimg, ev_text(r["recalled_passages"]), normalize=args.normalize) for r in rw]
        pa = [sc.answer_prob(q, opts, gold, qimg, ev_all(r["recalled_passages"]), normalize=args.normalize) for r in rw]
        spread_txt.append(max(pt) - min(pt))
        spread_all.append(max(pa) - min(pa))
        ptxt_level.append(st.mean(pt)); pall_level.append(st.mean(pa)); pbase_level.append(p_base)
        n += 1
        print(f"[{o['qid']}] p_base={p_base:.3f} | 纯文本 p={[round(x,3) for x in pt]} 跨度={max(pt)-min(pt):.3f}"
              f" | 图文 p={[round(x,3) for x in pa]} 跨度={max(pa)-min(pa):.3f}", flush=True)
        if n >= args.limit:
            break

    print("\n==== 汇总(%d题) ====" % n)
    print("组内改写概率跨度(越大=改写越有区分度/梯度):")
    print("  纯文本证据:  均值 %.3f  中位 %.3f" % (st.mean(spread_txt), st.median(spread_txt)))
    print("  图文混合证据:均值 %.3f  中位 %.3f" % (st.mean(spread_all), st.median(spread_all)))
    diluted = sum(1 for a, b in zip(spread_all, spread_txt) if a < b - 1e-6)
    print("  图文跨度<纯文本跨度(被稀释)的题占比: %.2f" % (diluted / n))
    print("概率水平(看饱和): p_base均值 %.3f | 纯文本 %.3f | 图文 %.3f"
          % (st.mean(pbase_level), st.mean(ptxt_level), st.mean(pall_level)))


if __name__ == "__main__":
    main()
