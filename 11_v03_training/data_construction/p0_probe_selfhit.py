#!/usr/bin/env python3
"""无排除对照实验：验证题图自命中的存在性与过滤键匹配。

背景：3 题冒烟中 overfetch k=40 未遇到题图自身（self=0），与旧数据
"top1 就是题图（score>0.999）"矛盾。本脚本不加排除直接检索，逐条打印
候选与题目 gold 键（sample_id / image_path / doc_id）的匹配情况，
确认：(a) 题图是否在索引里被召回；(b) 过滤键格式是否对得上。

用法：
  /mnt/data_1/yds/venvs/qwen35-train/bin/python p0_probe_selfhit.py \
      --config ../../10_harness/harness_config.json \
      --questions /mnt/data_1/yds/多模态/rerank_image_and_text/agentic/outputs/mcq_image_v2_4000/train.jsonl \
      --n 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leave_one_out_retriever import load_first_stage_retriever  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--k", type=int, default=40)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        retrieval_cfg = json.load(f)["retrieval"]
    retrieval_cfg["text_k"] = args.k
    retrieval_cfg["image_k"] = args.k

    retriever = load_first_stage_retriever(retrieval_cfg)
    try:
        with open(args.questions, encoding="utf-8") as f:
            for line in itertools.islice(f, args.n):
                item = json.loads(line)
                src = item.get("source") or {}
                qimg = str(item.get("query_image_path") or "")
                print("=" * 100)
                print(f"qid={item['qid']} gold_sample_id={src.get('sample_id')} gold_doc_id={src.get('doc_id')}")
                print(f"query_image_path={qimg}")
                # 图像路：不加任何排除
                hits = retriever.search_image(qimg, k=args.k)
                for h in hits[:5]:
                    same_img = "SAME_IMG" if str(h.get("image_path") or "") == qimg else ""
                    same_sid = "SAME_SID" if str(h.get("sample_id") or "") == str(src.get("sample_id")) else ""
                    print(
                        f"  img[{h.get('rank')}] score={float(h.get('score') or 0):.4f} "
                        f"sid={h.get('sample_id')} doc={h.get('doc_id')} {same_img}{same_sid}"
                    )
                n_self_img = sum(
                    1 for h in hits
                    if str(h.get("image_path") or "") == qimg or str(h.get("sample_id") or "") == str(src.get("sample_id"))
                )
                n_book_img = sum(1 for h in hits if str(h.get("doc_id") or "") == str(src.get("doc_id")))
                print(f"  => 图像路 top{args.k}: 自命中 {n_self_img} 条, 同书 {n_book_img} 条")
                # 文本路：题干+选项
                opts = item.get("options") or {}
                qtext = f"{item.get('question', '')} 选项：{'；'.join(f'{k}. {v}' for k, v in opts.items())}"
                thits = retriever.search_text(qtext, k=args.k)
                n_book_t = sum(1 for h in thits if str(h.get("doc_id") or "") == str(src.get("doc_id")))
                n_self_t = sum(1 for h in thits if str(h.get("sample_id") or "") == str(src.get("sample_id")))
                print(f"  => 文本路 top{args.k}: 自命中 {n_self_t} 条, 同书 {n_book_t} 条; top1 样例:")
                if thits:
                    h = thits[0]
                    print(f"    txt[1] score={float(h.get('score') or 0):.4f} sid={h.get('sample_id')} doc={h.get('doc_id')} text={str(h.get('text') or '')[:60]}")
    finally:
        retriever.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
