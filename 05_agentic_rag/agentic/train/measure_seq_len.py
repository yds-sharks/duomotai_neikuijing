#!/usr/bin/env python3
"""Measure the exact full-sequence token length of each SFT sample.

Uses the SAME processor + image settings as train_ctrl_sft_full.py (build_messages
with max_images/query_edge/ev_edge, include_target=True) so the measured length is
exactly what the trainer feeds the model (text + vision tokens). CPU-only; does not
touch the training GPUs. Writes per-sample lengths + prints a discard table so we
can pick a max-len threshold that DROPS (not truncates) oversized samples.

Usage:
  python train/measure_seq_len.py --input train/sft_ctrl_train.jsonl \
    --out train/seq_len_train.jsonl --max-images 8 --query-edge 768 --ev-edge 384
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctrl_data_common import build_messages, render_chat  # noqa: E402

DEFAULT_MODEL = "/mnt/data_1/yds/models/hf_hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-images", type=int, default=8)
    ap.add_argument("--query-edge", type=int, default=768)
    ap.add_argument("--ev-edge", type=int, default=384)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    lens = []
    fout = open(args.out, "w", encoding="utf-8")
    for i, s in enumerate(rows):
        msgs, images = build_messages(s, max_images=args.max_images,
                                      query_edge=args.query_edge, ev_edge=args.ev_edge,
                                      include_target=True)
        text = render_chat(proc, msgs, add_generation_prompt=False)
        enc = proc(text=[text], images=images if images else None, return_tensors="pt")
        n = int(enc["input_ids"].shape[1])
        n_img = len(images)
        lens.append(n)
        fout.write(json.dumps({"qid": s.get("qid", ""), "round_idx": s.get("round_idx", 0),
                               "n_tokens": n, "n_images": n_img}, ensure_ascii=False) + "\n")
        if (i + 1) % args.log_every == 0:
            fout.flush()
            print(f"[{i+1}/{len(rows)}] running... last={n}", flush=True)
        del enc
    fout.close()

    lens.sort()
    n = len(lens)
    def pct(p): return lens[min(int(n * p), n - 1)]
    print("=== 完整序列 token 长度分布(文本+图像, 与训练一致) ===", flush=True)
    print(f"样本={n} 均值={sum(lens)/n:.0f} 中位={pct(0.5)} p90={pct(0.9)} "
          f"p95={pct(0.95)} p99={pct(0.99)} 最大={lens[-1]}", flush=True)
    print("=== 丢弃表(阈值 -> 保留/丢弃) ===", flush=True)
    for thr in [3072, 3584, 4096, 4608, 5120, 5632]:
        drop = sum(1 for x in lens if x > thr)
        print(f"  max_len={thr}: 保留 {n-drop} ({(n-drop)/n*100:.1f}%)  丢弃 {drop} ({drop/n*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
