#!/usr/bin/env python3
"""Evaluate controller checkpoints on the val set: keep/drop decision hit-rate.

For each val sample we build the SAME prompt the trainer used (system v11 + user_text
+ query image + evidence images), greedily generate the controller JSON, parse the
keep/drop + ACCEPT/REWRITE action, and compare against the teacher `target`.

Metrics per checkpoint:
  - parse_ok       : fraction of outputs that are valid controller JSON
  - keep_exact     : fraction where predicted keep-set == teacher keep-set
  - keep P/R/F1    : per-candidate binary (label = "kept"), micro-averaged
  - action_acc     : ACCEPT/REWRITE match rate

Supports multiple checkpoints (evaluated sequentially, model freed between) and
prints a comparison table + a recommended pick (by keep-F1, tie-break keep_exact).

Usage:
  python train/eval_ctrl_ckpt.py \
    --ckpts train/ckpt_qwen35_ctrl_full_v1_u101,train/ckpt_qwen35_ctrl_full_v1_u151,train/ckpt_qwen35_ctrl_full_v1 \
    --val train/sft_ctrl_val.jsonl --val-len-file train/seq_len_val.jsonl \
    --max-len 4608 --device cuda:0 --out-dir train/eval_ctrl
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctrl_data_common import build_messages, render_chat  # noqa: E402
from grpo_reward import parse_action  # noqa: E402


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def filter_by_len(rows: List[Dict[str, Any]], len_file: str, max_len: int) -> List[Dict[str, Any]]:
    if not (len_file and max_len > 0):
        return rows
    lens = [int(json.loads(l)["n_tokens"]) for l in open(len_file, encoding="utf-8") if l.strip()]
    if len(lens) != len(rows):
        raise ValueError(f"len_file rows {len(lens)} != data rows {len(rows)}")
    return [r for r, n in zip(rows, lens) if n <= max_len]


def teacher_action(sample: Dict[str, Any], n: int) -> Dict[str, Any]:
    tgt = sample.get("target", "")
    return parse_action(tgt if isinstance(tgt, str) else json.dumps(tgt), n)


def eval_ckpt(ckpt: str, rows: List[Dict[str, Any]], *, model_cls, processor_cls, torch,
              device: str, max_images: int, query_edge: int, ev_edge: int,
              max_new_tokens: int, out_path: Path) -> Dict[str, Any]:
    processor = processor_cls.from_pretrained(ckpt, trust_remote_code=True)
    model = model_cls.from_pretrained(ckpt, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                      trust_remote_code=True).to(device)
    model.eval()
    model.config.use_cache = True
    eos = processor.tokenizer.eos_token_id

    n_parse = n_exact = n_action = 0
    tp = fp = fn = 0
    fout = open(out_path, "w", encoding="utf-8")
    for s in rows:
        n = int(s.get("n_candidates", 0) or 0)
        tea = teacher_action(s, n)
        msgs, images = build_messages(s, max_images=max_images, query_edge=query_edge,
                                      ev_edge=ev_edge, include_target=False)
        text = render_chat(processor, msgs, add_generation_prompt=True)
        enc = processor(text=[text], images=images if images else None, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                 pad_token_id=eos)
        new = gen[0][enc["input_ids"].shape[1]:]
        pred_text = processor.tokenizer.decode(new, skip_special_tokens=True)
        pred = parse_action(pred_text, n)

        pk, tk = set(pred["keep"]), set(tea["keep"])
        n_parse += int(pred["parsed"])
        n_exact += int(pk == tk)
        n_action += int(pred["action"] == tea["action"])
        tp += len(pk & tk); fp += len(pk - tk); fn += len(tk - pk)
        fout.write(json.dumps({"qid": s.get("qid"), "round_idx": s.get("round_idx"),
                               "teacher_keep": sorted(tk), "pred_keep": sorted(pk),
                               "teacher_action": tea["action"], "pred_action": pred["action"],
                               "parsed": pred["parsed"]}, ensure_ascii=False) + "\n")
    fout.close()

    m = len(rows)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"ckpt": ckpt, "n": m, "parse_ok": n_parse / m, "keep_exact": n_exact / m,
            "keep_prec": prec, "keep_rec": rec, "keep_f1": f1, "action_acc": n_action / m}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", required=True, help="comma-separated checkpoint dirs")
    ap.add_argument("--val", required=True)
    ap.add_argument("--val-len-file", default="")
    ap.add_argument("--max-len", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=8)
    ap.add_argument("--query-edge", type=int, default=768)
    ap.add_argument("--ev-edge", type=int, default=384)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="train/eval_ctrl")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = filter_by_len(read_jsonl(args.val), args.val_len_file, args.max_len)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] val samples={len(rows)} (max_len={args.max_len})", flush=True)

    results = []
    for ckpt in [c.strip() for c in args.ckpts.split(",") if c.strip()]:
        if not Path(ckpt).exists():
            print(f"[skip] not found: {ckpt}", flush=True)
            continue
        tag = Path(ckpt).name
        print(f"\n===== evaluating {tag} =====", flush=True)
        r = eval_ckpt(ckpt, rows, model_cls=AutoModelForImageTextToText,
                      processor_cls=AutoProcessor, torch=torch, device=args.device,
                      max_images=args.max_images, query_edge=args.query_edge,
                      ev_edge=args.ev_edge, max_new_tokens=args.max_new_tokens,
                      out_path=out_dir / f"pred_{tag}.jsonl")
        results.append(r)
        print(f"[{tag}] parse_ok={r['parse_ok']:.3f} keep_exact={r['keep_exact']:.3f} "
              f"keep_P={r['keep_prec']:.3f} keep_R={r['keep_rec']:.3f} keep_F1={r['keep_f1']:.3f} "
              f"action_acc={r['action_acc']:.3f}", flush=True)

    if results:
        print("\n===== comparison =====", flush=True)
        print(f"{'ckpt':<40}{'parse':>7}{'exact':>7}{'keepP':>7}{'keepR':>7}{'keepF1':>8}{'act':>7}", flush=True)
        for r in results:
            print(f"{Path(r['ckpt']).name:<40}{r['parse_ok']:>7.3f}{r['keep_exact']:>7.3f}"
                  f"{r['keep_prec']:>7.3f}{r['keep_rec']:>7.3f}{r['keep_f1']:>8.3f}{r['action_acc']:>7.3f}", flush=True)
        best = max(results, key=lambda r: (r["keep_f1"], r["keep_exact"]))
        print(f"\n[recommended] {Path(best['ckpt']).name}  (keep_F1={best['keep_f1']:.3f}, "
              f"keep_exact={best['keep_exact']:.3f}) -> GRPO init", flush=True)
        (out_dir / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[save] {out_dir/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
