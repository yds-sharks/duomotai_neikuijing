#!/usr/bin/env python3
"""Stage C:改写后的第二次 keep/drop(多步 agent 的第 2 步)。

Stage A 只捕获了 round-0 决策;对 REWRITE 分支,改写会检索到一批新的图文对候选
(Stage B 产出 recalled_passages)。本阶段让**同一策略**观测这批新候选,再做一次
keep/drop 挑选 —— 这才是"agent 负责挑选"的完整闭环。最终喂给生成器/算 reward 的
证据 = 策略选中的子集,而不是 top-k 全量。

  - ACCEPT 分支:无需第二步,selected = round-0 keep 应用到 obs_candidates。
  - REWRITE 分支:观测新候选 -> **贪心** keep/drop 挑选(每个改写一个确定性选择,
    保证组内差异来自改写本身而非挑选噪声) -> selected = 选中的新候选子集。

env: qwen35-train(与 Stage A 同策略)。可分片(--start/--limit)+断点续(按 qid)。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import sys

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

from ctrl_data_common import build_messages, render_chat  # noqa: E402
from grpo_reward import parse_action  # noqa: E402
from gen_policy_actions import build_state_sample, _first_json  # noqa: E402

DEFAULT_MODEL = str(HERE / "ckpt_qwen35_ctrl_full_v1")


def _apply_keep(cands: List[Dict[str, Any]], keep: List[int]) -> List[Dict[str, Any]]:
    return [cands[i] for i in keep if 0 <= i < len(cands)]


def load_done_qids(out_path: str) -> set:
    done: set = set()
    p = Path(out_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line).get("qid"))
                    except Exception:
                        pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Stage B output (rollouts_recalled_v2*.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--policy-device", default="cuda:0")
    ap.add_argument("--do-sample", action="store_true", help="sample instead of greedy (default: greedy)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--max-images", type=int, default=8)
    ap.add_argument("--query-edge", type=int, default=768)
    ap.add_argument("--ev-edge", type=int, default=384)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    pdev = torch.device(args.policy_device)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    eos_id = processor.tokenizer.eos_token_id
    policy = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(pdev)
    policy.eval()
    policy.config.use_cache = True
    print(f"[策略] 已加载 {Path(args.model).name} 于 {pdev} (round-1 挑选)", flush=True)

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
    done = load_done_qids(args.out)
    print(f"[数据] 待处理 {len(rows)} 题(start={args.start} limit={args.limit}) 已完成 {len(done)}", flush=True)

    gen_kwargs = dict(do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p,
                      max_new_tokens=args.max_new_tokens, pad_token_id=eos_id)

    out_f = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    n = 0
    n_sel = 0
    for row in rows:
        qid = row.get("qid")
        if qid in done:
            continue
        obs = row.get("obs_candidates", [])
        for r in row.get("group", []):
            action = r.get("action")
            if action == "REWRITE" and r.get("recalled_passages"):
                cands = r["recalled_passages"]
                state = {"query": r.get("rewrite_query", ""), "candidates": cands}
                sample = build_state_sample(row, state)
                msgs, images = build_messages(sample, max_images=args.max_images,
                                              query_edge=args.query_edge, ev_edge=args.ev_edge,
                                              include_target=False)
                prompt_text = render_chat(processor, msgs, add_generation_prompt=True)
                inp = processor(text=[prompt_text], images=images if images else None,
                                return_tensors="pt").to(pdev)
                plen = int(inp["input_ids"].shape[1])
                with torch.no_grad():
                    seq = policy.generate(**inp, **gen_kwargs)
                text = processor.tokenizer.decode(seq[0, plen:], skip_special_tokens=True)
                act1 = parse_action(_first_json(text), len(cands))
                r["round1_action"] = act1["action"]
                r["round1_keep"] = act1["keep"]
                r["round1_parsed"] = act1["parsed"]
                r["round1_raw_text"] = text
                r["selected_passages"] = _apply_keep(cands, act1["keep"])
                n_sel += 1
            elif action == "ACCEPT":
                # round-0 keep 应用到 obs 图文对
                r["selected_passages"] = _apply_keep(obs, r.get("keep", []))
            else:
                r["selected_passages"] = []
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()
        n += 1
        if n % args.log_every == 0:
            dt = time.time() - t0
            print(f"[{n}/{len(rows)}] {dt / n:.2f}s/题 round1挑选次数={n_sel}", flush=True)

    out_f.close()
    print(f"[完成] 写出 {n} 题 -> {args.out}(round-1 挑选 {n_sel} 次)", flush=True)


if __name__ == "__main__":
    main()
