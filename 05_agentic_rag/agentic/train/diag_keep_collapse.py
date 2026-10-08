#!/usr/bin/env python3
"""Diag: is the all-empty keep caused by the forced-ACCEPT prefix or by the new prompt?

Runs, on a few selection-tier questions:
  a) free sampling (no prefix, new prompt) x N
  b) forced-ACCEPT prefix sampling x N
and prints the keep sets of each. env: qwen35-train.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

from ctrl_data_common import build_messages, render_chat  # noqa: E402
from grpo_reward import parse_action  # noqa: E402
from gen_policy_actions import build_state_sample, round0_of, _first_json  # noqa: E402
from gen_policy_sample import _forced_accept_prefix, load_tier_map  # noqa: E402

DEFAULT_MODEL = str(HERE / "ckpt_qwen35_ctrl_full_v1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs", default="outputs/stage2_calibration/agent_context_v11_train3200_sep.jsonl")
    ap.add_argument("--tier", default="outputs/stage2_calibration/judge_tier_sep.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-q", type=int, default=6)
    ap.add_argument("--n-sample", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.4)
    args = ap.parse_args()

    tier_map = load_tier_map(args.tier)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    eos_id = processor.tokenizer.eos_token_id
    policy = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(torch.device(args.device))
    policy.eval()
    policy.config.use_cache = True

    trajs = [json.loads(l) for l in open(args.obs) if l.strip()]
    picked = [t for t in trajs if (tier_map.get(t.get("qid")) or {}).get("tier") == "selection"][:args.n_q]

    acc_prefix = _forced_accept_prefix()
    for traj in picked:
        qid = traj["qid"]
        rd0 = round0_of(traj)
        obs = rd0.get("candidates", [])
        n_cand = len(obs)
        sample = build_state_sample(traj, {"query": rd0.get("query", ""), "candidates": obs})

        msgs, images = build_messages(sample, include_target=False)
        prompt_text = render_chat(processor, msgs, add_generation_prompt=True)
        base_in = processor(text=[prompt_text], images=images if images else None,
                            return_tensors="pt").to(args.device)
        blen = int(base_in["input_ids"].shape[1])

        msgs_a, images_a = build_messages(sample, include_target=True, target_text=acc_prefix)
        try:
            atxt = processor.apply_chat_template(msgs_a, tokenize=False, add_generation_prompt=False,
                                                 continue_final_message=True, enable_thinking=False)
        except TypeError:
            atxt = processor.apply_chat_template(msgs_a, tokenize=False, add_generation_prompt=False,
                                                 continue_final_message=True)
        acc_in = processor(text=[atxt], images=images_a if images_a else None,
                           return_tensors="pt").to(args.device)
        alen = int(acc_in["input_ids"].shape[1])

        kw = dict(do_sample=True, temperature=args.temperature, top_p=0.95,
                  max_new_tokens=192, pad_token_id=eos_id)
        free_keeps, forced_keeps = [], []
        with torch.no_grad():
            for _ in range(args.n_sample):
                seq = policy.generate(**base_in, **kw)
                txt = processor.tokenizer.decode(seq[0, blen:], skip_special_tokens=True)
                a = parse_action(_first_json(txt), n_cand)
                free_keeps.append((a["keep"], a["action"]))
            for _ in range(args.n_sample):
                seq = policy.generate(**acc_in, **kw)
                cont = processor.tokenizer.decode(seq[0, alen:], skip_special_tokens=True)
                a = parse_action(_first_json(acc_prefix + cont), n_cand)
                forced_keeps.append((a["keep"], a["action"]))
        print(f"{qid} n_cand={n_cand}")
        print(f"  free  : {free_keeps}")
        print(f"  forced: {forced_keeps}")


if __name__ == "__main__":
    main()
