#!/usr/bin/env python3
"""Stage A -- policy action sampling for real-augmentation GRPO (qwen35-train env).

For each question, the observation is the recorded round-0 initial-retrieval window
(NOT any teacher rewrite). The SFT policy samples G high-temperature actions; each
action is ACCEPT (evidence = round-0[keep]) or REWRITE(query). We only record the
decisions here -- NO retrieval, NO scoring (both are later stages).

Output: one JSON per question:
  {qid, question, options, answer, answer_text, query_image_path, original_query,
   gold_source, obs_candidates(=round-0), group:[{idx, action, keep, rewrite_query,
   parsed, raw_text}, ...]}

Run (policy-only; shardable across GPUs with --start/--limit):
  CUDA_VISIBLE_DEVICES=0 python train/gen_policy_actions.py \
    --input outputs/stage2_calibration/agent_context_v11_train3200.jsonl \
    --out train/policy_actions_v1.jsonl \
    --model train/ckpt_qwen35_ctrl_full_v1 \
    --policy-device cuda:0 --group-size 6 --temperature 1.4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

from ctrl_data_common import build_messages, render_chat  # noqa: E402
from context_agent import SYSTEM_PROMPT_V11, USER_TEMPLATE, format_candidates  # noqa: E402
from grpo_reward import parse_action  # noqa: E402

DEFAULT_MODEL = str(HERE / "ckpt_qwen35_ctrl_full_v1")


def _first_json(text: str) -> str:
    """Return the first balanced {...} object (guards against post-JSON rambling)."""
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def _forced_rewrite_prefix(n_cand: int) -> str:
    """Assistant-turn prefix that commits to REWRITE and opens rewrite_query.

    keep=[]/drop=all is the natural semantics of a rewrite (discard current window,
    re-retrieve). The model only samples the query text that follows.
    """
    drop = ", ".join(str(i) for i in range(n_cand))
    return '{"keep": [], "drop": [' + drop + '], "action": "REWRITE", "rewrite_query": "'



def build_state_sample(traj: Dict[str, Any], rd: Dict[str, Any]) -> Dict[str, Any]:
    """Identical prompt construction to SFT/GRPO (train_ctrl_grpo.build_state_sample)."""
    cands = rd.get("candidates", [])
    user_text = USER_TEMPLATE.format(
        qid=traj.get("qid", ""), query_type=traj.get("query_type", ""),
        original_query=traj.get("original_query", ""), current_query=rd.get("query", ""),
        question=traj.get("question", ""),
        options=json.dumps(traj.get("options", {}), ensure_ascii=False),
        candidate_block=format_candidates(cands),
    )
    return {
        "system": SYSTEM_PROMPT_V11,
        "user_text": user_text,
        "query_image_path": traj.get("query_image_path", ""),
        "evidence_image_paths": [str(c.get("image_path") or "") for c in cands],
    }


def round0_of(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Return the round-0 record (initial retrieval, no breadcrumb) or {}."""
    for rd in traj.get("rounds", []):
        if int(rd.get("round_idx", 0)) == 0 and rd.get("candidates"):
            return rd
    rounds = traj.get("rounds", [])
    return rounds[0] if rounds and rounds[0].get("candidates") else {}


def load_done_qids(out_path: str) -> set:
    done: set = set()
    p = Path(out_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line).get("qid"))
                except Exception:
                    pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--policy-device", default="cuda:0")
    ap.add_argument("--n-natural", type=int, default=2, help="unconstrained rollouts per question (ACCEPT anchors)")
    ap.add_argument("--n-force-rewrite", type=int, default=4, help="forced-REWRITE rollouts per question (prefix-conditioned)")
    ap.add_argument("--temperature", type=float, default=1.4)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=200, help="cap for natural rollouts")
    ap.add_argument("--force-max-new-tokens", type=int, default=96, help="cap for forced-rewrite query completion")
    ap.add_argument("--max-images", type=int, default=8)
    ap.add_argument("--query-edge", type=int, default=768)
    ap.add_argument("--ev-edge", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
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
    print(f"[策略] 已加载 {Path(args.model).name} 于 {pdev}", flush=True)
    print(f"[采样] 每题 自然={args.n_natural} 强制改写={args.n_force_rewrite} "
          f"温度={args.temperature} top_p={args.top_p}", flush=True)

    done = load_done_qids(args.out)
    print(f"[断点] 已完成 {len(done)} 题,将跳过", flush=True)

    trajs: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajs.append(json.loads(line))
    if args.start:
        trajs = trajs[args.start:]
    if args.limit:
        trajs = trajs[:args.limit]
    print(f"[数据] 待处理 {len(trajs)} 题(start={args.start} limit={args.limit})", flush=True)

    out_f = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    n_done = 0
    n_skip_noobs = 0
    for qi, traj in enumerate(trajs):
        qid = traj.get("qid", "")
        if qid in done:
            continue
        rd0 = round0_of(traj)
        obs = rd0.get("candidates", [])
        if not obs:
            n_skip_noobs += 1
            continue
        rd0_state = {"query": rd0.get("query", traj.get("original_query", "")), "candidates": obs}
        sample = build_state_sample(traj, rd0_state)
        n_cand = len(obs)

        msgs, images = build_messages(sample, max_images=args.max_images,
                                      query_edge=args.query_edge, ev_edge=args.ev_edge,
                                      include_target=False)
        prompt_text = render_chat(processor, msgs, add_generation_prompt=True)
        base_inputs = processor(text=[prompt_text], images=images if images else None,
                                return_tensors="pt").to(pdev)
        prompt_len = int(base_inputs["input_ids"].shape[1])

        # forced-REWRITE prefix as a to-be-CONTINUED assistant turn (fixes action=REWRITE,
        # opens rewrite_query"); re-encode in one shot so multimodal tensors stay aligned.
        prefix_text = _forced_rewrite_prefix(n_cand)
        msgs_f, images_f = build_messages(sample, max_images=args.max_images,
                                          query_edge=args.query_edge, ev_edge=args.ev_edge,
                                          include_target=True, target_text=prefix_text)
        try:
            forced_text = processor.apply_chat_template(
                msgs_f, tokenize=False, add_generation_prompt=False,
                continue_final_message=True, enable_thinking=False)
        except TypeError:
            forced_text = processor.apply_chat_template(
                msgs_f, tokenize=False, add_generation_prompt=False,
                continue_final_message=True)
        forced_inputs = processor(text=[forced_text], images=images_f if images_f else None,
                                  return_tensors="pt").to(pdev)
        forced_len = int(forced_inputs["input_ids"].shape[1])

        nat_kwargs = dict(do_sample=True, temperature=args.temperature, top_p=args.top_p,
                          max_new_tokens=args.max_new_tokens, pad_token_id=eos_id)
        force_kwargs = dict(do_sample=True, temperature=args.temperature, top_p=args.top_p,
                            max_new_tokens=args.force_max_new_tokens, pad_token_id=eos_id)

        group: List[Dict[str, Any]] = []
        n_rw = 0
        n_ac = 0
        idx = 0
        with torch.no_grad():
            # natural rollouts (unconstrained) -> ACCEPT anchors (usually)
            for _ in range(args.n_natural):
                seq = policy.generate(**base_inputs, **nat_kwargs)
                text = processor.tokenizer.decode(seq[0, prompt_len:], skip_special_tokens=True)
                action = parse_action(_first_json(text), n_cand)
                is_rw = action["action"] == "REWRITE" and action["rewrite_query"].strip()
                group.append({
                    "idx": idx, "forced": False, "action": action["action"],
                    "keep": action["keep"], "rewrite_query": action["rewrite_query"].strip(),
                    "parsed": action["parsed"], "raw_text": text,
                })
                n_rw += int(bool(is_rw))
                n_ac += int(not is_rw)
                idx += 1
            # forced-REWRITE rollouts -> diverse rewrite queries
            for _ in range(args.n_force_rewrite):
                seq = policy.generate(**forced_inputs, **force_kwargs)
                cont = processor.tokenizer.decode(seq[0, forced_len:], skip_special_tokens=True)
                # rewrite_query = text up to the closing quote (Chinese queries rarely contain ")
                rq = cont.split('"')[0].strip()
                group.append({
                    "idx": idx, "forced": True, "action": "REWRITE",
                    "keep": [], "rewrite_query": rq,
                    "parsed": bool(rq), "raw_text": prefix_text + cont,
                })
                n_rw += int(bool(rq))
                idx += 1

        uniq_rw = len({r["rewrite_query"] for r in group if r["action"] == "REWRITE" and r["rewrite_query"]})
        out = {
            "qid": qid, "query_type": traj.get("query_type", ""),
            "question": traj.get("question", ""), "options": traj.get("options", {}),
            "answer": traj.get("answer", ""), "answer_text": traj.get("answer_text", ""),
            "query_image_path": traj.get("query_image_path", ""),
            "original_query": traj.get("original_query", ""),
            "gold_source": traj.get("gold_source", {}),
            "obs_candidates": obs,
            "group": group,
        }
        out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
        out_f.flush()
        n_done += 1
        if args.log_every and qi % args.log_every == 0:
            dt = time.time() - t0
            print(f"[题{qi} {qid}] 候选={n_cand} 改写={n_rw} 接受={n_ac} 去重改写={uniq_rw} "
                  f"| {dt / max(n_done, 1):.1f}s/题", flush=True)

    out_f.close()
    print(f"[完成] 写出 {n_done} 题 -> {args.out}(无观测跳过 {n_skip_noobs})", flush=True)


if __name__ == "__main__":
    main()
