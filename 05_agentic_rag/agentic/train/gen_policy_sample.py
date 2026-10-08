#!/usr/bin/env python3
"""Stage 3 -- tier-aware policy sampling (qwen35-train env), v2 group format.

For each question sample/build a GRPO group (one JSON per question, "group" key):
  - selection tier: N_self forced-ACCEPT rollouts (policy's own keep-sets over the
    original 12 candidates, dedup, NO post-hoc truncation -- the prompt itself caps
    keep at 3) + K=1 forced-REWRITE rollout + an anchor_none member (u=0).
  - rewrite tier:   K=8 forced-REWRITE rollouts (dedup) + anchor_original member
    (the original retrieval window, keep=all, scored per-passage downstream) +
    anchor_none.
  - trivial tier:   kept with probability --include-trivial-frac (deterministic by
    crc32(qid), shard-safe). Group = constructed keep-1 anchor + 1 forced-ACCEPT
    rollout + 1 forced-REWRITE rollout + anchor_none. Teaches "good enough -> ACCEPT".

NO retrieval, NO scoring here (later stages). Merges u/cor from the judge_tier file so
downstream can build priors without re-reading it. keep<=3 lives in the prompt
(context_agent.SYSTEM_PROMPT_V11), not in post-processing.

Output one JSON per question:
  {qid, tier, question, options, answer, answer_text, query_image_path, original_query,
   gold_source, obs_candidates(=round-0 + u/cor/flip),
   group:[{member_id, kind, action, keep, rewrite_query, raw_text}, ...]}
  kind in {select, rewrite, anchor_original, anchor_none}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

from ctrl_data_common import build_messages, render_chat  # noqa: E402
from grpo_reward import parse_action  # noqa: E402
from gen_policy_actions import (  # noqa: E402
    build_state_sample, round0_of, _first_json, _forced_rewrite_prefix,
)

DEFAULT_MODEL = str(HERE / "ckpt_qwen35_ctrl_full_v1")


def _forced_accept_prefix() -> str:
    """Assistant-turn prefix that commits to ACCEPT and opens the keep list."""
    return '{"keep": ['


def _include_trivial(qid: str, frac: float) -> bool:
    """Deterministic per-qid inclusion (shard-safe: stable across --start/--limit)."""
    if frac <= 0:
        return False
    if frac >= 1:
        return True
    return (zlib.crc32(str(qid).encode("utf-8")) % 10000) < frac * 10000


def load_tier_map(path: str) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            m[d.get("qid")] = d
    return m


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
    ap.add_argument("--obs", default="outputs/stage2_calibration/agent_context_v11_train3200_sep.jsonl")
    ap.add_argument("--tier", default="outputs/stage2_calibration/judge_tier_sep.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--policy-device", default="cuda:0")
    ap.add_argument("--n-self-select", type=int, default=3, help="self-selection rollouts (selection tier)")
    ap.add_argument("--include-trivial-frac", type=float, default=0.15,
                    help="fraction of trivial-tier questions to keep (crc32(qid) deterministic)")
    ap.add_argument("--k-rewrite-selection", type=int, default=1)
    ap.add_argument("--k-rewrite-rewrite", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.4)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--accept-max-new-tokens", type=int, default=192,
                    help="enough for keep+drop+action+reason to close (64 truncates -> parse fail)")
    ap.add_argument("--force-max-new-tokens", type=int, default=96)
    ap.add_argument("--max-images", type=int, default=8)
    ap.add_argument("--query-edge", type=int, default=768)
    ap.add_argument("--ev-edge", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    pdev = torch.device(args.policy_device)

    tier_map = load_tier_map(args.tier)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    eos_id = processor.tokenizer.eos_token_id
    policy = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(pdev)
    policy.eval()
    policy.config.use_cache = True
    print(f"[policy] loaded {Path(args.model).name} on {pdev}", flush=True)

    done = load_done_qids(args.out)
    trajs: List[Dict[str, Any]] = []
    with open(args.obs, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajs.append(json.loads(line))
    if args.start:
        trajs = trajs[args.start:]
    if args.limit:
        trajs = trajs[:args.limit]
    print(f"[data] pending {len(trajs)} (start={args.start} limit={args.limit}) done {len(done)}", flush=True)

    out_f = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    n_done = 0
    n_skip = 0
    for qi, traj in enumerate(trajs):
        qid = traj.get("qid", "")
        if qid in done:
            continue
        tinfo = tier_map.get(qid)
        tier = tinfo.get("tier") if tinfo else None
        if tier == "trivial":
            if not _include_trivial(qid, args.include_trivial_frac):
                n_skip += 1
                continue
        elif tier not in ("selection", "rewrite"):
            n_skip += 1
            continue

        rd0 = round0_of(traj)
        obs = rd0.get("candidates", [])
        if not obs:
            n_skip += 1
            continue
        # merge u/cor/flip from judge_tier
        tcands = {c["idx"]: c for c in (tinfo.get("candidates") or [])}
        for i, c in enumerate(obs):
            tc = tcands.get(i, {})
            c["u"] = tc.get("u")
            c["cor"] = tc.get("cor")
            c["flip"] = tc.get("flip")

        rd0_state = {"query": rd0.get("query", traj.get("original_query", "")), "candidates": obs}
        sample = build_state_sample(traj, rd0_state)
        n_cand = len(obs)

        msgs, images = build_messages(sample, max_images=args.max_images,
                                      query_edge=args.query_edge, ev_edge=args.ev_edge,
                                      include_target=False)
        prompt_text = render_chat(processor, msgs, add_generation_prompt=True)
        base_inputs = processor(text=[prompt_text], images=images if images else None,
                                return_tensors="pt").to(pdev)

        # forced-ACCEPT prefix (self-selection): commit to ACCEPT, open the keep list
        acc_prefix = _forced_accept_prefix()
        msgs_a, images_a = build_messages(sample, max_images=args.max_images,
                                          query_edge=args.query_edge, ev_edge=args.ev_edge,
                                          include_target=True, target_text=acc_prefix)
        # forced-REWRITE prefix
        rw_prefix = _forced_rewrite_prefix(n_cand)
        msgs_r, images_r = build_messages(sample, max_images=args.max_images,
                                          query_edge=args.query_edge, ev_edge=args.ev_edge,
                                          include_target=True, target_text=rw_prefix)

        def _forced_inputs(msgs_x, images_x):
            try:
                txt = processor.apply_chat_template(
                    msgs_x, tokenize=False, add_generation_prompt=False,
                    continue_final_message=True, enable_thinking=False)
            except TypeError:
                txt = processor.apply_chat_template(
                    msgs_x, tokenize=False, add_generation_prompt=False,
                    continue_final_message=True)
            inp = processor(text=[txt], images=images_x if images_x else None,
                            return_tensors="pt").to(pdev)
            return inp, int(inp["input_ids"].shape[1])

        acc_inputs, acc_len = _forced_inputs(msgs_a, images_a)
        rw_inputs, rw_len = _forced_inputs(msgs_r, images_r)

        acc_kwargs = dict(do_sample=True, temperature=args.temperature, top_p=args.top_p,
                          max_new_tokens=args.accept_max_new_tokens, pad_token_id=eos_id)
        rw_kwargs = dict(do_sample=True, temperature=args.temperature, top_p=args.top_p,
                         max_new_tokens=args.force_max_new_tokens, pad_token_id=eos_id)

        self_selects: List[Dict[str, Any]] = []
        rewrites: List[Dict[str, Any]] = []
        seen_keep = set()
        seen_rw = set()
        if tier == "selection":
            k_rw = args.k_rewrite_selection
            n_self = args.n_self_select
        elif tier == "rewrite":
            k_rw = args.k_rewrite_rewrite
            n_self = 0
        else:  # trivial: 1 self-select + 1 rewrite teach "good enough -> ACCEPT"
            k_rw = 1
            n_self = 1

        with torch.no_grad():
            for _ in range(n_self):
                seq = policy.generate(**acc_inputs, **acc_kwargs)
                cont = processor.tokenizer.decode(seq[0, acc_len:], skip_special_tokens=True)
                action = parse_action(_first_json(acc_prefix + cont), n_cand)
                # no post-hoc truncation: keep<=3 is enforced by the prompt
                keep = sorted(set(int(x) for x in action["keep"] if 0 <= int(x) < n_cand))
                if not action["parsed"]:
                    # truncated JSON: recover the keep list leniently (it sits right after the prefix)
                    m = re.search(r'"keep"\s*:\s*\[([^\]]*)\]', acc_prefix + cont)
                    if not m:
                        continue
                    keep = sorted(set(int(x) for x in re.findall(r"\d+", m.group(1))
                                      if 0 <= int(x) < n_cand))
                key = tuple(keep)
                if key in seen_keep:
                    continue
                seen_keep.add(key)
                self_selects.append({"keep": keep, "raw_text": acc_prefix + cont})
            for _ in range(k_rw):
                seq = policy.generate(**rw_inputs, **rw_kwargs)
                cont = processor.tokenizer.decode(seq[0, rw_len:], skip_special_tokens=True)
                rq = cont.split('"')[0].strip()
                if not rq or rq in seen_rw:
                    continue
                seen_rw.add(rq)
                rewrites.append({"idx": len(rewrites), "rewrite_query": rq,
                                 "raw_text": rw_prefix + cont})

        # ---- assemble the GRPO group (member kinds drive Stage D scoring) ----
        group: List[Dict[str, Any]] = []
        if tier == "rewrite":
            # original retrieval window as the no-rewrite anchor (per-passage scoring)
            group.append({
                "member_id": "anchor_orig", "kind": "anchor_original",
                "action": "ACCEPT", "keep": list(range(n_cand)),
                "rewrite_query": "", "raw_text": "",
            })
        if tier == "trivial":
            dup_keep1 = any(tuple(ss["keep"]) == (0,) for ss in self_selects)
            if not dup_keep1 and n_cand:
                group.append({
                    "member_id": "keep1", "kind": "select",
                    "action": "ACCEPT", "keep": [0],
                    "rewrite_query": "", "raw_text": "",
                })
        for i, ss in enumerate(self_selects):
            group.append({
                "member_id": f"self{i}", "kind": "select",
                "action": "ACCEPT", "keep": ss["keep"],
                "rewrite_query": "", "raw_text": ss["raw_text"],
            })
        for rw in rewrites:
            group.append({
                "member_id": f"rw{rw['idx']}", "kind": "rewrite",
                "action": "REWRITE", "keep": [],
                "rewrite_query": rw["rewrite_query"], "raw_text": rw["raw_text"],
            })
        group.append({
            "member_id": "anchor_none", "kind": "anchor_none",
            "action": "ACCEPT", "keep": [], "rewrite_query": "", "raw_text": "",
        })

        out = {
            "qid": qid, "tier": tier, "query_type": traj.get("query_type", ""),
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
        if args.log_every and n_done % args.log_every == 0:
            dt = time.time() - t0
            print(f"[{n_done}] {qid} tier={tier} self={len(self_selects)} rw={len(rewrites)} "
                  f"grp={len(group)} | {dt / max(n_done, 1):.1f}s/q", flush=True)

    out_f.close()
    print(f"[done] {n_done} -> {args.out} (skipped {n_skip})", flush=True)


if __name__ == "__main__":
    main()
