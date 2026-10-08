#!/usr/bin/env python3
"""GRPO for the Qwen3.5-4B evidence controller (single-GPU runnable smoke first).

Pipeline per state (a recorded retrieval round):
  1. Build the same faithful multimodal prompt used at SFT time.
  2. Sample G controller decisions from the *policy* at high temperature
     (--n-forced-rewrite of them behind a forced-REWRITE prefix for exploration;
     prefix tokens are excluded from PG/KL so gradients only touch the free
     continuation).
  3. Reward each rollout with the UNIFIED answer-utility reward
     u = Δ logP_options(gold) vs the no-evidence baseline (log-domain, no
     saturation; same metric as judge_tier / Stage D). Only HARD leakage
     ("答案是X"/"最终答案") is penalized; answer_text hits are monitor-only.
  4. Policy-gradient: loss = mean_i[ -A_i * mean_token_logp(rollout_i) ]
     + beta_kl * KL(policy || frozen_ref)   (k3 estimator, per-token).

Tier-aware input filtering: --tier-file judge_tier_sep.jsonl keeps selection/
rewrite tiers and subsamples trivial by --include-trivial-frac (crc32-stable).

Reference = the SFT checkpoint (default = --model). Start from the SFT model so
sampling explores sensible keep/drop decisions; high temperature exposes both
higher- and lower-reward actions so the group baseline yields gradient signal.

Smoke (1 GPU, base model as stand-in until SFT ckpt exists):
  CUDA_VISIBLE_DEVICES=0 python train/train_ctrl_grpo.py \
    --input outputs/stage2_calibration/agent_context_v11_train3200.jsonl \
    --out-dir train/ckpt_grpo_smoke --limit 4 --group-size 4 --max-new-tokens 128 \
    --smoke

Full (after SFT):
  CUDA_VISIBLE_DEVICES=0 python train/train_ctrl_grpo.py \
    --input outputs/stage2_calibration/agent_context_v11_train3200.jsonl \
    --model train/ckpt_qwen35_ctrl_full_v1 --out-dir train/ckpt_qwen35_ctrl_grpo_v1 \
    --group-size 8 --temperature 1.1 --top-p 0.95 --lr 1e-6 --beta-kl 0.02
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.optimization import Adafactor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from ctrl_data_common import build_messages, render_chat  # noqa: E402
from context_agent import SYSTEM_PROMPT_V11, USER_TEMPLATE, format_candidates  # noqa: E402
from grpo_reward import action_reward, group_advantages, parse_action  # noqa: E402
from gen_policy_actions import _forced_rewrite_prefix  # noqa: E402
from reward_model import hard_leakage, leakage_penalty  # noqa: E402

DEFAULT_GEN_MODEL = "/mnt/data_10/mwx/huggingface_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

DEFAULT_MODEL = "/mnt/data_1/yds/models/hf_hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def _include_trivial(qid: str, frac: float) -> bool:
    """Deterministic per-qid inclusion (same rule as gen_policy_sample)."""
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
            if line:
                d = json.loads(line)
                m[d.get("qid")] = d
    return m


def read_states(path: str, limit: int, *, tier_map: Optional[Dict[str, Dict[str, Any]]] = None,
                tiers: Optional[set] = None, trivial_frac: float = 0.0) -> List[Dict[str, Any]]:
    states: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traj = json.loads(line)
            if traj.get("status") not in (None, "ok"):
                continue
            if tier_map is not None:
                t = (tier_map.get(traj.get("qid")) or {}).get("tier")
                if t == "trivial":
                    if not _include_trivial(traj.get("qid"), trivial_frac):
                        continue
                elif tiers and t not in tiers:
                    continue
            for rd in traj.get("rounds", []):
                if rd.get("candidates"):
                    states.append({"traj": traj, "rd": rd})
            if limit and len(states) >= limit:
                break
    return states[:limit] if limit else states


def build_state_sample(traj: Dict[str, Any], rd: Dict[str, Any]) -> Dict[str, Any]:
    cands = rd.get("candidates", [])
    user_text = USER_TEMPLATE.format(
        qid=traj.get("qid", ""), query_type=traj.get("query_type", ""),
        original_query=traj.get("original_query", ""), current_query=rd.get("query", ""),
        question=traj.get("question", ""), options=json.dumps(traj.get("options", {}), ensure_ascii=False),
        candidate_block=format_candidates(cands),
    )
    return {
        "system": SYSTEM_PROMPT_V11,
        "user_text": user_text,
        "query_image_path": traj.get("query_image_path", ""),
        "evidence_image_paths": [str(c.get("image_path") or "") for c in cands],
    }


def encode_target(processor, sample: Dict[str, Any], target_text: str, *,
                  max_images: int, query_edge: int, ev_edge: int, device,
                  prefix_text: str = "") -> Dict[str, Any]:
    """Re-encode the full chat (prompt + sampled decision) via the processor.

    Hand-concatenating token IDs breaks Qwen3.5's 3D-RoPE token-type bookkeeping,
    so we tokenize the whole conversation in one shot (same path as SFT/DPO) and
    locate the target span via a prompt-only render. When prefix_text is given
    (forced-REWRITE exploration), we additionally locate the continuation start
    (render with the prefix as a to-be-continued assistant turn) so the forced
    prefix tokens can be EXCLUDED from PG/KL: gradients only touch what the
    policy freely sampled.
    """
    msg_full, images = build_messages(sample, max_images=max_images, query_edge=query_edge,
                                      ev_edge=ev_edge, include_target=True, target_text=target_text)
    msg_prompt, _ = build_messages(sample, max_images=max_images, query_edge=query_edge,
                                   ev_edge=ev_edge, include_target=False)
    full_text = render_chat(processor, msg_full, add_generation_prompt=False)
    prompt_text = render_chat(processor, msg_prompt, add_generation_prompt=True)
    img_arg = images if images else None
    full = processor(text=[full_text], images=img_arg, return_tensors="pt").to(device)
    prompt = processor(text=[prompt_text], images=img_arg, return_tensors="pt")
    enc = {k: v for k, v in full.items()}
    n_prompt = int(prompt["input_ids"].shape[1])
    enc["n_prompt"] = n_prompt
    n_prefix = 0
    if prefix_text:
        msg_pre, _ = build_messages(sample, max_images=max_images, query_edge=query_edge,
                                    ev_edge=ev_edge, include_target=True, target_text=prefix_text)
        try:
            pre_text = processor.apply_chat_template(
                msg_pre, tokenize=False, add_generation_prompt=False,
                continue_final_message=True, enable_thinking=False)
        except TypeError:
            pre_text = processor.apply_chat_template(
                msg_pre, tokenize=False, add_generation_prompt=False,
                continue_final_message=True)
        pre = processor(text=[pre_text], images=img_arg, return_tensors="pt")
        n_prefix = max(0, int(pre["input_ids"].shape[1]) - n_prompt)
    enc["n_prefix"] = n_prefix
    return enc


def target_token_logprobs(model, enc: Dict[str, Any]) -> torch.Tensor:
    """Per-token log-prob of the assistant-target tokens (logits_to_keep trick)."""
    n_prompt = int(enc["n_prompt"])
    input_ids = enc["input_ids"]
    total_len = int(input_ids.shape[1])
    keep = max(2, total_len - n_prompt + 1)
    fwd = {k: v for k, v in enc.items() if k not in ("n_prompt", "n_prefix")}
    out = model(**fwd, logits_to_keep=keep)
    shift_logits = out.logits[:, :-1, :]                        # (1, keep-1, V)
    shift_labels = input_ids[:, total_len - (keep - 1):]         # (1, keep-1)
    logp = torch.log_softmax(shift_logits.float(), dim=-1)
    return logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)[0]  # (keep-1,)


def build_pg_batch(encs: List[Dict[str, Any]], pad_id: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[int], int]:
    """Stack per-rollout encodings into ONE right-padded multimodal batch.

    All rollouts share the identical prompt (same images), so pixel_values /
    image_grid_thw are concatenated in sample order (== tiling the shared
    patches), matching the image-token layout of each input_ids row. Every
    per-token tensor of shape [1, L] (input_ids, attention_mask,
    mm_token_type_ids, ...) is right-padded and stacked along batch. Returns
    (fwd_dict, labels[B,K], true_lens, n_prompt) where K = max_len - n_prompt.
    """
    n_prompt = int(encs[0]["n_prompt"])
    lens = [int(e["input_ids"].shape[1]) for e in encs]
    max_len = max(lens)
    B = len(encs)
    dev = encs[0]["input_ids"].device
    fwd: Dict[str, torch.Tensor] = {}
    # generic per-token [1, L] tensors -> right-pad + stack along batch
    per_token_keys = [k for k, v in encs[0].items()
                      if torch.is_tensor(v) and v.dim() == 2 and v.shape[0] == 1 and v.shape[1] == lens[0]]
    for k in per_token_keys:
        fill = pad_id if k == "input_ids" else 0
        buf = torch.full((B, max_len), fill, dtype=encs[0][k].dtype, device=dev)
        for i, e in enumerate(encs):
            buf[i, :lens[i]] = e[k][0]
        fwd[k] = buf
    # multimodal patch tensors -> concatenate in sample order
    pvs = [e["pixel_values"] for e in encs if "pixel_values" in e]
    gthw = [e["image_grid_thw"] for e in encs if "image_grid_thw" in e]
    if pvs:
        fwd["pixel_values"] = torch.cat(pvs, dim=0)
    if gthw:
        fwd["image_grid_thw"] = torch.cat(gthw, dim=0)
    labels = fwd["input_ids"][:, n_prompt:]                # (B, K)
    return fwd, labels, lens, n_prompt


def batched_target_logprobs(model, fwd: Dict[str, torch.Tensor], labels: torch.Tensor, keep: int) -> torch.Tensor:
    """Batched per-token log-prob over the continuation window (logits_to_keep)."""
    out = model(**fwd, logits_to_keep=keep)
    shift_logits = out.logits[:, :-1, :]                  # (B, K, V)
    logp = torch.log_softmax(shift_logits.float(), dim=-1)
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)   # (B, K)


def score_evidence_parallel(scorers, question, options, gold_letter, q_img, evidences):
    """Score P(gold) for many evidence sets, spreading forwards across scorer GPUs.

    One scorer  -> sequential (original single-GPU behavior).
    >=2 scorers -> round-robin the jobs onto each GPU and run one worker thread per
    GPU; the GIL is released during the CUDA forwards, so the GPUs score concurrently
    and the dominant answer-utility scoring cost is cut ~Nx for N scorer replicas.
    """
    if not evidences:
        return []
    if len(scorers) == 1:
        sc = scorers[0]
        return [sc.answer_prob(question, options, gold_letter, q_img, evidence=ev)
                for ev in evidences]
    from concurrent.futures import ThreadPoolExecutor  # noqa: E402
    groups = [[] for _ in scorers]
    for i, ev in enumerate(evidences):
        groups[i % len(scorers)].append((i, ev))
    results = [0.0] * len(evidences)

    def _run_group(gidx):
        sc = scorers[gidx]
        for i, ev in groups[gidx]:
            results[i] = sc.answer_prob(question, options, gold_letter, q_img, evidence=ev)

    with ThreadPoolExecutor(max_workers=len(scorers)) as ex:
        list(ex.map(_run_group, range(len(scorers))))
    return results


def _hit_to_ev(h: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a retrieval hit to the scorer's evidence schema."""
    return {"text": h.get("text", ""), "image_path": h.get("image_path", ""),
            "doc_name": h.get("doc_name", ""), "page_idx": h.get("page_idx", ""),
            "score": h.get("score", 0.0)}


def _tile_inputs(inputs, n):
    """Repeat a single-sample processor batch n times along dim 0 for batched generate.

    Works for the multimodal keys: input_ids/attention_mask [1,L]->[n,L], and the
    vision keys pixel_values [P,D]->[n*P,D], image_grid_thw [I,3]->[n*I,3] (each
    tensor's leading dim scales with the batch, so a plain dim-0 repeat stays aligned).
    """
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            out[k] = v.repeat(n, *([1] * (v.dim() - 1)))
        else:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ref-model", default="")
    ap.add_argument("--ref-device", default="", help="put frozen ref on its own GPU (e.g. cuda:1); default=same as --device")
    ap.add_argument("--grad-ckpt", type=int, default=1, help="1=gradient checkpointing on (save mem), 0=off (faster backward)")
    ap.add_argument("--pg-batch", type=int, default=4, help="rollouts per batched PG forward/backward (memory vs speed)")
    ap.add_argument("--clip-grad-norm", type=float, default=5.0, help="max grad norm for clipping (raw ~75; 1.0=very conservative, 5.0=relaxed)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.4)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta-kl", type=float, default=0.02)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0, help="save mid-run ckpt every N optimizer updates; 0=only final")
    ap.add_argument("--max-images", type=int, default=8)
    ap.add_argument("--reward", choices=["answer_utility", "evidence_hit"], default="answer_utility",
                    help="answer_utility=frozen-generator P(correct) delta vs no-evidence baseline")
    ap.add_argument("--gen-model", default=DEFAULT_GEN_MODEL, help="frozen generator for answer_utility reward")
    ap.add_argument("--gen-device", default="cuda:1", help="device for the frozen generator scorer")
    ap.add_argument("--gen-device2", default="",
                    help="optional 2nd scorer GPU; scoring is split across both to ~2x the reward throughput")
    ap.add_argument("--leak-penalty", type=float, default=0.5,
                    help="weight on HARD leakage only ('答案是X'/'最终答案'); answer_text hit is monitor-only")
    ap.add_argument("--rewrite-cost", type=float, default=0.0,
                    help="subtracted from REWRITE rollouts' reward (online default 0)")
    ap.add_argument("--gate-eps", type=float, default=0.05,
                    help="answer_utility only: skip a state (before rollout) if generator logP(correct) "
                         "over {none,top3,all} evidence spans < this (evidence can't move the answer)")
    ap.add_argument("--tier-file", default="",
                    help="judge_tier jsonl -> tier-aware state filtering (selection/rewrite + trivial subsample)")
    ap.add_argument("--tiers", default="selection,rewrite",
                    help="comma-separated tiers to keep when --tier-file is set (trivial handled by frac)")
    ap.add_argument("--include-trivial-frac", type=float, default=0.15,
                    help="fraction of trivial-tier states kept (crc32(qid) deterministic)")
    ap.add_argument("--n-forced-rewrite", type=int, default=0,
                    help="forced-REWRITE prefix rollouts per group for exploration (Phase 4: use 2 of G=8)")
    ap.add_argument("--adv-std-floor", type=float, default=0.5,
                    help="floor on the group-advantage std denominator; >0 stops tiny-spread "
                         "(noise-only) groups from being inflated to full +-1 advantages")
    ap.add_argument("--min-spread", type=float, default=0.3,
                    help="skip the PG update when max-min group reward < this (noise-only group)")
    ap.add_argument("--rewrite-retrieval", type=int, default=1,
                    help="1=REWRITE rollouts re-run real text retrieval with the rewritten query and "
                         "are scored on the retrieved evidence (true rewrite-quality reward); 0=off")
    ap.add_argument("--rewrite-topk", type=int, default=5,
                    help="top-k retrieved chunks scored as the REWRITE rollout's evidence")
    ap.add_argument("--probe-every", type=int, default=20,
                    help="run a greedy eval probe on held-out states every N optimizer updates; 0=off")
    ap.add_argument("--probe-states", type=int, default=24,
                    help="number of trailing states reserved (excluded from training) for the probe")
    ap.add_argument("--query-edge", type=int, default=768)
    ap.add_argument("--ev-edge", type=int, default=384)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="1 optimizer step then exit")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ref_device = torch.device(args.ref_device) if (args.ref_device and torch.cuda.is_available()) else device
    ref_path = args.ref_model or args.model

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    eos_id = processor.tokenizer.eos_token_id
    tier_map = None
    tiers = None
    if args.tier_file:
        tier_map = load_tier_map(args.tier_file)
        tiers = {t.strip() for t in args.tiers.split(",") if t.strip()}
    states = read_states(args.input, args.limit, tier_map=tier_map, tiers=tiers,
                         trivial_frac=args.include_trivial_frac)
    probe_states: List[Dict[str, Any]] = []
    if args.probe_every > 0 and args.probe_states > 0 and len(states) > args.probe_states * 4:
        probe_states = states[-args.probe_states:]
        states = states[:-args.probe_states]
    print(f"[cfg] states={len(states)} G={args.group_size} T={args.temperature} "
          f"beta_kl={args.beta_kl} reward={args.reward} grad_ckpt={args.grad_ckpt} "
          f"forced_rw={args.n_forced_rewrite} tier_file={bool(args.tier_file)} "
          f"probe={len(probe_states)} rw_retr={args.rewrite_retrieval} "
          f"policy={device} ref={ref_device} gen={args.gen_device}", flush=True)

    policy = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device)
    policy.config.use_cache = True
    policy.enable_input_require_grads()
    ref = AutoModelForImageTextToText.from_pretrained(
        ref_path, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(ref_device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    scorer = None
    scorers = []
    if args.reward == "answer_utility":
        from gen_scorer import AnswerScorer  # noqa: E402
        scorer = AnswerScorer(args.gen_model, device=args.gen_device,
                              query_edge=args.query_edge, ev_edge=args.ev_edge,
                              max_images=args.max_images)
        scorers.append(scorer)
        print(f"[gen] answer-utility scorer loaded: {Path(args.gen_model).name} on {args.gen_device}", flush=True)
        if args.gen_device2:
            scorer2 = AnswerScorer(args.gen_model, device=args.gen_device2,
                                   query_edge=args.query_edge, ev_edge=args.ev_edge,
                                   max_images=args.max_images)
            scorers.append(scorer2)
            print(f"[gen] 2nd scorer loaded on {args.gen_device2} (parallel scoring x{len(scorers)})", flush=True)

    opt = Adafactor(policy.parameters(), lr=args.lr, scale_parameter=False,
                    relative_step=False, warmup_init=False)

    # ---- real retrieval for REWRITE rewards: BGE-M3 + Milvus text search ----
    # The rewritten query only affects TEXT retrieval at runtime (the query image is
    # unchanged), so scoring top-k text hits measures true rewrite quality.
    retr = None
    if args.rewrite_retrieval and args.reward == "answer_utility":
        from retrieval_adapter import FirstStageRetriever, load_config as load_retr_config  # noqa: E402
        retr_cfg = load_retr_config()
        # small BGE-M3 encoder rides on the 2nd scorer GPU (plenty of headroom)
        retr_cfg["retrieval"]["text_device"] = args.gen_device2 or args.gen_device
        retr = FirstStageRetriever(retr_cfg)
        # warm up encoder + collection so the first state isn't slow
        retr.search_text("warmup", k=1)
        print(f"[retr] rewrite retrieval online: text_device={retr_cfg['retrieval']['text_device']} "
              f"topk={args.rewrite_topk}", flush=True)

    gen_kwargs = dict(do_sample=True, temperature=args.temperature, top_p=args.top_p,
                      max_new_tokens=args.max_new_tokens, pad_token_id=eos_id)

    # ---- greedy eval probe on held-out states: the TRUE progress metric ----
    # training R_mean is measured on T=1.4 samples, so sampling noise buries any
    # policy improvement; a periodic greedy decode on fixed unseen states shows
    # whether the policy's deterministic behavior actually gets better.
    probe_cache: Dict[str, float] = {}

    def run_probe(tag: str) -> None:
        if not probe_states or scorer is None:
            return
        t_p0 = time.time()
        policy.eval()
        policy.gradient_checkpointing_disable()
        policy.config.use_cache = True
        us: List[float] = []
        n_rw = 0
        n_bad = 0
        with torch.no_grad():
            for pst in probe_states:
                ptraj, prd = pst["traj"], pst["rd"]
                popts = ptraj.get("options", {}) or {}
                pgold = str(ptraj.get("answer", "") or "")
                if not popts or not pgold:
                    continue
                pq_img = ptraj.get("query_image_path", "")
                pquestion = ptraj.get("question", "")
                pcands = prd.get("candidates", [])
                psample = build_state_sample(ptraj, prd)
                pmsgs, pimages = build_messages(psample, max_images=args.max_images,
                                                query_edge=args.query_edge, ev_edge=args.ev_edge,
                                                include_target=False)
                ptxt = render_chat(processor, pmsgs, add_generation_prompt=True)
                pinp = processor(text=[ptxt], images=pimages if pimages else None,
                                 return_tensors="pt").to(device)
                pseq = policy.generate(**pinp, do_sample=False,
                                       max_new_tokens=args.max_new_tokens, pad_token_id=eos_id)
                ptext = processor.tokenizer.decode(pseq[0, pinp["input_ids"].shape[1]:],
                                                   skip_special_tokens=True)
                pact = parse_action(ptext, len(pcands))
                if not pact["parsed"]:
                    n_bad += 1
                key = f"{ptraj.get('qid', '')}#{prd.get('round_idx', '')}"
                if key not in probe_cache:
                    pb = scorer.answer_prob(pquestion, popts, pgold, pq_img, evidence=[])
                    probe_cache[key] = math.log(max(pb, 1e-6))
                plp_base = probe_cache[key]
                pev = [{"text": pcands[i].get("text", ""), "image_path": pcands[i].get("image_path", ""),
                        "doc_name": pcands[i].get("doc_name", ""), "page_idx": pcands[i].get("page_idx", ""),
                        "score": pcands[i].get("score", 0.0)}
                       for i in pact["keep"] if 0 <= i < len(pcands)]
                if pact["action"] == "REWRITE":
                    n_rw += 1
                    prq = pact.get("rewrite_query", "")
                    if retr is not None and prq:
                        try:
                            phits = retr.search_text(prq, k=args.rewrite_topk)
                        except Exception:
                            phits = []
                        pev = pev + [_hit_to_ev(h) for h in phits[:args.rewrite_topk]]
                if pev:
                    pk = scorer.answer_prob(pquestion, popts, pgold, pq_img, evidence=pev)
                    us.append(math.log(max(pk, 1e-6)) - plp_base)
                else:
                    us.append(0.0)
        if us:
            print(f"[probe {tag}] mean_u={sum(us) / len(us):+.4f} n={len(us)} rw={n_rw} "
                  f"bad_json={n_bad} t={time.time() - t_p0:.0f}s", flush=True)

    run_probe("u0")

    n_epochs = int(math.ceil(args.epochs))
    update = 0
    accum = 0
    n_flat = 0
    last_gnorm = 0.0
    for epoch in range(n_epochs):
        for si, st in enumerate(states):
            t0 = time.time()
            traj, rd = st["traj"], st["rd"]
            gold = traj.get("gold_source", {}) or {}
            sample = build_state_sample(traj, rd)
            cands = rd.get("candidates", [])
            n_cand = len(cands)
            msgs, images = build_messages(sample, max_images=args.max_images,
                                          query_edge=args.query_edge, ev_edge=args.ev_edge,
                                          include_target=False)
            prompt_text = render_chat(processor, msgs, add_generation_prompt=True)
            base_inputs = processor(text=[prompt_text], images=images if images else None,
                                    return_tensors="pt").to(device)
            prompt_ids = base_inputs["input_ids"]
            prompt_len = int(prompt_ids.shape[1])

            # ---- answer-utility baseline: generator logP(correct) with NO evidence ----
            question = traj.get("question", "")
            options = traj.get("options", {}) or {}
            gold_letter = str(traj.get("answer", "") or "")
            q_img = traj.get("query_image_path", "")
            p_base = 0.0
            lp_base = 0.0
            if scorer is not None and options and gold_letter:
                p_base = scorer.answer_prob(question, options, gold_letter, q_img, evidence=[])
                lp_base = math.log(max(p_base, 1e-6))

            def _kept_evidence(keep):
                return [{"text": cands[i].get("text", ""),
                         "image_path": cands[i].get("image_path", ""),
                         "doc_name": cands[i].get("doc_name", ""),
                         "page_idx": cands[i].get("page_idx", ""),
                         "score": cands[i].get("score", 0.0)}
                        for i in keep if 0 <= i < n_cand]

            # ---- cheap gate: skip states where evidence can't move the answer ----
            if scorer is not None and options and gold_letter and args.gate_eps > 0:
                all_ev = _kept_evidence(list(range(n_cand)))
                gp = score_evidence_parallel(scorers, question, options, gold_letter, q_img,
                                             [all_ev[:3], all_ev])
                lp_t3 = math.log(max(gp[0], 1e-6))
                lp_all = math.log(max(gp[1], 1e-6))
                if max(lp_base, lp_all, lp_t3) - min(lp_base, lp_all, lp_t3) < args.gate_eps:
                    n_flat += 1
                    if args.log_every and si % args.log_every == 0:
                        print(f"[e{epoch} s{si}] flat-skip: lp[none/t3/all]={lp_base:.2f}/{lp_t3:.2f}/{lp_all:.2f}", flush=True)
                    continue

            # ---- rollout: sample G decisions; first n_forced behind a REWRITE prefix ----
            n_forced = min(args.n_forced_rewrite, args.group_size) if n_cand else 0
            forced_inputs = None
            forced_len = 0
            forced_prefix = ""
            if n_forced:
                forced_prefix = _forced_rewrite_prefix(n_cand)
                msgs_f, images_f = build_messages(sample, max_images=args.max_images,
                                                  query_edge=args.query_edge, ev_edge=args.ev_edge,
                                                  include_target=True, target_text=forced_prefix)
                try:
                    ftxt = processor.apply_chat_template(
                        msgs_f, tokenize=False, add_generation_prompt=False,
                        continue_final_message=True, enable_thinking=False)
                except TypeError:
                    ftxt = processor.apply_chat_template(
                        msgs_f, tokenize=False, add_generation_prompt=False,
                        continue_final_message=True)
                forced_inputs = processor(text=[ftxt], images=images_f if images_f else None,
                                          return_tensors="pt").to(device)
                forced_len = int(forced_inputs["input_ids"].shape[1])

            t_gen0 = time.time()
            rollouts: List[Dict[str, Any]] = []
            policy.eval()
            policy.gradient_checkpointing_disable()
            policy.config.use_cache = True
            n_natural = args.group_size - n_forced
            with torch.no_grad():
                # forced-REWRITE rollouts: one batched generate over the shared prefix
                if n_forced:
                    f_seq = policy.generate(**_tile_inputs(forced_inputs, n_forced), **gen_kwargs)
                    for gi in range(n_forced):
                        cont = processor.tokenizer.decode(f_seq[gi, forced_len:], skip_special_tokens=True)
                        # continuation may not close the JSON, so construct the action
                        # directly (same as the offline pipeline) instead of parsing it.
                        rq = cont.split('"')[0].strip()
                        rollouts.append({"text": forced_prefix + cont, "keep": [], "action": "REWRITE",
                                         "rewrite_query": rq, "parsed": bool(rq), "prefix": forced_prefix})
                # natural rollouts: one batched generate over the shared prompt
                if n_natural > 0:
                    n_seq = policy.generate(**_tile_inputs(base_inputs, n_natural), **gen_kwargs)
                    for gi in range(n_natural):
                        text = processor.tokenizer.decode(n_seq[gi, prompt_len:], skip_special_tokens=True)
                        action = parse_action(text, n_cand)
                        if not action["parsed"]:
                            # truncated JSON: recover keep leniently; keep parsed=False flag
                            m = re.search(r'"keep"\s*:\s*\[([^\]]*)\]', text)
                            if m:
                                action["keep"] = sorted(set(
                                    int(x) for x in re.findall(r"\d+", m.group(1))
                                    if 0 <= int(x) < n_cand))
                        rollouts.append({"text": text, "keep": action["keep"], "action": action["action"],
                                         "rewrite_query": action.get("rewrite_query", ""),
                                         "parsed": action.get("parsed", True), "prefix": ""})

            t_gen1 = time.time()
            # ---- score every rollout's kept evidence in parallel across scorer GPUs ----
            if scorer is not None and options and gold_letter:
                # REWRITE rollouts: re-run REAL text retrieval with the rewritten query
                # (query image unchanged at runtime -> text search is the only delta) and
                # score kept + retrieved evidence, so the reward finally reflects rewrite
                # QUALITY instead of a constant  u=0 - cost  that carried no signal.
                rw_ev: Dict[int, List[Dict[str, Any]]] = {}
                if retr is not None:
                    seen_q: Dict[str, List[Dict[str, Any]]] = {}
                    for ri, r in enumerate(rollouts):
                        if r["action"] == "REWRITE" and r["rewrite_query"]:
                            rq = r["rewrite_query"]
                            if rq not in seen_q:
                                try:
                                    hits = retr.search_text(rq, k=args.rewrite_topk)
                                except Exception:
                                    hits = []
                                seen_q[rq] = [_hit_to_ev(h) for h in hits[:args.rewrite_topk]]
                            rw_ev[ri] = seen_q[rq]
                jobs = []
                for ri, r in enumerate(rollouts):
                    ev = _kept_evidence(r["keep"])
                    if r["action"] == "REWRITE":
                        ev = ev + rw_ev.get(ri, [])
                    if ev:
                        jobs.append((ri, ev))
                probs = score_evidence_parallel(scorers, question, options, gold_letter, q_img,
                                                [ev for _, ev in jobs])
                pmap = {ri: p for (ri, _), p in zip(jobs, probs)}
                for ri, r in enumerate(rollouts):
                    query = (r["rewrite_query"] if (r["action"] == "REWRITE" and r["rewrite_query"])
                             else rd.get("query", ""))
                    p_kept = pmap.get(ri, p_base)
                    # unified: u = Δ logP_options(gold) (log-domain, no saturation)
                    u = math.log(max(p_kept, 1e-6)) - lp_base
                    leak = leakage_penalty(query, answer=traj.get("answer", ""),
                                           answer_text=traj.get("answer_text", ""))
                    reward = u - args.leak_penalty * hard_leakage(query, answer=traj.get("answer", ""))
                    if r["action"] == "REWRITE":
                        reward -= args.rewrite_cost
                    if not r["parsed"]:
                        reward -= 0.5
                    r["reward"] = reward
                    r["comp"] = {"p_kept": p_kept, "p_base": p_base, "u": u, "leak": leak,
                                 "parsed": r["parsed"]}
            else:
                for r in rollouts:
                    action = {"keep": r["keep"], "action": r["action"],
                              "rewrite_query": r["rewrite_query"], "parsed": r["parsed"]}
                    reward, comp = action_reward(rd, action, gold_doc=str(gold.get("doc_id") or ""),
                                                 gold_page=gold.get("page_idx"),
                                                 answer=traj.get("answer", ""),
                                                 answer_text=traj.get("answer_text", ""))
                    r["reward"] = reward
                    r["comp"] = comp
            t_score1 = time.time()
            rewards = [r["reward"] for r in rollouts]
            # noise gate: a group whose rewards barely differ carries no usable
            # ranking signal -- updating on it is a pure random walk, skip it.
            spread = max(rewards) - min(rewards)
            if spread < args.min_spread:
                n_flat += 1
                if args.log_every and si % args.log_every == 0:
                    print(f"[e{epoch} s{si}] spread-skip: spread={spread:.3f} "
                          f"mean={sum(rewards) / len(rewards):+.3f}", flush=True)
                continue
            advs = group_advantages(rewards, std_floor=args.adv_std_floor)
            if max(abs(a) for a in advs) < 1e-6:
                if args.log_every and si % args.log_every == 0:
                    print(f"[e{epoch} s{si}] skip: zero-variance rewards mean={sum(rewards)/len(rewards):.3f}", flush=True)
                continue

            # ---- policy gradient with KL to frozen ref ----
            policy.train()
            policy.config.use_cache = False
            if args.grad_ckpt:
                policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            step_loss = 0.0
            step_kl = 0.0
            # encode all rollouts once (CPU), then do chunked BATCHED fwd+bwd:
            # the identical prompt/images are tiled across the batch, so the vision
            # tower + LLM prompt are processed in one batched kernel per chunk
            # instead of len(rollouts) sequential under-utilized launches.
            encs = [encode_target(processor, sample, r["text"], max_images=args.max_images,
                                  query_edge=args.query_edge, ev_edge=args.ev_edge, device=device,
                                  prefix_text=r.get("prefix", "")) for r in rollouts]
            G = len(rollouts)
            pg_b = max(1, int(args.pg_batch))
            for c0 in range(0, G, pg_b):
                chunk = encs[c0:c0 + pg_b]
                chunk_adv = advs[c0:c0 + pg_b]
                fwd, labels, lens, n_prompt = build_pg_batch(chunk, eos_id)
                K = labels.shape[1]
                keep = K + 1
                pol_logp = batched_target_logprobs(policy, fwd, labels, keep)      # (c, K) grad
                with torch.no_grad():
                    if ref_device != device:
                        fwd_ref = {k: (v.to(ref_device) if torch.is_tensor(v) else v) for k, v in fwd.items()}
                        ref_logp = batched_target_logprobs(ref, fwd_ref, labels.to(ref_device), keep).to(device)
                    else:
                        ref_logp = batched_target_logprobs(ref, fwd, labels, keep)
                # mask: valid continuation token j in [n_prefix_i, true_len_i - n_prompt)
                js = torch.arange(K, device=device)
                lower = torch.tensor([int(e.get("n_prefix", 0)) for e in chunk], device=device)
                upper = torch.tensor([lens[i] - n_prompt for i in range(len(chunk))], device=device)
                mask = ((js[None, :] >= lower[:, None]) & (js[None, :] < upper[:, None])).float()
                denom = mask.sum(1).clamp(min=1.0)
                pol_mean = (pol_logp * mask).sum(1) / denom                       # (c,)
                diff = ref_logp - pol_logp
                kl_tok = torch.exp(diff) - diff - 1.0                             # k3 estimator
                kl_mean = (kl_tok * mask).sum(1) / denom                          # (c,)
                adv_t = torch.tensor(chunk_adv, device=device, dtype=torch.float32)
                pg = -adv_t * pol_mean                                            # (c,)
                loss = (pg.sum() + args.beta_kl * kl_mean.sum()) / G
                loss.backward()
                step_loss += float(pg.sum().detach())
                step_kl += float(kl_mean.sum().detach())
            t_bwd1 = time.time()
            accum += 1
            if accum % args.grad_accum == 0 or si == len(states) - 1:
                last_gnorm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip_grad_norm))
                opt.step()
                opt.zero_grad()
                update += 1
                if args.probe_every and update % args.probe_every == 0:
                    run_probe(f"u{update}")
                if args.save_every and update % args.save_every == 0:
                    mid = Path(f"{args.out_dir}_u{update}")
                    mid.mkdir(parents=True, exist_ok=True)
                    policy.save_pretrained(str(mid))
                    processor.save_pretrained(str(mid))
                    print(f"[save] mid-run ckpt -> {mid}", flush=True)
            if args.log_every and si % args.log_every == 0:
                extra = ""
                if scorer is not None:
                    us = [r["comp"].get("u", 0.0) for r in rollouts]
                    extra = f" lp_base={lp_base:.3f} u=[{min(us):.3f},{max(us):.3f}]"
                print(f"[e{epoch} s{si}] R_mean={sum(rewards)/len(rewards):+.3f} "
                      f"R_max={max(rewards):+.3f} pg={step_loss/len(rollouts):+.4f} "
                      f"kl={step_kl/len(rollouts):.4f}{extra} gnorm={last_gnorm:.3f} u={update} "
                      f"keep0={rollouts[0]['keep']} "
                      f"t[pre={t_gen0-t0:.0f} gen={t_gen1-t_gen0:.0f} score={t_score1-t_gen1:.0f} "
                      f"bwd={t_bwd1-t_score1:.0f} tot={t_bwd1-t0:.0f}]", flush=True)
            if args.smoke and update >= 1:
                print("[smoke] one optimizer step done -> exit", flush=True)
                break
        if args.smoke:
            break

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(out))
    processor.save_pretrained(str(out))
    run_probe(f"final_u{update}")
    print(f"[done] GRPO -> {out}", flush=True)


if __name__ == "__main__":
    main()
