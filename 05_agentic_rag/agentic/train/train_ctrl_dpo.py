#!/usr/bin/env python3
"""Full-parameter FSDP DPO for the Qwen3.5-4B evidence controller.

Preference optimization on top of the SFT cold-start. Consumes preference pairs
(chosen vs rejected controller decisions for the SAME state) and optimizes the
DPO loss against a frozen reference (the SFT checkpoint):

    L = -log sigmoid( beta * [ (logp_pol(chosen) - logp_ref(chosen))
                              - (logp_pol(rejected) - logp_ref(rejected)) ] )

where logp is the summed log-prob of the assistant target tokens only (prompt is
masked; computed via logits_to_keep + manual gather, same trick as SFT).

Pair data (one JSON per line) is produced by build_dpo_pairs.py and has fields:
    system, user_text, query_image_path, evidence_image_paths, chosen, rejected
(chosen/rejected are the assistant target strings; the preference is by
answer-utility P_G(a*|kept) measured with the frozen generator.)

Launch (3 GPUs):
  CUDA_VISIBLE_DEVICES=0,1,2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  accelerate launch --config_file train/fsdp_qwen35.yaml train/train_ctrl_dpo.py \
    --pairs train/dpo_pairs.jsonl --model train/ckpt_qwen35_ctrl_full_v1 \
    --out-dir train/ckpt_qwen35_ctrl_dpo_v1 --beta 0.1 --epochs 1 --lr 5e-6
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.optimization import Adafactor
from accelerate import Accelerator

from ctrl_data_common import build_messages, render_chat

DEFAULT_MODEL = "/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/ckpt_qwen35_ctrl_full_v1"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class PairDataset(Dataset):
    def __init__(self, path: str, processor, *, max_images: int, query_edge: int, ev_edge: int):
        self.rows = read_jsonl(path)
        self.processor = processor
        self.max_images = max_images
        self.query_edge = query_edge
        self.ev_edge = ev_edge

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, sample: Dict[str, Any], target_text: str) -> Dict[str, Any]:
        msg_full, images = build_messages(
            sample, max_images=self.max_images, query_edge=self.query_edge,
            ev_edge=self.ev_edge, include_target=True, target_text=target_text,
        )
        msg_prompt, _ = build_messages(
            sample, max_images=self.max_images, query_edge=self.query_edge,
            ev_edge=self.ev_edge, include_target=False,
        )
        full_text = render_chat(self.processor, msg_full, add_generation_prompt=False)
        prompt_text = render_chat(self.processor, msg_prompt, add_generation_prompt=True)
        img_arg = images if images else None
        full = self.processor(text=[full_text], images=img_arg, return_tensors="pt")
        prompt = self.processor(text=[prompt_text], images=img_arg, return_tensors="pt")
        enc = {k: v for k, v in full.items()}
        enc["n_prompt"] = int(prompt["input_ids"].shape[1])
        return enc

    def __getitem__(self, i: int) -> Dict[str, Any]:
        s = self.rows[i]
        return {
            "chosen": self._encode(s, str(s.get("chosen") or "")),
            "rejected": self._encode(s, str(s.get("rejected") or "")),
        }


def collate(batch):
    return batch[0]


def seq_logprob(model, enc: Dict[str, Any]) -> torch.Tensor:
    """Summed log-prob of the assistant-target tokens for one sequence."""
    n_prompt = int(enc.pop("n_prompt")) if "n_prompt" in enc else enc["n_prompt"]
    enc.pop("labels", None)
    input_ids = enc["input_ids"]
    total_len = int(input_ids.shape[1])
    keep = max(2, total_len - n_prompt + 1)
    out = model(**{k: v for k, v in enc.items() if k != "n_prompt"}, logits_to_keep=keep)
    shift_logits = out.logits[:, :-1, :]                       # (1, keep-1, V)
    shift_labels = input_ids[:, total_len - (keep - 1):]        # (1, keep-1)
    logp = torch.log_softmax(shift_logits.float(), dim=-1)
    tok_logp = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (1, keep-1)
    return tok_logp.sum()


def lr_at(step: int, total: int, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return step / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(prog, 0.0), 1.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="policy init = SFT checkpoint")
    ap.add_argument("--ref-model", default="", help="frozen reference; default = --model")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-images", type=int, default=8)
    ap.add_argument("--query-edge", type=int, default=768)
    ap.add_argument("--ev-edge", type=int, default=384)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
    torch.manual_seed(args.seed)
    ref_path = args.ref_model or args.model

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    ds = PairDataset(args.pairs, processor, max_images=args.max_images,
                     query_edge=args.query_edge, ev_edge=args.ev_edge)
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate, num_workers=2)

    policy = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    policy.enable_input_require_grads()

    ref = AutoModelForImageTextToText.from_pretrained(
        ref_path, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    ref.config.use_cache = False
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    opt = Adafactor(policy.parameters(), lr=args.lr, scale_parameter=False,
                    relative_step=False, warmup_init=False)

    policy, opt, dl = accelerator.prepare(policy, opt, dl)
    ref = ref.to(accelerator.device)

    steps_per_epoch = math.ceil(len(ds) / args.grad_accum)
    total_updates = max(int(steps_per_epoch * args.epochs), 1)
    warmup = int(total_updates * args.warmup_ratio)
    if accelerator.is_main_process:
        print(f"[cfg] pairs={len(ds)} total_updates={total_updates} beta={args.beta} "
              f"lr={args.lr} ref={ref_path}", flush=True)

    def save_ckpt(path: Path) -> None:
        accelerator.wait_for_everyone()
        state = accelerator.get_state_dict(policy)
        unwrapped = accelerator.unwrap_model(policy)
        is_main = accelerator.is_main_process
        if is_main:
            path.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(str(path), is_main_process=is_main,
                                  state_dict=state, save_function=accelerator.save)
        if is_main:
            processor.save_pretrained(str(path))
            print(f"[save] {path}", flush=True)

    update = 0
    n_epochs = int(math.ceil(args.epochs))
    policy.train()
    for epoch in range(n_epochs):
        for step, batch in enumerate(dl):
            with accelerator.accumulate(policy):
                chosen = {k: v for k, v in batch["chosen"].items()}
                rejected = {k: v for k, v in batch["rejected"].items()}
                nc, nr = chosen["n_prompt"], rejected["n_prompt"]
                pol_c = seq_logprob(policy, {**chosen, "n_prompt": nc})
                pol_r = seq_logprob(policy, {**rejected, "n_prompt": nr})
                with torch.no_grad():
                    ref_c = seq_logprob(ref, {**chosen, "n_prompt": nc})
                    ref_r = seq_logprob(ref, {**rejected, "n_prompt": nr})
                logits = (pol_c - ref_c) - (pol_r - ref_r)
                loss = -F.logsigmoid(args.beta * logits)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy.parameters(), 1.0)
                    update += 1
                    for g in opt.param_groups:
                        g["lr"] = args.lr * lr_at(update, total_updates, warmup)
                opt.step()
                opt.zero_grad()
            if accelerator.sync_gradients and accelerator.is_main_process and update % args.log_every == 0:
                acc = float((logits > 0).float().mean().item())
                print(f"[e{epoch} u{update}/{total_updates}] loss={loss.item():.4f} "
                      f"margin={logits.item():.3f} acc={acc:.2f} lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        save_ckpt(Path(args.out_dir).parent / f"{Path(args.out_dir).name}_epoch{epoch + 1}")

    save_ckpt(Path(args.out_dir))
    if accelerator.is_main_process:
        print("[done] DPO training complete", flush=True)


if __name__ == "__main__":
    main()
