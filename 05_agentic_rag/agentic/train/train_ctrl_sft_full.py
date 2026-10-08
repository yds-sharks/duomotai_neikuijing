#!/usr/bin/env python3
"""Full-parameter FSDP SFT for the Qwen3.5-4B evidence controller.

Cold-start behaviour cloning: given (system v11 prompt, user_text with numbered
candidate block, query image, evidence images), reproduce the teacher's JSON
decision (keep/drop + ACCEPT/REWRITE + rewrite_query + reason). Each retrieval
round is one sample.

Memory strategy for full-param on 3x48GB (RTX A6000):
  - FSDP FULL_SHARD (accelerate config train/fsdp_qwen35.yaml)
  - Adafactor optimizer (factored 2nd moments -> near-zero optimizer state)
  - logits_to_keep: compute logits only for the trailing assistant tokens, then
    manual cross-entropy (avoids the LM-head-over-full-seq x vocab memory spike)
  - HF reentrant gradient checkpointing (non-reentrant trips Qwen3.5 attention)

Launch (3 GPUs):
  CUDA_VISIBLE_DEVICES=0,1,2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  accelerate launch --config_file train/fsdp_qwen35.yaml train/train_ctrl_sft_full.py \
    --train train/sft_ctrl_train.jsonl --val train/sft_ctrl_val.jsonl \
    --out-dir train/ckpt_qwen35_ctrl_full_v1 --epochs 2 --grad-accum 8 --lr 1e-5
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

DEFAULT_MODEL = "/mnt/data_1/yds/models/hf_hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class ControllerDataset(Dataset):
    def __init__(self, path: str, processor, *, max_images: int, query_edge: int, ev_edge: int,
                 max_len: int = 0, len_file: str = ""):
        self.rows = read_jsonl(path)
        self.n_total = len(self.rows)
        self.n_dropped = 0
        if max_len > 0 and len_file:
            lens = [int(json.loads(l)["n_tokens"]) for l in open(len_file, encoding="utf-8") if l.strip()]
            if len(lens) != self.n_total:
                raise ValueError(f"len_file rows {len(lens)} != data rows {self.n_total} for {path}")
            kept = [r for r, n in zip(self.rows, lens) if n <= max_len]
            self.n_dropped = self.n_total - len(kept)
            self.rows = kept
        self.processor = processor
        self.max_images = max_images
        self.query_edge = query_edge
        self.ev_edge = ev_edge

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        sample = self.rows[i]
        msg_full, images = build_messages(
            sample, max_images=self.max_images, query_edge=self.query_edge,
            ev_edge=self.ev_edge, include_target=True,
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
        n_prompt = int(prompt["input_ids"].shape[1])
        item = {k: v for k, v in full.items()}
        item["n_prompt"] = n_prompt
        return item


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # batch size fixed to 1 (variable image grids); pass through
    return batch[0]


def lr_at(step: int, total: int, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return step / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(prog, 0.0), 1.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-images", type=int, default=8)
    ap.add_argument("--query-edge", type=int, default=768)
    ap.add_argument("--ev-edge", type=int, default=384)
    ap.add_argument("--max-len", type=int, default=0, help="drop (not truncate) samples whose full-seq tokens exceed this")
    ap.add_argument("--train-len-file", default="", help="per-line n_tokens for --train (from measure_seq_len.py)")
    ap.add_argument("--val-len-file", default="", help="per-line n_tokens for --val")
    ap.add_argument("--optim", default="adafactor", choices=["adafactor", "adamw"])
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=25, help="run val (forward-only) every N updates; 0 disables")
    ap.add_argument("--save-updates", default="", help="comma-sep update indices to save mid-run ckpts (e.g. 101,151); final is always saved")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
    torch.manual_seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    ds = ControllerDataset(args.train, processor, max_images=args.max_images,
                           query_edge=args.query_edge, ev_edge=args.ev_edge,
                           max_len=args.max_len, len_file=args.train_len_file)
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate, num_workers=2)

    val_dl = None
    val_ds = None
    if args.val and args.eval_every > 0:
        val_ds = ControllerDataset(args.val, processor, max_images=args.max_images,
                                   query_edge=args.query_edge, ev_edge=args.ev_edge,
                                   max_len=args.max_len, len_file=args.val_len_file)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=2)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.enable_input_require_grads()

    if args.optim == "adafactor":
        opt = Adafactor(model.parameters(), lr=args.lr, scale_parameter=False,
                        relative_step=False, warmup_init=False, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model, opt, dl = accelerator.prepare(model, opt, dl)
    if val_dl is not None:
        val_dl = accelerator.prepare(val_dl)

    steps_per_epoch = max(math.ceil(len(dl) / args.grad_accum), 1)  # len(dl) is per-process (sharded)
    total_updates = max(int(steps_per_epoch * args.epochs), 1)
    warmup = int(total_updates * args.warmup_ratio)
    save_at = {int(x) for x in args.save_updates.split(",") if x.strip()}
    if accelerator.is_main_process:
        print(f"[cfg] samples={len(ds)} (dropped {ds.n_dropped}/{ds.n_total} > max_len={args.max_len}) "
              f"val={len(val_ds) if val_ds else 0} eval_every={args.eval_every} save_at={sorted(save_at)} "
              f"per_proc_batches={len(dl)} steps/epoch={steps_per_epoch} "
              f"total_updates={total_updates} warmup={warmup} optim={args.optim} lr={args.lr}", flush=True)

    def save_ckpt(path: Path) -> None:
        accelerator.wait_for_everyone()
        state = accelerator.get_state_dict(model)
        unwrapped = accelerator.unwrap_model(model)
        is_main = accelerator.is_main_process
        if is_main:
            path.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(str(path), is_main_process=is_main,
                                  state_dict=state, save_function=accelerator.save)
        if is_main:
            processor.save_pretrained(str(path))
            print(f"[save] {path}", flush=True)

    @torch.no_grad()
    def evaluate() -> float:
        if val_dl is None:
            return float("nan")
        model.eval()
        tot = torch.zeros(2, device=accelerator.device)  # [sum_loss, count]
        for batch in val_dl:
            n_prompt = int(batch.pop("n_prompt"))
            batch.pop("labels", None)
            input_ids = batch["input_ids"]
            total_len = int(input_ids.shape[1])
            keep = max(2, total_len - n_prompt + 1)
            out = model(**batch, logits_to_keep=keep)
            shift_logits = out.logits[:, :-1, :]
            shift_labels = input_ids[:, total_len - (keep - 1):]
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                shift_labels.reshape(-1),
            )
            tot[0] += loss.detach()
            tot[1] += 1
        tot = accelerator.reduce(tot, reduction="sum")
        model.train()
        return (tot[0] / tot[1].clamp(min=1)).item()

    update = 0
    n_epochs = int(math.ceil(args.epochs))
    model.train()
    for epoch in range(n_epochs):
        for step, batch in enumerate(dl):
            with accelerator.accumulate(model):
                n_prompt = int(batch.pop("n_prompt"))
                batch.pop("labels", None)
                input_ids = batch["input_ids"]
                total_len = int(input_ids.shape[1])
                keep = max(2, total_len - n_prompt + 1)
                out = model(**batch, logits_to_keep=keep)
                shift_logits = out.logits[:, :-1, :]
                shift_labels = input_ids[:, total_len - (keep - 1):]
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                    shift_labels.reshape(-1),
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    update += 1
                    for g in opt.param_groups:
                        g["lr"] = args.lr * lr_at(update, total_updates, warmup)
                opt.step()
                opt.zero_grad()
            if accelerator.sync_gradients and accelerator.is_main_process and update % args.log_every == 0:
                print(f"[e{epoch} u{update}/{total_updates}] loss={loss.item():.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
            if accelerator.sync_gradients and args.eval_every > 0 and update > 0 and update % args.eval_every == 0:
                vl = evaluate()
                if accelerator.is_main_process:
                    print(f"[eval u{update}/{total_updates}] val_loss={vl:.4f}", flush=True)
            if accelerator.sync_gradients and update in save_at:
                save_ckpt(Path(args.out_dir).parent / f"{Path(args.out_dir).name}_u{update}")
        if args.eval_every > 0:
            vl = evaluate()
            if accelerator.is_main_process:
                print(f"[eval epoch{epoch + 1} u{update}/{total_updates}] val_loss={vl:.4f}", flush=True)

    save_ckpt(Path(args.out_dir))
    if accelerator.is_main_process:
        print("[done] SFT full-param training complete", flush=True)


if __name__ == "__main__":
    main()
