#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEV (Supportiveness Evaluation) Reranking for Multimodal RAG.

Uses a fine-tuned Qwen2.5-7B cross-encoder to score candidate passages/descriptions
for their supportiveness to the question, then fuses with first-stage retrieval scores.

Usage:
    CUDA_VISIBLE_DEVICES=0 python rerank_sev_multimodal.py \
        --retrieval-export-jsonl <input> \
        --output-jsonl <output> \
        --limit 500
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from tqdm import tqdm


# ─── Model Definition ───────────────────────────────────────────────────────

class QwenRerankerModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        hidden_size = self.backbone.config.hidden_size
        self.score_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        self.cls_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, T, D]
        mask = attention_mask.unsqueeze(-1).float()
        sent = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)  # [B, D]
        sent = sent.to(self.score_head[0].weight.dtype)
        score = self.score_head(sent).squeeze(-1)      # [B] in [0,1]
        logit = self.cls_head(sent).squeeze(-1)        # [B]
        return score, logit


# ─── SEV Inference Engine ────────────────────────────────────────────────────

class SEVInferenceEngine:
    def __init__(
        self,
        base_model_path: str,
        adapter_path: str,
        cls_head_path: str,
        score_head_path: Optional[str] = None,
        device: str = "cuda:0",
        max_length: int = 2048,
        dtype: str = "bf16",
    ):
        self.device = device
        self.max_length = max_length

        torch_dtype = None
        if device.startswith("cuda"):
            if dtype in ("bf16", "bfloat16"):
                torch_dtype = torch.bfloat16
            elif dtype in ("fp16", "float16"):
                torch_dtype = torch.float16

        print(f"[SEV] Loading tokenizer from {base_model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[SEV] Loading base model ...")
        base = AutoModel.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        print(f"[SEV] Loading LoRA adapter from {adapter_path} ...")
        base = PeftModel.from_pretrained(base, adapter_path)

        self.model = QwenRerankerModel(base)

        print(f"[SEV] Loading cls_head from {cls_head_path} ...")
        self.model.cls_head.load_state_dict(torch.load(cls_head_path, map_location="cpu"))

        if score_head_path and os.path.exists(score_head_path):
            print(f"[SEV] Loading score_head from {score_head_path} ...")
            self.model.score_head.load_state_dict(torch.load(score_head_path, map_location="cpu"))

        self.model.to(self.device).eval()
        print(f"[SEV] Model loaded on {device}, dtype={dtype}, max_length={max_length}")

    @torch.no_grad()
    def predict_support_scores(self, questions: List[str], passages: List[str], batch_size: int = 8) -> List[float]:
        """Predict P(support) for each (question, passage) pair."""
        assert len(questions) == len(passages)
        if not passages:
            return []

        # Build prompts matching the fine-tuning template
        prompts = [f"问题：{q}\n内容：{p}" for q, p in zip(questions, passages)]
        all_scores: List[float] = []

        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            pred_scores, logits = self.model(enc["input_ids"], enc["attention_mask"])

            # Use sigmoid on cls_head logits as P(support)
            probs = torch.sigmoid(logits.float()).cpu().tolist()
            all_scores.extend(probs)

        return all_scores


# ─── Main Reranking Logic ────────────────────────────────────────────────────

def get_first_stage_score(item: dict) -> float:
    """Extract first-stage retrieval score from a candidate item."""
    if "weighted_score" in item and item["weighted_score"] is not None:
        return float(item["weighted_score"])
    if "raw_score" in item and item["raw_score"] is not None:
        return float(item["raw_score"])
    if "score" in item and item["score"] is not None:
        return float(item["score"])
    return 0.0


def normalize_scores(scores: List[float]) -> List[float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    if mx - mn < 1e-9:
        return [0.5] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def process_single_question(
    sample: dict,
    retrieval: dict,
    engine: SEVInferenceEngine,
    alpha: float = 0.5,
    select_k: int = 3,
    batch_size: int = 8,
) -> dict:
    """Process a single question: score all candidates with SEV, fuse, select top-k."""
    question = sample.get("question", "")
    options = sample.get("options", {})

    # Build a richer question that includes options for context
    options_text = " ".join([f"{k}.{v}" for k, v in sorted(options.items())]) if options else ""
    full_question = f"{question} {options_text}".strip() if options_text else question

    # Collect all candidates
    text_candidates = retrieval.get("text_top20", [])
    image_candidates = retrieval.get("image_top20", [])

    candidates = []
    for item in text_candidates:
        candidates.append({
            "source": "text",
            "text": item.get("text", ""),
            "first_stage_score": get_first_stage_score(item),
            "doc_name": item.get("doc_name", ""),
            "page_idx": item.get("page_idx"),
            "sample_id": item.get("sample_id", ""),
            "image_path": item.get("image_path", ""),
        })
    for item in image_candidates:
        candidates.append({
            "source": "image",
            "text": item.get("text", ""),
            "first_stage_score": get_first_stage_score(item),
            "doc_name": item.get("doc_name", ""),
            "page_idx": item.get("page_idx"),
            "sample_id": item.get("sample_id", ""),
            "image_path": item.get("image_path", ""),
            "image_id": item.get("image_id", ""),
        })

    if not candidates:
        return {"index": sample.get("index"), "retrieval": []}

    # Run SEV model
    questions_batch = [full_question] * len(candidates)
    passages_batch = [c["text"] for c in candidates]
    sev_scores = engine.predict_support_scores(questions_batch, passages_batch, batch_size=batch_size)

    # Normalize first-stage scores
    first_stage_scores = [c["first_stage_score"] for c in candidates]
    first_stage_norm = normalize_scores(first_stage_scores)

    # Fuse scores: final = alpha * first_stage_norm + (1-alpha) * sev_score
    fused_scores = [
        alpha * fs + (1 - alpha) * ss
        for fs, ss in zip(first_stage_norm, sev_scores)
    ]

    # Sort by fused score descending
    indexed = list(enumerate(fused_scores))
    indexed.sort(key=lambda x: x[1], reverse=True)

    # Select top-k
    selected = []
    for rank_idx, (orig_idx, score) in enumerate(indexed[:select_k], start=1):
        c = candidates[orig_idx]
        entry = {
            "source": c["source"],
            "rank": rank_idx,
            "text": c["text"],
            "doc_name": c["doc_name"],
            "page_idx": c["page_idx"],
            "fused_score": round(score, 4),
            "sev_score": round(sev_scores[orig_idx], 4),
            "first_stage_score_norm": round(first_stage_norm[orig_idx], 4),
        }
        if c.get("image_path"):
            entry["image_path"] = c["image_path"]
        selected.append(entry)

    return {"index": sample.get("index"), "retrieval": selected}


def generate_baseline_top3(sample: dict, retrieval: dict, select_k: int = 3) -> dict:
    """Generate baseline: simply take original top-k by first_stage_score."""
    text_candidates = retrieval.get("text_top20", [])
    image_candidates = retrieval.get("image_top20", [])

    all_candidates = []
    for item in text_candidates:
        all_candidates.append({
            "source": "text",
            "text": item.get("text", ""),
            "score": get_first_stage_score(item),
            "doc_name": item.get("doc_name", ""),
            "page_idx": item.get("page_idx"),
            "image_path": item.get("image_path", ""),
        })
    for item in image_candidates:
        all_candidates.append({
            "source": "image",
            "text": item.get("text", ""),
            "score": get_first_stage_score(item),
            "doc_name": item.get("doc_name", ""),
            "page_idx": item.get("page_idx"),
            "image_path": item.get("image_path", ""),
        })

    # Sort by original score descending
    all_candidates.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    for rank_idx, c in enumerate(all_candidates[:select_k], start=1):
        entry = {
            "source": c["source"],
            "rank": rank_idx,
            "text": c["text"],
            "doc_name": c["doc_name"],
            "page_idx": c["page_idx"],
        }
        if c.get("image_path"):
            entry["image_path"] = c["image_path"]
        selected.append(entry)

    return {"index": sample.get("index"), "retrieval": selected}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SEV multimodal reranking")
    parser.add_argument("--retrieval-export-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True, help="SEV reranked output")
    parser.add_argument("--baseline-jsonl", default=None, help="Baseline (no reranking) output")
    parser.add_argument("--debug-jsonl", default=None, help="Debug output with all scores")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--select-k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Fusion weight: alpha*first_stage + (1-alpha)*sev")
    parser.add_argument("--max-length", type=int, default=2048)

    # Model paths (defaults from config_sev_qwen25_listwise.yaml)
    parser.add_argument("--base-model-path", 
                        default="/mnt/data_1/yds/微调/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path",
                        default="/mnt/data_1/yds/微调/分类模型微调/数据集构造_new/微调BGE/训练/listwise微调/save_weights_ddp")
    parser.add_argument("--cls-head-path",
                        default="/mnt/data_1/yds/微调/分类模型微调/数据集构造_new/微调BGE/训练/listwise微调/save_weights_ddp/rank_cls_head.pt")
    parser.add_argument("--score-head-path",
                        default="/mnt/data_1/yds/微调/分类模型微调/数据集构造_new/微调BGE/训练/listwise微调/save_weights_ddp/rank_score_head.pt")

    args = parser.parse_args()

    # Load data
    print(f"[MAIN] Loading {args.retrieval_export_jsonl} ...")
    records = []
    with open(args.retrieval_export_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit:
        records = records[:args.limit]
    print(f"[MAIN] Loaded {len(records)} records (limit={args.limit})")

    # Initialize SEV engine
    engine = SEVInferenceEngine(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        cls_head_path=args.cls_head_path,
        score_head_path=args.score_head_path,
        device=args.device,
        max_length=args.max_length,
    )

    # Process all records
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    if args.baseline_jsonl:
        os.makedirs(os.path.dirname(args.baseline_jsonl) or ".", exist_ok=True)

    f_out = open(args.output_jsonl, "w", encoding="utf-8")
    f_base = open(args.baseline_jsonl, "w", encoding="utf-8") if args.baseline_jsonl else None
    f_debug = open(args.debug_jsonl, "w", encoding="utf-8") if args.debug_jsonl else None

    t0 = time.time()
    for i, record in enumerate(tqdm(records, desc="SEV Reranking")):
        sample = record.get("sample", {})
        retrieval = record.get("retrieval", {})

        # SEV reranked result
        result = process_single_question(
            sample=sample,
            retrieval=retrieval,
            engine=engine,
            alpha=args.alpha,
            select_k=args.select_k,
            batch_size=args.batch_size,
        )
        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Baseline (no reranking)
        if f_base:
            baseline = generate_baseline_top3(sample, retrieval, select_k=args.select_k)
            f_base.write(json.dumps(baseline, ensure_ascii=False) + "\n")

        # Debug info
        if f_debug:
            debug_info = {
                "index": sample.get("index"),
                "question": sample.get("question", ""),
                "sev_result": result,
                "baseline_top3": generate_baseline_top3(sample, retrieval, select_k=args.select_k) if not f_base else None,
            }
            f_debug.write(json.dumps(debug_info, ensure_ascii=False) + "\n")

    f_out.close()
    if f_base:
        f_base.close()
    if f_debug:
        f_debug.close()

    elapsed = time.time() - t0
    qps = len(records) / elapsed if elapsed > 0 else 0
    print(f"[MAIN] Done. {len(records)} records in {elapsed:.1f}s ({qps:.2f} q/s)")
    print(f"[MAIN] Output: {args.output_jsonl}")
    if args.baseline_jsonl:
        print(f"[MAIN] Baseline: {args.baseline_jsonl}")


if __name__ == "__main__":
    main()
