#!/usr/bin/env python3
"""Frozen generator (Qwen3-VL-8B) answer-probability scorer for the GRPO reward.

Reward signal for the evidence controller = the *frozen generator*'s confidence in
the CORRECT option given the kept evidence, read deterministically (no sampling):

    p = softmax_over_options( next-token logits at {A,B,C,D} )[gold]

Feeding different kept-evidence sets changes p continuously, which is exactly the
dense signal GRPO needs (unlike the discrete evidence-hit reward). The controller
reward is then  logp(kept) - logp(no-evidence baseline)  (options-normalized).

The prompt mirrors the PRODUCTION RAG prompt (code/rag_prompting.build_rag_prompt):
same RAG_INSTRUCTION, same evidence block (format_evidence), same question/options
layout, no system message. Only the answer trigger is a direct letter head
("...答案：") so the next-token option logits can be read off in one forward.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

CODE_DIR = Path(__file__).resolve().parent.parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from rag_prompting import (  # noqa: E402
    RAG_INSTRUCTION, format_evidence, format_options, valid_options,
)


def _load_image(path: str, max_edge: int) -> Optional[Image.Image]:
    if not path:
        return None
    try:
        img = Image.open(path).convert("RGB")
        if max(img.size) > max_edge:
            img.thumbnail((max_edge, max_edge), Image.LANCZOS)
        return img
    except Exception:
        return None


class AnswerScorer:
    """Loads Qwen3-VL-8B once; scores P(correct option | question, options, evidence)."""

    def __init__(self, model_path: str, device: str = "cuda:1", *,
                 query_edge: int = 768, ev_edge: int = 384, max_images: int = 8):
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tok = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.query_edge = query_edge
        self.ev_edge = ev_edge
        self.max_images = max_images
        self._letter_cache: Dict[str, List[int]] = {}

    def _letter_ids(self, letter: str) -> List[int]:
        """Single-token ids that render the option letter (bare and space-prefixed)."""
        if letter in self._letter_cache:
            return self._letter_cache[letter]
        ids = set()
        for s in (letter, " " + letter):
            enc = self.tok.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(int(enc[0]))
        out = sorted(ids)
        self._letter_cache[letter] = out
        return out

    @torch.no_grad()
    def _option_logits(self, question: str, options: Dict[str, str],
                       query_image_path: str, evidence: List[Dict[str, str]],
                       use_query_image: bool = True):
        """One forward pass -> (letters, stacked option logits, full next-token logits)."""
        content: List[Dict[str, Any]] = []
        images: List[Image.Image] = []
        qimg = _load_image(query_image_path, self.query_edge) if use_query_image else None
        if qimg is not None:
            content.append({"type": "image"})
            images.append(qimg)
        budget = self.max_images - len(images)
        for ev in evidence[:max(0, budget)]:
            im = _load_image(ev.get("image_path", ""), self.ev_edge)
            if im is not None:
                content.append({"type": "image"})
                images.append(im)

        letters = valid_options(options) or list(options.keys())
        opt_block = format_options(options)
        if evidence:
            ev_txt = format_evidence(evidence)
            ctx = f"{RAG_INSTRUCTION}\n\n参考资料：\n{ev_txt}\n\n"
        else:
            ctx = ""
        text = (
            f"{ctx}{question}\n{opt_block}\n\n"
            f"请只输出唯一正确选项的字母（{'/'.join(letters)}）。答案："
        )
        content.append({"type": "text", "text": text})

        messages = [
            {"role": "user", "content": content},
        ]
        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[chat], images=images if images else None, return_tensors="pt",
        ).to(self.device)

        out = self.model(**inputs)
        logits = out.logits[0, -1, :].float()  # next-token distribution
        opt_logits = []
        for L in letters:
            ids = self._letter_ids(L)
            if ids:
                opt_logits.append(torch.logsumexp(logits[ids], dim=0))
            else:
                opt_logits.append(torch.tensor(-1e9, device=logits.device))
        return letters, torch.stack(opt_logits), logits

    @torch.no_grad()
    def judge(self, question: str, options: Dict[str, str], gold_letter: str,
              query_image_path: str, evidence: List[Dict[str, str]],
              use_query_image: bool = True) -> Dict[str, Any]:
        """Unified primitive: one forward -> 4-option distribution, prediction,
        correctness, and log P(gold). Correctness = argmax over {A,B,C,D} == gold."""
        letters, stacked, _ = self._option_logits(
            question, options, query_image_path, evidence, use_query_image)
        logprobs = torch.log_softmax(stacked, dim=0)
        probs = torch.softmax(stacked, dim=0)
        pred_idx = int(torch.argmax(stacked).item())
        pred = letters[pred_idx]
        gi = letters.index(gold_letter) if gold_letter in letters else -1
        return {
            "probs": {L: float(probs[i].item()) for i, L in enumerate(letters)},
            "pred": pred,
            "correct": bool(gi >= 0 and pred == gold_letter),
            "logp_gold": float(logprobs[gi].item()) if gi >= 0 else float("-1e9"),
            "p_gold": float(probs[gi].item()) if gi >= 0 else 0.0,
        }

    @torch.no_grad()
    def answer_prob(self, question: str, options: Dict[str, str], gold_letter: str,
                    query_image_path: str, evidence: List[Dict[str, str]],
                    normalize: str = "options", use_query_image: bool = True) -> float:
        letters = valid_options(options) or list(options.keys())
        if gold_letter not in letters:
            return 0.0
        if normalize == "vocab":
            _, _, logits = self._option_logits(
                question, options, query_image_path, evidence, use_query_image)
            probs_full = torch.softmax(logits, dim=0)
            ids = self._letter_ids(gold_letter)
            return float(probs_full[ids].sum().item()) if ids else 0.0
        letters, stacked, _ = self._option_logits(
            question, options, query_image_path, evidence, use_query_image)
        probs = torch.softmax(stacked, dim=0)
        return float(probs[letters.index(gold_letter)].item())

