#!/usr/bin/env python3
"""In-process transformers backend for the Qwen3.5-4B brain/generator.

vLLM 0.11 does not support the Qwen3_5 architecture, so real smoke runs load the
model directly with AutoModelForImageTextToText (same class as FSDP training).
One shared model instance serves brain / evidence-filter / generator roles within
a process (module-level cache avoids double GPU memory).

Switch back to vLLM later by setting backend="openai" in harness_config.json.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

from backends.llm_backend import extract_json
from backends.reward_backend import extract_option_letter, render_generator_prompt

_MODEL_CACHE: Dict[str, Any] = {}
_LOCK = threading.Lock()


def _load(model_path: str, device: str):
    key = f"{model_path}@{device}"
    with _LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device
        )
        model.eval()
        _MODEL_CACHE[key] = (model, processor)
        return _MODEL_CACHE[key]


class TransformersChat:
    """Chat over a local Qwen3.5-4B; exposes decide() (JSON protocol) and generate()."""

    def __init__(self, cfg: Dict[str, Any]):
        self.model_path = cfg.get("model_path") or cfg["model"]
        self.device = cfg.get("device", "cuda:0")
        self.model, self.processor = _load(self.model_path, self.device)
        self.temperature = float(cfg.get("temperature", 0.2))
        self.top_p = float(cfg.get("top_p", 0.9))
        self.max_tokens = int(cfg.get("max_tokens", 512))
        self.enable_thinking = bool(cfg.get("enable_thinking", cfg.get("chat_template_kwargs", {}).get("enable_thinking", False)))

    def chat(self, messages: List[Dict[str, str]], image_path: str = "") -> str:
        import torch

        # multimodal call: prepend an image placeholder to the last user turn
        # (same convention as gen_scorer._option_logits in the v0.5 eval pkg)
        msgs = [dict(m) for m in messages]
        pil_imgs = []
        if image_path:
            from PIL import Image

            pil = Image.open(image_path).convert("RGB")
            pil_imgs.append(pil)
            msgs[-1] = {
                "role": msgs[-1].get("role", "user"),
                "content": [{"type": "image"}, {"type": "text", "text": msgs[-1].get("content", "")}],
            }
        prompt = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking
        )
        inputs = self.processor(
            text=[prompt], images=pil_imgs or None, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-4),
                top_p=self.top_p,
            )
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    # --- brain / evidence-filter protocol ---
    def decide(self, messages: List[Dict[str, str]], image_path: str = "") -> Dict[str, Any]:
        return extract_json(self.chat(messages, image_path=image_path))

    # --- generator protocol (RewardBackend) ---
    def generate(self, question: str, options: Dict[str, Any], evidence: List[Dict[str, Any]], image_path: str = "") -> Dict[str, Any]:
        prompt = render_generator_prompt(question, options, evidence, image_attached=bool(image_path))
        text = self.chat([{"role": "user", "content": prompt}], image_path=image_path)
        return {"response": text, "prediction": extract_option_letter(text, options)}

    def utility_from_generation(self, out: Dict[str, Any], gold_answer: str) -> float:
        return 1.0 if gold_answer and out.get("prediction") == gold_answer else 0.0

    def answer_utility(self, question, options, evidence, gold_answer: str = "", image_path: str = "") -> float:
        gen = self.generate(question, options, evidence, image_path=image_path)
        return self.utility_from_generation(gen, gold_answer)

    def close(self) -> None: ...


def reset_model_cache() -> None:
    """Free the shared model (e.g. between runner processes)."""
    _MODEL_CACHE.clear()
    import torch

    torch.cuda.empty_cache()
