#!/usr/bin/env python3
"""Shared data utilities for controller SFT / DPO training.

Builds the *faithful multimodal* chat used at training time, mirroring what the
GPT teacher saw when the trajectory was recorded:

    system : v11 controller prompt
    user   : [ user_text,
               "查询图像：", <query image>,
               ("候选证据图像 [idx]：", <evidence image>) x up to max_images ]
    assistant (target only for the "full" render) : the teacher JSON decision

Evidence images are attached in candidate order using their ORIGINAL candidate
index (breadcrumb / empty paths are skipped), so the "[idx]" label matches the
numbering inside user_text.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


def load_image(path: str, max_edge: int) -> Optional[Image.Image]:
    if not path:
        return None
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        if max(img.size) > max_edge:
            img.thumbnail((max_edge, max_edge), Image.LANCZOS)
        return img
    except Exception:
        return None


def build_messages(
    sample: Dict[str, Any],
    *,
    max_images: int = 8,
    query_edge: int = 768,
    ev_edge: int = 384,
    include_target: bool = True,
    target_text: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    """Return (messages, images) where images is the flat PIL list in order."""
    images: List[Image.Image] = []
    content: List[Dict[str, Any]] = [{"type": "text", "text": str(sample.get("user_text") or "")}]

    q_img = load_image(str(sample.get("query_image_path") or ""), query_edge)
    if q_img is not None:
        content.append({"type": "text", "text": "查询图像："})
        content.append({"type": "image", "image": q_img})
        images.append(q_img)

    attached = 0
    for idx, ip in enumerate(sample.get("evidence_image_paths") or []):
        if attached >= max_images:
            break
        ev_img = load_image(str(ip or ""), ev_edge)
        if ev_img is None:
            continue
        content.append({"type": "text", "text": f"候选证据图像 [{idx}]："})
        content.append({"type": "image", "image": ev_img})
        images.append(ev_img)
        attached += 1

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": str(sample.get("system") or "")},
        {"role": "user", "content": content},
    ]
    if include_target:
        tgt = target_text if target_text is not None else str(sample.get("target") or "")
        messages.append({"role": "assistant", "content": [{"type": "text", "text": tgt}]})
    return messages, images


def render_chat(processor, messages: List[Dict[str, Any]], *, add_generation_prompt: bool) -> str:
    try:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
        )
    except TypeError:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
