#!/usr/bin/env python3
"""
Multimodal RAG Evaluation: Pass BOTH images and text to VLM.

Key difference from quick_eval_compare.py:
  - For image candidates: pass the actual image file to Qwen3-VL via multimodal content
  - For text candidates: pass text as before
  - This allows VLM to actually "see" retrieved images, not just read text descriptions

Usage:
    python eval_multimodal.py \
        --ppr-v2-jsonl <reranked_output> \
        --baseline-jsonl <baseline_output> \
        --retrieval-export-jsonl <source_data> \
        --limit 50
"""

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Remove proxy env vars
for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(key, None)

RETRIEVAL_EXPORT = "/mnt/data_10/mwx/workspace/multi_modal_rag/evaluate_benchmark/rag_retrieval_export/output/20260511_151007/retrieval_export.jsonl"
VLLM_URL = "http://127.0.0.1:8889/v1"


def build_multimodal_prompt(question: str, options: dict, retrieval_items: list, topk: int = 3) -> List[Dict]:
    """
    Build multimodal prompt with images and text for VLM.
    
    Returns a list of content items (text + image_url) for the chat message.
    """
    content_parts = []
    
    # Add context header
    content_parts.append({
        "type": "text",
        "text": "请根据以下参考资料回答选择题，只需输出选项字母（如A、B、C、D）。\n\n参考资料：\n"
    })
    
    # Add retrieval evidence (text + images)
    has_evidence = False
    for i, item in enumerate(retrieval_items[:topk], start=1):
        source = item.get("source", "text")
        text = item.get("text", "").strip()
        image_path = item.get("image_path", "")
        
        if source == "image" and image_path and os.path.exists(image_path):
            # Encode image as base64 data URL
            try:
                with open(image_path, "rb") as img_f:
                    img_bytes = img_f.read()
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                ext = os.path.splitext(image_path)[1].lower()
                mime = {"jpg": "image/jpeg", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
                data_url = f"data:{mime};base64,{b64}"
                content_parts.append({
                    "type": "text",
                    "text": f"\n[证据{i} - 图像]："
                })
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
            except Exception:
                # Fallback to text if image read fails
                if text:
                    content_parts.append({
                        "type": "text",
                        "text": f"\n[证据{i} - 图像描述] {text[:800]}"
                    })
                    has_evidence = True
                continue
            # Also add text description if available
            if text:
                content_parts.append({
                    "type": "text",
                    "text": f"\n图像描述：{text[:500]}"
                })
            has_evidence = True
        elif text:
            # Text-only candidate
            content_parts.append({
                "type": "text",
                "text": f"\n[证据{i} - 文本] {text[:800]}"
            })
            has_evidence = True
    
    if not has_evidence:
        content_parts.append({
            "type": "text",
            "text": "无相关参考资料"
        })
    
    # Add question and options
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
    content_parts.append({
        "type": "text",
        "text": f"\n\n问题：{question}\n选项：\n{options_text}\n\n答案是："
    })
    
    return content_parts


def build_text_only_prompt(question: str, options: dict, retrieval_items: list, topk: int = 3) -> List[Dict]:
    """Fallback: text-only prompt (same as before)."""
    parts = []
    for i, item in enumerate(retrieval_items[:topk], 1):
        text = item.get("text", "").strip()
        if text:
            parts.append(f"[证据{i}] {text[:800]}")
    context = "\n".join(parts) if parts else "无相关参考资料"
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
    
    full_text = f"请根据以下参考资料回答选择题，只需输出选项字母（如A、B、C、D）。\n\n参考资料：\n{context}\n\n问题：{question}\n选项：\n{options_text}\n\n答案是："
    return [{"type": "text", "text": full_text}]


def call_vllm_multimodal(content: List[Dict], model: str, base_url: str = VLLM_URL) -> str:
    """Call vLLM with multimodal content (text + images)."""
    resp = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 32,
            "temperature": 0.0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def parse_answer(response: str) -> str:
    """Extract answer letter from model response."""
    response = response.strip()
    m = re.search(r'\b([A-F])\b', response)
    if m:
        return m.group(1)
    if response and response[0] in "ABCDEF":
        return response[0]
    return ""


def has_valid_image(retrieval_items: list, topk: int = 3) -> bool:
    """Check if any of the top-k items has a valid image path."""
    for item in retrieval_items[:topk]:
        if item.get("source") == "image" and item.get("image_path") and os.path.exists(item.get("image_path", "")):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG Evaluation (image+text input to VLM)")
    parser.add_argument("--ppr-v2-jsonl", required=True, help="PPR v2 reranked output")
    parser.add_argument("--baseline-jsonl", required=True, help="Baseline top-3 output")
    parser.add_argument("--retrieval-export-jsonl", default=RETRIEVAL_EXPORT)
    parser.add_argument("--base-url", default=VLLM_URL)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--text-only", action="store_true", help="Use text-only mode (no images)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Auto-detect model
    if not args.model:
        resp = requests.get(f"{args.base_url}/models")
        models = resp.json()["data"]
        args.model = models[0]["id"]
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Mode: {'text-only' if args.text_only else 'MULTIMODAL (image+text)'}")

    # Load retrieval export indexed by sample index
    print("[INFO] Loading retrieval export...")
    records_by_index = {}
    with open(args.retrieval_export_jsonl) as f:
        for line in f:
            r = json.loads(line)
            records_by_index[r["sample"]["index"]] = r

    # Load results
    ppr_v2 = [json.loads(l) for l in open(args.ppr_v2_jsonl)]
    baseline = [json.loads(l) for l in open(args.baseline_jsonl)]
    
    n = min(len(ppr_v2), len(baseline), args.limit)
    ppr_v2 = ppr_v2[:n]
    baseline = baseline[:n]
    print(f"[INFO] Evaluating {n} questions")

    # Evaluate
    base_correct = ppr_correct = 0
    both_correct = only_base = only_ppr = both_wrong = 0
    img_used_count = 0
    details = []

    t0 = time.time()
    for i, (p_result, b_result) in enumerate(zip(ppr_v2, baseline)):
        idx = p_result["index"]
        record = records_by_index.get(idx)
        if not record:
            continue
        
        sample = record["sample"]
        question = sample.get("question", "")
        options = sample.get("options", {})
        gt = sample.get("answer", "")
        
        b_items = b_result.get("retrieval", [])
        p_items = p_result.get("retrieval", [])
        
        # Build prompts (multimodal or text-only)
        if args.text_only:
            b_content = build_text_only_prompt(question, options, b_items)
            p_content = build_text_only_prompt(question, options, p_items)
        else:
            b_content = build_multimodal_prompt(question, options, b_items)
            p_content = build_multimodal_prompt(question, options, p_items)
            if has_valid_image(b_items) or has_valid_image(p_items):
                img_used_count += 1
        
        # Call VLM
        try:
            b_resp = call_vllm_multimodal(b_content, args.model, args.base_url)
            b_pred = parse_answer(b_resp)
        except Exception as e:
            b_pred = ""
            print(f"  [WARN] Baseline error at {idx}: {e}")
        
        try:
            p_resp = call_vllm_multimodal(p_content, args.model, args.base_url)
            p_pred = parse_answer(p_resp)
        except Exception as e:
            p_pred = ""
            print(f"  [WARN] PPR v2 error at {idx}: {e}")
        
        b_ok = (b_pred == gt)
        p_ok = (p_pred == gt)
        
        if b_ok: base_correct += 1
        if p_ok: ppr_correct += 1
        if b_ok and p_ok: both_correct += 1
        elif b_ok and not p_ok: only_base += 1
        elif not b_ok and p_ok: only_ppr += 1
        else: both_wrong += 1
        
        details.append({
            "index": idx, "gt": gt, 
            "base_pred": b_pred, "ppr_pred": p_pred,
            "base_ok": b_ok, "ppr_ok": p_ok,
        })
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n}] Base={base_correct}/{i+1} PPR_v2={ppr_correct}/{i+1} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    mode_str = "TEXT-ONLY" if args.text_only else "MULTIMODAL"
    
    print(f"""
{'='*60}
多模态评测结果 [{mode_str}] (n={n}, {elapsed:.0f}s)
{'='*60}
Baseline:  {base_correct}/{n} = {100*base_correct/n:.1f}%
PPR v2:    {ppr_correct}/{n} = {100*ppr_correct/n:.1f}%
差值:      {100*(ppr_correct-base_correct)/n:+.1f}%
{'='*60}
两者都对: {both_correct}  仅Baseline对: {only_base}
仅PPR v2对: {only_ppr}  两者都错: {both_wrong}
PPR v2 净收益: {only_ppr - only_base} 题
图片实际传入VLM的题数: {img_used_count}/{n}
""")

    # Save results
    if args.output:
        output_data = {
            "mode": mode_str,
            "n": n,
            "baseline_acc": base_correct / n if n > 0 else 0,
            "ppr_v2_acc": ppr_correct / n if n > 0 else 0,
            "diff": (ppr_correct - base_correct) / n if n > 0 else 0,
            "only_base": only_base,
            "only_ppr": only_ppr,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "img_used_count": img_used_count,
            "elapsed_seconds": elapsed,
            "details": details,
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {args.output}")


if __name__ == "__main__":
    main()
