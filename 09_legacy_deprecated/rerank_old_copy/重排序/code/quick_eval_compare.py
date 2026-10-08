#!/usr/bin/env python3
"""
Quick RAG evaluation: compare baseline vs SEV reranking accuracy.
Calls local vLLM server (Qwen3-VL-2B) with retrieval context and checks MCQ accuracy.
"""
import json
import re
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# Default benchmark path
DEFAULT_BENCHMARK_JSONL = "/mnt/data_10/mwx/workspace/multi_modal_rag/evaluate_benchmark/rag_retrieval_export/output/20260511_151007/retrieval_export.jsonl"

def build_prompt(question: str, options: dict, retrieval_items: list, topk: int = 3) -> str:
    """Build RAG prompt with retrieved context."""
    # Build context from retrieval
    context_parts = []
    for i, item in enumerate(retrieval_items[:topk], start=1):
        text = item.get("text", "").strip()
        if text:
            context_parts.append(f"[证据{i}] {text[:800]}")
    
    context = "\n".join(context_parts) if context_parts else "无相关参考资料"
    
    # Build options text
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
    
    prompt = f"""请根据以下参考资料回答选择题，只需输出选项字母（如A、B、C、D）。

参考资料：
{context}

问题：{question}
选项：
{options_text}

答案是："""
    return prompt


def call_vllm(prompt: str, base_url: str, model: str) -> str:
    """Call vLLM server and return generated text."""
    resp = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0.0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def parse_answer(response: str) -> str:
    """Extract answer letter from model response."""
    # Try to find a single letter A-F
    response = response.strip()
    # Common patterns
    match = re.search(r'\b([A-F])\b', response)
    if match:
        return match.group(1)
    # If response starts with a letter
    if response and response[0] in "ABCDEF":
        return response[0]
    return ""


def evaluate_condition(
    records: list,
    retrieval_results: list,
    base_url: str,
    model: str,
    label: str,
    max_workers: int = 4,
) -> dict:
    """Evaluate a set of retrieval results."""
    correct = 0
    total = 0
    errors = 0
    
    results = []
    
    for i, (record, retr_result) in enumerate(zip(records, retrieval_results)):
        sample = record["sample"]
        question = sample.get("question", "")
        options = sample.get("options", {})
        gt_answer = sample.get("answer", "")
        
        retrieval_items = retr_result.get("retrieval", [])
        
        prompt = build_prompt(question, options, retrieval_items)
        
        try:
            response = call_vllm(prompt, base_url, model)
            predicted = parse_answer(response)
            is_correct = (predicted == gt_answer)
            if is_correct:
                correct += 1
            total += 1
            results.append({
                "index": sample.get("index"),
                "gt": gt_answer,
                "pred": predicted,
                "correct": is_correct,
                "response": response[:100],
            })
        except Exception as e:
            errors += 1
            total += 1
            results.append({
                "index": sample.get("index"),
                "gt": gt_answer,
                "pred": "",
                "correct": False,
                "error": str(e)[:100],
            })
        
        if (i + 1) % 50 == 0:
            acc = correct / total if total > 0 else 0
            print(f"  [{label}] {i+1}/{len(records)} done, accuracy so far: {correct}/{total} = {acc:.3f}")
    
    accuracy = correct / total if total > 0 else 0
    return {
        "label": label,
        "correct": correct,
        "total": total,
        "errors": errors,
        "accuracy": accuracy,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-jsonl", required=True)
    parser.add_argument("--sev-jsonl", required=True)
    parser.add_argument("--retrieval-export-jsonl", default=DEFAULT_BENCHMARK_JSONL)
    parser.add_argument("--base-url", default="http://127.0.0.1:8889/v1")
    parser.add_argument("--model", default=None, help="Model name (auto-detect from server)")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Auto-detect model name
    if not args.model:
        resp = requests.get(f"{args.base_url}/models")
        models = resp.json()["data"]
        args.model = models[0]["id"]
        print(f"[INFO] Using model: {args.model}")

    # Load data
    print(f"[INFO] Loading benchmark data ...")
    records = []
    with open(args.retrieval_export_jsonl, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))
            if len(records) >= args.limit:
                break

    baseline_results = []
    with open(args.baseline_jsonl, "r") as f:
        for line in f:
            baseline_results.append(json.loads(line.strip()))

    sev_results = []
    with open(args.sev_jsonl, "r") as f:
        for line in f:
            sev_results.append(json.loads(line.strip()))

    n = min(len(records), len(baseline_results), len(sev_results))
    records = records[:n]
    baseline_results = baseline_results[:n]
    sev_results = sev_results[:n]
    print(f"[INFO] Evaluating {n} questions")

    # Run baseline evaluation
    print(f"\n{'='*60}")
    print(f"Evaluating BASELINE (原始 top-3) ...")
    print(f"{'='*60}")
    baseline_eval = evaluate_condition(records, baseline_results, args.base_url, args.model, "Baseline")

    # Run SEV evaluation
    print(f"\n{'='*60}")
    print(f"Evaluating SEV RERANKED (α=0.5) ...")
    print(f"{'='*60}")
    sev_eval = evaluate_condition(records, sev_results, args.base_url, args.model, "SEV")

    # Print comparison
    print(f"\n{'='*60}")
    print(f"{'='*60}")
    print(f"最终对比结果 (n={n})")
    print(f"{'='*60}")
    print(f"Baseline 准确率: {baseline_eval['correct']}/{baseline_eval['total']} = {baseline_eval['accuracy']:.4f} ({baseline_eval['accuracy']*100:.1f}%)")
    print(f"SEV重排序 准确率: {sev_eval['correct']}/{sev_eval['total']} = {sev_eval['accuracy']:.4f} ({sev_eval['accuracy']*100:.1f}%)")
    diff = sev_eval['accuracy'] - baseline_eval['accuracy']
    print(f"差值: {diff:+.4f} ({diff*100:+.1f}%)")
    print(f"{'='*60}")

    # Per-question diff analysis
    both_correct = 0
    only_baseline = 0
    only_sev = 0
    both_wrong = 0
    for br, sr in zip(baseline_eval['results'], sev_eval['results']):
        bc = br.get("correct", False)
        sc = sr.get("correct", False)
        if bc and sc:
            both_correct += 1
        elif bc and not sc:
            only_baseline += 1
        elif not bc and sc:
            only_sev += 1
        else:
            both_wrong += 1

    print(f"\n逐题对比:")
    print(f"  两者都对: {both_correct}")
    print(f"  仅Baseline对: {only_baseline}")
    print(f"  仅SEV对: {only_sev}")
    print(f"  两者都错: {both_wrong}")
    print(f"  SEV净收益: {only_sev - only_baseline} 题")

    # Save results
    if args.output:
        output = {
            "baseline": {k: v for k, v in baseline_eval.items() if k != "results"},
            "sev": {k: v for k, v in sev_eval.items() if k != "results"},
            "diff": diff,
            "per_question": {
                "both_correct": both_correct,
                "only_baseline": only_baseline,
                "only_sev": only_sev,
                "both_wrong": both_wrong,
            },
        }
        with open(args.output, "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
