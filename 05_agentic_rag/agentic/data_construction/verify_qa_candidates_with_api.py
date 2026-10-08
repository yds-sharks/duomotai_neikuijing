#!/usr/bin/env python3
"""Stage-1 blind-answer verification + gold assembly.

Reads accepted generation candidates, calls the verification API WITH the query
image attached (blind: candidate_answer NOT shown), enforces the strict accept
gate, deduplicates per source, and assembles the gold QA dataset with train/dev/
internal_test splits. 三件套: timing+ETA / streaming append+fsync / --resume.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from qa_api_common import (
    StreamWriter,
    accumulate_usage,
    build_client,
    build_image_messages,
    extract_usage,
    load_api_config,
    load_done_ids,
    parse_json_content,
)
from qa_api_prompts import QA_VERIFICATION_SYSTEM_PROMPT, render_verification_user_prompt
from qa_stage1_common import iter_jsonl, write_json_atomic

DEFAULT_INPUT = "/mnt/data_1/yds/多模态/agentic/outputs/qa_stage1_generated/generated_qa_candidates.jsonl"
DEFAULT_OUTPUT_DIR = "/mnt/data_1/yds/多模态/agentic/outputs/qa_stage1_verified"
DEFAULT_CONFIG = "data_construction/api_config.json"

DEPENDENCY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _is_accepted(ver: Dict[str, Any]) -> Optional[str]:
    """Return None if accepted, else a short reject reason."""
    if ver.get("verifier_answer") not in {"A", "B", "C", "D"}:
        return "missing_verifier_answer"
    if not ver.get("answer_supported"):
        return "answer_not_supported"
    if ver.get("image_dependency") in {"low", None}:
        return "low_image_dependency"
    if ver.get("question_naturalness") in {"low", None}:
        return "low_naturalness"
    if not ver.get("benchmark_style_match"):
        return "no_benchmark_style_match"
    if ver.get("multi_organ_ambiguity"):
        return "multi_organ_ambiguity"
    if ver.get("option_quality") != "good":
        return "option_quality_not_good"
    if ver.get("text_leakage"):
        return "text_leakage"
    if not ver.get("all_options_same_granularity"):
        return "mixed_option_granularity"
    if ver.get("multiple_correct_options"):
        return "multiple_correct_options"
    return None


def _gold_record(cand: Dict[str, Any], ver: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble a gold record matching design §6 schema."""
    return {
        "qid": f"QA-{cand.get('candidate_id', '')}",
        "split": cand.get("split", "unknown"),
        "query_type": cand.get("query_type", ""),
        "question": cand.get("question", ""),
        "options": cand.get("options", {}),
        "answer": cand.get("candidate_answer", ""),
        "answer_text": cand.get("candidate_answer_text", ""),
        "low_information_query_seed": cand.get("low_information_query_seed", ""),
        "query_image_path": cand.get("query_image_path", ""),
        "source": cand.get("source", {}),
        "generation": {
            "generator_model": (cand.get("generation") or {}).get("generator_model", ""),
            "alternative_low_information_query_seed": (cand.get("generation") or {}).get("alternative_low_information_query_seed", ""),
            "evidence_basis": cand.get("evidence_basis", ""),
            "image_dependency": cand.get("image_dependency", ""),
            "ambiguity": bool(cand.get("ambiguity", False)),
        },
        "verification": {
            "verifier_answer": ver.get("verifier_answer"),
            "verifier_confidence": ver.get("verifier_confidence"),
            "answer_supported": ver.get("answer_supported"),
            "image_dependency": ver.get("image_dependency"),
            "question_naturalness": ver.get("question_naturalness"),
            "benchmark_style_match": ver.get("benchmark_style_match"),
            "multi_organ_ambiguity": ver.get("multi_organ_ambiguity"),
            "option_quality": ver.get("option_quality"),
            "text_leakage": ver.get("text_leakage"),
            "all_options_same_granularity": ver.get("all_options_same_granularity"),
            "multiple_correct_options": ver.get("multiple_correct_options"),
            "reason": ver.get("reason", ""),
        },
        "provenance": {
            "candidate_id": cand.get("candidate_id"),
            "source_id": cand.get("source_id"),
            "query_type": cand.get("query_type"),
        },
    }


def process_one(cand: Dict[str, Any], client, cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    candidate_id = str(cand.get("candidate_id") or "")
    image_path = str(cand.get("query_image_path") or "")
    base = {
        "candidate_id": candidate_id,
        "source_id": cand.get("source_id"),
        "query_type": cand.get("query_type"),
    }

    if not image_path or not Path(image_path).exists():
        return {**base, "verification_status": "no_image", "api_usage": {}}

    user_text = render_verification_user_prompt(cand)
    try:
        messages = build_image_messages(QA_VERIFICATION_SYSTEM_PROMPT, user_text, image_path)
        result = client.chat(
            model=cfg["model"],
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001
        return {**base, "verification_status": "api_error", "error": str(exc)[:300], "api_usage": {}}

    usage = extract_usage(result)
    ver = parse_json_content(result.primary_text)
    if ver is None:
        return {**base, "verification_status": "invalid_json", "api_usage": usage,
                "raw_reply": result.primary_text[:500]}

    fail = _is_accepted(ver)
    if fail is not None:
        cand_answer = str(cand.get("candidate_answer") or "")
        agreed = ver.get("verifier_answer") == cand_answer
        return {**base, "verification_status": f"rejected:{fail}",
                "agrees_with_candidate": agreed,
                "verifier_answer": ver.get("verifier_answer"),
                "api_usage": usage, "verification": ver}

    cand_answer = str(cand.get("candidate_answer") or "")
    if ver.get("verifier_answer") != cand_answer:
        return {**base, "verification_status": "rejected:answer_mismatch",
                "agrees_with_candidate": False,
                "verifier_answer": ver.get("verifier_answer"),
                "api_usage": usage, "verification": ver}

    return {
        **base,
        "verification_status": "accepted",
        "agrees_with_candidate": True,
        "api_usage": usage,
        "verification": ver,
    }


def _pick_best(records: List[Dict[str, Any]], max_per_source: int) -> List[Dict[str, Any]]:
    """Per-source dedup: keep up to max_per_source, prefer high image_dependency then high naturalness."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        sid = rec.get("source_id") or rec.get("candidate_id", "unknown")
        grouped[sid].append(rec)

    picked: List[Dict[str, Any]] = []
    for sid, recs in grouped.items():
        recs.sort(key=lambda r: (
            DEPENDENCY_ORDER.get((r.get("verification") or {}).get("image_dependency", "low"), 2),
            {"high": 0, "medium": 1, "low": 2}.get((r.get("verification") or {}).get("question_naturalness", "low"), 2),
        ))
        picked.extend(recs[:max_per_source])
    return picked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-per-source", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Debug: cap number of candidates.")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_api_config(args.config)
    client = build_client(cfg)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    verified_path = out_dir / "verified_candidates.jsonl"

    candidates = [c for c in iter_jsonl(Path(args.input))
                  if c.get("generation_status") == "accepted"]
    if args.limit > 0:
        candidates = candidates[: args.limit]

    done = set() if args.no_resume else load_done_ids(verified_path, "candidate_id")
    todo = [c for c in candidates if str(c.get("candidate_id") or "") not in done]
    print(f"[info] total_accepted={len(candidates)} done={len(done)} todo={len(todo)} workers={args.workers}")

    writer = StreamWriter(verified_path)
    status_counts: Counter = Counter()
    usage_total: Dict[str, int] = {}
    agg_lock = Lock()
    t0 = time.time()
    processed = 0
    sleep_s = float(cfg.get("sleep", 0.0))

    def worker(cand: Dict[str, Any]) -> Dict[str, Any]:
        rec = process_one(cand, client, cfg, args)
        if sleep_s > 0:
            time.sleep(sleep_s)
        return rec

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(worker, cand) for cand in todo]
            for fut in as_completed(futures):
                rec = fut.result()
                writer.write(rec)
                with agg_lock:
                    processed += 1
                    status_counts[rec.get("verification_status", "unknown")] += 1
                    accumulate_usage(usage_total, rec.get("api_usage") or {})
                    if processed % 25 == 0 or processed == len(todo):
                        elapsed = time.time() - t0
                        rate = processed / elapsed if elapsed > 0 else 0.0
                        eta = (len(todo) - processed) / rate if rate > 0 else 0.0
                        print(f"[progress] {processed}/{len(todo)} accepted={status_counts['accepted']} "
                              f"rate={rate:.2f}/s eta={eta:.0f}s usage={usage_total.get('total_tokens', 0)}")
    finally:
        writer.close()

    # --- assemble gold dataset with per-source dedup ---
    accepted = [c for c in iter_jsonl(verified_path) if c.get("verification_status") == "accepted"]
    picked = _pick_best(accepted, args.max_per_source)

    gold_records = []
    for rec in picked:
        cand = next((c for c in candidates if str(c.get("candidate_id") or "") == str(rec.get("candidate_id"))), None)
        if cand is not None:
            gold_records.append(_gold_record(cand, rec.get("verification") or {}))

    # write gold
    gold_path = out_dir / "qa_gold.jsonl"
    with gold_path.open("w", encoding="utf-8") as fh:
        for rec in gold_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # split by source.split
    splits: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in gold_records:
        splits[rec["split"]].append(rec)

    split_files: Dict[str, str] = {}
    for split_name, records in splits.items():
        sp = out_dir / f"qa_{split_name}.jsonl"
        with sp.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        split_files[split_name] = str(sp)

    report = {
        "input": args.input,
        "verified_output": str(verified_path),
        "gold_output": str(gold_path),
        "split_files": split_files,
        "model": cfg["model"],
        "total_accepted_for_verification": len(candidates),
        "processed_this_run": processed,
        "resumed_from": len(done),
        "verified_accepted": status_counts.get("accepted", 0),
        "after_dedup_gold_count": len(gold_records),
        "max_per_source": args.max_per_source,
        "elapsed_s": round(time.time() - t0, 1),
        "verification_status_counts": dict(status_counts),
        "api_usage_total_this_run": usage_total,
        "split_counts": {k: len(v) for k, v in splits.items()},
    }
    write_json_atomic(out_dir / "construction_report.json", report)
    print("[done]", report)


if __name__ == "__main__":
    main()
