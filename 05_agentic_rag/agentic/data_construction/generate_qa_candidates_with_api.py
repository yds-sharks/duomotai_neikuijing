#!/usr/bin/env python3
"""Stage-1 candidate QA generation via an image-capable API.

Reads routed candidates, calls the generation API WITH the query image attached,
parses the JSON reply, enforces hard constraints, and streams accepted/rejected
records. 代码三件套: timing+ETA / streaming append+fsync / --resume. API usage is
recorded per record and totaled in the report.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
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
from qa_api_prompts import QA_GENERATION_SYSTEM_PROMPT, render_generation_user_prompt
from qa_stage1_common import iter_jsonl, write_json_atomic

DEFAULT_INPUT = "/mnt/data_1/yds/多模态/agentic/outputs/qa_stage1_prepared/routed_qa_candidates.jsonl"
DEFAULT_OUTPUT_DIR = "/mnt/data_1/yds/多模态/agentic/outputs/qa_stage1_generated"
DEFAULT_CONFIG = "data_construction/api_config.json"

SOURCE_KEYS = [
    "sample_id", "group_id", "doc_id", "doc_name", "origin_pdf", "source_path",
    "content_list_path", "page_idx", "pdf_context_pages", "image_item_index",
    "image_local_context_text", "caption_or_pair_text", "pdf_context_text",
    "organ_tags", "primary_knowledge_type", "secondary_knowledge_types", "labels",
]
VALID_OPTIONS = {"A", "B", "C", "D"}


def source_subset(cand: Dict[str, Any]) -> Dict[str, Any]:
    return {key: cand.get(key) for key in SOURCE_KEYS}


def validate_generation(gen: Dict[str, Any]) -> Optional[str]:
    """Return None if valid, else a short failure reason."""
    options = gen.get("options")
    if not isinstance(options, dict) or set(options) != VALID_OPTIONS:
        return "invalid_options"
    if len({str(v) for v in options.values()}) != 4:
        return "duplicate_option_values"
    ans = gen.get("candidate_answer")
    if ans not in VALID_OPTIONS:
        return "invalid_candidate_answer"
    if str(gen.get("candidate_answer_text") or "") != str(options.get(ans) or ""):
        return "answer_text_mismatch"
    if not str(gen.get("question") or "").strip():
        return "empty_question"
    if not str(gen.get("evidence_basis") or "").strip():
        return "missing_evidence_basis"
    if gen.get("image_dependency") not in {"high", "medium", "low"}:
        return "invalid_image_dependency"
    if gen.get("image_dependency") == "low":
        return "image_dependency_low"
    return None


def process_one(cand: Dict[str, Any], client, cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    candidate_id = str(cand.get("candidate_id") or "")
    query_type = str(cand.get("candidate_query_type") or "")
    image_path = str(cand.get("query_image_path") or "")
    base = {
        "candidate_id": candidate_id,
        "source_id": cand.get("source_id"),
        "query_type": query_type,
        "split": cand.get("split"),
        "query_image_path": image_path,
        "low_information_query_seed": cand.get("low_information_query_seed"),
        "source": source_subset(cand),
    }

    if not image_path or not Path(image_path).exists():
        return {**base, "generation_status": "no_image", "api_usage": {}}

    user_text = render_generation_user_prompt(cand, query_type)
    try:
        messages = build_image_messages(QA_GENERATION_SYSTEM_PROMPT, user_text, image_path)
        result = client.chat(
            model=cfg["model"],
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001
        return {**base, "generation_status": "api_error", "error": str(exc)[:300], "api_usage": {}}

    usage = extract_usage(result)
    gen = parse_json_content(result.primary_text)
    if gen is None:
        return {**base, "generation_status": "invalid_json", "api_usage": usage,
                "raw_reply": result.primary_text[:500]}

    if gen.get("accept_source") is False:
        return {**base, "generation_status": "source_rejected", "api_usage": usage,
                "reject_reason": str(gen.get("reject_reason") or "")[:300]}

    fail = validate_generation(gen)
    if fail is not None:
        return {**base, "generation_status": f"invalid:{fail}", "api_usage": usage,
                "raw_generation": gen}

    return {
        **base,
        "generation_status": "accepted",
        "question": gen.get("question"),
        "options": gen.get("options"),
        "candidate_answer": gen.get("candidate_answer"),
        "candidate_answer_text": gen.get("candidate_answer_text"),
        "evidence_basis": gen.get("evidence_basis"),
        "image_dependency": gen.get("image_dependency"),
        "ambiguity": bool(gen.get("ambiguity", False)),
        "generation": {
            "generator_model": cfg["model"],
            "accept_source": True,
            # rule default stays authoritative; API alternative is only recorded, not applied
            "alternative_low_information_query_seed": gen.get("alternative_low_information_query_seed", ""),
        },
        "api_usage": usage,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--limit", type=int, default=0, help="Debug: cap number of candidates.")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_api_config(args.config)
    client = build_client(cfg)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generated_qa_candidates.jsonl"

    candidates = list(iter_jsonl(Path(args.input)))
    if args.limit > 0:
        candidates = candidates[: args.limit]

    done = set() if args.no_resume else load_done_ids(out_path, "candidate_id")
    todo = [c for c in candidates if str(c.get("candidate_id") or "") not in done]
    print(f"[info] total={len(candidates)} done={len(done)} todo={len(todo)} workers={args.workers}")

    writer = StreamWriter(out_path)
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
                    status_counts[rec.get("generation_status", "unknown")] += 1
                    accumulate_usage(usage_total, rec.get("api_usage") or {})
                    if processed % 25 == 0 or processed == len(todo):
                        elapsed = time.time() - t0
                        rate = processed / elapsed if elapsed > 0 else 0.0
                        eta = (len(todo) - processed) / rate if rate > 0 else 0.0
                        print(f"[progress] {processed}/{len(todo)} accepted={status_counts['accepted']} "
                              f"rate={rate:.2f}/s eta={eta:.0f}s usage={usage_total.get('total_tokens', 0)}")
    finally:
        writer.close()

    report = {
        "input": args.input,
        "output": str(out_path),
        "model": cfg["model"],
        "total_candidates": len(candidates),
        "processed_this_run": processed,
        "resumed_from": len(done),
        "elapsed_s": round(time.time() - t0, 1),
        "status_counts": dict(status_counts),
        "api_usage_total_this_run": usage_total,
    }
    write_json_atomic(out_dir / "generation_report.json", report)
    print("[done]", report)


if __name__ == "__main__":
    main()
