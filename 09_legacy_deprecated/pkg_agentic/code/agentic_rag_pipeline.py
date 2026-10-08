#!/usr/bin/env python3
"""Run the frozen RAG stack for Stage-2/3 agentic experiments.

Input: Stage-1 QA JSONL or candidate JSONL.
Output: one JSON row per query with first-stage retrieval, the score-truncated
top-k evidence (no reranker model), and optional generator response/prediction.

This is the no-agent baseline: the full top-k evidence is fed to the generator.
The evidence-selection + rewrite agent lives in run_gpt_agent_rollout_smoke.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from evidence_selection import select_top_evidence
from generator_adapter import OpenAICompatibleGenerator
from rag_prompting import build_rag_prompt, parse_prediction
from retrieval_adapter import FirstStageRetriever, load_config


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def get_options(row: Dict[str, Any]) -> Dict[str, Any]:
    options = row.get("options")
    if isinstance(options, dict):
        return options
    out = {}
    for key in "ABCDEF":
        if key in row:
            out[key] = row.get(key)
    return out


def get_answer(row: Dict[str, Any]) -> str:
    return str(row.get("answer") or row.get("candidate_answer") or "").strip()


def get_answer_text(row: Dict[str, Any]) -> str:
    return str(row.get("answer_text") or row.get("candidate_answer_text") or "").strip()


def get_qid(row: Dict[str, Any], idx: int) -> str:
    for key in ("qid", "candidate_id", "source_id", "sample_id", "index"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return f"row_{idx:06d}"


def get_query_text(row: Dict[str, Any], args: argparse.Namespace) -> str:
    if args.query_field and row.get(args.query_field):
        return str(row.get(args.query_field)).strip()
    for key in ("low_information_query_seed", "original_query", "retrieval_query", "question"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def get_question(row: Dict[str, Any]) -> str:
    return str(row.get("question") or row.get("candidate_question") or "").strip()


def get_image_path(row: Dict[str, Any]) -> str:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    for key in ("query_image_path", "image_path"):
        value = row.get(key) or source.get(key)
        if value:
            return str(value)
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "agentic_runtime_config.json"))
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--query-field", default="low_information_query_seed")
    parser.add_argument("--text-k", type=int, default=0)
    parser.add_argument("--image-k", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--skip-generator", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(Path(args.input_jsonl)))
    start = max(int(args.offset), 0)
    stop = len(rows) if args.limit <= 0 else min(len(rows), start + int(args.limit))
    rows = rows[start:stop]

    generator = None if args.skip_generator else OpenAICompatibleGenerator(config)

    with FirstStageRetriever(config) as retriever, out_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=start):
            qid = get_qid(row, idx)
            query_text = get_query_text(row, args)
            image_path = get_image_path(row)
            question = get_question(row)
            options = get_options(row)
            answer = get_answer(row)
            answer_text = get_answer_text(row)
            try:
                retrieval = retriever.retrieve(
                    query_text,
                    image_path,
                    text_k=args.text_k or None,
                    image_k=args.image_k or None,
                )
                top5 = select_top_evidence(retrieval["combined"], select_k=args.topk)
                response = ""
                pred = ""
                correct = None
                if generator is not None and question and options:
                    prompt, valid = build_rag_prompt(question, options, top5)
                    response = generator.generate(prompt, image_path=image_path)
                    pred = parse_prediction(response, valid)
                    correct = (pred == answer) if answer else None
                record = {
                    "qid": qid,
                    "row_index": idx,
                    "query_text": query_text,
                    "query_image_path": image_path,
                    "question": question,
                    "options": options,
                    "answer": answer,
                    "answer_text": answer_text,
                    "retrieval_text": retrieval["text"],
                    "retrieval_image": retrieval["image"],
                    "top5_evidence": top5,
                    "generator_response": response,
                    "prediction": pred,
                    "correct": correct,
                    "raw_input": row,
                }
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                record = {
                    "qid": qid,
                    "row_index": idx,
                    "query_text": query_text,
                    "query_image_path": image_path,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "raw_input": row,
                }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
    print(f"[done] rows={len(rows)} output={out_path}")


if __name__ == "__main__":
    main()
