#!/usr/bin/env python3
"""v0.3 agentic RAG runtime pipeline.

Flow (no reranker):
  query + image → retrieve (text top-20 + image top-20, ~40 combined)
    → agent policy: keep/drop + ACCEPT/REWRITE
      → ACCEPT: generator answers MCQ with kept evidence
      → REWRITE: suppress dropped/seen IDs → retrieve again (up to T rounds)

All retrieval / generator frozen; only agent is trainable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from generator_adapter import OpenAICompatibleGenerator
from gpt_agent_adapter import GPTAgent
from rag_prompting import build_rag_prompt, parse_prediction
from retrieval_adapter import FirstStageRetriever, load_config

CODE_DIR = Path(__file__).resolve().parent
AGENTIC_ROOT = CODE_DIR.parent
DEFAULT_API_CONFIG = AGENTIC_ROOT / "data_construction" / "api_config.local.json"


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


def apply_keep_drop(evidence: List[Dict[str, Any]], keep: List[int]) -> List[Dict[str, Any]]:
    """按 agent 的 keep 索引取 kept 证据子集。"""
    return [evidence[i] for i in keep if 0 <= i < len(evidence)]


def collect_evidence_ids(evidence: List[Dict[str, Any]], indices: List[int]) -> Set[str]:
    """收集指定索引的证据 ID，用于 suppression。"""
    ids: Set[str] = set()
    for i in indices:
        if 0 <= i < len(evidence):
            item = evidence[i]
            eid = item.get("sample_id") or item.get("doc_id") or ""
            if eid:
                ids.add(str(eid))
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CODE_DIR / "agentic_runtime_config.json"))
    parser.add_argument("--api-config", default=str(DEFAULT_API_CONFIG))
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--query-field", default="low_information_query_seed")
    parser.add_argument("--text-k", type=int, default=20)
    parser.add_argument("--image-k", type=int, default=20)
    parser.add_argument("--max-rounds", type=int, default=2, help="Max REWRITE rounds (T).")
    parser.add_argument("--skip-agent", action="store_true", help="Skip agent; use all evidence as-is.")
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

    agent_cfg = config.get("agent", {})
    agent = None if args.skip_agent else GPTAgent(
        api_config_path=args.api_config,
        max_evidence_images=int(agent_cfg.get("max_evidence_images", 20)),
    )
    generator = None if args.skip_generator else OpenAICompatibleGenerator(config)

    t0 = time.time()
    with FirstStageRetriever(config) as retriever, out_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=start):
            t_row = time.time()
            qid = get_qid(row, idx)
            query_text = get_query_text(row, args)
            image_path = get_image_path(row)
            question = get_question(row)
            options = get_options(row)
            answer = get_answer(row)
            answer_text = get_answer_text(row)

            try:
                suppressed_ids: Set[str] = set()
                rounds_log: List[Dict[str, Any]] = []
                final_evidence: List[Dict[str, Any]] = []
                final_action = "ACCEPT"
                final_rewrite_query = ""
                round_count = 0

                for round_idx in range(args.max_rounds + 1):
                    round_count = round_idx + 1
                    retrieval = retriever.retrieve(
                        query_text, image_path,
                        text_k=args.text_k or None, image_k=args.image_k or None,
                        exclude_ids=suppressed_ids,
                    )
                    evidence = retrieval["combined"]

                    if agent is not None:
                        agent_out = agent.decide(
                            qid=qid,
                            query_type=str(row.get("query_type") or ""),
                            original_query=query_text,
                            question=question,
                            options=options,
                            evidence=evidence,
                            image_path=image_path,
                        )
                    else:
                        # 跳过 agent：keep all, ACCEPT
                        agent_out = {
                            "keep": list(range(len(evidence))),
                            "drop": [],
                            "action": "ACCEPT",
                            "rewrite_query": "",
                            "reason": "skip_agent",
                        }

                    kept = apply_keep_drop(evidence, agent_out["keep"])
                    dropped_ids = collect_evidence_ids(evidence, agent_out["drop"])
                    suppressed_ids |= dropped_ids
                    # 也抑制已见证据（避免多轮打转）
                    seen_ids = collect_evidence_ids(evidence, agent_out["keep"])
                    suppressed_ids |= seen_ids

                    rounds_log.append({
                        "round": round_idx,
                        "query": query_text,
                        "num_evidence": len(evidence),
                        "keep": agent_out["keep"],
                        "drop": agent_out["drop"],
                        "action": agent_out["action"],
                        "rewrite_query": agent_out.get("rewrite_query", ""),
                        "reason": agent_out.get("reason", ""),
                    })

                    if agent_out["action"] == "ACCEPT" or round_idx >= args.max_rounds:
                        final_evidence = kept
                        final_action = agent_out["action"]
                        break

                    # REWRITE → 下一轮用 rewrite_query
                    query_text = agent_out.get("rewrite_query") or query_text
                    final_rewrite_query = agent_out.get("rewrite_query", "")

                # Generator
                response = ""
                pred = ""
                correct = None
                if generator is not None and question and options:
                    prompt, valid = build_rag_prompt(question, options, final_evidence)
                    response = generator.generate(prompt, image_path=image_path)
                    pred = parse_prediction(response, valid)
                    correct = (pred == answer) if answer else None

                record = {
                    "qid": qid,
                    "row_index": idx,
                    "query_text": get_query_text(row, args),
                    "query_image_path": image_path,
                    "question": question,
                    "options": options,
                    "answer": answer,
                    "answer_text": answer_text,
                    "final_action": final_action,
                    "final_evidence": final_evidence,
                    "final_rewrite_query": final_rewrite_query,
                    "round_count": round_count,
                    "rounds": rounds_log,
                    "generator_response": response,
                    "prediction": pred,
                    "correct": correct,
                    "elapsed_s": round(time.time() - t_row, 2),
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
                }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            if (idx - start + 1) % 10 == 0 or (idx - start + 1) == len(rows):
                elapsed = time.time() - t0
                rate = (idx - start + 1) / elapsed if elapsed > 0 else 0.0
                eta = (len(rows) - (idx - start + 1)) / rate if rate > 0 else 0.0
                print(f"[progress] {idx - start + 1}/{len(rows)} rate={rate:.1f}/s eta={eta:.0f}s")

    print(f"[done] rows={len(rows)} output={out_path} elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
