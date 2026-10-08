#!/usr/bin/env python3
"""Two-sample GPT-agent rollout smoke for Stage 2/3.

This script verifies that the full control path works:
1. original low-information query -> retrieval/rerank
2. GPT API rewrite agent -> ACCEPT/REWRITE + candidates
3. each rewrite candidate -> retrieval/rerank
4. reward + group advantage/loss-weight calculation

It intentionally supports `--no-ppr-rerank` for fast connectivity tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from gpt_agent_adapter import GPTRewriteAgent
from rerank_adapter import NoOpReranker, PPRReranker
from retrieval_adapter import FirstStageRetriever, load_config
from reward_model import attach_group_advantages, evidence_hit, score_candidate

DEFAULT_INPUT = "/mnt/data_1/yds/多模态/agentic/outputs/mcq_image_v2_4000/agentic_image_mcq_4000.jsonl"
DEFAULT_OUTPUT = "/mnt/data_1/yds/多模态/agentic/outputs/stage2_gpt_agent_smoke/rollout_rewards.jsonl"

LOW_QUERY_BY_TYPE = {
    "image_organ_identification": "这张图是什么部位？",
    "anatomical_site_recognition": "这张图是什么部位？",
    "image_content_type_identification": "这张图主要展示什么？",
    "lesion_or_finding_identification": "图中是什么异常？",
    "procedure_or_operation_recognition": "图中在做什么操作？",
    "spatial_region_understanding": "图中异常在哪里？",
}


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


def get_qid(row: Dict[str, Any], idx: int) -> str:
    return str(row.get("qid") or row.get("candidate_id") or row.get("sample_id") or f"row_{idx:06d}")


def get_options(row: Dict[str, Any]) -> Dict[str, Any]:
    opts = row.get("options")
    return opts if isinstance(opts, dict) else {k: row[k] for k in "ABCDEF" if k in row}


def get_query_image(row: Dict[str, Any]) -> str:
    src = row.get("source") if isinstance(row.get("source"), dict) else {}
    return str(row.get("query_image_path") or row.get("image_path") or src.get("image_path") or "")


def get_original_query(row: Dict[str, Any]) -> str:
    for key in ("low_information_query_seed", "original_query", "retrieval_query"):
        if row.get(key):
            return str(row[key]).strip()
    return LOW_QUERY_BY_TYPE.get(str(row.get("query_type") or ""), str(row.get("question") or "").strip())


def get_gold_source(row: Dict[str, Any]) -> Dict[str, Any]:
    src = row.get("source") if isinstance(row.get("source"), dict) else {}
    return {
        "doc_id": str(src.get("doc_id") or row.get("doc_id") or ""),
        "page_idx": src.get("page_idx", row.get("page_idx")),
        "sample_id": str(src.get("sample_id") or row.get("sample_id") or ""),
    }


def select_rows(rows: Iterable[Dict[str, Any]], *, limit: int, query_type: str) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in rows:
        if query_type and row.get("query_type") != query_type:
            continue
        if not row.get("question") or not row.get("options") or not get_query_image(row):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def run_one_query(retriever: FirstStageRetriever, reranker: Any, query: str, image_path: str, *, topk: int, text_k: int, image_k: int) -> Dict[str, Any]:
    retrieval = retriever.retrieve(query, image_path, text_k=text_k, image_k=image_k)
    reranked = reranker.rerank(query, retrieval["combined"], select_k=topk)
    return {
        "query": query,
        "retrieval_text": retrieval["text"],
        "retrieval_image": retrieval["image"],
        "reranked_evidence": reranked,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "agentic_runtime_config.json"))
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT)
    parser.add_argument("--api-config", default="/mnt/data_1/yds/多模态/agentic/data_construction/api_config.local.json")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--query-type", default="image_organ_identification")
    parser.add_argument("--text-k", type=int, default=8)
    parser.add_argument("--image-k", type=int, default=8)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--no-ppr-rerank", action="store_true")
    parser.add_argument("--max-rewrites", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = select_rows(iter_jsonl(Path(args.input_jsonl)), limit=args.limit, query_type=args.query_type)
    if not rows:
        raise SystemExit("No usable smoke rows selected")

    reranker = NoOpReranker(select_k=args.topk) if args.no_ppr_rerank else PPRReranker(config)
    agent = GPTRewriteAgent(api_config_path=args.api_config)

    with FirstStageRetriever(config) as retriever, out_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows):
            qid = get_qid(row, idx)
            question = str(row.get("question") or "")
            options = get_options(row)
            answer = str(row.get("answer") or "")
            answer_text = str(row.get("answer_text") or "")
            image_path = get_query_image(row)
            original_query = get_original_query(row)
            gold = get_gold_source(row)

            original = run_one_query(
                retriever,
                reranker,
                original_query,
                image_path,
                topk=args.topk,
                text_k=args.text_k,
                image_k=args.image_k,
            )
            original_hit = evidence_hit(original["reranked_evidence"], gold_doc_id=gold["doc_id"], gold_page_idx=gold["page_idx"])

            agent_output = agent.decide(
                qid=qid,
                query_type=str(row.get("query_type") or ""),
                original_query=original_query,
                question=question,
                options=options,
                evidence=original["reranked_evidence"],
                image_path=image_path,
            )

            candidate_records: List[Dict[str, Any]] = []
            original_score = score_candidate(
                original_hit=original_hit,
                candidate_hit=original_hit,
                query=original_query,
                answer=answer,
                answer_text=answer_text,
                is_original=True,
                agent_action=agent_output["action"],
            )
            candidate_records.append({
                "candidate_id": "original",
                "query": original_query,
                "is_original": True,
                "evidence_hit": bool(original_hit),
                "reward": original_score["reward"],
                "reward_components": original_score["components"],
                "reranked_evidence": original["reranked_evidence"],
            })

            rewrites = list(agent_output.get("rewrite_candidates") or [])[: max(args.max_rewrites, 0)]
            for ridx, rewrite_query in enumerate(rewrites, start=1):
                rewritten = run_one_query(
                    retriever,
                    reranker,
                    rewrite_query,
                    image_path,
                    topk=args.topk,
                    text_k=args.text_k,
                    image_k=args.image_k,
                )
                rewrite_hit = evidence_hit(rewritten["reranked_evidence"], gold_doc_id=gold["doc_id"], gold_page_idx=gold["page_idx"])
                score = score_candidate(
                    original_hit=original_hit,
                    candidate_hit=rewrite_hit,
                    query=rewrite_query,
                    answer=answer,
                    answer_text=answer_text,
                    is_original=False,
                    agent_action=agent_output["action"],
                )
                candidate_records.append({
                    "candidate_id": f"rewrite_{ridx}",
                    "query": rewrite_query,
                    "is_original": False,
                    "evidence_hit": bool(rewrite_hit),
                    "reward": score["reward"],
                    "reward_components": score["components"],
                    "reranked_evidence": rewritten["reranked_evidence"],
                })

            reward_group = attach_group_advantages(candidate_records)
            record = {
                "qid": qid,
                "query_type": row.get("query_type"),
                "question": question,
                "options": options,
                "answer": answer,
                "answer_text": answer_text,
                "query_image_path": image_path,
                "gold_source": gold,
                "agent_output": agent_output,
                "original_result": original,
                "reward_group": reward_group,
                "smoke_note": "Uses old v2 MCQ for engineering connectivity only; not final training data.",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(json.dumps({
                "qid": qid,
                "agent_action": agent_output["action"],
                "num_rewrites": len(rewrites),
                "original_hit": original_hit,
                "best_candidate_id": reward_group.get("best_candidate_id"),
                "rewards": [round(float(c.get("reward", 0.0)), 4) for c in reward_group.get("candidates", [])],
            }, ensure_ascii=False))

    print(f"[done] rows={len(rows)} output={out_path}")


if __name__ == "__main__":
    main()
