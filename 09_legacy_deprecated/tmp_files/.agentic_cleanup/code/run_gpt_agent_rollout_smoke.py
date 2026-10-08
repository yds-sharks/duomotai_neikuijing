#!/usr/bin/env python3
"""v0.3 GPT agent rollout smoke test.

Flow (per query):
1. original query → retrieve → agent(keep/drop+ACCEPT/REWRITE)
2. If REWRITE: rewrite query → retrieve again → agent again (up to max_rounds)
3. For each round's action, measure P_G(a*|E) via generator logprobs
4. Compute answer-utility reward = P_G(a*|E) - P_G(a*|∅)
5. GRPO group advantage across all actions

No reranker. Retrieval results go directly to agent.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from generator_adapter import OpenAICompatibleGenerator
from gpt_agent_adapter import GPTAgent
from retrieval_adapter import FirstStageRetriever, load_config
from reward_model import (
    attach_group_advantages,
    evidence_hit,
    measure_p_correct,
    score_answer_utility,
)

CODE_DIR = Path(__file__).resolve().parent
AGENTIC_ROOT = CODE_DIR.parent
DEFAULT_INPUT = str(AGENTIC_ROOT / "outputs" / "qa_stage1_verified" / "qa_gold.jsonl")
DEFAULT_OUTPUT = str(AGENTIC_ROOT / "outputs" / "stage2_gpt_agent_smoke" / "rollout_rewards.jsonl")
DEFAULT_API_CONFIG = str(AGENTIC_ROOT / "data_construction" / "api_config.local.json")

LOW_QUERY_BY_TYPE = {
    "anatomical_site_recognition": "这张图是什么部位？",
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


def apply_keep_drop(evidence: List[Dict[str, Any]], keep: List[int]) -> List[Dict[str, Any]]:
    return [evidence[i] for i in keep if 0 <= i < len(evidence)]


def collect_ids(evidence: List[Dict[str, Any]], indices: List[int]) -> Set[str]:
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
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT)
    parser.add_argument("--api-config", default=DEFAULT_API_CONFIG)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--query-type", default="")
    parser.add_argument("--text-k", type=int, default=20)
    parser.add_argument("--image-k", type=int, default=20)
    parser.add_argument("--max-rounds", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = select_rows(iter_jsonl(Path(args.input_jsonl)), limit=args.limit, query_type=args.query_type)
    if not rows:
        raise SystemExit("No usable smoke rows selected")

    agent_cfg = config.get("agent", {})
    agent = GPTAgent(
        api_config_path=args.api_config,
        max_evidence_images=int(agent_cfg.get("max_evidence_images", 20)),
    )
    generator = OpenAICompatibleGenerator(config)

    t0 = time.time()
    with FirstStageRetriever(config) as retriever, out_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows):
            t_row = time.time()
            qid = get_qid(row, idx)
            question = str(row.get("question") or "")
            options = get_options(row)
            answer = str(row.get("answer") or "")
            answer_text = str(row.get("answer_text") or "")
            image_path = get_query_image(row)
            original_query = get_original_query(row)
            gold = get_gold_source(row)

            # Baseline: P_G(a*|∅)
            p_baseline, _ = measure_p_correct(
                generator, question=question, options=options, answer=answer,
                query_image_path=image_path, evidence=None,
            )

            # Run agent loop
            query_text = original_query
            suppressed_ids: Set[str] = set()
            candidate_records: List[Dict[str, Any]] = []
            rounds_log: List[Dict[str, Any]] = []

            for round_idx in range(args.max_rounds + 1):
                retrieval = retriever.retrieve(
                    query_text, image_path,
                    text_k=args.text_k, image_k=args.image_k,
                    exclude_ids=suppressed_ids,
                )
                evidence = retrieval["combined"]

                agent_out = agent.decide(
                    qid=qid,
                    query_type=str(row.get("query_type") or ""),
                    original_query=original_query if round_idx == 0 else query_text,
                    question=question,
                    options=options,
                    evidence=evidence,
                    image_path=image_path,
                )

                kept = apply_keep_drop(evidence, agent_out["keep"])
                dropped_ids = collect_ids(evidence, agent_out["drop"])
                suppressed_ids |= dropped_ids
                suppressed_ids |= collect_ids(evidence, agent_out["keep"])

                # Measure P_G(a*|kept evidence)
                p_with_kept, _ = measure_p_correct(
                    generator, question=question, options=options, answer=answer,
                    query_image_path=image_path, evidence=kept,
                )

                # Offline analysis: evidence_hit (not in reward)
                hit = evidence_hit(kept, gold_doc_id=gold["doc_id"], gold_page_idx=gold["page_idx"])

                original_utility = float(p_baseline) if round_idx == 0 else 0.0
                score = score_answer_utility(
                    p_correct_with_evidence=p_with_kept,
                    p_correct_no_evidence=p_baseline,
                    query=query_text,
                    answer=answer,
                    answer_text=answer_text,
                    is_original=(round_idx == 0 and agent_out["action"] == "ACCEPT"),
                    agent_action=agent_out["action"],
                    original_utility=original_utility,
                )

                cid = "original_accept" if round_idx == 0 else f"round_{round_idx}_{agent_out['action'].lower()}"
                candidate_records.append({
                    "candidate_id": cid,
                    "action": agent_out["action"],
                    "keep": agent_out["keep"],
                    "drop": agent_out["drop"],
                    "rewrite_query": agent_out.get("rewrite_query", ""),
                    "query": query_text,
                    "p_correct_with": p_with_kept,
                    "p_correct_baseline": p_baseline,
                    "reward": score["reward"],
                    "reward_components": score["components"],
                    "evidence_hit": bool(hit),
                    "num_kept": len(kept),
                    "round": round_idx,
                })

                rounds_log.append({
                    "round": round_idx,
                    "query": query_text,
                    "num_evidence": len(evidence),
                    "keep": agent_out["keep"],
                    "drop": agent_out["drop"],
                    "action": agent_out["action"],
                    "rewrite_query": agent_out.get("rewrite_query", ""),
                    "p_correct_with": p_with_kept,
                    "evidence_hit": bool(hit),
                })

                if agent_out["action"] == "ACCEPT" or round_idx >= args.max_rounds:
                    break

                query_text = agent_out.get("rewrite_query") or query_text

            # GRPO group advantage
            reward_group = attach_group_advantages(candidate_records)

            record = {
                "qid": qid,
                "query_type": row.get("query_type"),
                "question": question,
                "options": options,
                "answer": answer,
                "answer_text": answer_text,
                "query_image_path": image_path,
                "original_query": original_query,
                "gold_source": gold,
                "p_correct_no_evidence": p_baseline,
                "rounds": rounds_log,
                "reward_group": reward_group,
                "elapsed_s": round(time.time() - t_row, 2),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(json.dumps({
                "qid": qid,
                "p_baseline": round(p_baseline, 4),
                "num_rounds": len(rounds_log),
                "best_candidate_id": reward_group.get("best_candidate_id"),
                "rewards": [round(float(c.get("reward", 0.0)), 4) for c in reward_group.get("candidates", [])],
                "elapsed_s": round(time.time() - t_row, 2),
            }, ensure_ascii=False))

    print(f"[done] rows={len(rows)} output={out_path} elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
