#!/usr/bin/env python3
"""Per-round trajectory runtime for Stage-2 SFT trajectory generation.

Runs the multi-round control loop for a single question:

    round r:
        retrieve(query, image)  ->  top-k candidate window
        [optionally prepend a breadcrumb (search_history) memory item]
        agent.decide(keep/drop + ACCEPT/REWRITE + rewrite_query)
        record the round
        if ACCEPT or last round: stop
        else: suppress seen candidates, set next query = rewrite_query,
              and if this REWRITE round kept NO real evidence -> leave a
              breadcrumb (this round's query) for the next round.

Each round becomes one behaviour-cloning sample downstream. The breadcrumb gives
the next round a rewrite-direction signal when a rewrite produced nothing useful.
"""

from __future__ import annotations

import hashlib
from threading import Lock
from typing import Any, Dict, List, Optional

from evidence_selection import select_top_evidence

RECORD_KEYS = (
    "origin", "source", "score", "text", "doc_id", "doc_name",
    "page_idx", "block_id", "sample_id", "group_id", "image_path", "image_id",
)


def search_history_item(query: str) -> Dict[str, Any]:
    """Synthetic breadcrumb candidate injected at index 0 of the next round."""
    return {
        "origin": "search_history",
        "source": "memory",
        "score": 0.0,
        "text": (
            f"检索历史：上一轮检索 query = 「{query}」，但未保留任何有效证据。"
            "请据此调整检索方向，换用更具区分度的关键词（部位/病变/操作/选项特征）。"
        ),
        "doc_id": "",
        "doc_name": "search_history",
        "page_idx": None,
        "block_id": None,
        "sample_id": "",
        "group_id": "",
        "image_path": "",
        "image_id": "",
    }


def retained_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Mark a candidate as retained (M+): cross-round kept evidence."""
    ri = dict(item)
    ri["origin"] = "retained"
    return ri


def search_history_item_multi(failed_queries: List[str]) -> Dict[str, Any]:
    """M- breadcrumb: list all queries that led to poor retrieval."""
    if len(failed_queries) == 1:
        text = (
            f"检索历史：上一轮检索 query = 「{failed_queries[0]}」，但召回不佳。"
            "请据此调整检索方向，换用更具区分度的关键词（部位/病变/操作/选项特征）。"
        )
    else:
        lines = "\n".join(f"- 「{q}」" for q in failed_queries)
        text = (
            f"检索历史（M-）：已尝试以下 query 但召回不佳：\n{lines}\n"
            "请避免重复，换用更具区分度的关键词（部位/病变/操作/选项特征）。"
        )
    return {
        "origin": "search_history",
        "source": "memory",
        "score": 0.0,
        "text": text,
        "doc_id": "",
        "doc_name": "search_history",
        "page_idx": None,
        "block_id": None,
        "sample_id": "",
        "group_id": "",
        "image_path": "",
        "image_id": "",
    }


def build_round_candidates(
    retained: List[Dict[str, Any]],
    failed_queries: List[str],
    new_cands: List[Dict[str, Any]],
    *,
    no_mplus: bool = False,
    no_mminus: bool = False,
) -> tuple:
    """Build the candidate list for a round.

    Order: [retained M+] + [breadcrumb M-] + [new retrieval].
    Returns (cand, n_retained, n_bc) where n_bc is the breadcrumb count.
    """
    cand: List[Dict[str, Any]] = []
    n_retained = 0
    if retained and not no_mplus:
        cand += [retained_item(it) for it in retained]
        n_retained = len(cand)
    n_bc = 0
    if failed_queries and not no_mminus:
        cand += [search_history_item_multi(failed_queries)]
        n_bc = 1
    cand += new_cands
    return cand, n_retained, n_bc


def normalize_candidate(hit: Dict[str, Any]) -> Dict[str, Any]:
    item = {k: hit.get(k) for k in RECORD_KEYS}
    item["origin"] = str(hit.get("origin") or hit.get("source") or "retrieval")
    item["text"] = str(hit.get("text") or hit.get("content") or "")
    item["image_path"] = str(hit.get("image_path") or "")
    return item


def candidate_id(item: Dict[str, Any]) -> str:
    """Stable id for cross-round suppression of seen evidence."""
    sid = str(item.get("sample_id") or "")
    if sid:
        return f"s:{sid}"
    doc = str(item.get("doc_id") or item.get("doc_name") or "")
    page = item.get("page_idx")
    block = item.get("block_id")
    if doc:
        return f"d:{doc}:{page}:{block}"
    text = str(item.get("text") or "")
    if text:
        return "t:" + hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
    return ""


def run_trajectory(
    *,
    qid: str,
    query_type: str,
    original_query: str,
    question: str,
    options: Dict[str, Any],
    answer: str,
    answer_text: str,
    image_path: str,
    gold_source: Dict[str, Any],
    retriever: Any,
    agent: Any,
    max_rounds: int = 2,
    select_k: int = 12,
    text_k: int = 20,
    image_k: int = 20,
    retriever_lock: Optional[Lock] = None,
) -> Dict[str, Any]:
    suppressed: set = set()
    retained: List[Dict[str, Any]] = []   # M+ : cross-round kept evidence
    failed_queries: List[str] = []         # M- : queries that led to poor retrieval
    query = original_query
    rounds: List[Dict[str, Any]] = []
    status = "ok"

    for r in range(max(int(max_rounds), 1)):
        if retriever_lock is not None:
            with retriever_lock:
                retrieval = retriever.retrieve(query, image_path, text_k=text_k, image_k=image_k)
        else:
            retrieval = retriever.retrieve(query, image_path, text_k=text_k, image_k=image_k)

        pool = [h for h in retrieval.get("combined", [])
                if candidate_id(normalize_candidate(h)) not in suppressed]
        top = select_top_evidence(pool, select_k=select_k)
        new_cands = [normalize_candidate(h) for h in top]

        # build candidate list: [retained M+] + [breadcrumb M-] + [new retrieval]
        cand, n_retained, n_bc = build_round_candidates(
            retained, failed_queries, new_cands)

        try:
            agent_out = agent.decide(
                qid=qid,
                query_type=query_type,
                original_query=original_query,
                current_query=query,
                question=question,
                options=options,
                candidates=cand,
                image_path=image_path,
            )
        except Exception as exc:  # keep the trajectory, mark failure
            status = f"agent_error:{type(exc).__name__}"
            rounds.append({
                "round_idx": r,
                "query": query,
                "candidates": cand,
                "n_retained": n_retained,
                "n_breadcrumb": n_bc,
                "agent": {"keep": [], "drop": list(range(len(cand))), "action": "ACCEPT", "rewrite_query": "", "reason": f"error: {exc}"},
                "is_final": True,
                "error": str(exc),
            })
            break

        is_final = (r == max_rounds - 1) or (agent_out["action"] == "ACCEPT")
        rounds.append({
            "round_idx": r,
            "query": query,
            "candidates": cand,
            "n_retained": n_retained,
            "n_breadcrumb": n_bc,
            "agent": agent_out,
            "is_final": is_final,
        })

        # suppress only new retrieval candidates (retained were suppressed in prior rounds)
        for item in new_cands:
            cid = candidate_id(item)
            if cid:
                suppressed.add(cid)

        if is_final:
            break

        # M- : record query on REWRITE (retrieval was insufficient)
        if agent_out["action"] == "REWRITE":
            failed_queries.append(query)

        # M+ : update retained = kept candidates excluding breadcrumb
        bc_start = n_retained
        bc_end = n_retained + n_bc
        kept = agent_out.get("keep", [])
        retained = [cand[i] for i in kept
                    if i < len(cand) and not (bc_start <= i < bc_end)]

        query = agent_out.get("rewrite_query") or query

    return {
        "qid": qid,
        "query_type": query_type,
        "question": question,
        "options": options,
        "answer": answer,
        "answer_text": answer_text,
        "query_image_path": image_path,
        "gold_source": gold_source,
        "original_query": original_query,
        "num_rounds": len(rounds),
        "final_action": rounds[-1]["agent"]["action"] if rounds else "",
        "status": status,
        "rounds": rounds,
    }
