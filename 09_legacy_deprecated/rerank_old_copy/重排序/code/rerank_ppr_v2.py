#!/usr/bin/env python3
"""
PPR v2: Retrieval-Anchored Personalized PageRank for Multimodal Reranking.

Key difference from PPR v1:
  - Restart vector is initialized from FIRST-STAGE retrieval scores (not discarded!)
  - Optionally blends first-stage score with fresh query-candidate embedding similarity
  - Graph edges still use Qwen3-VL candidate-candidate similarity
  - Cross-modal pair edges bridge text and image candidates

This preserves the retrieval quality while allowing PPR to propagate relevance
through the candidate graph.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Candidate:
    source: str         # "text" or "image"
    rank: int           # original rank from retrieval
    first_stage_score: float  # weighted_score or raw_score
    text: str
    doc_name: str
    page_idx: Optional[int]
    doc_id: str = ""
    sample_id: str = ""
    group_id: str = ""
    image_path: str = ""
    image_id: str = ""


@dataclass
class QueryBundle:
    index: int
    query: str
    options: Dict[str, str]
    text_candidates: List[Candidate]
    image_candidates: List[Candidate]


# ─── Encoder ─────────────────────────────────────────────────────────────────

class Qwen3VLEncoder:
    def __init__(self, model_path: str, device: str, batch_size: int):
        self.device = device
        self.batch_size = batch_size
        print(f"[Encoder] Loading {model_path} on {device} ...")
        self.model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
        print(f"[Encoder] Ready.")

    def encode_query(self, query: str) -> np.ndarray:
        emb = self.model.encode(
            [query],
            prompt_name="query",
            batch_size=1,
            show_progress_bar=False,
        )
        return l2_normalize(emb[0:1])[0]

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        emb = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        return l2_normalize(emb)

    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        if not image_paths:
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)
        emb = self.model.encode(
            [{"image": p} for p in image_paths],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return l2_normalize(emb)


# ─── Utilities ───────────────────────────────────────────────────────────────

def l2_normalize(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return emb / norms


def normalize_match_text(text: str) -> str:
    if not text:
        return ""
    return "".join(text.split())


def get_first_stage_score(item: dict) -> float:
    if "weighted_score" in item and item["weighted_score"] is not None:
        return float(item["weighted_score"])
    if "raw_score" in item and item["raw_score"] is not None:
        return float(item["raw_score"])
    if "score" in item and item["score"] is not None:
        return float(item["score"])
    return 0.0


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    mn = scores.min()
    mx = scores.max()
    if mx - mn < 1e-9:
        return np.full_like(scores, 0.5)
    return (scores - mn) / (mx - mn)


# ─── Core PPR v2 ─────────────────────────────────────────────────────────────

def build_adjacency(
    candidate_embs: np.ndarray,
    candidates: Sequence[Candidate],
    *,
    top_m_neighbors: int = 5,
    pair_boost: float = 0.15,
    pair_min_weight: float = 0.45,
) -> np.ndarray:
    """Build adjacency matrix from candidate embeddings + cross-modal pair edges."""
    sim = candidate_embs @ candidate_embs.T
    sim = np.maximum(sim, 0.0)
    np.fill_diagonal(sim, 0.0)

    n = sim.shape[0]
    adjacency = np.zeros_like(sim, dtype=np.float32)

    for i in range(n):
        row = sim[i]
        if top_m_neighbors >= n:
            neighbor_idx = np.argsort(-row)
        else:
            neighbor_idx = np.argpartition(-row, min(top_m_neighbors, n - 1))[:top_m_neighbors]
            neighbor_idx = neighbor_idx[np.argsort(-row[neighbor_idx])]
        for j in neighbor_idx:
            if i == j or row[j] <= 0:
                continue
            adjacency[i, j] = float(row[j])

    # Symmetrize
    adjacency = np.maximum(adjacency, adjacency.T)

    # Cross-modal pair edges
    normalized_texts = [normalize_match_text(c.text) for c in candidates]
    for i in range(n):
        for j in range(i + 1, n):
            if candidates[i].source == candidates[j].source:
                continue
            pair_matched = False
            if candidates[i].group_id and candidates[j].group_id and candidates[i].group_id == candidates[j].group_id:
                pair_matched = True
            elif normalized_texts[i] and normalized_texts[i] == normalized_texts[j]:
                pair_matched = True
            if pair_matched:
                boosted = max(float(adjacency[i, j]) + pair_boost, pair_min_weight)
                boosted = min(boosted, 1.0)
                adjacency[i, j] = boosted
                adjacency[j, i] = boosted

    return adjacency


def build_restart_vector(
    candidates: Sequence[Candidate],
    candidate_embs: np.ndarray,
    query_emb: np.ndarray,
    *,
    beta: float = 0.7,
) -> np.ndarray:
    """
    Build PPR restart vector blending first-stage scores with query-candidate similarity.
    
    restart[i] = beta * first_stage_norm[i] + (1-beta) * query_cand_sim_norm[i]
    
    beta=1.0: purely first-stage scores (trust retrieval completely)
    beta=0.0: purely embedding similarity (original PPR v1 behavior)
    beta=0.7: recommended - retrieval-anchored with some embedding signal
    """
    n = len(candidates)
    
    # First-stage scores
    first_stage = np.array([c.first_stage_score for c in candidates], dtype=np.float32)
    first_stage_norm = normalize_scores(first_stage)
    
    # Fresh query-candidate similarity from embeddings
    query_sim = np.maximum(candidate_embs @ query_emb.astype(np.float32), 0.0)
    query_sim_norm = normalize_scores(query_sim)
    
    # Blend
    restart = beta * first_stage_norm + (1.0 - beta) * query_sim_norm
    
    # Normalize to probability distribution
    total = restart.sum()
    if total < 1e-8:
        restart = np.ones(n, dtype=np.float32) / n
    else:
        restart = restart / total
    
    return restart


def run_ppr(
    adjacency: np.ndarray,
    restart: np.ndarray,
    *,
    alpha: float = 0.7,
    iters: int = 10,
) -> np.ndarray:
    """Run Personalized PageRank."""
    row_sums = adjacency.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    transition = adjacency / row_sums
    scores = restart.astype(np.float32).copy()
    for _ in range(iters):
        scores = alpha * restart + (1.0 - alpha) * (transition.T @ scores)
    return scores


def greedy_select(
    candidates: Sequence[Candidate],
    ppr_scores: np.ndarray,
    candidate_embs: np.ndarray,
    *,
    select_k: int = 3,
    redundancy_weight: float = 0.3,
    pair_complete_bonus: float = 0.1,
    min_gain: float = 0.01,
) -> List[int]:
    """Greedily select top-k diverse candidates."""
    selected: List[int] = []
    remaining = list(np.argsort(-ppr_scores))

    while remaining and len(selected) < select_k:
        best_idx = None
        best_gain = -math.inf
        for idx in remaining:
            redundancy = 0.0
            if selected:
                redundancy = float(np.max(candidate_embs[idx] @ candidate_embs[selected].T))

            pair_bonus = 0.0
            for sel_idx in selected:
                if (candidates[idx].source != candidates[sel_idx].source and
                    candidates[idx].group_id and candidates[sel_idx].group_id and
                    candidates[idx].group_id == candidates[sel_idx].group_id):
                    pair_bonus = pair_complete_bonus
                    break

            gain = float(ppr_scores[idx]) - redundancy_weight * redundancy + pair_bonus
            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx is None:
            break
        if best_gain < min_gain and selected:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    # Fallback: fill remaining slots by PPR score
    if len(selected) < select_k:
        for idx in np.argsort(-ppr_scores):
            if idx in selected:
                continue
            selected.append(int(idx))
            if len(selected) >= select_k:
                break
    return selected[:select_k]


# ─── Bundle Processing ───────────────────────────────────────────────────────

def process_bundle(
    bundle: QueryBundle,
    encoder: Qwen3VLEncoder,
    *,
    top_m_neighbors: int = 5,
    pair_boost: float = 0.15,
    pair_min_weight: float = 0.45,
    alpha: float = 0.7,
    ppr_iters: int = 10,
    select_k: int = 3,
    redundancy_weight: float = 0.3,
    pair_complete_bonus: float = 0.1,
    min_gain: float = 0.01,
    beta: float = 0.7,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Process a single query bundle through PPR v2."""
    
    # Prepare candidates
    image_candidates = [c for c in bundle.image_candidates if c.image_path]
    candidates: List[Candidate] = list(bundle.text_candidates) + image_candidates

    if not candidates:
        return (
            {"index": bundle.index, "retrieval": []},
            {"index": bundle.index, "query": bundle.query, "candidates": [], "selected": []},
        )

    # Encode
    text_texts = [c.text for c in bundle.text_candidates]
    image_paths = [c.image_path for c in image_candidates]

    query_emb = encoder.encode_query(bundle.query)
    text_embs = encoder.encode_texts(text_texts)
    image_embs = encoder.encode_images(image_paths)

    candidate_embs = np.concatenate([text_embs, image_embs], axis=0).astype(np.float32)
    candidate_embs = l2_normalize(candidate_embs)

    # Build restart vector (KEY CHANGE: uses first-stage scores!)
    restart = build_restart_vector(candidates, candidate_embs, query_emb, beta=beta)

    # Build graph and run PPR
    adjacency = build_adjacency(
        candidate_embs, candidates,
        top_m_neighbors=top_m_neighbors,
        pair_boost=pair_boost,
        pair_min_weight=pair_min_weight,
    )
    ppr_scores = run_ppr(adjacency, restart, alpha=alpha, iters=ppr_iters)

    # Select top-k
    selected = greedy_select(
        candidates, ppr_scores, candidate_embs,
        select_k=select_k,
        redundancy_weight=redundancy_weight,
        pair_complete_bonus=pair_complete_bonus,
        min_gain=min_gain,
    )

    # Build output
    retrieval_items = []
    for rank, idx in enumerate(selected, start=1):
        c = candidates[idx]
        retrieval_items.append({
            "source": c.source,
            "rank": rank,
            "text": c.text,
            "doc_name": c.doc_name,
            "page_idx": c.page_idx,
            "image_path": c.image_path,
        })

    # Debug info
    debug_record = {
        "index": bundle.index,
        "query": bundle.query,
        "beta": beta,
        "num_candidates": len(candidates),
        "selected_indices": [int(x) for x in selected],
        "candidates": [
            {
                "idx": i,
                "source": c.source,
                "rank_original": c.rank,
                "first_stage_score": round(float(c.first_stage_score), 4),
                "restart_weight": round(float(restart[i]), 6),
                "ppr_score": round(float(ppr_scores[i]), 6),
                "selected": i in selected,
                "text": c.text[:80],
            }
            for i, c in enumerate(candidates)
        ],
    }

    return {"index": bundle.index, "retrieval": retrieval_items}, debug_record


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_bundles(retrieval_export_jsonl: str, limit: Optional[int] = None) -> List[QueryBundle]:
    """Load query bundles from retrieval_export.jsonl."""
    bundles = []
    with open(retrieval_export_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample = record.get("sample", {})
            retrieval = record.get("retrieval", {})

            question = sample.get("question", "")
            options = sample.get("options", {})
            index = sample.get("index", line_no)

            # Build full query with options
            opts_text = " ".join([f"{k}.{v}" for k, v in sorted(options.items())]) if options else ""
            query = f"{question} {opts_text}".strip() if opts_text else question

            text_candidates = []
            for i, item in enumerate(retrieval.get("text_top20", [])):
                text_candidates.append(Candidate(
                    source="text",
                    rank=item.get("rank", i + 1),
                    first_stage_score=get_first_stage_score(item),
                    text=item.get("text", ""),
                    doc_name=item.get("doc_name", ""),
                    page_idx=item.get("page_idx"),
                    doc_id=item.get("doc_id", ""),
                    sample_id=item.get("sample_id", ""),
                    group_id=item.get("group_id", ""),
                    image_path=item.get("image_path", ""),
                    image_id=item.get("image_id", ""),
                ))

            image_candidates = []
            for i, item in enumerate(retrieval.get("image_top20", [])):
                image_candidates.append(Candidate(
                    source="image",
                    rank=item.get("rank", i + 1),
                    first_stage_score=get_first_stage_score(item),
                    text=item.get("text", ""),
                    doc_name=item.get("doc_name", ""),
                    page_idx=item.get("page_idx"),
                    doc_id=item.get("doc_id", ""),
                    sample_id=item.get("sample_id", ""),
                    group_id=item.get("group_id", ""),
                    image_path=item.get("image_path", ""),
                    image_id=item.get("image_id", ""),
                ))

            bundles.append(QueryBundle(
                index=index,
                query=query,
                options=options,
                text_candidates=text_candidates,
                image_candidates=image_candidates,
            ))

            if limit and len(bundles) >= limit:
                break

    return bundles


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PPR v2: Retrieval-Anchored Multimodal Reranking")
    parser.add_argument("--retrieval-export-jsonl", required=True)
    parser.add_argument("--model-path", default="/mnt/data_1/yds/models/Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0, help="Skip first N bundles")
    parser.add_argument("--mixed-only", action="store_true", help="Only process queries with both text and image candidates")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--baseline-jsonl", default=None, help="Also output no-rerank baseline")
    parser.add_argument("--debug-jsonl", default=None)
    parser.add_argument("--overwrite", action="store_true")

    # PPR v2 parameters
    parser.add_argument("--beta", type=float, default=0.7,
                        help="Blend weight: beta*first_stage + (1-beta)*embedding_sim. Default 0.7")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="PPR teleport probability. Default 0.7")
    parser.add_argument("--ppr-iters", type=int, default=10)
    parser.add_argument("--select-k", type=int, default=3)
    parser.add_argument("--top-m-neighbors", type=int, default=5)
    parser.add_argument("--pair-boost", type=float, default=0.15)
    parser.add_argument("--pair-min-weight", type=float, default=0.45)
    parser.add_argument("--redundancy-weight", type=float, default=0.3)
    parser.add_argument("--pair-complete-bonus", type=float, default=0.1)
    parser.add_argument("--min-gain", type=float, default=0.01)

    args = parser.parse_args()

    # Load data
    print(f"[MAIN] Loading bundles from {args.retrieval_export_jsonl} ...")
    bundles = load_bundles(args.retrieval_export_jsonl, limit=None)  # load all first
    
    # Skip
    if args.skip > 0:
        bundles = bundles[args.skip:]
        print(f"[MAIN] Skipped first {args.skip} bundles")
    
    # Filter mixed-only
    if args.mixed_only:
        bundles = [b for b in bundles if b.text_candidates and b.image_candidates]
        print(f"[MAIN] Filtered to mixed-only: {len(bundles)} bundles")
    
    # Apply limit
    if args.limit:
        bundles = bundles[:args.limit]
    
    print(f"[MAIN] Processing {len(bundles)} bundles (skip={args.skip}, limit={args.limit}, mixed_only={args.mixed_only})")

    # Initialize encoder
    encoder = Qwen3VLEncoder(args.model_path, args.device, args.batch_size)

    # Prepare output
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    f_out = open(args.output_jsonl, "w", encoding="utf-8")
    f_debug = open(args.debug_jsonl, "w", encoding="utf-8") if args.debug_jsonl else None
    f_base = None
    if args.baseline_jsonl:
        os.makedirs(os.path.dirname(args.baseline_jsonl) or ".", exist_ok=True)
        f_base = open(args.baseline_jsonl, "w", encoding="utf-8")

    t0 = time.time()
    for i, bundle in enumerate(bundles):
        try:
            export_record, debug_record = process_bundle(
                bundle, encoder,
                top_m_neighbors=args.top_m_neighbors,
                pair_boost=args.pair_boost,
                pair_min_weight=args.pair_min_weight,
                alpha=args.alpha,
                ppr_iters=args.ppr_iters,
                select_k=args.select_k,
                redundancy_weight=args.redundancy_weight,
                pair_complete_bonus=args.pair_complete_bonus,
                min_gain=args.min_gain,
                beta=args.beta,
            )
        except Exception as e:
            print(f"[ERROR] bundle {bundle.index}: {e}")
            traceback.print_exc()
            export_record = {"index": bundle.index, "retrieval": []}
            debug_record = {"index": bundle.index, "error": str(e)}

        f_out.write(json.dumps(export_record, ensure_ascii=False) + "\n")
        if f_debug:
            f_debug.write(json.dumps(debug_record, ensure_ascii=False) + "\n")

        # Baseline: just take original top-k by first_stage_score
        if f_base:
            all_cands = list(bundle.text_candidates) + [c for c in bundle.image_candidates if c.image_path]
            all_cands.sort(key=lambda c: c.first_stage_score, reverse=True)
            baseline_items = []
            for rank, c in enumerate(all_cands[:args.select_k], start=1):
                baseline_items.append({
                    "source": c.source, "rank": rank, "text": c.text,
                    "doc_name": c.doc_name, "page_idx": c.page_idx,
                    "image_path": c.image_path,
                })
            f_base.write(json.dumps({"index": bundle.index, "retrieval": baseline_items}, ensure_ascii=False) + "\n")

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            qps = (i + 1) / elapsed
            print(f"[PROG] {i+1}/{len(bundles)} done, {qps:.2f} q/s, elapsed {elapsed:.0f}s")

    f_out.close()
    if f_debug:
        f_debug.close()
    if f_base:
        f_base.close()

    elapsed = time.time() - t0
    print(f"[DONE] {len(bundles)} bundles in {elapsed:.1f}s ({len(bundles)/elapsed:.2f} q/s)")
    print(f"[DONE] Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
