#!/usr/bin/env python3
"""
VCMS: Vision-Guided Cluster-Medoid Selection for Multimodal Reranking.

Core idea:
  1. Encode all candidates into unified multimodal space (Qwen3-VL-Embedding-2B)
  2. Encode query image via vision tower to get visual query embedding
  3. Compute cross-modal distance matrix (query→candidates, candidate↔candidate)
  4. Hierarchical clustering on candidates
  5. Rank clusters by query affinity (average distance to query)
  6. Select medoid from top-M clusters as diverse, representative evidence
  7. Optional: complement with nearest non-selected candidate

Compared to PPR v2:
  - No graph propagation (simpler, faster)
  - Explicit clustering for diversity guarantee
  - Medoid = maximally representative within each cluster
  - Vision tower as cross-modal scoring bridge
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sentence_transformers import SentenceTransformer


# ─── Data structures (same as PPR v2) ────────────────────────────────────────

@dataclass
class Candidate:
    source: str         # "text" or "image"
    rank: int           # original rank from retrieval
    first_stage_score: float
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
    query_image: str    # query image path (if any)
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
        print(f"[Encoder] Ready. Dim={self.model.get_sentence_embedding_dimension()}")

    def encode_query(self, query: str) -> np.ndarray:
        emb = self.model.encode([query], prompt_name="query", batch_size=1, show_progress_bar=False)
        return l2_normalize(emb[0:1])[0]

    def encode_query_image(self, image_path: str) -> np.ndarray:
        """Encode query image via vision tower."""
        emb = self.model.encode([{"image": image_path}], batch_size=1, show_progress_bar=False)
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


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    mn = scores.min()
    mx = scores.max()
    if mx - mn < 1e-9:
        return np.full_like(scores, 0.5)
    return (scores - mn) / (mx - mn)


def get_first_stage_score(item: dict) -> float:
    if "weighted_score" in item and item["weighted_score"] is not None:
        return float(item["weighted_score"])
    if "raw_score" in item and item["raw_score"] is not None:
        return float(item["raw_score"])
    if "score" in item and item["score"] is not None:
        return float(item["score"])
    return 0.0


# ─── Core VCMS ───────────────────────────────────────────────────────────────

def vcms_select(
    candidates: Sequence[Candidate],
    candidate_embs: np.ndarray,
    query_emb: np.ndarray,
    query_image_emb: Optional[np.ndarray],
    *,
    n_clusters: int = 3,
    select_from_top_m: int = 2,
    select_k: int = 3,
    use_first_stage: bool = True,
    first_stage_weight: float = 0.3,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Vision-Guided Cluster-Medoid Selection.
    
    Args:
        candidates: list of candidates
        candidate_embs: (N, D) normalized embeddings
        query_emb: (D,) text query embedding
        query_image_emb: (D,) visual query embedding (None if no query image)
        n_clusters: number of clusters for hierarchical clustering
        select_from_top_m: number of top clusters to select medoids from
        select_k: total evidence items to select
        use_first_stage: whether to incorporate first-stage scores in cluster ranking
        first_stage_weight: weight for first-stage score in scoring
    
    Returns:
        selected_indices: list of selected candidate indices
        debug_info: dict with clustering details
    """
    n = len(candidates)
    if n == 0:
        return [], {}
    if n <= select_k:
        return list(range(n)), {"note": "n <= select_k, return all"}

    # Step 1: Compute query-candidate similarity
    # Use visual query embedding if available, otherwise text query embedding
    if query_image_emb is not None:
        # Cross-modal: vision tower → candidate similarities
        vision_sim = np.maximum(candidate_embs @ query_image_emb.astype(np.float32), 0.0)
        text_sim = np.maximum(candidate_embs @ query_emb.astype(np.float32), 0.0)
        # Combine visual and text similarities
        query_sim = 0.6 * vision_sim + 0.4 * text_sim
    else:
        # Text-only query: fall back to text similarity
        query_sim = np.maximum(candidate_embs @ query_emb.astype(np.float32), 0.0)

    # Step 2: Compute pairwise distance matrix between candidates
    # Use cosine distance: 1 - cos_sim
    cos_sim_matrix = candidate_embs @ candidate_embs.T
    cos_sim_matrix = np.clip(cos_sim_matrix, -1.0, 1.0)
    dist_matrix = 1.0 - cos_sim_matrix
    np.fill_diagonal(dist_matrix, 0.0)
    
    # Step 3: Hierarchical clustering
    actual_n_clusters = min(n_clusters, n)
    
    if n <= 2:
        # Too few candidates, just rank by similarity
        ranking = np.argsort(-query_sim)
        selected = ranking[:select_k].tolist()
        return selected, {"note": "n<=2, direct top-k", "query_sim": query_sim.tolist()}
    
    # Convert to condensed distance for linkage
    condensed_dist = squareform(dist_matrix, checks=False)
    # Replace NaN/inf with max distance
    condensed_dist = np.nan_to_num(condensed_dist, nan=1.0, posinf=1.0, neginf=0.0)
    
    Z = linkage(condensed_dist, method='ward')
    cluster_labels = fcluster(Z, t=actual_n_clusters, criterion='maxclust')
    # cluster_labels: 1-indexed, convert to 0-indexed
    cluster_labels = cluster_labels - 1

    # Step 4: Compute cluster scores (affinity to query)
    cluster_info = {}
    for k in range(actual_n_clusters):
        members = np.where(cluster_labels == k)[0]
        if len(members) == 0:
            continue
        
        # Average query similarity of cluster members
        avg_query_sim = float(query_sim[members].mean())
        
        # Average first-stage score (if using)
        if use_first_stage:
            fs_scores = np.array([candidates[i].first_stage_score for i in members])
            avg_fs = float(normalize_scores(fs_scores).mean()) if len(fs_scores) > 1 else float(fs_scores[0])
        else:
            avg_fs = 0.0
        
        # Combined cluster score
        cluster_score = (1.0 - first_stage_weight) * avg_query_sim + first_stage_weight * avg_fs
        
        # Find medoid: member with minimum average distance to all other members
        if len(members) == 1:
            medoid_idx = int(members[0])
        else:
            sub_dist = dist_matrix[np.ix_(members, members)]
            avg_dists = sub_dist.mean(axis=1)
            medoid_local = np.argmin(avg_dists)
            medoid_idx = int(members[medoid_local])
        
        cluster_info[k] = {
            "members": [int(m) for m in members],
            "size": len(members),
            "avg_query_sim": avg_query_sim,
            "avg_fs_score": avg_fs,
            "cluster_score": cluster_score,
            "medoid_idx": medoid_idx,
        }
    
    # Step 5: Rank clusters by score, select medoids from top-M clusters
    sorted_clusters = sorted(cluster_info.items(), key=lambda x: x[1]["cluster_score"], reverse=True)
    
    selected = []
    for cluster_id, info in sorted_clusters[:select_from_top_m]:
        medoid = info["medoid_idx"]
        if medoid not in selected:
            selected.append(medoid)
    
    # Step 6: Fill remaining slots
    # Strategy: from remaining candidates, pick by combined score (query_sim + first_stage)
    if len(selected) < select_k:
        # Compute combined ranking score for non-selected candidates
        fs_all = np.array([c.first_stage_score for c in candidates], dtype=np.float32)
        fs_norm = normalize_scores(fs_all)
        qs_norm = normalize_scores(query_sim)
        combined = 0.5 * qs_norm + 0.5 * fs_norm
        
        # From unselected candidates, prefer those NOT in already-selected clusters
        selected_cluster_ids = set()
        for idx in selected:
            selected_cluster_ids.add(int(cluster_labels[idx]))
        
        # First try: candidates from non-selected clusters
        remaining_scores = []
        for idx in range(n):
            if idx in selected:
                continue
            bonus = 0.1 if int(cluster_labels[idx]) not in selected_cluster_ids else 0.0
            remaining_scores.append((idx, float(combined[idx]) + bonus))
        
        remaining_scores.sort(key=lambda x: x[1], reverse=True)
        for idx, _ in remaining_scores:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= select_k:
                break
    
    # Debug info
    debug = {
        "n_candidates": n,
        "n_clusters": actual_n_clusters,
        "has_query_image": query_image_emb is not None,
        "clusters": {str(k): {
            "size": info["size"],
            "score": round(info["cluster_score"], 4),
            "medoid": info["medoid_idx"],
        } for k, info in sorted_clusters},
        "selected": [int(x) for x in selected],
        "query_sim_selected": [round(float(query_sim[i]), 4) for i in selected],
    }
    
    return selected[:select_k], debug


# ─── Bundle Processing ───────────────────────────────────────────────────────

def process_bundle(
    bundle: QueryBundle,
    encoder: Qwen3VLEncoder,
    *,
    n_clusters: int = 3,
    select_from_top_m: int = 2,
    select_k: int = 3,
    use_first_stage: bool = True,
    first_stage_weight: float = 0.3,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Process a single query bundle through VCMS."""
    
    # Prepare candidates
    image_candidates = [c for c in bundle.image_candidates if c.image_path]
    candidates: List[Candidate] = list(bundle.text_candidates) + image_candidates

    if not candidates:
        return (
            {"index": bundle.index, "retrieval": []},
            {"index": bundle.index, "query": bundle.query, "vcms": {}, "selected": []},
        )

    # Encode candidates
    text_texts = [c.text for c in bundle.text_candidates]
    image_paths = [c.image_path for c in image_candidates]

    query_emb = encoder.encode_query(bundle.query)
    text_embs = encoder.encode_texts(text_texts)
    image_embs = encoder.encode_images(image_paths)

    candidate_embs = np.concatenate([text_embs, image_embs], axis=0).astype(np.float32)
    candidate_embs = l2_normalize(candidate_embs)

    # Encode query image (if available) - THIS IS THE KEY DIFFERENCE
    query_image_emb = None
    if bundle.query_image and os.path.exists(bundle.query_image):
        query_image_emb = encoder.encode_query_image(bundle.query_image)

    # VCMS selection
    selected, vcms_debug = vcms_select(
        candidates, candidate_embs, query_emb, query_image_emb,
        n_clusters=n_clusters,
        select_from_top_m=select_from_top_m,
        select_k=select_k,
        use_first_stage=use_first_stage,
        first_stage_weight=first_stage_weight,
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

    # Debug record
    debug_record = {
        "index": bundle.index,
        "query": bundle.query[:100],
        "has_query_image": bundle.query_image != "",
        "num_candidates": len(candidates),
        "num_text": len(bundle.text_candidates),
        "num_image": len(image_candidates),
        "vcms": vcms_debug,
        "selected_indices": [int(x) for x in selected],
        "candidates_summary": [
            {
                "idx": i,
                "source": c.source,
                "rank_original": c.rank,
                "first_stage_score": round(float(c.first_stage_score), 4),
                "selected": i in selected,
                "text": c.text[:60],
            }
            for i, c in enumerate(candidates)
        ],
    }

    return {"index": bundle.index, "retrieval": retrieval_items}, debug_record


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_bundles(retrieval_export_jsonl: str, limit: Optional[int] = None,
                 skip: int = 0, mixed_only: bool = False) -> List[QueryBundle]:
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
            # Query image path
            query_image = sample.get("image_path", "") or sample.get("image", "")

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
                query_image=query_image,
                options=options,
                text_candidates=text_candidates,
                image_candidates=image_candidates,
            ))

    # Apply skip
    if skip > 0:
        bundles = bundles[skip:]
    
    # Filter mixed-only
    if mixed_only:
        bundles = [b for b in bundles if b.text_candidates and b.image_candidates]
    
    # Apply limit
    if limit:
        bundles = bundles[:limit]

    return bundles


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VCMS: Vision-Guided Cluster-Medoid Selection")
    parser.add_argument("--retrieval-export-jsonl", required=True)
    parser.add_argument("--model-path", default="/mnt/data_1/yds/models/Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--mixed-only", action="store_true")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--baseline-jsonl", default=None)
    parser.add_argument("--debug-jsonl", default=None)

    # VCMS parameters
    parser.add_argument("--n-clusters", type=int, default=3, help="Number of clusters. Default 3")
    parser.add_argument("--select-from-top-m", type=int, default=2, help="Select medoids from top-M clusters. Default 2")
    parser.add_argument("--select-k", type=int, default=3, help="Total evidence items to select. Default 3")
    parser.add_argument("--use-first-stage", action="store_true", default=True)
    parser.add_argument("--no-first-stage", action="store_true", help="Disable first-stage score integration")
    parser.add_argument("--first-stage-weight", type=float, default=0.3,
                        help="Weight for first-stage score in cluster ranking. Default 0.3")

    args = parser.parse_args()
    if args.no_first_stage:
        args.use_first_stage = False

    # Load data
    print(f"[MAIN] Loading bundles from {args.retrieval_export_jsonl} ...")
    bundles = load_bundles(
        args.retrieval_export_jsonl,
        limit=args.limit,
        skip=args.skip,
        mixed_only=args.mixed_only,
    )
    print(f"[MAIN] Processing {len(bundles)} bundles")
    print(f"[MAIN] VCMS params: n_clusters={args.n_clusters}, top_m={args.select_from_top_m}, "
          f"select_k={args.select_k}, first_stage_weight={args.first_stage_weight}")

    # Count queries with images
    img_query_count = sum(1 for b in bundles if b.query_image and os.path.exists(b.query_image))
    print(f"[MAIN] Queries with valid image: {img_query_count}/{len(bundles)}")

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
                n_clusters=args.n_clusters,
                select_from_top_m=args.select_from_top_m,
                select_k=args.select_k,
                use_first_stage=args.use_first_stage,
                first_stage_weight=args.first_stage_weight,
            )
        except Exception as e:
            print(f"[ERROR] bundle {bundle.index}: {e}")
            traceback.print_exc()
            export_record = {"index": bundle.index, "retrieval": []}
            debug_record = {"index": bundle.index, "error": str(e)}

        f_out.write(json.dumps(export_record, ensure_ascii=False) + "\n")
        if f_debug:
            f_debug.write(json.dumps(debug_record, ensure_ascii=False) + "\n")

        # Baseline
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

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            qps = (i + 1) / elapsed
            print(f"[PROG] {i+1}/{len(bundles)} done, {qps:.2f} q/s, elapsed {elapsed:.0f}s")

    f_out.close()
    if f_debug:
        f_debug.close()
    if f_base:
        f_base.close()

    elapsed = time.time() - t0
    print(f"\n[DONE] {len(bundles)} bundles in {elapsed:.1f}s ({len(bundles)/elapsed:.2f} q/s)")
    print(f"[DONE] Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
