#!/usr/bin/env python3
"""PPR rerank for EndoBench multimodal retrieval results.

Pipeline
--------
1. Read existing text/image retrieval results.
2. Use Qwen3-VL-Embedding-2B to encode:
   - query text
   - text candidates
   - raw image candidates
3. Build a candidate graph with:
   - candidate-candidate similarity edges
   - boosted text-image edges when candidates can be matched
4. Run query-aware PPR
5. Greedily select non-redundant evidence
6. Export retrieval JSONL for downstream RAG evaluation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_TEXT_JSONL = "data/legacy/endobench_text_to_text_hybrid_recall_breakdown.jsonl"
DEFAULT_IMAGE_JSONL = "data/legacy/endobench_image_to_image_recall_breakdown.jsonl"
DEFAULT_RETRIEVAL_EXPORT_JSONL = "data/retrieval_export.jsonl"
DEFAULT_MAIN_DB = "data/multimodal_samples.db"
DEFAULT_MODEL_PATH = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_OUTPUT_JSONL = "output/endobench_multimodal_ppr_top3.jsonl"
DEFAULT_ERROR_JSONL = "output/endobench_multimodal_ppr_top3_errors.jsonl"
DEFAULT_PROGRESS_LOG = "output/endobench_multimodal_ppr_top3_progress.log"


@dataclass
class Candidate:
    source: str
    rank: int
    first_stage_score: float
    text: str
    doc_name: str
    page_idx: Optional[int]
    doc_id: str = ""
    block_id: Optional[int] = None
    sample_id: str = ""
    group_id: str = ""
    image_path: str = ""
    image_id: str = ""


@dataclass
class QueryBundle:
    index: int
    query: str
    text_candidates: List[Candidate]
    image_candidates: List[Candidate]


def log_message(message: str, progress_handle: Optional[Any] = None) -> None:
    print(message, flush=True)
    if progress_handle is None:
        return
    progress_handle.write(message + "\n")
    progress_handle.flush()
    os.fsync(progress_handle.fileno())


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


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


def load_block_mapping(db_path: Path) -> Dict[Tuple[str, int], Dict[str, str]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT doc_id, block_id, sample_id, group_id
            FROM multimodal_samples
            WHERE block_id IS NOT NULL
            """
        ).fetchall()
    finally:
        conn.close()

    mapping: Dict[Tuple[str, int], Dict[str, str]] = {}
    for row in rows:
        doc_id = str(row["doc_id"] or "")
        block_id = row["block_id"]
        if not doc_id or block_id is None:
            continue
        key = (doc_id, int(block_id))
        if key not in mapping:
            mapping[key] = {
                "sample_id": str(row["sample_id"] or ""),
                "group_id": str(row["group_id"] or ""),
            }
    return mapping


def load_sample_mapping(db_path: Path) -> Dict[str, Dict[str, str]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT sample_id, group_id
            FROM multimodal_samples
            """
        ).fetchall()
    finally:
        conn.close()

    mapping: Dict[str, Dict[str, str]] = {}
    for row in rows:
        sample_id = str(row["sample_id"] or "")
        if not sample_id:
            continue
        mapping[sample_id] = {
            "group_id": str(row["group_id"] or ""),
        }
    return mapping


def candidate_score(raw: Dict[str, Any]) -> float:
    for key in ("weighted_score", "raw_score", "score"):
        value = raw.get(key)
        if value is not None:
            return float(value)
    return 0.0


def normalize_text_candidate(raw: Dict[str, Any], mapping: Dict[Tuple[str, int], Dict[str, str]]) -> Candidate:
    doc_id = str(raw.get("doc_id") or "")
    block_id = raw.get("block_id")
    resolved = {}
    if block_id is not None and doc_id:
        resolved = mapping.get((doc_id, int(block_id)), {})
    return Candidate(
        source="text",
        rank=int(raw.get("rank") or 0),
        first_stage_score=candidate_score(raw),
        text=str(raw.get("text") or ""),
        doc_name=str(raw.get("doc_name") or ""),
        page_idx=raw.get("page_idx"),
        doc_id=doc_id,
        block_id=int(block_id) if block_id is not None else None,
        sample_id=str(resolved.get("sample_id") or ""),
        group_id=str(resolved.get("group_id") or ""),
    )


def normalize_image_candidate(raw: Dict[str, Any], sample_mapping: Optional[Dict[str, Dict[str, str]]] = None) -> Candidate:
    sample_id = str(raw.get("sample_id") or "")
    resolved = (sample_mapping or {}).get(sample_id, {})
    return Candidate(
        source="image",
        rank=int(raw.get("rank") or 0),
        first_stage_score=candidate_score(raw),
        text=str(raw.get("text") or raw.get("content") or ""),
        doc_name=str(raw.get("doc_name") or ""),
        page_idx=raw.get("page_idx"),
        doc_id=str(raw.get("doc_id") or ""),
        sample_id=sample_id,
        group_id=str(raw.get("group_id") or resolved.get("group_id") or ""),
        image_path=str(raw.get("image_path") or ""),
        image_id=str(raw.get("image_id") or ""),
    )


def build_query_bundles_from_recall(
    text_rows: Sequence[Dict[str, Any]],
    image_rows: Sequence[Dict[str, Any]],
    block_mapping: Dict[Tuple[str, int], Dict[str, str]],
    sample_mapping: Dict[str, Dict[str, str]],
    *,
    text_topk: int,
    image_topk: int,
) -> List[QueryBundle]:
    image_by_index = {int(row["index"]): row for row in image_rows}
    bundles: List[QueryBundle] = []
    for text_row in text_rows:
        index = int(text_row["index"])
        image_row = image_by_index.get(index)
        if image_row is None:
            continue

        query = str(text_row.get("text_query") or text_row.get("question") or "").strip()
        text_candidates = [
            normalize_text_candidate(item, block_mapping)
            for item in list(text_row.get("dense_top20") or [])[:text_topk]
        ]
        image_candidates = [
            normalize_image_candidate(item, sample_mapping)
            for item in list(image_row.get("top20_image_hits") or [])[:image_topk]
        ]
        bundles.append(
            QueryBundle(
                index=index,
                query=query,
                text_candidates=text_candidates,
                image_candidates=image_candidates,
            )
        )
    return bundles


def build_query_bundles_from_retrieval_export(
    rows: Sequence[Dict[str, Any]],
    block_mapping: Dict[Tuple[str, int], Dict[str, str]],
    sample_mapping: Dict[str, Dict[str, str]],
    *,
    text_topk: int,
    image_topk: int,
) -> List[QueryBundle]:
    bundles: List[QueryBundle] = []
    for row in rows:
        sample = row.get("sample") or {}
        retrieval = row.get("retrieval") or {}
        index_value = sample.get("index", row.get("index"))
        if index_value is None:
            continue
        query = str(sample.get("question") or row.get("question") or "").strip()
        text_candidates = [
            normalize_text_candidate(item, block_mapping)
            for item in list(retrieval.get("text_top20") or [])[:text_topk]
        ]
        image_candidates = [
            normalize_image_candidate(item, sample_mapping)
            for item in list(retrieval.get("image_top20") or [])[:image_topk]
        ]
        bundles.append(
            QueryBundle(
                index=int(index_value),
                query=query,
                text_candidates=text_candidates,
                image_candidates=image_candidates,
            )
        )
    return bundles


def build_query_bundle_from_retrieval_export_row(
    row: Dict[str, Any],
    block_mapping: Dict[Tuple[str, int], Dict[str, str]],
    sample_mapping: Dict[str, Dict[str, str]],
    *,
    text_topk: int,
    image_topk: int,
) -> Optional[QueryBundle]:
    sample = row.get("sample") or {}
    retrieval = row.get("retrieval") or {}
    index_value = sample.get("index", row.get("index"))
    if index_value is None:
        return None
    query = str(sample.get("question") or row.get("question") or "").strip()
    text_candidates = [
        normalize_text_candidate(item, block_mapping)
        for item in list(retrieval.get("text_top20") or [])[:text_topk]
    ]
    image_candidates = [
        normalize_image_candidate(item, sample_mapping)
        for item in list(retrieval.get("image_top20") or [])[:image_topk]
    ]
    return QueryBundle(
        index=int(index_value),
        query=query,
        text_candidates=text_candidates,
        image_candidates=image_candidates,
    )


def iter_retrieval_export_bundles(
    path: Path,
    block_mapping: Dict[Tuple[str, int], Dict[str, str]],
    sample_mapping: Dict[str, Dict[str, str]],
    *,
    text_topk: int,
    image_topk: int,
    num_shards: int,
    shard_index: int,
    limit: int,
) -> Iterable[QueryBundle]:
    seen_valid_rows = 0
    for row in iter_jsonl(path):
        bundle = build_query_bundle_from_retrieval_export_row(
            row,
            block_mapping,
            sample_mapping,
            text_topk=text_topk,
            image_topk=image_topk,
        )
        if bundle is None:
            continue
        seen_valid_rows += 1
        if limit > 0 and seen_valid_rows > limit:
            break
        if bundle.index % num_shards != shard_index:
            continue
        yield bundle


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def normalize_match_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", "", text)
    return text


class Qwen3VLEncoder:
    def __init__(self, model_path: str, device: str, batch_size: int):
        self.model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
        self.batch_size = batch_size

    def encode_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [query],
            batch_size=1,
            prompt="Retrieve relevant medical multimodal evidence for the query.",
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0].astype(np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        valid_inputs = [{"image": path} for path in image_paths]
        if not valid_inputs:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        embeddings = self.model.encode(
            valid_inputs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


def build_adjacency(
    candidate_embs: np.ndarray,
    candidates: Sequence[Candidate],
    *,
    top_m_neighbors: int,
    pair_boost: float,
    pair_min_weight: float,
) -> np.ndarray:
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
            neighbor_idx = np.argpartition(-row, top_m_neighbors)[:top_m_neighbors]
            neighbor_idx = neighbor_idx[np.argsort(-row[neighbor_idx])]
        for j in neighbor_idx:
            if i == j or row[j] <= 0:
                continue
            adjacency[i, j] = float(row[j])

    # Symmetrize before pair boost.
    adjacency = np.maximum(adjacency, adjacency.T)

    normalized_texts = [normalize_match_text(item.text) for item in candidates]
    for i in range(n):
        for j in range(i + 1, n):
            left = candidates[i]
            right = candidates[j]
            if left.source == right.source:
                continue
            pair_matched = False
            if left.group_id and right.group_id and left.group_id == right.group_id:
                pair_matched = True
            elif normalized_texts[i] and normalized_texts[i] == normalized_texts[j]:
                pair_matched = True
            if not pair_matched:
                continue
            boosted = max(float(adjacency[i, j]) + pair_boost, pair_min_weight)
            boosted = min(boosted, 1.0)
            adjacency[i, j] = boosted
            adjacency[j, i] = boosted

    return adjacency


def run_ppr(
    adjacency: np.ndarray,
    restart: np.ndarray,
    *,
    alpha: float,
    iters: int,
) -> np.ndarray:
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
    select_k: int,
    redundancy_weight: float,
    pair_complete_bonus: float,
    min_gain: float,
) -> List[int]:
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
                left = candidates[idx]
                right = candidates[sel_idx]
                if (
                    left.group_id
                    and right.group_id
                    and left.group_id == right.group_id
                    and left.source != right.source
                ):
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

    if len(selected) < select_k:
        for idx in np.argsort(-ppr_scores):
            if idx in selected:
                continue
            selected.append(int(idx))
            if len(selected) >= select_k:
                break
    return selected[:select_k]


def build_export_items(candidates: Sequence[Candidate], chosen_indices: Sequence[int]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for rank, idx in enumerate(chosen_indices, start=1):
        candidate = candidates[idx]
        items.append(
            {
                "source": candidate.source,
                "rank": rank,
                "doc_name": candidate.doc_name,
                "page_idx": candidate.page_idx,
                "doc_id": candidate.doc_id,
                "block_id": candidate.block_id,
                "sample_id": candidate.sample_id,
                "group_id": candidate.group_id,
                "image_id": candidate.image_id,
                "image_path": candidate.image_path,
                "text": candidate.text,
            }
        )
    return items


def build_debug_record(
    bundle: QueryBundle,
    candidates: Sequence[Candidate],
    query_scores: np.ndarray,
    ppr_scores: np.ndarray,
    selected: Sequence[int],
) -> Dict[str, Any]:
    selected_set = set(selected)
    detail = []
    for idx, candidate in enumerate(candidates):
        detail.append(
            {
                "candidate_index": idx,
                "selected": idx in selected_set,
                "source": candidate.source,
                "doc_name": candidate.doc_name,
                "page_idx": candidate.page_idx,
                "doc_id": candidate.doc_id,
                "block_id": candidate.block_id,
                "sample_id": candidate.sample_id,
                "group_id": candidate.group_id,
                "first_stage_rank": candidate.rank,
                "first_stage_score": candidate.first_stage_score,
                "query_score": float(query_scores[idx]),
                "ppr_score": float(ppr_scores[idx]),
                "text": candidate.text,
                "image_path": candidate.image_path,
            }
        )
    return {
        "index": bundle.index,
        "query": bundle.query,
        "selected_candidate_indices": list(selected),
        "candidates": detail,
    }


def process_bundle(
    bundle: QueryBundle,
    encoder: Qwen3VLEncoder,
    *,
    top_m_neighbors: int,
    pair_boost: float,
    pair_min_weight: float,
    alpha: float,
    ppr_iters: int,
    select_k: int,
    redundancy_weight: float,
    pair_complete_bonus: float,
    min_gain: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    text_texts = [candidate.text for candidate in bundle.text_candidates]
    image_paths = [candidate.image_path for candidate in bundle.image_candidates if candidate.image_path]
    image_candidates = [candidate for candidate in bundle.image_candidates if candidate.image_path]

    candidates: List[Candidate] = list(bundle.text_candidates) + image_candidates
    if not candidates:
        export_record = {
            "index": bundle.index,
            "retrieval": [],
        }
        debug_record = {
            "index": bundle.index,
            "query": bundle.query,
            "selected_candidate_indices": [],
            "candidates": [],
        }
        return export_record, debug_record

    query_emb = encoder.encode_query(bundle.query)
    text_embs = encoder.encode_texts(text_texts)
    image_embs = encoder.encode_images(image_paths)

    candidate_embs = np.concatenate([text_embs, image_embs], axis=0).astype(np.float32)
    candidate_embs = l2_normalize(candidate_embs)

    query_sim = np.maximum(candidate_embs @ query_emb.astype(np.float32), 0.0)
    query_scores = query_sim / (float(query_sim.sum()) + 1e-8)

    adjacency = build_adjacency(
        candidate_embs,
        candidates,
        top_m_neighbors=top_m_neighbors,
        pair_boost=pair_boost,
        pair_min_weight=pair_min_weight,
    )
    ppr_scores = run_ppr(adjacency, query_scores, alpha=alpha, iters=ppr_iters)
    selected = greedy_select(
        candidates,
        ppr_scores,
        candidate_embs,
        select_k=select_k,
        redundancy_weight=redundancy_weight,
        pair_complete_bonus=pair_complete_bonus,
        min_gain=min_gain,
    )

    export_record = {
        "index": bundle.index,
        "retrieval": build_export_items(candidates, selected),
    }
    debug_record = build_debug_record(bundle, candidates, query_scores, ppr_scores, selected)
    return export_record, debug_record


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_processed_indices(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            index = row.get("index")
            if index is None:
                continue
            try:
                done.add(int(index))
            except (TypeError, ValueError):
                continue
    return done


def write_jsonl_record(handle: Any, record: Dict[str, Any]) -> None:
    handle.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multimodal PPR rerank for EndoBench retrieval results.")
    parser.add_argument(
        "--retrieval-export-jsonl",
        default=DEFAULT_RETRIEVAL_EXPORT_JSONL,
        help="Single JSONL exported by rag_retrieval_export. Set to empty string to use legacy recall JSONLs.",
    )
    parser.add_argument("--text-jsonl", default=DEFAULT_TEXT_JSONL, help="Legacy text recall JSONL.")
    parser.add_argument("--image-jsonl", default=DEFAULT_IMAGE_JSONL, help="Legacy image recall JSONL.")
    parser.add_argument("--main-db", default=DEFAULT_MAIN_DB)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--debug-jsonl", default="")
    parser.add_argument("--error-jsonl", default=DEFAULT_ERROR_JSONL)
    parser.add_argument("--progress-log", default=DEFAULT_PROGRESS_LOG)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--text-topk", type=int, default=20)
    parser.add_argument("--image-topk", type=int, default=20)
    parser.add_argument("--top-m-neighbors", type=int, default=5)
    parser.add_argument("--pair-boost", type=float, default=0.15)
    parser.add_argument("--pair-min-weight", type=float, default=0.45)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--ppr-iters", type=int, default=10)
    parser.add_argument("--select-k", type=int, default=3)
    parser.add_argument("--redundancy-weight", type=float, default=0.3)
    parser.add_argument("--pair-complete-bonus", type=float, default=0.1)
    parser.add_argument("--min-gain", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N queries for smoke test.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files instead of resuming.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    main_db_path = Path(args.main_db)
    block_mapping = load_block_mapping(main_db_path)
    sample_mapping = load_sample_mapping(main_db_path)

    bundles: Optional[List[QueryBundle]] = None
    if not args.retrieval_export_jsonl:
        text_rows = load_jsonl(Path(args.text_jsonl))
        image_rows = load_jsonl(Path(args.image_jsonl))
        bundles = build_query_bundles_from_recall(
            text_rows,
            image_rows,
            block_mapping,
            sample_mapping,
            text_topk=args.text_topk,
            image_topk=args.image_topk,
        )

    encoder = Qwen3VLEncoder(args.model_path, device=args.device, batch_size=args.batch_size)
    output_path = Path(args.output_jsonl)
    debug_path = Path(args.debug_jsonl) if args.debug_jsonl else None
    error_path = Path(args.error_jsonl) if args.error_jsonl else None
    progress_path = Path(args.progress_log) if args.progress_log else None
    ensure_parent_dir(output_path)
    if debug_path is not None:
        ensure_parent_dir(debug_path)
    if error_path is not None:
        ensure_parent_dir(error_path)
    if progress_path is not None:
        ensure_parent_dir(progress_path)

    if args.overwrite:
        if output_path.exists():
            output_path.unlink()
        if debug_path is not None and debug_path.exists():
            debug_path.unlink()
        if error_path is not None and error_path.exists():
            error_path.unlink()
        if progress_path is not None and progress_path.exists():
            progress_path.unlink()

    processed_indices = load_processed_indices(output_path)
    if debug_path is not None and not args.overwrite:
        # Keep debug and main outputs aligned when resuming.
        processed_indices &= load_processed_indices(debug_path) if debug_path.exists() else set()

    if bundles is not None:
        if args.limit > 0:
            bundles = bundles[: args.limit]
        if args.num_shards > 1:
            bundles = [bundle for bundle in bundles if bundle.index % args.num_shards == args.shard_index]
        bundle_iterable: Iterable[QueryBundle] = bundles
        expected_total: Optional[int] = len(bundles)
    else:
        bundle_iterable = iter_retrieval_export_bundles(
            Path(args.retrieval_export_jsonl),
            block_mapping,
            sample_mapping,
            text_topk=args.text_topk,
            image_topk=args.image_topk,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            limit=args.limit,
        )
        expected_total = None

    skipped = 0
    processed = 0
    progress_mode = "a" if progress_path is not None and progress_path.exists() and not args.overwrite else "w"
    progress_handle = progress_path.open(progress_mode, encoding="utf-8") if progress_path is not None else None
    log_message(
        (
            f"[resume] total={expected_total if expected_total is not None else 'stream'} "
            f"processed_index_count={len(processed_indices)} shard={args.shard_index}/{args.num_shards}"
        ),
        progress_handle,
    )

    output_mode = "a" if output_path.exists() and not args.overwrite else "w"
    debug_mode = "a" if debug_path is not None and debug_path.exists() and not args.overwrite else "w"
    error_mode = "a" if error_path is not None and error_path.exists() and not args.overwrite else "w"

    with output_path.open(output_mode, encoding="utf-8") as output_handle:
        debug_handle = None
        error_handle = None
        if debug_path is not None:
            debug_handle = debug_path.open(debug_mode, encoding="utf-8")
        if error_path is not None:
            error_handle = error_path.open(error_mode, encoding="utf-8")
        try:
            for bundle in bundle_iterable:
                if bundle.index in processed_indices:
                    skipped += 1
                    continue
                try:
                    export_record, debug_record = process_bundle(
                        bundle,
                        encoder,
                        top_m_neighbors=args.top_m_neighbors,
                        pair_boost=args.pair_boost,
                        pair_min_weight=args.pair_min_weight,
                        alpha=args.alpha,
                        ppr_iters=args.ppr_iters,
                        select_k=args.select_k,
                        redundancy_weight=args.redundancy_weight,
                        pair_complete_bonus=args.pair_complete_bonus,
                        min_gain=args.min_gain,
                    )
                    write_jsonl_record(output_handle, export_record)
                    if debug_handle is not None:
                        write_jsonl_record(debug_handle, debug_record)
                    processed += 1
                except Exception as exc:
                    if error_handle is not None:
                        write_jsonl_record(
                            error_handle,
                            {
                                "index": bundle.index,
                                "query": bundle.query,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            },
                        )
                    log_message(
                        f"[error] index={bundle.index} type={type(exc).__name__} msg={exc}",
                        progress_handle,
                    )
                    continue
                if processed % 10 == 0:
                    log_message(
                        f"[progress] processed={processed} skipped_existing={skipped} shard={args.shard_index}/{args.num_shards}",
                        progress_handle,
                    )
        finally:
            if debug_handle is not None:
                debug_handle.close()
            if error_handle is not None:
                error_handle.close()
            if progress_handle is not None:
                progress_handle.close()

    print(f"[done] output_jsonl={output_path}")
    if debug_path is not None:
        print(f"[done] debug_jsonl={debug_path}")
    if error_path is not None:
        print(f"[done] error_jsonl={error_path}")


if __name__ == "__main__":
    main()
