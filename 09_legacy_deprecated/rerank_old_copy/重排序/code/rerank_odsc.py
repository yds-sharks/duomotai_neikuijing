#!/usr/bin/env python3
"""Option-Discriminative Submodular Composition (ODSC) reranker.

Pipeline
--------
1. Read retrieval_export JSONL (same format as PPR reranker).
2. Build option-aware sub-queries: question + each option.
3. Encode sub-queries and candidates with Qwen3-VL-Embedding-2B.
   - Text candidates: encode text.
   - Image candidates: encode {image + caption} jointly.
4. Compute per-candidate support vector over options.
5. Greedy submodular selection optimizing:
   a) Discriminability — candidate strongly supports one option over others.
   b) Option coverage  — selected set covers all option dimensions.
   c) Cross-modal coherence — text+image evidence for same concept.
6. Export retrieval JSONL for downstream RAG evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_RETRIEVAL_EXPORT_JSONL = "data/retrieval_export.jsonl"
DEFAULT_MODEL_PATH = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_OUTPUT_JSONL = "output/endobench_odsc_top3.jsonl"
DEFAULT_ERROR_JSONL = "output/endobench_odsc_top3_errors.jsonl"
DEFAULT_PROGRESS_LOG = "output/endobench_odsc_top3_progress.log"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    source: str            # "text" or "image"
    rank: int
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
    question: str
    options: Dict[str, str]    # {"A": "食管", "B": "小肠", ...}
    answer: str                # ground-truth key, only for debug
    text_candidates: List[Candidate] = field(default_factory=list)
    image_candidates: List[Candidate] = field(default_factory=list)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

_write_lock = threading.Lock()


def log_message(message: str, handle: Optional[Any] = None) -> None:
    print(message, flush=True)
    if handle is None:
        return
    handle.write(message + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{lineno}: {exc}") from exc


def write_jsonl_record(handle: Any, record: Dict[str, Any]) -> None:
    with _write_lock:
        handle.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(i) for i in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def load_processed_indices(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = row.get("index")
            if idx is not None:
                try:
                    done.add(int(idx))
                except (TypeError, ValueError):
                    pass
    return done


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Candidate score extraction (same logic as PPR reranker)
# ---------------------------------------------------------------------------

def _candidate_score(raw: Dict[str, Any]) -> float:
    for key in ("weighted_score", "raw_score", "score"):
        v = raw.get(key)
        if v is not None:
            return float(v)
    return 0.0


# ---------------------------------------------------------------------------
# Build QueryBundle from retrieval_export row
# ---------------------------------------------------------------------------

def build_bundle(row: Dict[str, Any], *, text_topk: int, image_topk: int) -> Optional[QueryBundle]:
    sample = row.get("sample") or {}
    retrieval = row.get("retrieval") or {}

    index_val = sample.get("index", row.get("index"))
    if index_val is None:
        return None

    question = str(sample.get("question") or row.get("question") or "").strip()
    options = sample.get("options") or {}
    answer = str(sample.get("answer") or "")

    text_cands = [
        Candidate(
            source="text",
            rank=int(r.get("rank") or 0),
            first_stage_score=_candidate_score(r),
            text=str(r.get("text") or ""),
            doc_name=str(r.get("doc_name") or ""),
            page_idx=r.get("page_idx"),
            doc_id=str(r.get("doc_id") or ""),
        )
        for r in list(retrieval.get("text_top20") or [])[:text_topk]
    ]

    image_cands = [
        Candidate(
            source="image",
            rank=int(r.get("rank") or 0),
            first_stage_score=_candidate_score(r),
            text=str(r.get("text") or r.get("content") or ""),
            doc_name=str(r.get("doc_name") or ""),
            page_idx=r.get("page_idx"),
            doc_id=str(r.get("doc_id") or ""),
            sample_id=str(r.get("sample_id") or ""),
            group_id=str(r.get("group_id") or ""),
            image_path=str(r.get("image_path") or ""),
            image_id=str(r.get("image_id") or ""),
        )
        for r in list(retrieval.get("image_top20") or [])[:image_topk]
    ]

    return QueryBundle(
        index=int(index_val),
        question=question,
        options=options,
        answer=answer,
        text_candidates=text_cands,
        image_candidates=image_cands,
    )


def iter_bundles(
    path: Path,
    *,
    text_topk: int,
    image_topk: int,
    num_shards: int,
    shard_index: int,
    limit: int,
) -> Iterable[QueryBundle]:
    seen = 0
    for row in iter_jsonl(path):
        bundle = build_bundle(row, text_topk=text_topk, image_topk=image_topk)
        if bundle is None:
            continue
        seen += 1
        if 0 < limit < seen:
            break
        if bundle.index % num_shards != shard_index:
            continue
        yield bundle


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class VLEncoder:
    def __init__(self, model_path: str, device: str, batch_size: int):
        self.model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
        self.batch_size = batch_size
        self._dim = self.model.get_sentence_embedding_dimension()

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.asarray(
            self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                prompt="Retrieve relevant medical multimodal evidence for the query.",
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.asarray(
            self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def encode_images_with_text(self, pairs: Sequence[Dict[str, str]]) -> np.ndarray:
        """Encode image candidates with their caption jointly."""
        if not pairs:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.asarray(
            self.model.encode(
                list(pairs),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vecs / norms


# ---------------------------------------------------------------------------
# Step 1-2: Build option-aware sub-queries and support vectors
# ---------------------------------------------------------------------------

def build_sub_queries(question: str, options: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Return (option_keys, sub_query_texts) sorted by key."""
    keys = sorted(options.keys())
    queries = [f"{question} 答案：{options[k]}" for k in keys]
    return keys, queries


def compute_support_matrix(
    sub_query_embs: np.ndarray,   # (num_options, dim)
    candidate_embs: np.ndarray,   # (num_candidates, dim)
) -> np.ndarray:
    """Return (num_candidates, num_options) support matrix."""
    sim = candidate_embs @ sub_query_embs.T   # (C, O)
    return np.maximum(sim, 0.0)


# ---------------------------------------------------------------------------
# Step 3: Discriminability score
# ---------------------------------------------------------------------------

def compute_discriminability(support: np.ndarray) -> np.ndarray:
    """Per-candidate discriminability = top1 - top2 support.

    Returns (num_candidates,) array.
    """
    if support.shape[1] < 2:
        return support.max(axis=1)
    sorted_desc = np.sort(support, axis=1)[:, ::-1]
    return sorted_desc[:, 0] - sorted_desc[:, 1]


# ---------------------------------------------------------------------------
# Step 4: Submodular greedy selection
# ---------------------------------------------------------------------------

def submodular_select(
    candidates: Sequence[Candidate],
    candidate_embs: np.ndarray,
    support: np.ndarray,
    discriminability: np.ndarray,
    *,
    select_k: int,
    lambda_disc: float,
    lambda_cover: float,
    lambda_cross: float,
    cross_sim_threshold: float,
) -> Tuple[List[int], Dict[str, Any]]:
    """Greedy submodular selection.

    Objective f(S) = lambda_disc * Disc(S)
                   + lambda_cover * Cover(S)
                   + lambda_cross * Cross(S)

    Returns (selected_indices, debug_info).
    """
    n = len(candidates)
    num_options = support.shape[1]
    selected: List[int] = []
    remaining = list(range(n))
    trace: List[Dict[str, Any]] = []

    # Precompute cross-modal similarity clusters.
    # For each pair (i, j) where source differs, check embedding sim > threshold.
    # We'll compute these on-the-fly in marginal gain for efficiency.

    def _option_coverage(sel: List[int]) -> float:
        if not sel:
            return 0.0
        # For each option, how much total support from selected candidates (capped at 1).
        total = support[sel].sum(axis=0)   # (num_options,)
        return float(np.minimum(total, 1.0).sum())

    def _cross_modal_coherence(sel: List[int]) -> float:
        if len(sel) < 2:
            return 0.0
        bonus = 0.0
        for i_pos, i_idx in enumerate(sel):
            for j_idx in sel[i_pos + 1:]:
                if candidates[i_idx].source == candidates[j_idx].source:
                    continue
                sim = float(candidate_embs[i_idx] @ candidate_embs[j_idx])
                if sim >= cross_sim_threshold:
                    bonus += sim
        return bonus

    def _disc_score(sel: List[int]) -> float:
        if not sel:
            return 0.0
        return float(discriminability[sel].sum())

    def _f(sel: List[int]) -> float:
        return (lambda_disc * _disc_score(sel)
                + lambda_cover * _option_coverage(sel)
                + lambda_cross * _cross_modal_coherence(sel))

    for step in range(select_k):
        best_idx = -1
        best_gain = -math.inf
        current_val = _f(selected)

        for idx in remaining:
            trial = selected + [idx]
            gain = _f(trial) - current_val
            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx < 0:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        trace.append({
            "step": step + 1,
            "selected_index": best_idx,
            "marginal_gain": best_gain,
            "source": candidates[best_idx].source,
            "doc_name": candidates[best_idx].doc_name,
        })

    # If we didn't select enough, fill by discriminability.
    if len(selected) < select_k:
        for idx in np.argsort(-discriminability):
            if int(idx) not in selected:
                selected.append(int(idx))
                if len(selected) >= select_k:
                    break

    debug_info = {
        "final_objective": _f(selected),
        "disc_component": _disc_score(selected),
        "cover_component": _option_coverage(selected),
        "cross_component": _cross_modal_coherence(selected),
        "trace": trace,
    }
    return selected[:select_k], debug_info


# ---------------------------------------------------------------------------
# Build export records
# ---------------------------------------------------------------------------

def build_export_record(
    bundle: QueryBundle,
    candidates: Sequence[Candidate],
    selected: Sequence[int],
) -> Dict[str, Any]:
    items = []
    for rank, idx in enumerate(selected, 1):
        c = candidates[idx]
        items.append({
            "source": c.source,
            "rank": rank,
            "doc_name": c.doc_name,
            "page_idx": c.page_idx,
            "doc_id": c.doc_id,
            "sample_id": c.sample_id,
            "group_id": c.group_id,
            "image_id": c.image_id,
            "image_path": c.image_path,
            "text": c.text,
        })
    return {"index": bundle.index, "retrieval": items}


def build_debug_record(
    bundle: QueryBundle,
    candidates: Sequence[Candidate],
    support: np.ndarray,
    discriminability: np.ndarray,
    selected: Sequence[int],
    option_keys: Sequence[str],
    submod_debug: Dict[str, Any],
) -> Dict[str, Any]:
    selected_set = set(selected)
    details = []
    for idx, c in enumerate(candidates):
        details.append({
            "candidate_index": idx,
            "selected": idx in selected_set,
            "source": c.source,
            "doc_name": c.doc_name,
            "first_stage_rank": c.rank,
            "first_stage_score": c.first_stage_score,
            "discriminability": float(discriminability[idx]),
            "support_vector": {k: float(support[idx, i]) for i, k in enumerate(option_keys)},
            "text": c.text[:200],
            "image_path": c.image_path,
        })
    return {
        "index": bundle.index,
        "question": bundle.question,
        "options": bundle.options,
        "answer": bundle.answer,
        "submodular": submod_debug,
        "candidates": details,
    }


# ---------------------------------------------------------------------------
# Process one bundle
# ---------------------------------------------------------------------------

def process_bundle(
    bundle: QueryBundle,
    encoder: VLEncoder,
    *,
    select_k: int,
    lambda_disc: float,
    lambda_cover: float,
    lambda_cross: float,
    cross_sim_threshold: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Filter image candidates with valid paths.
    image_cands = [c for c in bundle.image_candidates if c.image_path]
    candidates: List[Candidate] = list(bundle.text_candidates) + image_cands

    if not candidates:
        return (
            {"index": bundle.index, "retrieval": []},
            {"index": bundle.index, "question": bundle.question, "candidates": []},
        )

    # Build option-aware sub-queries.
    option_keys, sub_query_texts = build_sub_queries(bundle.question, bundle.options)

    # Encode sub-queries.
    sub_query_embs = encoder.encode_queries(sub_query_texts)   # (O, dim)

    # Encode candidates.
    text_texts = [c.text for c in bundle.text_candidates]
    text_embs = encoder.encode_texts(text_texts)

    image_pairs = [{"image": c.image_path, "text": c.text} for c in image_cands]
    image_embs = encoder.encode_images_with_text(image_pairs)

    if text_embs.shape[0] > 0 and image_embs.shape[0] > 0:
        candidate_embs = np.concatenate([text_embs, image_embs], axis=0)
    elif text_embs.shape[0] > 0:
        candidate_embs = text_embs
    else:
        candidate_embs = image_embs

    candidate_embs = l2_normalize(candidate_embs.astype(np.float32))

    # Compute support matrix and discriminability.
    support = compute_support_matrix(sub_query_embs, candidate_embs)
    disc = compute_discriminability(support)

    # Submodular selection.
    selected, submod_debug = submodular_select(
        candidates,
        candidate_embs,
        support,
        disc,
        select_k=select_k,
        lambda_disc=lambda_disc,
        lambda_cover=lambda_cover,
        lambda_cross=lambda_cross,
        cross_sim_threshold=cross_sim_threshold,
    )

    export_rec = build_export_record(bundle, candidates, selected)
    debug_rec = build_debug_record(
        bundle, candidates, support, disc, selected, option_keys, submod_debug,
    )
    return export_rec, debug_rec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ODSC reranker for EndoBench multimodal retrieval.")
    p.add_argument("--retrieval-export-jsonl", default=DEFAULT_RETRIEVAL_EXPORT_JSONL)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    p.add_argument("--debug-jsonl", default="")
    p.add_argument("--error-jsonl", default=DEFAULT_ERROR_JSONL)
    p.add_argument("--progress-log", default=DEFAULT_PROGRESS_LOG)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--text-topk", type=int, default=20)
    p.add_argument("--image-topk", type=int, default=20)
    p.add_argument("--select-k", type=int, default=3)
    # Submodular weights.
    p.add_argument("--lambda-disc", type=float, default=1.0,
                   help="Weight for discriminability component.")
    p.add_argument("--lambda-cover", type=float, default=0.5,
                   help="Weight for option coverage component.")
    p.add_argument("--lambda-cross", type=float, default=0.3,
                   help="Weight for cross-modal coherence component.")
    p.add_argument("--cross-sim-threshold", type=float, default=0.5,
                   help="Embedding similarity threshold for cross-modal bonus.")
    # Execution.
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t_start = time.time()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    output_path = Path(args.output_jsonl)
    debug_path = Path(args.debug_jsonl) if args.debug_jsonl else None
    error_path = Path(args.error_jsonl) if args.error_jsonl else None
    progress_path = Path(args.progress_log) if args.progress_log else None

    for p in [output_path, debug_path, error_path, progress_path]:
        if p is not None:
            ensure_parent(p)

    if args.overwrite:
        for p in [output_path, debug_path, error_path, progress_path]:
            if p is not None and p.exists():
                p.unlink()

    processed_indices = load_processed_indices(output_path)
    if debug_path is not None and not args.overwrite:
        processed_indices &= (load_processed_indices(debug_path) if debug_path.exists() else set())

    encoder = VLEncoder(args.model_path, device=args.device, batch_size=args.batch_size)

    bundle_iter = iter_bundles(
        Path(args.retrieval_export_jsonl),
        text_topk=args.text_topk,
        image_topk=args.image_topk,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        limit=args.limit,
    )

    mode = "a" if not args.overwrite else "w"
    skipped = 0
    processed = 0

    progress_handle = (
        progress_path.open(mode, encoding="utf-8") if progress_path else None
    )
    log_message(
        f"[start] shard={args.shard_index}/{args.num_shards} "
        f"resume_count={len(processed_indices)} "
        f"lambda_disc={args.lambda_disc} lambda_cover={args.lambda_cover} "
        f"lambda_cross={args.lambda_cross} cross_sim_threshold={args.cross_sim_threshold}",
        progress_handle,
    )

    with output_path.open(mode, encoding="utf-8") as out_f:
        debug_f = debug_path.open(mode, encoding="utf-8") if debug_path else None
        error_f = error_path.open(mode, encoding="utf-8") if error_path else None
        try:
            for bundle in bundle_iter:
                if bundle.index in processed_indices:
                    skipped += 1
                    continue

                t_item = time.time()
                try:
                    export_rec, debug_rec = process_bundle(
                        bundle,
                        encoder,
                        select_k=args.select_k,
                        lambda_disc=args.lambda_disc,
                        lambda_cover=args.lambda_cover,
                        lambda_cross=args.lambda_cross,
                        cross_sim_threshold=args.cross_sim_threshold,
                    )
                    write_jsonl_record(out_f, export_rec)
                    if debug_f is not None:
                        write_jsonl_record(debug_f, debug_rec)
                    processed += 1
                except Exception as exc:
                    if error_f is not None:
                        write_jsonl_record(error_f, {
                            "index": bundle.index,
                            "question": bundle.question,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        })
                    log_message(
                        f"[error] index={bundle.index} {type(exc).__name__}: {exc}",
                        progress_handle,
                    )
                    continue

                elapsed_item = time.time() - t_item
                if processed % 10 == 0:
                    elapsed_total = time.time() - t_start
                    speed = processed / elapsed_total if elapsed_total > 0 else 0
                    log_message(
                        f"[progress] processed={processed} skipped={skipped} "
                        f"last={elapsed_item:.2f}s speed={speed:.1f}q/s "
                        f"shard={args.shard_index}/{args.num_shards}",
                        progress_handle,
                    )
        finally:
            if debug_f is not None:
                debug_f.close()
            if error_f is not None:
                error_f.close()
            if progress_handle is not None:
                progress_handle.close()

    elapsed = time.time() - t_start
    print(f"[done] processed={processed} skipped={skipped} "
          f"elapsed={elapsed:.1f}s output={output_path}")


if __name__ == "__main__":
    main()
