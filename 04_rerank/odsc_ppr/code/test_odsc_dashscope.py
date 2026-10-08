#!/usr/bin/env python3
"""Real embedding test for ODSC using DashScope text-embedding API.

Uses sample data from retrieval_export.sample.jsonl.
Since DashScope embedding is text-only, image candidates use their caption text.
This tests the core ODSC logic (support vectors, discriminability, submodular selection)
with real semantic embeddings instead of random vectors.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
import unittest.mock
from pathlib import Path
from typing import Dict, List, Sequence

import dashscope
import numpy as np

# Stub sentence_transformers for import.
fake_st = types.ModuleType("sentence_transformers")
fake_st.SentenceTransformer = unittest.mock.MagicMock
sys.modules["sentence_transformers"] = fake_st

sys.path.insert(0, str(Path(__file__).parent))
from rerank_odsc import (
    build_bundle,
    build_sub_queries,
    compute_discriminability,
    compute_support_matrix,
    submodular_select,
    build_export_record,
    iter_jsonl,
)

SAMPLE_PATH = Path(__file__).parent.parent / "data" / "persistent" / "retrieval_export.sample.jsonl"
MODEL_NAME = "text-embedding-v3"


def dashscope_embed(texts: Sequence[str]) -> np.ndarray:
    """Call DashScope text-embedding API. Handles batching (max 6 per call)."""
    all_embs = []
    batch_size = 6
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        resp = dashscope.TextEmbedding.call(
            model=MODEL_NAME,
            input=batch,
            dimension=1024,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope API error: {resp.code} {resp.message}")
        for item in resp.output["embeddings"]:
            all_embs.append(item["embedding"])
    return np.array(all_embs, dtype=np.float32)


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)


def run_odsc_for_bundle(row: dict, idx_label: str) -> dict:
    """Run full ODSC pipeline for one sample."""
    bundle = build_bundle(row, text_topk=20, image_topk=20)
    if bundle is None:
        return {}

    image_cands = [c for c in bundle.image_candidates if c.image_path]
    candidates = list(bundle.text_candidates) + image_cands

    if not candidates:
        return {"index": bundle.index, "retrieval": [], "note": "no candidates"}

    # Build sub-queries.
    option_keys, sub_query_texts = build_sub_queries(bundle.question, bundle.options)
    print(f"\n{'='*60}")
    print(f"[{idx_label}] index={bundle.index}")
    print(f"  Question: {bundle.question}")
    print(f"  Options: {bundle.options}")
    print(f"  Answer: {bundle.answer} ({bundle.options.get(bundle.answer, '?')})")
    print(f"  Candidates: {len(bundle.text_candidates)} text + {len(image_cands)} image")

    # Encode with DashScope.
    print(f"  Encoding {len(sub_query_texts)} sub-queries + {len(candidates)} candidates...")
    t0 = time.time()

    sub_query_embs = l2_normalize(dashscope_embed(sub_query_texts))
    cand_texts = [c.text for c in candidates]
    candidate_embs = l2_normalize(dashscope_embed(cand_texts))

    elapsed = time.time() - t0
    print(f"  Encoding done in {elapsed:.2f}s")

    # Compute support and discriminability.
    support = compute_support_matrix(sub_query_embs, candidate_embs)
    disc = compute_discriminability(support)

    # Show support distribution.
    print(f"\n  Support matrix stats:")
    for i, k in enumerate(option_keys):
        col = support[:, i]
        print(f"    Option {k} ({bundle.options[k]}): "
              f"min={col.min():.4f} max={col.max():.4f} mean={col.mean():.4f}")

    print(f"\n  Discriminability: min={disc.min():.4f} max={disc.max():.4f} mean={disc.mean():.4f}")

    # Show top-5 most discriminative.
    top5 = np.argsort(-disc)[:5]
    print(f"\n  Top-5 discriminative candidates:")
    for rank, idx in enumerate(top5, 1):
        c = candidates[idx]
        sup_str = " ".join(f"{k}={support[idx, i]:.3f}" for i, k in enumerate(option_keys))
        best_option = option_keys[np.argmax(support[idx])]
        print(f"    #{rank} [idx={idx}] disc={disc[idx]:.4f} best={best_option}({bundle.options[best_option]}) "
              f"source={c.source} | {sup_str}")
        print(f"         text: {c.text[:80]}...")

    # Submodular selection.
    selected, submod_debug = submodular_select(
        candidates, candidate_embs, support, disc,
        select_k=3,
        lambda_disc=1.0,
        lambda_cover=0.5,
        lambda_cross=0.3,
        cross_sim_threshold=0.5,
    )

    print(f"\n  Submodular selection result:")
    print(f"    Objective: {submod_debug['final_objective']:.4f} "
          f"(disc={submod_debug['disc_component']:.4f} "
          f"cover={submod_debug['cover_component']:.4f} "
          f"cross={submod_debug['cross_component']:.4f})")

    export_rec = build_export_record(bundle, candidates, selected)
    print(f"\n  Selected top-3:")
    for item in export_rec["retrieval"]:
        idx_in_pool = selected[item["rank"] - 1]
        sup_str = " ".join(f"{k}={support[idx_in_pool, i]:.3f}" for i, k in enumerate(option_keys))
        best_opt = option_keys[np.argmax(support[idx_in_pool])]
        print(f"    rank={item['rank']} source={item['source']} "
              f"best_option={best_opt}({bundle.options[best_opt]}) "
              f"disc={disc[idx_in_pool]:.4f}")
        print(f"      {sup_str}")
        print(f"      doc: {item['doc_name'][:50]}")
        print(f"      text: {item['text'][:80]}...")

    # Check if selected evidence points to correct answer.
    correct_key = bundle.answer
    correct_count = 0
    for idx in selected:
        best_opt = option_keys[np.argmax(support[idx])]
        if best_opt == correct_key:
            correct_count += 1
    print(f"\n  Verdict: {correct_count}/3 selected candidates point to correct answer "
          f"({correct_key}: {bundle.options.get(correct_key, '?')})")

    return export_rec


def main():
    print("ODSC Real Embedding Test (DashScope text-embedding-v3)")
    print(f"Sample: {SAMPLE_PATH}\n")

    rows = list(iter_jsonl(SAMPLE_PATH))
    print(f"Loaded {len(rows)} samples")

    results = []
    total_api_usage = {"total_tokens": 0}

    for i, row in enumerate(rows[:3]):
        result = run_odsc_for_bundle(row, f"Sample {i+1}/3")
        results.append(result)

    # Save results.
    out_path = Path(__file__).parent.parent / "data" / "tmp" / "odsc_dashscope_test.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n{'='*60}")
    print(f"Results saved to: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
