#!/usr/bin/env python3
"""Offline logic test for ODSC reranker.

Uses sample data + random embeddings to verify:
1. Data loading and bundle construction
2. Support matrix / discriminability computation
3. Submodular greedy selection
4. Export record format compatibility with downstream eval
"""

import importlib
import json
import sys
import time
import types
import unittest.mock
from pathlib import Path

import numpy as np

# Stub out sentence_transformers so we can import rerank_odsc without GPU/model.
fake_st = types.ModuleType("sentence_transformers")
fake_st.SentenceTransformer = unittest.mock.MagicMock
sys.modules["sentence_transformers"] = fake_st

sys.path.insert(0, str(Path(__file__).parent))
from rerank_odsc import (
    Candidate,
    QueryBundle,
    build_bundle,
    build_sub_queries,
    compute_discriminability,
    compute_support_matrix,
    submodular_select,
    build_export_record,
    build_debug_record,
    iter_jsonl,
)

SAMPLE_PATH = Path(__file__).parent.parent / "data" / "persistent" / "retrieval_export.sample.jsonl"
PPR_OUTPUT_PATH = Path(__file__).parent.parent / "data" / "persistent" / "endobench_multimodal_ppr_top3_smoke.jsonl"
DIM = 64  # fake embedding dimension


def fake_embeddings(n: int) -> np.ndarray:
    rng = np.random.RandomState(42)
    vecs = rng.randn(n, DIM).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)


def test_data_loading():
    print("=" * 60)
    print("TEST 1: Data loading and bundle construction")
    print("=" * 60)

    rows = list(iter_jsonl(SAMPLE_PATH))
    print(f"  Loaded {len(rows)} rows from sample data")

    for row in rows:
        bundle = build_bundle(row, text_topk=20, image_topk=20)
        assert bundle is not None, "bundle should not be None"
        assert bundle.question, "question should not be empty"
        assert bundle.options, "options should not be empty"
        print(f"  index={bundle.index}: question='{bundle.question[:30]}...' "
              f"options={list(bundle.options.keys())} "
              f"text_cands={len(bundle.text_candidates)} "
              f"image_cands={len(bundle.image_candidates)} "
              f"answer={bundle.answer}")

    print("  PASSED\n")
    return rows


def test_sub_queries(rows):
    print("=" * 60)
    print("TEST 2: Option-aware sub-query construction")
    print("=" * 60)

    bundle = build_bundle(rows[0], text_topk=20, image_topk=20)
    keys, queries = build_sub_queries(bundle.question, bundle.options)

    print(f"  Option keys: {keys}")
    for k, q in zip(keys, queries):
        print(f"  {k}: '{q}'")

    assert len(keys) == len(bundle.options), "should have one sub-query per option"
    assert all(bundle.options[k] in q for k, q in zip(keys, queries)), \
        "each sub-query should contain its option text"

    print("  PASSED\n")
    return bundle, keys


def test_support_and_discriminability(bundle, option_keys):
    print("=" * 60)
    print("TEST 3: Support matrix and discriminability")
    print("=" * 60)

    image_cands = [c for c in bundle.image_candidates if c.image_path]
    candidates = list(bundle.text_candidates) + image_cands
    n_cands = len(candidates)
    n_options = len(option_keys)

    sub_query_embs = fake_embeddings(n_options)
    candidate_embs = fake_embeddings(n_cands)

    support = compute_support_matrix(sub_query_embs, candidate_embs)
    print(f"  Support matrix shape: {support.shape} (expected ({n_cands}, {n_options}))")
    assert support.shape == (n_cands, n_options)
    assert (support >= 0).all(), "support should be non-negative"

    disc = compute_discriminability(support)
    print(f"  Discriminability shape: {disc.shape} (expected ({n_cands},))")
    assert disc.shape == (n_cands,)
    assert (disc >= 0).all(), "discriminability should be non-negative"

    print(f"  Disc stats: min={disc.min():.4f} max={disc.max():.4f} mean={disc.mean():.4f}")
    print(f"  Top-3 discriminative candidates: {np.argsort(-disc)[:3].tolist()}")

    print("  PASSED\n")
    return candidates, candidate_embs, support, disc


def test_submodular_selection(candidates, candidate_embs, support, disc):
    print("=" * 60)
    print("TEST 4: Submodular greedy selection")
    print("=" * 60)

    selected, debug_info = submodular_select(
        candidates,
        candidate_embs,
        support,
        disc,
        select_k=3,
        lambda_disc=1.0,
        lambda_cover=0.5,
        lambda_cross=0.3,
        cross_sim_threshold=0.5,
    )

    print(f"  Selected indices: {selected}")
    print(f"  Selected sources: {[candidates[i].source for i in selected]}")
    assert len(selected) == 3, f"should select exactly 3, got {len(selected)}"
    assert len(set(selected)) == 3, "should not have duplicates"

    print(f"  Final objective: {debug_info['final_objective']:.4f}")
    print(f"    disc={debug_info['disc_component']:.4f} "
          f"cover={debug_info['cover_component']:.4f} "
          f"cross={debug_info['cross_component']:.4f}")

    for step in debug_info["trace"]:
        print(f"    Step {step['step']}: idx={step['selected_index']} "
              f"gain={step['marginal_gain']:.4f} "
              f"source={step['source']} doc={step['doc_name'][:30]}")

    print("  PASSED\n")
    return selected, debug_info


def test_export_format(bundle, candidates, support, disc, selected, option_keys, debug_info):
    print("=" * 60)
    print("TEST 5: Export format compatibility")
    print("=" * 60)

    export_rec = build_export_record(bundle, candidates, selected)
    print(f"  Export record keys: {list(export_rec.keys())}")
    assert "index" in export_rec
    assert "retrieval" in export_rec
    assert len(export_rec["retrieval"]) == 3

    for item in export_rec["retrieval"]:
        assert "source" in item
        assert "rank" in item
        assert "doc_name" in item
        assert "text" in item
        print(f"    rank={item['rank']} source={item['source']} "
              f"doc={item['doc_name'][:40]}...")

    # Verify format matches PPR output.
    if PPR_OUTPUT_PATH.exists():
        with PPR_OUTPUT_PATH.open() as f:
            ppr_rec = json.loads(f.readline())
        ppr_keys = set(ppr_rec["retrieval"][0].keys())
        odsc_keys = set(export_rec["retrieval"][0].keys())
        print(f"  PPR output keys:  {sorted(ppr_keys)}")
        print(f"  ODSC output keys: {sorted(odsc_keys)}")
        missing = ppr_keys - odsc_keys
        assert not missing, f"ODSC missing keys from PPR: {missing}"
        extra = odsc_keys - ppr_keys
        print(f"  ODSC superset of PPR (extra fields: {sorted(extra)})")
        print("  Format compatible with PPR output!")

    # Debug record.
    debug_rec = build_debug_record(
        bundle, candidates, support, disc, selected, option_keys, debug_info,
    )
    assert "support_vector" in debug_rec["candidates"][0]
    assert "discriminability" in debug_rec["candidates"][0]
    print(f"  Debug record has {len(debug_rec['candidates'])} candidates with support vectors")

    print("  PASSED\n")


def test_edge_cases():
    print("=" * 60)
    print("TEST 6: Edge cases")
    print("=" * 60)

    # Empty candidates.
    support = np.zeros((0, 4), dtype=np.float32)
    disc = compute_discriminability(support)
    assert disc.shape == (0,), "empty input should return empty"
    print("  Empty candidates: PASSED")

    # Single candidate.
    support = np.array([[0.8, 0.2, 0.1, 0.3]], dtype=np.float32)
    disc = compute_discriminability(support)
    assert disc.shape == (1,)
    assert abs(disc[0] - 0.5) < 1e-6, f"expected 0.8-0.3=0.5, got {disc[0]}"
    print(f"  Single candidate disc={disc[0]:.4f}: PASSED")

    # Two options only.
    support = np.array([[0.9, 0.1], [0.5, 0.5]], dtype=np.float32)
    disc = compute_discriminability(support)
    assert abs(disc[0] - 0.8) < 1e-6
    assert abs(disc[1] - 0.0) < 1e-6
    print(f"  Two-option disc=[{disc[0]:.1f}, {disc[1]:.1f}]: PASSED")

    print("  ALL EDGE CASES PASSED\n")


def main():
    t0 = time.time()
    print(f"\nODSC Logic Test Suite")
    print(f"Sample data: {SAMPLE_PATH}")
    print(f"PPR output:  {PPR_OUTPUT_PATH}\n")

    assert SAMPLE_PATH.exists(), f"Sample data not found: {SAMPLE_PATH}"

    rows = test_data_loading()
    bundle, option_keys = test_sub_queries(rows)
    candidates, candidate_embs, support, disc = test_support_and_discriminability(bundle, option_keys)
    selected, debug_info = test_submodular_selection(candidates, candidate_embs, support, disc)
    test_export_format(bundle, candidates, support, disc, selected, option_keys, debug_info)
    test_edge_cases()

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"ALL TESTS PASSED ({elapsed:.2f}s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
