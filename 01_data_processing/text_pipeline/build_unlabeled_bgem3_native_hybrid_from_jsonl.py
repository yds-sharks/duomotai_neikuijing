#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a fresh Milvus-Lite collection for the unlabeled corpus using
official BGE-M3 native hybrid outputs from the cleaned source JSONL.

Input:
    cleaned text-only JSONL produced by the recent MinerU resume pipeline

Output collection fields:
    - dense_vector: official BGE-M3 dense embedding
    - bge_m3_lexical_sparse: official BGE-M3 lexical sparse weights
    - text + basic metadata for retrieval inspection
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from FlagEmbedding import BGEM3FlagModel
import torch
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

import sys
sys.path.insert(0, "/mnt/data_1/yds/多模态/insert")
from bm25_bge_vectorstore_v2.text_utils import clean_text, detect_lang
from transformers import AutoModel
from huggingface_hub import snapshot_download
from FlagEmbedding.finetune.embedder.encoder_only.m3 import EncoderOnlyEmbedderM3Runner


def patch_flagembedding_dtype_bug() -> None:
    def _patched_get_model(
        model_name_or_path: str,
        trust_remote_code: bool = False,
        colbert_dim: int = -1,
        cache_dir: str = None,
        torch_dtype: torch.dtype | None = None,
    ):
        cache_folder = os.getenv("HF_HUB_CACHE", None) if cache_dir is None else cache_dir
        if not os.path.exists(model_name_or_path):
            model_name_or_path = snapshot_download(
                repo_id=model_name_or_path,
                cache_dir=cache_folder,
                ignore_patterns=["flax_model.msgpack", "rust_model.ot", "tf_model.h5"],
            )

        model = AutoModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_folder,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        colbert_linear = torch.nn.Linear(
            in_features=model.config.hidden_size,
            out_features=model.config.hidden_size if colbert_dim <= 0 else colbert_dim,
            dtype=torch_dtype,
        )
        sparse_linear = torch.nn.Linear(
            in_features=model.config.hidden_size,
            out_features=1,
            dtype=torch_dtype,
        )

        colbert_model_path = os.path.join(model_name_or_path, "colbert_linear.pt")
        sparse_model_path = os.path.join(model_name_or_path, "sparse_linear.pt")
        if os.path.exists(colbert_model_path) and os.path.exists(sparse_model_path):
            colbert_state_dict = torch.load(colbert_model_path, map_location="cpu", weights_only=True)
            sparse_state_dict = torch.load(sparse_model_path, map_location="cpu", weights_only=True)
            colbert_linear.load_state_dict(colbert_state_dict)
            sparse_linear.load_state_dict(sparse_state_dict)

        return {
            "model": model,
            "colbert_linear": colbert_linear,
            "sparse_linear": sparse_linear,
        }

    EncoderOnlyEmbedderM3Runner.get_model = staticmethod(_patched_get_model)


def extract_doc_id(doc_out_dir: str) -> str:
    name = Path(doc_out_dir).name
    if "__" in name:
        return name.split("__")[-1][:32]
    return name[:32]


def extract_doc_name(doc_out_dir: str) -> str:
    return Path(doc_out_dir).name[:256]


def build_schema(dense_dim: int) -> CollectionSchema:
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="block_id", dtype=DataType.INT64),
        FieldSchema(name="text_hash", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="page_idx", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="lang", dtype=DataType.VARCHAR, max_length=8),
        FieldSchema(name="bge_m3_lexical_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
    ]
    return CollectionSchema(
        fields=fields,
        description="unlabeled text blocks with official BGE-M3 native hybrid fields",
    )


def batched(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def iter_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def build_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "bbox_norm1000": item.get("bbox_norm1000"),
        "coord_sys": item.get("coord_sys"),
        "mineru_type": item.get("mineru_type"),
        "type": item.get("type"),
        "is_noise": item.get("is_noise"),
        "mineru_backend": item.get("mineru_backend"),
        "mineru_version": item.get("mineru_version"),
        "canonical_view": item.get("canonical_view"),
        "extra": item.get("extra") or {},
    }


def prepare_row(pk: int, item: Dict[str, Any]) -> Dict[str, Any]:
    text = clean_text(str(item.get("text") or ""))[:8192]
    doc_out_dir = str(item.get("doc_out_dir") or "")
    doc_id = extract_doc_id(doc_out_dir)
    doc_name = extract_doc_name(doc_out_dir)
    lang = detect_lang(text)
    return {
        "pk": int(pk),
        "block_id": int(pk),
        "text_hash": str(item.get("content_hash") or "")[:64],
        "doc_id": doc_id,
        "doc_name": doc_name,
        "page_idx": int(item.get("page_idx") or 0),
        "text": text,
        "summary": text[:512],
        "metadata": build_metadata(item),
        "lang": lang[:8],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-jsonl",
        default="/mnt/data_1/yds/多模态/data_house/test_ingest_188w_20260508_bgem3/assetpack_resume552_20260508.text_only_denoised.jsonl",
    )
    ap.add_argument(
        "--milvus-db",
        default="/mnt/data_1/yds/多模态/data_house/test_ingest_188w_20260508_bgem3/u_bgem3_hybrid.db",
    )
    ap.add_argument("--collection", default="text_blocks_bge_m3_native_hybrid")
    ap.add_argument(
        "--model-path",
        default="/mnt/data_1/yds/RAG/Hybrid_milvus/总版/pretrained_models/BAAI/bge-m3",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--insert-batch-size", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--expected-rows", type=int, default=1318084)
    ap.add_argument("--drop-collection", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.milvus_db), exist_ok=True)
    lock_path = os.path.join(
        os.path.dirname(args.milvus_db),
        f".{os.path.basename(args.milvus_db)}.lock",
    )
    if os.path.exists(lock_path):
        try:
            os.remove(lock_path)
            print(f"[info] removed stale lock: {lock_path}", flush=True)
        except Exception:
            pass

    print(
        f"[start] input={args.input_jsonl} db={args.milvus_db} collection={args.collection}",
        flush=True,
    )
    patch_flagembedding_dtype_bug()
    model = BGEM3FlagModel(
        args.model_path,
        normalize_embeddings=True,
        use_fp16=str(args.device).startswith("cuda"),
        devices=args.device,
        batch_size=int(args.batch_size),
        passage_max_length=int(args.max_length),
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    print(f"[model] device={args.device} batch_size={args.batch_size}", flush=True)

    connections.connect(alias="default", uri=args.milvus_db)
    if utility.has_collection(args.collection):
        if args.drop_collection:
            Collection(args.collection).drop()
            print(f"[step] dropped existing collection={args.collection}", flush=True)
        else:
            raise RuntimeError(
                f"collection exists: {args.collection} (use --drop-collection)"
            )

    collection = Collection(name=args.collection, schema=build_schema(dense_dim=1024))
    print(f"[step] created collection={args.collection}", flush=True)

    total_expected = int(args.expected_rows)
    rows_buffer: List[Dict[str, Any]] = []
    texts_buffer: List[str] = []
    written = 0
    seen = 0
    t0 = time.time()
    encode_seconds = 0.0
    insert_seconds = 0.0

    def flush_batch() -> None:
        nonlocal written, encode_seconds, insert_seconds, rows_buffer, texts_buffer
        if not rows_buffer:
            return

        t_enc0 = time.time()
        encode_batch_size = max(int(args.batch_size) + 1, len(texts_buffer) + 1)
        out = model.encode(
            texts_buffer,
            batch_size=encode_batch_size,
            max_length=int(args.max_length),
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        encode_seconds += time.time() - t_enc0

        dense_vecs = out["dense_vecs"]
        lexical_weights = out["lexical_weights"]

        entities: List[Dict[str, Any]] = []
        for row, dense_vec, lex in zip(rows_buffer, dense_vecs, lexical_weights):
            sparse_vec = {int(k): float(v) for k, v in lex.items() if float(v) > 0.0}
            if not sparse_vec:
                sparse_vec = {0: 1e-12}
            entity = dict(row)
            entity["dense_vector"] = dense_vec.tolist()
            entity["bge_m3_lexical_sparse"] = sparse_vec
            entities.append(entity)

        t_ins0 = time.time()
        for chunk in batched(entities, int(args.insert_batch_size)):
            collection.insert(chunk)
            written += len(chunk)
        insert_seconds += time.time() - t_ins0

        rows_buffer = []
        texts_buffer = []

    for line_no, item in iter_jsonl(args.input_jsonl):
        text = clean_text(str(item.get("text") or ""))
        if not text:
            continue
        seen += 1
        rows_buffer.append(prepare_row(seen, item))
        texts_buffer.append(text[:8192])

        if len(rows_buffer) >= int(args.batch_size):
            flush_batch()
            if written and written % max(int(args.batch_size) * 20, 1) == 0:
                collection.flush()
                elapsed = time.time() - t0
                speed = written / max(elapsed, 1e-6)
                pct = (written / total_expected * 100.0) if total_expected > 0 else 0.0
                eta_min = (
                    (max(total_expected - written, 0) / max(speed, 1e-6)) / 60.0
                    if total_expected > 0
                    else 0.0
                )
                print(
                    f"[progress] line={line_no} written={written} pct={pct:.2f}% "
                    f"elapsed={elapsed/60:.1f}m speed={speed:.1f} rows/s "
                    f"encode={encode_seconds/60:.1f}m insert={insert_seconds/60:.1f}m "
                    f"eta={eta_min:.1f}m",
                    flush=True,
                )
        if args.max_rows > 0 and seen >= int(args.max_rows):
            print(f"[stop] max_rows={args.max_rows}", flush=True)
            break

    flush_batch()
    collection.flush()
    print("[step] building indexes", flush=True)
    collection.create_index(
        field_name="dense_vector",
        index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}},
    )
    print("[index] dense_vector ok", flush=True)
    collection.create_index(
        field_name="bge_m3_lexical_sparse",
        index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP", "params": {}},
    )
    print("[index] bge_m3_lexical_sparse ok", flush=True)

    elapsed = time.time() - t0
    print("=" * 60, flush=True)
    print("[done]", flush=True)
    print(f"collection={args.collection}", flush=True)
    print(f"written={written}", flush=True)
    print(f"elapsed={elapsed/60:.2f} min speed={written/max(elapsed,1e-6):.2f} rows/s", flush=True)
    print("fields=dense_vector,bge_m3_lexical_sparse", flush=True)
    print("=" * 60, flush=True)
    connections.disconnect("default")


if __name__ == "__main__":
    main()
