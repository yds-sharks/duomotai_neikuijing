#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为无标签 Milvus-Lite 库重建一个新集合，新增 BGE-M3 原生 lexical sparse 字段。

设计目标：
1. 保留现有 dense_vector / sparse_vector / sparse_vector_v2；
2. 新增 bge_m3_lexical_sparse；
3. 为 dense_vector 与 bge_m3_lexical_sparse 建索引；
4. 作为完整 BGE-M3 native hybrid 的底座集合使用。

注意：
- Milvus-Lite 不支持给已有集合原地追加向量字段，因此采用新集合方案。
- 当前脚本只处理无标签 120w 集合。
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch import nn
import ujson
from google.protobuf.internal.decoder import _DecodeVarint32
from transformers import AutoConfig, AutoModel, AutoTokenizer
from pymilvus.client import entity_helper
from pymilvus.grpc_gen import schema_pb2
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


def build_dst_schema(dense_dim: int) -> CollectionSchema:
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="block_id", dtype=DataType.INT64),
        FieldSchema(name="text_hash", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="page_idx", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="lang", dtype=DataType.VARCHAR, max_length=8),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="sparse_vector_v2", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="bge_m3_lexical_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
    ]
    return CollectionSchema(fields=fields, description="unlabeled text blocks with BGE-M3 lexical sparse")


class BGEM3LexicalEncoder:
    def __init__(self, model_path: str, device: str, use_fp16: bool = True):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.encoder = AutoModel.from_pretrained(model_path).to(device).eval()

        cfg = AutoConfig.from_pretrained(model_path)
        self.sparse_linear = nn.Linear(cfg.hidden_size, 1).to(device).eval()
        sparse_linear_path = Path(model_path) / "sparse_linear.pt"
        state = torch.load(sparse_linear_path, map_location="cpu")
        self.sparse_linear.load_state_dict(state)

        if use_fp16 and device.startswith("cuda"):
            self.encoder = self.encoder.half()
            self.sparse_linear = self.sparse_linear.half()

        self.special_token_ids = set(self.tokenizer.all_special_ids)

    def _process_lexical_weights(
        self,
        input_ids: List[int],
        attention_mask: List[int],
        token_weights: List[float],
    ) -> Dict[int, float]:
        lexical: Dict[int, float] = {}
        for token_id, mask, weight in zip(input_ids, attention_mask, token_weights):
            if not mask:
                continue
            if token_id in self.special_token_ids:
                continue
            if weight <= 0.0:
                continue
            token_id = int(token_id)
            prev = lexical.get(token_id, 0.0)
            if weight > prev:
                lexical[token_id] = float(weight)
        return lexical

    def encode_sparse(
        self,
        texts: List[str],
        max_length: int,
    ) -> List[Dict[int, float]]:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = self.encoder(**batch, return_dict=True)
            hidden = outputs.last_hidden_state
            token_weights = torch.relu(self.sparse_linear(hidden)).squeeze(-1)

        input_ids = batch["input_ids"].cpu().tolist()
        attention_mask = batch["attention_mask"].cpu().tolist()
        token_weights = token_weights.float().cpu().tolist()

        sparse_maps: List[Dict[int, float]] = []
        for ids, mask, weights in zip(input_ids, attention_mask, token_weights):
            sparse_maps.append(self._process_lexical_weights(ids, mask, weights))
        return sparse_maps


def batched(it: Iterable[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    bucket: List[Dict[str, Any]] = []
    for row in it:
        bucket.append(row)
        if len(bucket) >= size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def open_sqlite_ro(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = bytes
    return conn


def parse_blob_row(blob: bytes) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    i = 0
    while i < len(blob):
        if blob[i] != 0x0A:
            break
        i += 1
        size, next_i = _DecodeVarint32(blob, i)
        i = next_i
        msg_bytes = blob[i : i + size]
        i += size

        fd = schema_pb2.FieldData()
        fd.ParseFromString(msg_bytes)
        name, dtype = fd.field_name, fd.type

        if name in ("RowID", "Timestamp"):
            continue
        if dtype == DataType.INT64:
            row[name] = int(fd.scalars.long_data.data[0])
            continue
        if dtype in (DataType.INT8, DataType.INT16, DataType.INT32):
            row[name] = int(fd.scalars.int_data.data[0])
            continue
        if dtype == DataType.VARCHAR:
            row[name] = fd.scalars.string_data.data[0]
            continue
        if dtype == DataType.JSON:
            row[name] = ujson.loads(fd.scalars.json_data.data[0])
            continue
        if dtype == DataType.FLOAT_VECTOR:
            row[name] = list(fd.vectors.float_vector.data)
            continue
        if dtype == DataType.SPARSE_FLOAT_VECTOR:
            row[name] = list(entity_helper.sparse_proto_to_rows(fd.vectors.sparse_float_vector, 0, 1))[0]
            continue
        raise RuntimeError(f"unsupported field type in sqlite row: {name} dtype={dtype}")
    return row


def sqlite_blob_batches(db_path: str, table_name: str, fetch_size: int) -> Iterable[List[bytes]]:
    last_id = 0
    while True:
        conn = open_sqlite_ro(db_path)
        cur = conn.cursor()
        cur.execute(
            f"SELECT id, data FROM {table_name} WHERE id > ? ORDER BY id LIMIT ?",
            (last_id, fetch_size),
        )
        rows = cur.fetchall()
        conn.close()
        if not rows:
            break
        last_id = int(rows[-1][0])
        yield [r[1] for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--src", default="text_blocks_sparse_v2")
    ap.add_argument("--dst", default="text_blocks_bge_m3_hybrid_v1")
    ap.add_argument("--model-path", default="/mnt/data_1/yds/RAG/Hybrid_milvus/总版/pretrained_models/BAAI/bge-m3")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--insert-batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default="")
    ap.add_argument("--drop-dst", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    device = args.device.strip() or ("cuda:0" if torch.cuda.is_available() else "cpu")

    lock_path = os.path.join(os.path.dirname(args.db), f".{os.path.basename(args.db)}.lock")
    if os.path.exists(lock_path):
        try:
            os.remove(lock_path)
            print(f"[info] removed stale lock: {lock_path}", flush=True)
        except Exception:
            pass

    print(f"[start] db={args.db} src={args.src} dst={args.dst}", flush=True)
    sqlite_conn = open_sqlite_ro(args.db)
    cur = sqlite_conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {args.src}")
    src_count = int(cur.fetchone()[0])
    sqlite_conn.close()
    print(f"[step] sqlite source ready rows={src_count}", flush=True)

    connections.connect(alias="default", uri=args.db)
    print("[step] milvus connected", flush=True)

    src = Collection(args.src)
    print(f"[step] opened src schema={args.src} entities={src_count}", flush=True)

    dense_dim = None
    for f in src.schema.fields:
        if f.name == "dense_vector":
            dense_dim = int(f.params.get("dim"))
            break
    if dense_dim is None:
        raise RuntimeError("source collection has no dense_vector")

    if utility.has_collection(args.dst):
        if args.drop_dst:
            Collection(args.dst).drop()
            print(f"[step] dropped existing dst={args.dst}", flush=True)
        else:
            raise RuntimeError(f"destination collection exists: {args.dst} (use --drop-dst)")

    dst = Collection(name=args.dst, schema=build_dst_schema(dense_dim))
    print(f"[step] created dst={args.dst}", flush=True)

    encoder = BGEM3LexicalEncoder(
        model_path=args.model_path,
        device=device,
        use_fp16=True,
    )
    print(f"[model] device={device} model={args.model_path}", flush=True)

    output_fields = [
        "pk",
        "block_id",
        "text_hash",
        "doc_id",
        "doc_name",
        "page_idx",
        "text",
        "summary",
        "metadata",
        "lang",
        "sparse_vector",
        "sparse_vector_v2",
        "dense_vector",
    ]
    _ = output_fields
    print(f"[step] sqlite iterator ready batch_size={args.batch_size} max_length={args.max_length}", flush=True)

    written = 0
    batch_no = 0
    t0 = time.time()
    encode_seconds = 0.0

    for blobs in sqlite_blob_batches(args.db, args.src, int(args.batch_size)):
        rows = [parse_blob_row(blob) for blob in blobs]

        texts = [(r.get("text") or "").strip()[:8192] for r in rows]
        t_enc0 = time.time()
        lexical_sparse = encoder.encode_sparse(texts, max_length=int(args.max_length))
        encode_seconds += time.time() - t_enc0

        entities: List[Dict[str, Any]] = []
        for r, lex_sp, text in zip(rows, lexical_sparse, texts):
            entities.append(
                {
                    "pk": int(r["pk"]),
                    "block_id": int(r.get("block_id") or 0),
                    "text_hash": r.get("text_hash") or "",
                    "doc_id": r.get("doc_id") or "",
                    "doc_name": r.get("doc_name") or "",
                    "page_idx": int(r.get("page_idx") or 0),
                    "text": text,
                    "summary": (r.get("summary") or "")[:512],
                    "metadata": r.get("metadata") or {},
                    "lang": (r.get("lang") or "")[:8],
                    "sparse_vector": r.get("sparse_vector") or {},
                    "sparse_vector_v2": r.get("sparse_vector_v2") or {},
                    "bge_m3_lexical_sparse": lex_sp,
                    "dense_vector": r.get("dense_vector") or [],
                }
            )

        if entities:
            for insert_rows in batched(entities, int(args.insert_batch_size)):
                dst.insert(insert_rows)
                written += len(insert_rows)

        batch_no += 1
        if batch_no % 5 == 0:
            dst.flush()
            elapsed = time.time() - t0
            speed = written / max(elapsed, 1e-6)
            pct = (written / src_count * 100.0) if src_count else 0.0
            eta_min = ((src_count - written) / max(speed, 1e-6)) / 60.0 if src_count and written < src_count else 0.0
            print(
                f"[progress] batch={batch_no} written={written} "
                f"pct={pct:.2f}% elapsed={elapsed/60:.1f}m speed={speed:.1f} rows/s "
                f"encode={encode_seconds/60:.1f}m eta={eta_min:.1f}m"
            , flush=True)

        if args.max_rows > 0 and written >= args.max_rows:
            print(f"[stop] max_rows={args.max_rows}", flush=True)
            break

    dst.flush()
    print("[step] creating indexes", flush=True)

    try:
        dst.create_index(
            field_name="dense_vector",
            index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}},
        )
        print("[index] dense_vector ok", flush=True)
    except Exception as e:
        print(f"[index] dense_vector warn: {e}", flush=True)

    try:
        dst.create_index(
            field_name="bge_m3_lexical_sparse",
            index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP", "params": {}},
        )
        print("[index] bge_m3_lexical_sparse ok", flush=True)
    except Exception as e:
        print(f"[index] bge_m3_lexical_sparse warn: {e}", flush=True)

    elapsed = time.time() - t0
    print("=" * 60, flush=True)
    print("[done]", flush=True)
    print(f"src={args.src} src_entities={src_count}", flush=True)
    print(f"dst={args.dst} dst_entities={dst.num_entities}", flush=True)
    print(f"elapsed={elapsed/60:.2f} min speed={written/max(elapsed,1e-6):.2f} rows/s", flush=True)
    print("new field: bge_m3_lexical_sparse", flush=True)
    print("=" * 60, flush=True)

    connections.disconnect("default")


if __name__ == "__main__":
    main()
