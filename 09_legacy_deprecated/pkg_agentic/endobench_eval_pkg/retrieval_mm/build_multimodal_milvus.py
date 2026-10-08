#!/usr/bin/env python3
"""Build text_vector_index and image_vector_index from multimodal_samples.db."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

try:
    from .common import DEFAULT_DB_PATH, DEFAULT_QWEN3_VL_MODEL_PATH, load_row_json_fields
    from .image_dense_module import embed_images, image_record_from_sample, load_qwen3vl_image_encoder
    from .text_dense_module import embed_texts, load_bge_m3_encoder, text_record_from_sample
except ImportError:
    from common import DEFAULT_DB_PATH, DEFAULT_QWEN3_VL_MODEL_PATH, load_row_json_fields
    from image_dense_module import embed_images, image_record_from_sample, load_qwen3vl_image_encoder
    from text_dense_module import embed_texts, load_bge_m3_encoder, text_record_from_sample


DEFAULT_MILVUS_DB_PATH = os.environ.get(
    "MILVUS_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "assets", "multimodal_vector_indexes.db"),
)
TEXT_COLLECTION_NAME = "text_vector_index"
IMAGE_COLLECTION_NAME = "image_vector_index"
TEXT_VECTOR_DIM = 1024
IMAGE_VECTOR_DIM = 4096

TEXT_BATCH_SQL = """
SELECT
    sample_id,
    group_id,
    source_type,
    have_image,
    doc_id,
    doc_name,
    page_idx,
    block_id,
    text,
    text_role,
    organ_tags,
    primary_knowledge_type,
    secondary_knowledge_types,
    retrieval_flags,
    retrieval_meta,
    images
FROM multimodal_samples
WHERE text IS NOT NULL
  AND TRIM(text) != ''
ORDER BY sample_id
LIMIT ? OFFSET ?
"""

IMAGE_BATCH_SQL = """
SELECT
    sample_id,
    group_id,
    doc_id,
    organ_tags,
    primary_knowledge_type,
    retrieval_flags,
    retrieval_meta,
    images
FROM multimodal_samples
WHERE images IS NOT NULL
  AND TRIM(images) != '[]'
ORDER BY sample_id
LIMIT ? OFFSET ?
"""


def json_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    return value


def _load_db_batch(db_path: str, sql: str, params: Sequence[Any]) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql, tuple(params))
        return [load_row_json_fields(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def build_text_schema() -> CollectionSchema:
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True, auto_id=False),
        FieldSchema(name="sample_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="group_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="image_ids", dtype=DataType.JSON),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="text_role", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="have_image", dtype=DataType.BOOL),
        FieldSchema(name="body_site_main", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="body_site_all", dtype=DataType.JSON),
        FieldSchema(name="knowledge_type_main", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="organ_tags", dtype=DataType.JSON),
        FieldSchema(name="primary_knowledge_type", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="secondary_knowledge_types", dtype=DataType.JSON),
        FieldSchema(name="is_general", dtype=DataType.BOOL),
        FieldSchema(name="is_mixed", dtype=DataType.BOOL),
        FieldSchema(name="is_key_knowledge", dtype=DataType.BOOL),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="page_idx", dtype=DataType.INT64),
        FieldSchema(name="block_id", dtype=DataType.INT64),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=TEXT_VECTOR_DIM),
    ]
    return CollectionSchema(fields, description="Text vector index from multimodal_samples")


def build_image_schema() -> CollectionSchema:
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True, auto_id=False),
        FieldSchema(name="image_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="parent_sample_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="group_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="image_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="is_primary", dtype=DataType.BOOL),
        FieldSchema(name="body_site_main", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="body_site_all", dtype=DataType.JSON),
        FieldSchema(name="knowledge_type_main", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="organ_tags", dtype=DataType.JSON),
        FieldSchema(name="primary_knowledge_type", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="is_general", dtype=DataType.BOOL),
        FieldSchema(name="is_mixed", dtype=DataType.BOOL),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=IMAGE_VECTOR_DIM),
    ]
    return CollectionSchema(fields, description="Image vector index from multimodal_samples")


def recreate_collection(collection_name: str, schema: CollectionSchema) -> Collection:
    if utility.has_collection(collection_name):
        Collection(collection_name).drop()
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(
        field_name="dense_vector",
        index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}},
    )
    return collection


def text_row_to_milvus(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": record["id"],
        "sample_id": record["sample_id"],
        "group_id": record["group_id"],
        "image_ids": json_value(record["image_ids"]),
        "text": record["text"],
        "source_type": record["source_type"],
        "text_role": record["text_role"],
        "have_image": bool(record["have_image"]),
        "body_site_main": record["body_site_main"],
        "body_site_all": json_value(record["body_site_all"]),
        "knowledge_type_main": record["knowledge_type_main"],
        "organ_tags": json_value(record["organ_tags"]),
        "primary_knowledge_type": record["primary_knowledge_type"],
        "secondary_knowledge_types": json_value(record["secondary_knowledge_types"]),
        "is_general": bool(record["is_general"]),
        "is_mixed": bool(record["is_mixed"]),
        "is_key_knowledge": bool(record["is_key_knowledge"]),
        "doc_id": record["doc_id"] or "",
        "doc_name": record["doc_name"] or "",
        "page_idx": int(record["page_idx"] if record["page_idx"] is not None else -1),
        "block_id": int(record["block_id"] if record["block_id"] is not None else -1),
        "dense_vector": record["embedding"],
    }


def image_row_to_milvus(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": record["id"],
        "image_id": record["image_id"],
        "parent_sample_id": record["parent_sample_id"],
        "group_id": record["group_id"],
        "image_path": record["image_path"],
        "image_type": record["image_type"],
        "is_primary": bool(record["is_primary"]),
        "body_site_main": record["body_site_main"],
        "body_site_all": json_value(record["body_site_all"]),
        "knowledge_type_main": record["knowledge_type_main"],
        "organ_tags": json_value(record["organ_tags"]),
        "primary_knowledge_type": record["primary_knowledge_type"],
        "is_general": bool(record["is_general"]),
        "is_mixed": bool(record["is_mixed"]),
        "doc_id": record["doc_id"] or "",
        "dense_vector": record["embedding"],
    }


def build_text_collection(
    *,
    db_path: str,
    collection: Collection,
    model_path: str,
    device: str,
    fetch_batch_size: int,
    embed_batch_size: int,
    limit: int | None = None,
) -> int:
    encoder = load_bge_m3_encoder(model_path=model_path, device=device)
    inserted = 0
    offset = 0

    while True:
        remaining = fetch_batch_size if limit is None else min(fetch_batch_size, max(limit - inserted, 0))
        if remaining <= 0:
            break
        samples = _load_db_batch(db_path, TEXT_BATCH_SQL, (remaining, offset))
        if not samples:
            break

        records = [text_record_from_sample(sample) for sample in samples]
        vectors = embed_texts(
            [record["text"] for record in records],
            encoder=encoder,
            batch_size=embed_batch_size,
        )
        if vectors:
            if len(vectors[0]) != TEXT_VECTOR_DIM:
                raise ValueError(f"Unexpected text vector dim: {len(vectors[0])} != {TEXT_VECTOR_DIM}")
        for record, vector in zip(records, vectors):
            record["embedding"] = vector

        collection.insert([text_row_to_milvus(record) for record in records])
        inserted += len(records)
        offset += len(samples)
        print(f"[text] inserted {inserted}")

    collection.flush()
    collection.load()
    return inserted


def build_image_collection(
    *,
    db_path: str,
    collection: Collection,
    model_path: str,
    device: str,
    torch_dtype: str,
    fetch_batch_size: int,
    embed_batch_size: int,
    limit: int | None = None,
) -> int:
    processor, model = load_qwen3vl_image_encoder(
        model_path=model_path,
        device=device,
        torch_dtype=torch_dtype,
    )
    inserted = 0
    sample_offset = 0

    while True:
        sample_remaining = fetch_batch_size if limit is None else min(fetch_batch_size, max(limit - inserted, 0))
        if sample_remaining <= 0:
            break
        samples = _load_db_batch(db_path, IMAGE_BATCH_SQL, (sample_remaining, sample_offset))
        if not samples:
            break

        records: List[Dict[str, Any]] = []
        for sample in samples:
            records.extend(image_record_from_sample(sample))
        if not records:
            sample_offset += len(samples)
            continue

        for start in range(0, len(records), embed_batch_size):
            batch = records[start : start + embed_batch_size]
            vectors = embed_images(
                [record["image_path"] for record in batch],
                processor=processor,
                model=model,
                device=device,
                pooling="mean",
            )
            if vectors:
                if len(vectors[0]) != IMAGE_VECTOR_DIM:
                    raise ValueError(f"Unexpected image vector dim: {len(vectors[0])} != {IMAGE_VECTOR_DIM}")
            for record, vector in zip(batch, vectors):
                record["embedding"] = vector
            collection.insert([image_row_to_milvus(record) for record in batch])
            inserted += len(batch)
            print(f"[image] inserted {inserted}")
            if limit is not None and inserted >= limit:
                break

        sample_offset += len(samples)
        if limit is not None and inserted >= limit:
            break

    collection.flush()
    collection.load()
    return inserted


def _write_image_shards(
    *,
    db_path: str,
    fetch_batch_size: int,
    shard_paths: Sequence[str],
    limit: int | None = None,
) -> int:
    shard_files = [open(path, "w", encoding="utf-8") for path in shard_paths]
    sample_offset = 0
    record_count = 0
    try:
        while True:
            sample_remaining = fetch_batch_size if limit is None else min(fetch_batch_size, max(limit - record_count, 0))
            if sample_remaining <= 0:
                break
            samples = _load_db_batch(db_path, IMAGE_BATCH_SQL, (sample_remaining, sample_offset))
            if not samples:
                break

            for sample in samples:
                for record in image_record_from_sample(sample):
                    shard_index = record_count % len(shard_files)
                    shard_files[shard_index].write(json.dumps(record, ensure_ascii=False) + "\n")
                    record_count += 1
                    if limit is not None and record_count >= limit:
                        break
                if limit is not None and record_count >= limit:
                    break

            sample_offset += len(samples)
            if limit is not None and record_count >= limit:
                break
    finally:
        for shard_file in shard_files:
            shard_file.close()

    return record_count


def count_jsonl_lines(path: str) -> int:
    total = 0
    with open(path, "r", encoding="utf-8") as handle:
        for _ in handle:
            total += 1
    return total


def count_pickle_records(path: str) -> int:
    if not os.path.exists(path):
        return 0

    total = 0
    with open(path, "rb") as handle:
        while True:
            try:
                batch = pickle.load(handle)
            except EOFError:
                break
            total += len(batch)
    return total


def insert_pickle_records(
    *,
    collection: Collection,
    output_paths: Sequence[str],
) -> int:
    inserted = 0
    for output_path in output_paths:
        if not os.path.exists(output_path):
            continue
        with open(output_path, "rb") as handle:
            while True:
                try:
                    batch = pickle.load(handle)
                except EOFError:
                    break
                collection.insert([image_row_to_milvus(record) for record in batch])
                inserted += len(batch)
                print(f"[image] inserted {inserted}", flush=True)

    collection.flush()
    collection.load()
    return inserted


def run_image_worker(
    *,
    shard_path: str,
    output_path: str,
    model_path: str,
    device: str,
    torch_dtype: str,
    batch_size: int,
    resume_embedded_count: int = 0,
) -> int:
    processor, model = load_qwen3vl_image_encoder(
        model_path=model_path,
        device=device,
        torch_dtype=torch_dtype,
    )
    with open(shard_path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    embedded = resume_embedded_count
    skipped = 0
    file_mode = "ab" if resume_embedded_count > 0 and os.path.exists(output_path) else "wb"
    with open(output_path, file_mode) as handle:
        for start in range(resume_embedded_count, len(records), batch_size):
            batch = records[start : start + batch_size]
            valid_batch = [record for record in batch if os.path.exists(record["image_path"])]
            skipped += len(batch) - len(valid_batch)
            if not valid_batch:
                continue

            vectors = embed_images(
                [record["image_path"] for record in valid_batch],
                processor=processor,
                model=model,
                device=device,
                pooling="mean",
            )
            if vectors:
                if len(vectors[0]) != IMAGE_VECTOR_DIM:
                    raise ValueError(f"Unexpected image vector dim: {len(vectors[0])} != {IMAGE_VECTOR_DIM}")
            for record, vector in zip(valid_batch, vectors):
                record["embedding"] = vector

            pickle.dump(valid_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
            embedded += len(valid_batch)
            print(f"[image-worker {device}] embedded {embedded}/{len(records)} skipped={skipped}", flush=True)

    return embedded


def build_image_collection_parallel(
    *,
    db_path: str,
    collection: Collection,
    model_path: str,
    devices: Sequence[str],
    torch_dtype: str,
    fetch_batch_size: int,
    embed_batch_size: int,
    limit: int | None = None,
    resume_temp_dir: str = "",
) -> int:
    if resume_temp_dir:
        temp_dir = resume_temp_dir
        shard_paths = [os.path.join(temp_dir, f"image_records_shard_{idx}.jsonl") for idx in range(len(devices))]
        output_paths = [os.path.join(temp_dir, f"image_embeddings_shard_{idx}.pkl") for idx in range(len(devices))]
        total_records = sum(count_jsonl_lines(path) for path in shard_paths)
        print(f"[image-parallel] resuming from {temp_dir}", flush=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="image_parallel_", dir=os.path.dirname(DEFAULT_MILVUS_DB_PATH) or None)
        shard_paths = [os.path.join(temp_dir, f"image_records_shard_{idx}.jsonl") for idx in range(len(devices))]
        output_paths = [os.path.join(temp_dir, f"image_embeddings_shard_{idx}.pkl") for idx in range(len(devices))]
        total_records = _write_image_shards(
            db_path=db_path,
            fetch_batch_size=fetch_batch_size,
            shard_paths=shard_paths,
            limit=limit,
        )
        print(f"[image-parallel] prepared {total_records} image records across {len(devices)} shards", flush=True)

    processes: List[subprocess.Popen[str]] = []
    try:
        existing_counts = [count_pickle_records(path) for path in output_paths]
        for idx, existing_count in enumerate(existing_counts):
            print(
                f"[image-parallel] shard={idx} existing={existing_count}/{count_jsonl_lines(shard_paths[idx])}",
                flush=True,
            )

        for idx, device in enumerate(devices):
            shard_total = count_jsonl_lines(shard_paths[idx])
            if existing_counts[idx] >= shard_total:
                print(f"[image-parallel] shard={idx} already complete, skipping worker launch", flush=True)
                continue
            cmd = [
                sys.executable,
                __file__,
                "--image-worker-shard",
                shard_paths[idx],
                "--image-worker-output",
                output_paths[idx],
                "--image-worker-device",
                device,
                "--image-worker-resume-count",
                str(existing_counts[idx]),
                "--image-model-path",
                model_path,
                "--image-torch-dtype",
                torch_dtype,
                "--image-embed-batch-size",
                str(embed_batch_size),
            ]
            processes.append(subprocess.Popen(cmd))

        for process in processes:
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"Image worker failed with exit code {return_code}")

        return insert_pickle_records(collection=collection, output_paths=output_paths)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multimodal vector indexes in Milvus Lite")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to multimodal_samples.db")
    parser.add_argument(
        "--milvus-db-path",
        default=DEFAULT_MILVUS_DB_PATH,
        help="Target Milvus Lite database path",
    )
    parser.add_argument("--text-model-path", default=os.environ.get("BGE_M3_PATH", "BAAI/bge-m3"))
    parser.add_argument("--image-model-path", default=DEFAULT_QWEN3_VL_MODEL_PATH)
    parser.add_argument("--text-device", default="cuda", help="Device for BGE text embedding")
    parser.add_argument("--image-device", default="cuda", help="Device for Qwen3-VL image embedding")
    parser.add_argument("--image-torch-dtype", default="auto", help="Torch dtype for Qwen3-VL image encoder")
    parser.add_argument("--image-devices", default="", help="Comma-separated GPU devices for parallel image embedding")
    parser.add_argument("--text-fetch-batch-size", type=int, default=512)
    parser.add_argument("--text-embed-batch-size", type=int, default=64)
    parser.add_argument("--image-fetch-batch-size", type=int, default=256)
    parser.add_argument("--image-embed-batch-size", type=int, default=4)
    parser.add_argument("--text-limit", type=int, default=None)
    parser.add_argument("--image-limit", type=int, default=None)
    parser.add_argument("--skip-text", action="store_true", help="Reuse existing text collection without rebuilding it")
    parser.add_argument("--skip-image", action="store_true", help="Skip image collection build")
    parser.add_argument("--resume-image-temp-dir", default="", help="Resume image parallel build from an existing temp shard directory")
    parser.add_argument("--image-worker-shard", default="", help=argparse.SUPPRESS)
    parser.add_argument("--image-worker-output", default="", help=argparse.SUPPRESS)
    parser.add_argument("--image-worker-device", default="", help=argparse.SUPPRESS)
    parser.add_argument("--image-worker-resume-count", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_worker_shard:
        run_image_worker(
            shard_path=args.image_worker_shard,
            output_path=args.image_worker_output,
            model_path=args.image_model_path,
            device=args.image_worker_device or args.image_device,
            torch_dtype=args.image_torch_dtype,
            batch_size=args.image_embed_batch_size,
            resume_embedded_count=args.image_worker_resume_count,
        )
        return

    milvus_dir = os.path.dirname(args.milvus_db_path)
    if milvus_dir:
        os.makedirs(milvus_dir, exist_ok=True)
    connections.connect(alias="default", uri=args.milvus_db_path)
    try:
        if args.skip_text:
            if not utility.has_collection(TEXT_COLLECTION_NAME):
                raise ValueError(f"Cannot skip text build: missing collection {TEXT_COLLECTION_NAME}")
            text_collection = Collection(TEXT_COLLECTION_NAME)
            text_count = text_collection.num_entities
            print(f"[text] reusing existing collection with {text_count} entities", flush=True)
        else:
            text_collection = recreate_collection(TEXT_COLLECTION_NAME, build_text_schema())
            text_count = build_text_collection(
                db_path=args.db_path,
                collection=text_collection,
                model_path=args.text_model_path,
                device=args.text_device,
                fetch_batch_size=args.text_fetch_batch_size,
                embed_batch_size=args.text_embed_batch_size,
                limit=args.text_limit,
            )

        if args.skip_image:
            if not utility.has_collection(IMAGE_COLLECTION_NAME):
                raise ValueError(f"Cannot skip image build: missing collection {IMAGE_COLLECTION_NAME}")
            image_collection = Collection(IMAGE_COLLECTION_NAME)
            image_count = image_collection.num_entities
            print(f"[image] reusing existing collection with {image_count} entities", flush=True)
        else:
            image_collection = recreate_collection(IMAGE_COLLECTION_NAME, build_image_schema())
            image_devices = [device.strip() for device in args.image_devices.split(",") if device.strip()]
            if len(image_devices) > 1:
                image_count = build_image_collection_parallel(
                    db_path=args.db_path,
                    collection=image_collection,
                    model_path=args.image_model_path,
                    devices=image_devices,
                    torch_dtype=args.image_torch_dtype,
                    fetch_batch_size=args.image_fetch_batch_size,
                    embed_batch_size=args.image_embed_batch_size,
                    limit=args.image_limit,
                    resume_temp_dir=args.resume_image_temp_dir,
                )
            else:
                image_count = build_image_collection(
                    db_path=args.db_path,
                    collection=image_collection,
                    model_path=args.image_model_path,
                    device=args.image_device,
                    torch_dtype=args.image_torch_dtype,
                    fetch_batch_size=args.image_fetch_batch_size,
                    embed_batch_size=args.image_embed_batch_size,
                    limit=args.image_limit,
                )
        print(f"[done] text_entities={text_count} image_entities={image_count} db={args.milvus_db_path}")
    finally:
        connections.disconnect("default")


if __name__ == "__main__":
    main()
