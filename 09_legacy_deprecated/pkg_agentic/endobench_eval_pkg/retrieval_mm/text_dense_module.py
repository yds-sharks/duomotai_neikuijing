#!/usr/bin/env python3
"""Extract text vector records from multimodal_samples with BGE-m3."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

try:
    from .common import DEFAULT_BGE_MODEL_PATH, DEFAULT_DB_PATH, image_ids_from_images, load_db_rows
except ImportError:
    from common import DEFAULT_BGE_MODEL_PATH, DEFAULT_DB_PATH, image_ids_from_images, load_db_rows


TEXT_SELECT_SQL = """
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
"""


def load_bge_m3_encoder(model_path: str = DEFAULT_BGE_MODEL_PATH, device: str = "cpu"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_path, device=device)


def text_record_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    retrieval_flags = sample["retrieval_flags"]
    retrieval_meta = sample["retrieval_meta"]
    return {
        "id": sample["sample_id"],
        "sample_id": sample["sample_id"],
        "group_id": sample["group_id"],
        "image_ids": image_ids_from_images(sample["images"]),
        "text": sample["text"],
        "source_type": sample["source_type"],
        "text_role": sample["text_role"],
        "have_image": bool(sample["have_image"]),
        "body_site_main": retrieval_meta.get("body_site_main", ""),
        "body_site_all": retrieval_meta.get("body_site_all", []),
        "knowledge_type_main": retrieval_meta.get("knowledge_type_main", ""),
        "organ_tags": sample["organ_tags"],
        "primary_knowledge_type": sample["primary_knowledge_type"],
        "secondary_knowledge_types": sample["secondary_knowledge_types"],
        "is_general": bool(retrieval_flags.get("is_general", False)),
        "is_mixed": bool(retrieval_flags.get("is_mixed", False)),
        "is_key_knowledge": bool(retrieval_flags.get("is_key_knowledge", False)),
        "doc_id": sample.get("doc_id", ""),
        "doc_name": sample.get("doc_name", ""),
        "page_idx": sample.get("page_idx"),
        "block_id": sample.get("block_id"),
    }


def fetch_text_vector_records(
    db_path: str = DEFAULT_DB_PATH,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    sql = TEXT_SELECT_SQL
    params: Sequence[Any] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    samples = load_db_rows(db_path, sql, params)
    return [text_record_from_sample(sample) for sample in samples]


def embed_texts(
    texts: Sequence[str],
    *,
    encoder,
    batch_size: int = 32,
) -> List[List[float]]:
    vectors = encoder.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def build_text_vector_records(
    db_path: str = DEFAULT_DB_PATH,
    *,
    model_path: str = DEFAULT_BGE_MODEL_PATH,
    device: str = "cpu",
    batch_size: int = 32,
    limit: int | None = None,
    with_vectors: bool = True,
) -> List[Dict[str, Any]]:
    records = fetch_text_vector_records(db_path=db_path, limit=limit)
    if not with_vectors or not records:
        return records

    encoder = load_bge_m3_encoder(model_path=model_path, device=device)
    vectors = embed_texts([record["text"] for record in records], encoder=encoder, batch_size=batch_size)
    for record, vector in zip(records, vectors):
        record["embedding"] = vector
    return records
