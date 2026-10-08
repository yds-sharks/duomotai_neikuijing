#!/usr/bin/env python3
"""Extract image vector records from multimodal_samples with Qwen3-VL encoder only."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    from .common import DEFAULT_DB_PATH, DEFAULT_QWEN3_VL_MODEL_PATH, load_db_rows
except ImportError:
    from common import DEFAULT_DB_PATH, DEFAULT_QWEN3_VL_MODEL_PATH, load_db_rows


IMAGE_SELECT_SQL = """
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
"""


def load_qwen3vl_image_encoder(
    model_path: str = DEFAULT_QWEN3_VL_MODEL_PATH,
    device: str = "cuda",
    torch_dtype: str = "auto",
):
    """
    Load the Qwen3-VL model for image encoding only.

    Notes:
    - This must run in an environment whose transformers version supports qwen3_vl.
    - We use `get_image_features()` and do not use the decoder output or lm_head.
    """
    import torch
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    dtype = torch_dtype
    if torch_dtype == "auto":
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.eval()
    model.to(device)
    return processor, model


def image_record_from_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    retrieval_flags = sample["retrieval_flags"]
    retrieval_meta = sample["retrieval_meta"]

    records: List[Dict[str, Any]] = []
    for image in sample["images"]:
        image_id = str(image.get("image_id") or "").strip()
        image_path = str(image.get("image_path") or "").strip()
        if not image_id or not image_path:
            continue
        records.append(
            {
                "id": image_id,
                "image_id": image_id,
                "parent_sample_id": sample["sample_id"],
                "group_id": sample["group_id"],
                "image_path": image_path,
                "image_type": image.get("image_type", "unknown"),
                "is_primary": bool(image.get("is_primary", False)),
                "body_site_main": retrieval_meta.get("body_site_main", ""),
                "body_site_all": retrieval_meta.get("body_site_all", []),
                "knowledge_type_main": retrieval_meta.get("knowledge_type_main", ""),
                "organ_tags": sample["organ_tags"],
                "primary_knowledge_type": sample["primary_knowledge_type"],
                "is_general": bool(retrieval_flags.get("is_general", False)),
                "is_mixed": bool(retrieval_flags.get("is_mixed", False)),
                "doc_id": sample.get("doc_id", ""),
            }
        )
    return records


def fetch_image_vector_records(
    db_path: str = DEFAULT_DB_PATH,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    sql = IMAGE_SELECT_SQL
    params: Sequence[Any] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    samples = load_db_rows(db_path, sql, params)
    records: List[Dict[str, Any]] = []
    for sample in samples:
        records.extend(image_record_from_sample(sample))
    return records


def _load_images(image_paths: Sequence[str]):
    from PIL import Image

    images = []
    for image_path in image_paths:
        image = Image.open(Path(image_path)).convert("RGB")
        images.append(image)
    return images


def embed_images(
    image_paths: Sequence[str],
    *,
    processor,
    model,
    device: str = "cuda",
    pooling: str = "mean",
) -> List[List[float]]:
    import torch
    import torch.nn.functional as F

    if pooling != "mean":
        raise ValueError(f"Unsupported pooling mode: {pooling}")

    images = _load_images(image_paths)
    image_inputs = processor.image_processor(images=images, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device)
    image_grid_thw = image_inputs["image_grid_thw"].to(device)

    with torch.no_grad():
        _result = model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        # transformers >=4.57 returns BaseModelOutputWithDeepstackFeatures.
        # pooler_output[0] = projected image embeds (4096-dim for Qwen3-VL-8B),
        # matching the Milvus index dimension. last_hidden_state is the raw
        # 1152-dim vision-encoder output (pre-projection) — do NOT use it.
        if hasattr(_result, "pooler_output") and _result.pooler_output is not None:
            image_embeds = _result.pooler_output[0]  # [patches, 4096]
        elif hasattr(_result, "last_hidden_state"):
            image_embeds = _result.last_hidden_state
        else:
            image_embeds = _result[0] if isinstance(_result, (tuple, list)) else _result

    pooled_vectors = []
    # last_hidden_state shape: [total_patches, hidden] (no batch dim for single image)
    if image_embeds.dim() == 2:
        image_embeds = image_embeds.unsqueeze(0)  # [1, patches, hidden]
    for embeds in image_embeds:
        vector = embeds.mean(dim=0)
        vector = F.normalize(vector, p=2, dim=0)
        pooled_vectors.append(vector.float().cpu().tolist())
    return pooled_vectors


def build_image_vector_records(
    db_path: str = DEFAULT_DB_PATH,
    *,
    model_path: str = DEFAULT_QWEN3_VL_MODEL_PATH,
    device: str = "cuda",
    torch_dtype: str = "auto",
    batch_size: int = 4,
    limit: int | None = None,
    with_vectors: bool = True,
    pooling: str = "mean",
) -> List[Dict[str, Any]]:
    records = fetch_image_vector_records(db_path=db_path, limit=limit)
    if not with_vectors or not records:
        return records

    processor, model = load_qwen3vl_image_encoder(
        model_path=model_path,
        device=device,
        torch_dtype=torch_dtype,
    )

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        vectors = embed_images(
            [record["image_path"] for record in batch],
            processor=processor,
            model=model,
            device=device,
            pooling=pooling,
        )
        for record, vector in zip(batch, vectors):
            record["embedding"] = vector
    return records
