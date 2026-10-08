#!/usr/bin/env python3
"""Multimodal vector extraction helpers."""

from .common import (
    DEFAULT_BGE_MODEL_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_QWEN3_VL_MODEL_PATH,
)
from .image_dense_module import build_image_vector_records, fetch_image_vector_records
from .search_multimodal_vector_store import (
    MultimodalVectorSearchEngine,
    RetrievalRequest,
    SearchConfig,
    request_from_payload,
)
from .text_dense_module import build_text_vector_records, fetch_text_vector_records

__all__ = [
    "DEFAULT_BGE_MODEL_PATH",
    "DEFAULT_DB_PATH",
    "DEFAULT_QWEN3_VL_MODEL_PATH",
    "fetch_text_vector_records",
    "build_text_vector_records",
    "fetch_image_vector_records",
    "build_image_vector_records",
    "SearchConfig",
    "RetrievalRequest",
    "MultimodalVectorSearchEngine",
    "request_from_payload",
]
