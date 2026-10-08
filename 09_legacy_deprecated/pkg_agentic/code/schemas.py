#!/usr/bin/env python3
"""Shared dataclasses for the agentic RAG runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceItem:
    source: str
    rank: int
    score: float
    text: str = ""
    doc_id: str = ""
    doc_name: str = ""
    page_idx: Optional[int] = None
    block_id: Optional[int] = None
    sample_id: str = ""
    group_id: str = ""
    image_path: str = ""
    image_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RagRequest:
    qid: str
    query_text: str
    query_image_path: str = ""
    question: str = ""
    options: Dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    answer_text: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RagResult:
    qid: str
    query_text: str
    retrieval_text: List[Dict[str, Any]]
    retrieval_image: List[Dict[str, Any]]
    reranked_evidence: List[Dict[str, Any]]
    generator_response: str = ""
    prediction: str = ""
    answer: str = ""
    answer_text: str = ""
    correct: Optional[bool] = None
    error: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
