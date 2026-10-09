#!/usr/bin/env python3
"""Retrieval backend: wraps the v0.5 FirstStageRetriever, with a Mock for smoke tests.

The real backend dynamically loads `retrieval_adapter.py` from the archived agentic
code dir (same FirstStageRetriever used by v0.5 evals), so candidate fields stay
identical across the training chain.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Protocol


class RetrievalBackend(Protocol):
    def search_text(self, query: str, k: int = 20) -> List[Dict[str, Any]]: ...
    def search_image(self, image_path: str, k: int = 20) -> List[Dict[str, Any]]: ...
    def close(self) -> None: ...


def load_first_stage_retriever(config: Dict[str, Any]) -> RetrievalBackend:
    """Build FirstStageRetriever from the archived agentic code dir.

    FirstStageRetriever expects the FULL config shape {"paths": {...}, "retrieval": {...}}
    (it reads paths.multimodal_search_dir / paths.main_db_path); harness_config.json
    keeps everything under "retrieval", so re-wrap here.
    """
    code_dir = Path(config["retrieval"]["code_dir"])
    adapter_path = code_dir / "retrieval_adapter.py"
    if not adapter_path.exists():
        raise FileNotFoundError(f"retrieval_adapter.py not found: {adapter_path}")
    mod_name = "_harness_retrieval_adapter"
    spec = importlib.util.spec_from_file_location(mod_name, adapter_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    rcfg = dict(config["retrieval"])
    rcfg.setdefault("first_stage_text_k", rcfg.get("text_k", 20))
    rcfg.setdefault("first_stage_image_k", rcfg.get("image_k", 20))
    full_cfg = {
        "paths": {
            "multimodal_search_dir": rcfg["multimodal_search_dir"],
            "main_db_path": rcfg["main_db_path"],
        },
        "retrieval": rcfg,
    }
    return module.FirstStageRetriever(config=full_cfg)  # type: ignore[return-value]


class MockRetrievalBackend:
    """Deterministic fake hits for smoke tests — no GPU/DB/network."""

    def __init__(self, n_text: int = 8, n_image: int = 8):
        self._n_text = n_text
        self._n_image = n_image

    @staticmethod
    def _hit(i: int, origin: str) -> Dict[str, Any]:
        return {
            "origin": origin,
            "source": origin,
            "score": 1.0 - i * 0.01,
            "text": f"[mock {origin} passage {i}] 该内镜图像显示黏膜表面光滑，可见环状皱襞。",
            "doc_id": f"mockdoc{origin}",
            "doc_name": f"mock_doc_{origin}",
            "page_idx": i,
            "block_id": i * 10,
            "sample_id": f"mock_{origin}_{i:04d}",
            "group_id": f"grp_mock_{origin}_{i:04d}",
            "image_path": "",
            "image_id": "",
        }

    def search_text(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        return [self._hit(i, "text") for i in range(min(k, self._n_text))]

    def search_image(self, image_path: str, k: int = 20) -> List[Dict[str, Any]]:
        hits = [self._hit(i, "image") for i in range(min(k, self._n_image))]
        for h in hits:
            h["image_path"] = "mock://image.jpg"
        return hits

    def close(self) -> None: ...
