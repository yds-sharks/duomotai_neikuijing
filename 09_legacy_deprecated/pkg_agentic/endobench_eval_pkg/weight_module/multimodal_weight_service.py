#!/usr/bin/env python3
"""Text-only weight service for semantic density and image dependency."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from semantic_density_service import SemanticDensityService


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class MultimodalWeightService:
    """Current public interface: text-only analysis."""

    DEPENDENCY_PUBLIC_MAP = {
        "0": "R1",
        "1": "R2",
        "2": "R3",
    }
    DEPENDENCY_PUBLIC_DESC = {
        "R1": "低图片依赖：仅看文本通常也可以完成判断。",
        "R2": "中图片依赖：文本提供了部分信息，但结合图片更稳妥。",
        "R3": "高图片依赖：如果不看图片，通常难以可靠判断。",
    }

    def __init__(
        self,
        config_path: str | Path = os.environ.get(
            "WEIGHT_MODULE_CONFIG",
            str(Path(__file__).resolve().parent / "weight_module_runtime.yaml")),
    ):
        self.config_path = Path(config_path)
        self.cfg = load_yaml(self.config_path)
        self.root_dir = Path(self.cfg["runtime"]["workspace_root"])
        self._text_service: Optional[SemanticDensityService] = None

    def _ensure_text_service(self) -> SemanticDensityService:
        if self._text_service is None:
            device = self.cfg.get("text_analysis", {}).get("device", "auto")
            self._text_service = SemanticDensityService(root_dir=self.root_dir, device=device)
        return self._text_service

    def close(self) -> None:
        return None

    def __enter__(self) -> "MultimodalWeightService":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def analyze_text(self, query_text: str) -> Dict[str, Any]:
        service = self._ensure_text_service()
        result = service.predict(query_text)

        density = result["density"]
        dependency = result["image_dependency"]
        dependency_level = self.DEPENDENCY_PUBLIC_MAP[dependency["label_name"]]
        dependency_probs = {
            self.DEPENDENCY_PUBLIC_MAP[k]: v for k, v in dependency["probabilities"].items()
        }

        return {
            "query_text": str(query_text).strip(),
            "density_level": density["label_name"],
            "density_description": density["description"],
            "image_dependency_level": dependency_level,
            "image_dependency_description": self.DEPENDENCY_PUBLIC_DESC[dependency_level],
            "model_outputs": {
                "density_confidence": density["confidence"],
                "density_probabilities": density["probabilities"],
                "image_dependency_confidence": dependency["confidence"],
                "image_dependency_probabilities": dependency_probs,
            },
        }

    def analyze(
        self,
        *,
        query_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not query_text or not str(query_text).strip():
            raise ValueError("当前接口只支持文本侧分析，请提供非空 query_text。")
        return {"text_analysis": self.analyze_text(query_text)}


def pretty_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)
