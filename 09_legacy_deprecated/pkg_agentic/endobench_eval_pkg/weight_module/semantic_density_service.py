#!/usr/bin/env python3
"""Inference service for semantic-density related models."""

from __future__ import annotations

import json
import math
import os
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def require_runtime_deps():
    try:
        import torch
        from torch import nn
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise SystemExit(
            "推理依赖缺失，请使用已安装 torch/transformers 的 Python 环境运行。\n"
            "推荐解释器：/mnt/data_1/yds/home/miniconda3/bin/python\n"
            f"原始错误：{exc}"
        )
    return torch, nn, AutoModel, AutoTokenizer


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def softmax(values: List[float]) -> List[float]:
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class TextClassifier:
    def __init__(self, torch, nn, auto_model, config_path: Path, best_model_dir: Path, device: str):
        cfg = load_yaml(config_path)
        self.task = cfg["task"]
        self.num_labels = int(cfg["model"]["num_labels"])
        self.max_length = int(cfg["model"].get("max_length", 96))
        self.label_names = [f"L{i}" for i in range(self.num_labels)] if self.task == "density_cls" else [str(i) for i in range(self.num_labels)]
        self.base_model_path = cfg["model"]["name_or_path"]
        self.torch = torch
        self.device = device
        self.tokenizer = self._load_tokenizer(best_model_dir)
        self.model = self._build_model(nn, auto_model, float(cfg["model"].get("dropout", 0.1)))
        state = torch.load(best_model_dir / "pytorch_model.bin", map_location=device)
        state = {k: v for k, v in state.items() if not k.startswith("loss_fn.")}
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def _load_tokenizer(self, best_model_dir: Path):
        _, _, _, auto_tokenizer = require_runtime_deps()
        return auto_tokenizer.from_pretrained(best_model_dir)

    def _build_model(self, nn, auto_model, dropout: float):
        encoder = auto_model.from_pretrained(self.base_model_path)
        hidden_size = int(getattr(encoder.config, "hidden_size"))
        classifier = nn.Linear(hidden_size, self.num_labels)

        class WrappedClassifier(nn.Module):
            def __init__(self, encoder_, dropout_, classifier_):
                super().__init__()
                self.encoder = encoder_
                self.dropout = dropout_
                self.classifier = classifier_

            def forward(self, encodings):
                outputs = self.encoder(**encodings)
                pooled = outputs.last_hidden_state[:, 0]
                return self.classifier(self.dropout(pooled))

        model = WrappedClassifier(encoder, nn.Dropout(dropout), classifier).to(self.device).float()
        return model

    def predict(self, query_text: str) -> Dict[str, Any]:
        with self.torch.no_grad():
            enc = self.tokenizer(
                [query_text],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(enc)[0].detach().cpu().tolist()
        probs = softmax(logits)
        label_id = int(max(range(len(probs)), key=lambda i: probs[i]))
        confidence = probs[label_id]
        prob_map = {self.label_names[i]: round(float(prob), 6) for i, prob in enumerate(probs)}
        return {
            "label_id": label_id,
            "label_name": self.label_names[label_id],
            "confidence": round(float(confidence), 6),
            "probabilities": prob_map,
            "raw_logits": [round(float(v), 6) for v in logits],
        }


class PairwiseRanker:
    def __init__(self, torch, nn, auto_model, auto_tokenizer, config_path: Path, best_model_dir: Path, device: str):
        cfg = load_yaml(config_path)
        self.base_model_path = cfg["model"]["name_or_path"]
        self.max_length = int(cfg["model"].get("max_length", 96))
        self.device = device
        self.torch = torch
        self.tokenizer = auto_tokenizer.from_pretrained(best_model_dir)
        self.model = self._build_model(nn, auto_model, float(cfg["model"].get("dropout", 0.1)))
        state = torch.load(best_model_dir / "pytorch_model.bin", map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()

    def _build_model(self, nn, auto_model, dropout: float):
        encoder = auto_model.from_pretrained(self.base_model_path)
        hidden_size = int(getattr(encoder.config, "hidden_size"))
        score_head = nn.Linear(hidden_size, 1)

        class WrappedRanker(nn.Module):
            def __init__(self, encoder_, dropout_, score_head_):
                super().__init__()
                self.encoder = encoder_
                self.dropout = dropout_
                self.score_head = score_head_

            def score(self, encodings):
                outputs = self.encoder(**encodings)
                pooled = outputs.last_hidden_state[:, 0]
                return self.score_head(self.dropout(pooled)).squeeze(-1)

        model = WrappedRanker(encoder, nn.Dropout(dropout), score_head).to(self.device).float()
        return model

    def score_queries(self, query_texts: List[str], batch_size: int = 64) -> List[float]:
        scores: List[float] = []
        with self.torch.no_grad():
            for start in range(0, len(query_texts), batch_size):
                batch = query_texts[start : start + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                batch_scores = self.model.score(enc).detach().cpu().tolist()
                scores.extend(float(s) for s in batch_scores)
        return scores

    def score_query(self, query_text: str) -> float:
        return self.score_queries([query_text], batch_size=1)[0]

    def compare(self, query_a: str, query_b: str) -> Dict[str, Any]:
        score_a, score_b = self.score_queries([query_a, query_b], batch_size=2)
        logit = float(score_b - score_a)
        prob_b = sigmoid(logit)
        return {
            "score_a": round(score_a, 6),
            "score_b": round(score_b, 6),
            "logit_b_minus_a": round(logit, 6),
            "prob_b_denser_than_a": round(prob_b, 6),
            "prob_a_denser_than_b": round(1.0 - prob_b, 6),
            "winner": "query_b" if prob_b >= 0.5 else "query_a",
        }


class SemanticDensityService:
    DENSITY_LABEL_TEXT = {
        "L0": "极低密度，几乎不提供可定位语义，需要强依赖图像或上下文。",
        "L1": "低密度，提供了很弱的方向，但仍然比较泛。",
        "L2": "中密度，已经包含部位或病变中的一部分关键信息。",
        "L3": "高密度，具备较完整的判别线索，但还没到最细。",
        "L4": "极高密度，问题本身已经非常具体，包含多项明确约束。",
    }
    DEPENDENCY_LABEL_TEXT = {
        "0": "低图像依赖：仅看文本也较容易回答。",
        "1": "中图像依赖：文本有一定信息，但最好结合图像判断。",
        "2": "高图像依赖：如果不看图，基本无法可靠回答。",
    }

    def __init__(
        self,
        root_dir: str | Path = None,
        device: str = "auto",
        rank_reference_path: Optional[str | Path] = None,
    ):
        torch, nn, auto_model, auto_tokenizer = require_runtime_deps()
        self.torch = torch
        self.root_dir = Path(root_dir) if root_dir else Path(os.environ.get("WEIGHT_MODULE_DIR", "."))
        self.device = self._resolve_device(device)

        density_root = self.root_dir / "checkpoints/density_cls_v10_mengzi"
        dependency_root = self.root_dir / "checkpoints/dependency_cls_v10_mengzi"
        rank_root = self.root_dir / "checkpoints/density_rank_v10_mengzi"

        self.density_model = TextClassifier(
            torch,
            nn,
            auto_model,
            density_root / "config.resolved.yaml",
            density_root / "best_model",
            self.device,
        )
        self.dependency_model = TextClassifier(
            torch,
            nn,
            auto_model,
            dependency_root / "config.resolved.yaml",
            dependency_root / "best_model",
            self.device,
        )
        self.rank_model = PairwiseRanker(
            torch,
            nn,
            auto_model,
            auto_tokenizer,
            rank_root / "config.resolved.yaml",
            rank_root / "best_model",
            self.device,
        )

        self.rank_reference_path = Path(rank_reference_path) if rank_reference_path else rank_root / "rank_score_reference.json"
        self.rank_reference = self._load_rank_reference(self.rank_reference_path)

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if self.torch.cuda.is_available() else "cpu"

    def _load_rank_reference(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _rank_signal(self, raw_score: float) -> Dict[str, Any]:
        signal = {
            "raw_score": round(float(raw_score), 6),
            "percentile": None,
            "closest_level_by_median": None,
            "reference_available": self.rank_reference is not None,
            "note": "raw_score 是排序模型内部标量，只保证相对大小有意义；推荐优先看 percentile。",
        }
        if not self.rank_reference:
            return signal

        sorted_scores = self.rank_reference["sorted_scores"]
        position = bisect_right(sorted_scores, raw_score)
        percentile = 100.0 * position / max(len(sorted_scores), 1)
        signal["percentile"] = round(percentile, 4)

        level_stats = self.rank_reference.get("level_stats", {})
        best_level = None
        best_gap = None
        for level_name, stats in level_stats.items():
            gap = abs(raw_score - float(stats["p50"]))
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_level = level_name
        signal["closest_level_by_median"] = best_level
        return signal

    def predict(self, query_text: str) -> Dict[str, Any]:
        query_text = str(query_text).strip()
        density = self.density_model.predict(query_text)
        dependency = self.dependency_model.predict(query_text)
        rank_raw = self.rank_model.score_query(query_text)
        rank_signal = self._rank_signal(rank_raw)

        density["description"] = self.DENSITY_LABEL_TEXT[density["label_name"]]
        dependency["description"] = self.DEPENDENCY_LABEL_TEXT[dependency["label_name"]]
        return {
            "query_text": query_text,
            "device": self.device,
            "density": density,
            "image_dependency": dependency,
            "rank_signal": rank_signal,
        }

    def compare(self, query_a: str, query_b: str) -> Dict[str, Any]:
        single_a = self.predict(query_a)
        single_b = self.predict(query_b)
        pairwise = self.rank_model.compare(query_a, query_b)
        return {
            "query_a": single_a,
            "query_b": single_b,
            "pairwise_rank": pairwise,
        }
