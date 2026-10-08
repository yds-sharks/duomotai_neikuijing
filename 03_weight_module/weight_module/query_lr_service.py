#!/usr/bin/env python3
"""Query-only LR service.

Input: one text query
Output:
- L label: binary coarse density, mapped to public labels L0/L1
- R label: image dependency label R1/R2/R3
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from semantic_density_service import require_runtime_deps, softmax


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class QueryTextClassifier:
    def __init__(
        self,
        torch,
        nn,
        auto_model,
        auto_tokenizer,
        *,
        config_path: Path,
        checkpoint_dir: Path,
        label_names: List[str],
        device: str,
    ):
        cfg = load_yaml(config_path)
        self.max_length = int(cfg["model"].get("max_length", 96))
        self.base_model_path = str(cfg["model"]["name_or_path"])
        self.num_labels = int(cfg["model"]["num_labels"])
        self.label_names = list(label_names)
        self.torch = torch
        self.device = device
        self.tokenizer = auto_tokenizer.from_pretrained(checkpoint_dir)
        self.model = self._build_model(
            nn=nn,
            auto_model=auto_model,
            dropout=float(cfg["model"].get("dropout", 0.1)),
        )
        state = torch.load(checkpoint_dir / "pytorch_model.bin", map_location=device)
        state = {k: v for k, v in state.items() if not k.startswith("loss_fn.")}
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

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

        return WrappedClassifier(encoder, nn.Dropout(dropout), classifier).to(self.device)

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
        return {
            "label_id": label_id,
            "label_name": self.label_names[label_id],
            "confidence": round(float(probs[label_id]), 6),
        }


class QueryLRService:
    """Public query-only service for coarse L + R labels."""

    DENSITY_PUBLIC_LABELS = ["L0", "L1"]
    ROUTING_PUBLIC_LABELS = ["R1", "R2", "R3"]

    def __init__(
        self,
        root_dir: str | Path = "/mnt/data_1/yds/多模态/权重模块",
        device: str = "auto",
        density_checkpoint_root: Optional[str | Path] = None,
        routing_checkpoint_root: Optional[str | Path] = None,
    ):
        torch, nn, auto_model, auto_tokenizer = require_runtime_deps()
        self.torch = torch
        self.root_dir = Path(root_dir)
        self.device = self._resolve_device(device)

        density_root = Path(density_checkpoint_root) if density_checkpoint_root else self.root_dir / "checkpoints/density2_coarsefilter_mengzi"
        routing_root = Path(routing_checkpoint_root) if routing_checkpoint_root else self.root_dir / "checkpoints/routing_cls_api10k_mengzi"

        self.density_model = QueryTextClassifier(
            torch,
            nn,
            auto_model,
            auto_tokenizer,
            config_path=density_root / "config.resolved.yaml",
            checkpoint_dir=density_root / "best_model",
            label_names=self.DENSITY_PUBLIC_LABELS,
            device=self.device,
        )
        self.routing_model = QueryTextClassifier(
            torch,
            nn,
            auto_model,
            auto_tokenizer,
            config_path=routing_root / "config.resolved.yaml",
            checkpoint_dir=routing_root / "best_model",
            label_names=self.ROUTING_PUBLIC_LABELS,
            device=self.device,
        )

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if self.torch.cuda.is_available() else "cpu"

    def predict(self, query_text: str) -> Dict[str, Any]:
        query = str(query_text).strip()
        if not query:
            raise ValueError("query_text 不能为空。")
        density = self.density_model.predict(query)
        routing = self.routing_model.predict(query)
        return {
            "query_text": query,
            "L_label": density["label_name"],
            "L_confidence": density["confidence"],
            "R_label": routing["label_name"],
            "R_confidence": routing["confidence"],
        }
