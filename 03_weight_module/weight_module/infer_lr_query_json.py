#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import yaml
from transformers import AutoModel, AutoTokenizer


DENSITY_LABELS = ["L01", "L234"]
ROUTING_LABELS = ["R1", "R2", "R3"]


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class TextClassifier(torch.nn.Module):
    def __init__(self, model_name_or_path: str, num_labels: int, dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = getattr(self.encoder.config, "hidden_size")
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, enc):
        outputs = self.encoder(**enc)
        pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))


def load_classifier(
    checkpoint_root: Path,
    resolved_config: Path,
    device: torch.device,
) -> tuple[TextClassifier, Any]:
    cfg = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
    model_name_or_path = str(cfg["model"]["name_or_path"])
    num_labels = int(cfg["model"]["num_labels"])
    dropout = float(cfg["model"].get("dropout", 0.1))

    best_model_dir = checkpoint_root / "best_model"
    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    model = TextClassifier(
        model_name_or_path=model_name_or_path,
        num_labels=num_labels,
        dropout=dropout,
    )
    state = torch.load(best_model_dir / "pytorch_model.bin", map_location="cpu")
    state = {k: v for k, v in state.items() if not k.startswith("loss_fn.")}
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    return model, tokenizer


def batch_iter(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def predict_batch(
    model: TextClassifier,
    tokenizer: Any,
    texts: List[str],
    labels: List[str],
    max_length: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(enc)
    probs = torch.softmax(logits, dim=-1).cpu().tolist()
    preds = torch.argmax(logits, dim=-1).cpu().tolist()

    out = []
    for pred, prob in zip(preds, probs):
        out.append(
            {
                "label": labels[pred],
                "confidence": float(prob[pred]),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--question-field", type=str, default="question")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument(
        "--density-checkpoint-root",
        type=Path,
        default=Path("/mnt/data_1/yds/多模态/权重模块/checkpoints/density2_coarsefilter_mengzi"),
    )
    parser.add_argument(
        "--density-resolved-config",
        type=Path,
        default=Path("/mnt/data_1/yds/多模态/权重模块/checkpoints/density2_coarsefilter_mengzi/config.resolved.yaml"),
    )
    parser.add_argument(
        "--routing-checkpoint-root",
        type=Path,
        default=Path("/mnt/data_1/yds/多模态/权重模块/checkpoints/routing_cls_api10k_mengzi"),
    )
    parser.add_argument(
        "--routing-resolved-config",
        type=Path,
        default=Path("/mnt/data_1/yds/多模态/权重模块/checkpoints/routing_cls_api10k_mengzi/config.resolved.yaml"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    density_model, density_tokenizer = load_classifier(
        checkpoint_root=args.density_checkpoint_root,
        resolved_config=args.density_resolved_config,
        device=device,
    )
    routing_model, routing_tokenizer = load_classifier(
        checkpoint_root=args.routing_checkpoint_root,
        resolved_config=args.routing_resolved_config,
        device=device,
    )

    rows = list(read_jsonl(args.input_jsonl))
    output_rows = []
    with torch.no_grad():
        for batch_rows in batch_iter(rows, args.batch_size):
            texts = [str(row.get(args.question_field, "")).strip() for row in batch_rows]
            density_preds = predict_batch(
                model=density_model,
                tokenizer=density_tokenizer,
                texts=texts,
                labels=DENSITY_LABELS,
                max_length=args.max_length,
                device=device,
            )
            routing_preds = predict_batch(
                model=routing_model,
                tokenizer=routing_tokenizer,
                texts=texts,
                labels=ROUTING_LABELS,
                max_length=args.max_length,
                device=device,
            )
            for row, density_pred, routing_pred in zip(batch_rows, density_preds, routing_preds):
                output_rows.append(
                    {
                        "index": row.get("index"),
                        "query": str(row.get(args.question_field, "")).strip(),
                        "L_label": density_pred["label"],
                        "L_confidence": density_pred["confidence"],
                        "R_label": routing_pred["label"],
                        "R_confidence": routing_pred["confidence"],
                    }
                )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "input": str(args.input_jsonl),
                "output": str(args.output_json),
                "count": len(output_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
