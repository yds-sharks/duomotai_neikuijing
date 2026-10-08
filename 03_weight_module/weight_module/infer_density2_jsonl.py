#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from transformers import AutoModel, AutoTokenizer
import yaml


LABELS = ["L01", "L234"]


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


def batch_iter(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--question-field", type=str, default="question")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/mnt/data_1/yds/多模态/权重模块/checkpoints/density2_coarsefilter_mengzi/best_model"),
    )
    parser.add_argument(
        "--resolved-config",
        type=Path,
        default=Path("/mnt/data_1/yds/多模态/权重模块/checkpoints/density2_coarsefilter_mengzi/config.resolved.yaml"),
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    cfg = yaml.safe_load(args.resolved_config.read_text(encoding="utf-8"))
    model_name_or_path = str(cfg["model"]["name_or_path"])
    num_labels = int(cfg["model"]["num_labels"])
    dropout = float(cfg["model"].get("dropout", 0.1))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = TextClassifier(model_name_or_path=model_name_or_path, num_labels=num_labels, dropout=dropout)
    state = torch.load(checkpoint_dir / "pytorch_model.bin", map_location="cpu")
    state = {k: v for k, v in state.items() if not k.startswith("loss_fn.")}
    model.load_state_dict(state, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    rows = list(read_jsonl(args.input_jsonl))
    results = []
    with torch.no_grad():
        for batch_rows in batch_iter(rows, args.batch_size):
            texts = [str(row.get(args.question_field, "")).strip() for row in batch_rows]
            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(enc)
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            for row, pred, prob in zip(batch_rows, preds, probs):
                results.append(
                    {
                        **row,
                        "density2_prediction": LABELS[pred],
                        "density2_prediction_id": pred,
                        "density2_probabilities": {
                            LABELS[0]: prob[0],
                            LABELS[1]: prob[1],
                        },
                    }
                )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input_jsonl),
        "output": str(args.output_jsonl),
        "count": len(results),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
