#!/usr/bin/env python3
"""Train semantic-density classifiers and pairwise rankers.

The training inputs are query-only by default. This keeps the module focused on
the linguistic density of the question instead of memorizing anchor context.
"""

import argparse
import json
import math
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from tqdm import tqdm


def require_training_deps():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise SystemExit(
            "训练依赖缺失，请先安装 train/requirements_train.txt 中的依赖。\n"
            "推荐：python3 -m pip install -r /mnt/data_1/yds/多模态/权重模块/train/requirements_train.txt\n"
            f"原始错误：{exc}"
        )
    return torch, nn, DataLoader, Dataset, AutoModel, AutoTokenizer


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_config(path: Path, cfg: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def build_input_text(row: Dict[str, Any], input_mode: str) -> str:
    query = str(row.get("query_text", "")).strip()
    if input_mode == "query_only":
        return query
    if input_mode == "query_with_core":
        core = str(row.get("semantic_core", "")).strip()
        return f"问题：{query}\n语义核心：{core}".strip()
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def split_rows(
    path: Path,
    split_name: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        if row.get("split") == split_name:
            rows.append(row)
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def macro_f1(labels: List[int], preds: List[int], num_labels: int) -> Dict[str, Any]:
    confusion = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for y, p in zip(labels, preds):
        if 0 <= y < num_labels and 0 <= p < num_labels:
            confusion[y][p] += 1

    per_class = {}
    f1_values = []
    for cls_id in range(num_labels):
        tp = confusion[cls_id][cls_id]
        fp = sum(confusion[r][cls_id] for r in range(num_labels) if r != cls_id)
        fn = sum(confusion[cls_id][c] for c in range(num_labels) if c != cls_id)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(cls_id)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[cls_id]),
        }
        f1_values.append(f1)

    correct = sum(1 for y, p in zip(labels, preds) if y == p)
    total = len(labels)
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


class TextClassifierBase:
    pass


def make_models(nn, AutoModel):
    class TextClassifier(nn.Module):
        def __init__(self, model_name_or_path: str, num_labels: int, dropout: float):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name_or_path)
            hidden_size = getattr(self.encoder.config, "hidden_size")
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_size, num_labels)
            self.loss_fn = nn.CrossEntropyLoss()

        def forward(self, enc, labels=None):
            outputs = self.encoder(**enc)
            pooled = outputs.last_hidden_state[:, 0]
            logits = self.classifier(self.dropout(pooled))
            loss = None
            if labels is not None:
                loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}

    class PairwiseRanker(nn.Module):
        def __init__(self, model_name_or_path: str, dropout: float):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name_or_path)
            hidden_size = getattr(self.encoder.config, "hidden_size")
            self.dropout = nn.Dropout(dropout)
            self.score_head = nn.Linear(hidden_size, 1)

        def score(self, enc):
            outputs = self.encoder(**enc)
            pooled = outputs.last_hidden_state[:, 0]
            return self.score_head(self.dropout(pooled)).squeeze(-1)

        def forward(self, enc_a, enc_b, labels=None):
            score_a = self.score(enc_a)
            score_b = self.score(enc_b)
            logits = score_b - score_a
            loss = None
            if labels is not None:
                loss = nn.BCEWithLogitsLoss()(logits, labels.float())
            return {"loss": loss, "logits": logits, "score_a": score_a, "score_b": score_b}

    return TextClassifier, PairwiseRanker


def make_datasets(torch, Dataset):
    class ClassificationDataset(Dataset):
        def __init__(self, rows: List[Dict[str, Any]], label_field: str, input_mode: str):
            self.items = []
            for row in rows:
                self.items.append(
                    {
                        "text": build_input_text(row, input_mode),
                        "label": int(row[label_field]),
                        "meta": {
                            "query_id": row.get("query_id"),
                            "candidate_id": row.get("candidate_id"),
                            "split": row.get("split"),
                            "query_text": row.get("query_text"),
                            "density_level": row.get("density_level"),
                        },
                    }
                )

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    class RankDataset(Dataset):
        def __init__(self, rows: List[Dict[str, Any]], augment_reverse: bool):
            self.items = []
            for row in rows:
                self.items.append(
                    {
                        "text_a": str(row.get("query_text_low", "")).strip(),
                        "text_b": str(row.get("query_text_high", "")).strip(),
                        "label": 1,
                        "meta": {
                            "pair_id": row.get("pair_id"),
                            "candidate_id": row.get("candidate_id"),
                            "split": row.get("split"),
                            "query_id_a": row.get("query_id_low"),
                            "query_id_b": row.get("query_id_high"),
                            "density_level_a": row.get("density_level_low"),
                            "density_level_b": row.get("density_level_high"),
                        },
                    }
                )
                if augment_reverse:
                    self.items.append(
                        {
                            "text_a": str(row.get("query_text_high", "")).strip(),
                            "text_b": str(row.get("query_text_low", "")).strip(),
                            "label": 0,
                            "meta": {
                                "pair_id": f"{row.get('pair_id')}_reverse",
                                "candidate_id": row.get("candidate_id"),
                                "split": row.get("split"),
                                "query_id_a": row.get("query_id_high"),
                                "query_id_b": row.get("query_id_low"),
                                "density_level_a": row.get("density_level_high"),
                                "density_level_b": row.get("density_level_low"),
                            },
                        }
                    )

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    return ClassificationDataset, RankDataset


def make_collators(torch, tokenizer, max_length: int):
    def cls_collate(batch):
        texts = [x["text"] for x in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        return {"enc": enc, "labels": labels, "meta": [x["meta"] for x in batch]}

    def rank_collate(batch):
        text_a = [x["text_a"] for x in batch]
        text_b = [x["text_b"] for x in batch]
        enc_a = tokenizer(
            text_a,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc_b = tokenizer(
            text_b,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.float)
        return {"enc_a": enc_a, "enc_b": enc_b, "labels": labels, "meta": [x["meta"] for x in batch]}

    return cls_collate, rank_collate


def to_device(batch: Dict[str, Any], device):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {k: v.to(device) for k, v in value.items()}
        elif hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def count_labels(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(Counter(str(row.get(field)) for row in rows))


def build_class_weights(torch, rows: List[Dict[str, Any]], label_field: str, num_labels: int, mode: str, device):
    if mode != "auto_balanced":
        return None
    counts = Counter(int(row[label_field]) for row in rows)
    total = sum(counts.values())
    weights = []
    for label_id in range(num_labels):
        count = counts.get(label_id, 0)
        weights.append(total / (num_labels * count) if count else 0.0)
    return torch.tensor(weights, dtype=torch.float, device=device)


def train_classification(cfg: Dict[str, Any], dry_run: bool = False) -> None:
    data_cfg = cfg["data"]
    train_rows = split_rows(Path(data_cfg["path"]), "train", data_cfg.get("max_train_samples"))
    val_rows = split_rows(Path(data_cfg["path"]), "val", data_cfg.get("max_val_samples"))
    test_rows = split_rows(Path(data_cfg["path"]), "test", data_cfg.get("max_test_samples"))
    label_field = data_cfg["label_field"]

    print(json.dumps(
        {
            "task": cfg["task"],
            "input_mode": data_cfg.get("input_mode", "query_only"),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "train_label_distribution": count_labels(train_rows, label_field),
        },
        ensure_ascii=False,
        indent=2,
    ))
    if dry_run:
        return

    torch, nn, DataLoader, Dataset, AutoModel, AutoTokenizer = require_training_deps()
    TextClassifier, _ = make_models(nn, AutoModel)
    ClassificationDataset, _ = make_datasets(torch, Dataset)
    set_seed(int(cfg["train"].get("seed", 20260423)))
    torch.manual_seed(int(cfg["train"].get("seed", 20260423)))

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("device", "auto") != "cpu" else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name_or_path"])
    cls_collate, _ = make_collators(torch, tokenizer, int(cfg["model"].get("max_length", 96)))

    input_mode = data_cfg.get("input_mode", "query_only")
    train_ds = ClassificationDataset(train_rows, label_field, input_mode)
    val_ds = ClassificationDataset(val_rows, label_field, input_mode)
    test_ds = ClassificationDataset(test_rows, label_field, input_mode)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=cls_collate,
    )
    val_loader = DataLoader(val_ds, batch_size=int(cfg["eval"]["batch_size"]), shuffle=False, collate_fn=cls_collate)
    test_loader = DataLoader(test_ds, batch_size=int(cfg["eval"]["batch_size"]), shuffle=False, collate_fn=cls_collate)

    model = TextClassifier(
        cfg["model"]["name_or_path"],
        num_labels=int(cfg["model"]["num_labels"]),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    ).to(device)
    class_weights = build_class_weights(
        torch,
        train_rows,
        label_field,
        int(cfg["model"]["num_labels"]),
        cfg["train"].get("class_weight", "none"),
        device,
    )
    if class_weights is not None:
        model.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print(json.dumps({"class_weights": class_weights.detach().cpu().tolist()}, ensure_ascii=False, indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["learning_rate"]), weight_decay=float(cfg["train"].get("weight_decay", 0.01)))

    out_dir = Path(cfg["output"]["dir"])
    best_dir = out_dir / "best_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(out_dir / "config.resolved.yaml", cfg)
    tokenizer.save_pretrained(best_dir)

    use_amp = bool(cfg["train"].get("fp16", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_accum = int(cfg["train"].get("gradient_accumulation_steps", 1))
    best_metric = -1.0
    history = []
    has_val = len(val_rows) > 0

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"epoch {epoch} train")
        for step, batch in enumerate(progress, start=1):
            batch = to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(batch["enc"], batch["labels"])
                loss = out["loss"] / grad_accum
            scaler.scale(loss).backward()
            if step % grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("max_grad_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item()) * grad_accum
            progress.set_postfix(loss=f"{total_loss / step:.4f}")

        val_metrics, _ = evaluate_classification(torch, model, val_loader, device, int(cfg["model"]["num_labels"]))
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = total_loss / max(len(train_loader), 1)
        history.append(val_metrics)
        print(json.dumps({"epoch": epoch, "val": val_metrics}, ensure_ascii=False, indent=2))

        if not has_val:
            save_model(
                torch,
                model,
                tokenizer,
                best_dir,
                cfg,
                {"best_epoch": epoch, "best_metric": None, "selection": "last_epoch_no_val"},
            )
        elif val_metrics["macro_f1"] > best_metric:
            best_metric = val_metrics["macro_f1"]
            save_model(torch, model, tokenizer, best_dir, cfg, {"best_epoch": epoch, "best_metric": best_metric})

    model.load_state_dict(torch.load(best_dir / "pytorch_model.bin", map_location=device))
    test_metrics, predictions = evaluate_classification(torch, model, test_loader, device, int(cfg["model"]["num_labels"]), collect_predictions=True)
    write_jsonl(out_dir / "predictions.test.jsonl", predictions)
    final_report = {
        "history": history,
        "best_val_macro_f1": best_metric if has_val else None,
        "model_selection": "best_val_macro_f1" if has_val else "last_epoch_no_val",
        "test": test_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final_report, ensure_ascii=False, indent=2))


def evaluate_classification(torch, model, loader, device, num_labels: int, collect_predictions: bool = False):
    model.eval()
    labels_all, preds_all = [], []
    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            batch = to_device(batch, device)
            out = model(batch["enc"])
            probs = torch.softmax(out["logits"], dim=-1)
            preds = torch.argmax(probs, dim=-1)
            labels = batch["labels"]
            labels_all.extend(labels.cpu().tolist())
            preds_all.extend(preds.cpu().tolist())
            if collect_predictions:
                for meta, label, pred, prob in zip(batch["meta"], labels.cpu().tolist(), preds.cpu().tolist(), probs.cpu().tolist()):
                    predictions.append({**meta, "label": label, "prediction": pred, "probabilities": prob})
    return macro_f1(labels_all, preds_all, num_labels), predictions


def train_rank(cfg: Dict[str, Any], dry_run: bool = False) -> None:
    data_cfg = cfg["data"]
    train_rows = split_rows(Path(data_cfg["path"]), "train", data_cfg.get("max_train_samples"))
    val_rows = split_rows(Path(data_cfg["path"]), "val", data_cfg.get("max_val_samples"))
    test_rows = split_rows(Path(data_cfg["path"]), "test", data_cfg.get("max_test_samples"))
    augment_reverse = bool(data_cfg.get("augment_reverse", True))

    print(json.dumps(
        {
            "task": cfg["task"],
            "train_pairs": len(train_rows),
            "val_pairs": len(val_rows),
            "test_pairs": len(test_rows),
            "augment_reverse": augment_reverse,
            "effective_train_examples": len(train_rows) * (2 if augment_reverse else 1),
        },
        ensure_ascii=False,
        indent=2,
    ))
    if dry_run:
        return

    torch, nn, DataLoader, Dataset, AutoModel, AutoTokenizer = require_training_deps()
    _, PairwiseRanker = make_models(nn, AutoModel)
    _, RankDataset = make_datasets(torch, Dataset)
    set_seed(int(cfg["train"].get("seed", 20260423)))
    torch.manual_seed(int(cfg["train"].get("seed", 20260423)))

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("device", "auto") != "cpu" else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name_or_path"])
    _, rank_collate = make_collators(torch, tokenizer, int(cfg["model"].get("max_length", 96)))
    train_ds = RankDataset(train_rows, augment_reverse)
    val_ds = RankDataset(val_rows, augment_reverse)
    test_ds = RankDataset(test_rows, augment_reverse)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=int(cfg["train"].get("num_workers", 0)), collate_fn=rank_collate)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["eval"]["batch_size"]), shuffle=False, collate_fn=rank_collate)
    test_loader = DataLoader(test_ds, batch_size=int(cfg["eval"]["batch_size"]), shuffle=False, collate_fn=rank_collate)

    model = PairwiseRanker(cfg["model"]["name_or_path"], dropout=float(cfg["model"].get("dropout", 0.1))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["learning_rate"]), weight_decay=float(cfg["train"].get("weight_decay", 0.01)))

    out_dir = Path(cfg["output"]["dir"])
    best_dir = out_dir / "best_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(out_dir / "config.resolved.yaml", cfg)
    tokenizer.save_pretrained(best_dir)

    use_amp = bool(cfg["train"].get("fp16", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_accum = int(cfg["train"].get("gradient_accumulation_steps", 1))
    best_metric = -1.0
    history = []

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"epoch {epoch} rank-train")
        for step, batch in enumerate(progress, start=1):
            batch = to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(batch["enc_a"], batch["enc_b"], batch["labels"])
                loss = out["loss"] / grad_accum
            scaler.scale(loss).backward()
            if step % grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("max_grad_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item()) * grad_accum
            progress.set_postfix(loss=f"{total_loss / step:.4f}")

        val_metrics, _ = evaluate_rank(torch, model, val_loader, device)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = total_loss / max(len(train_loader), 1)
        history.append(val_metrics)
        print(json.dumps({"epoch": epoch, "val": val_metrics}, ensure_ascii=False, indent=2))
        if val_metrics["accuracy"] > best_metric:
            best_metric = val_metrics["accuracy"]
            save_model(torch, model, tokenizer, best_dir, cfg, {"best_epoch": epoch, "best_metric": best_metric})

    model.load_state_dict(torch.load(best_dir / "pytorch_model.bin", map_location=device))
    test_metrics, predictions = evaluate_rank(torch, model, test_loader, device, collect_predictions=True)
    write_jsonl(out_dir / "predictions.test.jsonl", predictions)
    final_report = {"history": history, "best_val_accuracy": best_metric, "test": test_metrics}
    (out_dir / "metrics.json").write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final_report, ensure_ascii=False, indent=2))


def evaluate_rank(torch, model, loader, device, collect_predictions: bool = False):
    model.eval()
    labels_all, preds_all = [], []
    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval-rank", leave=False):
            batch = to_device(batch, device)
            out = model(batch["enc_a"], batch["enc_b"])
            probs = torch.sigmoid(out["logits"])
            preds = (probs >= 0.5).long()
            labels = batch["labels"].long()
            labels_all.extend(labels.cpu().tolist())
            preds_all.extend(preds.cpu().tolist())
            if collect_predictions:
                for meta, label, pred, prob, score_a, score_b in zip(
                    batch["meta"],
                    labels.cpu().tolist(),
                    preds.cpu().tolist(),
                    probs.cpu().tolist(),
                    out["score_a"].cpu().tolist(),
                    out["score_b"].cpu().tolist(),
                ):
                    predictions.append({**meta, "label": label, "prediction": pred, "prob_second_denser": prob, "score_a": score_a, "score_b": score_b})
    return macro_f1(labels_all, preds_all, 2), predictions


def save_model(torch, model, tokenizer, out_dir: Path, cfg: Dict[str, Any], meta: Dict[str, Any]) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    torch.save(model.state_dict(), out_dir / "pytorch_model.bin")
    (out_dir / "model_meta.json").write_text(json.dumps({"task": cfg["task"], **meta}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML training config")
    parser.add_argument("--dry-run", action="store_true", help="Only inspect data and config; no torch import.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    task = cfg["task"]
    if task in {"density_cls", "dependency_cls"}:
        train_classification(cfg, dry_run=args.dry_run)
    elif task == "density_rank":
        train_rank(cfg, dry_run=args.dry_run)
    else:
        raise SystemExit(f"Unsupported task: {task}")


if __name__ == "__main__":
    main()
