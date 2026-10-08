#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: run_sev_cache_and_sweep.py
Purpose:
  1) Cache SEV signals (cls_prob and score_prob) for (qid, pk) pairs from a TREC run.
  2) Sweep thresholds around 0.5 WITHOUT re-running the model: generate partitioned TREC runs + eval.

Key design:
  - One heavy pass: model inference over all (qid, topk_in) pairs => cache.jsonl
  - Many cheap passes: threshold sweep => run_out + metrics.csv

Assumptions:
  - run_in is TREC format: qid \t Q0 \t docid(pk) \t rank \t score \t tag
  - qid in run matches qid as string index in input_json enumeration (same as your baseline scripts)
  - Milvus primary key field is obtained from collection.schema.primary_field.name
"""

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import os, sys
from pathlib import Path

# ---- FIX: make project root importable (so `import insert...` works everywhere) ----
_PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if not _PROJECT_ROOT:
    # file is: <root>/insert/exp_path3/run_sev_cache_and_sweep.py
    _PROJECT_ROOT = str(Path(__file__).resolve().parents[2])  # -> <root>
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# -----------------------------------------------------------------------------------

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import yaml
from tqdm import tqdm


# -----------------------------
# Your QwenRerankerModel (local)
# -----------------------------
class QwenRerankerModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        hidden_size = self.backbone.config.hidden_size
        self.score_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        self.cls_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, T, D]
        mask = attention_mask.unsqueeze(-1).float()
        sent = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)  # [B, D]
        sent = sent.to(self.score_head[0].weight.dtype)
        score = self.score_head(sent).squeeze(-1)      # [B] in [0,1]
        logit = self.cls_head(sent).squeeze(-1)        # [B]
        return score, logit


def sigmoid_prob(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


# -----------------------------
# Helpers: load queries
# -----------------------------
def load_qids(path: str) -> Optional[set]:
    if not path:
        return None
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                s.add(t)
    return s


def load_questions(input_json: str, eligible_qids: Optional[set]) -> Dict[str, str]:
    """
    Map qid(str(index)) -> question
    """
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    qmap = {}
    for i, item in enumerate(data):
        qid = str(i)
        if eligible_qids is not None and qid not in eligible_qids:
            continue
        q = (item.get("question") or item.get("query") or "").strip()
        qmap[qid] = q
    return qmap


# -----------------------------
# Helpers: parse TREC run
# -----------------------------
def read_trec_run(run_path: str, topk_in: int, eligible_qids: Optional[set]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return: qid -> list of {pk, orig_rank, retr_score}
    Only keep up to topk_in per qid by orig_rank.
    """
    q2docs: Dict[str, List[Dict[str, Any]]] = {}
    with open(run_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                # tolerate space-separated
                parts = line.split()
                if len(parts) < 6:
                    continue
            qid, _, docid, rank, score, _tag = parts[:6]
            if eligible_qids is not None and qid not in eligible_qids:
                continue
            try:
                pk = int(docid)
            except Exception:
                # allow non-int docids but still store string
                pk = docid
            try:
                rnk = int(rank)
            except Exception:
                rnk = 0
            try:
                sc = float(score)
            except Exception:
                sc = 0.0
            q2docs.setdefault(qid, []).append({"pk": pk, "orig_rank": rnk, "retr_score": sc})

    # sort and cut
    for qid in list(q2docs.keys()):
        q2docs[qid].sort(key=lambda x: x["orig_rank"])
        q2docs[qid] = q2docs[qid][:topk_in]
    return q2docs


# -----------------------------
# Helpers: Milvus fetch texts by pk
# -----------------------------
def chunked(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def fetch_texts_by_pks(collection, pks: List[Any], text_field: str = "text", chunk_size: int = 512) -> Dict[Any, str]:
    """
    Robustly fetch texts by Milvus primary key field name.
    """
    pk_field = collection.schema.primary_field.name
    out: Dict[Any, str] = {}

    for batch in chunked(pks, chunk_size):
        # Milvus expr expects list repr
        expr = f"{pk_field} in {batch}"
        rows = collection.query(expr=expr, output_fields=[pk_field, text_field])
        for r in rows:
            out[r[pk_field]] = (r.get(text_field) or "")
    return out


# -----------------------------
# Inference wrapper
# -----------------------------
class SEVInfer:
    def __init__(
        self,
        model_path: str,
        adapter_path: str,
        cls_head_path: str,
        score_head_path: str,
        dtype: str = "bf16",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        base = PeftModel.from_pretrained(base, adapter_path)

        self.model = QwenRerankerModel(base)

        # heads
        self.model.cls_head.load_state_dict(torch.load(cls_head_path, map_location="cpu"))
        self.model.score_head.load_state_dict(torch.load(score_head_path, map_location="cpu"))

        self.model.eval().to(self.device)

        # dtype casting (model weights)
        if self.device.startswith("cuda"):
            if dtype == "bf16":
                self.model = self.model.to(torch.bfloat16)
            elif dtype == "fp16":
                self.model = self.model.to(torch.float16)
            else:
                self.model = self.model.to(torch.float32)

        print(f"[INIT] device={self.device}, dtype={dtype}")

    @torch.no_grad()
    def predict(
        self,
        prompts: List[str],
        batch_size: int,
        max_length: int,
    ) -> Tuple[List[float], List[float]]:
        """
        Returns:
          score_prob_list: score_head output in [0,1]
          cls_prob_list: sigmoid(cls_logit) in [0,1]
        """
        score_all: List[float] = []
        cls_all: List[float] = []

        for s in range(0, len(prompts), batch_size):
            chunk = prompts[s:s + batch_size]
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            score, logit = self.model(enc["input_ids"], enc["attention_mask"])

            # ensure float32 on cpu for stability
            score = score.detach().float().cpu()
            logit = logit.detach().float().cpu()
            cls_prob = sigmoid_prob(logit)

            score_all.extend(score.view(-1).tolist())
            cls_all.extend(cls_prob.view(-1).tolist())

        return score_all, cls_all


# -----------------------------
# Cache writing
# -----------------------------
def build_prompt(question: str, passage: str, template: str) -> str:
    return template.format(question=question, passage=passage)


def cache_all_pairs(
    cfg_path: str,
    input_json: str,
    qids_file: str,
    run_in: str,
    topk_in: int,
    cache_out: str,
    model_path: str,
    adapter_path: str,
    cls_head_path: str,
    score_head_path: str,
    batch_size: int,
    max_length: int,
    dtype: str,
    prompt_template: str,
    text_field: str = "text",
):
    eligible = load_qids(qids_file)
    qmap = load_questions(input_json, eligible)
    q2docs = read_trec_run(run_in, topk_in=topk_in, eligible_qids=eligible)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # HybridRetriever gives us collection handle
    from insert.bm25_bge_vectorstore_v2.retrieval_service import HybridRetriever
    retr = HybridRetriever(cfg)
    collection = retr.collection

    infer = SEVInfer(
        model_path=model_path,
        adapter_path=adapter_path,
        cls_head_path=cls_head_path,
        score_head_path=score_head_path,
        dtype=dtype,
    )

    outp = Path(cache_out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    missing_q = 0
    missing_text = 0
    total_pairs = 0

    with outp.open("w", encoding="utf-8") as fout:
        for qid in tqdm(sorted(q2docs.keys(), key=lambda x: int(x) if x.isdigit() else x), desc="SEV cache"):
            docs = q2docs[qid]
            question = qmap.get(qid, "")
            if not question:
                missing_q += 1

            pks = [d["pk"] for d in docs]
            text_map = fetch_texts_by_pks(collection, pks, text_field=text_field)

            prompts = []
            valid_idx = []
            for idx, d in enumerate(docs):
                pk = d["pk"]
                txt = (text_map.get(pk) or "").strip()
                if not txt:
                    missing_text += 1
                prompts.append(build_prompt(question, txt, prompt_template))
                valid_idx.append(idx)

            score_prob, cls_prob = infer.predict(prompts, batch_size=batch_size, max_length=max_length)

            for i, d in enumerate(docs):
                rec = {
                    "qid": qid,
                    "pk": d["pk"],
                    "orig_rank": d["orig_rank"],
                    "retr_score": d["retr_score"],
                    "score_prob": float(score_prob[i]),  # from score_head
                    "cls_prob": float(cls_prob[i]),      # sigmoid(cls_logit)
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_pairs += 1

    print(json.dumps({
        "cache_out": str(outp),
        "num_qids": len(q2docs),
        "total_pairs": total_pairs,
        "missing_question_qids": missing_q,
        "missing_text_count": missing_text,
        "batch_size": batch_size,
        "max_length": max_length,
        "dtype": dtype,
        "prompt_template": prompt_template,
    }, ensure_ascii=False, indent=2))


# -----------------------------
# Threshold sweep (fast)
# -----------------------------
def load_cache(cache_path: str) -> Dict[str, List[Dict[str, Any]]]:
    q2 = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q2.setdefault(r["qid"], []).append(r)
    # keep stable by orig_rank
    for qid in q2:
        q2[qid].sort(key=lambda x: x["orig_rank"])
    return q2


def write_partitioned_run(
    q2pairs: Dict[str, List[Dict[str, Any]]],
    out_run: str,
    run_name: str,
    topk_out: int,
    threshold: float,
    signal: str,           # "cls_prob" or "score_prob"
):
    """
    Stable partition by threshold:
      relevant first (signal>=threshold) keep orig order,
      then others keep orig order.
    Use retr_score as TREC score to avoid re-sorting surprises.
    """
    Path(out_run).parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    with open(out_run, "w", encoding="utf-8") as fp:
        for qid, pairs in q2pairs.items():
            rel = []
            irr = []
            for p in pairs:
                if float(p[signal]) >= threshold:
                    rel.append(p)
                else:
                    irr.append(p)
            merged = (rel + irr)[:topk_out]
            for new_rank, p in enumerate(merged, start=1):
                fp.write(f"{qid}\tQ0\t{p['pk']}\t{new_rank}\t{float(p['retr_score'])}\t{run_name}\n")
                n_lines += 1
    return n_lines


def eval_one_run(eval_py: str, qrels: str, run_path: str, metrics_out: str, per_query_out: str) -> Dict[str, float]:
    cmd = [
        "python", eval_py,
        "--qrels", qrels,
        "--runs", run_path,
        "--metrics_out", metrics_out,
        "--per_query_out", per_query_out,
    ]
    subprocess.check_call(cmd)

    # parse last line of metrics csv (header + one row)
    with open(metrics_out, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty metrics_out: {metrics_out}")
    r = rows[-1]
    return {
        "nDCG@10": float(r["nDCG@10"]),
        "MRR@10": float(r["MRR@10"]),
        "Recall@10": float(r["Recall@10"]),
        "Recall@50": float(r["Recall@50"]),
        "Recall@100": float(r["Recall@100"]),
    }


def sweep_thresholds(
    cache_path: str,
    out_dir: str,
    qrels: str,
    eval_py: str,
    topk_out: int,
    thresholds: List[float],
    signal: str,  # "cls_prob" or "score_prob"
    run_prefix: str,
):
    out_dir = str(Path(out_dir))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    q2pairs = load_cache(cache_path)

    summary_csv = Path(out_dir) / f"{run_prefix}_{signal}_sweep.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal", "threshold", "nDCG@10", "MRR@10", "Recall@10", "Recall@50", "Recall@100", "run_path"])

        for th in thresholds:
            th_str = f"{th:.4f}".rstrip("0").rstrip(".")
            run_path = str(Path(out_dir) / f"{run_prefix}_{signal}_th{th_str}.trec")
            run_name = f"{run_prefix}_{signal}_th{th_str}"

            n_lines = write_partitioned_run(
                q2pairs=q2pairs,
                out_run=run_path,
                run_name=run_name,
                topk_out=topk_out,
                threshold=th,
                signal=signal,
            )

            metrics_out = str(Path(out_dir) / f"_tmp_{signal}_th{th_str}.csv")
            perq_out = str(Path(out_dir) / f"_tmp_{signal}_th{th_str}.json")
            m = eval_one_run(eval_py=eval_py, qrels=qrels, run_path=run_path, metrics_out=metrics_out, per_query_out=perq_out)

            w.writerow([signal, th, m["nDCG@10"], m["MRR@10"], m["Recall@10"], m["Recall@50"], m["Recall@100"], run_path])
            print(json.dumps({"signal": signal, "threshold": th, "lines": n_lines, **m}, ensure_ascii=False))

    print(f"[OK] sweep summary -> {summary_csv}")


def parse_thresholds(th_center: float, th_span: float, th_step: float) -> List[float]:
    """
    e.g., center=0.5 span=0.06 step=0.01 => [0.44..0.56]
    """
    start = th_center - th_span
    end = th_center + th_span
    out = []
    x = start
    # robust floating stepping
    while x <= end + 1e-12:
        out.append(round(x, 6))
        x += th_step
    return out


def main():
    ap = argparse.ArgumentParser()

    # common
    ap.add_argument("--config", required=True)
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--qids_file", required=True)
    ap.add_argument("--run_in", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--eval_py", default="insert/exp_path3/eval_runs.py")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--topk_in", type=int, default=100)
    ap.add_argument("--topk_out", type=int, default=100)

    # caching
    ap.add_argument("--cache_out", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--prompt_template", default="问请判断下列文段是否支持回答问题：{question}\n文段如下：{passage}")
    ap.add_argument("--text_field", default="text")

    # model paths
    ap.add_argument("--model_path", default="/mnt/data_1/yds/微调/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_path", default="/mnt/data_1/yds/微调/分类模型微调/数据集构造_new/微调BGE/训练/listwise微调/save_weights_ddp")
    ap.add_argument("--cls_head_path", default="/mnt/data_1/yds/微调/分类模型微调/数据集构造_new/微调BGE/训练/listwise微调/save_weights_ddp/rank_cls_head.pt")
    ap.add_argument("--score_head_path", default="/mnt/data_1/yds/微调/分类模型微调/数据集构造_new/微调BGE/训练/listwise微调/save_weights_ddp/rank_score_head.pt")

    # sweep config (around 0.5)
    ap.add_argument("--th_center", type=float, default=0.5)
    ap.add_argument("--th_span", type=float, default=0.06, help="sweep [center-span, center+span]")
    ap.add_argument("--th_step", type=float, default=0.01)
    ap.add_argument("--run_prefix", default="sev_partition")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) cache once
    cache_all_pairs(
        cfg_path=args.config,
        input_json=args.input_json,
        qids_file=args.qids_file,
        run_in=args.run_in,
        topk_in=args.topk_in,
        cache_out=args.cache_out,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        cls_head_path=args.cls_head_path,
        score_head_path=args.score_head_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
        prompt_template=args.prompt_template,
        text_field=args.text_field,
    )

    # 2) sweep thresholds (fast) for both heads
    thresholds = parse_thresholds(args.th_center, args.th_span, args.th_step)

    sweep_thresholds(
        cache_path=args.cache_out,
        out_dir=str(out_dir),
        qrels=args.qrels,
        eval_py=args.eval_py,
        topk_out=args.topk_out,
        thresholds=thresholds,
        signal="cls_prob",
        run_prefix=args.run_prefix,
    )

    sweep_thresholds(
        cache_path=args.cache_out,
        out_dir=str(out_dir),
        qrels=args.qrels,
        eval_py=args.eval_py,
        topk_out=args.topk_out,
        thresholds=thresholds,
        signal="score_prob",
        run_prefix=args.run_prefix,
    )


if __name__ == "__main__":
    main()
