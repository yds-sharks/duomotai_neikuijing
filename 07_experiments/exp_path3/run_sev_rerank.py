#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: run_sev_rerank.py
Purpose:
  Take an existing TREC run (qid->pk list), fetch passage text by pk from Milvus,
  run SEV (cls_head) to estimate P(support), then rerank and output a new TREC run.

Inputs:
  - --config: same YAML as retrieval (has milvus uri + collection)
  - --input_json: same as Path3 input (contains question/query)
  - --qids_file: eligible qids (optional)
  - --run_in: a .trec file (dense/sparse/hybrid)
Outputs:
  - --run_out: reranked .trec file for eval_runs.py
"""

import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from tqdm import tqdm

# your local model wrapper
from QwenRerankerModel import QwenRerankerModel


def support_prob_from_logits(logits: torch.Tensor, support_index: int = 1) -> torch.Tensor:
    if logits.ndim == 2 and logits.size(-1) == 2:
        return torch.softmax(logits, dim=-1)[:, support_index]
    elif (logits.ndim == 2 and logits.size(-1) == 1) or logits.ndim == 1:
        return torch.sigmoid(logits.view(-1))
    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


class OnlyCLSInfer:
    def __init__(
        self,
        model_path: str,
        adapter_path: str,
        cls_head_path: str,
        score_head_path: Optional[str] = None,
        device: Optional[str] = None,
        support_index: int = 1,
        max_length: int = 512,
        dtype: str = "bf16",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.support_index = int(support_index)
        self.max_length = int(max_length)

        torch_dtype = None
        if self.device.startswith("cuda"):
            if dtype.lower() in ("bf16", "bfloat16"):
                torch_dtype = torch.bfloat16
            elif dtype.lower() in ("fp16", "float16"):
                torch_dtype = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        base = PeftModel.from_pretrained(base, adapter_path)

        self.model = QwenRerankerModel(base)

        self.model.cls_head.load_state_dict(torch.load(cls_head_path, map_location="cpu"))
        if score_head_path:
            try:
                self.model.score_head.load_state_dict(torch.load(score_head_path, map_location="cpu"))
            except Exception as e:
                print(f"[WARN] score_head not loaded: {e}")

        self.model.to(self.device).eval()
        print(f"[INIT] device={self.device}, dtype={dtype}")

        self._shape_printed = False

    @torch.no_grad()
    def predict_probs(self, questions: List[str], texts: List[str], batch_size: int) -> List[float]:
        assert len(questions) == len(texts)
        if not texts:
            return []
        probs_all: List[float] = []

        # prompt: keep identical across experiments
        prompts = [f"问题：{q}\n内容：{t}" for q, t in zip(questions, texts)]

        for s in range(0, len(prompts), batch_size):
            chunk = prompts[s:s + batch_size]
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            pred_scores, logits = self.model(enc["input_ids"], enc["attention_mask"])
            if not self._shape_printed:
                print(f"[CHK] logits.shape = {tuple(logits.shape)}")
                self._shape_printed = True

            logits = logits.detach().float().cpu()
            probs = support_prob_from_logits(logits, support_index=self.support_index).cpu().view(-1).tolist()
            probs_all.extend([float(p) for p in probs])

        return probs_all


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


def iter_items_json_or_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt:
        return
    if path.endswith(".jsonl"):
        for line in txt.splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)
    else:
        obj = json.loads(txt)
        if isinstance(obj, list):
            for it in obj:
                yield it
        else:
            yield obj


def read_qid_to_question(input_json: str, eligible_qids: Optional[set], qid_field: str = "") -> Dict[str, str]:
    qmap: Dict[str, str] = {}
    for i, item in enumerate(iter_items_json_or_jsonl(input_json)):
        qid = str(item.get(qid_field)) if qid_field else str(i)
        if eligible_qids is not None and qid not in eligible_qids:
            continue
        q = (item.get("question") or item.get("query") or "").strip()
        qmap[qid] = q
    return qmap


def read_trec_run(run_in: str, topk_in: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    returns: qid -> list of {"pk": str, "rank": int, "score": float}
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(run_in, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, score, _ = parts[:6]
            if len(groups[qid]) < topk_in:
                groups[qid].append({"pk": str(docid), "rank": int(rank), "score": float(score)})
    return groups


def fetch_texts_by_pks(retr, pks: List[str], field: str = "text", batch: int = 512) -> Dict[str, str]:
    """
    Best-effort batch fetch. We try a few common access patterns.
    You may need to adjust this if your retrieval_service exposes a different API.
    """
    out: Dict[str, str] = {}
    # ensure unique
    uniq = []
    seen = set()
    for pk in pks:
        if pk not in seen:
            uniq.append(pk)
            seen.add(pk)

    for s in range(0, len(uniq), batch):
        chunk = uniq[s:s + batch]

        # Try: MilvusClient.query style
        got = None
        try:
            # pymilvus: collection.query(expr=..., output_fields=[...])
            expr = f"pk in [{', '.join(chunk)}]"
            got = retr.collection.query(expr=expr, output_fields=["pk", field])
        except Exception:
            pass

        if got is None:
            try:
                # MilvusClient.get / get_entities_by_ids
                got = retr.m.get(collection_name=retr.collection_name, ids=[int(x) for x in chunk], output_fields=[field])
            except Exception:
                got = None

        if got is None:
            raise RuntimeError(
                "[FETCH_FAIL] Cannot fetch texts by pk. "
                "Please expose retr.collection (pymilvus Collection) or retr.m.get/query in HybridRetriever."
            )

        # normalize output
        if isinstance(got, dict) and "data" in got:
            rows = got["data"]
        else:
            rows = got

        for r in rows:
            try:
                pk = str(r.get("pk") if isinstance(r, dict) else getattr(r, "pk"))
            except Exception:
                # sometimes returns {"id":..., "entity":{...}}
                pk = str(r.get("id")) if isinstance(r, dict) and "id" in r else None

            if pk is None:
                continue

            if isinstance(r, dict):
                if field in r:
                    out[pk] = (r.get(field) or "")
                elif "entity" in r and isinstance(r["entity"], dict):
                    out[pk] = (r["entity"].get(field) or "")
            else:
                out[pk] = getattr(r, field, "") or ""

    return out


def write_trec_run(run_out: str, qid: str, docs: List[Dict[str, Any]], run_name: str):
    with open(run_out, "a", encoding="utf-8") as fp:
        for i, d in enumerate(docs, start=1):
            fp.write(f"{qid}\tQ0\t{d['pk']}\t{i}\t{float(d['new_score'])}\t{run_name}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--run_in", required=True)
    ap.add_argument("--run_out", required=True)
    ap.add_argument("--qids_file", default="")
    ap.add_argument("--qid_field", default="")
    ap.add_argument("--topk_in", type=int, default=100)
    ap.add_argument("--topk_out", type=int, default=100)

    # SEV rerank behavior
    ap.add_argument("--sev_mode", choices=["partition", "score"], default="score")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=1.0, help="weight for original retr score")
    ap.add_argument("--beta", type=float, default=1.0, help="weight for cls_prob")

    # model
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--support_index", type=int, default=1)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--model_path", default="/mnt/data_1/yds/微调/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_path", default="/mnt/data_1/yds/微调/分类模型微调/数据集构造_new/微调BGE/训练/listwise微调/save_weights_ddp")
    ap.add_argument("--cls_head_path", default="/mnt/data_1/yds/微调/分类模型微调/数据集构造_new/微调BGE/训练/listwise微调/save_weights_ddp/rank_cls_head.pt")
    ap.add_argument("--score_head_path", default="/mnt/data_1/yds/微调/分类模型微调/数据集构造_new/微调BGE/训练/listwise微调/save_weights_ddp/rank_score_head.pt")

    args = ap.parse_args()

    # load cfg + retriever (for milvus fetch)
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from insert.bm25_bge_vectorstore_v2.retrieval_service import HybridRetriever
    retr = HybridRetriever(cfg)

    eligible = load_qids(args.qids_file)
    qmap = read_qid_to_question(args.input_json, eligible, qid_field=args.qid_field)

    groups = read_trec_run(args.run_in, topk_in=args.topk_in)

    # init model
    infer = OnlyCLSInfer(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        cls_head_path=args.cls_head_path,
        score_head_path=args.score_head_path,
        support_index=args.support_index,
        max_length=args.max_length,
        dtype=args.dtype,
    )

    run_out_path = Path(args.run_out)
    run_out_path.parent.mkdir(parents=True, exist_ok=True)
    if run_out_path.exists():
        run_out_path.unlink()

    # global stats
    total_pairs = 0
    total_relevant = 0

    for qid in tqdm(sorted(groups.keys(), key=lambda x: int(x) if x.isdigit() else x), desc="SEV rerank"):
        docs = groups[qid]
        if qid not in qmap:
            continue
        q = qmap[qid]

        pks = [d["pk"] for d in docs]
        pk2text = fetch_texts_by_pks(retr, pks, field="text", batch=512)

        texts = []
        qs = []
        valid_docs = []
        for d in docs:
            t = (pk2text.get(d["pk"]) or "").strip()
            if not t:
                # still keep it, but cls_prob=0
                d["cls_prob"] = 0.0
                d["cls_label"] = "irrelevant"
                d["new_score"] = float(d["score"])  # fallback
                valid_docs.append(d)
            else:
                qs.append(q)
                texts.append(t)
                valid_docs.append(d)

        # run model only for docs with text
        if texts:
            probs = infer.predict_probs(qs, texts, batch_size=args.batch_size)
        else:
            probs = []

        # write back probs in order
        p_idx = 0
        for d in valid_docs:
            if "cls_prob" in d:
                continue
            p = float(probs[p_idx])
            p_idx += 1
            d["cls_prob"] = p
            d["cls_label"] = "relevant" if p >= args.threshold else "irrelevant"

        # stats
        total_pairs += len(valid_docs)
        total_relevant += sum(1 for d in valid_docs if d["cls_label"] == "relevant")

        # rerank
        if args.sev_mode == "partition":
            rel = [d for d in valid_docs if d["cls_label"] == "relevant"]
            irrel = [d for d in valid_docs if d["cls_label"] != "relevant"]
            reranked = rel + irrel
            # keep a monotonic score for TREC (rank-based)
            for i, d in enumerate(reranked, start=1):
                d["new_score"] = float(len(reranked) - i + 1)
        else:
            # score fusion: sev_score = alpha*orig_score + beta*cls_prob
            for d in valid_docs:
                d["new_score"] = float(args.alpha) * float(d["score"]) + float(args.beta) * float(d["cls_prob"])
            reranked = sorted(valid_docs, key=lambda x: x["new_score"], reverse=True)

        reranked = reranked[: args.topk_out]

        write_trec_run(str(run_out_path), qid, reranked, run_name=f"sev_{args.sev_mode}_a{args.alpha}_b{args.beta}_t{args.threshold}")

    print(json.dumps({
        "run_in": args.run_in,
        "run_out": str(run_out_path),
        "num_qids": len(groups),
        "topk_in": args.topk_in,
        "topk_out": args.topk_out,
        "sev_mode": args.sev_mode,
        "threshold": args.threshold,
        "alpha": args.alpha,
        "beta": args.beta,
        "total_pairs": total_pairs,
        "relevant_ratio": (total_relevant / total_pairs) if total_pairs else 0.0
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
