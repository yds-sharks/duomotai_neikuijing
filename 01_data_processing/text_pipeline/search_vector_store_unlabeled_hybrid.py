#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无标签测试向量库：完整 BGE-M3 原生 hybrid 检索接口。

当前版本针对重建后的新 collection：
1. 全库存储 BGE-M3 dense_vector；
2. 全库存储 BGE-M3 原生 lexical sparse 字段；
3. query 同时走 dense / lexical 双路召回；
4. 对候选并集内每条文段重新计算 dense / lexical 分数，再做原生 hybrid 加权融合。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
import atexit
import argparse
import json
import os
import glob
import sys

from FlagEmbedding import BGEM3FlagModel
import numpy as np
import torch
from pymilvus import Collection, connections
from transformers import AutoModel
from huggingface_hub import snapshot_download
from FlagEmbedding.finetune.embedder.encoder_only.m3 import EncoderOnlyEmbedderM3Runner

sys.path.insert(0, "/mnt/data_1/yds/多模态/insert")
from bm25_bge_vectorstore_v2.text_utils import clean_text, detect_lang


def _normalize_pair(w_dense: float, w_sparse: float) -> Dict[str, float]:
    wd = max(0.0, float(w_dense))
    ws = max(0.0, float(w_sparse))
    s = wd + ws
    if s <= 0:
        return {"w_dense": 1.0, "w_sparse": 0.0}
    return {"w_dense": wd / s, "w_sparse": ws / s}


def patch_flagembedding_dtype_bug() -> None:
    def _patched_get_model(
        model_name_or_path: str,
        trust_remote_code: bool = False,
        colbert_dim: int = -1,
        cache_dir: str = None,
        torch_dtype: torch.dtype | None = None,
    ):
        cache_folder = os.getenv("HF_HUB_CACHE", None) if cache_dir is None else cache_dir
        if not os.path.exists(model_name_or_path):
            model_name_or_path = snapshot_download(
                repo_id=model_name_or_path,
                cache_dir=cache_folder,
                ignore_patterns=["flax_model.msgpack", "rust_model.ot", "tf_model.h5"],
            )

        model = AutoModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_folder,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        colbert_linear = torch.nn.Linear(
            in_features=model.config.hidden_size,
            out_features=model.config.hidden_size if colbert_dim <= 0 else colbert_dim,
            dtype=torch_dtype,
        )
        sparse_linear = torch.nn.Linear(
            in_features=model.config.hidden_size,
            out_features=1,
            dtype=torch_dtype,
        )

        colbert_model_path = os.path.join(model_name_or_path, "colbert_linear.pt")
        sparse_model_path = os.path.join(model_name_or_path, "sparse_linear.pt")
        if os.path.exists(colbert_model_path) and os.path.exists(sparse_model_path):
            colbert_state_dict = torch.load(colbert_model_path, map_location="cpu", weights_only=True)
            sparse_state_dict = torch.load(sparse_model_path, map_location="cpu", weights_only=True)
            colbert_linear.load_state_dict(colbert_state_dict)
            sparse_linear.load_state_dict(sparse_state_dict)

        return {
            "model": model,
            "colbert_linear": colbert_linear,
            "sparse_linear": sparse_linear,
        }

    EncoderOnlyEmbedderM3Runner.get_model = staticmethod(_patched_get_model)

@dataclass
class HybridSearchConfig:
    milvus_db_path: str = (
        "/mnt/data_1/yds/多模态/data_house/"
        "test_ingest_188w_20260508_bgem3/u_bgem3_hybrid.db"
    )
    collection_name: str = "text_blocks_bge_m3_native_hybrid"
    dense_field: str = "dense_vector"
    sparse_field: str = "bge_m3_lexical_sparse"
    model_path: str = "/mnt/data_1/yds/RAG/Hybrid_milvus/总版/pretrained_models/BAAI/bge-m3"
    device: Optional[str] = None
    nprobe: int = 64
    dense_topk: int = 10
    sparse_topk: int = 10
    max_query_length: int = 256
    max_passage_length: int = 512
    use_fp16: bool = True
    w_dense: float = 0.5
    w_sparse: float = 0.5
    zh_sparse: float = 0.5
    en_sparse: float = 0.5
    disable_lang_adapt: bool = True


def _find_milvus_lite_pids(db_path: str) -> List[int]:
    db_real = os.path.realpath(db_path)
    pids: List[int] = []
    for proc_dir in glob.glob("/proc/[0-9]*"):
        pid_str = os.path.basename(proc_dir)
        cmdline_path = os.path.join(proc_dir, "cmdline")
        try:
            raw = Path(cmdline_path).read_bytes()
        except Exception:
            continue
        if not raw:
            continue
        parts = [p.decode("utf-8", errors="ignore") for p in raw.split(b"\x00") if p]
        joined = " ".join(parts)
        if "milvus_lite/lib/milvus" in joined and db_real in joined:
            try:
                pids.append(int(pid_str))
            except Exception:
                pass
    return pids


def _prepare_milvus_lite_lock(db_path: str) -> None:
    lock_path = os.path.join(os.path.dirname(db_path), f".{os.path.basename(db_path)}.lock")
    if not os.path.exists(lock_path):
        return

    live_pids = _find_milvus_lite_pids(db_path)
    if live_pids:
        raise RuntimeError(
            f"Milvus Lite database is busy: db={db_path} lock={lock_path} active_pids={live_pids}"
        )

    try:
        os.remove(lock_path)
        print(f"[info] removed stale lock: {lock_path}", flush=True)
    except Exception as e:
        raise RuntimeError(f"Failed to remove stale Milvus Lite lock {lock_path}: {e}") from e


class NativeBGEM3HybridEngine:
    def __init__(self, config: Optional[HybridSearchConfig] = None):
        self.config = config or HybridSearchConfig()
        if self.config.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        _prepare_milvus_lite_lock(self.config.milvus_db_path)
        self.alias = f"lite_{os.getpid()}"
        self.is_lite = not str(self.config.milvus_db_path).lower().startswith(("http://", "https://", "tcp://"))
        connections.connect(alias=self.alias, uri=self.config.milvus_db_path)
        self.collection = Collection(self.config.collection_name, using=self.alias)
        if not self.is_lite:
            self.collection.load()
        atexit.register(self.close)

        patch_flagembedding_dtype_bug()
        self.model = BGEM3FlagModel(
            self.config.model_path,
            normalize_embeddings=True,
            use_fp16=bool(self.config.use_fp16 and str(self.device).startswith("cuda")),
            devices=self.device,
            batch_size=32,
            query_max_length=int(self.config.max_query_length),
            passage_max_length=int(self.config.max_passage_length),
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        field_names = [f.name for f in self.collection.schema.fields]
        if self.config.dense_field not in field_names:
            raise ValueError(
                f"dense field not found: {self.config.dense_field}, "
                f"available={field_names}"
            )
        if self.config.sparse_field not in field_names:
            raise ValueError(
                f"sparse field not found: {self.config.sparse_field}, "
                f"available={field_names}"
            )

    def close(self) -> None:
        try:
            self.collection.release()
        except Exception:
            pass
        try:
            connections.disconnect(self.alias)
        except Exception:
            pass

    def _search_with_retry(self, data: List[Any], anns_field: str, param: Dict[str, Any], limit: int, output_fields: List[str]):
        try:
            return self.collection.search(
                data=data,
                anns_field=anns_field,
                param=param,
                limit=limit,
                output_fields=output_fields,
            )[0]
        except Exception as e:
            msg = str(e).lower()
            if self.is_lite and ("load" in msg or "loaded" in msg or "memory" in msg):
                self.collection.load()
                return self.collection.search(
                    data=data,
                    anns_field=anns_field,
                    param=param,
                    limit=limit,
                    output_fields=output_fields,
                )[0]
            raise

    def _weights(self, lang: str, has_lexical: bool) -> Dict[str, float]:
        if not has_lexical:
            return {"w_dense": 1.0, "w_sparse": 0.0}

        w_dense = float(self.config.w_dense)
        w_sparse = float(self.config.w_sparse)
        if not self.config.disable_lang_adapt:
            if lang == "zh":
                w_sparse = float(self.config.zh_sparse)
                w_dense = 1.0 - w_sparse
            elif lang == "en":
                w_sparse = float(self.config.en_sparse)
                w_dense = 1.0 - w_sparse
        return _normalize_pair(w_dense, w_sparse)

    def _encode_query(self, query: str) -> Dict[str, Any]:
        out = self.model.encode(
            [query],
            batch_size=1,
            max_length=int(self.config.max_query_length),
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vec = out["dense_vecs"][0]
        lexical = {int(k): float(v) for k, v in out["lexical_weights"][0].items() if float(v) > 0.0}
        return {"dense_vec": dense_vec, "lexical_weights": lexical}

    @staticmethod
    def _normalize_sparse_vector(vec: Any) -> Dict[int, float]:
        if not vec:
            return {}
        if isinstance(vec, dict):
            return {int(k): float(v) for k, v in vec.items() if float(v) != 0.0}
        return {}

    @staticmethod
    def _dense_dot(query_vec: np.ndarray, doc_vec: Any) -> float:
        if doc_vec is None:
            return 0.0
        doc_arr = np.asarray(doc_vec, dtype=np.float32)
        if doc_arr.size == 0:
            return 0.0
        return float(np.dot(query_vec, doc_arr))

    @staticmethod
    def _sparse_dot(query_lexical: Dict[int, float], doc_lexical: Dict[int, float]) -> float:
        if not query_lexical or not doc_lexical:
            return 0.0
        if len(query_lexical) > len(doc_lexical):
            query_lexical, doc_lexical = doc_lexical, query_lexical
        return float(sum(float(v) * float(doc_lexical.get(k, 0.0)) for k, v in query_lexical.items()))

    def search(self, query: str, topk: int = 10) -> Dict[str, Any]:
        q = clean_text(str(query or ""))
        if not q:
            return {
                "query": "",
                "lang": "mix",
                "weights": {"w_dense": 0.0, "w_sparse": 0.0},
                "mode": "bge-m3-native-hybrid-rerank",
                "candidate_pool_size": 0,
                "query_lexical_nonzero": 0,
                "results": [],
            }

        q_lang = detect_lang(q)
        out_fields = [
            "doc_name",
            "doc_id",
            "page_idx",
            "text",
            "summary",
            "block_id",
            "metadata",
            "lang",
            self.config.dense_field,
            self.config.sparse_field,
        ]

        q_enc = self._encode_query(q)
        q_dense = q_enc["dense_vec"]
        q_lexical = q_enc["lexical_weights"]

        dense_k = max(int(topk), int(self.config.dense_topk))
        dense_hits = self._search_with_retry(
            data=[q_dense.tolist()],
            anns_field=self.config.dense_field,
            param={"metric_type": "IP", "params": {"nprobe": int(self.config.nprobe)}},
            limit=dense_k,
            output_fields=out_fields,
        )

        weights = self._weights(q_lang, has_lexical=bool(q_lexical))
        wd = float(weights["w_dense"])
        ws = float(weights["w_sparse"])

        sparse_hits = []
        if q_lexical:
            sparse_k = max(int(topk), int(self.config.sparse_topk))
            sparse_hits = self._search_with_retry(
                data=[q_lexical],
                anns_field=self.config.sparse_field,
                param={"metric_type": "IP", "params": {}},
                limit=sparse_k,
                output_fields=out_fields,
            )

        merged: Dict[int, Dict[str, Any]] = {}
        for hit in dense_hits:
            ent = hit.entity
            pk = int(hit.id)
            merged[pk] = {
                "pk": pk,
                "doc_name": ent.get("doc_name"),
                "doc_id": ent.get("doc_id"),
                "page_idx": ent.get("page_idx"),
                "text": ent.get("text"),
                "summary": ent.get("summary"),
                "block_id": ent.get("block_id"),
                "metadata": ent.get("metadata"),
                "lang": ent.get("lang"),
                "_dense_vector": ent.get(self.config.dense_field),
                "_sparse_vector": self._normalize_sparse_vector(ent.get(self.config.sparse_field)),
                "dense_recalled": True,
                "sparse_recalled": False,
                "dense_recall_score": float(hit.distance),
                "sparse_recall_score": 0.0,
                "dense_score_raw": float(hit.distance),
                "dense_score": float(hit.distance),
                "lexical_score_raw": 0.0,
                "lexical_score": 0.0,
                "sparse_score_raw": 0.0,
                "sparse_score": 0.0,
            }

        for hit in sparse_hits:
            ent = hit.entity
            pk = int(hit.id)
            row = merged.get(pk)
            if row is None:
                row = {
                    "pk": pk,
                    "doc_name": ent.get("doc_name"),
                    "doc_id": ent.get("doc_id"),
                    "page_idx": ent.get("page_idx"),
                    "text": ent.get("text"),
                    "summary": ent.get("summary"),
                    "block_id": ent.get("block_id"),
                    "metadata": ent.get("metadata"),
                    "lang": ent.get("lang"),
                    "_dense_vector": ent.get(self.config.dense_field),
                    "_sparse_vector": self._normalize_sparse_vector(ent.get(self.config.sparse_field)),
                    "dense_recalled": False,
                    "sparse_recalled": True,
                    "dense_recall_score": 0.0,
                    "sparse_recall_score": 0.0,
                    "dense_score_raw": 0.0,
                    "dense_score": 0.0,
                    "lexical_score_raw": 0.0,
                    "lexical_score": 0.0,
                    "sparse_score_raw": 0.0,
                    "sparse_score": 0.0,
                }
                merged[pk] = row
            if row.get("_dense_vector") is None:
                row["_dense_vector"] = ent.get(self.config.dense_field)
            if not row.get("_sparse_vector"):
                row["_sparse_vector"] = self._normalize_sparse_vector(ent.get(self.config.sparse_field))
            row["sparse_recalled"] = True
            row["sparse_recall_score"] = float(hit.distance)

        q_dense_arr = np.asarray(q_dense, dtype=np.float32)

        ranked: List[Dict[str, Any]] = []
        for item in merged.values():
            dense_score = self._dense_dot(q_dense_arr, item.get("_dense_vector"))
            lexical_score = self._sparse_dot(
                q_lexical,
                item.get("_sparse_vector") or {},
            )
            score = wd * dense_score + ws * lexical_score
            row = dict(item)
            row.pop("_dense_vector", None)
            row.pop("_sparse_vector", None)
            row["dense_score_raw"] = float(dense_score)
            row["dense_score"] = float(dense_score)
            row["lexical_score_raw"] = float(lexical_score)
            row["lexical_score"] = float(lexical_score)
            row["sparse_score_raw"] = float(lexical_score)
            row["sparse_score"] = float(lexical_score)
            row["fusion_raw"] = float(score)
            row["fusion_score"] = float(score)
            row["score"] = float(score)
            row["w_dense"] = wd
            row["w_sparse"] = ws
            ranked.append(row)

        ranked.sort(key=lambda x: x["score"], reverse=True)
        results = []
        for rank, item in enumerate(ranked[: max(1, int(topk))], start=1):
            row = dict(item)
            row["rank"] = rank
            results.append(row)

        return {
            "query": q,
            "lang": q_lang,
            "mode": "bge-m3-native-hybrid-candidate-rescore",
            "weights": {"w_dense": wd, "w_sparse": ws},
            "candidate_pool_size": len(merged),
            "dense_candidate_count": len(dense_hits),
            "sparse_candidate_count": len(sparse_hits),
            "query_lexical_nonzero": len(q_lexical),
            "notes": {
                "retrieval_stage": "full-corpus dense + lexical retrieval from Milvus",
                "rerank_stage": "recompute dense + lexical on candidate union",
                "normalization": "none",
            },
            "results": results,
        }

    def batch_search(self, queries: List[str], topk: int = 10) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for q in queries:
            try:
                out[q] = self.search(q, topk=topk)
            except Exception as e:
                out[q] = {
                    "query": q,
                    "error": str(e),
                    "results": [],
                }
        return out


_global_engine: Optional[NativeBGEM3HybridEngine] = None


def _get_global_engine() -> NativeBGEM3HybridEngine:
    global _global_engine
    if _global_engine is None:
        _global_engine = NativeBGEM3HybridEngine()
    return _global_engine


def search(query: str, topk: int = 10) -> Dict[str, Any]:
    engine = _get_global_engine()
    return engine.search(query, topk=topk)


def batch_search(queries: List[str], topk: int = 10) -> Dict[str, Dict[str, Any]]:
    engine = _get_global_engine()
    return engine.batch_search(queries, topk=topk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BGE-M3 native full hybrid search for unlabeled vector store")
    parser.add_argument("--query", type=str, default="溃疡性结肠炎的Mayo内镜评分标准是什么？")
    parser.add_argument("--milvus-db", type=str, default="")
    parser.add_argument("--collection", type=str, default="")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--dense-topk", type=int, default=120)
    parser.add_argument("--sparse-topk", type=int, default=120)
    parser.add_argument("--max-query-length", type=int, default=256)
    parser.add_argument("--max-passage-length", type=int, default=512)
    parser.add_argument("--w-dense", type=float, default=0.5)
    parser.add_argument("--w-sparse", type=float, default=0.5)
    parser.add_argument("--zh-sparse", type=float, default=0.5)
    parser.add_argument("--en-sparse", type=float, default=0.5)
    parser.add_argument("--disable-lang-adapt", action="store_true")
    parser.add_argument("--json", action="store_true", help="print full JSON")
    args = parser.parse_args()

    cfg = HybridSearchConfig(
        milvus_db_path=args.milvus_db or HybridSearchConfig.milvus_db_path,
        collection_name=args.collection or HybridSearchConfig.collection_name,
        dense_topk=int(args.dense_topk),
        sparse_topk=int(args.sparse_topk),
        max_query_length=int(args.max_query_length),
        max_passage_length=int(args.max_passage_length),
        w_dense=float(args.w_dense),
        w_sparse=float(args.w_sparse),
        zh_sparse=float(args.zh_sparse),
        en_sparse=float(args.en_sparse),
        disable_lang_adapt=bool(args.disable_lang_adapt),
    )
    engine = NativeBGEM3HybridEngine(config=cfg)
    try:
        out = engine.search(args.query, topk=args.topk)
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(f"query: {out['query']}")
            print(
                f"lang: {out.get('lang')} mode={out.get('mode')} "
                f"weights={out.get('weights')} candidate_pool={out.get('candidate_pool_size')} "
                f"dense_candidates={out.get('dense_candidate_count')} "
                f"sparse_candidates={out.get('sparse_candidate_count')} "
                f"query_lexical_nonzero={out.get('query_lexical_nonzero')}"
            )
            for r in out.get("results", []):
                print(
                    f"[{r['rank']}] score={r['score']:.4f} "
                    f"dense={r['dense_score_raw']:.4f} lexical={r['lexical_score_raw']:.4f} "
                    f"recall_dense={r['dense_recall_score']:.4f} "
                    f"{r.get('doc_name')} (p.{r.get('page_idx')})"
                )
    finally:
        engine.close()
