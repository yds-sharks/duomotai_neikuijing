#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import yaml
from pymilvus import Collection, connections

sys.path.insert(0, "/mnt/data_1/yds/多模态")
from insert.bm25_bge_vectorstore_v2.bm25_vectorizer import BM25Vectorizer
from insert.bm25_bge_vectorstore_v2.dense_embedder import BGEDense
from insert.bm25_bge_vectorstore_v2.text_utils import clean_text, detect_lang
from insert.bm25_bge_vectorstore_v2.tokenizers import Tokenizer


def zscore(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    sd = float(x.std())
    if not np.isfinite(sd) or sd < 1e-6:
        return x
    return (x - float(x.mean())) / sd


def to_sparse_map(vec: Dict) -> Dict[int, float]:
    idx = [int(i) for i in vec.get("indices", [])]
    val = [float(v) for v in vec.get("values", [])]
    acc = defaultdict(float)
    for i, v in zip(idx, val):
        if i >= 0 and np.isfinite(v):
            acc[i] += float(v)
    return {int(i): float(acc[i]) for i in sorted(acc.keys())}


def sparse_from_hits(hits, out_fields) -> List[Dict]:
    docs = []
    for h in hits:
        item = {"pk": int(h.id), "score_raw": float(h.distance)}
        ent = h.entity
        for f in out_fields:
            item[f] = ent.get(f)
        docs.append(item)
    return docs


def dense_from_hits(hits, out_fields) -> List[Dict]:
    docs = []
    for h in hits:
        item = {"pk": int(h.id), "score_raw": float(h.distance)}
        ent = h.entity
        for f in out_fields:
            item[f] = ent.get(f)
        docs.append(item)
    return docs


def build_retrieval(
    queries: List[str],
    *,
    topk: int,
    k_dense: int,
    k_sparse: int,
    w_dense: float,
    w_sparse: float,
    disable_lang_adapt: bool,
    nprobe: int,
) -> Dict[str, List[Dict]]:
    cfg = yaml.safe_load(open("/mnt/data_1/yds/多模态/insert/config.yaml", "r", encoding="utf-8"))
    stats_path = "/mnt/data_10/mwx/workspace/multi_modal_rag/build_database/project_ver2/build_vector_database/corpus_stats/corpus_stats.json"
    vocab_path = "/mnt/data_10/mwx/workspace/multi_modal_rag/build_database/project_ver2/build_vector_database/corpus_stats/vocab.txt"
    model_path = cfg["paths"]["model_path"]

    stats = json.load(open(stats_path, "r", encoding="utf-8"))
    tok = Tokenizer(
        bm25_vocab_path=vocab_path,
        jieba_userdict_path=cfg["paths"]["jieba_userdict"],
        stopwords_zh_path=cfg["paths"]["stopwords_zh"],
        stopwords_en_path=cfg["paths"]["stopwords_en"],
        forbid_single_char=cfg["bm25"]["forbid_single_char"],
        max_terms_per_doc=cfg["bm25"]["max_terms_per_doc"],
    )
    vectorizer = BM25Vectorizer(
        df=stats["df"],
        N=stats["N"],
        avgdl=stats["avgdl"],
        token2id=tok.token2id,
        k1=cfg["bm25"]["k1"],
        b=cfg["bm25"]["b"],
        idf_clip_min=cfg["bm25"]["idf_clip_min"],
        min_df=cfg["bm25"]["min_df"],
    )
    embedder = BGEDense(model_path, device="cuda")

    uri = "/mnt/data_1/yds/多模态/retrieval/test/vector_store_s3_filtered_labeled.db"
    connections.connect(alias="default", uri=uri)
    col = Collection("text_blocks")
    col.load()

    out_fields = ["doc_name", "doc_id", "page_idx", "block_id", "text", "summary", "metadata", "lang"]

    cleaned = [clean_text(q or "") for q in queries]
    langs = [detect_lang(q) if q else "unk" for q in cleaned]

    dense_vecs = embedder.encode_batch([q if q else "" for q in cleaned], batch_size=64)
    dense_res_all = col.search(
        data=[v.tolist() for v in dense_vecs],
        anns_field="dense_vector",
        param={"metric_type": "IP", "params": {"nprobe": int(nprobe)}},
        limit=int(k_dense),
        output_fields=out_fields,
    )

    sparse_queries = []
    sparse_valid_idx = []
    for i, q in enumerate(cleaned):
        toks = tok.tokenize_mixed(q) if q else []
        q_sparse = vectorizer.vectorize_query(toks)
        q_map = to_sparse_map(q_sparse)
        if q_map:
            sparse_valid_idx.append(i)
            sparse_queries.append(q_map)

    sparse_res_map = {}
    if sparse_queries:
        sparse_res_all = col.search(
            data=sparse_queries,
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=int(k_sparse),
            output_fields=out_fields,
        )
        for qi, hits in zip(sparse_valid_idx, sparse_res_all):
            sparse_res_map[qi] = hits

    result = {}
    for i, q in enumerate(cleaned):
        d_docs = dense_from_hits(dense_res_all[i], out_fields)
        s_hits = sparse_res_map.get(i, [])
        s_docs = sparse_from_hits(s_hits, out_fields)

        d_arr = np.array([x["score_raw"] for x in d_docs], dtype=np.float32)
        s_arr = np.array([x["score_raw"] for x in s_docs], dtype=np.float32)
        d_norm = zscore(d_arr) if d_arr.size else d_arr
        s_norm = zscore(s_arr) if s_arr.size else s_arr

        wd = float(w_dense)
        ws = float(w_sparse)
        if not disable_lang_adapt:
            if langs[i] == "zh":
                ws = float(cfg["retrieval"]["fusion"]["lang_adapt"]["zh_sparse"])
                wd = 1.0 - ws
            elif langs[i] == "en":
                ws = float(cfg["retrieval"]["fusion"]["lang_adapt"]["en_sparse"])
                wd = 1.0 - ws
        if len(s_docs) == 0:
            wd, ws = 1.0, 0.0

        pool = {}
        for idx, doc in enumerate(d_docs):
            pk = doc["pk"]
            if pk not in pool:
                pool[pk] = {
                    "pk": pk,
                    "doc_name": doc.get("doc_name"),
                    "doc_id": doc.get("doc_id"),
                    "page_idx": doc.get("page_idx"),
                    "block_id": doc.get("block_id"),
                    "text": doc.get("text"),
                    "summary": doc.get("summary"),
                    "metadata": doc.get("metadata"),
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                }
            pool[pk]["dense_score"] = float(d_norm[idx]) if idx < len(d_norm) else 0.0

        for idx, doc in enumerate(s_docs):
            pk = doc["pk"]
            if pk not in pool:
                pool[pk] = {
                    "pk": pk,
                    "doc_name": doc.get("doc_name"),
                    "doc_id": doc.get("doc_id"),
                    "page_idx": doc.get("page_idx"),
                    "block_id": doc.get("block_id"),
                    "text": doc.get("text"),
                    "summary": doc.get("summary"),
                    "metadata": doc.get("metadata"),
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                }
            pool[pk]["sparse_score"] = float(s_norm[idx]) if idx < len(s_norm) else 0.0

        ranked = []
        for v in pool.values():
            v["score"] = wd * float(v["dense_score"]) + ws * float(v["sparse_score"])
            ranked.append(v)
        ranked.sort(key=lambda x: x["score"], reverse=True)

        docs = []
        for rank, v in enumerate(ranked[: topk], 1):
            docs.append(
                {
                    "rank": rank,
                    "score": float(v["score"]),
                    "dense_score": float(v["dense_score"]),
                    "sparse_score": float(v["sparse_score"]),
                    "doc_name": v.get("doc_name"),
                    "doc_id": v.get("doc_id"),
                    "page_idx": v.get("page_idx"),
                    "block_id": v.get("block_id"),
                    "text": v.get("text"),
                    "summary": v.get("summary"),
                    "metadata": v.get("metadata"),
                }
            )
        result[q] = docs

    connections.disconnect("default")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--query-field", default="retrieval_query_rewritten")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--k-dense", type=int, default=200)
    p.add_argument("--k-sparse", type=int, default=200)
    p.add_argument("--w-dense", type=float, required=True)
    p.add_argument("--w-sparse", type=float, required=True)
    p.add_argument("--disable-lang-adapt", action="store_true")
    p.add_argument("--nprobe", type=int, default=64)
    args = p.parse_args()

    records = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    queries = []
    for r in records:
        q = r.get(args.query_field) or r.get("retrieval_query") or ""
        q = clean_text(q)
        if q:
            queries.append(q)
    unique_queries = list(dict.fromkeys(queries))

    retrieval_map = build_retrieval(
        unique_queries,
        topk=args.topk,
        k_dense=args.k_dense,
        k_sparse=args.k_sparse,
        w_dense=args.w_dense,
        w_sparse=args.w_sparse,
        disable_lang_adapt=bool(args.disable_lang_adapt),
        nprobe=args.nprobe,
    )

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for q in unique_queries:
            f.write(json.dumps({"query": q, "retrieval_docs": retrieval_map.get(q, [])}, ensure_ascii=False) + "\n")

    print(f"[done] queries={len(unique_queries)} output={args.output_jsonl}")


if __name__ == "__main__":
    main()
