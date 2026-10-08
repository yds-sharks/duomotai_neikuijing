#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从旧集合读取：pk, text, summary, file_name, metadata, lang, dense_vector, text_hash
用【新词表+新 df/N/avgdl】重算稀疏向量，插入到新集合；复用原 dense_vector
注意：不使用 offset 翻页；改为按 pk 的区间递归切片，规避 16384 限制。
"""

import os, json, yaml, math, sys, hashlib
from typing import List, Dict, Any, Tuple, Set
from pymilvus import (
    connections, utility, Collection, CollectionSchema, FieldSchema, DataType
)

# ======= 配置 =======
CFG = "/mnt/data_1/yds/RAG/Hybrid_milvus/总版/insert/config.yaml"
SRC_COLLECTION = "cervix_v2"   # 旧集合
DST_COLLECTION = "cervix_v3"   # 新集合
BATCH = 5000                   # 插入批大小
PK_WINDOW_START = 0            # pk 起始扫描位置
PK_WINDOW_STEP  = 10_000_000   # 初始窗口宽度；窗口内>16384条会自动二分

sys.path.insert(0, "/mnt/data_1/yds/RAG/Hybrid_milvus/总版/insert")
from bm25_bge_vectorstore_v2.tokenizers import Tokenizer
from bm25_bge_vectorstore_v2.bm25_vectorizer import BM25Vectorizer

def sha1_hex(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

# ======= 工具 =======
def clone_schema(src: Collection) -> CollectionSchema:
    new_fields: List[FieldSchema] = []
    for f in src.schema.fields:
        kw = dict(
            name=f.name,
            dtype=f.dtype,
            is_primary=f.is_primary,
            auto_id=f.auto_id,
            description=f.description or "",
        )
        if f.dtype == DataType.VARCHAR:
            kw["max_length"] = int(getattr(f, "max_length", 1024) or 1024)
        elif f.dtype == DataType.FLOAT_VECTOR:
            dim = None
            if hasattr(f, "params") and isinstance(f.params, dict):
                dim = int(f.params.get("dim") or f.params.get("max_length") or 0)
            if not dim:
                dim = int(getattr(f, "dim", 0)) or 1024
            kw["dim"] = dim
        elif f.dtype == DataType.SPARSE_FLOAT_VECTOR:
            pass
        new_fields.append(FieldSchema(**kw))
    return CollectionSchema(fields=new_fields, description="rebuild sparse only")

def build_dst_if_needed(src_col: Collection, dst_name: str) -> Collection:
    if utility.has_collection(dst_name):
        return Collection(dst_name)
    schema = clone_schema(src_col)
    c = Collection(name=dst_name, schema=schema, shards_num=2)
    return c

def to_sparse_row(vec: Dict[str, Any]) -> Dict[str, Any]:
    # 统一成 {"indices":[int...], "values":[float...]}
    idx = [int(i) for i in vec.get("indices", [])]
    val = [float(v) for v in vec.get("values", [])]
    return {"indices": idx, "values": val}

def filter_tokens_to_vocab(tok, tokens: List[str]) -> List[str]:
    t2i = tok.token2id
    kept = []
    for t in tokens:
        if t in t2i:
            kept.append(t); continue
        t_norm = t.lower()
        if t_norm in t2i:
            kept.append(t_norm)
    return kept

# ======= 主流程 =======
def main():
    cfg = yaml.safe_load(open(CFG, "r", encoding="utf-8"))
    mil = cfg["milvus"]
    connections.connect(
        "default",
        uri=mil["uri"],
        user=mil.get("user", ""),
        password=mil.get("password", ""),
        db_name=mil.get("db_name", "default"),
    )

    src = Collection(SRC_COLLECTION); src.load()
    print("[src] num_entities =", src.num_entities)

    # tokenizer & BM25（使用你合并后的词表 + 对齐后的 df/N/avgdl）
    tok = Tokenizer(
        bm25_vocab_path=cfg["paths"]["vocab_path"],
        jieba_userdict_path=cfg["paths"]["jieba_userdict"],
        stopwords_zh_path=cfg["paths"]["stopwords_zh"],
        stopwords_en_path=cfg["paths"]["stopwords_en"],
        forbid_single_char=cfg["bm25"]["forbid_single_char"],
        max_terms_per_doc=cfg["bm25"]["max_terms_per_doc"],
    )
    stats = json.load(open(os.path.join(cfg["paths"]["stats_out_dir"], "corpus_stats.json"), "r", encoding="utf-8"))
    vectorizer = BM25Vectorizer(
        df=stats["df"], N=stats["N"], avgdl=stats["avgdl"],
        token2id=tok.token2id,
        k1=cfg["bm25"]["k1"], b=cfg["bm25"]["b"],
        idf_clip_min=cfg["bm25"]["idf_clip_min"], min_df=cfg["bm25"]["min_df"],
    )

    dst = build_dst_if_needed(src, DST_COLLECTION)

    # 目标列顺序严格按目标 schema 来
    field_names = [f.name for f in dst.schema.fields]
    print("[schema] src PK={[f.name for f in src.schema.fields if f.is_primary] } | dst PK={[f.name for f in dst.schema.fields if f.is_primary] }")
    print("[schema] src fields:", [f.name for f in src.schema.fields])
    print("[schema] dst fields:", field_names)

    # ✅ 必须包含 text_hash，否则会插入 None 触发 ParamError
    required_fields = set(["pk","text_hash","text","summary","file_name","metadata","lang","dense_vector","sparse_vector"])
    missing = required_fields - set(field_names)
    if missing:
        raise RuntimeError(f"目标集合缺少字段: {missing}. 请检查 schema。")

    # 扫描控制
    total_target = src.num_entities
    seen_pks: Set[int] = set()
    insert_buffer: Dict[str, list] = {name: [] for name in field_names}

    # 将一批 rows 处理 -> 插入缓冲
    def process_rows(rows: List[Dict[str, Any]]):
        nonlocal insert_buffer, seen_pks
        if not rows: return

        idx_keep = []
        for i, r in enumerate(rows):
            pk = int(r["pk"])
            if pk in seen_pks:
                continue
            seen_pks.add(pk)

            text = r.get("text") or ""
            toks = tok.tokenize_mixed(text)
            toks = filter_tokens_to_vocab(tok, toks)
            sp = vectorizer.vectorize_doc(toks)
            if not sp.get("indices"):
                # 空稀疏直接跳过该行（可按需放开）
                continue
            idx_keep.append(i)

            # 先填充非稀疏列，稀疏向量稍后统一追加
            for name in field_names:
                if name == "sparse_vector": 
                    continue
                if name == "text_hash":
                    # 优先用旧集合的 text_hash；若不存在或为空，则按 text 现算
                    th = r.get("text_hash")
                    if not th:
                        th = sha1_hex(text)
                    insert_buffer[name].append(th)
                    continue
                if name in r:
                    insert_buffer[name].append(r[name])
                else:
                    # 兼容可能没有的字段（尽量避免，特别是 VARCHAR 禁止 None）
                    if name in ("file_name","lang","summary","text"):
                        insert_buffer[name].append(r.get(name, ""))  # 用空串兜底
                    elif name == "metadata":
                        insert_buffer[name].append(r.get(name, {}))
                    else:
                        insert_buffer[name].append(None)

        # 稀疏列
        for i in idx_keep:
            text = rows[i].get("text") or ""
            toks = tok.tokenize_mixed(text)
            toks = filter_tokens_to_vocab(tok, toks)
            sp = vectorizer.vectorize_doc(toks)
            insert_buffer["sparse_vector"].append(to_sparse_row(sp))

        # 如果缓冲达到批大小，落库
        if len(insert_buffer["pk"]) >= BATCH:
            flush_buffer()

    def flush_buffer():
        nonlocal insert_buffer
        n = len(insert_buffer["pk"])
        if n == 0: return
        columns = [insert_buffer[name] for name in field_names]
        # 安全检查：所有列长度一致
        lens = list(map(len, columns))
        if len(set(lens)) != 1:
            raise RuntimeError(f"列长度不一致: {dict(zip(field_names, lens))}")
        dst.insert(columns)
        for name in field_names:
            insert_buffer[name] = []
        # 定期 flush
        if len(seen_pks) % (BATCH * 10) == 0:
            dst.flush()
        print(f"[write] total_inserted={len(seen_pks)}")

    # 递归拉取一个 pk 区间 [lo, hi)
    def drain_range(lo: int, hi: int):
        # 返回数量不超过 16384；超过则二分
        expr = f"pk >= {lo} && pk < {hi}"
        rows = src.query(expr=expr, limit=16384, output_fields=[
            # ✅ 一定要把 text_hash 拉出来；旧库没有也不怕，我们会在插入前兜底
            "pk","text_hash","text","summary","file_name","metadata","lang","dense_vector"
        ])
        if not rows:
            return
        if len(rows) >= 16384:
            mid = (lo + hi) // 2
            if mid == lo or mid == hi:
                process_rows(rows)
                return
            drain_range(lo, mid)
            drain_range(mid, hi)
        else:
            process_rows(rows)

    print(f"[run] total={total_target} | batch={BATCH}")
    # 逐窗口推进；直到处理满 total_target 条为止
    lo = PK_WINDOW_START
    step = PK_WINDOW_STEP
    while len(seen_pks) < total_target:
        hi = lo + step
        drain_range(lo, hi)
        lo = hi
        # 安全退出条件，避免极端 pk 稀疏拉不到足量数据
        if lo > (1 << 62) and len(seen_pks) == 0:
            raise RuntimeError("未能拉取到任何行，请检查 pk 字段/表达式。")
        # 进度日志
        print(f"[scan] pk_window=[{hi-step}, {hi}) | seen={len(seen_pks)}/{total_target}")

    # flush & 建索引
    flush_buffer()
    dst.flush()
    print(f"[done insert] total written(unique pk) = {len(seen_pks)}")

    # === 全量插入完成后再创建索引 ===
    idx_d = mil["index_dense"]; idx_s = mil["index_sparse"]

    print("[index] building dense index ...")
    dst.create_index(
        field_name="dense_vector",
        index_name="dense_ivf",
        index_params={
            "index_type": idx_d["type"],
            "metric_type": idx_d.get("metric") or idx_d.get("metric_type") or "IP",
            "params": idx_d.get("params", {}),
        },
    )
    print("[index] building sparse index ...")
    dst.create_index(
        field_name="sparse_vector",
        index_name="sparse_sii",
        index_params={
            "index_type": idx_s["type"],
            "metric_type": idx_s.get("metric") or idx_s.get("metric_type") or "IP",
            "params": idx_s.get("params", {}),
        },
    )

    dst.load()
    print("[dst] num_entities =", dst.num_entities)
    print("[ok] rebuilt sparse only, reused dense. Switch config.yaml to collection =", DST_COLLECTION)


if __name__ == "__main__":
    main()
