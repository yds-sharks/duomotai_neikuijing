#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块名：batch_retrieve_top30_inproc.py
功能：同进程批量检索，输出 BGE-only 与 Hybrid 两份 Top-30 结果（供后续+Filter）。
"""
import argparse, json, pathlib, yaml
from bm25_bge_vectorstore_v2.retrieval_service import HybridRetriever

def read_questions(p: pathlib.Path):
    txt = p.read_text(encoding="utf-8").strip()
    try:
        obj = json.loads(txt)
        items = obj if isinstance(obj, list) else [obj]
    except Exception:
        # JSONL
        items = [json.loads(ln) for ln in txt.splitlines() if ln.strip()]
    seen, qs = set(), []
    for it in items:
        q = (it.get("question") or "").strip()
        if q and q not in seen:
            seen.add(q); qs.append(q)
    return qs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--embed-batch", type=int, default=64)
    ap.add_argument("--k-dense", type=int, default=120)   # 初选池 > topk，利于融合
    ap.add_argument("--k-sparse", type=int, default=800)
    args = ap.parse_args()

    inp = pathlib.Path(args.input).resolve()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text(encoding="utf-8"))
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    qs = read_questions(inp)
    print(f"[INFO] questions: {len(qs)}")

    # 构建一次即可，复用所有资源（jieba/bge/milvus）
    retriever = HybridRetriever(cfg)

    # 1) BGE-only（禁用语言自适配，确保权重固定为(1,0)）
    res_dense = retriever.search_batch(
        qs, topk=args.topk, mode="dense",
        disable_lang_adapt=True,
        k_dense=args.k_dense, k_sparse=0,
        embed_batch_size=args.embed_batch
    )
    p_dense = out_dir / f"BGE_only_top{args.topk}.jsonl"
    with p_dense.open("w", encoding="utf-8") as f:
        for r in res_dense: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] {p_dense}")

    # 2) Hybrid（沿用你 YAML 中的融合/自适配设置）
    res_hybrid = retriever.search_batch(
        qs, topk=args.topk, mode="hybrid",
        disable_lang_adapt=False,
        k_dense=args.k_dense, k_sparse=args.k_sparse,
        embed_batch_size=args.embed_batch
    )
    p_hybrid = out_dir / f"Hybrid_top{args.topk}.jsonl"
    with p_hybrid.open("w", encoding="utf-8") as f:
        for r in res_hybrid: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] {p_hybrid}")

if __name__ == "__main__":
    main()
