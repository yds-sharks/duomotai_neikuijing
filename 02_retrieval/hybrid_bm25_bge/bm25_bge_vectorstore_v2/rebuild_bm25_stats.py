#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重建 BM25 语料统计：
- 读取 config.yaml 的 paths.input_root
- 用 Tokenizer.tokenize_mixed（含用户词典/停用词/单字过滤）分词
- 生成 df(N 文档频次)、avgdl，并写到 paths.stats_out_dir/corpus_stats.json
- 注意：vocab_BM25.txt 由你已整理好的中英词汇构成；Tokenizer 会用它来构建 token2id
"""
import os, json, yaml, glob, collections

CFG = "/mnt/data_1/yds/RAG/Hybrid_milvus/总版/insert/config.yaml"

import sys
sys.path.insert(0, "/mnt/data_1/yds/RAG/Hybrid_milvus/总版/insert")
from bm25_bge_vectorstore_v2.tokenizers import Tokenizer

CAND_FIELDS = ["text","content","body","paragraph","chunk","passage","summary","abstract","raw","clean_text"]

def extract_text(obj):
    if isinstance(obj, dict):
        for k in CAND_FIELDS:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v
        # 兜底：找值里最长的字符串
        best = ""
        for v in obj.values():
            if isinstance(v, str) and len(v) > len(best):
                best = v
        return best
    return obj if isinstance(obj, str) else ""

def iter_docs(input_root: str):
    files = []
    for ext in ("*.json", "*.jsonl"):
        files += glob.glob(os.path.join(input_root, "**", ext), recursive=True)
    for p in files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                if p.endswith(".jsonl"):
                    for line in f:
                        line=line.strip()
                        if not line: continue
                        yield extract_text(json.loads(line)) or ""
                else:
                    obj = json.load(f)
                    if isinstance(obj, list):
                        for o in obj: yield extract_text(o) or ""
                    elif isinstance(obj, dict):
                        yield extract_text(obj) or ""
        except Exception as e:
            print(f"[warn] skip {p}: {e}")

def main():
    cfg = yaml.safe_load(open(CFG, "r", encoding="utf-8"))
    paths = cfg["paths"]
    input_root = paths["input_root"]
    vocab_path = paths["vocab_path"]
    stats_dir  = paths["stats_out_dir"]
    os.makedirs(stats_dir, exist_ok=True)
    stats_out = os.path.join(stats_dir, "corpus_stats.json")

    tok = Tokenizer(
        bm25_vocab_path=vocab_path,
        jieba_userdict_path=paths["jieba_userdict"],
        stopwords_zh_path=paths["stopwords_zh"],
        stopwords_en_path=paths["stopwords_en"],
        forbid_single_char=cfg["bm25"]["forbid_single_char"],
        max_terms_per_doc=cfg["bm25"]["max_terms_per_doc"],
    )

    N = 0
    df = collections.Counter()
    dl_sum = 0

    print(f"[scan] input_root = {input_root}")
    for text in iter_docs(input_root):
        toks = tok.tokenize_mixed(text or "")
        if not toks: continue
        N += 1
        dl_sum += len(toks)
        for w in set(toks):  # 文档频次
            df[w] += 1
        if N % 10000 == 0:
            print(f"  ... {N} docs")

    if N == 0:
        raise SystemExit("No docs scanned. Check input_root.")

    avgdl = dl_sum / N
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump({"df": df, "N": N, "avgdl": avgdl}, f, ensure_ascii=False)
    print(f"[done] N={N}, avgdl={avgdl:.2f}, df_size={len(df)} -> {stats_out}")

if __name__ == "__main__":
    main()
