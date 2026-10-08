#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, yaml, random, shutil
from collections import defaultdict

CFG = "/mnt/data_1/yds/RAG/Hybrid_milvus/总版/insert/config.yaml"

# ---- 载入配置与组件 ----
cfg = yaml.safe_load(open(CFG,"r",encoding="utf-8"))
vocab_path     = cfg["paths"]["vocab_path"]
stats_out_dir  = cfg["paths"]["stats_out_dir"]
stats_path     = os.path.join(stats_out_dir, "corpus_stats.json")

import sys
sys.path.insert(0, "/mnt/data_1/yds/RAG/Hybrid_milvus/总版/insert")
from bm25_bge_vectorstore_v2.tokenizers import Tokenizer

tok = Tokenizer(
    bm25_vocab_path=vocab_path,
    jieba_userdict_path=cfg["paths"]["jieba_userdict"],
    stopwords_zh_path=cfg["paths"]["stopwords_zh"],
    stopwords_en_path=cfg["paths"]["stopwords_en"],
    forbid_single_char=cfg["bm25"]["forbid_single_char"],
    max_terms_per_doc=cfg["bm25"]["max_terms_per_doc"],
)

# ---- 读取 stats ----
stats = json.load(open(stats_path,"r",encoding="utf-8"))
df_raw = stats["df"]
N      = int(stats["N"])
avgdl  = float(stats["avgdl"])

print(f"[INFO] vocab size={len(tok.token2id)} | df_size={len(df_raw)} | N={N} | avgdl={avgdl}")

# 采样看键类型
sample_keys = []
for k in df_raw.keys():
    sample_keys.append(k)
    if len(sample_keys) >= 10:
        break

def looks_like_int_string(s: str) -> bool:
    return isinstance(s, str) and s.isdigit()

mode = None
if isinstance(sample_keys[0], int):
    mode = "id_int"
elif looks_like_int_string(sample_keys[0]):
    mode = "id_str"
else:
    mode = "token_str"

print(f"[INFO] detected df-key mode = {mode}")

# ---- 对齐 / 转换 ----
if mode == "id_int":
    # 已经是 int id，无需改
    df_new = {int(k): int(v) for k, v in df_raw.items()}
elif mode == "id_str":
    # 字符串形式的 id，转成 int
    df_new = {int(k): int(v) for k, v in df_raw.items() if looks_like_int_string(k)}
else:
    # token 文本做键，需要映射到 token id
    missing, hit = 0, 0
    acc = defaultdict(int)
    for token, v in df_raw.items():
        tid = tok.token2id.get(token)
        if tid is None:
            missing += 1
            continue
        # 多个文本可能映射到同一 id（极少），取 df 最大值更稳妥
        acc[tid] = max(acc[tid], int(v))
        hit += 1
    df_new = dict(acc)
    print(f"[INFO] token→id mapped: hit={hit}, missing={missing}")

# ---- 保存覆盖（先备份） ----
bak = stats_path + ".bak"
if not os.path.exists(bak):
    shutil.copyfile(stats_path, bak)
    print(f"[INFO] backup written: {bak}")

stats["df"] = {str(k): int(v) for k, v in df_new.items()}  # JSON 里依旧用 str 键，BM25Vectorizer 读取时需转 int
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False)

print(f"[DONE] fixed df; new df_size={len(df_new)} (should be <= vocab size={len(tok.token2id)})")
