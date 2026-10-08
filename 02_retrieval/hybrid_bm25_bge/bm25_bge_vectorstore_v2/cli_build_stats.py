#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from collections import defaultdict
from glob import glob
from typing import Dict, Iterable, List, Set, Tuple

import jieba
import yaml
from tqdm import tqdm

# 英文分词正则：字母/数字/常见撇号/连接符
# 放在文件顶端 regex 附近
_EN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9’'_-]*")  # 必须以字母开头

def _is_bad_en_token(t: str) -> bool:
    t = t.strip().lower()
    if len(t) < 3 or len(t) > 40:
        return True
    # 去掉两端符号后再判断
    t_stripped = t.strip("-'_")
    if not t_stripped or len(t_stripped) < 2:
        return True
    # 不允许首尾是连字符/撇号
    if t[0] in "-'_" or t[-1] in "-'_":
        return True
    # 连续同字符>=4（多为噪声/拉长）
    if re.search(r"(.)\1\1\1", t):
        return True
    # 只含 A/C/G/T 且很长（DNA 串），阈值可调
    if re.fullmatch(r"[acgt]+", t) and len(t) >= 12:
        return True
    # 由多个词粘连且无元音迹象（极端保守，可按需关闭）
    if not re.search(r"[aeiou]", t) and not re.search(r"\d", t):
        return True
    return False



def _load_set(path: str) -> Set[str]:
    if not path or not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {w.strip() for w in f if w.strip()}


def _iter_files(root: str) -> List[str]:
    pats = ["**/*.json", "**/*.jsonl"]
    files: List[str] = []
    for p in pats:
        files.extend(glob(os.path.join(root, p), recursive=True))
    return files


def _iter_json_records(fp: str, field: str) -> Iterable[str]:
    """逐条产出指定字段文本（原文/摘要）。"""
    if fp.lower().endswith(".jsonl"):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    t = obj.get(field, "")
                    if isinstance(t, str) and t.strip():
                        yield t
                except Exception:
                    continue
    else:
        with open(fp, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                data = []
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    t = obj.get(field, "")
                    if isinstance(t, str) and t.strip():
                        yield t


def _tokenize_zh(text: str, stop_zh: Set[str], forbid_single_char: bool) -> List[str]:
    toks: List[str] = []
    for w in jieba.cut(text, cut_all=False, HMM=True):
        w = w.strip()
        if not w:
            continue
        if forbid_single_char and len(w) == 1:
            # 单字且不强制保留，直接丢弃
            continue
        if w in stop_zh:
            continue
        toks.append(w)
    return toks


def _tokenize_en(text: str, stop_en: Set[str]) -> List[str]:
    words = _EN_TOKEN_RE.findall(text)
    out = []
    for w in words:
        wl = w.lower()
        if wl in stop_en:
            continue
        if _is_bad_en_token(wl):
            continue
        out.append(wl)
    return out



def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="构建 BM25 词表与统计（可开关中/英），写入 vocab.txt 与 corpus_stats.json"
    )
    ap.add_argument("--config", required=True, help="路径到 config.yaml")

    # 可覆盖/附加的命令行参数
    ap.add_argument("--enable-zh", action="store_true", help="启用中文分词（默认从配置推断，命令行置位后强制启用）")
    ap.add_argument("--disable-zh", action="store_true", help="禁用中文分词（命令行置位后强制禁用）")
    ap.add_argument("--enable-en", action="store_true", help="启用英文分词（默认从配置推断，命令行置位后强制启用）")
    ap.add_argument("--disable-en", action="store_true", help="禁用英文分词（命令行置位后强制禁用）")

    ap.add_argument("--use-field", choices=["原文", "摘要"], help="从语料中选用的字段（默认读取配置，缺省用 原文）")
    ap.add_argument("--min-df", type=int, help="最小文档频次阈值，默认读配置 bm25.min_df")
    ap.add_argument("--top-k", type=int, default=0, help="仅输出频率最高的 Top-K（0 表示不限）")
    ap.add_argument("--forbid-single-char", action="store_true", help="过滤单字中文分词（默认读配置 bm25.forbid_single_char）")
    ap.add_argument("--allow-single-char", action="store_true", help="允许单字中文分词（与 --forbid-single-char 互斥）")
    ap.add_argument("--max-terms-per-doc", type=int, help="每文档最多保留多少 token（默认读配置 bm25.max_terms_per_doc）")

    # 输出覆盖
    ap.add_argument("--vocab-out", help="词表输出路径（默认使用 config.paths.vocab_path）")
    ap.add_argument("--stats-out-dir", help="统计输出目录（默认使用 config.paths.stats_out_dir）")

    return ap.parse_args()


def main():
    args = _parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    input_root = paths["input_root"]
    vocab_out = args.vocab_out or paths["vocab_path"]   # ✅ BM25 词表输出
    stats_out_dir = args.stats_out_dir or paths["stats_out_dir"]

    # 仅给 jieba 使用的中文用户词典（不会写）
    jieba_userdict = paths.get("jieba_userdict", "")
    if jieba_userdict and os.path.exists(jieba_userdict):
        jieba.load_userdict(jieba_userdict)

    stop_zh = _load_set(paths["stopwords_zh"])
    stop_en = _load_set(paths["stopwords_en"])

    # 配置与命令行融合
    bm25 = cfg.get("bm25", {})
    min_df = args.min_df if args.min_df is not None else int(bm25.get("min_df", 1))
    forbid_single_char_cfg = bool(bm25.get("forbid_single_char", True))
    if args.forbid_single_char:
        forbid_single_char = True
    elif args.allow_single_char:
        forbid_single_char = False
    else:
        forbid_single_char = forbid_single_char_cfg

    max_terms_per_doc = args.max_terms_per_doc if args.max_terms_per_doc else int(bm25.get("max_terms_per_doc", 4000))
    use_field = args.use_field or cfg.get("build_stats", {}).get("use_field", "原文")

    # 是否启用中/英：默认都启用，可被命令行强制关闭/开启
    enable_zh = True
    enable_en = True
    if args.enable_zh: enable_zh = True
    if args.disable_zh: enable_zh = False
    if args.enable_en: enable_en = True
    if args.disable_en: enable_en = False

    print("=== Build BM25 vocab/stats ===")
    print(f"input_root       : {input_root}")
    print(f"use_field        : {use_field}")
    print(f"enable_zh / en   : {enable_zh} / {enable_en}")
    print(f"min_df           : {min_df}")
    print(f"forbid_single_chr: {forbid_single_char}")
    print(f"max_terms_per_doc: {max_terms_per_doc}")
    print(f"vocab_out        : {vocab_out}")
    print(f"stats_out_dir    : {stats_out_dir}")
    print(f"jieba_userdict   : {jieba_userdict or '(none)'}")

    files = _iter_files(input_root)
    print(f"scan files       : {len(files)}")
    if not files:
        raise SystemExit("❌ 未发现 .json/.jsonl 文件，请检查 paths.input_root")

    # 统计 DF / N / avgdl
    df: Dict[str, int] = defaultdict(int)
    N = 0
    total_len = 0

    pbar = tqdm(files, desc="Scanning")
    for fp in pbar:
        for text in _iter_json_records(fp, use_field):
            toks: List[str] = []
            if enable_zh:
                toks.extend(_tokenize_zh(text, stop_zh, forbid_single_char))
            if enable_en:
                toks.extend(_tokenize_en(text, stop_en))

            if not toks:
                continue

            # 限制单文档 token 数
            if len(toks) > max_terms_per_doc:
                toks = toks[:max_terms_per_doc]

            N += 1
            total_len += len(toks)

            # DF 按文档去重
            for t in set(toks):
                df[t] += 1

    if N == 0:
        raise SystemExit("❌ 没有任何有效文本被分词，无法生成词表与统计。")

    avgdl = total_len / max(1, N)

    # 过滤 min_df，并排序（可选 top-k）
    vocab_tokens = [t for t, c in df.items() if c >= min_df]
    # 先按 token 排序，保证确定性；你也可以换成按 df 降序
    vocab_tokens.sort()

    if args.top_k and args.top_k > 0:
        # 取 DF 最高的前 K
        vocab_tokens = sorted(vocab_tokens, key=lambda t: df[t], reverse=True)[: args.top_k]

    # 落盘
    os.makedirs(os.path.dirname(vocab_out), exist_ok=True)
    os.makedirs(stats_out_dir, exist_ok=True)

    with open(vocab_out, "w", encoding="utf-8") as f:
        for t in vocab_tokens:
            f.write(t + "\n")

    stats_path = os.path.join(stats_out_dir, "corpus_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"N": N, "avgdl": avgdl, "df": df}, f, ensure_ascii=False)

    print(f"✅ Done. N={N}, avgdl={avgdl:.2f}, vocab_size={len(vocab_tokens)}")
    print(f"   vocab  → {vocab_out}")
    print(f"   stats  → {stats_path}")


if __name__ == "__main__":
    main()
