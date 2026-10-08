import json
import math
import os
from collections import Counter
from glob import glob
from typing import Dict, Iterable, List, Tuple
from tqdm import tqdm

from .text_utils import iter_json_records, clean_text
from .tokenizers import Tokenizer

class CorpusStats:
    def __init__(self, N: int, avgdl: float, df: Dict[str, int]):
        self.N = N
        self.avgdl = avgdl
        self.df = df

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "corpus_stats.json"), "w", encoding="utf-8") as f:
            json.dump({"N": self.N, "avgdl": self.avgdl, "df": self.df}, f, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> "CorpusStats":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return CorpusStats(obj["N"], obj["avgdl"], obj["df"])


def _iter_files(root: str) -> List[str]:
    pats = ["**/*.json", "**/*.jsonl"]
    files: List[str] = []
    for p in pats:
        files.extend(glob(os.path.join(root, p), recursive=True))
    return files


def build_corpus_stats(
    input_root: str,
    tokenizer: Tokenizer,
    shard_docs: int = 200_000,
    out_dir: str = "./stats_v2",
) -> CorpusStats:
    os.makedirs(out_dir, exist_ok=True)
    files = _iter_files(input_root)
    N = 0
    sum_len = 0
    df_counter = Counter()
    shard_idx = 0
    shard_paths: List[str] = []

    for fp in tqdm(files, desc="Scanning files"):
        for rec in iter_json_records(fp):
            text = clean_text(str(rec.get("原文", "")))
            if not text:
                continue
            toks = tokenizer.tokenize_mixed(text)
            if not toks:
                continue
            N += 1
            sum_len += len(toks)
            df_counter.update(set(toks))  # 文档频次，用 set 去重

            if N % shard_docs == 0:
                shard_path = os.path.join(out_dir, f"df_shard_{shard_idx}.json")
                with open(shard_path, "w", encoding="utf-8") as f:
                    json.dump(dict(df_counter), f, ensure_ascii=False)
                shard_paths.append(shard_path)
                df_counter.clear()
                shard_idx += 1

    # dump tail shard
    if df_counter:
        shard_path = os.path.join(out_dir, f"df_shard_{shard_idx}.json")
        with open(shard_path, "w", encoding="utf-8") as f:
            json.dump(dict(df_counter), f, ensure_ascii=False)
        shard_paths.append(shard_path)

    # merge shards
    merged = Counter()
    for sp in tqdm(shard_paths, desc="Merging shards"):
        with open(sp, "r", encoding="utf-8") as f:
            part = json.load(f)
        merged.update(part)

    avgdl = (sum_len / N) if N > 0 else 0.0
    stats = CorpusStats(N=N, avgdl=avgdl, df=dict(merged))
    stats.save(out_dir)
    return stats