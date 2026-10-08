import json
import os
from glob import glob
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
import torch

from .bm25_vectorizer import BM25Vectorizer
from .dense_embedder import BGEDense
from .milvus_client import MilvusClient
from .text_utils import clean_text, detect_lang, iter_json_records, sha1_hex
from .tokenizers import Tokenizer

def _to_sparse_map(s):
    # 把 {"indices": [...], "values": [...]} 转成 {idx: val}
    if isinstance(s, dict) and "indices" in s and "values" in s:
        return {int(i): float(v) for i, v in zip(s["indices"], s["values"]) if v}
    return s or {}

def _iter_files(root: str) -> List[str]:
    pats = ["**/*.json", "**/*.jsonl"]
    files: List[str] = []
    for p in pats:
        files.extend(glob(os.path.join(root, p), recursive=True))
    return files


def _validate_sparse_dict(sd: Dict) -> Tuple[bool, Dict]:
    """
    对稀疏向量做强校验与规整：
    - indices: int, 非负；values: float, 有限且非 0
    - 合并重复 index
    - indices 严格升序
    返回 (ok, cleaned_dict)
    """
    if not sd or "indices" not in sd or "values" not in sd:
        return False, {"indices": [], "values": []}

    try:
        idx = [int(i) for i in sd["indices"]]
        val = [float(v) for v in sd["values"]]
    except Exception:
        return False, {"indices": [], "values": []}

    if len(idx) != len(val) or len(idx) == 0:
        return False, {"indices": [], "values": []}

    # 过滤：非负、有限、非零
    pairs = [(i, v) for i, v in zip(idx, val) if (i >= 0 and np.isfinite(v) and v != 0.0)]
    if not pairs:
        return False, {"indices": [], "values": []}

    # 合并重复 index
    from collections import defaultdict
    acc = defaultdict(float)
    for i, v in pairs:
        acc[i] += float(v)

    # 升序 & 严格递增
    idx_sorted = sorted(acc.keys())
    vals_sorted = [float(acc[i]) for i in idx_sorted]
    if len(idx_sorted) == 0:
        return False, {"indices": [], "values": []}
    # 防守：严格递增（已去重，这里恒为真；保留校验）
    for a, b in zip(idx_sorted, idx_sorted[1:]):
        if not (a < b):
            return False, {"indices": [], "values": []}

    return True, {"indices": idx_sorted, "values": vals_sorted}


class Inserter:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.tok = Tokenizer(
            bm25_vocab_path=cfg["paths"]["vocab_path"],
            jieba_userdict_path=cfg["paths"]["jieba_userdict"],
            stopwords_zh_path=cfg["paths"]["stopwords_zh"],
            stopwords_en_path=cfg["paths"]["stopwords_en"],
            forbid_single_char=cfg["bm25"]["forbid_single_char"],
            max_terms_per_doc=cfg["bm25"]["max_terms_per_doc"],
        )

        # 载入 BM25 统计
        stats_path = os.path.join(cfg["paths"]["stats_out_dir"], "corpus_stats.json")
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        self.vectorizer = BM25Vectorizer(
            df=stats["df"],
            N=stats["N"],
            avgdl=stats["avgdl"],
            token2id=self.tok.token2id,
            k1=cfg["bm25"]["k1"],
            b=cfg["bm25"]["b"],
            idf_clip_min=cfg["bm25"]["idf_clip_min"],
            min_df=cfg["bm25"]["min_df"],
        )

        self.embedder = BGEDense(
            cfg["paths"]["model_path"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        self.m = MilvusClient(
            cfg["milvus"]["uri"],
            cfg["milvus"].get("user", ""),
            cfg["milvus"].get("password", ""),
            cfg["milvus"].get("db_name", "default"),
            index_dense_cfg=cfg["milvus"]["index_dense"],
            index_sparse_cfg=cfg["milvus"]["index_sparse"],
        )

        self.collection = self.m.ensure_collection(
            cfg["milvus"]["collection"],
            cfg["milvus"]["dims_dense"],
            cfg["milvus"]["max_text_len"],
            cfg["milvus"]["max_summary_len"],
            cfg["milvus"]["partitions"],
        )

        self.batch_size = int(cfg["insert"]["batch_size"])
        self.flush_interval = int(cfg["insert"]["flush_interval_batches"])
        self.checkpoint_dir = cfg["insert"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.audit_log = bool(cfg["insert"].get("audit_log", True))

    def _checkpoint_path(self, file_name: str) -> str:
        base = os.path.basename(file_name)
        return os.path.join(self.checkpoint_dir, f"{base}.checkpoint.jsonl")

    def _failed_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "failed_batches.jsonl")

    def _append_failed(self, rec: Dict):
        with open(self._failed_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _append_checkpoint(self, rec: Dict, ckpt_path: str):
        with open(ckpt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _prep_row(self, rec: Dict) -> Dict | None:
        text = clean_text(str(rec.get("原文", "")))
        if not text:
            return None

        summary = clean_text(str(rec.get("摘要", "")))
        if not summary:
            summary = text[: self.cfg["milvus"]["max_summary_len"]]

        file_name = str(rec.get("来源文件", ""))
        lang = detect_lang(text)

        # BM25 稀疏
        toks = self.tok.tokenize_mixed(text)
        sparse_raw = self.vectorizer.vectorize_doc(toks)
        ok, sparse = _validate_sparse_dict(sparse_raw)
        if not ok:
            return None
        # 转成 Milvus 需要的 {idx: val} 稀疏映射（你也可以直接用你上面的 _to_sparse_map）
        sparse_map = {int(i): float(v) for i, v in zip(sparse["indices"], sparse["values"]) if v}
        if not sparse_map:
            return None

        # 暂存原文用于批量 dense 编码
        row = {
            "text_hash": sha1_hex(text),
            "file_name": file_name,
            "lang": lang,
            "text": text[: self.cfg["milvus"]["max_text_len"]],
            "summary": summary[: self.cfg["milvus"]["max_summary_len"]],
            "metadata": {"source": file_name},
            "sparse_vector": sparse_map,
            "_dense_text": text,
        }
        return row

    def insert_dir(self, input_root: str):
        files = _iter_files(input_root)
        batch_rows: List[Dict] = []
        batch_dense_texts: List[str] = []
        inserted = 0
        flush_countdown = self.flush_interval

        pbar = tqdm(files, desc="Files")
        for fp in pbar:
            ckpt_path = self._checkpoint_path(fp)
            for rec in iter_json_records(fp):
                row = self._prep_row(rec)
                if not row:
                    continue
                batch_rows.append(row)
                batch_dense_texts.append(row.pop("_dense_text"))

                if len(batch_rows) >= self.batch_size:
                    try:
                        self._flush_batch(batch_rows, batch_dense_texts)
                        inserted += len(batch_rows)
                        self._append_checkpoint({"file": fp, "inserted": inserted}, ckpt_path)
                    except Exception as e:
                        # 记录整批失败（会触发二分定位）
                        self._append_failed({"file": fp, "error": str(e), "stage": "batch"})
                        self._recover_and_insert(batch_rows)
                    finally:
                        batch_rows, batch_dense_texts = [], []
                        flush_countdown -= 1
                        if flush_countdown <= 0:
                            self.collection.flush()
                            flush_countdown = self.flush_interval

            # 文件结束也做一次 flush
            if batch_rows:
                try:
                    self._flush_batch(batch_rows, batch_dense_texts)
                    inserted += len(batch_rows)
                    self._append_checkpoint({"file": fp, "inserted": inserted}, ckpt_path)
                except Exception as e:
                    self._append_failed({"file": fp, "error": str(e), "stage": "tail"})
                    self._recover_and_insert(batch_rows)
                finally:
                    batch_rows, batch_dense_texts = [], []

        # 尾部 flush
        if flush_countdown != self.flush_interval:
            self.collection.flush()

    def _flush_batch(self, batch_rows: List[Dict], batch_dense_texts: List[str]):
        # 去重：按 text_hash
        hashes = [r["text_hash"] for r in batch_rows]
        existed = self.m.query_existing_by_hash(self.collection, hashes)

        # 同步掩码，避免错位
        mask = [r["text_hash"] not in existed for r in batch_rows]
        rows = [r for r, keep in zip(batch_rows, mask) if keep]
        dense_texts = [t for t, keep in zip(batch_dense_texts, mask) if keep]
        if not rows:
            return

        # 稠密编码
        dense_vecs = self.embedder.encode_batch(dense_texts)
        for r, v in zip(rows, dense_vecs):
            r["dense_vector"] = v.tolist()

        # 分区路由（Lite 下无分区）
        by_part: Dict[str, List[Dict]] = {"zh": [], "en": [], "mix": []}
        for r in rows:
            by_part.get(r.get("lang", "mix"), by_part["mix"]).append(r)

        lite_mode = self.m._is_lite()
        for part, rows_part in by_part.items():
            if not rows_part:
                continue
            if lite_mode:
                self.m.insert_rows(self.collection, rows_part)
            else:
                self.m.insert_rows(self.collection, rows_part, partition=part)

    # ---------- 关键：批量插入失败后的二分排错，剔除脏样本 ----------
    def _recover_and_insert(self, rows: List[Dict]):
        """
        针对 ParamError: invalid input for sparse float vector
        用二分搜索定位坏样本，坏样本写入 failed 日志，其余继续入库
        """
        if not rows:
            return

        def try_insert(sub: List[Dict]) -> bool:
            try:
                # 简化：不分区，直接插入（Lite 下与生产等价）
                self.m.insert_rows(self.collection, sub)
                return True
            except Exception as e:
                # 只要是 ParamError/格式问题都算失败，继续分割
                return False

        def bisect_and_insert(sub: List[Dict]):
            if not sub:
                return
            if len(sub) == 1:
                # 单条仍失败，记录并跳过
                bad = sub[0]
                self._append_failed({
                    "stage": "bisect",
                    "text_hash": bad.get("text_hash"),
                    "file_name": bad.get("file_name"),
                    "reason": "invalid sparse vector or other param error"
                })
                return
            mid = len(sub) // 2
            left, right = sub[:mid], sub[mid:]
            if not try_insert(left):
                bisect_and_insert(left)
            if not try_insert(right):
                bisect_and_insert(right)

        # 先直接尝试整批插入，若失败才二分
        if not try_insert(rows):
            bisect_and_insert(rows)
