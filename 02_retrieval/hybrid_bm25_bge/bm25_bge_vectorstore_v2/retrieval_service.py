from typing import Dict
import numpy as np
import torch

from .bm25_vectorizer import BM25Vectorizer
from .dense_embedder import BGEDense
from .milvus_client import MilvusClient
from .text_utils import clean_text, detect_lang
from .tokenizers import Tokenizer
import json, os
from typing import Dict, List, Optional, Tuple

def _zscore(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    sd = x.std()
    # 方差过小（或 NaN/Inf）直接跳过归一化，避免全 0
    if not np.isfinite(sd) or sd < 1e-6:
        return x
    mu = x.mean()
    return (x - mu) / sd



def _orient_scores(scores: np.ndarray, metric: str) -> np.ndarray:
    """
    将得分统一成“越大越好”：
    - IP/COSINE：原样
    - L2/EUCLIDEAN：取相反数
    """
    m = (metric or "IP").upper()
    if m in ("L2", "EUCLIDEAN"):
        return -scores
    return scores


class HybridRetriever:
    def __init__(self, cfg: Dict):
        self.cfg = cfg

        # 分词与 BM25
        self.tok = Tokenizer(
            bm25_vocab_path=cfg["paths"]["vocab_path"],
            jieba_userdict_path=cfg["paths"]["jieba_userdict"],
            stopwords_zh_path=cfg["paths"]["stopwords_zh"],
            stopwords_en_path=cfg["paths"]["stopwords_en"],
            forbid_single_char=cfg["bm25"]["forbid_single_char"],
            max_terms_per_doc=cfg["bm25"]["max_terms_per_doc"],
        )
        with open(os.path.join(cfg["paths"]["stats_out_dir"], "corpus_stats.json"), "r", encoding="utf-8") as f:
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

        # 稠密编码器
        self.embedder = BGEDense(
            cfg["paths"]["model_path"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Milvus
        self.m = MilvusClient(
            cfg["milvus"]["uri"],
            cfg["milvus"].get("user", ""),
            cfg["milvus"].get("password", ""),
            cfg["milvus"].get("db_name", "default"),
            index_dense_cfg=cfg["milvus"]["index_dense"],
            index_sparse_cfg=cfg["milvus"]["index_sparse"],
        )
        self.collection = self.m.get_collection(cfg["milvus"]["collection"])

        # 超参
        self.search_dense_params = cfg["milvus"]["index_dense"]
        self.topk = int(cfg["retrieval"]["topk"])
        self.fusion_cfg = cfg["retrieval"]["fusion"]

    def search(self, query: str) -> Dict:
        q = clean_text(query)
        if not q:
            return {
                "query": "",
                "lang": "unk",
                "fusion": {"w_dense": 0.0, "w_sparse": 0.0, "norm": self.fusion_cfg.get("normalize", {})},
                "results": [],
            }

        q_lang = detect_lang(q)
        q_toks = self.tok.tokenize_mixed(q)

        # 稀疏查询向量
        q_sparse = self.vectorizer.vectorize_query(q_toks)

        # —— 稠密先算，防止未赋值使用 ——
        q_dense_vec = self.embedder.encode_batch([q])[0]            # np.ndarray (D,)
        q_dense_list = q_dense_vec.tolist()

        # 稠密检索
        dense_res = self.m.search_dense(self.collection, [q_dense_list], self.topk, self.search_dense_params)[0]

        # 稀疏检索（仅当 indices 非空）
        do_sparse = bool(q_sparse.get("indices"))
        if do_sparse:
            # 保护性清洗（通常 vectorizer 已是干净的，这里双保险）
            idx = [int(i) for i in q_sparse["indices"]]
            val = [float(v) for v in q_sparse["values"]]
            pairs = [(i, v) for i, v in zip(idx, val) if (i >= 0 and np.isfinite(v))]
            from collections import defaultdict
            acc = defaultdict(float)
            for i, v in pairs:
                acc[i] += float(v)
            idx_sorted = sorted(acc.keys())
            q_sparse = {"indices": idx_sorted, "values": [float(acc[i]) for i in idx_sorted]} if idx_sorted else {"indices": [], "values": []}

        sparse_res = self.m.search_sparse(self.collection, [q_sparse], self.topk)[0] if do_sparse and q_sparse["indices"] else []

        # ---- 取分数并归一化/对齐方向 ----
        def _scores(rs):
            return np.array([hit.distance for hit in rs], dtype=np.float32)

        s_dense = _scores(dense_res)
        s_sparse = _scores(sparse_res)

        # 方向统一（根据 dense metric；sparse 这里固定 IP）
        dense_metric = (self.search_dense_params.get("metric") or self.search_dense_params.get("metric_type") or "IP")
        s_dense = _orient_scores(s_dense, dense_metric)
        s_sparse = _orient_scores(s_sparse, "IP")

        # 可选归一化
        norm_cfg = self.fusion_cfg.get("normalize", {})
        def _maybe_norm(x: np.ndarray, how: str | None):
            if how == "zscore":
                return _zscore(x)
            return x

        s_dense_n = _maybe_norm(s_dense, norm_cfg.get("dense")) if s_dense.size else s_dense
        s_sparse_n = _maybe_norm(s_sparse, norm_cfg.get("sparse")) if s_sparse.size else s_sparse
        print(f"[DBG] sparse_hits={len(sparse_res)}")
        if s_sparse.size:
            print("[DBG] sparse raw distances (first 5):", s_sparse[:5].tolist())
            print("[DBG] sparse norm distances (first 5):", _maybe_norm(s_sparse, norm_cfg.get("sparse"))[:5].tolist())


        # 语言自适应权重
        w_dense = float(self.fusion_cfg["weights"]["dense"])
        w_sparse = float(self.fusion_cfg["weights"]["sparse"])
        if q_lang == "zh":
            w_sparse = float(self.fusion_cfg["lang_adapt"]["zh_sparse"])
            w_dense = 1.0 - w_sparse
        elif q_lang == "en":
            w_sparse = float(self.fusion_cfg["lang_adapt"]["en_sparse"])
            w_dense = 1.0 - w_sparse

        if not do_sparse:
            w_dense, w_sparse = 1.0, 0.0

        # ---- 融合（按 pk 合并）----
        pool: Dict[int, Dict] = {}

        def _push(rs, scores, key):
            for hit, sc in zip(rs, scores):
                # 统一主键类型，避免 str/int/np.int64 混淆
                try:
                    pk = int(getattr(hit, "id"))
                except Exception:
                    pk = int(getattr(hit, "pk"))  # 兜底

                if pk not in pool:
                    pool[pk] = {
                        "pk": pk,
                        "text": hit.entity.get("text"),
                        "summary": hit.entity.get("summary"),
                        "file_name": hit.entity.get("file_name"),
                        "metadata": hit.entity.get("metadata"),
                        "lang": hit.entity.get("lang"),
                        "dense": None,
                        "sparse": None,
                        "score": 0.0,
                    }
                pool[pk][key] = float(sc)

        _push(dense_res, s_dense_n, "dense")
        _push(sparse_res, s_sparse_n, "sparse")

        for item in pool.values():
            d = 0.0 if item["dense"] is None else item["dense"]
            s = 0.0 if item["sparse"] is None else item["sparse"]
            item["score"] = w_dense * d + w_sparse * s

        ranked = sorted(pool.values(), key=lambda x: x["score"], reverse=True)[: self.topk]

        # debug
        print(f"[DEBUG] query={q}")
        print(f"[DEBUG] tokens={q_toks}")
        print(f"[DEBUG] q_sparse={q_sparse if do_sparse else {'indices': [], 'values': []}}")
        print(f"[DEBUG] weights={{'dense': {w_dense}, 'sparse': {w_sparse}}}")

        return {
            "query": q,
            "lang": q_lang,
            "fusion": {"w_dense": w_dense, "w_sparse": w_sparse, "norm": norm_cfg},
            "results": [
                {
                    "pk": r["pk"],
                    "score": r["score"],
                    "dense": r["dense"] if r["dense"] is not None else 0.0,
                    "sparse": r["sparse"] if r["sparse"] is not None else 0.0,
                    "file_name": r["file_name"],
                    "text": r["text"],
                    "summary": r["summary"],
                    "metadata": r["metadata"],
                }
                for r in ranked
            ],
        }
    def search_batch(
        self,
        queries: List[str],
        topk: Optional[int] = None,
        mode: str = "hybrid",                     # "dense" / "sparse" / "hybrid"
        weights_override: Optional[Tuple[float, float]] = None,  # (w_dense, w_sparse)
        disable_lang_adapt: bool = False,         # 置 True 可保证权重不被语言自适配改写
        k_dense: Optional[int] = None,            # 稠密侧初选池（建议 >= topk，融合更稳）
        k_sparse: Optional[int] = None,           # 稀疏侧初选池（建议 >= topk）
        embed_batch_size: int = 64,               # BGE 向量化批大小（按显存调整）
        return_internal: bool = False,            # 调试：是否在结果里带上中间分数
    ) -> List[Dict]:
        """
        批量检索：输入一组 queries，输出与 queries 对齐的一组检索结果字典。
        返回元素结构与单条 search 基本一致：{"query","lang","fusion","results":[...]}。
        """
        import numpy as np
        from collections import defaultdict

        if topk is None: topk = self.topk
        k_dense = int(k_dense or topk)
        k_sparse = int(k_sparse or topk)

        # --- 预处理 ---
        Q = []
        Lang = []
        Toks = []
        for q in queries:
            q_ = clean_text(q or "")
            Q.append(q_)
            Lang.append(detect_lang(q_) if q_ else "unk")
            Toks.append(self.tok.tokenize_mixed(q_) if q_ else [])

        # 稀疏查询向量（逐条生成，但可一次性提交给 Milvus）
        Q_sparse = []
        for toks in Toks:
            qs = self.vectorizer.vectorize_query(toks)
            if qs.get("indices"):
                # 去重/清洗
                idx = [int(i) for i in qs["indices"]]
                val = [float(v) for v in qs["values"]]
                acc = defaultdict(float)
                for i, v in zip(idx, val):
                    if i >= 0 and np.isfinite(v): acc[i] += v
                ids = sorted(acc.keys())
                qs = {"indices": ids, "values": [float(acc[i]) for i in ids]} if ids else {"indices": [], "values": []}
            else:
                qs = {"indices": [], "values": []}
            Q_sparse.append(qs)

        # 稠密编码（批量）
        Q_texts = [q if q else "" for q in Q]
        Q_dense_vecs: List[np.ndarray] = []
        with torch.no_grad():
            for s in range(0, len(Q_texts), embed_batch_size):
                batch = Q_texts[s:s+embed_batch_size]
                vecs = self.embedder.encode_batch(batch)      # -> np.ndarray [B, D]
                Q_dense_vecs.extend([v for v in vecs])

        # --- Milvus 批量检索 ---
        # 你的 MilvusClient.search_* 已支持“列表输入返回列表输出”（你之前传 [vec] 再取 [0]）
        dense_lists = self.m.search_dense(
            self.collection, [v.tolist() for v in Q_dense_vecs], k_dense, self.search_dense_params
        )
        # 稀疏：只要有任何 query 的 indices 非空，就整体批量搜；否则统一为空
        if any(qs["indices"] for qs in Q_sparse):
            sparse_lists = self.m.search_sparse(
                self.collection, Q_sparse, k_sparse
            )
        else:
            sparse_lists = [[] for _ in Q_sparse]

        # --- 归一化与融合 ---
        def _scores(rs): return np.array([hit.distance for hit in rs], dtype=np.float32)
        dense_metric = (self.search_dense_params.get("metric") or
                        self.search_dense_params.get("metric_type") or "IP")

        norm_cfg = self.fusion_cfg.get("normalize", {})
        def _maybe_norm(x: np.ndarray, how: Optional[str]):
            if x.size == 0: return x
            if how == "zscore": return _zscore(x)
            return x

        results_all: List[Dict] = []
        for i, q in enumerate(Q):
            d_res = dense_lists[i] if i < len(dense_lists) else []
            s_res = sparse_lists[i] if i < len(sparse_lists) else []

            s_dense = _orient_scores(_scores(d_res), dense_metric)
            s_sparse = _orient_scores(_scores(s_res), "IP")

            s_dense_n = _maybe_norm(s_dense, norm_cfg.get("dense"))
            s_sparse_n = _maybe_norm(s_sparse, norm_cfg.get("sparse"))

            # 权重设定
            if mode == "dense":
                w_d, w_s = 1.0, 0.0
            elif mode == "sparse":
                w_d, w_s = 0.0, 1.0
            else:
                w_d = float(self.fusion_cfg["weights"]["dense"])
                w_s = float(self.fusion_cfg["weights"]["sparse"])

            if weights_override is not None:
                w_d, w_s = float(weights_override[0]), float(weights_override[1])

            if not disable_lang_adapt:
                if   Lang[i] == "zh": w_s = float(self.fusion_cfg["lang_adapt"]["zh_sparse"])
                elif Lang[i] == "en": w_s = float(self.fusion_cfg["lang_adapt"]["en_sparse"])
                if mode == "hybrid" and weights_override is None:
                    w_d = 1.0 - w_s

            # 如无稀疏候选，强制稠密-only
            if not Q_sparse[i]["indices"]:
                w_d, w_s = 1.0, 0.0

            # 融合：按 pk 合并
            pool: Dict[int, Dict] = {}
            def _push(rs, scores, key):
                for hit, sc in zip(rs, scores):
                    try: pk = int(getattr(hit, "id"))
                    except Exception: pk = int(getattr(hit, "pk"))
                    if pk not in pool:
                        pool[pk] = {
                            "pk": pk,
                            "text": hit.entity.get("text"),
                            "summary": hit.entity.get("summary"),
                            "file_name": hit.entity.get("file_name"),
                            "metadata": hit.entity.get("metadata"),
                            "lang": hit.entity.get("lang"),
                            "dense": None, "sparse": None, "score": 0.0
                        }
                    pool[pk][key] = float(sc)
            # ✅ 不同 mode 只允许对应候选进入 pool，避免 baseline 被另一侧污染
            if mode in ("dense", "hybrid"):
                _push(d_res, s_dense_n, "dense")
            if mode in ("sparse", "hybrid"):
                _push(s_res, s_sparse_n, "sparse")
            

            for item in pool.values():
                d = 0.0 if item["dense"] is None else item["dense"]
                s = 0.0 if item["sparse"] is None else item["sparse"]
                item["score"] = w_d * d + w_s * s

            ranked = sorted(pool.values(), key=lambda x: x["score"], reverse=True)[: topk]

            out = {
                "query": q,
                "lang": Lang[i],
                "fusion": {"w_dense": w_d, "w_sparse": w_s, "norm": norm_cfg},
                "results": [
                    {
                        "pk": r["pk"],
                        "score": r["score"],
                        "dense": r["dense"] if r["dense"] is not None else 0.0,
                        "sparse": r["sparse"] if r["sparse"] is not None else 0.0,
                        "file_name": r["file_name"],
                        "text": r["text"],
                        "summary": r["summary"],
                        "metadata": r["metadata"],
                    } for r in ranked
                ],
            }
            if return_internal:
                out["_dense_raw"]  = s_dense.tolist()
                out["_sparse_raw"] = s_sparse.tolist()
            results_all.append(out)

        return results_all
