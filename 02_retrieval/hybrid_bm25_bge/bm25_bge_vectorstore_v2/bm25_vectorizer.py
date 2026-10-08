import math
from collections import Counter
from typing import Dict, List, Tuple

class BM25Vectorizer:
    def __init__(self, df: Dict[str, int], N: int, avgdl: float, token2id: Dict[str, int], k1=1.2, b=0.75, idf_clip_min=0.0, min_df: int = 1):
        self.df = df
        self.N = N
        self.avgdl = max(1e-9, avgdl)
        self.k1 = k1
        self.b = b
        self.idf_clip_min = idf_clip_min
        self.token2id = token2id
        self.min_df = min_df

    def _idf(self, t: str) -> float:
        df_t = self.df.get(t, 0)
        if df_t < self.min_df:
            return None
        val = math.log((self.N - df_t + 0.5) / (df_t + 0.5))
        if val < self.idf_clip_min:
            val = self.idf_clip_min
        return val

    def vectorize_doc(self, tokens: List[str]) -> Dict[str, List]:
        # 文档向量：BM25 权重
        tf = Counter(tokens)
        dl = sum(tf.values())
        indices: List[int] = []
        values: List[float] = []
        for t, f in tf.items():
            if t not in self.token2id:
                continue
            idf = self._idf(t)
            if idf is None:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            w = idf * (f * (self.k1 + 1)) / denom
            indices.append(self.token2id[t])
            values.append(float(w))
        return {"indices": indices, "values": values}

    def vectorize_query(self, tokens: List[str]) -> Dict[str, List]:
        # 查询向量：常用近似——仅用 idf 作为权重；也可改为 tf-idf
        seen = set()
        indices: List[int] = []
        values: List[float] = []
        for t in tokens:
            if t in seen or t not in self.token2id:
                continue
            seen.add(t)
            idf = self._idf(t)
            if idf is None:
                continue
            indices.append(self.token2id[t])
            values.append(float(idf))
        return {"indices": indices, "values": values}
