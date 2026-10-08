import os
import re
from typing import List, Set
import jieba

# 英文分词正则
_EN_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9’'_\-]+\b")


class Tokenizer:
    def __init__(
        self,
        bm25_vocab_path: str,       # BM25 稀疏向量词表（中英混合）
        jieba_userdict_path: str,   # jieba 中文用户词典（医学专用）
        stopwords_zh_path: str,
        stopwords_en_path: str,
        forbid_single_char: bool = True,
        max_terms_per_doc: int = 4000,
    ):
        # 加载 BM25 词表（中英混合）
        self.vocab = self._load_vocab(bm25_vocab_path)
        self.token2id = {w: i for i, w in enumerate(self.vocab)}

        # 加载停词表
        self.stop_zh = self._load_set(stopwords_zh_path)
        self.stop_en = self._load_set(stopwords_en_path)

        # 参数
        self.forbid_single_char = forbid_single_char
        self.max_terms = max_terms_per_doc

        # 加载 jieba 用户词典（医学专用）
        if jieba_userdict_path and os.path.exists(jieba_userdict_path):
            jieba.load_userdict(jieba_userdict_path)

    @staticmethod
    def _load_vocab(path: str) -> List[str]:
        if not path or not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [w.strip() for w in f if w.strip()]

    @staticmethod
    def _load_set(path: str) -> Set[str]:
        if not path or not os.path.exists(path):
            return set()
        with open(path, "r", encoding="utf-8") as f:
            return {w.strip() for w in f if w.strip()}

    def tokenize_zh(self, text: str) -> List[str]:
        """中文分词"""
        words = list(jieba.cut(text, cut_all=False, HMM=True))
        toks: List[str] = []
        for w in words:
            w = w.strip()
            if not w:
                continue
            if self.forbid_single_char and (len(w) == 1) and (w not in self.token2id):
                continue
            if w in self.stop_zh:
                continue
            toks.append(w)
        return toks

    def tokenize_en(self, text: str) -> List[str]:
        """英文分词"""
        text = text.lower()
        words = _EN_TOKEN_RE.findall(text)
        return [w for w in words if w not in self.stop_en]

    def tokenize_mixed(self, text: str) -> List[str]:
        """中英文混合分词"""
        zh = self.tokenize_zh(text)
        en = self.tokenize_en(text)
        toks = zh + en
        if len(toks) > self.max_terms:
            toks = toks[: self.max_terms]
        return toks

    def to_ids(self, tokens: List[str]) -> List[int]:
        """token 转 ID，依赖 BM25 词表"""
        return [self.token2id[t] for t in tokens if t in self.token2id]
