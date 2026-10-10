#!/usr/bin/env python3
"""留一检索封装：v0.3 数据构造的检索环境（P0 检索环境修正）。

修正旧环境的两个结构性问题（详见 07_experiments/v0.3_rag_agent_training/ 两份文档）：
1. 题图自命中泄露：检索库包含题目来源样本，题图的 caption 直接带着答案被召回
   （旧 train3200 中 3200/3200 题自命中、2143 题排 top1）。底层 RetrievalRequest
   不支持排除参数，因此在应用层"过量召回 → 排除过滤 → 配额截断"。
2. 文本路被挤掉：文本(BGE-M3)与图像(视觉塔)候选的原始分数量纲不同，旧代码
   全局混排取 top-k，图像 0.9–1.0 的分数占满全部候选，文本路 0 条。修正为
   两路各自配额（默认文本 8 / 图像 4）后再合并。

排除键（与 v0.5/harness 候选字段对齐）：
- 同书：候选 doc_id ∈ exclude_doc_ids
- 自命中：候选 sample_id ∈ exclude_sample_ids，或候选 image_path ∈ exclude_image_paths
字段为空的候选不参与该项判断（不会误杀）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def filter_leakage(
    hits: List[Dict[str, Any]],
    exclude_sample_ids: Sequence[str] = (),
    exclude_image_paths: Sequence[str] = (),
    exclude_doc_ids: Sequence[str] = (),
) -> Tuple[List[Dict[str, Any]], int, int]:
    """按排除键过滤候选，保持原顺序。返回 (kept, n_same_book, n_self)。"""
    sid = {str(s) for s in exclude_sample_ids if s}
    img = {str(p) for p in exclude_image_paths if p}
    doc = {str(d) for d in exclude_doc_ids if d}
    kept: List[Dict[str, Any]] = []
    n_book = n_self = 0
    for h in hits:
        # 先判自命中再判同书：题图自身必属于同书，若先判同书，自命中统计会被吸收进 drop_same_book
        sample_id = str(h.get("sample_id") or "")
        image_path = str(h.get("image_path") or "")
        if (sample_id and sample_id in sid) or (image_path and image_path in img):
            n_self += 1
            continue
        doc_id = str(h.get("doc_id") or "")
        if doc_id and doc_id in doc:
            n_book += 1
            continue
        kept.append(h)
    return kept, n_book, n_self


def load_first_stage_retriever(retrieval_cfg: Dict[str, Any]):
    """动态加载归档 agentic 代码的 FirstStageRetriever（同 10_harness 方式）。

    FirstStageRetriever 期望完整 config 形状 {"paths": {...}, "retrieval": {...}}。
    """
    code_dir = Path(retrieval_cfg["code_dir"])
    adapter_path = code_dir / "retrieval_adapter.py"
    if not adapter_path.exists():
        raise FileNotFoundError(f"retrieval_adapter.py not found: {adapter_path}")
    mod_name = "_v03_retrieval_adapter"
    spec = importlib.util.spec_from_file_location(mod_name, adapter_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    rcfg = dict(retrieval_cfg)
    rcfg.setdefault("first_stage_text_k", rcfg.get("text_k", 20))
    rcfg.setdefault("first_stage_image_k", rcfg.get("image_k", 20))
    full_cfg = {
        "paths": {
            "multimodal_search_dir": rcfg["multimodal_search_dir"],
            "main_db_path": rcfg["main_db_path"],
        },
        "retrieval": rcfg,
    }
    return module.FirstStageRetriever(config=full_cfg)  # type: ignore[return-value]


class LeaveOneOutRetriever:
    """过量召回 → 留一/同书过滤 → 两路配额合并的检索封装。

    retriever 可注入（单测用 Mock）；默认按 retrieval_cfg 动态加载真实检索器。
    """

    def __init__(
        self,
        retrieval_cfg: Dict[str, Any],
        retriever: Optional[Any] = None,
        *,
        text_k: int = 8,
        image_k: int = 4,
        overfetch_k: int = 40,
    ):
        self._retriever = retriever if retriever is not None else load_first_stage_retriever(retrieval_cfg)
        self.text_k = int(text_k)
        self.image_k = int(image_k)
        self.overfetch_k = int(overfetch_k)

    def retrieve(
        self,
        query_text: str,
        query_image_path: str = "",
        *,
        exclude_sample_ids: Sequence[str] = (),
        exclude_image_paths: Sequence[str] = (),
        exclude_doc_ids: Sequence[str] = (),
        text_k: Optional[int] = None,
        image_k: Optional[int] = None,
        overfetch_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """双路检索并过滤。

        返回 {"text": [...], "image": [...], "combined": [...], "stats": {...}}。
        combined 为 text + image 拼接（保持路内按分数排序），stats 记录过滤明细。
        """
        tk = int(text_k if text_k is not None else self.text_k)
        ik = int(image_k if image_k is not None else self.image_k)
        over = int(overfetch_k if overfetch_k is not None else self.overfetch_k)

        text_hits: List[Dict[str, Any]] = []
        if query_text and str(query_text).strip():
            text_hits = self._retriever.search_text(str(query_text).strip(), k=over)
        image_hits: List[Dict[str, Any]] = []
        if query_image_path and str(query_image_path).strip():
            # 路径存在性由底层 search_image 判断（不存在时返回空），此处不重复检查
            image_hits = self._retriever.search_image(str(query_image_path).strip(), k=over)

        text_kept, t_book, t_self = filter_leakage(text_hits, exclude_sample_ids, exclude_image_paths, exclude_doc_ids)
        image_kept, i_book, i_self = filter_leakage(image_hits, exclude_sample_ids, exclude_image_paths, exclude_doc_ids)

        text_final = text_kept[:tk]
        image_final = image_kept[:ik]
        stats = {
            "overfetch_k": over,
            "text": {"overfetch": len(text_hits), "drop_same_book": t_book, "drop_self": t_self, "kept": len(text_final)},
            "image": {"overfetch": len(image_hits), "drop_same_book": i_book, "drop_self": i_self, "kept": len(image_final)},
            "quota": {"text_k": tk, "image_k": ik},
        }
        return {"text": text_final, "image": image_final, "combined": text_final + image_final, "stats": stats}

    def close(self) -> None:
        close = getattr(self._retriever, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "LeaveOneOutRetriever":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
