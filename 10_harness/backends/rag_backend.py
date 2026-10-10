#!/usr/bin/env python3
"""RAG pipeline backend: dual-path retrieval + internal evidence selection.

This is the INSIDE of the single rag_search tool. The former v0.5 evidence-selection
agent is embedded here (AgentEvidenceFilter): first-stage hits never reach the
central brain — it only receives the final selected passages.

Filters (injectable):
  - AgentEvidenceFilter: LLM keep/drop over numbered hits (v0.5 agent protocol),
    falls back to TopKFilter on any error.
  - TopKFilter: score-ordered truncation (deterministic baseline / mock / ablation).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from backends.retrieval_backend import RetrievalBackend, load_first_stage_retriever


# --------------------------------------------------------------------- filters
class EvidenceFilter(Protocol):
    def select(self, hits: List[Dict[str, Any]], question: str, options: Dict[str, Any]) -> List[Dict[str, Any]]: ...


class TopKFilter:
    """Deterministic baseline: keep the top-k hits by retrieval score."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def select(self, hits: List[Dict[str, Any]], question: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
        return sorted(hits, key=lambda h: h.get("score", 0.0), reverse=True)[: self.top_k]


class AgentEvidenceFilter:
    """v0.5 evidence-selection agent, embedded inside the RAG tool.

    One LLM call: numbered hit list -> {"keep": [nums]}; invalid output or errors
    fall back to TopKFilter so the pipeline never breaks.
    """

    def __init__(self, llm, fallback: TopKFilter, max_keep: int = 5, text_chars: int = 200):
        self.llm = llm  # object with decide(messages) -> dict (backends.llm_backend)
        self.fallback = fallback
        self.max_keep = max_keep
        self.text_chars = text_chars

    def select(self, hits: List[Dict[str, Any]], question: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not hits:
            return []
        try:
            keep = self._ask(hits, question, options)
            if keep:
                by_no = {i + 1: h for i, h in enumerate(hits)}
                return [by_no[n] for n in keep if n in by_no]
        except Exception:
            pass
        return self.fallback.select(hits, question, options)

    def _ask(self, hits: List[Dict[str, Any]], question: str, options: Dict[str, Any]) -> List[int]:
        opt_s = ", ".join(f"{k}. {v}" for k, v in options.items())
        lines = []
        for i, h in enumerate(hits, 1):
            text = (h.get("text", "") or "").strip().replace("\n", " ")[: self.text_chars]
            lines.append(f"[{i}] {text}")
        system = (
            "你是多选题的证据筛选器。给定问题和编号的候选文段，返回可能改变答案的"
            "文段编号。只回复一个 JSON 对象：{\"keep\": [int, ...]}"
        )
        user = f"问题: {question}\n选项: {opt_s}\n\n候选文段:\n" + "\n".join(lines)
        out = self.llm.decide([{"role": "system", "content": system}, {"role": "user", "content": user}])
        raw = out.get("keep", [])
        if not isinstance(raw, list):
            return []
        return [int(n) for n in raw[: self.max_keep] if isinstance(n, (int, float))]


def build_evidence_filter(cfg: Dict[str, Any]) -> EvidenceFilter:
    kind = cfg.get("backend", "topk")
    fallback = TopKFilter(int(cfg.get("max_keep", 5)))
    if kind == "topk":
        return fallback
    if kind == "agent":
        def build_llm(c):
            if c.get("backend", "openai") == "transformers":
                from backends.transformers_backend import TransformersChat

                return TransformersChat(c)
            from backends.llm_backend import OpenAIChatBrain

            return OpenAIChatBrain(c)

        llm = build_llm(cfg["llm"])
        return AgentEvidenceFilter(llm=llm, fallback=fallback, max_keep=int(cfg.get("max_keep", 5)))
    raise ValueError(f"unknown evidence filter backend: {kind}")


# --------------------------------------------------------------------- pipeline
class RagPipelineBackend(Protocol):
    def search(
        self,
        query: str,
        image_path: str,
        question: str,
        options: Dict[str, Any],
        *,
        exclude_sample_ids: tuple = (),
        exclude_doc_ids: tuple = (),
        exclude_image_paths: tuple = (),
    ) -> List[Dict[str, Any]]: ...
    def close(self) -> None: ...


class FullRagPipeline:
    """Dual-path retrieval -> leakage filter -> per-path quota -> evidence filter.

    v0.3 P0 检索环境修正（同 11_v03_training/data_construction/leave_one_out_retriever.py）：
    1. 留一去泄露：排除题图自身样本（sample_id / image_path）与同书 doc_id；
    2. 两路按配额合并：文本(BGE-M3)与图像(视觉塔)分数量纲不同，旧全局混排取
       top-k 会把文本路挤掉（0.9–1.0 的图像分占满候选）；配额默认文本 8 / 图像 4，
       filter 拿到的已是配额后列表（TopKFilter 回落也因此不再受混排影响）。
    """

    def __init__(self, config: Dict[str, Any]):
        rcfg = config["retrieval"]
        self.retrieval: RetrievalBackend = load_first_stage_retriever(config)
        self.overfetch_k = int(rcfg.get("overfetch_k", 40))
        self.text_quota = int(rcfg.get("text_quota", 8))
        self.image_quota = int(rcfg.get("image_quota", 4))
        self.filter = build_evidence_filter(config.get("rag", {}).get("evidence_filter", {}))

    @staticmethod
    def _filter_excluded(
        hits: List[Dict[str, Any]],
        exclude_sample_ids: tuple,
        exclude_image_paths: tuple,
        exclude_doc_ids: tuple,
    ) -> List[Dict[str, Any]]:
        """同书/自命中排除；字段为空的候选不参与该项判断（不会误杀）。"""
        sid = {str(s) for s in exclude_sample_ids if s}
        img = {str(p) for p in exclude_image_paths if p}
        doc = {str(d) for d in exclude_doc_ids if d}
        kept: List[Dict[str, Any]] = []
        for h in hits:
            # 先判自命中再判同书（题图自身必属于同书，顺序与 11_v03 filter_leakage 一致）
            sample_id = str(h.get("sample_id") or "")
            image_path = str(h.get("image_path") or "")
            if (sample_id and sample_id in sid) or (image_path and image_path in img):
                continue
            doc_id = str(h.get("doc_id") or "")
            if doc_id and doc_id in doc:
                continue
            kept.append(h)
        return kept

    def search(
        self,
        query: str,
        image_path: str,
        question: str,
        options: Dict[str, Any],
        *,
        exclude_sample_ids: tuple = (),
        exclude_doc_ids: tuple = (),
        exclude_image_paths: tuple = (),
    ) -> List[Dict[str, Any]]:
        text_hits: List[Dict[str, Any]] = []
        if query and str(query).strip():
            text_hits = self.retrieval.search_text(str(query).strip(), k=self.overfetch_k)
            text_hits = self._filter_excluded(text_hits, exclude_sample_ids, exclude_image_paths, exclude_doc_ids)
            text_hits = text_hits[: self.text_quota]
        image_hits: List[Dict[str, Any]] = []
        if image_path:
            image_hits = self.retrieval.search_image(image_path, k=self.overfetch_k)
            img_excl = tuple(exclude_image_paths) + (image_path,)  # 题图本身必排除
            image_hits = self._filter_excluded(image_hits, exclude_sample_ids, img_excl, exclude_doc_ids)
            image_hits = image_hits[: self.image_quota]
        return self.filter.select(text_hits + image_hits, question, options)

    def close(self) -> None:
        self.retrieval.close()


class MockRagPipeline:
    """Deterministic final passages for smoke tests — simulates the whole pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.calls = 0

    def search(self, query: str, image_path: str, question: str, options: Dict[str, Any], **_kwargs) -> List[Dict[str, Any]]:
        self.calls += 1
        if self.calls == 1:
            return [self._p(1, "img"), self._p(2, "txt")]
        return [self._p(3, "txt")]  # second, rephrased query yields one fresh passage

    @staticmethod
    def _p(i: int, origin: str) -> Dict[str, Any]:
        return {
            "origin": origin,
            "source": origin,
            "score": 1.0 - i * 0.01,
            "text": f"[mock final passage {i}] 胃镜下可见胃窦部黏膜光滑，蠕动正常。",
            "doc_id": f"mockdoc{origin}",
            "doc_name": f"mock_doc_{origin}",
            "page_idx": i,
            "block_id": i * 10,
            "sample_id": f"mock_{origin}_{i:04d}",
            "group_id": f"grp_mock_{origin}_{i:04d}",
            "image_path": "mock://image.jpg" if origin == "img" else "",
            "image_id": "",
        }

    def close(self) -> None: ...


def build_rag_pipeline(config: Dict[str, Any], mock: bool = False) -> RagPipelineBackend:
    if mock:
        return MockRagPipeline(config)
    return FullRagPipeline(config)
