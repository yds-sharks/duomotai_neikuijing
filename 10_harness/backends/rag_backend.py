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
            "You select evidence for a medical MCQ. Given the question and numbered "
            "candidate passages, return the numbers of passages that could change the "
            "answer. Reply EXACTLY one JSON object: {\"keep\": [int, ...]}"
        )
        user = f"Question: {question}\nOptions: {opt_s}\n\nCandidates:\n" + "\n".join(lines)
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
        from backends.llm_backend import OpenAIChatBrain

        llm = OpenAIChatBrain(cfg["llm"])
        return AgentEvidenceFilter(llm=llm, fallback=fallback, max_keep=int(cfg.get("max_keep", 5)))
    raise ValueError(f"unknown evidence filter backend: {kind}")


# --------------------------------------------------------------------- pipeline
class RagPipelineBackend(Protocol):
    def search(self, query: str, image_path: str, question: str, options: Dict[str, Any]) -> List[Dict[str, Any]]: ...
    def close(self) -> None: ...


class FullRagPipeline:
    """Dual-path first-stage retrieval -> internal evidence filter -> final passages."""

    def __init__(self, config: Dict[str, Any]):
        rcfg = config["retrieval"]
        self.retrieval: RetrievalBackend = load_first_stage_retriever(config)
        self.text_k = int(rcfg.get("text_k", 20))
        self.image_k = int(rcfg.get("image_k", 20))
        self.filter = build_evidence_filter(config.get("rag", {}).get("evidence_filter", {}))

    def search(self, query: str, image_path: str, question: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
        hits = self.retrieval.search_text(query, k=self.text_k)
        if image_path:
            hits += self.retrieval.search_image(image_path, k=self.image_k)
        return self.filter.select(hits, question, options)

    def close(self) -> None:
        self.retrieval.close()


class MockRagPipeline:
    """Deterministic final passages for smoke tests — simulates the whole pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.calls = 0

    def search(self, query: str, image_path: str, question: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
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
