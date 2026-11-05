"""Semantic retriever: query embedding -> cosine similarity -> top-K/threshold.

This module provides a clean interface for Phase 3B (RAG generation).
It does not call Gemini, does not use a vector DB, and does not change routing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from agent.kb.loader import KnowledgeBaseError, load_knowledge
from agent.kb.retrieval.embeddings import EmbeddingModel, EmbeddingModelError
from agent.kb.retrieval.records import KnowledgeRecord, build_knowledge_records
from agent.kb.retrieval.similarity import RetrievalHit, cosine_similarity_scores, rank_top_k

# Provisional: based on all-MiniLM-L6-v2 scores against the current 16-query
# evaluation set (13 relevant + 3 unrelated). Not a universally reliable threshold;
# recalibrate as verified records grow / evaluation expands.
DEFAULT_TOP_K = 3
DEFAULT_RELEVANCE_THRESHOLD = 0.25


class RetrievalError(RuntimeError):
    """Raised for retrieval-time failures (model, embeddings, data)."""


@dataclass
class RetrievalResult:
    hits: list[RetrievalHit] = field(default_factory=list)
    has_relevant_context: bool = False
    query: str = ""
    top_k: int = DEFAULT_TOP_K
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    error: str | None = None

    @property
    def records(self) -> list[KnowledgeRecord]:
        return [h.record for h in self.hits]

    @property
    def scores(self) -> list[float]:
        return [h.score for h in self.hits]


class SemanticRetriever:
    """In-memory semantic retriever with lazy model loading and KB embedding cache."""

    def __init__(
        self,
        *,
        embedding_model: EmbeddingModel | None = None,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        kb_path: str | None = None,
    ):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.top_k = top_k
        self.threshold = threshold
        self.kb_path = kb_path
        self._records: list[KnowledgeRecord] = []
        self._doc_vectors: np.ndarray | None = None
        self._kb_fingerprint: Any = None

    @property
    def is_ready(self) -> bool:
        return self._doc_vectors is not None and len(self._records) > 0

    def clear_cache(self) -> None:
        self._records = []
        self._doc_vectors = None
        self._kb_fingerprint = None

    def _load_records(self) -> list[KnowledgeRecord]:
        try:
            kb = load_knowledge(self.kb_path)
        except KnowledgeBaseError as exc:
            raise RetrievalError(f"Unable to load knowledge base: {exc}") from exc
        return build_knowledge_records(kb)

    def index(self, *, force: bool = False) -> None:
        """Build/cache KB embeddings (no-op when already cached unless forced)."""
        if np is None:
            raise RetrievalError("NumPy is required for semantic retrieval. Install project dependencies.")
        records = self._load_records()
        fingerprint = tuple(r.id + "\x1f" + r.text for r in records)
        if not force and self._kb_fingerprint == fingerprint and self._doc_vectors is not None:
            return
        self._records = records
        if not records:
            self._doc_vectors = np.asarray([], dtype=float)
            self._kb_fingerprint = fingerprint
            return
        try:
            vectors = self.embedding_model.encode([r.text for r in records])
        except EmbeddingModelError as exc:
            raise RetrievalError(str(exc)) from exc
        arr = np.asarray(vectors, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(len(records), -1) if len(records) == 1 else arr.reshape(len(records), -1)
        if arr.ndim != 2 or arr.shape[0] != len(records):
            raise RetrievalError("Embedding model returned vectors with unexpected shape.")
        self._doc_vectors = arr
        self._kb_fingerprint = fingerprint

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> RetrievalResult:
        k = self.top_k if top_k is None else top_k
        t = self.threshold if threshold is None else threshold
        result = RetrievalResult(query=query or "", top_k=k, threshold=t)
        try:
            self.index()
            if not query or not query.strip():
                result.error = "empty_query"
                return result
            if not self._records or self._doc_vectors is None or self._doc_vectors.size == 0:
                return result
            q_vec = self.embedding_model.encode([query])
            q_arr = np.asarray(q_vec, dtype=float).reshape(-1)
            scores = cosine_similarity_scores(q_arr, self._doc_vectors)
            hits = rank_top_k(self._records, scores, top_k=k, threshold=t)
            result.hits = hits
            result.has_relevant_context = len(hits) > 0
            return result
        except RetrievalError as exc:
            result.error = str(exc)
            return result
        except (EmbeddingModelError, KnowledgeBaseError) as exc:
            result.error = str(exc)
            return result
        except Exception as exc:  # pragma: no cover
            result.error = f"retrieval_failed: {exc}"
            return result
