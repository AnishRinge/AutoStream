"""NumPy cosine similarity and top-K ranking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from agent.kb.retrieval.records import KnowledgeRecord


@dataclass(frozen=True)
class RetrievalHit:
    record: KnowledgeRecord
    score: float


def _require_numpy():
    if np is None:
        raise RuntimeError("NumPy is required for semantic retrieval. Install project dependencies.")


def _as_matrix(vectors):
    _require_numpy()
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("Embeddings must be a 1D or 2D numeric array.")
    if arr.shape[0] == 0:
        return arr.reshape(0, 0)
    return arr


def _safe_norms(matrix):
    norms = np.linalg.norm(matrix, axis=1)
    norms = np.where(norms == 0.0, 1.0, norms)
    return norms


def cosine_similarity_scores(query_vector, doc_vectors):
    """Return cosine similarity scores for query vs each document vector.

    Invalid/zero norms are handled safely (score becomes 0 for zero vectors).
    """
    _require_numpy()
    query = np.asarray(query_vector, dtype=float).reshape(-1)
    docs = _as_matrix(doc_vectors)
    if docs.size == 0 or query.size == 0:
        return np.asarray([], dtype=float)
    if docs.shape[1] != query.shape[0]:
        raise ValueError("Query and document embedding dimensions do not match.")

    q_norm = np.linalg.norm(query)
    if q_norm == 0.0:
        return np.zeros(docs.shape[0], dtype=float)

    d_norms = _safe_norms(docs)
    dots = np.dot(docs, query)
    scores = dots / (d_norms * q_norm)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores.astype(float)


def rank_top_k(
    records: Sequence[KnowledgeRecord],
    scores: Sequence[float],
    *,
    top_k: int = 3,
    threshold: float | None = None,
) -> list[RetrievalHit]:
    if len(records) != len(scores):
        raise ValueError("records and scores must have the same length.")
    hits = [RetrievalHit(record=r, score=float(s)) for r, s in zip(records, scores)]
    hits.sort(key=lambda h: h.score, reverse=True)
    if threshold is not None:
        hits = [h for h in hits if h.score >= threshold]
    if top_k is not None and top_k >= 0:
        hits = hits[:top_k]
    return hits
