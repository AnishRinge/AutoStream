"""Embedding model adapter (Sentence Transformers) with lazy loading."""
from __future__ import annotations

from typing import Any, Sequence

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModelError(RuntimeError):
    """Raised when the embedding model cannot be loaded or used."""


class EmbeddingModel:
    """Lazy-loading SentenceTransformer wrapper.

    The heavy model is loaded on first use. Tests can inject a fake model via
    `model=` or monkeypatch `SentenceTransformer`.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, *, model: Any = None):
        self.model_name = model_name
        self._model = model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise EmbeddingModelError(
                "sentence-transformers is not available; install it to enable semantic retrieval."
            ) from exc
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            raise EmbeddingModelError(
                f"Failed to load embedding model '{self.model_name}'."
            ) from exc
        return self._model

    def encode(self, texts: Sequence[str]):
        model = self._ensure_model()
        try:
            return model.encode(list(texts), show_progress_bar=False)
        except Exception as exc:
            raise EmbeddingModelError("Embedding generation failed.") from exc
