import unittest
from unittest.mock import MagicMock, patch

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from agent.kb.retrieval.embeddings import EMBEDDING_MODEL_NAME, EmbeddingModel, EmbeddingModelError
from agent.kb.retrieval.records import build_knowledge_records

if np is not None:
    from agent.kb.retrieval.retriever import (
        DEFAULT_RELEVANCE_THRESHOLD,
        DEFAULT_TOP_K,
        RetrievalError,
        SemanticRetriever,
    )


KB = {
    "pricing": {
        "basic": {"price": "$29/month", "features": ["10 videos per month", "720p resolution"]},
        "pro": {"price": "$79/month", "features": ["Unlimited videos", "4K resolution", "AI captions"]},
    },
    "policies": {
        "refund": "No refunds after 7 days",
        "support": "Pro Plan customers receive 24/7 support via email at support@autostream.com.",
    },
}


class FakeEncoder:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.calls = 0

    def encode(self, texts, show_progress_bar=False):
        self.calls += 1
        vecs = []
        for t in texts:
            for key, vec in self.mapping.items():
                if key.lower() in str(t).lower():
                    vecs.append(vec)
                    break
            else:
                vecs.append(np.ones(8, dtype=float))
        return np.asarray(vecs, dtype=float)


@unittest.skipUnless(np is not None, "NumPy not installed in this environment")
class TestEmbeddingsAndRetriever(unittest.TestCase):
    def test_lazy_load_not_on_init(self):
        model = EmbeddingModel()
        self.assertFalse(model.is_loaded)
        self.assertEqual(model.model_name, EMBEDDING_MODEL_NAME)

    def test_injected_model_encodes(self):
        fake = FakeEncoder()
        model = EmbeddingModel(model=fake)
        out = model.encode(["hello"])
        self.assertEqual(out.shape[0], 1)
        self.assertTrue(model.is_loaded)

    def test_model_load_failure(self):
        with patch("agent.kb.retrieval.embeddings.EmbeddingModel._ensure_model", side_effect=EmbeddingModelError("boom")):
            model = EmbeddingModel()
            with self.assertRaises(EmbeddingModelError):
                model.encode(["x"])

    def test_retriever_indexes_and_caches_embeddings(self):
        records = build_knowledge_records(KB)
        fake = MagicMock()
        # query encoding only; doc vectors pre-cached
        fake.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]])
        retriever = SemanticRetriever(embedding_model=EmbeddingModel(model=fake), top_k=2, threshold=-1.0)
        retriever._records = records
        retriever._doc_vectors = np.ones((len(records), 4), dtype=float)
        retriever._kb_fingerprint = "test"
        scores = np.zeros(len(records), dtype=float)
        refund_idx = next(i for i, r in enumerate(records) if r.id == "policy.refund")
        scores[refund_idx] = 0.9
        scores[0] = 0.4
        with patch("agent.kb.retrieval.retriever.cosine_similarity_scores", return_value=scores), patch.object(
            retriever, "index"
        ):
            result = retriever.retrieve("refund policy question")
        self.assertTrue(result.has_relevant_context)
        self.assertEqual(len(result.hits), 2)
        self.assertEqual(result.hits[0].record.id, "policy.refund")
        fake.encode.assert_called_once()  # query only; docs pre-cached

    def test_threshold_abstention(self):
        records = build_knowledge_records(KB)
        doc_vecs = np.ones((len(records), 4), dtype=float)
        fake = MagicMock()
        fake.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]])
        retriever = SemanticRetriever(embedding_model=EmbeddingModel(model=fake), threshold=0.99)
        retriever._records = records
        retriever._doc_vectors = doc_vecs
        retriever._kb_fingerprint = "test"
        result = retriever.retrieve("unrelated")
        self.assertFalse(result.has_relevant_context)
        self.assertEqual(result.hits, [])

    def test_empty_knowledge_base(self):
        fake = MagicMock()
        retriever = SemanticRetriever(embedding_model=EmbeddingModel(model=fake), threshold=0.1)
        retriever._records = []
        retriever._doc_vectors = np.asarray([], dtype=float)
        retriever._kb_fingerprint = "empty"
        result = retriever.retrieve("anything")
        self.assertFalse(result.has_relevant_context)

    def test_empty_query(self):
        retriever = SemanticRetriever(embedding_model=EmbeddingModel(model=MagicMock()))
        result = retriever.retrieve("   ")
        self.assertEqual(result.error, "empty_query")

    def test_invalid_embedding_shape_error(self):
        records = build_knowledge_records(KB)
        fake = MagicMock()
        fake.encode.return_value = np.array([1.0, 2.0])
        retriever = SemanticRetriever(embedding_model=EmbeddingModel(model=fake))
        with patch.object(retriever, "_load_records", return_value=records):
            retriever.clear_cache()
            result = retriever.retrieve("price")
        self.assertIsNotNone(result.error)

    def test_configurable_defaults(self):
        self.assertEqual(DEFAULT_TOP_K, 3)
        self.assertIsInstance(DEFAULT_RELEVANCE_THRESHOLD, float)
        self.assertEqual(DEFAULT_RELEVANCE_THRESHOLD, 0.25)


if __name__ == "__main__":
    unittest.main()
