import unittest

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from agent.kb.retrieval.records import KnowledgeRecord

if np is not None:
    from agent.kb.retrieval.similarity import cosine_similarity_scores, rank_top_k


def rec(i, text="t", cat="c"):
    return KnowledgeRecord(id=f"id{i}", text=text, category=cat, source="s", metadata={})


@unittest.skipUnless(np is not None, "NumPy not installed in this environment")
class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        scores = cosine_similarity_scores([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[1], 0.0)

    def test_zero_query_returns_zeros(self):
        scores = cosine_similarity_scores([0.0, 0.0], [[1.0, 0.0], [2.0, 3.0]])
        np.testing.assert_allclose(scores, [0.0, 0.0])

    def test_zero_document_safe(self):
        scores = cosine_similarity_scores([1.0, 0.0], [[0.0, 0.0], [1.0, 0.0]])
        self.assertAlmostEqual(scores[0], 0.0)
        self.assertAlmostEqual(scores[1], 1.0)

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cosine_similarity_scores([1.0, 0.0], [[1.0, 0.0, 0.0]])

    def test_empty_docs(self):
        scores = cosine_similarity_scores([1.0, 0.0], [])
        self.assertEqual(len(scores), 0)


@unittest.skipUnless(np is not None, "NumPy not installed in this environment")
class TestRanking(unittest.TestCase):
    def test_top_k_and_threshold(self):
        records = [rec(0), rec(1), rec(2)]
        scores = [0.1, 0.9, 0.5]
        hits = rank_top_k(records, scores, top_k=2, threshold=0.2)
        self.assertEqual([h.record.id for h in hits], ["id1", "id2"])
        self.assertAlmostEqual(hits[0].score, 0.9)

    def test_abstention_when_below_threshold(self):
        hits = rank_top_k([rec(0), rec(1)], [0.1, 0.2], top_k=3, threshold=0.5)
        self.assertEqual(hits, [])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            rank_top_k([rec(0)], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
