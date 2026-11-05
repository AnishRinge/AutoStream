import json
import os
import unittest
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from agent.kb.retrieval.embeddings import EmbeddingModel
from agent.kb.retrieval.retriever import SemanticRetriever

# Integration retrieval tests require numpy and sentence-transformers. They also
# load the real embedding model unless RUN_RETRIEVAL_EVAL=1 is set with network/
# model cache available. By default they are skipped to keep CI/local runs fast
# and network-free.
RUN_EVAL = os.environ.get("RUN_RETRIEVAL_EVAL") == "1"


@unittest.skipUnless(RUN_EVAL and np is not None, "Set RUN_RETRIEVAL_EVAL=1 with deps+model available")
class TestSemanticRetrievalIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = SemanticRetriever(embedding_model=EmbeddingModel(), top_k=3, threshold=0.30)
        cls.retriever.index()

    def _top_id(self, query, threshold=None):
        res = self.retriever.retrieve(query, threshold=self.retriever.threshold if threshold is None else threshold)
        return res, (res.hits[0].record.id if res.hits else None)

    def test_pricing_query(self):
        res, top = self._top_id("How much does AutoStream cost?")
        self.assertTrue(res.has_relevant_context)
        self.assertTrue(top and top.startswith("pricing."))

    def test_plan_feature_query(self):
        res, top = self._top_id("Does the Pro plan include AI captions and 4K?")
        self.assertTrue(res.has_relevant_context)
        self.assertTrue(top and ("pricing.pro" in top))

    def test_refund_policy_query(self):
        res, top = self._top_id("Can I get my money back after buying?")
        self.assertTrue(res.has_relevant_context)
        self.assertEqual(top, "policy.refund")

    def test_support_policy_query(self):
        res, top = self._top_id("How do I contact customer support?")
        self.assertTrue(res.has_relevant_context)
        self.assertTrue(top and "policy.support" in top)

    def test_paraphrase_query(self):
        res, _ = self._top_id("What is the monthly subscription price?")
        self.assertTrue(res.has_relevant_context)

    def test_unrelated_abstains_at_threshold(self):
        res, _ = self._top_id("What is the weather on Mars tomorrow?", threshold=0.45)
        self.assertFalse(res.has_relevant_context)


EVALUATION_SET = [
    ("What plans do you offer?", "pricing"),
    ("How much is the Pro plan?", "pricing.pro"),
    ("Basic plan price", "pricing.basic"),
    ("Does the Pro plan have 4K resolution?", "pricing.pro"),
    ("AI captions feature", "pricing.pro"),
    ("10 videos per month limit", "pricing.basic"),
    ("What is your refund policy?", "policy.refund"),
    ("Can I get a refund after a week?", "policy.refund"),
    ("How do I contact support?", "policy.support"),
    ("Is there 24/7 email support?", "policy.support"),
    ("monthly subscription cost", "pricing"),
    ("money back guarantee", "policy.refund"),
    ("customer service email", "policy.support"),
    ("Tell me a joke", None),
    ("Who won the World Cup?", None),
    ("Recipe for pasta", None),
]


def run_threshold_evaluation(threshold=0.30):
    """Manual evaluation helper (not auto-run; may download/load model)."""
    retriever = SemanticRetriever(embedding_model=EmbeddingModel(), top_k=3, threshold=threshold)
    retriever.index()
    rows = []
    for query, expected_prefix in EVALUATION_SET:
        res = retriever.retrieve(query)
        top = res.hits[0] if res.hits else None
        rows.append(
            {
                "query": query,
                "expected": expected_prefix,
                "top_id": top.record.id if top else None,
                "top_score": round(top.score, 4) if top else None,
                "has_context": res.has_relevant_context,
            }
        )
    return rows


if __name__ == "__main__":
    print(json.dumps(run_threshold_evaluation(), indent=2))
