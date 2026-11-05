import unittest
from unittest.mock import MagicMock, patch

from agent.handlers import business
from agent.kb.lookup import LookupResult, LookupStatus
from agent.kb.retrieval import generation
from agent.kb.retrieval.records import KnowledgeRecord
from agent.kb.retrieval.retriever import RetrievalResult
from agent.kb.retrieval.similarity import RetrievalHit
from agent.llm.gemini_client import GeminiClientError
from agent.state import INTENT_BUSINESS_QUERY


def hit():
    return RetrievalHit(
        record=KnowledgeRecord(id="pricing.pro", text="Pro $79", category="pricing", source="kb", metadata={}),
        score=0.9,
    )


class TestDefaultLifecycle(unittest.TestCase):
    def setUp(self):
        business.reset_default_retriever()
        generation.reset_default_gemini_client()

    def tearDown(self):
        business.reset_default_retriever()
        generation.reset_default_gemini_client()

    @patch("agent.handlers.business.generate_grounded_response", return_value="ok")
    @patch("agent.handlers.business.lookup_business_query")
    def test_two_deferred_queries_reuse_default_retriever(self, mock_lookup, _mock_gen):
        mock_lookup.return_value = LookupResult(LookupStatus.DEFER_RAG, "defer")
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(has_relevant_context=True, hits=[hit()])

        with patch("agent.handlers.business.SemanticRetriever", return_value=retriever) as ctor:
            business.handle_business({"intent": INTENT_BUSINESS_QUERY, "input": "q1"})
            business.handle_business({"intent": INTENT_BUSINESS_QUERY, "input": "q2"})

        self.assertEqual(ctor.call_count, 1)
        self.assertEqual(retriever.retrieve.call_count, 2)

    @patch("agent.handlers.business.generate_grounded_response", return_value="ok")
    @patch("agent.handlers.business.lookup_business_query")
    def test_kb_embeddings_not_regenerated_when_retriever_reused(self, mock_lookup, _mock_gen):
        mock_lookup.return_value = LookupResult(LookupStatus.DEFER_RAG, "defer")
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(has_relevant_context=True, hits=[hit()])

        with patch("agent.handlers.business.SemanticRetriever", return_value=retriever):
            business.handle_business({"intent": INTENT_BUSINESS_QUERY, "input": "q1"})
            business.handle_business({"intent": INTENT_BUSINESS_QUERY, "input": "q2"})

        # Retriever instance is reused; index/encoding lifecycle is internal to that instance.
        self.assertIs(business.get_default_retriever(), retriever)

    def test_two_gemini_requests_reuse_sdk_client(self):
        fake_sdk = MagicMock()
        fake_sdk.models.generate_content.return_value = type("R", (), {"text": "answer"})()
        fake_genai = MagicMock()
        fake_genai.Client.return_value = fake_sdk

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False), patch.dict(
            "sys.modules", {"google": MagicMock(genai=fake_genai), "google.genai": fake_genai}
        ):
            generation.generate_grounded_response("q1", [hit()], client=None)
            generation.generate_grounded_response("q2", [hit()], client=None)

        self.assertEqual(fake_genai.Client.call_count, 1)
        self.assertEqual(fake_sdk.models.generate_content.call_count, 2)

    @patch("agent.handlers.business.lookup_business_query")
    def test_direct_lookup_does_not_initialize_gemini(self, mock_lookup):
        mock_lookup.return_value = LookupResult(LookupStatus.ANSWERED, "verified")
        with patch("agent.kb.retrieval.generation.get_default_gemini_client") as get_client, patch(
            "agent.handlers.business.get_default_retriever"
        ) as get_ret:
            out = business.handle_business({"intent": INTENT_BUSINESS_QUERY, "input": "simple"})
        self.assertEqual(out, "verified")
        get_client.assert_not_called()
        get_ret.assert_not_called()

    @patch("agent.handlers.business.lookup_business_query")
    def test_missing_api_key_handled_gracefully(self, mock_lookup):
        mock_lookup.return_value = LookupResult(LookupStatus.DEFER_RAG, "defer")
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(has_relevant_context=True, hits=[hit()])
        with patch("agent.kb.retrieval.generation.get_default_gemini_client") as get_client:
            client = MagicMock()
            client.generate_text.side_effect = GeminiClientError("Gemini is not configured (missing API key).")
            get_client.return_value = client
            out = business.handle_business({"intent": INTENT_BUSINESS_QUERY, "input": "q"}, retriever=retriever)
        self.assertIn("try again", out.lower())
        self.assertNotIn("AIza", out)

    @patch("agent.handlers.business.lookup_business_query")
    def test_internal_retrieval_failure_distinct_from_no_context(self, mock_lookup):
        mock_lookup.return_value = LookupResult(LookupStatus.DEFER_RAG, "defer")
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(has_relevant_context=False, hits=[], error="model load failed")
        out = business.handle_business({"intent": INTENT_BUSINESS_QUERY, "input": "q"}, retriever=retriever)
        self.assertEqual(out, generation.RETRIEVAL_FAILURE_RESPONSE)
        self.assertNotIn("knowledge base to answer", out.lower())


if __name__ == "__main__":
    unittest.main()
