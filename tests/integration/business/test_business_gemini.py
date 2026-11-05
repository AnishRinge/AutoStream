import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.handlers.business import handle_business
from agent.kb.lookup import LookupResult, LookupStatus
from agent.kb.retrieval.generation import INSUFFICIENT_CONTEXT_RESPONSE, TEMPORARY_FAILURE_RESPONSE
from agent.kb.retrieval.records import KnowledgeRecord
from agent.kb.retrieval.retriever import RetrievalResult
from agent.kb.retrieval.similarity import RetrievalHit
from agent.llm.gemini_client import GeminiClientError
from agent.state import (
    INTENT_BUSINESS_QUERY,
    INTENT_LEAD_INTEREST,
    LEAD_STATE_AWAITING_NAME,
    TRANSIENT_STATE_KEYS,
)


def hit():
    return RetrievalHit(
        record=KnowledgeRecord(id="pricing.pro", text="Pro $79", category="pricing", source="kb", metadata={}),
        score=0.9,
    )


class TestBusinessGeminiIntegration(unittest.TestCase):
    def test_direct_lookup_answered_no_gemini(self):
        retriever = MagicMock()
        client = MagicMock()
        state = {"intent": INTENT_BUSINESS_QUERY, "input": "What is your refund policy?"}
        out = handle_business(state, retriever=retriever, gemini_client=client)
        self.assertIn("No refunds after 7 days", out)
        retriever.retrieve.assert_not_called()
        client.generate_text.assert_not_called()

    def test_direct_lookup_not_found_no_gemini(self):
        retriever = MagicMock()
        client = MagicMock()
        state = {"intent": INTENT_BUSINESS_QUERY, "input": "What integrations do you support?"}
        out = handle_business(state, retriever=retriever, gemini_client=client)
        self.assertIn("verified", out.lower())
        retriever.retrieve.assert_not_called()
        client.generate_text.assert_not_called()

    @patch("agent.handlers.business.lookup_business_query")
    def test_direct_lookup_error_no_gemini(self, mock_lookup):
        mock_lookup.return_value = LookupResult(LookupStatus.ERROR, "kb error")
        retriever = MagicMock()
        client = MagicMock()
        out = handle_business({"intent": INTENT_BUSINESS_QUERY, "input": "q"}, retriever=retriever, gemini_client=client)
        self.assertEqual(out, "kb error")
        retriever.retrieve.assert_not_called()
        client.generate_text.assert_not_called()

    @patch("agent.handlers.business.lookup_business_query")
    def test_defer_rag_invokes_retrieval(self, mock_lookup):
        mock_lookup.return_value = LookupResult(LookupStatus.DEFER_RAG, "defer")
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(has_relevant_context=False, hits=[])
        out = handle_business({"intent": INTENT_BUSINESS_QUERY, "input": "complex"}, retriever=retriever)
        retriever.retrieve.assert_called_once()
        self.assertEqual(out, INSUFFICIENT_CONTEXT_RESPONSE)

    @patch("agent.handlers.business.lookup_business_query")
    def test_internal_retrieval_error_returns_temporary_failure(self, mock_lookup):
        mock_lookup.return_value = LookupResult(LookupStatus.DEFER_RAG, "defer")
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(has_relevant_context=False, hits=[], error="boom")
        out = handle_business({"intent": INTENT_BUSINESS_QUERY, "input": "complex"}, retriever=retriever)
        self.assertIn("trouble accessing", out.lower())

    @patch("agent.handlers.business.generate_grounded_response")
    @patch("agent.handlers.business.lookup_business_query")
    def test_no_relevant_context_does_not_call_gemini(self, mock_lookup, mock_gen):
        mock_lookup.return_value = LookupResult(LookupStatus.DEFER_RAG, "defer")
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(has_relevant_context=False, hits=[], error=None)
        handle_business({"intent": INTENT_BUSINESS_QUERY, "input": "complex"}, retriever=retriever)
        mock_gen.assert_not_called()

    @patch("agent.handlers.business.generate_grounded_response", return_value="Grounded answer")
    @patch("agent.handlers.business.lookup_business_query")
    def test_relevant_context_calls_gemini(self, mock_lookup, mock_gen):
        mock_lookup.return_value = LookupResult(LookupStatus.DEFER_RAG, "defer")
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(has_relevant_context=True, hits=[hit()])
        client = MagicMock()
        out = handle_business(
            {"intent": INTENT_BUSINESS_QUERY, "input": "complex"}, retriever=retriever, gemini_client=client
        )
        self.assertEqual(out, "Grounded answer")
        mock_gen.assert_called_once()

    @patch("agent.handlers.business.generate_grounded_response", side_effect=GeminiClientError("fail"))
    @patch("agent.handlers.business.lookup_business_query")
    def test_gemini_failure_graceful(self, mock_lookup, _mock_gen):
        mock_lookup.return_value = LookupResult(LookupStatus.DEFER_RAG, "defer")
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(has_relevant_context=True, hits=[hit()])
        out = handle_business({"intent": INTENT_BUSINESS_QUERY, "input": "complex"}, retriever=retriever)
        self.assertEqual(out, TEMPORARY_FAILURE_RESPONSE)

    @patch("agent.handlers.business.generate_grounded_response", return_value="Grounded answer")
    @patch("agent.handlers.business.lookup_business_query")
    def test_active_lead_state_preserved_during_gemini_interruption(self, mock_lookup, _mock_gen):
        from agent.graph import build_graph

        mock_lookup.return_value = LookupResult(LookupStatus.DEFER_RAG, "defer")
        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = RetrievalResult(has_relevant_context=True, hits=[hit()])
        fake_client = MagicMock()

        # Exercise handler directly to keep graph tests deterministic and avoid live model.
        state = {
            "intent": INTENT_BUSINESS_QUERY,
            "input": "Which plan is better for a small team?",
            "lead_state": LEAD_STATE_AWAITING_NAME,
            "lead_name": None,
        }
        out = handle_business(state, retriever=fake_retriever, gemini_client=fake_client)
        self.assertEqual(out, "Grounded answer")
        self.assertEqual(state.get("lead_state"), LEAD_STATE_AWAITING_NAME)
        self.assertIn("lead_name", state)

    def test_lead_interest_no_gemini(self):
        client = MagicMock()
        out = handle_business({"intent": INTENT_LEAD_INTEREST, "input": "sign me up", "lead_state": "idle"}, gemini_client=client)
        self.assertIn("name", out.lower())
        client.generate_text.assert_not_called()


if __name__ == "__main__":
    unittest.main()
