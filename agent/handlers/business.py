"""Business query handler: direct lookup -> semantic retrieval -> grounded Gemini.

Phase 3B behavior:
- Direct lookup ANSWERED/NOT_FOUND/ERROR: deterministic, no Gemini.
- DEFER_RAG + relevant retrieval context: grounded Gemini generation.
- DEFER_RAG + no relevant context (genuine abstention): insufficient verified info.
- Retrieval internal failure (model/embeddings/load): temporary retrieval failure.
- Gemini failure: graceful temporary response; does not mutate lead state.
"""
from __future__ import annotations

from agent.kb.lookup import LookupStatus, lookup_business_query
from agent.kb.retrieval.generation import (
    INSUFFICIENT_CONTEXT_RESPONSE,
    RETRIEVAL_FAILURE_RESPONSE,
    generate_grounded_response,
    safe_generation_fallback,
)
from agent.kb.retrieval.retriever import SemanticRetriever
from agent.state import INTENT_LEAD_INTEREST

_DEFAULT_RETRIEVER: SemanticRetriever | None = None


def get_default_retriever() -> SemanticRetriever:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        # Preserves Phase 3A defaults: all-MiniLM-L6-v2, NumPy cosine, top_k=3, threshold=0.25.
        _DEFAULT_RETRIEVER = SemanticRetriever()
    return _DEFAULT_RETRIEVER


def reset_default_retriever() -> None:
    """Test/helper hook to clear the default retriever and its caches."""
    global _DEFAULT_RETRIEVER
    _DEFAULT_RETRIEVER = None


def handle_business(
    state: dict,
    *,
    retriever: SemanticRetriever | None = None,
    gemini_client=None,
) -> str:
    """Handle business/lead-intent turns using verified KB data and grounded RAG."""
    if state.get("intent") == INTENT_LEAD_INTEREST:
        # Lead interest is handled by the lead FSM via routing; keep a safe fallback
        # for direct handler calls/tests that bypass graph routing.
        if (state.get("lead_state") or "idle") == "idle":
            return (
                "I can help you get started with AutoStream. What's your name?"
            )
        return "I can help you continue getting started with AutoStream."

    query = state.get("input", "")
    result = lookup_business_query(query)

    if result.status in {LookupStatus.ANSWERED, LookupStatus.NOT_FOUND, LookupStatus.ERROR}:
        return result.response

    # DEFER_RAG: semantic retrieval first; Gemini only when relevant context exists.
    active_retriever = retriever or get_default_retriever()
    try:
        retrieval = active_retriever.retrieve(query)
    except Exception:
        # Internal retrieval failure (model/embeddings/runtime), not "no answer in KB".
        return RETRIEVAL_FAILURE_RESPONSE

    if getattr(retrieval, "error", None):
        return RETRIEVAL_FAILURE_RESPONSE
    if not retrieval.has_relevant_context or not retrieval.hits:
        # Genuine threshold-based abstention: no sufficiently relevant verified context.
        return INSUFFICIENT_CONTEXT_RESPONSE

    try:
        return generate_grounded_response(query, retrieval.hits, client=gemini_client)
    except Exception as exc:
        return safe_generation_fallback(exc)
