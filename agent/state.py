"""Shared LangGraph conversation state.

Persistent state survives across turns (CLI merges these between invocations).
Transient state is produced/consumed within a single graph run.
"""

from __future__ import annotations

from typing import Any, List, TypedDict

from typing_extensions import NotRequired

# Intent labels produced by deterministic classification.
INTENT_GREETING = "greeting"
INTENT_BUSINESS_QUERY = "business_query"
INTENT_LEAD_INTEREST = "lead_interest"
INTENT_THANKS = "thanks"
INTENT_GOODBYE = "goodbye"
INTENT_CANCEL = "cancel"
INTENT_CONFIRM = "confirm"
INTENT_UNKNOWN = "unknown"

VALID_INTENTS = {
    INTENT_GREETING,
    INTENT_BUSINESS_QUERY,
    INTENT_LEAD_INTEREST,
    INTENT_THANKS,
    INTENT_GOODBYE,
    INTENT_CANCEL,
    INTENT_CONFIRM,
    INTENT_UNKNOWN,
}

# Lead qualification FSM stages (Phase 4A). Clear constants, no scattered literals.
LEAD_STATE_IDLE = "idle"
LEAD_STATE_AWAITING_NAME = "awaiting_name"
LEAD_STATE_AWAITING_EMAIL = "awaiting_email"
LEAD_STATE_AWAITING_PLATFORM = "awaiting_platform"
LEAD_STATE_AWAITING_CONFIRMATION = "awaiting_confirmation"
LEAD_STATE_CONFIRMED = "confirmed"
LEAD_STATE_CANCELLED = "cancelled"
# Back-compat alias if referenced; canonical terminal success is CONFIRMED.
LEAD_STATE_COMPLETED = LEAD_STATE_CONFIRMED

LEAD_COLLECTION_STAGES = {
    LEAD_STATE_AWAITING_NAME,
    LEAD_STATE_AWAITING_EMAIL,
    LEAD_STATE_AWAITING_PLATFORM,
    LEAD_STATE_AWAITING_CONFIRMATION,
}
ACTIVE_LEAD_STATES = set(LEAD_COLLECTION_STAGES)
TERMINAL_LEAD_STATES = {LEAD_STATE_CONFIRMED, LEAD_STATE_CANCELLED}

# Keys treated as transient per turn by the CLI boundary.
TRANSIENT_STATE_KEYS = {
    "input",
    "response",
    "intent",
    "route",
    "validation_error",
    "retrieved_context",
    "retrieval_scores",
}


class AgentState(TypedDict):
    """TypedDict state passed through the LangGraph workflow.

    Persistent (carried across turns when present):
        conversation_history: Prior user/assistant turns for future context.
        lead_state: Current lead qualification stage.
        lead_name, lead_email, lead_platform: Collected lead fields.
        confirmation_status: Explicit confirmation state (future phase).
        lead_submission_id: Stable id for one confirmed submission (persistence retries).
        lead_persisted: True when confirmed submission has been durably stored.

    Transient (set/overwritten per turn as appropriate):
        input: Current user message (required on entry).
        intent: Classified intent for the current turn.
        response: Final assistant response for the current turn.
        route: Resolved route label (debuggable; not required by callers).
        validation_error: Field/flow validation message (future phase).
        retrieved_context: Retrieved KB records/text (future RAG).
        retrieval_scores: Similarity scores for retrieved items (future RAG).
    """

    input: str
    conversation_history: NotRequired[List[dict[str, Any]]]
    intent: NotRequired[str]
    response: NotRequired[str]
    route: NotRequired[str]
    lead_state: NotRequired[str]
    lead_name: NotRequired[str | None]
    lead_email: NotRequired[str | None]
    lead_platform: NotRequired[str | None]
    validation_error: NotRequired[str | None]
    confirmation_status: NotRequired[str | None]
    lead_submission_id: NotRequired[str | None]
    lead_persisted: NotRequired[bool]
    retrieved_context: NotRequired[Any]
    retrieval_scores: NotRequired[List[float]]
