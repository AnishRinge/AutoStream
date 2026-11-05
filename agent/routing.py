"""Centralized, deterministic routing for LangGraph conditional edges.

Routing decisions are pure functions of structured state (intent + lead state).
They do not perform keyword matching or I/O.
"""

from __future__ import annotations

from agent.state import (
    INTENT_BUSINESS_QUERY,
    INTENT_CANCEL,
    INTENT_CONFIRM,
    INTENT_GOODBYE,
    INTENT_GREETING,
    INTENT_LEAD_INTEREST,
    INTENT_THANKS,
    INTENT_UNKNOWN,
    LEAD_COLLECTION_STAGES,
    LEAD_STATE_IDLE,
    TERMINAL_LEAD_STATES,
)

ROUTE_GREETING = "greeting"
ROUTE_BUSINESS = "business"
ROUTE_LEAD = "lead"
ROUTE_FALLBACK = "fallback"

_INTENT_TO_ROUTE = {
    INTENT_GREETING: ROUTE_GREETING,
    INTENT_THANKS: ROUTE_GREETING,
    INTENT_GOODBYE: ROUTE_GREETING,
    INTENT_BUSINESS_QUERY: ROUTE_BUSINESS,
    INTENT_LEAD_INTEREST: ROUTE_LEAD,
    INTENT_CANCEL: ROUTE_FALLBACK,
    INTENT_CONFIRM: ROUTE_FALLBACK,
    INTENT_UNKNOWN: ROUTE_FALLBACK,
}


def is_active_lead_state(lead_state: str | None) -> bool:
    return (lead_state or LEAD_STATE_IDLE) in LEAD_COLLECTION_STAGES


def is_lead_flow_state(lead_state: str | None) -> bool:
    state = lead_state or LEAD_STATE_IDLE
    return state in LEAD_COLLECTION_STAGES or state in TERMINAL_LEAD_STATES


def resolve_route(intent: str | None, lead_state: str | None) -> str:
    """Resolve a route label from intent and lead qualification state.

    Active collection/confirmation:
    - Business questions route to business (answered) while preserving lead state;
      next turn resumes the same stage.
    - Cancel/confirm and field/correction input route to lead FSM.
    Terminal lead states (confirmed/cancelled): allow new lead interest to start
    fresh via lead route; otherwise route by intent.
    """
    current_intent = intent or INTENT_UNKNOWN
    state = lead_state or LEAD_STATE_IDLE
    if state in LEAD_COLLECTION_STAGES:
        if current_intent == INTENT_BUSINESS_QUERY:
            return ROUTE_BUSINESS
        if current_intent in {INTENT_CANCEL, INTENT_CONFIRM, INTENT_LEAD_INTEREST}:
            return ROUTE_LEAD
        return ROUTE_LEAD
    if state in TERMINAL_LEAD_STATES and current_intent == INTENT_LEAD_INTEREST:
        return ROUTE_LEAD
    if state in TERMINAL_LEAD_STATES and current_intent == INTENT_CONFIRM:
        # Allow confirmation/restart-like explicit commands to be handled by lead FSM.
        return ROUTE_LEAD

    return _INTENT_TO_ROUTE.get(current_intent, ROUTE_FALLBACK)


def route_from_state(state: dict) -> str:
    """LangGraph-compatible routing function."""
    return resolve_route(state.get("intent"), state.get("lead_state"))

