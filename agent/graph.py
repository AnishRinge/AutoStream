"""LangGraph orchestrator for the AutoStream social-to-lead assistant.

Phase 1 builds the core conversational architecture:
intent classification -> centralized routing -> handler -> formatted response.

Future phases will extend handlers/routing for direct JSON lookup, RAG,
Gemini grounded generation, lead FSM, validation, confirmation, and SQLite.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agent.handlers.business import handle_business
from agent.handlers.fallback import handle_fallback
from agent.handlers.greeting import handle_greeting
from agent.handlers.lead import handle_lead
from agent.handlers.response import format_response
from agent.intent import detect_intent
from agent.routing import (
    ROUTE_BUSINESS,
    ROUTE_FALLBACK,
    ROUTE_GREETING,
    ROUTE_LEAD,
    route_from_state,
)
from agent.state import AgentState


def intent_node(state: AgentState) -> AgentState:
    state["intent"] = detect_intent(state.get("input", ""))
    return state


def greeting_node(state: AgentState) -> AgentState:
    state["route"] = ROUTE_GREETING
    state["response"] = format_response(handle_greeting(state))
    return state


def business_node(state: AgentState) -> AgentState:
    """Handle business queries and lead-route placeholder turns.

    The dedicated `lead` graph node uses this handler as a placeholder until
    the full lead FSM is implemented. Route reporting reflects the resolved
    route for the turn so callers can distinguish lead-hook handling from a
    business interruption.
    """
    state["route"] = route_from_state(state)
    state["response"] = format_response(handle_business(state))
    return state


def lead_node(state: AgentState) -> AgentState:
    """Deterministic lead qualification FSM (no Gemini, no persistence)."""
    updated = handle_lead(state)
    updated["route"] = ROUTE_LEAD
    updated["response"] = format_response(updated.get("response"))
    return updated


def route_node(state: AgentState) -> AgentState:
    """Record the resolved route for observability and future lead-state hooks."""
    state["route"] = route_from_state(state)
    return state


def fallback_node(state: AgentState) -> AgentState:
    state["route"] = ROUTE_FALLBACK
    state["response"] = format_response(handle_fallback(state))
    return state


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intent", intent_node)
    graph.add_node("route", route_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("business", business_node)
    graph.add_node("lead", lead_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("intent")
    graph.add_edge("intent", "route")
    graph.add_conditional_edges(
        "route",
        route_from_state,
        {
            ROUTE_GREETING: "greeting",
            ROUTE_BUSINESS: "business",
            ROUTE_LEAD: "lead",
            ROUTE_FALLBACK: "fallback",
        },
    )
    graph.add_edge("greeting", END)
    graph.add_edge("business", END)
    graph.add_edge("lead", END)
    graph.add_edge("fallback", END)

    return graph.compile()
