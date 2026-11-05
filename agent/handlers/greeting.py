"""Deterministic greeting handler (no LLM, no RAG)."""

from __future__ import annotations

from agent.state import INTENT_GOODBYE, INTENT_THANKS


def handle_greeting(state: dict) -> str:
    intent = state.get("intent")
    if intent == INTENT_THANKS:
        return "You're welcome. If you need help with AutoStream, I can assist."
    if intent == INTENT_GOODBYE:
        return "Goodbye. If you need AutoStream help later, I'm here."
    return "Hello. I can help with AutoStream pricing, plans, policies, or getting started."
