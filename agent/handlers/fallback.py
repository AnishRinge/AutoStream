"""Fallback/clarification handler.

Handles unknown, ambiguous, and unsupported requests without inventing facts.
"""

from __future__ import annotations


def handle_fallback(state: dict) -> str:
    return (
        "I can help with AutoStream pricing, plans, policies, or getting started. "
        "Could you rephrase your request?"
    )
