"""Deterministic direct lookup over the verified JSON knowledge base.

Simple factual questions return verified facts. Recognized topics with no
verified answer are acknowledged. Complex/semantic questions return a deferred
status so Phase 3 RAG can handle them through a stable interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from agent.kb.loader import KnowledgeBaseError, load_knowledge
from agent.kb.text import contains_any, normalize


class LookupStatus(str, Enum):
    ANSWERED = "answered"
    NOT_FOUND = "not_found"
    DEFER_RAG = "defer_rag"
    ERROR = "error"


@dataclass(frozen=True)
class LookupResult:
    status: LookupStatus
    response: str
    topic: str | None = None


_COMPLEX_MARKERS = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    "difference between",
    "better",
    "best",
    "recommend",
    "should i",
    "which plan",
    "what plan should",
    "for me",
    "if i",
    "when i",
    "can i",
    "do you offer anything",
    "cheapest for",
    "worth it",
)

_PRICING_TERMS = (
    "price",
    "pricing",
    "cost",
    "how much",
    "plan",
    "plans",
    "subscription",
    "offer",
)
_PLAN_TERMS = ("plan", "plans", "offer", "available")
_REFUND_TERMS = ("refund", "money back", "cancelation", "cancellation")
_SUPPORT_TERMS = ("support", "contact", "help", "customer service", "reach", "email")


def _plan_name(text: str) -> str | None:
    if "pro" in text:
        return "pro"
    if "basic" in text:
        return "basic"
    return None


def _format_plan(plan_id: str, plan: dict[str, Any]) -> str:
    title = "Pro Plan" if plan_id == "pro" else "Basic Plan"
    features = "\n".join(f"- {feature}" for feature in plan.get("features", []))
    return f"{title}: {plan.get('price')}\n{features}".strip()


def _format_all_plans(pricing: dict[str, Any]) -> str:
    parts = []
    for plan_id in ("basic", "pro"):
        if plan_id in pricing:
            parts.append(_format_plan(plan_id, pricing[plan_id]))
    if not parts:
        return "I don't have verified pricing information available right now."
    return "Here are AutoStream's available plans:\n\n" + "\n\n".join(parts)


def lookup_business_query(query: str, path: str | None = None) -> LookupResult:
    """Resolve simple factual business questions from verified KB data."""
    text = normalize(query)
    if not text:
        return LookupResult(
            LookupStatus.NOT_FOUND,
            "I couldn't find verified information for that question.",
        )

    try:
        kb = load_knowledge(path)
    except KnowledgeBaseError:
        return LookupResult(
            LookupStatus.ERROR,
            "I couldn't access verified AutoStream information right now.",
            topic="kb_error",
        )

    pricing = kb.get("pricing", {})
    policies = kb.get("policies", {})

    if contains_any(text, _REFUND_TERMS):
        refund = policies.get("refund")
        if refund:
            return LookupResult(LookupStatus.ANSWERED, f"Refund policy: {refund}", "refund")
        return LookupResult(
            LookupStatus.NOT_FOUND,
            "I don't have verified refund policy information available.",
            "refund",
        )

    asks_integrations = contains_any(text, ("integration", "integrations", "integrate", "api", "zapier"))
    if asks_integrations and not contains_any(text, ("plan", "plans", "price", "cost", "refund")):
        return LookupResult(
            LookupStatus.NOT_FOUND,
            "I don't have verified information available for that specific question.",
            "support_unknown",
        )

    plan = _plan_name(text)

    # Simple factual intents can include "can I" (e.g. how many videos can I make);
    # complex/comparative/semantic forms are deferred to Phase 3 RAG.
    is_simple_capacity = bool(re.search(r"how many videos? can i", text))
    is_simple_support_contact = bool(re.search(r"how (can|do) i (contact|reach)", text))
    complex = contains_any(text, _COMPLEX_MARKERS)
    if is_simple_capacity or is_simple_support_contact:
        complex = False
    elif re.search(r"\bcan i\b", text):
        complex = True
    if complex:
        return LookupResult(
            LookupStatus.DEFER_RAG,
            "This question needs a more detailed answer path that will be added in a later phase.",
            topic="complex",
        )

    if contains_any(text, _SUPPORT_TERMS) and not contains_any(text, ("plan", "plans", "price", "cost")):
        support = policies.get("support")
        if support:
            return LookupResult(LookupStatus.ANSWERED, f"Support: {support}", "support")
        return LookupResult(
            LookupStatus.NOT_FOUND,
            "I don't have verified support information available.",
            "support",
        )

    if contains_any(text, _PRICING_TERMS) or plan or contains_any(text, ("feature", "features", "include", "includes", "video", "videos", "caption", "captions", "resolution", "4k", "720p")):
        if plan and plan in pricing:
            if any(term in text for term in ("video", "videos", "how many")):
                features = pricing[plan].get("features", [])
                video_info = next((f for f in features if "video" in f.lower()), None)
                if video_info:
                    title = "Pro plan" if plan == "pro" else "Basic plan"
                    return LookupResult(
                        LookupStatus.ANSWERED,
                        f"The {title} includes {video_info.lower()}.",
                        f"pricing.{plan}.videos",
                    )
            if any(term in text for term in ("price", "cost", "how much", "subscription")):
                title = "Pro plan" if plan == "pro" else "Basic plan"
                return LookupResult(
                    LookupStatus.ANSWERED,
                    f"The {title} costs {pricing[plan].get('price')}.",
                    f"pricing.{plan}.price",
                )
            return LookupResult(
                LookupStatus.ANSWERED, _format_plan(plan, pricing[plan]), f"pricing.{plan}"
            )

        if contains_any(text, _PLAN_TERMS) or any(
            term in text for term in ("price", "pricing", "cost", "how much", "subscription")
        ):
            return LookupResult(LookupStatus.ANSWERED, _format_all_plans(pricing), "pricing")

        return LookupResult(
            LookupStatus.NOT_FOUND,
            "I don't have verified pricing information available for that request.",
            "pricing",
        )

    # Recognized as business intent by routing, but no verified factual match.
    if contains_any(text, ("autostream", "product", "service", "company")):
        return LookupResult(
            LookupStatus.ANSWERED,
            "I can help with verified AutoStream information about pricing, plans, the refund policy, and support contact details.",
            topic="business_overview",
        )
    return LookupResult(
        LookupStatus.NOT_FOUND,
        "I don't have verified information available for that specific question.",
        topic="business_unknown",
    )
