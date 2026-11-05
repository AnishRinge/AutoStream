"""Deterministic, rule-based intent classification.

This module is intentionally independent from LangGraph and routing.
It performs no I/O, no model calls, and no embedding lookups.
"""

from __future__ import annotations

import re

from agent.state import (
    INTENT_BUSINESS_QUERY,
    INTENT_CANCEL,
    INTENT_CONFIRM,
    INTENT_GOODBYE,
    INTENT_GREETING,
    INTENT_LEAD_INTEREST,
    INTENT_THANKS,
    INTENT_UNKNOWN,
)

# Preserved reference: prior uncommitted high-intent phrases informed these
# patterns; the missing-comma bug between "interested" and "purchase" is fixed.

_GREETING_PHRASES = (
    "good morning",
    "good afternoon",
    "good evening",
)

_GREETING_WORDS = ("hi", "hello", "hey", "hola")

_THANKS_PHRASES = (
    "thank you",
    "thanks a lot",
    "thanks so much",
    "many thanks",
    "appreciate it",
    "much appreciated",
)

_THANKS_WORDS = ("thanks", "thx", "ty")

_GOODBYE_PHRASES = (
    "good night",
    "see you",
    "talk to you later",
    "have a good day",
    "have a nice day",
)

_GOODBYE_WORDS = ("bye", "goodbye", "farewell", "later", "cya")

# Explicit flow-control commands (FSM/confirmation). Classified separately so
# routing/lead handling can distinguish them from ordinary field input.
_CANCEL_PHRASES = (
    "cancel",
    "cancel lead",
    "stop collecting my details",
    "stop collecting",
    "stop",
    "never mind",
    "nevermind",
    "don't continue",
    "dont continue",
    "quit this",
    "exit lead",
)

_CONFIRM_PHRASES = (
    "confirm",
    "yes that's correct",
    "yes thats correct",
    "that's correct",
    "thats correct",
    "looks good",
    "submit it",
    "submit",
    "go ahead",
)

# Explicit negative confirmation/correction signal (routed to lead when active).
_REJECT_CONFIRM_PHRASES = (
    "no that's wrong",
    "no thats wrong",
    "not correct",
    "incorrect",
    "don't submit",
    "dont submit",
)

# Lead interest requires explicit intent to proceed/sign up/buy/try a plan.
# Broad words like "plan" or "interested in pricing" are handled as business.
_LEAD_INTEREST_PHRASES = (
    "sign up",
    "sign me up",
    "get started",
    "want to try",
    "wanna try",
    "want to join",
    "wanna join",
    "want to buy",
    "wanna buy",
    "want to purchase",
    "wanna purchase",
    "i would like to buy",
    "i'd like to buy",
    "i would like to purchase",
    "i'd like to purchase",
    "i want the pro plan",
    "i want pro plan",
    "try the pro plan",
    "try pro plan",
    "try the basic plan",
    "try basic plan",
    "proceed with the pro plan",
    "proceed with pro plan",
    "proceed with the basic plan",
    "proceed with basic plan",
    "ready to proceed",
    "ready to sign up",
    "i am ready to sign up",
    "i'm ready to sign up",
    "i want to sign up",
    "i want to proceed",
    "i would like to proceed",
    "i'd like to proceed",
    "let's start",
    "lets start",
    "let's get started",
    "lets get started",
    "i am ready",
    "i'm ready",
    "subscribe",
    "subscribing",
    "buy it",
    "purchase it",
)

_BUSINESS_KEYWORDS = (
    "price",
    "pricing",
    "plan",
    "plans",
    "cost",
    "subscription",
    "subscriptions",
    "feature",
    "features",
    "details",
    "policy",
    "policies",
    "refund",
    "support",
    "video",
    "videos",
    "caption",
    "captions",
    "resolution",
    "4k",
    "720p",
    "autostream",
    "basic",
    "pro",
)


def _normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.split())


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _contains_word(text: str, words: tuple[str, ...]) -> bool:
    padded = f" {text} "
    return any(f" {word} " in padded for word in words)


def detect_intent(query: str) -> str:
    """Return a deterministic intent label for a user message.

    Classification is independent from routing. Explicit cancel/confirm are
    distinguishable labels for future lead FSM/confirmation phases; routing
    decides whether they apply in the current lead context.
    """
    text = _normalize(query)
    if not text:
        return INTENT_UNKNOWN

    if _contains_phrase(text, _CANCEL_PHRASES):
        return INTENT_CANCEL

    if _contains_phrase(text, _CONFIRM_PHRASES) or _contains_phrase(text, _REJECT_CONFIRM_PHRASES):
        return INTENT_CONFIRM

    if _contains_phrase(text, _LEAD_INTEREST_PHRASES):
        return INTENT_LEAD_INTEREST

    if _contains_phrase(text, _GOODBYE_PHRASES) or _contains_word(text, _GOODBYE_WORDS):
        return INTENT_GOODBYE

    if _contains_phrase(text, _THANKS_PHRASES) or _contains_word(text, _THANKS_WORDS):
        return INTENT_THANKS

    if _contains_phrase(text, _GREETING_PHRASES) or _contains_word(text, _GREETING_WORDS):
        return INTENT_GREETING

    if any(keyword in text for keyword in _BUSINESS_KEYWORDS):
        return INTENT_BUSINESS_QUERY

    return INTENT_UNKNOWN
