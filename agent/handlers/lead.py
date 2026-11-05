"""Deterministic lead qualification FSM handler (Phase 4A).

No LLM, no SQLite, no external submission. State transitions live here.
"""
from __future__ import annotations

import re
import uuid
from typing import Any

from agent.handlers.lead_validation import validate_email, validate_name, validate_platform
from agent.routing import ROUTE_LEAD
from agent.state import (
    INTENT_BUSINESS_QUERY,
    INTENT_CANCEL,
    INTENT_CONFIRM,
    INTENT_LEAD_INTEREST,
    LEAD_STATE_AWAITING_CONFIRMATION,
    LEAD_STATE_AWAITING_EMAIL,
    LEAD_STATE_AWAITING_NAME,
    LEAD_STATE_AWAITING_PLATFORM,
    LEAD_STATE_CANCELLED,
    LEAD_STATE_CONFIRMED,
    LEAD_STATE_IDLE,
    TERMINAL_LEAD_STATES,
)


def _text(state: dict) -> str:
    return (state.get("input") or "").strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _clear_lead_fields(state: dict[str, Any]) -> None:
    state["lead_name"] = None
    state["lead_email"] = None
    state["lead_platform"] = None
    state["validation_error"] = None
    state["confirmation_status"] = None
    state["lead_submission_id"] = None
    state["lead_persisted"] = False


def _start_lead(state: dict[str, Any]) -> str:
    _clear_lead_fields(state)
    state["lead_state"] = LEAD_STATE_AWAITING_NAME
    return "Great — I can help you get started. What's your name?"


def _summary(state: dict[str, Any]) -> str:
    return (
        "Please confirm these details:\n"
        f"- Name: {state.get('lead_name')}\n"
        f"- Email: {state.get('lead_email')}\n"
        f"- Platform: {state.get('lead_platform')}\n"
        "Reply with an explicit 'confirm' to submit, 'no' to correct/cancel, or tell me what to change (for example, 'change my email to name@example.com')."
    )


def _prompt_for_stage(stage: str) -> str:
    if stage == LEAD_STATE_AWAITING_NAME:
        return "What's your name?"
    if stage == LEAD_STATE_AWAITING_EMAIL:
        return "What's your email address?"
    if stage == LEAD_STATE_AWAITING_PLATFORM:
        return "Which platform do you use or intend to use?"
    if stage == LEAD_STATE_AWAITING_CONFIRMATION:
        return "Please reply with 'confirm' to submit, 'no' to correct/cancel, or tell me which detail to change."
    return "How can I help with AutoStream?"


def _cancel(state: dict[str, Any]) -> str:
    _clear_lead_fields(state)
    state["lead_state"] = LEAD_STATE_CANCELLED
    state["confirmation_status"] = "cancelled"
    return "Lead collection cancelled. If you'd like, I can still help with AutoStream pricing, plans, or policies."


def _is_explicit_reject(text: str, intent: str | None = None) -> bool:
    t = _norm(text)
    # Route-level INTENT_CONFIRM is used for reject/correction phrases too; require text evidence.
    if t in {"no", "nope", "nah", "reject", "not correct", "incorrect", "no thanks"}:
        return True
    if t.startswith("no,") or t.startswith("no "):
        return True
    if "not correct" in t or "incorrect" in t or "don't submit" in t or "dont submit" in t or "that's wrong" in t or "thats wrong" in t:
        return True
    return False


def _is_explicit_confirm(text: str, intent: str | None) -> bool:
    if _is_explicit_reject(text, intent):
        return False
    t = _norm(text)
    positives = {"yes", "yep", "yeah", "sure", "confirm", "submit", "ok", "okay", "correct", "that's correct", "thats correct", "yes confirm", "yes, confirm"}
    if t in positives or "confirm" in t.split() or t.startswith("yes"):
        # "yes, not correct" handled by reject check above
        if "not" in t or "no" == t.split(",")[0].strip():
            return False
        return True
    # intent==confirm alone is ambiguous (reject/correction phrases can share routing); require affirmative text
    if intent == INTENT_CONFIRM and any(w in t for w in ("yes", "confirm", "submit", "correct", "looks good", "go ahead")):
        return True
    return False


def _extract_correction(text: str) -> tuple[str, str] | None:
    """Return (field, value) for deterministic correction commands."""
    raw = text.strip()
    patterns = [
        (r"^(?:change|correct|update|fix)\s+(?:my\s+)?name\s+(?:to\s+)?(.+)$", "name"),
        (r"^(?:my\s+)?name\s+(?:should\s+be|is)\s+(.+)$", "name"),
        (r"^(?:change|correct|update|fix)\s+(?:my\s+)?email\s+(?:to\s+|address\s+to\s+)?(.+)$", "email"),
        (r"^(?:my\s+)?email\s+(?:should\s+be|is|address\s+is)\s+(.+)$", "email"),
        (r"^(?:change|correct|update|fix)\s+(?:my\s+)?platform\s+(?:to\s+)?(.+)$", "platform"),
        (r"^(?:my\s+)?platform\s+(?:should\s+be|is)\s+(.+)$", "platform"),
        (r"^(?:i\s+use|i\s'm\s+on|im\s+on|use)\s+(.+?)(?:\s*,?\s*not\s+.+)?$", "platform"),
    ]
    for pat, field in patterns:
        m = re.match(pat, raw, re.I)
        if m:
            value = m.group(1).strip().strip(".")
            if value:
                return field, value
    return None


def _apply_correction(state: dict[str, Any], field: str, value: str) -> str:
    validators = {"name": ("lead_name", validate_name), "email": ("lead_email", validate_email), "platform": ("lead_platform", validate_platform)}
    key, validator = validators[field]
    prev = state.get(key)
    result = validator(value)
    if not result.valid:
        state["validation_error"] = result.error
        if prev is not None:
            return f"{result.error} (Your current {field} is still saved.)"
        return result.error or "That value isn't valid."
    state[key] = result.value
    state["validation_error"] = None
    # If correction happens before/at confirmation, return to confirmation once all fields present.
    if all(state.get(k) for k in ("lead_name", "lead_email", "lead_platform")):
        state["lead_state"] = LEAD_STATE_AWAITING_CONFIRMATION
        state["confirmation_status"] = "pending"
        return f"Updated your {field}.\n\n{_summary(state)}"
    state["lead_state"] = LEAD_STATE_AWAITING_NAME if not state.get("lead_name") else LEAD_STATE_AWAITING_EMAIL if not state.get("lead_email") else LEAD_STATE_AWAITING_PLATFORM
    return f"Updated your {field}. {_prompt_for_stage(state['lead_state'])}"


def handle_lead(state: dict[str, Any]) -> dict[str, Any]:
    """Process one turn of lead qualification. Returns updated state."""
    state["route"] = ROUTE_LEAD
    state.setdefault("lead_state", LEAD_STATE_IDLE)
    intent = state.get("intent")
    stage = state.get("lead_state") or LEAD_STATE_IDLE
    text = _text(state)

    # Start fresh flow on explicit lead interest (including after terminal states).
    if intent == INTENT_LEAD_INTEREST and stage in (LEAD_STATE_IDLE, *TERMINAL_LEAD_STATES):
        state["response"] = _start_lead(state)
        return state

    # If lead route reached in idle/cancelled without lead interest (should be rare), be safe.
    if stage == LEAD_STATE_IDLE:
        state["response"] = "If you'd like to get started with AutoStream, tell me and I'll collect your details."
        return state
    if stage == LEAD_STATE_CANCELLED:
        if intent == INTENT_CONFIRM:
            state["response"] = _start_lead(state)
            return state
        state["response"] = "Your previous lead request was cancelled. If you'd like to start over, tell me you'd like to get started."
        return state

    if stage == LEAD_STATE_CONFIRMED:
        state["confirmation_status"] = "confirmed"
        state["response"] = (
            "Your details are already confirmed and ready for the next step. "
            "If you need to start over, tell me you'd like to begin again."
        )
        return state

    # Cancellation at any active stage.
    if intent == INTENT_CANCEL or _norm(text) in {"cancel", "cancel lead", "stop collecting my details", "never mind", "nevermind"}:
        state["response"] = _cancel(state)
        return state

    # Deterministic corrections (available during collection/confirmation).
    correction = _extract_correction(text)
    if correction:
        # Guard: don't mis-handle general business questions that happen to match loose forms.
        if intent == INTENT_BUSINESS_QUERY and not re.match(r"^(change|correct|update|fix|my\s+\w+\s+(should\s+be|is))", text.strip(), re.I):
            state["response"] = "I can update your name, email, or platform if you tell me which one to change."
            return state
        state["response"] = _apply_correction(state, correction[0], correction[1])
        return state

    if stage == LEAD_STATE_AWAITING_NAME:
        result = validate_name(text)
        if not result.valid:
            state["validation_error"] = result.error
            state["response"] = result.error or "Please share your name."
            return state
        state["lead_name"] = result.value
        state["validation_error"] = None
        state["lead_state"] = LEAD_STATE_AWAITING_EMAIL
        state["response"] = f"Thanks, {result.value}. What's your email address?"
        return state

    if stage == LEAD_STATE_AWAITING_EMAIL:
        result = validate_email(text)
        if not result.valid:
            state["validation_error"] = result.error
            state["response"] = result.error or "Please share a valid email address."
            return state
        state["lead_email"] = result.value
        state["validation_error"] = None
        state["lead_state"] = LEAD_STATE_AWAITING_PLATFORM
        state["response"] = "Thanks. Which platform do you use or intend to use?"
        return state

    if stage == LEAD_STATE_AWAITING_PLATFORM:
        result = validate_platform(text)
        if not result.valid:
            state["validation_error"] = result.error
            state["response"] = result.error or "Please share a platform."
            return state
        state["lead_platform"] = result.value
        state["validation_error"] = None
        state["lead_state"] = LEAD_STATE_AWAITING_CONFIRMATION
        state["confirmation_status"] = "pending"
        state["response"] = _summary(state)
        return state

    if stage == LEAD_STATE_AWAITING_CONFIRMATION:
        if _is_explicit_reject(text, intent):
            state["confirmation_status"] = "rejected"
            state["response"] = (
                "No problem — I won't submit these details. "
                "Tell me which detail to change (name, email, or platform), or reply 'cancel' to stop."
            )
            return state
        if _is_explicit_confirm(text, intent):
            state["lead_state"] = LEAD_STATE_CONFIRMED
            state["confirmation_status"] = "confirmed"
            state["validation_error"] = None
            if not state.get("lead_submission_id"):
                state["lead_submission_id"] = uuid.uuid4().hex
            state.setdefault("lead_persisted", False)
            if state.get("lead_persisted"):
                state["response"] = "Your details are already confirmed and saved."
            else:
                state["response"] = (
                    "Thanks — your details are confirmed. I'll save them now."
                )
            return state
        # Ambiguous: do not confirm.
        state["confirmation_status"] = "pending"
        state["response"] = (
            "I need an explicit confirmation. Reply with 'confirm' to submit, 'no' to correct/cancel, "
            "or tell me what to change."
        )
        return state

    state["response"] = "I couldn't process that lead step. Please tell me if you'd like to continue."
    return state
