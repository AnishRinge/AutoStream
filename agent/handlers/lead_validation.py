"""Deterministic field validators for lead qualification (no LLM)."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    value: str | None = None
    error: str | None = None


_EMAIL_RE = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$")
_URL_RE = re.compile(r"https?://|www\.|\.[a-z]{2,}/", re.I)


def _looks_like_unrelated_question(text: str) -> bool:
    t = text.strip().lower()
    if "?" in t:
        return True
    starters = ("what ", "how ", "why ", "when ", "where ", "who ", "which ", "can you", "could you", "tell me")
    return any(t.startswith(s) for s in starters)


def validate_name(value: str) -> ValidationResult:
    raw = (value or "").strip()
    if not raw:
        return ValidationResult(False, error="Please share your name.")
    if _URL_RE.search(raw):
        return ValidationResult(False, error="That doesn't look like a name. Please share your name.")
    if _looks_like_unrelated_question(raw):
        return ValidationResult(False, error="I still need your name to continue. What's your name?")
    # Allow letters, spaces, hyphens, apostrophes, periods (common in names); reject mostly punctuation/digits.
    cleaned = re.sub(r"[\s'\-\.]", "", raw)
    if len(cleaned) < 2:
        return ValidationResult(False, error="Please share your full name.")
    if re.fullmatch(r"[\W\d_]+", raw):
        return ValidationResult(False, error="That doesn't look like a name. Please share your name.")
    if sum(c.isalpha() for c in raw) < 2:
        return ValidationResult(False, error="Please share a valid name.")
    return ValidationResult(True, value=raw)


def validate_email(value: str) -> ValidationResult:
    raw = (value or "").strip()
    if not raw:
        return ValidationResult(False, error="Please share your email address.")
    if _looks_like_unrelated_question(raw) and "@" not in raw:
        return ValidationResult(False, error="I still need your email address. What's your email?")
    if not _EMAIL_RE.match(raw) or ".." in raw:
        return ValidationResult(False, error="That email address doesn't look valid. Please share a valid email (for example, name@example.com).")
    return ValidationResult(True, value=raw)


def validate_platform(value: str) -> ValidationResult:
    raw = (value or "").strip()
    if not raw:
        return ValidationResult(False, error="Please share the platform you use or intend to use.")
    if _URL_RE.search(raw):
        return ValidationResult(False, error="Please share the platform name (for example, YouTube, Instagram, or your website).")
    if _looks_like_unrelated_question(raw):
        return ValidationResult(False, error="I still need the platform you use or intend to use. Which platform is that?")
    if sum(c.isalnum() for c in raw) < 2:
        return ValidationResult(False, error="Please share a valid platform name.")
    return ValidationResult(True, value=raw)
