"""Consistent response formatting for CLI output."""

from __future__ import annotations


def format_response(response: str | None) -> str:
    text = (response or "").strip()
    return text or "I couldn't generate a response. Please try again."
