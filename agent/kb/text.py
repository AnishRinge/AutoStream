"""Lightweight text normalization for deterministic lookup."""

from __future__ import annotations

import re


def normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s$']", " ", text)
    return " ".join(text.split())


def contains_any(text: str, terms: tuple[str, ...] | list[str]) -> bool:
    return any(term in text for term in terms)
