"""Project-relative JSON knowledge base loader with simple in-memory caching.

The JSON file is the source of truth for verified AutoStream business facts.
This module does not mutate the file and performs no external I/O beyond
reading the local JSON.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KB_PATH = _PROJECT_ROOT / "knowledge_base" / "data.json"

_CACHE: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}


class KnowledgeBaseError(ValueError):
    """Raised when the knowledge base cannot be loaded or validated."""


def _validate_structure(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise KnowledgeBaseError("Knowledge base root must be a JSON object.")

    pricing = data.get("pricing")
    policies = data.get("policies")
    if not isinstance(pricing, dict):
        raise KnowledgeBaseError("Knowledge base 'pricing' must be an object.")
    if not isinstance(policies, dict):
        raise KnowledgeBaseError("Knowledge base 'policies' must be an object.")

    for plan_name in ("basic", "pro"):
        plan = pricing.get(plan_name)
        if not isinstance(plan, dict):
            raise KnowledgeBaseError(f"Pricing plan '{plan_name}' must be an object.")
        if not isinstance(plan.get("price"), str) or not plan.get("price").strip():
            raise KnowledgeBaseError(f"Pricing plan '{plan_name}' must include a price.")
        features = plan.get("features")
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            raise KnowledgeBaseError(
                f"Pricing plan '{plan_name}' features must be a list of strings."
            )

    for policy_name in ("refund", "support"):
        value = policies.get(policy_name)
        if not isinstance(value, str) or not value.strip():
            raise KnowledgeBaseError(f"Policy '{policy_name}' must be a non-empty string.")

    return data


def resolve_kb_path(path: str | Path | None = None) -> Path:
    candidate = Path(path) if path is not None else DEFAULT_KB_PATH
    if not candidate.is_absolute():
        candidate = _PROJECT_ROOT / candidate
    return candidate.resolve()


def clear_cache() -> None:
    """Clear the in-memory KB cache (primarily for tests)."""
    _CACHE.clear()


def load_knowledge(path: str | Path | None = None, *, force_reload: bool = False) -> dict[str, Any]:
    """Load and validate the knowledge base JSON.

    Results are cached in memory using the resolved path and file mtime/size.
    Returns a deep copy so callers cannot mutate cached data.
    """
    resolved = resolve_kb_path(path)
    key = str(resolved)

    try:
        stat = resolved.stat()
    except FileNotFoundError as exc:
        raise KnowledgeBaseError(f"Knowledge base file not found: {resolved}") from exc
    except OSError as exc:
        raise KnowledgeBaseError(f"Unable to access knowledge base file: {resolved}") from exc

    fingerprint = (stat.st_mtime_ns, stat.st_size)
    cached = _CACHE.get(key)
    if not force_reload and cached and cached[0] == fingerprint:
        return deepcopy(cached[1])

    try:
        with resolved.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise KnowledgeBaseError(f"Knowledge base JSON is malformed: {resolved}") from exc
    except OSError as exc:
        raise KnowledgeBaseError(f"Unable to read knowledge base file: {resolved}") from exc

    validated = _validate_structure(data)
    _CACHE[key] = (fingerprint, validated)
    return deepcopy(validated)


def get_pricing(path: str | Path | None = None) -> dict[str, Any]:
    return load_knowledge(path).get("pricing", {})


def get_policies(path: str | Path | None = None) -> dict[str, Any]:
    return load_knowledge(path).get("policies", {})
