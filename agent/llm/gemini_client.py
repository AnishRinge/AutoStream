"""Centralized Gemini client (google-genai SDK).

Uses `from google import genai` and `client.models.generate_content(...)`.
No live API calls are made by tests; tests mock the SDK/client.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
ENV_API_KEY = "GEMINI_API_KEY"
ENV_MODEL = "GEMINI_MODEL"


class GeminiClientError(RuntimeError):
    """Safe, user/log-safe Gemini error (never includes API keys)."""


def _sanitize_error_message(message: str, *, api_key: str | None = None) -> str:
    text = str(message or "Gemini request failed.")
    if api_key:
        text = text.replace(api_key, "[REDACTED_API_KEY]")
    # Redact common key-like patterns defensively.
    text = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "[REDACTED_API_KEY]", text)
    text = re.sub(r"(api[_-]?key\s*[=:]\s*)([^\s,;]+)", r"\1[REDACTED]", text, flags=re.I)
    return text


def classify_gemini_exception(exc: Exception, *, api_key: str | None = None) -> GeminiClientError:
    name = exc.__class__.__name__.lower()
    msg = _sanitize_error_message(str(exc), api_key=api_key).lower()
    if "missing" in msg and "api key" in msg:
        return GeminiClientError("Gemini is not configured (missing API key).")
    if "auth" in name or "permission" in name or "unauth" in msg or "401" in msg or "403" in msg or "api key" in msg:
        return GeminiClientError("Gemini authentication failed.")
    if "rate" in name or "quota" in msg or "429" in msg or "resource_exhausted" in msg:
        return GeminiClientError("Gemini rate limit reached. Please try again shortly.")
    if "timeout" in name or "timed out" in msg or "deadline" in msg:
        return GeminiClientError("Gemini request timed out. Please try again.")
    if "network" in name or "connection" in msg or "unavailable" in msg or "503" in msg:
        return GeminiClientError("Gemini service is temporarily unavailable.")
    return GeminiClientError("Gemini request failed.")


def extract_text(response: Any) -> str:
    """Extract text from google-genai response safely."""
    if response is None:
        return ""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    # Fallback: candidates/parts shape (best-effort, no SDK type dependency).
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        chunks: list[str] = []
        for part in parts:
            ptext = getattr(part, "text", None)
            if isinstance(ptext, str) and ptext:
                chunks.append(ptext)
        joined = "".join(chunks).strip()
        if joined:
            return joined
    return ""


@dataclass
class GeminiConfig:
    api_key: str | None
    model: str
    temperature: float = 0.2
    max_output_tokens: int = 400


class GeminiClient:
    """Centralized Gemini client; avoids unnecessary client instances per call."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        client: Any = None,
        temperature: float = 0.2,
        max_output_tokens: int = 400,
    ):
        resolved_key = api_key if api_key is not None else os.getenv(ENV_API_KEY)
        resolved_model = model or os.getenv(ENV_MODEL) or DEFAULT_GEMINI_MODEL
        self.config = GeminiConfig(
            api_key=resolved_key,
            model=resolved_model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        self._client = client

    @property
    def model(self) -> str:
        return self.config.model

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.config.api_key:
            raise GeminiClientError("Gemini is not configured (missing API key).")
        try:
            from google import genai  # type: ignore
        except Exception as exc:
            raise GeminiClientError("Gemini SDK is unavailable.") from exc
        try:
            self._client = genai.Client(api_key=self.config.api_key)
        except Exception as exc:
            raise classify_gemini_exception(exc, api_key=self.config.api_key) from exc
        return self._client

    def generate_text(self, prompt: str) -> str:
        if not prompt or not str(prompt).strip():
            raise GeminiClientError("Cannot generate a response from an empty prompt.")
        client = self._ensure_client()
        try:
            # google-genai: client.models.generate_content(model=..., contents=...)
            # Config is passed when supported; keep minimal/controlled.
            try:
                from google.genai import types  # type: ignore

                config = types.GenerateContentConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                )
                response = client.models.generate_content(
                    model=self.config.model,
                    contents=prompt,
                    config=config,
                )
            except Exception:
                response = client.models.generate_content(model=self.config.model, contents=prompt)
        except GeminiClientError:
            raise
        except Exception as exc:
            raise classify_gemini_exception(exc, api_key=self.config.api_key) from exc

        text = extract_text(response)
        if not text:
            raise GeminiClientError("Gemini returned an empty response.")
        return text
