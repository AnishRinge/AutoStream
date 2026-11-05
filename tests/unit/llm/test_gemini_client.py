import unittest
from unittest.mock import patch

from agent.llm.gemini_client import (
    DEFAULT_GEMINI_MODEL,
    ENV_API_KEY,
    ENV_MODEL,
    GeminiClient,
    GeminiClientError,
    _sanitize_error_message,
    classify_gemini_exception,
    extract_text,
)


class FakeModels:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return self.response


class FakeClient:
    def __init__(self, response=None, exc=None):
        self.models = FakeModels(response=response, exc=exc)


class TestGeminiClient(unittest.TestCase):
    def test_initialization_defaults_and_env(self):
        c = GeminiClient(client=FakeClient(response=type("R", (), {"text": "ok"})()))
        self.assertEqual(c.model, DEFAULT_GEMINI_MODEL)
        with patch.dict("os.environ", {ENV_API_KEY: "k", ENV_MODEL: "custom-model"}, clear=False):
            c2 = GeminiClient()
            self.assertEqual(c2.config.api_key, "k")
            self.assertEqual(c2.model, "custom-model")

    def test_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            c = GeminiClient(api_key=None)
            with self.assertRaises(GeminiClientError) as ctx:
                c.generate_text("hi")
        self.assertIn("missing API key", str(ctx.exception))
        self.assertNotIn("AIza", str(ctx.exception))

    def test_configurable_model_name(self):
        fake = FakeClient(response=type("R", (), {"text": "ok"})())
        c = GeminiClient(model="gemini-test", client=fake)
        c.generate_text("prompt")
        self.assertEqual(fake.models.calls[0]["model"], "gemini-test")

    def test_successful_generation_mocked(self):
        resp = type("R", (), {"text": " grounded answer "})()
        c = GeminiClient(client=FakeClient(response=resp))
        self.assertEqual(c.generate_text("prompt"), "grounded answer")

    def test_empty_malformed_response(self):
        c = GeminiClient(client=FakeClient(response=type("R", (), {"text": ""})()))
        with self.assertRaises(GeminiClientError):
            c.generate_text("prompt")

    def test_authentication_failure(self):
        err = classify_gemini_exception(Exception("401 unauthenticated"))
        self.assertIn("authentication", str(err).lower())

    def test_rate_limit_failure(self):
        err = classify_gemini_exception(Exception("429 RESOURCE_EXHAUSTED quota"))
        self.assertIn("rate limit", str(err).lower())

    def test_timeout_network_failure(self):
        self.assertIn("timed out", str(classify_gemini_exception(Exception("deadline timed out"))).lower())
        self.assertIn("unavailable", str(classify_gemini_exception(Exception("503 unavailable connection"))).lower())

    def test_api_key_never_in_errors_or_sanitized(self):
        secret = "AIzaSySECRET_TEST_KEY_1234567890"
        msg = _sanitize_error_message(f"failure with {secret} api_key={secret}", api_key=secret)
        self.assertNotIn(secret, msg)
        self.assertNotIn("AIzaSySECRET", msg)
        err = classify_gemini_exception(Exception(f"auth {secret}"), api_key=secret)
        self.assertNotIn(secret, str(err))

    def test_extract_text_candidates_fallback(self):
        part = type("P", (), {"text": "hello"})()
        content = type("C", (), {"parts": [part]})()
        cand = type("Cand", (), {"content": content})()
        resp = type("R", (), {"text": None, "candidates": [cand]})()
        self.assertEqual(extract_text(resp), "hello")


if __name__ == "__main__":
    unittest.main()
