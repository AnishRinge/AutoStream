import unittest
from unittest.mock import patch

from agent.handlers.lead import handle_lead
from agent.intent import detect_intent
from agent.state import (
    INTENT_BUSINESS_QUERY,
    INTENT_CANCEL,
    INTENT_CONFIRM,
    INTENT_LEAD_INTEREST,
    INTENT_UNKNOWN,
    LEAD_STATE_AWAITING_CONFIRMATION,
    LEAD_STATE_AWAITING_EMAIL,
    LEAD_STATE_AWAITING_NAME,
    LEAD_STATE_AWAITING_PLATFORM,
    LEAD_STATE_CANCELLED,
    LEAD_STATE_CONFIRMED,
    LEAD_STATE_IDLE,
)


def state(input_text, intent, **overrides):
    base = {
        "input": input_text,
        "intent": intent,
        "lead_state": LEAD_STATE_IDLE,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "confirmation_status": None,
    }
    base.update(overrides)
    return base


class TestLeadFSM(unittest.TestCase):
    def test_starts_collection(self):
        s = handle_lead(state("I want to sign up", INTENT_LEAD_INTEREST))
        self.assertEqual(s["lead_state"], LEAD_STATE_AWAITING_NAME)
        self.assertIn("name", s["response"].lower())

    def test_sequential_collection(self):
        s = handle_lead(state("sign up", INTENT_LEAD_INTEREST))
        s = handle_lead(state("Alex Rivera", INTENT_UNKNOWN, **{k: v for k, v in s.items() if k in {"lead_state", "lead_name", "lead_email", "lead_platform", "confirmation_status"}}))
        self.assertEqual(s["lead_name"], "Alex Rivera")
        self.assertEqual(s["lead_state"], LEAD_STATE_AWAITING_EMAIL)

        s = handle_lead(state("alex@example.com", INTENT_UNKNOWN, **{k: v for k, v in s.items() if k in {"lead_state", "lead_name", "lead_email", "lead_platform", "confirmation_status"}}))
        self.assertEqual(s["lead_email"], "alex@example.com")
        self.assertEqual(s["lead_state"], LEAD_STATE_AWAITING_PLATFORM)

        s = handle_lead(state("YouTube", INTENT_UNKNOWN, **{k: v for k, v in s.items() if k in {"lead_state", "lead_name", "lead_email", "lead_platform", "confirmation_status"}}))
        self.assertEqual(s["lead_platform"], "YouTube")
        self.assertEqual(s["lead_state"], LEAD_STATE_AWAITING_CONFIRMATION)
        self.assertIn("explicit", s["response"].lower())

    def test_invalid_email_retains_stage(self):
        s = handle_lead(
            state(
                "bad-email",
                INTENT_UNKNOWN,
                lead_state=LEAD_STATE_AWAITING_EMAIL,
                lead_name="Alex",
            )
        )
        self.assertEqual(s["lead_state"], LEAD_STATE_AWAITING_EMAIL)
        self.assertIsNone(s.get("lead_email"))
        self.assertIn("email", s["response"].lower())

    def test_correction_preserves_other_fields(self):
        s = handle_lead(
            state(
                "change my email to alex2@example.com",
                INTENT_CONFIRM,  # correction-like command routed via intent in graph; handler accepts text
                lead_state=LEAD_STATE_AWAITING_CONFIRMATION,
                lead_name="Alex",
                lead_email="old@example.com",
                lead_platform="YouTube",
            )
        )
        self.assertEqual(s["lead_email"], "alex2@example.com")
        self.assertEqual(s["lead_name"], "Alex")
        self.assertEqual(s["lead_platform"], "YouTube")
        self.assertEqual(s["lead_state"], LEAD_STATE_AWAITING_CONFIRMATION)

    def test_invalid_correction_does_not_mutate(self):
        s = handle_lead(
            state(
                "change my email to bad-email",
                INTENT_UNKNOWN,
                lead_state=LEAD_STATE_AWAITING_CONFIRMATION,
                lead_name="Alex",
                lead_email="old@example.com",
                lead_platform="YouTube",
            )
        )
        self.assertEqual(s["lead_email"], "old@example.com")
        self.assertEqual(s["lead_state"], LEAD_STATE_AWAITING_CONFIRMATION)
        self.assertIn("valid", s["response"].lower())

    def test_explicit_confirmation_required(self):
        base = dict(
            lead_state=LEAD_STATE_AWAITING_CONFIRMATION,
            lead_name="Alex",
            lead_email="alex@example.com",
            lead_platform="YouTube",
        )
        s = handle_lead(state("yes confirm", INTENT_CONFIRM, **base))
        self.assertEqual(s["lead_state"], LEAD_STATE_CONFIRMED)
        self.assertEqual(s["confirmation_status"], "confirmed")

    def test_negative_and_ambiguous_confirmation(self):
        base = dict(
            lead_state=LEAD_STATE_AWAITING_CONFIRMATION,
            lead_name="Alex",
            lead_email="alex@example.com",
            lead_platform="YouTube",
        )
        no = handle_lead(state("no", INTENT_UNKNOWN, **base))
        self.assertEqual(no["lead_state"], LEAD_STATE_AWAITING_CONFIRMATION)
        self.assertNotEqual(no["confirmation_status"], "confirmed")

        amb = handle_lead(state("maybe", INTENT_UNKNOWN, **base))
        self.assertEqual(amb["lead_state"], LEAD_STATE_AWAITING_CONFIRMATION)
        self.assertNotEqual(amb["confirmation_status"], "confirmed")
        self.assertIn("explicit", amb["response"].lower())

    def test_cancel_clears_active_fields(self):
        s = handle_lead(
            state(
                "cancel",
                INTENT_CANCEL if hasattr(__import__("agent.state"), "INTENT_CANCEL") else INTENT_UNKNOWN,
                lead_state=LEAD_STATE_AWAITING_PLATFORM,
                lead_name="Alex",
                lead_email="alex@example.com",
            )
        )
        self.assertEqual(s["lead_state"], LEAD_STATE_CANCELLED)
        self.assertIsNone(s.get("lead_name"))
        self.assertIsNone(s.get("lead_email"))
        self.assertIsNone(s.get("lead_platform"))
        self.assertEqual(s.get("confirmation_status"), "cancelled")

    def test_no_gemini_calls_in_lead_fsm(self):
        with patch("google.generativeai.configure", side_effect=AssertionError("Gemini called")) as c1, patch(
            "google.generativeai.GenerativeModel", side_effect=AssertionError("Gemini called")
        ) as c2:
            handle_lead(state("sign up", INTENT_LEAD_INTEREST))
            handle_lead(
                state(
                    "Alex",
                    INTENT_UNKNOWN,
                    lead_state=LEAD_STATE_AWAITING_NAME,
                )
            )
            handle_lead(
                state(
                    "yes confirm",
                    INTENT_CONFIRM if hasattr(__import__("agent.state"), "INTENT_CONFIRM") else INTENT_UNKNOWN,
                    lead_state=LEAD_STATE_AWAITING_CONFIRMATION,
                    lead_name="A",
                    lead_email="a@b.co",
                    lead_platform="YT",
                )
            )
        self.assertEqual(c1.call_count, 0)
        self.assertEqual(c2.call_count, 0)
