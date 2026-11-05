import unittest
from unittest.mock import patch

from agent.graph import build_graph
from agent.intent import detect_intent
from agent.state import (
    INTENT_CANCEL,
    LEAD_STATE_AWAITING_CONFIRMATION,
    LEAD_STATE_AWAITING_EMAIL,
    LEAD_STATE_AWAITING_NAME,
    LEAD_STATE_AWAITING_PLATFORM,
    LEAD_STATE_CANCELLED,
    LEAD_STATE_CONFIRMED,
    LEAD_STATE_IDLE,
    TRANSIENT_STATE_KEYS,
)


def build_turn_state(user_input, persisted_state):
    turn_state = {**persisted_state, "input": user_input}
    for key in TRANSIENT_STATE_KEYS:
        if key != "input":
            turn_state.pop(key, None)
    return turn_state


def extract_persisted_state(result_state):
    return {k: v for k, v in result_state.items() if k not in TRANSIENT_STATE_KEYS}


class TestLeadFSMIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = build_graph()

    def _turn(self, persisted, text):
        result = self.graph.invoke(build_turn_state(text, persisted))
        return result, extract_persisted_state(result)

    def test_full_collection_and_explicit_confirmation(self):
        p = {"lead_state": LEAD_STATE_IDLE}
        r, p = self._turn(p, "I want to sign up")
        self.assertEqual(p["lead_state"], LEAD_STATE_AWAITING_NAME)
        r, p = self._turn(p, "Alex Rivera")
        self.assertEqual(p["lead_name"], "Alex Rivera")
        self.assertEqual(p["lead_state"], LEAD_STATE_AWAITING_EMAIL)
        r, p = self._turn(p, "alex@example.com")
        self.assertEqual(p["lead_email"], "alex@example.com")
        self.assertEqual(p["lead_state"], LEAD_STATE_AWAITING_PLATFORM)
        r, p = self._turn(p, "YouTube")
        self.assertEqual(p["lead_platform"], "YouTube")
        self.assertEqual(p["lead_state"], LEAD_STATE_AWAITING_CONFIRMATION)
        r, p = self._turn(p, "yes, confirm")
        self.assertEqual(p["lead_state"], LEAD_STATE_CONFIRMED)
        self.assertEqual(p.get("confirmation_status"), "confirmed")

    def test_invalid_input_retains_stage(self):
        p = {"lead_state": LEAD_STATE_AWAITING_EMAIL, "lead_name": "Alex"}
        r, p = self._turn(p, "not-an-email")
        self.assertEqual(p["lead_state"], LEAD_STATE_AWAITING_EMAIL)
        self.assertIsNone(p.get("lead_email"))

    def test_corrections_update_only_target_field(self):
        p = {
            "lead_state": LEAD_STATE_AWAITING_CONFIRMATION,
            "lead_name": "Alex",
            "lead_email": "alex@example.com",
            "lead_platform": "YouTube",
        }
        r, p = self._turn(p, "change my platform to Instagram")
        self.assertEqual(p["lead_platform"], "Instagram")
        self.assertEqual(p["lead_name"], "Alex")
        self.assertEqual(p["lead_email"], "alex@example.com")
        self.assertEqual(p["lead_state"], LEAD_STATE_AWAITING_CONFIRMATION)

    def test_cancellation_at_every_stage(self):
        stages = [
            ({"lead_state": LEAD_STATE_AWAITING_NAME}, "cancel"),
            ({"lead_state": LEAD_STATE_AWAITING_EMAIL, "lead_name": "Alex"}, "cancel"),
            (
                {
                    "lead_state": LEAD_STATE_AWAITING_PLATFORM,
                    "lead_name": "Alex",
                    "lead_email": "a@b.co",
                },
                "cancel",
            ),
            (
                {
                    "lead_state": LEAD_STATE_AWAITING_CONFIRMATION,
                    "lead_name": "Alex",
                    "lead_email": "a@b.co",
                    "lead_platform": "YT",
                },
                "cancel",
            ),
        ]
        for persisted, text in stages:
            r, p = self._turn(persisted, text)
            self.assertEqual(p.get("lead_state"), LEAD_STATE_CANCELLED)
            self.assertIsNone(p.get("lead_name"))
            self.assertIsNone(p.get("lead_email"))
            self.assertIsNone(p.get("lead_platform"))
            self.assertEqual(p.get("confirmation_status"), "cancelled")

    def test_interruptions_at_every_stage_resume(self):
        # name stage interruption
        p = {"lead_state": LEAD_STATE_AWAITING_NAME}
        r, p = self._turn(p, "What is the refund policy?")
        self.assertIn("No refunds after 7 days", r["response"])
        self.assertEqual(p["lead_state"], LEAD_STATE_AWAITING_NAME)
        r, p = self._turn(p, "Alex")
        self.assertEqual(p["lead_name"], "Alex")

        # email interruption
        r, p = self._turn(p, "How much is Pro?")
        self.assertIn("$79", r["response"])
        self.assertEqual(p["lead_state"], LEAD_STATE_AWAITING_EMAIL)
        r, p = self._turn(p, "alex@example.com")
        self.assertEqual(p["lead_email"], "alex@example.com")

        # platform interruption
        r, p = self._turn(p, "Do you offer support?")
        self.assertIn("24/7", r["response"])
        self.assertEqual(p["lead_state"], LEAD_STATE_AWAITING_PLATFORM)
        r, p = self._turn(p, "YouTube")
        self.assertEqual(p["lead_platform"], "YouTube")

        # confirmation interruption
        self.assertEqual(p["lead_state"], LEAD_STATE_AWAITING_CONFIRMATION)
        r, p = self._turn(p, "What plans do you have?")
        self.assertTrue("Basic" in r["response"] or "Pro" in r["response"])
        self.assertEqual(p["lead_state"], LEAD_STATE_AWAITING_CONFIRMATION)
        r, p = self._turn(p, "yes confirm")
        self.assertEqual(p["lead_state"], LEAD_STATE_CONFIRMED)

    def test_explicit_negative_and_ambiguous_confirmation(self):
        base = {
            "lead_state": LEAD_STATE_AWAITING_CONFIRMATION,
            "lead_name": "Alex",
            "lead_email": "alex@example.com",
            "lead_platform": "YouTube",
        }
        r, p = self._turn(dict(base), "no, not correct")
        self.assertNotEqual(p.get("lead_state"), LEAD_STATE_CONFIRMED)
        r, p = self._turn(dict(base), "maybe later")
        self.assertNotEqual(p.get("lead_state"), LEAD_STATE_CONFIRMED)
        self.assertEqual(p.get("lead_state"), LEAD_STATE_AWAITING_CONFIRMATION)

    def test_new_lead_after_cancellation(self):
        p = {"lead_state": LEAD_STATE_AWAITING_EMAIL, "lead_name": "Alex"}
        r, p = self._turn(p, "cancel")
        self.assertEqual(p["lead_state"], LEAD_STATE_CANCELLED)
        r, p = self._turn(p, "I want to sign up")
        self.assertEqual(p["lead_state"], LEAD_STATE_AWAITING_NAME)
        self.assertIsNone(p.get("lead_name"))

    def test_persistence_across_turns(self):
        p = {}
        for text in ["sign up", "Alex", "alex@example.com", "YouTube"]:
            _, p = self._turn(p, text)
        self.assertEqual(p["lead_name"], "Alex")
        self.assertEqual(p["lead_email"], "alex@example.com")
        self.assertEqual(p["lead_platform"], "YouTube")
        self.assertEqual(p["lead_state"], LEAD_STATE_AWAITING_CONFIRMATION)

    def test_no_gemini_calls_any_lead_fsm_operation(self):
        with patch("google.generativeai.configure", side_effect=AssertionError("Gemini called")), patch(
            "google.generativeai.GenerativeModel", side_effect=AssertionError("Gemini called")
        ):
            p = {}
            for text in [
                "sign up",
                "Alex",
                "alex@example.com",
                "YouTube",
                "change my email to alex2@example.com",
                "yes confirm",
            ]:
                _, p = self._turn(p, text)
        self.assertEqual(p.get("lead_state"), LEAD_STATE_CONFIRMED)
