import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.db.lead_repository import LeadRepository, LeadRepositoryError
from agent.state import (
    LEAD_STATE_AWAITING_CONFIRMATION,
    LEAD_STATE_AWAITING_EMAIL,
    LEAD_STATE_AWAITING_NAME,
    LEAD_STATE_CANCELLED,
    LEAD_STATE_CONFIRMED,
    LEAD_STATE_IDLE,
    TRANSIENT_STATE_KEYS,
)
import app
from agent.graph import build_graph


def build_turn_state(user_input, persisted_state):
    return app.build_turn_state(user_input, persisted_state)


def extract_persisted_state(result_state):
    return app.extract_persisted_state(result_state)


class TestLeadPersistenceIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = build_graph()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "leads.db"
        self.repo = LeadRepository(self.db_path)

    def _turn(self, persisted, text):
        result = self.graph.invoke(build_turn_state(text, persisted))
        state = extract_persisted_state(result)
        app.persist_confirmed_lead_if_needed(state, repository=self.repo)
        return result, state

    def _confirm_flow(self):
        p = {}
        for text in ["sign up", "Alex Rivera", "alex@example.com", "YouTube", "yes confirm"]:
            _, p = self._turn(p, text)
        return p

    def test_confirmed_lead_saved_once(self):
        p = self._confirm_flow()
        self.assertEqual(p.get("lead_state"), LEAD_STATE_CONFIRMED)
        self.assertTrue(p.get("lead_persisted"))
        self.assertTrue(p.get("lead_submission_id"))
        leads = self.repo.list_leads()
        self.assertEqual(len(leads), 1)
        self.assertEqual(leads[0]["name"], "Alex Rivera")
        self.assertEqual(leads[0]["email"], "alex@example.com")
        self.assertEqual(leads[0]["platform"], "YouTube")
        self.assertEqual(leads[0]["submission_id"], p["lead_submission_id"])

    def test_unconfirmed_lead_not_saved(self):
        p = {}
        for text in ["sign up", "Alex", "alex@example.com", "YouTube"]:
            _, p = self._turn(p, text)
        self.assertEqual(p.get("lead_state"), LEAD_STATE_AWAITING_CONFIRMATION)
        self.assertEqual(self.repo.list_leads(), [])

    def test_cancelled_lead_not_saved(self):
        p = {"lead_state": LEAD_STATE_AWAITING_EMAIL, "lead_name": "Alex"}
        _, p = self._turn(p, "cancel")
        self.assertEqual(p.get("lead_state"), LEAD_STATE_CANCELLED)
        self.assertEqual(self.repo.list_leads(), [])

    def test_incomplete_lead_not_saved(self):
        p = {"lead_state": LEAD_STATE_CONFIRMED, "lead_name": "Alex", "lead_email": None, "lead_platform": "YT", "lead_submission_id": "x"}
        app.persist_confirmed_lead_if_needed(p, repository=self.repo)
        self.assertEqual(self.repo.list_leads(), [])

    def test_business_interruption_does_not_trigger_persistence(self):
        p = {"lead_state": LEAD_STATE_AWAITING_NAME}
        _, p = self._turn(p, "What is the refund policy?")
        self.assertEqual(p.get("lead_state"), LEAD_STATE_AWAITING_NAME)
        self.assertEqual(self.repo.list_leads(), [])

    def test_repeated_confirmation_does_not_duplicate(self):
        p = self._confirm_flow()
        submission_id = p["lead_submission_id"]
        # simulate repeated turn while already confirmed/persisted context
        _, p = self._turn(p, "yes confirm")
        self.assertEqual(len(self.repo.list_leads()), 1)
        self.assertEqual(p.get("lead_submission_id"), submission_id)

    def test_database_failure_preserves_confirmed_data(self):
        with patch.object(self.repo, "save_confirmed_lead", side_effect=LeadRepositoryError("db down")):
            p = self._confirm_flow()
        self.assertEqual(p.get("lead_state"), LEAD_STATE_CONFIRMED)
        self.assertEqual(p.get("lead_name"), "Alex Rivera")
        self.assertEqual(p.get("lead_email"), "alex@example.com")
        self.assertEqual(p.get("lead_platform"), "YouTube")
        self.assertTrue(p.get("lead_submission_id"))
        self.assertFalse(p.get("lead_persisted"))
        self.assertIn("couldn't save", p.get("response", "").lower())

    def test_retry_after_database_failure_succeeds(self):
        with patch.object(self.repo, "save_confirmed_lead", side_effect=LeadRepositoryError("db down")):
            p = self._confirm_flow()
        submission_id = p["lead_submission_id"]
        # retry persistence without recollecting
        app.persist_confirmed_lead_if_needed(p, repository=self.repo)
        self.assertTrue(p.get("lead_persisted"))
        self.assertEqual(p.get("lead_submission_id"), submission_id)
        self.assertEqual(p.get("lead_name"), "Alex Rivera")
        leads = self.repo.list_leads()
        self.assertEqual(len(leads), 1)
        self.assertEqual(leads[0]["submission_id"], submission_id)

    def test_successful_retry_marks_persisted(self):
        p = self._confirm_flow()
        self.assertTrue(p.get("lead_persisted"))
        self.assertIn("saved", p.get("response", "").lower())

    def test_existing_lead_fsm_behavior_unchanged(self):
        # quick regression: full collection reaches confirmation and confirms
        p = {}
        for text in ["sign up", "Alex", "alex@example.com", "YouTube"]:
            _, p = self._turn(p, text)
        self.assertEqual(p["lead_state"], LEAD_STATE_AWAITING_CONFIRMATION)
        _, p = self._turn(p, "yes confirm")
        self.assertEqual(p["lead_state"], LEAD_STATE_CONFIRMED)
        self.assertTrue(p.get("lead_persisted"))
