import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.db.lead_repository import LeadRepository, LeadRepositoryError


class TestLeadRepository(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def db_path(self, *parts):
        return Path(self.tmp.name).joinpath(*parts)

    def lead(self, submission_id="sub-1", **overrides):
        data = {
            "submission_id": submission_id,
            "name": "Alex Rivera",
            "email": "alex@example.com",
            "platform": "YouTube",
        }
        data.update(overrides)
        return data

    def test_initialize_creates_parent_and_table(self):
        path = self.db_path("nested", "dir", "leads.db")
        repo = LeadRepository(path)
        self.assertFalse(path.exists())
        repo.initialize()
        self.assertTrue(path.exists())
        conn = sqlite3.connect(path)
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='leads'"
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(row)

    def test_save_and_retrieve(self):
        repo = LeadRepository(self.db_path("leads.db"))
        result = repo.save_confirmed_lead(self.lead())
        self.assertTrue(result.saved)
        self.assertFalse(result.duplicate)
        got = repo.get_lead_by_submission_id("sub-1")
        self.assertIsNotNone(got)
        self.assertEqual(got["submission_id"], "sub-1")
        self.assertEqual(got["name"], "Alex Rivera")
        self.assertEqual(got["email"], "alex@example.com")
        self.assertEqual(got["platform"], "YouTube")
        self.assertTrue(got["created_at"].endswith("Z"))
        self.assertEqual(len(repo.list_leads()), 1)

    def test_duplicate_submission_id_is_idempotent(self):
        repo = LeadRepository(self.db_path("leads.db"))
        first = repo.save_confirmed_lead(self.lead("dup-1"))
        second = repo.save_confirmed_lead(self.lead("dup-1", name="Other"))
        self.assertTrue(first.saved and not first.duplicate)
        self.assertTrue(second.saved and second.duplicate)
        self.assertEqual(len(repo.list_leads()), 1)
        # original row preserved
        self.assertEqual(repo.get_lead_by_submission_id("dup-1")["name"], "Alex Rivera")

    def test_missing_or_invalid_fields_rejected(self):
        repo = LeadRepository(self.db_path("leads.db"))
        cases = [
            {"submission_id": "", "name": "A", "email": "a@b.co", "platform": "YT"},
            {"submission_id": "s", "name": "", "email": "a@b.co", "platform": "YT"},
            {"submission_id": "s", "name": "A", "email": "", "platform": "YT"},
            {"submission_id": "s", "name": "A", "email": "a@b.co", "platform": ""},
            {"name": "A", "email": "a@b.co", "platform": "YT"},
        ]
        for lead in cases:
            with self.assertRaises(LeadRepositoryError):
                repo.save_confirmed_lead(lead)
        self.assertEqual(repo.list_leads(), [])

    def test_env_path_configuration(self):
        path = self.db_path("env", "leads.db")
        with patch.dict(os.environ, {"LEADS_DB_PATH": str(path)}, clear=False):
            repo = LeadRepository(None)
            self.assertEqual(os.path.normcase(str(repo.db_path)), os.path.normcase(str(path.absolute())))
            repo.save_confirmed_lead(self.lead("env-1"))
        self.assertTrue(path.exists())

    def test_database_errors_are_safe(self):
        repo = LeadRepository(self.db_path("leads.db"))
        repo.initialize()
        with patch.object(repo, "_connect", side_effect=sqlite3.Error("secret /tmp/path email@x")):
            with self.assertRaises(LeadRepositoryError) as ctx:
                repo.save_confirmed_lead(self.lead())
        self.assertNotIn("secret", str(ctx.exception))
        self.assertNotIn("/tmp", str(ctx.exception))

    def test_transaction_rolls_back_on_unexpected_error(self):
        repo = LeadRepository(self.db_path("leads.db"))
        repo.initialize()
        real_connect = repo._connect

        class FailCommitConn:
            def __init__(self, inner):
                self._inner = inner
                self.row_factory = inner.row_factory

            def __enter__(self):
                self._inner.__enter__()
                return self

            def __exit__(self, *a):
                return self._inner.__exit__(*a)

            def execute(self, *a, **k):
                sql = a[0] if a else ""
                if isinstance(sql, str) and sql.strip().upper() == "COMMIT":
                    raise sqlite3.OperationalError("boom")
                return self._inner.execute(*a, **k)

            def commit(self):
                return self._inner.commit()

            def rollback(self):
                return self._inner.rollback()

            def close(self):
                return self._inner.close()

        def connect():
            return FailCommitConn(real_connect())

        with patch.object(repo, "_connect", side_effect=connect):
            with self.assertRaises(LeadRepositoryError):
                repo.save_confirmed_lead(self.lead("tx-1"))
        self.assertIsNone(repo.get_lead_by_submission_id("tx-1"))
