from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("data") / "leads.db"
ENV_DB_PATH = "LEADS_DB_PATH"


class LeadRepositoryError(Exception):
    """Safe repository error (no sensitive data)."""


@dataclass(frozen=True)
class SaveResult:
    saved: bool
    duplicate: bool = False


def resolve_db_path(db_path: str | os.PathLike[str] | None = None) -> Path:
    raw = db_path if db_path is not None else os.environ.get(ENV_DB_PATH)
    path = Path(raw) if raw else DEFAULT_DB_PATH
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        # Normalize without requiring the file to exist.
        try:
            path = Path(os.path.normpath(str(path.absolute())))
        except OSError:
            path = Path(os.path.normpath(str(path)))
    return path


class LeadRepository:
    """SQLite repository for confirmed leads. Uses parameterized SQL only."""

    def __init__(self, db_path: str | os.PathLike[str] | None = None):
        self.db_path = resolve_db_path(db_path)

    def _connect(self) -> sqlite3.Connection:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=10)
            conn.row_factory = sqlite3.Row
            return conn
        except OSError as exc:
            raise LeadRepositoryError("Could not initialize the lead database.") from exc
        except sqlite3.Error as exc:
            raise LeadRepositoryError("Could not connect to the lead database.") from exc

    def initialize(self) -> None:
        conn = None
        try:
            conn = self._connect()
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS leads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        except LeadRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise LeadRepositoryError("Could not initialize the lead database.") from exc
        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    def _normalize_lead(lead: dict[str, Any]) -> dict[str, str]:
        required = ("submission_id", "name", "email", "platform")
        normalized: dict[str, str] = {}
        for key in required:
            value = lead.get(key)
            if value is None or not str(value).strip():
                raise LeadRepositoryError("Lead data is incomplete.")
            normalized[key] = str(value).strip()
        return normalized

    def save_confirmed_lead(self, lead: dict[str, Any]) -> SaveResult:
        data = self._normalize_lead(lead)
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        try:
            self.initialize()
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT INTO leads (submission_id, name, email, platform, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        data["submission_id"],
                        data["name"],
                        data["email"],
                        data["platform"],
                        created_at,
                    ),
                )
                conn.execute("COMMIT")
                return SaveResult(saved=True, duplicate=False)
            except sqlite3.IntegrityError:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                row = conn.execute(
                    "SELECT 1 FROM leads WHERE submission_id = ? LIMIT 1",
                    (data["submission_id"],),
                ).fetchone()
                if row:
                    return SaveResult(saved=True, duplicate=True)
                raise LeadRepositoryError("Could not save the lead submission.")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()
        except LeadRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise LeadRepositoryError("Could not save the lead submission.") from exc

    def get_lead_by_submission_id(self, submission_id: str) -> dict[str, Any] | None:
        if not submission_id or not str(submission_id).strip():
            return None
        try:
            self.initialize()
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT id, submission_id, name, email, platform, created_at
                    FROM leads WHERE submission_id = ?
                    """,
                    (str(submission_id).strip(),),
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()
        except LeadRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise LeadRepositoryError("Could not read the lead database.") from exc

    def list_leads(self) -> list[dict[str, Any]]:
        try:
            self.initialize()
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT id, submission_id, name, email, platform, created_at FROM leads ORDER BY id"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()
        except LeadRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise LeadRepositoryError("Could not read the lead database.") from exc
