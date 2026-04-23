from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator


def adapt_json(value: dict | list) -> str:
    return json.dumps(value)


sqlite3.register_adapter(dict, adapt_json)
sqlite3.register_adapter(list, adapt_json)


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS candidate_profile (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    edited_draft TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id)
                );
                """
            )

    def is_writable(self) -> bool:
        try:
            with self.connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS readiness_probe (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        checked_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO readiness_probe (id, checked_at)
                    VALUES (1, ?)
                    ON CONFLICT(id) DO UPDATE SET checked_at = excluded.checked_at
                    """,
                    (datetime.utcnow().isoformat(),),
                )
            return True
        except sqlite3.Error:
            return False

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
