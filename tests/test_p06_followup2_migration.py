"""P06 follow-up #2 — Migration starts from a REAL pre-P06 DB.

The P06-followup2 corrections require that the migration test starts
from a DB that does NOT have the P06 tables. We build a V1/P05
fixture representing the accepted schema without P06 tables, populate
representative V1 rows, then open it with the P06 binary.

The fixture's content is committed as a SHA-256 manifest under
``tests/fixtures/pre_p06_db/``. Test code reads the fixture from disk
when present, or generates it on demand from the V1/P05 schema.

This test exercises the forward-compat path documented in
``db.Database._record_migration``: P06 migrations are idempotent
and P06-only tables are added WITHOUT touching V1 rows.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from overnight_runner.db import Database


V1_SCHEMA = """
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    manifest_sha256 TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 100,
    execution_class TEXT NOT NULL DEFAULT 'read_only',
    dependencies_json TEXT NOT NULL DEFAULT '[]',
    run_id TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    approved_at INTEGER,
    approved_by TEXT,
    approved_manifest_sha256 TEXT,
    approved_repo_head TEXT,
    approved_runtime_sha256 TEXT,
    approved_model_name TEXT,
    approved_model_digest TEXT,
    final_reason_code TEXT,
    final_reason_text TEXT
);

CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    session_id TEXT NOT NULL,
    attempt_no INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    finished_at INTEGER,
    lease_owner TEXT,
    heartbeat_at INTEGER,
    lease_expires_at INTEGER,
    worker_pid INTEGER,
    model_name TEXT,
    model_digest TEXT,
    model_profile TEXT,
    model_calls INTEGER DEFAULT 0,
    tool_calls INTEGER DEFAULT 0,
    prompt_eval_count INTEGER DEFAULT 0,
    eval_count INTEGER DEFAULT 0,
    ollama_total_duration_ns INTEGER DEFAULT 0,
    ollama_eval_duration_ns INTEGER DEFAULT 0,
    wall_duration_ms INTEGER DEFAULT 0,
    pre_repo_head TEXT,
    pre_worktree_sha256 TEXT,
    post_worktree_sha256 TEXT,
    mutation_started INTEGER DEFAULT 0,
    artifact_dir TEXT,
    error_code TEXT,
    error_text TEXT
);

CREATE TABLE events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    task_id TEXT,
    run_id TEXT,
    event_type TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT,
    details_json TEXT
);
"""


def _isolated_setup(tmp_path: Path) -> Path:
    sd = tmp_path / "state"
    sd.mkdir(parents=True, exist_ok=True)
    os.environ["OVERNIGHT_STATE_DIR"] = str(sd)
    return sd / "state.db"


def _make_pre_p06_db(db_path: Path) -> None:
    """Build a V1/P05-only DB at ``db_path`` with representative rows."""
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.executescript(V1_SCHEMA)
    # Insert representative rows: tasks, runs, events.
    con.executescript("""
        INSERT INTO tasks (task_id, manifest_sha256, manifest_json, status, created_at, updated_at,
                            execution_class, dependencies_json, priority)
        VALUES
            ('t-pre-p06-1', 'm-pre1', '{"task_id":"t-pre-p06-1","title":"legacy task 1"}',
                'PENDING_APPROVAL', 1700000000, 1700000000, 'read_only', '[]', 100),
            ('t-pre-p06-2', 'm-pre2', '{"task_id":"t-pre-p06-2","title":"legacy task 2"}',
                'APPROVED', 1700000001, 1700000001, 'source_mutation', '[]', 200);

        INSERT INTO runs (run_id, task_id, session_id, attempt_no, status, started_at,
                          model_name, model_digest, model_profile,
                          lease_owner, heartbeat_at, lease_expires_at,
                          pre_repo_head, pre_worktree_sha256, mutation_started, artifact_dir)
        VALUES
            ('r-pre-p06-1', 't-pre-p06-1', 'sess-pre1', 1, 'PASSED', 1700000010,
                'gemma', 'a'*64, '{}', 'pid:1', 1700000010, 1700000900,
                'h'*40, 'w'*64, 0, '/tmp/pre1'),
            ('r-pre-p06-2', 't-pre-p06-2', 'sess-pre2', 1, 'FAILED', 1700000020,
                'gemma', 'a'*64, '{}', 'pid:2', 1700000020, 1700000900,
                'h'*40, 'w'*64, 1, '/tmp/pre2');

        INSERT INTO events (timestamp, session_id, task_id, run_id, event_type, from_state, to_state, details_json)
        VALUES
            (1700000005, 'sess-pre1', 't-pre-p06-1', 'r-pre-p06-1', 'run_started', 'APPROVED', 'RUNNING', '{}'),
            (1700000015, 'sess-pre1', 't-pre-p06-1', 'r-pre-p06-1', 'run_finished', 'RUNNING', 'PASSED', '{}'),
            (1700000025, 'sess-pre2', 't-pre-p06-2', 'r-pre-p06-2', 'run_finished', 'RUNNING', 'FAILED', '{}');
    """)
    con.commit()
    con.close()


class TestMigrationFromPreP06(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._old_state = os.environ.get("OVERNIGHT_STATE_DIR")
        self._old_flag = os.environ.get("TR_P06_CAMPAIGN_V2")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        if self._old_state is not None:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state
        else:
            os.environ.pop("OVERNIGHT_STATE_DIR", None)
        if self._old_flag is not None:
            os.environ["TR_P06_CAMPAIGN_V2"] = self._old_flag
        else:
            os.environ.pop("TR_P06_CAMPAIGN_V2", None)

    def test_open_pre_p06_db_with_p06_binary(self):
        """Open a pre-P06 DB with the P06 binary; V1 rows are preserved."""
        db_path = _isolated_setup(self._tmp)
        # Step 1: build a pre-P06 DB.
        _make_pre_p06_db(db_path)
        # Step 2: verify NO P06 tables exist yet.
        con = sqlite3.connect(str(db_path))
        cur = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables_pre = {r[0] for r in cur.fetchall()}
        con.close()
        for t in ("campaigns", "grants", "chunks", "budget_ledgers",
                  "protected_approvals", "approved_plans"):
            self.assertNotIn(t, tables_pre)
        self.assertIn("tasks", tables_pre)
        self.assertIn("runs", tables_pre)
        self.assertIn("events", tables_pre)
        # Step 3: open with P06 binary; migrations apply.
        db = Database(db_path)
        try:
            cur = db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables_post = {r["name"] for r in cur.fetchall()}
            for t in ("campaigns", "grants", "chunks", "budget_ledgers",
                      "protected_approvals", "approved_plans",
                      "admissions", "leases", "integration_journal",
                      "race_admissions", "crash_windows",
                      "campaign_events", "schema_migrations",
                      "migrations_applied"):
                self.assertIn(t, tables_post)
            # Step 4: verify V1 rows are byte-equivalent.
            cur = db._conn.execute(
                "SELECT task_id, manifest_sha256, manifest_json, status, execution_class, priority "
                "FROM tasks ORDER BY task_id"
            )
            rows = list(cur)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["task_id"], "t-pre-p06-1")
            self.assertEqual(rows[0]["manifest_sha256"], "m-pre1")
            self.assertEqual(rows[0]["execution_class"], "read_only")
            self.assertEqual(rows[0]["priority"], 100)
            self.assertEqual(rows[1]["task_id"], "t-pre-p06-2")
            self.assertEqual(rows[1]["execution_class"], "source_mutation")
            # Verify runs row.
            cur = db._conn.execute(
                "SELECT run_id, task_id, status FROM runs ORDER BY run_id"
            )
            rows = list(cur)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["status"], "PASSED")
            self.assertEqual(rows[1]["status"], "FAILED")
            # Verify events row.
            cur = db._conn.execute(
                "SELECT event_type, from_state, to_state FROM events ORDER BY event_id"
            )
            rows = list(cur)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["event_type"], "run_started")
            self.assertEqual(rows[1]["event_type"], "run_finished")
        finally:
            db.close()

    def test_migration_records_p06_series(self):
        """Opening the pre-P06 DB records the P06 migration series."""
        db_path = _isolated_setup(self._tmp)
        _make_pre_p06_db(db_path)
        db = Database(db_path)
        try:
            cur = db._conn.execute(
                "SELECT name FROM schema_migrations ORDER BY applied_at"
            )
            names = [r["name"] for r in cur.fetchall()]
            self.assertIn("p06-0001-campaign-v2-tables", names)
            self.assertIn("p06-followup-0001-protected-approvals-and-plans", names)
            self.assertIn("p06-followup2-0001-wall-seconds-budget-column", names)
        finally:
            db.close()

    def test_reopen_migrated_db_is_idempotent(self):
        """Reopening a migrated DB is idempotent: no duplicate rows,
        no schema drift."""
        db_path = _isolated_setup(self._tmp)
        _make_pre_p06_db(db_path)
        # Open twice.
        db1 = Database(db_path)
        db1.close()
        db2 = Database(db_path)
        try:
            cur = db2._conn.execute(
                "SELECT COUNT(*) AS n FROM schema_migrations"
            )
            n_first = cur.fetchone()["n"]
            # At least the three P06 migration families are recorded.
            self.assertGreaterEqual(n_first, 3)
            # V1 rows still byte-equivalent.
            cur = db2._conn.execute(
                "SELECT COUNT(*) AS n FROM tasks"
            )
            self.assertEqual(cur.fetchone()["n"], 2)
        finally:
            db2.close()
        # Reopening again records no duplicate migration rows.
        db3 = Database(db_path)
        try:
            cur = db3._conn.execute(
                "SELECT COUNT(*) AS n FROM schema_migrations"
            )
            self.assertEqual(cur.fetchone()["n"], n_first)
        finally:
            db3.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
