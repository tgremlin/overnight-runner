"""SQLite persistence (Phase 2).

Schema: tasks, runs, events. WAL + synchronous=FULL + busy_timeout.
Short BEGIN IMMEDIATE transactions only. NO long-running writes.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .schemas import TaskStatus


SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    manifest_sha256 TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 100,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    approved_at INTEGER,
    approved_by TEXT,
    approved_manifest_sha256 TEXT,
    approved_repo_head TEXT,
    approved_runtime_sha256 TEXT,
    approved_model_digest TEXT,
    final_reason_code TEXT,
    final_reason_text TEXT
);
CREATE INDEX IF NOT EXISTS tasks_status ON tasks(status);

CREATE TABLE IF NOT EXISTS runs (
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
CREATE INDEX IF NOT EXISTS runs_task ON runs(task_id);
CREATE INDEX IF NOT EXISTS runs_status ON runs(status);

CREATE TABLE IF NOT EXISTS events (
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
CREATE INDEX IF NOT EXISTS events_task ON events(task_id);
CREATE INDEX IF NOT EXISTS events_type ON events(event_type);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), isolation_level=None, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = FULL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    @contextmanager
    def transaction(self):
        """Short IMMEDIATE transaction."""
        cur = self._conn.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE")
            yield cur
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    # ---------------- Task ops ----------------

    def upsert_task(
        self,
        task_id: str,
        manifest_sha256: str,
        manifest_json: str,
        status: TaskStatus,
        priority: int = 100,
    ) -> None:
        now = int(time.time())
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO tasks (task_id, manifest_sha256, manifest_json, status, priority, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    manifest_sha256=excluded.manifest_sha256,
                    manifest_json=excluded.manifest_json,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (task_id, manifest_sha256, manifest_json, status.value, priority, now, now),
            )

    def update_status(self, task_id: str, status: TaskStatus, **fields: Any) -> None:
        sets = ["status=?", "updated_at=?"]
        vals: list[Any] = [status.value, int(time.time())]
        for k, v in fields.items():
            sets.append(f"{k}=?")
            vals.append(v)
        vals.append(task_id)
        with self.transaction() as cur:
            cur.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE task_id=?", vals)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        cur = self._conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def list_tasks(self, status: TaskStatus | None = None) -> list[dict[str, Any]]:
        if status:
            cur = self._conn.execute("SELECT * FROM tasks WHERE status=? ORDER BY priority ASC, created_at ASC", (status.value,))
        else:
            cur = self._conn.execute("SELECT * FROM tasks ORDER BY priority ASC, created_at ASC")
        return [dict(r) for r in cur.fetchall()]

    def approve_task(self, task_id: str, *, approved_by: str, approval_envelope: dict[str, Any]) -> None:
        with self.transaction() as cur:
            cur.execute(
                """
                UPDATE tasks SET
                    status=?,
                    approved_at=?,
                    approved_by=?,
                    approved_manifest_sha256=?,
                    approved_repo_head=?,
                    approved_runtime_sha256=?,
                    approved_model_digest=?,
                    updated_at=?
                WHERE task_id=?
                """,
                (
                    TaskStatus.APPROVED.value,
                    int(time.time()),
                    approved_by,
                    approval_envelope.get("manifest_sha256"),
                    approval_envelope.get("approved_repo_head"),
                    approval_envelope.get("approved_runtime_sha256"),
                    approval_envelope.get("approved_model_digest"),
                    int(time.time()),
                    task_id,
                ),
            )

    # ---------------- Event ops ----------------

    def emit_event(
        self,
        session_id: str,
        event_type: str,
        *,
        task_id: str | None = None,
        run_id: str | None = None,
        from_state: str | None = None,
        to_state: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO events (timestamp, session_id, task_id, run_id, event_type, from_state, to_state, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(time.time()),
                session_id,
                task_id,
                run_id,
                event_type,
                from_state,
                to_state,
                json.dumps(details or {}),
            ),
        )

    # ---------------- Morning summary ----------------

    def summary(self) -> dict[str, Any]:
        cur = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status"
        )
        counts = {r["status"]: r["n"] for r in cur.fetchall()}
        return {"task_counts": counts}


def default_db_path() -> Path:
    from .runtime import state_dir
    return state_dir() / "state.db"
