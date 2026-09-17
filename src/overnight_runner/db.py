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
CREATE INDEX IF NOT EXISTS tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS tasks_priority ON tasks(priority, created_at);

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
        execution_class: str = "read_only",
        dependencies_json: str = "[]",
    ) -> None:
        now = int(time.time())
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO tasks (
                    task_id, manifest_sha256, manifest_json, status, priority,
                    execution_class, dependencies_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    manifest_sha256=excluded.manifest_sha256,
                    manifest_json=excluded.manifest_json,
                    status=excluded.status,
                    execution_class=COALESCE(NULLIF(excluded.execution_class, ''), tasks.execution_class),
                    dependencies_json=COALESCE(NULLIF(excluded.dependencies_json, ''), tasks.dependencies_json),
                    updated_at=excluded.updated_at
                """,
                (task_id, manifest_sha256, manifest_json, status.value, priority,
                 execution_class, dependencies_json, now, now),
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
                    approved_model_name=?,
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
                    approval_envelope.get("approved_model_name"),
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

    def find_stale_runs(self, now: int) -> list[dict[str, Any]]:
        """Find RUNNING runs whose lease has expired (worker disappeared).

        Strict semantics: lease_expires_at < now means stale. The lease window
        already provides the grace interval. No extra hidden grace.

        Mutation tasks MUST NOT auto-retry; they transition to REVIEW_REQUIRED.
        Read-only tasks MAY be retried (the caller decides).
        """
        cur = self._conn.execute(
            """
            SELECT r.* FROM runs r
            WHERE r.status='RUNNING'
              AND (
                r.lease_expires_at IS NULL
                OR r.lease_expires_at < ?
              )
            """,
            (now,),
        )
        return [dict(r) for r in cur.fetchall()]

    def find_orphan_running_tasks(self) -> list[dict[str, Any]]:
        """Find tasks stuck in RUNNING with no active RUNNING runs row.

        This is a legacy/edge-case safety net. The atomic claim in
        `claim_next_approved` should make this impossible going forward.
        """
        cur = self._conn.execute(
            """
            SELECT t.* FROM tasks t
            WHERE t.status='RUNNING'
              AND NOT EXISTS (
                SELECT 1 FROM runs r
                WHERE r.task_id = t.task_id AND r.status='RUNNING'
              )
            """
        )
        return [dict(r) for r in cur.fetchall()]

    def claim_next_approved(
        self,
        *,
        run_id: str,
        session_id: str,
        attempt_no: int,
        worker_pid: int,
        model_name: str,
        model_digest: str | None,
        model_profile_json: str,
        now: int,
        lease_expires_at: int,
        artifact_dir: str,
        execution_class_filter: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Atomically: pick eligible APPROVED task, create RUNNING runs row,
        transition task APPROVED -> RUNNING.

        Returns the joined claim record (task row + run row fields) or None
        if nothing eligible. Eligibility = status=APPROVED AND all declared
        dependencies have status=PASSED.
        """
        where_extra = ""
        params: list[Any] = []
        if execution_class_filter:
            placeholders = ",".join("?" for _ in execution_class_filter)
            where_extra = f" AND t.execution_class IN ({placeholders})"
            params.extend(execution_class_filter)
        with self.transaction() as cur:
            cur.execute(
                f"""
                SELECT t.* FROM tasks t
                WHERE t.status='APPROVED'
                  AND NOT EXISTS (
                    SELECT 1 FROM json_each(t.dependencies_json) d
                    WHERE d.required_state='PASSED'
                      AND NOT EXISTS (
                        SELECT 1 FROM tasks dep
                        WHERE dep.task_id = d.task_id AND dep.status='PASSED'
                      )
                  )
                  {where_extra}
                ORDER BY t.priority ASC, t.created_at ASC
                LIMIT 1
                """,
                tuple(params),
            )
            row = cur.fetchone()
            if row is None:
                return None
            task = dict(row)
            # Transition task RUNNING.
            cur.execute(
                "UPDATE tasks SET status=?, updated_at=? WHERE task_id=? AND status='APPROVED'",
                (TaskStatus.RUNNING.value, now, task["task_id"]),
            )
            if cur.rowcount != 1:
                return None  # Lost the race.
            # Insert RUNNING runs row.
            cur.execute(
                """
                INSERT INTO runs (
                    run_id, task_id, session_id, attempt_no, status, started_at,
                    worker_pid, model_name, model_digest, model_profile,
                    lease_owner, heartbeat_at, lease_expires_at,
                    pre_repo_head, pre_worktree_sha256, mutation_started, artifact_dir
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id, task["task_id"], session_id, attempt_no, "RUNNING", now,
                    worker_pid, model_name, model_digest, model_profile_json,
                    f"pid:{worker_pid}", now, lease_expires_at,
                    task["approved_repo_head"], "", 0, artifact_dir,
                ),
            )
            cur.execute(
                "UPDATE tasks SET run_id=? WHERE task_id=?",
                (run_id, task["task_id"]),
            )
        return {"task": task, "run_id": run_id, "attempt_no": attempt_no}

    def increment_attempt_for_task(self, task_id: str) -> int:
        """Return next attempt_no for this task (1, 2, ...)."""
        cur = self._conn.execute(
            "SELECT COALESCE(MAX(attempt_no), 0) AS m FROM runs WHERE task_id=?",
            (task_id,),
        )
        return int(cur.fetchone()["m"]) + 1

    def get_dep_blocked_approved(self) -> list[dict[str, Any]]:
        """List APPROVED tasks with at least one unmet dependency."""
        cur = self._conn.execute(
            """
            SELECT t.task_id,
                   (SELECT GROUP_CONCAT(json_extract(d.value, '$.task_id'))
                      FROM json_each(t.dependencies_json) d
                    WHERE json_extract(d.value, '$.required_state')='PASSED'
                      AND NOT EXISTS (
                        SELECT 1 FROM tasks dep
                        WHERE dep.task_id = json_extract(d.value, '$.task_id')
                          AND dep.status='PASSED'
                      )) AS blocked_by
            FROM tasks t
            WHERE t.status='APPROVED'
              AND EXISTS (
                SELECT 1 FROM json_each(t.dependencies_json) d
                WHERE json_extract(d.value, '$.required_state')='PASSED'
                  AND NOT EXISTS (
                    SELECT 1 FROM tasks dep
                    WHERE dep.task_id = json_extract(d.value, '$.task_id')
                      AND dep.status='PASSED'
                  )
              )
            """
        )
        return [dict(r) for r in cur.fetchall()]

    def insert_run(
        self,
        run_id: str,
        task_id: str,
        session_id: str,
        attempt_no: int,
        status: str,
        started_at: int,
        *,
        worker_pid: int,
        model_name: str,
        model_digest: str | None,
        model_profile: str,
        lease_expires_at: int,
        pre_repo_head: str,
        pre_worktree_sha256: str,
        artifact_dir: str,
        mutation_started: bool = False,
    ) -> None:
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO runs (
                    run_id, task_id, session_id, attempt_no, status, started_at,
                    worker_pid, model_name, model_digest, model_profile,
                    lease_owner, heartbeat_at, lease_expires_at,
                    pre_repo_head, pre_worktree_sha256, mutation_started, artifact_dir
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id, task_id, session_id, attempt_no, status, started_at,
                    worker_pid, model_name, model_digest, model_profile,
                    f"pid:{worker_pid}", started_at, lease_expires_at,
                    pre_repo_head, pre_worktree_sha256, int(mutation_started), artifact_dir,
                ),
            )

    def heartbeat(self, run_id: str, now: int, lease_expires_at: int,
                  *, prompt_eval_count: int = 0, eval_count: int = 0,
                  total_ns: int = 0, eval_ns: int = 0,
                  model_calls: int | None = None, tool_calls: int | None = None) -> None:
        sets = ["heartbeat_at=?", "lease_expires_at=?"]
        vals: list[Any] = [now, lease_expires_at]
        for k, v in (("prompt_eval_count", prompt_eval_count), ("eval_count", eval_count),
                     ("ollama_total_duration_ns", total_ns), ("ollama_eval_duration_ns", eval_ns)):
            sets.append(f"{k}=?")
            vals.append(v)
        if model_calls is not None:
            sets.append("model_calls=?"); vals.append(model_calls)
        if tool_calls is not None:
            sets.append("tool_calls=?"); vals.append(tool_calls)
        vals.append(run_id)
        self._conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE run_id=?", vals)

    def mark_mutation_started(self, run_id: str) -> None:
        self._conn.execute("UPDATE runs SET mutation_started=1 WHERE run_id=?", (run_id,))

    def finish_run(self, run_id: str, status: str, now: int, *,
                   post_worktree_sha256: str | None = None,
                   error_code: str | None = None,
                   error_text: str | None = None,
                   wall_duration_ms: int | None = None) -> None:
        sets = ["status=?", "finished_at=?"]
        vals: list[Any] = [status, now]
        if post_worktree_sha256 is not None:
            sets.append("post_worktree_sha256=?")
            vals.append(post_worktree_sha256)
        if error_code is not None:
            sets.append("error_code=?"); vals.append(error_code)
        if error_text is not None:
            sets.append("error_text=?"); vals.append(error_text)
        if wall_duration_ms is not None:
            sets.append("wall_duration_ms=?"); vals.append(wall_duration_ms)
        vals.append(run_id)
        self._conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE run_id=?", vals)

    # ---------------- Summary ----------------

    def summary(self) -> dict[str, Any]:
        cur = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status"
        )
        counts = {r["status"]: r["n"] for r in cur.fetchall()}
        return {"task_counts": counts}


def default_db_path() -> Path:
    from .runtime import state_dir
    return state_dir() / "state.db"
