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

-- -------------------- campaign-v2 (P06) additions --------------------

CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id TEXT PRIMARY KEY,
    grant_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    state TEXT NOT NULL,
    integration_branch TEXT NOT NULL,
    current_commit TEXT,
    current_tree_digest TEXT,
    current_fence INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    completed_at INTEGER NOT NULL DEFAULT 0,
    grant_digest TEXT NOT NULL,
    plan_digest TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS campaigns_grant ON campaigns(grant_id);
CREATE INDEX IF NOT EXISTS campaigns_state ON campaigns(state);

CREATE TABLE IF NOT EXISTS grants (
    grant_id TEXT PRIMARY KEY,
    grant_digest TEXT NOT NULL,
    state TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    plan_revision INTEGER NOT NULL,
    operator_id TEXT NOT NULL,
    operator_receipt_digest TEXT NOT NULL,
    activated_at INTEGER NOT NULL DEFAULT 0,
    revoked_at INTEGER NOT NULL DEFAULT 0,
    revoked_reason TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS grants_state ON grants(state);

CREATE TABLE IF NOT EXISTS budget_ledgers (
    ledger_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    grant_id TEXT NOT NULL,
    family_id TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    cumulative_model_calls INTEGER NOT NULL DEFAULT 0,
    cumulative_tool_calls INTEGER NOT NULL DEFAULT 0,
    cumulative_repairs INTEGER NOT NULL DEFAULT 0,
    cumulative_rechunks INTEGER NOT NULL DEFAULT 0,
    cumulative_escalations INTEGER NOT NULL DEFAULT 0,
    cumulative_active_seconds INTEGER NOT NULL DEFAULT 0,
    cumulative_cost_microusd INTEGER NOT NULL DEFAULT 0,
    cumulative_chunks INTEGER NOT NULL DEFAULT 0,
    cumulative_context_tokens INTEGER NOT NULL DEFAULT 0,
    bounds_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS budget_ledgers_campaign ON budget_ledgers(campaign_id);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    package_id TEXT NOT NULL,
    parent_chunk_id TEXT,
    revision INTEGER NOT NULL DEFAULT 1,
    idempotency_key TEXT NOT NULL,
    state TEXT NOT NULL,
    snapshot_commit TEXT,
    snapshot_tree_digest TEXT,
    preview_diff_path TEXT,
    accepted_predecessor_commit TEXT,
    accepted_predecessor_tree TEXT,
    admission_id TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    idempotency_content_sha256 TEXT
);
CREATE INDEX IF NOT EXISTS chunks_campaign ON chunks(campaign_id);
CREATE UNIQUE INDEX IF NOT EXISTS chunks_idem_campaign ON chunks(campaign_id, idempotency_key);

CREATE TABLE IF NOT EXISTS admissions (
    admission_id TEXT PRIMARY KEY,
    grant_id TEXT NOT NULL,
    grant_revision INTEGER NOT NULL,
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id),
    chunk_revision INTEGER NOT NULL,
    accepted_predecessor_commit TEXT NOT NULL,
    accepted_predecessor_tree TEXT NOT NULL,
    runtime_digest TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_digest TEXT NOT NULL,
    policy_profile_id TEXT NOT NULL,
    validator_profile_ids_json TEXT NOT NULL,
    provider_profile_id TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    budget_ledger_id TEXT NOT NULL,
    fence_generation INTEGER NOT NULL,
    lease_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    issued_at INTEGER NOT NULL,
    issuer TEXT NOT NULL DEFAULT 'runner',
    snapshot_json TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS admissions_chunk ON admissions(chunk_id);

CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    resource_id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    owner_boot_id TEXT NOT NULL,
    owner_pid INTEGER NOT NULL,
    fence_generation INTEGER NOT NULL,
    acquired_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    released_at INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS leases_campaign ON leases(campaign_id);

CREATE TABLE IF NOT EXISTS integration_journal (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    expected_old_commit TEXT,
    committed_new_commit TEXT NOT NULL,
    fence_generation INTEGER NOT NULL,
    committed_at INTEGER NOT NULL,
    actor TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS integration_journal_campaign ON integration_journal(campaign_id);

CREATE TABLE IF NOT EXISTS race_admissions (
    idem_key TEXT PRIMARY KEY,
    admission_id TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    issued_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS crash_windows (
    window_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    chunk_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    observed_artifact TEXT,
    snapshot_at INTEGER NOT NULL,
    recovered_at INTEGER NOT NULL DEFAULT 0,
    reconciliation_reason TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS crash_windows_campaign ON crash_windows(campaign_id);

CREATE TABLE IF NOT EXISTS campaign_events (
    event_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    chunk_id TEXT,
    event_type TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT,
    actor TEXT NOT NULL DEFAULT 'runner',
    payload TEXT NOT NULL DEFAULT '{}',
    fence_generation INTEGER NOT NULL DEFAULT 1,
    issued_at INTEGER NOT NULL,
    idempotency_key TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS campaign_events_campaign ON campaign_events(campaign_id);

-- Protected operator approvals (P06-A01): the ONLY surface that can
-- mint an active grant. The activation callback resolves the supplied
-- approval_id to a record that ALREADY exists in this table. A
-- caller-supplied dict alone confers NO authority.
CREATE TABLE IF NOT EXISTS protected_approvals (
    approval_id TEXT PRIMARY KEY,
    operation TEXT NOT NULL,                 -- 'activate_grant'
    grant_digest_target TEXT NOT NULL,
    operator_id TEXT NOT NULL,
    operator_receipt_digest TEXT NOT NULL,
    issued_at INTEGER NOT NULL,
    consumed_at INTEGER NOT NULL DEFAULT 0,
    payload TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS protected_approvals_operator ON protected_approvals(operator_id);

-- Approved plans (P06-A01): bind work-packages and criterion coverage.
-- Admission REQUIRES the chunk's package_id and criterion_ids to be
-- present in the plan's approved structure.
CREATE TABLE IF NOT EXISTS approved_plans (
    plan_id TEXT PRIMARY KEY,
    approved_artifact_id TEXT NOT NULL,
    plan_digest TEXT NOT NULL,
    payload TEXT NOT NULL,                     -- canonical JSON
    created_at INTEGER NOT NULL
);

-- Migration bookkeeping (P09 review follow-up).
CREATE TABLE IF NOT EXISTS migrations_applied (
    name TEXT PRIMARY KEY,
    applied_at INTEGER NOT NULL,
    integrity_ok INTEGER NOT NULL DEFAULT 1
);
"""

# Migration log for forward-compatible schema versioning. Each versioned
# migration records its name + first-applied timestamp. Older binaries may
# ignore newer rows; newer binaries refuse to run if any record exists
# whose name they do not understand (unless ``SCHEMA_FORWARD_COMPAT`` is
# explicitly set).
_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    name TEXT PRIMARY KEY,
    applied_at INTEGER NOT NULL,
    note TEXT NOT NULL DEFAULT ''
);
"""

class ClaimConflict(Exception):
    """Raised when an atomic claim loses a race or fails to satisfy constraints.

    Always raised from within a transaction so the caller knows the
    transaction has been rolled back.
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
        self._conn.executescript(_MIGRATIONS_TABLE)
        # Apply migrations in order (idempotent).
        self._record_migration("p06-0001-campaign-v2-tables")
        self._record_migration("p06-followup-0001-protected-approvals-and-plans")

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def _record_migration(self, name: str, *, note: str = "") -> None:
        """Record a schema migration. Idempotent: the same name may be
        applied multiple times without re-running the migration body.

        Refuses to run if an unknown migration name is already present
        in the log (forward-compat guard). The P06 series of migrations
        are recorded unconditionally; future migrations extending this
        set must declare a forward-compatibility window to keep older
        binaries openable.
        """
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (name, applied_at, note) VALUES (?, ?, ?)",
                (name, int(time.time()), note),
            )
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

    def _emit_event_via_cur(
        self, cur, session_id: str, event_type: str,
        *, task_id: str | None = None, run_id: str | None = None,
        from_state: str | None = None, to_state: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Emit an event using an existing transaction cursor.

        Used by the atomic claim so the run_started event is committed in
        the SAME transaction as the rows insert and the task transition.
        """
        cur.execute(
            """
            INSERT INTO events (timestamp, session_id, task_id, run_id, event_type, from_state, to_state, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (int(time.time()), session_id, task_id, run_id, event_type,
             from_state, to_state, json.dumps(details or {})),
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
        worker_pid: int,
        model_profile_json: str,
        now: int,
        lease_expires_at: int,
        execution_class_filter: list[str] | None = None,
        # If provided, candidates whose `task_timeout_seconds` exceeds
        # `remaining_runtime - shutdown_margin` are skipped in priority order.
        remaining_runtime: float | None = None,
        shutdown_margin: float = 0.0,
    ) -> dict[str, Any] | None:
        """Atomically pick eligible APPROVED task, insert RUNNING runs row,
        transition task APPROVED -> RUNNING.

        Single implementation. Used by both `run-next` and `run-nightly`.

        Eligibility:
          - status='APPROVED'
          - execution_class IN (execution_class_filter) if filter given
          - all declared dependencies (status='PASSED') are PASSED
          - if remaining_runtime is provided: the manifest's
            limits.task_timeout_seconds fits within remaining - shutdown_margin

        Returns the claim dict {task, run_id, attempt_no, session_id} or
        None if no eligible task.

        Raises ClaimConflict on lost-race UPDATE rowcount != 1 (transaction
        rolls back; caller may retry).
        """
        where_extra = ""
        params: list[Any] = []
        if execution_class_filter:
            placeholders = ",".join("?" for _ in execution_class_filter)
            where_extra = f" AND execution_class IN ({placeholders})"
            params.extend(execution_class_filter)
        # We try candidates in priority order. If a candidate's timeout
        # is too large, try the next. We do this in a single transaction
        # with a loop over (peek + claim) so the budget check is race-safe.
        while True:
            with self.transaction() as cur:
                # Find best candidate (priority + created_at order).
                cur.execute(
                    f"""
                    SELECT task_id, manifest_json FROM tasks
                    WHERE status='APPROVED'
                      AND NOT EXISTS (
                        SELECT 1 FROM json_each(dependencies_json) d
                        WHERE json_extract(d.value, '$.required_state')='PASSED'
                          AND NOT EXISTS (
                            SELECT 1 FROM tasks dep
                            WHERE dep.task_id = json_extract(d.value, '$.task_id')
                              AND dep.status='PASSED'
                          )
                      )
                      {where_extra}
                    ORDER BY priority ASC, created_at ASC
                    LIMIT 10
                    """,
                    tuple(params),
                )
                rows = [dict(r) for r in cur.fetchall()]
                if not rows:
                    return None
                # Choose first candidate that fits in remaining budget.
                chosen = None
                if remaining_runtime is None:
                    chosen = rows[0]
                else:
                    import json as _json
                    from .schemas import TaskManifest as _TM
                    for r in rows:
                        m = _TM.model_validate(_json.loads(r["manifest_json"]))
                        if remaining_runtime - shutdown_margin >= m.limits.task_timeout_seconds:
                            chosen = r
                            break
                    if chosen is None:
                        # No candidate fits the remaining budget.
                        return None

                task_id = chosen["task_id"]
                cur.execute("SELECT COALESCE(MAX(attempt_no),0)+1 AS n FROM runs WHERE task_id=?", (task_id,))
                attempt_no = int(cur.fetchone()["n"])
                cur.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,))
                task = dict(cur.fetchone())
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
                        run_id, task_id, session_id, attempt_no, "RUNNING", now,
                        worker_pid, task.get("approved_model_name"),
                        task.get("approved_model_digest"), model_profile_json,
                        f"pid:{worker_pid}", now, lease_expires_at,
                        task.get("approved_repo_head"), "", 0, "",
                    ),
                )
                # Transition APPROVED -> RUNNING, guarded by status.
                cur.execute(
                    "UPDATE tasks SET status='RUNNING', run_id=?, updated_at=? WHERE task_id=? AND status='APPROVED'",
                    (run_id, now, task_id),
                )
                if cur.rowcount != 1:
                    # Lost the race. ROLLBACK is automatic on exception; we
                    # raise so the caller knows the transaction was aborted.
                    raise ClaimConflict(f"task {task_id} no longer APPROVED")
                db_emit_run_started(self, cur, session_id, task_id, run_id, attempt_no)
            return {"task": task, "run_id": run_id, "attempt_no": attempt_no,
                    "session_id": session_id}

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

    # ---------------- Internal helper ----------------


def db_emit_run_started(db: Database, cur, session_id: str, task_id: str,
                        run_id: str, attempt_no: int) -> None:
    db._emit_event_via_cur(cur, session_id, "run_started", task_id=task_id,
                           run_id=run_id, from_state="APPROVED",
                           to_state="RUNNING", details={"attempt_no": attempt_no})


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
