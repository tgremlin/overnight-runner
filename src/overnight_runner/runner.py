"""Phase 2 queued runner: atomic claim, heartbeat, recovery, nightly session.

Single-source-of-truth for the lifecycle:

  claim (atomic DB transaction)
  -> heartbeat thread (own DB connection)
  -> worker.run() (NO DB held during Ollama / subprocess / mutation)
  -> finalize (DB transition + metrics + post worktree SHA)

Claim invariant:
  tasks.status='RUNNING'  <=>  there is a RUNNING runs row for it.

The claim is the ONLY place where the status transitions APPROVED->RUNNING and
the runs row is inserted. They are written in the same BEGIN IMMEDIATE
transaction. If the process dies before execution starts, the runs row
exists with lease_expires_at set; recovery_scan finds it.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .broker import Broker
from .db import Database, default_db_path
from .ollama_client import OllamaClient
from .runtime import is_paused, state_dir
from .worker import Approval, Worker
from .safety import git_head, git_worktree_sha
from .schemas import (
    Disposition, ExecutionClass, TaskManifest, TaskStatus, canonical_sha,
)
from .worker import Approval, Worker


LEASE_SECONDS = 120
HEARTBEAT_EVERY = 30
SHUTDOWN_MARGIN_SECONDS = 60


def _execute_via_worker(db: Database, task_row: dict[str, Any], run_id: str,
                        attempt_no: int, artifact_dir: Path,
                        session_id: str) -> QueuedRunResult:
    """Internal helper: run worker + finalize. Used by both claim path and
    legacy shim. Caller must have already inserted the runs row and
    transitioned the task."""
    manifest = TaskManifest.model_validate(json.loads(task_row["manifest_json"]))
    repo_root = Path(manifest.repo.path).resolve()
    approval = Approval(
        manifest_sha256=task_row["approved_manifest_sha256"] or task_row["manifest_sha256"],
        approved_repo_head=task_row["approved_repo_head"],
        approved_runtime_sha256=task_row["approved_runtime_sha256"],
        approved_model_name=manifest.model_profile.model_name,
        approved_model_digest=task_row.get("approved_model_digest"),
        approved_at=int(time.time()),
        approved_by=task_row.get("approved_by") or "queued",
    )
    (artifact_dir / "approval.json").write_text(json.dumps(approval.to_dict(), indent=2))
    metrics_state: dict[str, int] = {
        "prompt_eval_count": 0, "eval_count": 0,
        "ollama_total_duration_ns": 0, "ollama_eval_duration_ns": 0,
        "model_calls": 0, "tool_calls": 0,
    }
    stop_event = threading.Event()
    hb_db = Database(default_db_path())
    def hb_loop() -> None:
        try:
            while not stop_event.is_set():
                now = int(time.time())
                try:
                    hb_db.heartbeat(
                        run_id, now, now + LEASE_SECONDS,
                        prompt_eval_count=metrics_state["prompt_eval_count"],
                        eval_count=metrics_state["eval_count"],
                        total_ns=metrics_state["ollama_total_duration_ns"],
                        eval_ns=metrics_state["ollama_eval_duration_ns"],
                        model_calls=metrics_state["model_calls"],
                        tool_calls=metrics_state["tool_calls"],
                    )
                except Exception:
                    pass
                stop_event.wait(HEARTBEAT_EVERY)
        finally:
            hb_db.close()
    hb = threading.Thread(target=hb_loop, name=f"hb-{run_id}", daemon=True)
    hb.start()
    def on_mutation(entry) -> None:
        nl = Database(default_db_path())
        try:
            nl.mark_mutation_started(run_id)
        finally:
            nl.close()
    worker = Worker()
    worker.model_digest_resolver = OllamaClient().model_digest
    orig_build = worker._build_broker
    def _patched(m, r, a, ap):
        b = orig_build(m, r, a, ap)
        b.on_mutation = on_mutation
        return b
    worker._build_broker = _patched  # type: ignore[assignment]
    started = time.time()
    status, reason_code, reason_text = "RUNNING", "", ""
    try:
        result = worker.run(manifest, approval=approval, artifact_dir=artifact_dir,
                            on_metrics=_make_metrics_cb(metrics_state))
        status = result.status; reason_code = result.reason_code; reason_text = result.reason_text
    except Exception as e:
        status = "FAILED"; reason_code = type(e).__name__; reason_text = str(e)
    finally:
        stop_event.set()
        try: hb.join(timeout=HEARTBEAT_EVERY + 2)
        except Exception: pass
    finished_at = int(time.time())
    post_worktree = git_worktree_sha(repo_root) if repo_root.exists() else ""
    fd = Database(default_db_path())
    try:
        fd.finish_run(run_id, status, finished_at, post_worktree_sha256=post_worktree,
                      error_code=None if status == "PASSED" else reason_code,
                      error_text=None if status == "PASSED" else reason_text,
                      wall_duration_ms=int((time.time() - started) * 1000))
        try:
            fd.heartbeat(run_id, finished_at, finished_at,
                         prompt_eval_count=metrics_state["prompt_eval_count"],
                         eval_count=metrics_state["eval_count"],
                         total_ns=metrics_state["ollama_total_duration_ns"],
                         eval_ns=metrics_state["ollama_eval_duration_ns"],
                         model_calls=metrics_state["model_calls"],
                         tool_calls=metrics_state["tool_calls"])
        except Exception: pass
        fd.update_status(manifest.task_id, TaskStatus(status),
                         final_reason_code=reason_code,
                         final_reason_text=reason_text)
        fd.emit_event(session_id, "run_finished", task_id=manifest.task_id, run_id=run_id,
                      from_state="RUNNING", to_state=status,
                      details={"reason_code": reason_code, "reason_text": reason_text,
                               "artifact_dir": str(artifact_dir)})
    finally:
        fd.close()
    return QueuedRunResult(
        task_id=manifest.task_id, run_id=run_id, attempt_no=attempt_no,
        status=status, reason_code=reason_code, reason_text=reason_text,
        artifact_dir=str(artifact_dir), execution_class=manifest.execution_class.value,
    )


# ----------------------------- Result types -----------------------------

@dataclass
class QueuedRunResult:
    task_id: str
    run_id: str
    attempt_no: int
    status: str
    reason_code: str
    reason_text: str
    artifact_dir: str
    execution_class: str


@dataclass
class NightlyResult:
    session_id: str
    started_at: float
    finished_at: float
    stop_reason: str
    tasks_attempted: int = 0
    tasks_passed: int = 0
    tasks_failed: int = 0
    tasks_blocked: int = 0
    tasks_review_required: int = 0
    read_only_attempted: int = 0
    mutation_attempted: int = 0
    task_results: list[dict[str, Any]] = field(default_factory=list)
    dep_blocked_approved: list[dict[str, Any]] = field(default_factory=list)
    require_review: list[str] = field(default_factory=list)


# ----------------------------- Recovery -----------------------------

def recovery_scan(now: int | None = None) -> list[dict[str, Any]]:
    """Scan stale RUNNING runs AND orphan RUNNING tasks. Apply policy:
       - source_mutation stale -> REVIEW_REQUIRED, never retry
       - read_only stale       -> REVIEW_REQUIRED (caller may retry)
       - unreal_editor stale   -> REVIEW_REQUIRED
       - orphan RUNNING tasks  -> REVIEW_REQUIRED, ORPHANED_RUNNING_TASK
    """
    if now is None:
        now = int(time.time())
    out: list[dict[str, Any]] = []
    # 1. Stale runs
    db = Database(default_db_path())
    try:
        stale = db.find_stale_runs(now)
    finally:
        db.close()
    for s in stale:
        db = Database(default_db_path())
        try:
            t = db.get_task(s["task_id"])
            if not t:
                continue
            manifest = TaskManifest.model_validate(json.loads(t["manifest_json"]))
            cls = manifest.execution_class
            if cls == ExecutionClass.UNREAL_EDITOR:
                reason = "STALE_UNREAL"
            elif cls == ExecutionClass.SOURCE_MUTATION:
                reason = "STALE_MUTATION_NEVER_RETRY"
            else:
                reason = "STALE_READ_ONLY"
            db.update_status(s["task_id"], TaskStatus.REVIEW_REQUIRED,
                             final_reason_code=reason,
                             final_reason_text=f"stale run {s['run_id']}")
            db.finish_run(s["run_id"], "REVIEW_REQUIRED", now,
                          error_code=reason,
                          error_text=f"lease expired at {s.get('lease_expires_at')}")
            db.emit_event("recovery", "task_stale", task_id=s["task_id"], run_id=s["run_id"],
                          to_state=TaskStatus.REVIEW_REQUIRED.value,
                          details={"reason": reason})
            out.append({"task_id": s["task_id"], "run_id": s["run_id"],
                        "execution_class": cls.value, "action": reason})
        finally:
            db.close()

    # 2. Orphan RUNNING tasks (no active RUNNING runs row)
    db = Database(default_db_path())
    try:
        orphans = db.find_orphan_running_tasks()
    finally:
        db.close()
    for o in orphans:
        db = Database(default_db_path())
        try:
            manifest = TaskManifest.model_validate(json.loads(o["manifest_json"]))
            db.update_status(o["task_id"], TaskStatus.REVIEW_REQUIRED,
                             final_reason_code="ORPHANED_RUNNING_TASK",
                             final_reason_text="legacy orphan RUNNING task with no runs row")
            db.emit_event("recovery", "task_orphaned", task_id=o["task_id"],
                          to_state=TaskStatus.REVIEW_REQUIRED.value,
                          details={"reason": "ORPHANED_RUNNING_TASK"})
            out.append({"task_id": o["task_id"], "action": "ORPHANED_RUNNING_TASK",
                        "execution_class": manifest.execution_class.value})
        finally:
            db.close()
    return out


# ----------------------------- Atomic claim + run -----------------------------

def execute_claimed_task(
    db: Database, *,
    execution_class_filter: list[str] | None = None,
    session_id: str | None = None,
    remaining_runtime: float | None = None,
    shutdown_margin: float = 0.0,
) -> QueuedRunResult | None:
    """Atomic claim, run, finalize.

    Caller MUST hold the global runner_lock.
    Returns None if no eligible task.

    `remaining_runtime` is the budget (seconds) remaining in the calling
    session; combined with `shutdown_margin` it is used to skip candidates
    whose `task_timeout_seconds` would not fit.
    """
    now = int(time.time())
    sid = session_id or f"sess-{uuid.uuid4().hex[:8]}"
    run_id = f"run-{now}-{uuid.uuid4().hex[:6]}"
    try:
        claim = db.claim_next_approved(
            run_id=run_id, session_id=sid, worker_pid=os.getpid(),
            model_profile_json="{}",
            now=now, lease_expires_at=now + LEASE_SECONDS,
            execution_class_filter=execution_class_filter,
            remaining_runtime=remaining_runtime,
            shutdown_margin=shutdown_margin,
        )
    except ClaimConflict:
        return None
    if claim is None:
        return None
    task_row = claim["task"]
    run_id = claim["run_id"]
    attempt_no = claim["attempt_no"]
    sid = claim["session_id"]
    manifest = TaskManifest.model_validate(json.loads(task_row["manifest_json"]))
    repo_root = Path(manifest.repo.path).resolve()
    artifact_dir = state_dir() / "runs" / manifest.task_id / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "manifest.json").write_bytes(_canonical(manifest))
    db._conn.execute("UPDATE runs SET artifact_dir=? WHERE run_id=?",
                     (str(artifact_dir), run_id))
    return _execute_via_worker(db, task_row, run_id, attempt_no, artifact_dir, sid)


def _make_metrics_cb(state: dict[str, int]):
    """The worker reports per-turn metrics; we accumulate running totals."""
    def cb(delta: dict[str, int]) -> None:
        for k, v in delta.items():
            if isinstance(v, int):
                state[k] = state.get(k, 0) + v
    return cb


# ----------------------------- Nightly session -----------------------------

DEFAULT_NIGHTLY = {
    "max_wall_seconds": 8 * 3600,
    "max_tasks": 6,
    "max_mutation_tasks": 1,
    # automatic retries intentionally NOT implemented for MVP.
    # "safety > utilization". A failed task is left for human review.
    "retries": 0,
}


def run_nightly(session_id: str | None = None) -> NightlyResult:
    """Execute the full nightly session.

    Caller MUST hold the global runner_lock.

    Policy (per spec):
      - read-only tasks first
      - then at most one source_mutation
      - stop immediately after mutation
      - never execute unreal_editor
      - PAUSED stops new tasks
      - max_wall_seconds, max_tasks, max_mutation_tasks respected
    """
    from .runtime import require_not_paused  # raises SafetyError -> BLOCKED

    # Wall timestamps for audit/persistence; monotonic for scheduling.
    wall_started = time.time()
    mono_started = time.monotonic()
    session_id = session_id or f"nightly-{int(wall_started)}-{uuid.uuid4().hex[:6]}"
    result = NightlyResult(session_id=session_id, started_at=wall_started,
                           finished_at=wall_started, stop_reason="init")

    def remaining_budget() -> float:
        return DEFAULT_NIGHTLY["max_wall_seconds"] - (time.monotonic() - mono_started)

    def fits_in_budget(task_timeout_seconds: int) -> bool:
        return remaining_budget() - SHUTDOWN_MARGIN_SECONDS >= task_timeout_seconds

    try:
        require_not_paused()
    except Exception as e:
        result.stop_reason = f"PAUSED_AT_START: {e}"
        result.finished_at = time.time()
        _write_summary(result, [], session_id, wall_started, result.finished_at)
        return result

    # Recovery first.
    recovery_actions = recovery_scan()
    result.stop_reason = "ready"

    db = Database(default_db_path())
    try:
        # Phase 1: read-only loop
        while True:
            if is_paused():
                result.stop_reason = "PAUSED_BETWEEN_TASKS"
                break
            if result.tasks_attempted >= DEFAULT_NIGHTLY["max_tasks"]:
                result.stop_reason = "MAX_TASKS_REACHED"
                break
            if remaining_budget() - SHUTDOWN_MARGIN_SECONDS <= 0:
                result.stop_reason = "RUNTIME_BUDGET_EXHAUSTED"
                break
            # Peek at next eligible task timeout to ensure we don't claim
            # something we can't finish. If no candidate fits, leave it for
            # the next night and stop this one.
            from .schemas import TaskManifest as _TM
            with db.transaction() as cur:
                cur.execute(
                    """
                    SELECT manifest_json FROM tasks
                    WHERE status='APPROVED' AND execution_class='read_only'
                      AND NOT EXISTS (
                        SELECT 1 FROM json_each(dependencies_json) d
                        WHERE json_extract(d.value, '$.required_state')='PASSED'
                          AND NOT EXISTS (
                            SELECT 1 FROM tasks dep
                            WHERE dep.task_id = json_extract(d.value, '$.task_id')
                              AND dep.status='PASSED'
                          )
                      )
                    ORDER BY priority ASC, created_at ASC LIMIT 1
                    """
                )
                row = cur.fetchone()
                if row is None:
                    result.stop_reason = "NO_MORE_READ_ONLY"
                    break
                peek = _TM.model_validate(json.loads(row["manifest_json"]))
            if not fits_in_budget(peek.limits.task_timeout_seconds):
                result.stop_reason = "BUDGET_INSUFFICIENT_FOR_NEXT_TASK"
                break
            r = execute_claimed_task(
                db,
                execution_class_filter=[ExecutionClass.READ_ONLY.value],
                session_id=session_id,
                remaining_runtime=remaining_budget(),
                shutdown_margin=SHUTDOWN_MARGIN_SECONDS,
            )
            if r is None:
                result.stop_reason = "NO_MORE_READ_ONLY"
                break
            result.tasks_attempted += 1
            result.read_only_attempted += 1
            _absorb_task_result(result, r)
            if r.execution_class == "source_mutation":
                # Defensive: should never happen with the filter.
                break
            if r.status not in ("PASSED", "FAILED", "BLOCKED", "REVIEW_REQUIRED"):
                break

        # Phase 2: at most one mutation (if capacity remains and no mutation already done)
        if (result.mutation_attempted < DEFAULT_NIGHTLY["max_mutation_tasks"]
                and not is_paused()
                and result.tasks_attempted < DEFAULT_NIGHTLY["max_tasks"]):
            # Peek mutation timeout if any.
            with db.transaction() as cur:
                cur.execute(
                    """
                    SELECT manifest_json FROM tasks
                    WHERE status='APPROVED' AND execution_class='source_mutation'
                      AND NOT EXISTS (
                        SELECT 1 FROM json_each(dependencies_json) d
                        WHERE json_extract(d.value, '$.required_state')='PASSED'
                          AND NOT EXISTS (
                            SELECT 1 FROM tasks dep
                            WHERE dep.task_id = json_extract(d.value, '$.task_id')
                              AND dep.status='PASSED'
                          )
                      )
                    ORDER BY priority ASC, created_at ASC LIMIT 1
                    """
                )
                row = cur.fetchone()
            if row is None:
                if result.stop_reason in ("ready",):
                    result.stop_reason = "NO_ELIGIBLE_MUTATION"
            else:
                peek = _TM.model_validate(json.loads(row["manifest_json"]))
                if not fits_in_budget(peek.limits.task_timeout_seconds):
                    result.stop_reason = "BUDGET_INSUFFICIENT_FOR_MUTATION"
                else:
                    r = execute_claimed_task(
                        db,
                        execution_class_filter=[ExecutionClass.SOURCE_MUTATION.value],
                        session_id=session_id,
                        remaining_runtime=remaining_budget(),
                        shutdown_margin=SHUTDOWN_MARGIN_SECONDS,
                    )
                    if r is not None:
                        result.tasks_attempted += 1
                        result.mutation_attempted += 1
                        _absorb_task_result(result, r)
                        result.stop_reason = "MUTATION_DONE_STOPPING"
                    # else: leave stop_reason as the read-only reason.
    finally:
        # Capture dep-blocked tasks for the summary.
        try:
            result.dep_blocked_approved = db.get_dep_blocked_approved()
        except Exception:
            result.dep_blocked_approved = []
        db.close()

    result.finished_at = time.time()
    _write_summary(result, recovery_actions, session_id, wall_started, result.finished_at)
    return result


def _absorb_task_result(result: NightlyResult, r: QueuedRunResult) -> None:
    if r.status == "PASSED":
        result.tasks_passed += 1
    elif r.status == "BLOCKED":
        result.tasks_blocked += 1
    elif r.status == "REVIEW_REQUIRED":
        result.tasks_review_required += 1
        result.require_review.append(r.task_id)
    else:
        result.tasks_failed += 1
    result.task_results.append({
        "task_id": r.task_id, "run_id": r.run_id, "attempt": r.attempt_no,
        "execution_class": r.execution_class, "status": r.status,
        "reason_code": r.reason_code, "reason_text": r.reason_text,
        "artifact_dir": r.artifact_dir,
    })


def _write_summary(result: NightlyResult, recovery_actions: list[dict[str, Any]],
                   session_id: str, started: float, finished: float) -> None:
    """Persist morning summary as JSON + Markdown."""
    sess_dir = state_dir() / "sessions" / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "session_id": session_id,
        "started_at": int(started),
        "finished_at": int(finished),
        "wall_duration_seconds": int(finished - started),
        "stop_reason": result.stop_reason,
        "limits": DEFAULT_NIGHTLY,
        "counts": {
            "tasks_attempted": result.tasks_attempted,
            "tasks_passed": result.tasks_passed,
            "tasks_failed": result.tasks_failed,
            "tasks_blocked": result.tasks_blocked,
            "tasks_review_required": result.tasks_review_required,
            "read_only_attempted": result.read_only_attempted,
            "mutation_attempted": result.mutation_attempted,
        },
        "recovery_actions": recovery_actions,
        "dep_blocked_approved": result.dep_blocked_approved,
        "task_results": result.task_results,
        "require_review": result.require_review,
    }
    (sess_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    md_lines = [
        f"# Overnight Summary: {session_id}",
        "",
        f"- Started: {int(started)}",
        f"- Finished: {int(finished)}",
        f"- Wall duration: {int(finished - started)} seconds",
        f"- Stop reason: {result.stop_reason}",
        "",
        "## Limits",
        f"- Max wall: {DEFAULT_NIGHTLY['max_wall_seconds']} seconds",
        f"- Max tasks: {DEFAULT_NIGHTLY['max_tasks']}",
        f"- Max mutation tasks: {DEFAULT_NIGHTLY['max_mutation_tasks']}",
        f"- Auto-retries: {DEFAULT_NIGHTLY['retries']} (0 = none; tasks that fail are left for human review)",
        "",
        "## Counts",
        f"- Attempted: {result.tasks_attempted}",
        f"- Passed: {result.tasks_passed}",
        f"- Failed: {result.tasks_failed}",
        f"- Blocked: {result.tasks_blocked}",
        f"- Review required: {result.tasks_review_required}",
        f"- Read-only attempted: {result.read_only_attempted}",
        f"- Mutation attempted: {result.mutation_attempted}",
        "",
        "## Recovery actions",
    ]
    if not recovery_actions:
        md_lines.append("- (none)")
    else:
        for a in recovery_actions:
            md_lines.append(f"- {a.get('task_id')}: {a.get('action')}")
    md_lines += ["", "## Tasks"]
    for tr in result.task_results:
        md_lines.append(
            f"- {tr['task_id']} ({tr['execution_class']}) attempt={tr['attempt']} "
            f"-> {tr['status']} ({tr['reason_code']})"
        )
        md_lines.append(f"    artifact: {tr['artifact_dir']}")
    md_lines += ["", "## Dependency-blocked APPROVED"]
    if not result.dep_blocked_approved:
        md_lines.append("- (none)")
    else:
        for d in result.dep_blocked_approved:
            md_lines.append(f"- {d['task_id']} blocked_by=[{d.get('blocked_by','')}]")
    md_lines += ["", "## Require morning review"]
    if not result.require_review:
        md_lines.append("- (none)")
    else:
        for t in result.require_review:
            md_lines.append(f"- {t}")
    (sess_dir / "summary.md").write_text("\n".join(md_lines) + "\n")


def _canonical(m: TaskManifest) -> bytes:
    from .schemas import canonical_json
    return canonical_json(m)
