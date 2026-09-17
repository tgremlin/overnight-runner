"""Phase 2 queued runner: heartbeat thread + lease + recovery helpers.

Each queued run gets:
  - its own DB connection in the heartbeat thread;
  - a worker-owned process group for any subprocess;
  - periodic heartbeats extending the lease;
  - on entry: a 'runs' row; on exit: finished status + post worktree SHA.

This module is intentionally small.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import Database, default_db_path
from .runtime import state_dir
from .safety import git_head, git_worktree_sha
from .schemas import ExecutionClass, TaskManifest, TaskStatus, canonical_sha
from .worker import Approval, Worker


LEASE_SECONDS = 120  # heartbeat every 30s; lease expires in 120s
HEARTBEAT_EVERY = 30


@dataclass
class QueuedRunResult:
    task_id: str
    run_id: str
    status: str
    reason_code: str
    reason_text: str
    artifact_dir: str


def execute_queued_task(task_row: dict[str, Any], *, heartbeat_every: int = HEARTBEAT_EVERY,
                        lease_seconds: int = LEASE_SECONDS) -> QueuedRunResult:
    """Execute one queued task. Caller must hold the global runner lock."""
    task_id = task_row["task_id"]
    manifest = TaskManifest.model_validate(json.loads(task_row["manifest_json"]))
    repo_root = Path(manifest.repo.path).resolve()

    session_id = f"sess-{uuid.uuid4().hex[:8]}"
    run_id = f"run-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    artifact_root = state_dir() / "runs" / task_id / run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "manifest.json").write_bytes(_canonical(manifest))

    approval = Approval(
        manifest_sha256=task_row["approved_manifest_sha256"] or task_row["manifest_sha256"],
        approved_repo_head=task_row["approved_repo_head"],
        approved_runtime_sha256=task_row["approved_runtime_sha256"],
        approved_model_name=manifest.model_profile.model_name,
        approved_model_digest=task_row.get("approved_model_digest"),
        approved_at=int(time.time()),
        approved_by=task_row.get("approved_by") or "queued",
    )
    (artifact_root / "approval.json").write_text(json.dumps(approval.to_dict(), indent=2))

    db = Database(default_db_path())
    started_at = int(time.time())
    lease_expires_at = started_at + lease_seconds
    pre_repo_head = task_row["approved_repo_head"] or git_head(repo_root)
    pre_worktree = git_worktree_sha(repo_root)

    db.insert_run(
        run_id=run_id, task_id=task_id, session_id=session_id, attempt_no=1,
        status="RUNNING", started_at=started_at,
        worker_pid=os.getpid(),
        model_name=manifest.model_profile.model_name,
        model_digest=approval.approved_model_digest,
        model_profile=json.dumps({
            "num_ctx": manifest.model_profile.num_ctx,
            "num_predict": manifest.model_profile.num_predict,
            "temperature": manifest.model_profile.temperature,
            "seed": manifest.model_profile.seed,
        }),
        lease_expires_at=lease_expires_at,
        pre_repo_head=pre_repo_head,
        pre_worktree_sha256=pre_worktree,
        artifact_dir=str(artifact_root),
        mutation_started=manifest.execution_class == ExecutionClass.SOURCE_MUTATION,
    )
    db.emit_event(session_id, "run_started", task_id=task_id, run_id=run_id,
                  to_state="RUNNING")

    stop_event = threading.Event()
    metrics_state: dict[str, int] = {
        "prompt_eval_count": 0, "eval_count": 0,
        "ollama_total_duration_ns": 0, "ollama_eval_duration_ns": 0,
        "model_calls": 0, "tool_calls": 0,
    }

    def heartbeat_loop() -> None:
        own = Database(default_db_path())
        try:
            while not stop_event.is_set():
                now = int(time.time())
                try:
                    own.heartbeat(
                        run_id, now, now + lease_seconds,
                        prompt_eval_count=metrics_state["prompt_eval_count"],
                        eval_count=metrics_state["eval_count"],
                        total_ns=metrics_state["ollama_total_duration_ns"],
                        eval_ns=metrics_state["ollama_eval_duration_ns"],
                        model_calls=metrics_state["model_calls"],
                        tool_calls=metrics_state["tool_calls"],
                    )
                except Exception:
                    pass
                stop_event.wait(heartbeat_every)
        finally:
            own.close()

    hb = threading.Thread(target=heartbeat_loop, name=f"hb-{run_id}", daemon=True)
    hb.start()

    def on_mutation(entry) -> None:
        db.mark_mutation_started(run_id)

    worker = Worker(artifact_root=artifact_root)
    # Wrap broker's on_mutation to flag mutation_started.
    # We do this by patching after _build_broker: we instead pass via the
    # run() call indirectly by registering a hook. Simpler: subclass via
    # monkey-patch.
    orig_build = worker._build_broker
    def _patched(manifest, repo_root, artifact_dir, approval):
        b = orig_build(manifest, repo_root, artifact_dir, approval)
        b.on_mutation = on_mutation
        return b
    worker._build_broker = _patched  # type: ignore[assignment]

    started = time.time()
    try:
        result = worker.run(manifest, approval=approval)
        status = result.status
        reason_code = result.reason_code
        reason_text = result.reason_text
    except Exception as e:
        status = "FAILED"
        reason_code = type(e).__name__
        reason_text = str(e)
    finally:
        stop_event.set()
        try:
            hb.join(timeout=heartbeat_every + 2)
        except Exception:
            pass

    finished_at = int(time.time())
    post_worktree = git_worktree_sha(repo_root) if repo_root.exists() else ""
    try:
        db.finish_run(
            run_id, status, finished_at,
            post_worktree_sha256=post_worktree,
            error_code=None if status == "PASSED" else reason_code,
            error_text=None if status == "PASSED" else reason_text,
            wall_duration_ms=int((time.time() - started) * 1000),
        )
        db.update_status(task_id, TaskStatus(status),
                         final_reason_code=reason_code,
                         final_reason_text=reason_text)
        db.emit_event(session_id, "run_finished", task_id=task_id, run_id=run_id,
                      from_state="RUNNING", to_state=status,
                      details={"reason_code": reason_code, "reason_text": reason_text,
                               "artifact_dir": str(artifact_root)})
    finally:
        db.close()

    return QueuedRunResult(
        task_id=task_id, run_id=run_id, status=status,
        reason_code=reason_code, reason_text=reason_text,
        artifact_dir=str(artifact_root),
    )


def recovery_scan(now: int | None = None) -> list[dict[str, Any]]:
    """Scan stale RUNNING runs and apply the recovery policy.

    Rule (per sprint spec):
      - read_only stale: REVIEW_REQUIRED (caller may retry with policy)
      - source_mutation stale: REVIEW_REQUIRED, NEVER auto-retry
      - unreal_editor stale: REVIEW_REQUIRED
    """
    if now is None:
        now = int(time.time())
    db = Database(default_db_path())
    try:
        stale = db.find_stale_runs(now)
    finally:
        db.close()

    out: list[dict[str, Any]] = []
    for s in stale:
        db = Database(default_db_path())
        try:
            t = db.get_task(s["task_id"])
            if not t:
                continue
            ec = t["execution_class"] if "execution_class" in t.keys() else None
            manifest = TaskManifest.model_validate(json.loads(t["manifest_json"]))
            cls = manifest.execution_class
            if cls == ExecutionClass.UNREAL_EDITOR:
                new_status = TaskStatus.REVIEW_REQUIRED
                reason = "STALE_UNREAL"
            elif cls == ExecutionClass.SOURCE_MUTATION:
                new_status = TaskStatus.REVIEW_REQUIRED
                reason = "STALE_MUTATION_NEVER_RETRY"
            else:
                new_status = TaskStatus.REVIEW_REQUIRED
                reason = "STALE_READ_ONLY"
            db.update_status(s["task_id"], new_status,
                             final_reason_code=reason,
                             final_reason_text=f"stale run {s['run_id']}")
            db.finish_run(s["run_id"], "REVIEW_REQUIRED", now,
                          error_code=reason,
                          error_text=f"lease expired at {s.get('lease_expires_at')}")
            db.emit_event("recovery", "task_stale", task_id=s["task_id"], run_id=s["run_id"],
                          to_state=new_status.value, details={"reason": reason})
            out.append({"task_id": s["task_id"], "run_id": s["run_id"],
                        "execution_class": cls.value, "action": reason})
        finally:
            db.close()
    return out


def _canonical(m: TaskManifest) -> bytes:
    from .schemas import canonical_json
    return canonical_json(m)
