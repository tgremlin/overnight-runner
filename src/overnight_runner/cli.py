"""Tiny CLI. stdlib argparse."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .schemas import TaskManifest
from .worker import Worker


def cmd_validate(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    raw = json.loads(manifest_path.read_text())
    try:
        m = TaskManifest.model_validate(raw)
    except Exception as e:
        print(f"INVALID: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(f"OK  task_id={m.task_id}  class={m.execution_class.value}  sha256={_csha(m)}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    raw = json.loads(manifest_path.read_text())
    m = TaskManifest.model_validate(raw)
    if args.repo:
        m.repo.path = str(Path(args.repo).resolve())
    worker = Worker()
    result = worker.run(m)
    print(json.dumps({
        "status": result.status,
        "reason_code": result.reason_code,
        "reason_text": result.reason_text,
        "turns": result.turns,
        "tool_calls": result.tool_calls,
        "artifacts_dir": result.artifacts_dir,
        "proposal_count": result.proposal_count,
        "applied_proposal_count": result.applied_proposal_count,
    }, indent=2))
    return 0 if result.status in ("PASSED",) else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    from .ollama_client import DEFAULT_HOST
    import urllib.request
    try:
        with urllib.request.urlopen(f"{DEFAULT_HOST}/api/tags", timeout=3) as r:
            data = json.loads(r.read().decode())
        models = [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"Ollama unreachable: {e}", file=sys.stderr)
        return 1
    print(f"Ollama OK at {DEFAULT_HOST}")
    for n in models:
        print(f"  model: {n}")
    print(f"python: {sys.version.split()[0]}")
    print(f"cwd: {os.getcwd()}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    """Phase 2: validate manifest and queue into SQLite."""
    from .db import Database, default_db_path
    from .schemas import TaskStatus
    manifest_path = Path(args.manifest).resolve()
    raw = json.loads(manifest_path.read_text())
    m = TaskManifest.model_validate(raw)
    sha = _csha(m)
    db = Database(default_db_path())
    try:
        db.upsert_task(
            task_id=m.task_id,
            manifest_sha256=sha,
            manifest_json=manifest_path.read_text(),
            status=TaskStatus.PENDING_APPROVAL,
        )
        db.emit_event("cli", "task_imported", task_id=m.task_id, details={"sha256": sha})
    finally:
        db.close()
    print(f"imported task_id={m.task_id} sha256={sha} status=PENDING_APPROVAL")
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    """Phase 2: bind approval envelope and mark APPROVED."""
    from .db import Database, default_db_path
    from .schemas import TaskStatus
    from .safety import git_head
    from .runtime import runtime_fingerprint, state_dir
    db = Database(default_db_path())
    try:
        t = db.get_task(args.task_id)
        if not t:
            print(f"unknown task_id: {args.task_id}", file=sys.stderr)
            return 2
        manifest = TaskManifest.model_validate(json.loads(t["manifest_json"]))
        repo_root = Path(manifest.repo.path).resolve()
        head = git_head(repo_root)
        if not head:
            print("repo has no commits", file=sys.stderr)
            return 2
        env = {
            "manifest_sha256": t["manifest_sha256"],
            "approved_repo_head": head,
            "approved_runtime_sha256": runtime_fingerprint(
                [Path(__file__).parent, Path(__file__).parent.parent / "prompts"]
            ).sha256,
            "approved_model_digest": None,
        }
        db.approve_task(args.task_id, approved_by="manual-cli", approval_envelope=env)
        db.emit_event("cli", "task_approved", task_id=args.task_id, details={"head": head})
    finally:
        db.close()
    print(f"approved task_id={args.task_id}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .db import Database, default_db_path
    db = Database(default_db_path())
    try:
        rows = db.list_tasks()
    finally:
        db.close()
    if not rows:
        print("(no tasks)")
        return 0
    for r in rows:
        print(f"  {r['task_id']:32s}  {r['status']:16s}  priority={r['priority']:4d}  sha={r['manifest_sha256'][:12]}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    from .db import Database, default_db_path
    db = Database(default_db_path())
    try:
        s = db.summary()
    finally:
        db.close()
    print(json.dumps(s, indent=2))
    return 0


def cmd_run_next(args: argparse.Namespace) -> int:
    """Phase 2: claim and run the next approved task."""
    from .db import Database, default_db_path
    from .schemas import TaskStatus
    from .runtime import runner_lock
    db = Database(default_db_path())
    try:
        # Atomic claim: pick first APPROVED task and transition to RUNNING.
        with db.transaction() as cur:
            cur.execute("SELECT * FROM tasks WHERE status=? ORDER BY priority ASC, created_at ASC LIMIT 1", (TaskStatus.APPROVED.value,))
            row = cur.fetchone()
            if not row:
                print("(no approved tasks)")
                return 0
            task = dict(row)
            cur.execute("UPDATE tasks SET status=?, updated_at=? WHERE task_id=?",
                        (TaskStatus.RUNNING.value, int(time.time()), task["task_id"]))
    finally:
        pass
    manifest = TaskManifest.model_validate(json.loads(task["manifest_json"]))
    try:
        with runner_lock():
            worker = Worker()
            res = worker.run(manifest)
        # Final state
        final_status = TaskStatus(res.status)
        db.update_status(task["task_id"], final_status,
                         final_reason_code=res.reason_code,
                         final_reason_text=res.reason_text)
        db.emit_event("cli", "task_finished", task_id=task["task_id"],
                      from_state=TaskStatus.RUNNING.value, to_state=final_status.value,
                      details={"reason_code": res.reason_code, "reason_text": res.reason_text,
                               "artifact_dir": res.artifacts_dir})
        print(json.dumps({
            "task_id": task["task_id"],
            "status": res.status,
            "reason_code": res.reason_code,
            "reason_text": res.reason_text,
            "artifacts_dir": res.artifacts_dir,
        }, indent=2))
        return 0 if res.status == "PASSED" else 1
    finally:
        db.close()


def _csha(m: TaskManifest) -> str:
    from .schemas import canonical_sha
    return canonical_sha(m)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="overnight-runner", description="Bounded local-model work pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("validate", help="Validate a manifest JSON")
    s.add_argument("manifest")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("run", help="Phase 1: manually invoke a single task")
    s.add_argument("manifest")
    s.add_argument("--repo", default=None, help="Override repo.path")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("import", help="(Phase 2) Validate + queue")
    s.add_argument("manifest")
    s.set_defaults(func=cmd_import)

    s = sub.add_parser("approve", help="(Phase 2) Bind approval envelope + APPROVED")
    s.add_argument("task_id")
    s.set_defaults(func=cmd_approve)

    s = sub.add_parser("status", help="(Phase 2) List tasks")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("summary", help="(Phase 2) Morning summary")
    s.set_defaults(func=cmd_summary)

    s = sub.add_parser("run-next", help="(Phase 2) Claim + run next APPROVED task")
    s.set_defaults(func=cmd_run_next)

    s = sub.add_parser("doctor", help="Check Ollama + Python")
    s.set_defaults(func=cmd_doctor)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
