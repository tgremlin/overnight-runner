"""Tiny CLI. stdlib argparse."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .schemas import TaskManifest
from .worker import Approval, Worker


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
    """Phase 1 manual invocation. Held under the global runner lock so a
    human run cannot overlap a systemd-timer run.
    """
    from .runtime import runner_lock
    manifest_path = Path(args.manifest).resolve()
    raw = json.loads(manifest_path.read_text())
    m = TaskManifest.model_validate(raw)
    if args.repo:
        m.repo.path = str(Path(args.repo).resolve())
    worker = Worker()
    try:
        with runner_lock():
            result = worker.run(m)  # Phase 1 ad-hoc; ephemeral approval
    except RuntimeError as e:
        print(f"runner lock: {e}", file=sys.stderr)
        return 4
    print(json.dumps({
        "status": result.status,
        "reason_code": result.reason_code,
        "reason_text": result.reason_text,
        "turns": result.turns,
        "tool_calls": result.tool_calls,
        "artifacts_dir": result.artifacts_dir,
        "proposal_count": result.proposal_count,
        "applied_proposal_count": result.applied_proposal_count,
        "approval_ephemeral": result.extra.get("approval_ephemeral", True),
    }, indent=2))
    return 0 if result.status == "PASSED" else 1


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
    from .db import Database, default_db_path
    from .schemas import TaskStatus
    manifest_path = Path(args.manifest).resolve()
    raw = json.loads(manifest_path.read_text())
    m = TaskManifest.model_validate(raw)
    sha = _csha(m)
    db = Database(default_db_path())
    try:
        db.upsert_task(
            task_id=m.task_id, manifest_sha256=sha,
            manifest_json=manifest_path.read_text(),
            status=TaskStatus.PENDING_APPROVAL,
            execution_class=m.execution_class.value,
            dependencies_json=json.dumps([d.model_dump(mode="json") for d in m.dependencies]),
        )
        db.emit_event("cli", "task_imported", task_id=m.task_id,
                      details={"sha256": sha, "execution_class": m.execution_class.value})
    finally:
        db.close()
    print(f"imported task_id={m.task_id} sha256={sha} status=PENDING_APPROVAL class={m.execution_class.value}")
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    """Record an independently-stored approval envelope and transition to APPROVED.

    Resolves model digest via Ollama. Refuses approval if model cannot be
    resolved or digest has changed since import.
    """
    from .db import Database, default_db_path
    from .ollama_client import OllamaClient
    from .runtime import runtime_fingerprint
    from .safety import git_head
    from .schemas import TaskStatus
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
        # Resolve model digest.
        client = OllamaClient()
        digest = client.model_digest(manifest.model_profile.model_name)
        if not digest:
            print(
                f"refusing to approve: model {manifest.model_profile.model_name!r} not installed in Ollama",
                file=sys.stderr,
            )
            return 3
        env = {
            "manifest_sha256": t["manifest_sha256"],
            "approved_repo_head": head,
            "approved_runtime_sha256": runtime_fingerprint(
                [Path(__file__).parent, Path(__file__).parent.parent / "prompts"]
            ).sha256,
            "approved_model_name": manifest.model_profile.model_name,
            "approved_model_digest": digest,
        }
        db.approve_task(args.task_id, approved_by="manual-cli", approval_envelope=env)
        db.emit_event("cli", "task_approved", task_id=args.task_id,
                      details={"head": head, "model": manifest.model_profile.model_name,
                               "digest": digest})
    finally:
        db.close()
    print(f"approved task_id={args.task_id} model={manifest.model_profile.model_name} digest={digest[:12]}")
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
    """Claim and run the next eligible APPROVED task under the global lock."""
    from .db import Database, default_db_path
    from .runner import execute_claimed_task
    from .runtime import runner_lock
    from .schemas import TaskStatus
    try:
        with runner_lock():
            db = Database(default_db_path())
            try:
                res = execute_claimed_task(db, execution_class_filter=None)
            finally:
                db.close()
    except RuntimeError as e:
        print(f"runner lock: {e}", file=sys.stderr)
        return 4
    if res is None:
        print("(no eligible approved task)")
        return 0
    print(json.dumps({
        "task_id": res.task_id, "run_id": res.run_id, "attempt": res.attempt_no,
        "status": res.status, "reason_code": res.reason_code,
        "reason_text": res.reason_text, "artifact_dir": res.artifact_dir,
    }, indent=2))
    return 0 if res.status == "PASSED" else 1


def cmd_recover(args: argparse.Namespace) -> int:
    from .runner import recovery_scan
    out = recovery_scan()
    if not out:
        print("(no stale runs)")
        return 0
    print(json.dumps(out, indent=2))
    return 0


def cmd_run_nightly(args: argparse.Namespace) -> int:
    """Run the full nightly session under the global runner lock."""
    from .db import Database, default_db_path
    from .runner import run_nightly
    from .runtime import runner_lock
    session_id = args.session_id or None
    try:
        with runner_lock():
            db = Database(default_db_path())
            try:
                db.close()
            except Exception:
                pass
            res = run_nightly(session_id=session_id)
    except RuntimeError as e:
        print(f"runner lock: {e}", file=sys.stderr)
        return 4
    summary_dir = Path.home() / ".local" / "state" / "overnight-runner" / "sessions" / res.session_id
    print(json.dumps({
        "session_id": res.session_id,
        "started_at": res.started_at,
        "finished_at": res.finished_at,
        "wall_duration_seconds": int(res.finished_at - res.started_at),
        "stop_reason": res.stop_reason,
        "tasks_attempted": res.tasks_attempted,
        "tasks_passed": res.tasks_passed,
        "tasks_failed": res.tasks_failed,
        "tasks_blocked": res.tasks_blocked,
        "tasks_review_required": res.tasks_review_required,
        "read_only_attempted": res.read_only_attempted,
        "mutation_attempted": res.mutation_attempted,
        "summary_dir": str(summary_dir),
    }, indent=2))
    print(f"summary written to: {summary_dir}/summary.json and summary.md", file=sys.stderr)
    return 0


def _p07_db():
    from .db import Database, default_db_path
    return Database(default_db_path())


def cmd_hermes_status(args: argparse.Namespace) -> int:
    """P07 read-only: authoritative campaign/status projection (thin)."""
    from .p07 import project_campaign_status
    db = _p07_db()
    try:
        out = project_campaign_status(db, campaign_id=args.campaign_id)
    finally:
        db.close()
    print(json.dumps(out, indent=2))
    return 0


def cmd_hermes_cards(args: argparse.Namespace) -> int:
    """P07 read-only: rebuildable NON-authoritative projection cards (thin)."""
    from .p07 import project_hermes_cards
    db = _p07_db()
    try:
        out = project_hermes_cards(db, campaign_id=args.campaign_id)
    finally:
        db.close()
    print(json.dumps(out, indent=2))
    return 0


def cmd_hermes_capacity(args: argparse.Namespace) -> int:
    """P07 read-only: provider/account cooldown state (thin)."""
    from .p07 import list_cooldowns
    db = _p07_db()
    try:
        out = list_cooldowns(db)
    finally:
        db.close()
    print(json.dumps({"cooldowns": out}, indent=2))
    return 0


def cmd_hermes_wake(args: argparse.Namespace) -> int:
    """P07 bounded control: re-evaluate + at most one REAL claim (thin)."""
    from .p07 import wake_tick
    db = _p07_db()
    try:
        out = wake_tick(db, campaign_id=args.campaign_id,
                        trigger_id=args.trigger_id or "")
    finally:
        db.close()
    print(json.dumps(out, indent=2))
    return 0


def cmd_hermes_tick(args: argparse.Namespace) -> int:
    """P07 bounded control: runner-owned due-work tick over active campaigns."""
    from .p07 import hermes_tick
    db = _p07_db()
    try:
        out = hermes_tick(db, trigger_id=args.trigger_id or "")
    finally:
        db.close()
    print(json.dumps(out, indent=2))
    return 0


def cmd_hermes_job(args: argparse.Namespace) -> int:
    """P07 read-only: public durable job polling surface (thin)."""
    from .p07 import job_state
    db = _p07_db()
    try:
        out = job_state(db, args.job_id)
    finally:
        db.close()
    if out is None:
        print(json.dumps({"job_id": args.job_id, "state": "UNKNOWN"}, indent=2))
        return 1
    print(json.dumps(out, indent=2))
    return 0


def cmd_hermes_control(args: argparse.Namespace) -> int:
    """P07 bounded control: short-control request -> durable job id (thin)."""
    from .p07 import job_state, request_control
    db = _p07_db()
    try:
        job_id = request_control(db, operation=args.operation,
                                 campaign_id=args.campaign_id or "",
                                 window_key=args.window_key or "")
        out = job_state(db, job_id)
    finally:
        db.close()
    print(json.dumps({"job_id": job_id, "state": (out or {}).get("state", ""),
                      "schema_version": "trio.hermes-control.v1"}, indent=2))
    return 0


def cmd_hermes_adapters(args: argparse.Namespace) -> int:
    """P07 read-only: the schema-defined adapter surface Hermes may call."""
    from .p07 import ADAPTER_SCHEMA
    print(json.dumps(ADAPTER_SCHEMA, indent=2))
    return 0


def _csha(m: TaskManifest) -> str:
    from .schemas import canonical_sha
    return canonical_sha(m)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="overnight-runner", description="Bounded local-model work pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("validate", help="Validate a manifest JSON")
    s.add_argument("manifest")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("run", help="Phase 1: manually invoke a single task (ephemeral approval)")
    s.add_argument("manifest")
    s.add_argument("--repo", default=None)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("import", help="Phase 2: validate + queue")
    s.add_argument("manifest")
    s.set_defaults(func=cmd_import)

    s = sub.add_parser("approve", help="Phase 2: record approval envelope + transition to APPROVED")
    s.add_argument("task_id")
    s.set_defaults(func=cmd_approve)

    s = sub.add_parser("status", help="Phase 2: list tasks")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("summary", help="Phase 2: morning summary")
    s.set_defaults(func=cmd_summary)

    s = sub.add_parser("run-next", help="Phase 2: claim + run next eligible APPROVED task under global lock")
    s.set_defaults(func=cmd_run_next)

    s = sub.add_parser("run-nightly", help="Run the full nightly session under global lock")
    s.add_argument("--session-id", default=None, help="Override session id")
    s.set_defaults(func=cmd_run_nightly)

    s = sub.add_parser("recover", help="Phase 2: scan for stale RUNNING runs + orphan RUNNING tasks")
    s.set_defaults(func=cmd_recover)

    s = sub.add_parser("doctor", help="Check Ollama + Python")
    s.set_defaults(func=cmd_doctor)

    # ---- P07 Hermes foreman adapter surfaces (narrow, schema-defined) ----
    s = sub.add_parser("hermes-status", help="P07: authoritative campaign status projection")
    s.add_argument("campaign_id")
    s.set_defaults(func=cmd_hermes_status)

    s = sub.add_parser("hermes-cards", help="P07: rebuildable non-authoritative projection cards")
    s.add_argument("campaign_id")
    s.set_defaults(func=cmd_hermes_cards)

    s = sub.add_parser("hermes-capacity", help="P07: provider/account cooldown state")
    s.set_defaults(func=cmd_hermes_capacity)

    s = sub.add_parser("hermes-wake", help="P07: bounded wake (one real claim per obligation)")
    s.add_argument("campaign_id")
    s.add_argument("--trigger-id", dest="trigger_id", default=None, help="external evidence id")
    s.set_defaults(func=cmd_hermes_wake)

    s = sub.add_parser("hermes-tick", help="P07: runner-owned due-work tick over active campaigns")
    s.add_argument("--trigger-id", dest="trigger_id", default=None)
    s.set_defaults(func=cmd_hermes_tick)

    s = sub.add_parser("hermes-control", help="P07: short-control request -> durable job id")
    s.add_argument("operation")
    s.add_argument("--campaign-id", dest="campaign_id", default=None)
    s.add_argument("--window-key", dest="window_key", default=None)
    s.set_defaults(func=cmd_hermes_control)

    s = sub.add_parser("hermes-job", help="P07: public durable job polling surface")
    s.add_argument("job_id")
    s.set_defaults(func=cmd_hermes_job)

    s = sub.add_parser("hermes-adapters", help="P07: schema-defined adapter surface")
    s.set_defaults(func=cmd_hermes_adapters)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
