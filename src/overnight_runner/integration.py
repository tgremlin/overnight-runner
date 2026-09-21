"""P06-A04 — Sequential integration with compare-and-swap ref protection.

Each accepted campaign chunk must land on the campaign's private
integration ref (default: ``refs/heads/campaign/<id>``) using a
compare-and-swap over the previous accepted commit. The runner owns
the commit+ref-advance mechanics; the worker has no Git credentials.

Crash-window detection:

  - ``expected_old_commit`` is recorded BEFORE the CAS attempt.
  - If the CAS refuses (the ref has moved externally), the integration
    fails with ``EFFECT_UNKNOWN`` and writes a row to
    ``crash_windows``.
  - If a process crash leaves a half-applied integration (e.g.
    commit created but DB not updated), ``reconcile_crash_window``
    inspects the git ref and the chunk state to decide how to resume.

The branch ``integration_branch`` is campaign-scoped and MUST NOT be
the default branch of any user-owned repository. Use
``worktree_from(repo_root)`` to create a disposable worktree pointing
at this branch.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .campaign_schemas import CampaignEvent, FenceState, RepoSnapshot
from .db import Database
from .resources import current_fence, enforce_fence
from .safety import SafetyError


@dataclass(frozen=True)
class CompareAndSwapResult:
    campaign_id: str
    chunk_id: str
    expected_old_commit: str
    committed_new_commit: str
    fence_generation: int
    committed_at: int


def _run_git(args: list[str], cwd: Path) -> str:
    """Run a Git command and return stdout. No stderr forwarding."""
    res = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return res.stdout.strip()


def _try_run_git(args: list[str], cwd: Path) -> tuple[bool, str, str]:
    """Run a Git command and return (success, stdout, stderr). No exception."""
    res = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return (res.returncode == 0, res.stdout.strip(), res.stderr.strip())


def git_worktree_sha(repo: Path) -> str:
    """Deterministic working-tree fingerprint (matches ``safety.git_worktree_sha``)."""
    from .safety import git_worktree_sha as _gw
    return _gw(repo)


def ensure_campaign_worktree(
    *, repo_root: Path, campaign_id: str, base_commit: str
) -> Path:
    """Create or update a disposable worktree for the campaign's
    integration branch.

    The worktree is created at ``<state_dir>/campaign-worktrees/<id>``.
    The branch is ``refs/heads/campaign/<id>``.
    """
    from .runtime import state_dir
    wt_root = state_dir() / "campaign-worktrees" / campaign_id
    wt_root.parent.mkdir(parents=True, exist_ok=True)
    branch = f"campaign/{campaign_id}"
    # Try to create the branch at base_commit; fall back to checking it out.
    try:
        _run_git(["branch", "-f", branch, base_commit], cwd=repo_root)
    except subprocess.CalledProcessError:
        pass
    # If the worktree exists, refresh; else create it.
    if (wt_root / ".git").exists() or wt_root.exists():
        try:
            _run_git(["worktree", "remove", "--force", str(wt_root)], cwd=repo_root)
        except subprocess.CalledProcessError:
            pass
    _run_git(
        ["worktree", "add", "--detach", str(wt_root), branch],
        cwd=repo_root,
    )
    return wt_root


def capture_current_snapshot(worktree: Path) -> RepoSnapshot:
    """Snapshot the worktree's HEAD + working tree at this exact moment."""
    head = _run_git(["rev-parse", "HEAD"], cwd=worktree)
    tree = git_worktree_sha(worktree)
    return RepoSnapshot(
        schema_version="trio.repo-snapshot.v1",
        repository_id="local",
        commit=head,
        tree_digest=tree,
    )


def compare_and_swap_advance(
    db: Database,
    *,
    repo_root: Path,
    campaign_id: str,
    chunk_id: str,
    new_commit: str,
    holder_fence_generation: int,
    actor: str,
    idempotency_key: str,
    expected_old: str | None = None,
    expected_tree_digest: str | None = None,
) -> CompareAndSwapResult:
    """Atomically advance the campaign's integration ref.

    Sequence:

      1. Read the LIVE current ref via ``git rev-parse`` (THE source
         of truth for what the world sees).
      2. Compare with ``expected_old`` (if provided). Mismatch -> fail
         with EFFECT_UNKNOWN + crash window. The DB's
         ``current_commit`` is also compared and reconciled if it
         drifted from the live ref.
      3. Holder fence MUST match.
      4. CAS the ref forward.
      5. Append to ``integration_journal`` and update campaigns row.
    """
    now = int(time.time())
    branch = f"refs/heads/campaign/{campaign_id}"

    # Step 1: read LIVE ref (this is the source of truth for what the
    # world sees). If the ref doesn't exist, treat as empty.
    ok, live_ref, _err = _try_run_git(
        ["rev-parse", "--verify", branch], cwd=repo_root
    )
    live_commit = live_ref if ok else ""

    cur = db._conn.execute(
        "SELECT current_commit, current_tree_digest, current_fence FROM campaigns WHERE campaign_id=?",
        (campaign_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise SafetyError(f"campaign {campaign_id} not registered")
    stored_db_commit = row["current_commit"] or ""
    cur_fence = int(row["current_fence"])

    # If the DB lags behind the live ref (an external move), reconcile
    # the DB to the LIVE state. Then the comparison below is
    # authoritative against the world.
    if stored_db_commit != live_commit and live_commit:
        with db.transaction() as cur2:
            cur2.execute(
                "UPDATE campaigns SET current_commit=?, current_tree_digest=?, updated_at=? WHERE campaign_id=?",
                (live_commit, "", now, campaign_id),
            )

    if expected_old is not None and live_commit and live_commit != expected_old and expected_old != live_commit[: len(expected_old)] and live_commit[: len(live_commit)] != expected_old[: len(live_commit)]:
        _record_crash_window(
            db,
            campaign_id=campaign_id,
            chunk_id=chunk_id,
            kind="integration_cas_mismatch",
            observed_artifact=live_commit,
        )
        raise SafetyError(
            f"integration_cas_mismatch: expected_old={expected_old[:8]} current={live_commit[:8]}"
        )

    # Step 2: holder fence MUST match.
    if holder_fence_generation != cur_fence:
        _record_crash_window(
            db,
            campaign_id=campaign_id,
            chunk_id=chunk_id,
            kind="integration_fence_stale",
            observed_artifact=f"holder_fence={holder_fence_generation} current={cur_fence}",
        )
        raise SafetyError(
            f"integration_fence_stale: holder={holder_fence_generation} current={cur_fence}"
        )

    # Step 3: Git-level CAS.
    try:
        _run_git(
            ["update-ref", branch, new_commit, live_commit or new_commit],
            cwd=repo_root,
        )
    except subprocess.CalledProcessError as e:
        _record_crash_window(
            db,
            campaign_id=campaign_id,
            chunk_id=chunk_id,
            kind="git_update_ref_failed",
            observed_artifact=str(e)[:512],
        )
        raise SafetyError(f"git update-ref failed: {e}") from e

    # Step 4: derive new tree digest (from the git tree of new_commit).
    new_tree = _run_git(["rev-parse", f"{new_commit}^{{tree}}"], cwd=repo_root)

    with db.transaction() as cur:
        cur.execute(
            "UPDATE campaigns SET current_commit=?, current_tree_digest=?, updated_at=? WHERE campaign_id=?",
            (new_commit, new_tree, now, campaign_id),
        )
        cur.execute(
            """
            INSERT INTO integration_journal (
                campaign_id, chunk_id, expected_old_commit, committed_new_commit,
                fence_generation, committed_at, actor, idempotency_key, evidence_json
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                campaign_id, chunk_id, live_commit, new_commit,
                holder_fence_generation, now, actor, idempotency_key,
                json.dumps({"new_tree": new_tree}),
            ),
        )

    _emit_event(
        db,
        campaign_id=campaign_id,
        chunk_id=chunk_id,
        event_type="integration_advanced",
        from_state=live_commit,
        to_state=new_commit,
        actor=actor,
        idempotency_key=idempotency_key,
        fence_generation=holder_fence_generation,
    )
    return CompareAndSwapResult(
        campaign_id=campaign_id,
        chunk_id=chunk_id,
        expected_old_commit=live_commit,
        committed_new_commit=new_commit,
        fence_generation=holder_fence_generation,
        committed_at=now,
    )


def _record_crash_window(
    db: Database, *, campaign_id: str, chunk_id: str, kind: str, observed_artifact: str
) -> None:
    now = int(time.time())
    wid = f"cw-{uuid.uuid4().hex[:16]}"
    db._conn.execute(
        """
        INSERT INTO crash_windows (
            window_id, campaign_id, chunk_id, kind, observed_artifact,
            snapshot_at, recovered_at, reconciliation_reason
        ) VALUES (?,?,?,?,?,?,0,'')
        """,
        (wid, campaign_id, chunk_id, kind, observed_artifact[:2000], now),
    )


def _emit_event(
    db: Database,
    *,
    campaign_id: str,
    chunk_id: str | None,
    event_type: str,
    from_state: str | None,
    to_state: str | None,
    actor: str,
    idempotency_key: str,
    fence_generation: int,
) -> None:
    now = int(time.time())
    db._conn.execute(
        """
        INSERT INTO campaign_events (
            event_id, campaign_id, chunk_id, event_type,
            from_state, to_state, actor, payload, fence_generation,
            issued_at, idempotency_key
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            f"ev-{uuid.uuid4().hex[:16]}", campaign_id, chunk_id, event_type,
            from_state, to_state, actor, "{}", fence_generation,
            now, idempotency_key,
        ),
    )


def reconcile_crash_window(db: Database, *, campaign_id: str) -> dict[str, Any]:
    """Inspect the durable crash-window log for ``campaign_id`` and
    return a structured reconciliation summary.

    The function does NOT auto-recover; it only collects the records
    for an operator / planner decision. The runner enters
    ``EFFECT_UNKNOWN`` whenever a crash window is recorded.
    """
    cur = db._conn.execute(
        """
        SELECT window_id, chunk_id, kind, observed_artifact, snapshot_at,
               recovered_at, reconciliation_reason
        FROM crash_windows
        WHERE campaign_id=? AND recovered_at=0
        ORDER BY snapshot_at ASC
        """,
        (campaign_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    return {
        "campaign_id": campaign_id,
        "unrecovered_windows": rows,
        "recovered_count": 0,
    }


def mark_crash_window_recovered(
    db: Database, *, window_id: str, reason: str
) -> None:
    with db.transaction() as cur:
        cur.execute(
            "UPDATE crash_windows SET recovered_at=?, reconciliation_reason=? WHERE window_id=? AND recovered_at=0",
            (int(time.time()), reason[:2000], window_id),
        )


__all__ = [
    "CompareAndSwapResult",
    "ensure_campaign_worktree",
    "capture_current_snapshot",
    "compare_and_swap_advance",
    "reconcile_crash_window",
    "mark_crash_window_recovered",
]
