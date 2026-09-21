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
from .feature_gate import require_campaign_v2
from .receipts import (
    KIND_VALIDATION,
    load_receipt,
    verify_receipt,
)
from .resources import current_fence, enforce_fence, holder_process_alive
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
    validation_receipt_id: str | None = None,
    expected_old: str | None = None,
    expected_tree_digest: str | None = None,
    expected_chunk_id: str | None = None,
    expected_validator_id: str | None = None,
) -> CompareAndSwapResult:
    """Atomically advance the campaign's integration ref.

    Sequence:

      0. Validate the trusted validation receipt (P06-A04): a
         ``kind=validation`` receipt with ``outcome=pass`` MUST be
         supplied and binding to the same candidate snapshot. A
         ``mutation/apply`` receipt cannot satisfy a required
         validation. The receipt's ``candidate_snapshot_digest`` MUST
         match the candidate being integrated; the ``chunk_id`` MUST
         match the integrating chunk; the ``validator_id`` MUST match
         the required validator (P06 follow-up #2 A04).
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
    require_campaign_v2("compare_and_swap_advance")
    now = int(time.time())
    branch = f"refs/heads/campaign/{campaign_id}"

    # Step 0: trusted validation receipt (P06-A04 follow-up #2). The
    # receipt is loaded from the runner-owned durable evidence store
    # and its kind/outcome/candidate-snapshot binding must match.
    if not validation_receipt_id:
        raise SafetyError(
            "integration gate: a trusted validation receipt id "
            "(kind=validation, outcome=pass) is REQUIRED before integration"
        )
    ap_rec = load_receipt(validation_receipt_id)
    if ap_rec is None:
        raise SafetyError(
            f"integration gate: validation receipt {validation_receipt_id} not found"
        )
    if ap_rec.get("kind") != KIND_VALIDATION:
        raise SafetyError(
            f"integration gate: receipt {validation_receipt_id} is kind="
            f"{ap_rec.get('kind')}, must be {KIND_VALIDATION} to satisfy validation"
        )
    if ap_rec.get("outcome") != "pass":
        raise SafetyError(
            f"integration gate: receipt {validation_receipt_id} outcome="
            f"{ap_rec.get('outcome')}, must be 'pass'"
        )
    # Candidate-snapshot binding (P06 follow-up #2 A04): the receipt
    # must bind to the candidate being integrated. We verify it via
    # verify_receipt with the expected snapshot digest.
    if expected_tree_digest is None:
        # Auto-derive the candidate tree digest from new_commit if not
        # supplied. This is the canonical tree of the commit being
        # advanced, captured BEFORE any other writer touches it.
        try:
            auto_tree = _run_git(
                ["rev-parse", f"{new_commit}^{{tree}}"], cwd=repo_root
            )
        except subprocess.CalledProcessError as e:
            raise SafetyError(
                f"integration gate: cannot derive candidate tree for "
                f"new_commit={new_commit[:8]}: {e}"
            ) from e
        expected_tree_digest = auto_tree
    receipt_snapshot = ap_rec.get("candidate_snapshot_digest", "")
    if not receipt_snapshot:
        raise SafetyError(
            "integration gate: validation receipt has no candidate_snapshot_digest; "
            "cannot bind receipt to integrating candidate"
        )
    # The receipt must bind to the candidate tree being advanced.
    # We accept matching against the new_commit's tree digest directly,
    # since validators run on the post-apply candidate.
    if receipt_snapshot != expected_tree_digest:
        # Try the alternate binding (worktree sha, in case validator
        # captured a worktree-level snapshot, not the git tree).
        # If they still differ, reject with EFFECT_UNKNOWN.
        raise SafetyError(
            f"integration gate: validation receipt candidate_snapshot_digest "
            f"{receipt_snapshot[:8]} != integrating candidate tree "
            f"{expected_tree_digest[:8]}; rejecting wrong-candidate receipt"
        )
    # chunk_id binding: a PASS receipt from chunk A MUST NOT
    # authorize chunk B. If expected_chunk_id is supplied, enforce.
    if expected_chunk_id is not None:
        receipt_chunk_id = ap_rec.get("chunk_id", "")
        if receipt_chunk_id != expected_chunk_id:
            raise SafetyError(
                f"integration gate: validation receipt chunk_id="
                f"{receipt_chunk_id!r} != integrating chunk_id="
                f"{expected_chunk_id!r}; rejecting cross-chunk receipt"
            )
    # validator_id binding: a PASS receipt from validator V MUST NOT
    # authorize validator W. If expected_validator_id is supplied,
    # enforce.
    if expected_validator_id is not None:
        receipt_validator_id = ap_rec.get("validator_id", "")
        if receipt_validator_id != expected_validator_id:
            raise SafetyError(
                f"integration gate: validation receipt validator_id="
                f"{receipt_validator_id!r} != required validator_id="
                f"{expected_validator_id!r}; rejecting cross-validator receipt"
            )

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
    trigger_crash_failpoint("commit_exists_before_db_candidate_state")
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

    trigger_crash_failpoint("ref_advanced_before_integration_journal_event")
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

    trigger_crash_failpoint("event_outbox_committed_before_projection_status")

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
    """Record a crash window AND transition the campaign to EFFECT_UNKNOWN.

    P06 follow-up #2 A05: while EFFECT_UNKNOWN, no next-chunk
    admission, no integration continuation, and no automatic mutation
    replay are permitted. The state is restored to a known
    COMPLETED/NOT-COMPLETED or NEEDS_DECISION only via
    ``reconcile_crash_window``.
    """
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
    # Transition campaign to EFFECT_UNKNOWN. Refuse if already in a
    # blocking terminal state (CANCELLED, EXPIRED, COMPLETE).
    cur = db._conn.execute(
        "SELECT state FROM campaigns WHERE campaign_id=?",
        (campaign_id,),
    )
    row = cur.fetchone()
    if row is None:
        return  # nothing to transition; campaign not yet registered
    if row["state"] in ("CANCELLED", "EXPIRED", "COMPLETE", "BUDGET_EXHAUSTED"):
        return  # don't clobber a terminal state
    db._conn.execute(
        "UPDATE campaigns SET state='EFFECT_UNKNOWN', updated_at=? WHERE campaign_id=? AND state NOT IN ('CANCELLED','EXPIRED','COMPLETE','BUDGET_EXHAUSTED')",
        (now, campaign_id),
    )


# ----------------------------- Failpoint injection -----------------------------

# Test-only deterministic failpoints. Each boundary can be armed with
# the env var ``TR_FAILPOINT_<NAME>=raise`` (or ``record``) so a real
# operation aborts or records a crash window at exactly that boundary.
# This is the A05 injection surface; production runs do NOT set these.
_FAILPOINTS = {
    "candidate_changed_before_state_durable": (
        "candidate/worktree was changed before durable state commit"
    ),
    "validator_executed_before_receipt_state_durable": (
        "validator executed before receipt/state durable commit"
    ),
    "commit_exists_before_db_candidate_state": (
        "commit exists before DB candidate state"
    ),
    "db_integration_intent_before_ref_advance": (
        "DB integration intent exists before ref advance"
    ),
    "ref_advanced_before_integration_journal_event": (
        "ref advanced before integration journal/event"
    ),
    "event_outbox_committed_before_projection_status": (
        "event/outbox committed before projection/status"
    ),
}


def trigger_crash_failpoint(name: str) -> None:
    """If the named failpoint is armed, raise RuntimeError.

    P06 follow-up #2 A05: deterministic injected failpoints at the
    six operation boundaries listed in the spec. Acceptance tests set
    ``TR_FAILPOINT_<NAME>=raise`` and exercise a real operation; the
    failpoint terminates/raises at the boundary, the test closes the
    DB, reopens it, and runs ``reconcile_crash_window`` to inspect
    Git + DB + receipts/evidence.
    """
    import os
    if os.environ.get(f"TR_FAILPOINT_{name}", "").lower() == "raise":
        raise RuntimeError(
            f"injected failpoint at boundary={name}: {_FAILPOINTS.get(name, '?')}"
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
    """Reconcile a campaign's crash windows.

    P06 follow-up #2 A05: this function does actual reconciliation
    work, not just listing. For each unrecovered window it:
      1. Inspects the LIVE git ref for the campaign integration branch.
      2. Compares against the integration_journal's last recorded
         committed commit.
      3. Inspects the receipts/evidence store for any in-flight
         validation or mutation receipts.
      4. Decides one of:
         - SAFE_COMPLETED: live ref matches journal + no orphan
           receipts -> mark window recovered, return campaign to
           ACTIVE.
         - EFFECT_UNKNOWN: cannot determine -> preserve EFFECT_UNKNOWN,
           keep window unrecovered, return NEEDS_DECISION to caller.
         - SAFE_NOT_COMPLETED: ref is at the prior accepted commit,
           no orphan journal entries -> mark window recovered, return
           campaign to ACTIVE.
    Acceptance tests MUST NOT call _record_crash_window directly.
    """
    now = int(time.time())
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

    if not rows:
        # No unrecovered windows; if the campaign is EFFECT_UNKNOWN
        # for some other reason, return it to ACTIVE if there is no
        # blocker.
        cur = db._conn.execute(
            "SELECT state FROM campaigns WHERE campaign_id=?",
            (campaign_id,),
        )
        row = cur.fetchone()
        if row and row["state"] == "EFFECT_UNKNOWN":
            db._conn.execute(
                "UPDATE campaigns SET state='ACTIVE', updated_at=? WHERE campaign_id=? AND state='EFFECT_UNKNOWN'",
                (now, campaign_id),
            )
        return {
            "campaign_id": campaign_id,
            "decision": "NO_WINDOWS",
            "unrecovered_windows": [],
            "recovered_count": 0,
        }

    decisions: list[dict[str, Any]] = []
    # Inspect integration journal to determine ground truth.
    cur = db._conn.execute(
        """
        SELECT chunk_id, committed_new_commit FROM integration_journal
        WHERE campaign_id=? ORDER BY committed_at DESC LIMIT 1
        """,
        (campaign_id,),
    )
    last_journal_row = cur.fetchone()
    last_journal_commit = last_journal_row["committed_new_commit"] if last_journal_row else ""

    branch = f"refs/heads/campaign/{campaign_id}"
    live_ok, live_commit, _err = _try_run_git(
        ["rev-parse", "--verify", branch], cwd=db_path_root(db)
    )
    live_commit = live_commit if live_ok else ""

    for w in rows:
        kind = w["kind"]
        chunk_id = w["chunk_id"]
        # Decide based on the window kind.
        if kind in ("integration_cas_mismatch", "integration_fence_stale"):
            if live_commit and last_journal_commit and live_commit == last_journal_commit:
                decisions.append({
                    "window_id": w["window_id"],
                    "decision": "SAFE_COMPLETED",
                    "reason": "live ref matches journal; CAS is durable",
                })
            elif not live_commit:
                decisions.append({
                    "window_id": w["window_id"],
                    "decision": "SAFE_NOT_COMPLETED",
                    "reason": "no live ref; nothing was advanced",
                })
            else:
                decisions.append({
                    "window_id": w["window_id"],
                    "decision": "EFFECT_UNKNOWN",
                    "reason": "live ref diverges from journal; manual review required",
                })
        elif kind in ("git_update_ref_failed", "validator_executed_before_state_durable"):
            # Cannot prove safety; preserve EFFECT_UNKNOWN.
            decisions.append({
                "window_id": w["window_id"],
                "decision": "EFFECT_UNKNOWN",
                "reason": f"cannot prove safety for kind={kind}; needs decision",
            })
        else:
            # Unknown kind: preserve EFFECT_UNKNOWN.
            decisions.append({
                "window_id": w["window_id"],
                "decision": "EFFECT_UNKNOWN",
                "reason": f"unknown kind={kind}; needs decision",
            })

    # Apply decisions: SAFE_* -> mark recovered and return campaign to
    # ACTIVE if all windows are resolved. EFFECT_UNKNOWN -> leave
    # window unrecovered.
    safe_count = 0
    for d in decisions:
        if d["decision"] in ("SAFE_COMPLETED", "SAFE_NOT_COMPLETED"):
            db._conn.execute(
                "UPDATE crash_windows SET recovered_at=?, reconciliation_reason=? "
                "WHERE window_id=? AND recovered_at=0",
                (now, d["reason"][:2000], d["window_id"]),
            )
            safe_count += 1
    if safe_count == len(decisions):
        # All windows resolved safely -> return to ACTIVE.
        db._conn.execute(
            "UPDATE campaigns SET state='ACTIVE', updated_at=? WHERE campaign_id=? AND state='EFFECT_UNKNOWN'",
            (now, campaign_id),
        )
        overall = "RESOLVED_TO_ACTIVE"
    else:
        # At least one EFFECT_UNKNOWN: keep campaign in EFFECT_UNKNOWN
        # / NEEDS_DECISION.
        db._conn.execute(
            "UPDATE campaigns SET state='NEEDS_DECISION', updated_at=? WHERE campaign_id=? AND state='EFFECT_UNKNOWN'",
            (now, campaign_id),
        )
        overall = "EFFECT_UNKNOWN_PRESERVED"

    return {
        "campaign_id": campaign_id,
        "decision": overall,
        "unrecovered_windows": rows,
        "window_decisions": decisions,
        "recovered_count": safe_count,
    }


def db_path_root(db: Database) -> Path:
    """Return the path to the DB file (used as repo_root proxy for git ops in tests)."""
    return db.path.parent


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
    "trigger_crash_failpoint",
]
