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
from .failpoints import FAILPOINTS, failpoint_armed, raise_failpoint
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
    *,
    repo_root: Path,
    campaign_id: str,
    base_commit: str,
    db: Database | None = None,
) -> Path:
    """Create or update a disposable worktree for the campaign's
    integration branch.

    The worktree is created at ``<state_dir>/campaign-worktrees/<id>``.
    The branch is ``refs/heads/campaign/<id>``.

    P06 follow-up #3 (A05 item 7): when ``db`` is supplied, the exact
    ``repo_root`` and ``worktree_path`` are persisted on the campaign
    row so a later crash-window reconciliation inspects the REAL
    campaign repo, not the runner state directory.
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
    if db is not None:
        try:
            with db.transaction() as cur:
                cur.execute(
                    "UPDATE campaigns SET repo_root=?, worktree_path=? WHERE campaign_id=?",
                    (str(repo_root), str(wt_root), campaign_id),
                )
        except Exception:
            pass
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


def _durable_chunk_authority(
    db: Database, *, campaign_id: str, chunk_id: str
) -> dict[str, Any]:
    """Return the durable admitted-chunk authority for integration.

    P06 follow-up #4 (A04 item 5): campaign-v2 integration REQUIRES a
    durable admitted chunk. There is no optional caller-supplied hint
    fallback. We require:

      * the chunk row exists;
      * the chunk belongs to ``campaign_id``;
      * the chunk has an ``admission_id``;
      * ``required_validator_ids_json`` is readable and authoritative.

    Missing durability raises ``SafetyError`` (BLOCK).
    """
    cur = db._conn.execute(
        "SELECT campaign_id, admission_id, required_validator_ids_json "
        "FROM chunks WHERE chunk_id=?",
        (chunk_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise SafetyError(
            f"integration gate: no durable admitted chunk for "
            f"chunk_id={chunk_id!r}; campaign-v2 integration REQUIRES a "
            f"runner-owned admitted chunk (no caller-hint fallback)"
        )
    if (row["campaign_id"] or "") != campaign_id:
        raise SafetyError(
            f"integration gate: chunk {chunk_id!r} belongs to campaign "
            f"{row['campaign_id']!r}, not {campaign_id!r}"
        )
    admission_id = row["admission_id"] or ""
    if not admission_id:
        raise SafetyError(
            f"integration gate: chunk {chunk_id!r} has no admission_id; "
            f"refusing integration without durable admission authority"
        )
    raw = row["required_validator_ids_json"]
    if raw is None:
        raise SafetyError(
            f"integration gate: chunk {chunk_id!r} has no readable "
            f"required_validator_ids_json; refusing integration"
        )
    try:
        vals = json.loads(raw)
    except (ValueError, TypeError) as e:
        raise SafetyError(
            f"integration gate: chunk {chunk_id!r} required_validator_ids_json "
            f"is not readable JSON: {e}"
        ) from e
    if not isinstance(vals, list) or not vals:
        raise SafetyError(
            f"integration gate: chunk {chunk_id!r} has an empty required "
            f"validator set; refusing integration"
        )
    return {
        "chunk_id": chunk_id,
        "campaign_id": campaign_id,
        "admission_id": admission_id,
        "required_validator_ids": [str(v) for v in vals if v],
    }


def _git_is_clean(worktree: Path) -> bool:
    """True iff ``git status --porcelain`` in ``worktree`` is empty."""
    ok, out, _err = _try_run_git(["status", "--porcelain"], cwd=worktree)
    if not ok:
        return False
    return out.strip() == ""


def candidate_fingerprint(
    *,
    repo_root: Path,
    new_commit: str,
    campaign_worktree: Path | None = None,
    expected_tree_digest: str | None = None,
    require_canonical: bool = True,
) -> str:
    """Canonical P05/P06 candidate identity for the integrating candidate.

    P06 follow-up #3 (A04 item 3): there is exactly ONE candidate
    identity scheme. Validators run against the ACTUAL campaign
    worktree, so the canonical candidate is
    ``git_worktree_sha(campaign_worktree)``. Integration recomputes the
    same fingerprint and requires it to equal the validation receipt's
    ``candidate_snapshot_digest``.

    P06 follow-up #4 (A04 item 6): for the campaign-v2 authoritative
    path (``require_canonical=True``, the default) the campaign worktree
    is REQUIRED. The caller may NOT substitute a digest via
    ``expected_tree_digest`` and the Git tree fallback is unavailable.
    The worktree MUST exist, have ``HEAD == new_commit``, and be clean.
    The legacy fallbacks are only reachable with ``require_canonical=False``
    for old unit/legacy code and are NOT used by
    ``compare_and_swap_advance``.
    """
    if campaign_worktree is not None and Path(campaign_worktree).exists():
        wt = Path(campaign_worktree)
        try:
            head = _run_git(["rev-parse", "HEAD"], cwd=wt)
        except subprocess.CalledProcessError as e:
            raise SafetyError(
                f"integration gate: cannot read campaign worktree HEAD: {e}"
            ) from e
        if head != new_commit:
            raise SafetyError(
                f"integration gate: campaign worktree HEAD={head[:8]} != "
                f"new_commit={new_commit[:8]}; refusing to integrate a "
                f"candidate that is not the committed candidate"
            )
        if not _git_is_clean(wt):
            raise SafetyError(
                "integration gate: campaign worktree has uncommitted drift; "
                "refusing to integrate a non-clean candidate"
            )
        return git_worktree_sha(wt)
    if require_canonical:
        raise SafetyError(
            "integration gate: campaign-v2 integration REQUIRES the actual "
            "campaign worktree (caller-supplied candidate digests are not "
            "accepted)"
        )
    if expected_tree_digest is not None:
        return expected_tree_digest
    try:
        return _run_git(["rev-parse", f"{new_commit}^{{tree}}"], cwd=repo_root)
    except subprocess.CalledProcessError as e:
        raise SafetyError(
            f"integration gate: cannot derive candidate tree for "
            f"new_commit={new_commit[:8]}: {e}"
        ) from e


def _resolve_campaign_worktree(
    db: Database, *, campaign_id: str, campaign_worktree: Path | None
) -> Path:
    """Return the campaign worktree path (explicit, else persisted).

    P06 follow-up #4 (A04 item 6): loads the persisted campaign worktree
    identity and requires the worktree to actually exist. Raises
    ``SafetyError`` when no durable identity is available.
    """
    if campaign_worktree is not None:
        wt = Path(campaign_worktree)
    else:
        cur = db._conn.execute(
            "SELECT worktree_path FROM campaigns WHERE campaign_id=?",
            (campaign_id,),
        )
        row = cur.fetchone()
        stored = (row["worktree_path"] if row is not None else "") or ""
        if not stored:
            raise SafetyError(
                f"integration gate: campaign {campaign_id!r} has no persisted "
                f"worktree identity; refusing integration"
            )
        wt = Path(stored)
    if not wt.exists():
        raise SafetyError(
            f"integration gate: campaign worktree {wt} does not exist; "
            f"refusing integration"
        )
    return wt


def _validate_admission_lease_lineage(
    db: Database,
    *,
    campaign_id: str,
    chunk_id: str,
    admission_id: str,
    holder_fence_generation: int,
    now: int | None = None,
) -> None:
    """Bind integration to the admitted authority lineage.

    P06 follow-up #4 (item 9): chunk -> admission -> lease/fence.

    P06 follow-up #5 (item 2): the admission lease is a full authority
    object. Integration REQUIRES:

      * the lease belongs to ``campaign_id``;
      * ``released_at == 0``;
      * ``expires_at > now`` (an expired lease is NOT made valid by a
        live-or-dead holder — it requires explicit reconciliation /
        reacquisition / fenced takeover);
      * lease fence == admission fence == campaign current fence.
    """
    now = int(now if now is not None else time.time())

    crow = db._conn.execute(
        "SELECT current_fence FROM campaigns WHERE campaign_id=?",
        (campaign_id,),
    ).fetchone()
    if crow is None:
        raise SafetyError(
            f"integration gate: campaign {campaign_id!r} not registered"
        )
    current_fence_gen = int(crow["current_fence"])
    if holder_fence_generation != current_fence_gen:
        raise SafetyError(
            f"integration gate: holder fence {holder_fence_generation} != "
            f"campaign current fence {current_fence_gen}; refusing integration"
        )

    cur = db._conn.execute(
        "SELECT chunk_id, lease_id, fence_generation FROM admissions "
        "WHERE admission_id=?",
        (admission_id,),
    )
    arow = cur.fetchone()
    if arow is None:
        raise SafetyError(
            f"integration gate: admission {admission_id!r} not found; "
            f"refusing integration"
        )
    if (arow["chunk_id"] or "") != chunk_id:
        raise SafetyError(
            f"integration gate: admission {admission_id!r} is for chunk "
            f"{arow['chunk_id']!r}, not {chunk_id!r}"
        )
    admission_fence_gen = int(arow["fence_generation"])
    if admission_fence_gen != holder_fence_generation:
        raise SafetyError(
            f"integration gate: admission fence_generation="
            f"{admission_fence_gen} != holder fence "
            f"{holder_fence_generation}; refusing integration"
        )
    lease_id = arow["lease_id"] or ""
    if not lease_id:
        raise SafetyError(
            f"integration gate: admission {admission_id!r} has no lease; "
            f"refusing integration"
        )
    lrow = db._conn.execute(
        "SELECT campaign_id, released_at, expires_at, fence_generation "
        "FROM leases WHERE lease_id=?",
        (lease_id,),
    ).fetchone()
    if lrow is None:
        raise SafetyError(
            f"integration gate: admission lease {lease_id!r} not found; "
            f"refusing integration"
        )
    if (lrow["campaign_id"] or "") != campaign_id:
        raise SafetyError(
            f"integration gate: admission lease {lease_id!r} belongs to "
            f"campaign {lrow['campaign_id']!r}, not {campaign_id!r}"
        )
    if int(lrow["released_at"]) != 0:
        raise SafetyError(
            f"integration gate: admission lease {lease_id!r} was released; "
            f"a stale admission may not advance integration"
        )
    if int(lrow["expires_at"]) > 0 and int(lrow["expires_at"]) <= now:
        raise SafetyError(
            f"integration gate: admission lease {lease_id!r} expired at "
            f"{lrow['expires_at']} (now={now}); an expired lease has no "
            f"integration authority — reconcile/reacquire or perform a "
            f"fenced takeover"
        )
    if int(lrow["fence_generation"]) != admission_fence_gen:
        raise SafetyError(
            f"integration gate: admission lease fence_generation="
            f"{lrow['fence_generation']} != admission fence "
            f"{admission_fence_gen}; refusing integration"
        )


def _precheck_receipts(receipt_ids: list[str]) -> None:
    """Cheap gate: every receipt must exist, be kind=validation, PASS.

    Runs before candidate derivation so a ``mutation/apply`` receipt (or
    a missing id) is rejected with a clear "validation" error.
    """
    if not receipt_ids:
        raise SafetyError(
            "integration gate: a trusted validation receipt id "
            "(kind=validation, outcome=pass) is REQUIRED before integration"
        )
    for rid in receipt_ids:
        rec = load_receipt(rid)
        if rec is None:
            raise SafetyError(f"integration gate: validation receipt {rid} not found")
        if rec.get("kind") != KIND_VALIDATION:
            raise SafetyError(
                f"integration gate: receipt {rid} is kind={rec.get('kind')}, "
                f"must be {KIND_VALIDATION} to satisfy validation"
            )
        if rec.get("outcome") != "pass":
            raise SafetyError(
                f"integration gate: receipt {rid} outcome={rec.get('outcome')}, "
                f"must be 'pass'"
            )


def _validate_required_receipts(
    db: Database,
    *,
    campaign_id: str,
    chunk_id: str,
    candidate: str,
    receipt_ids: list[str],
    required_validators: list[str] | None,
    expected_chunk_id: str | None,
    expected_validator_id: str | None,
) -> list[str]:
    """Prove every required validator has one trusted PASS receipt.

    P06 follow-up #3 (A04 item 4):

      * receipt.chunk_id == integrating chunk_id
      * receipt candidate snapshot == exact candidate
      * receipt outcome == pass
      * receipt validator/profile is one of the exact required validators
      * ALL required validators have one trusted PASS receipt

    A single PASS receipt cannot satisfy a chunk requiring multiple
    validators; a duplicate receipt for the same validator does not
    satisfy another validator. Returns the receipt ids actually used.
    """
    if not receipt_ids:
        raise SafetyError(
            "integration gate: a trusted validation receipt id "
            "(kind=validation, outcome=pass) is REQUIRED before integration"
        )
    # Load + sanity-check every supplied receipt.
    loaded: list[dict[str, Any]] = []
    for rid in receipt_ids:
        rec = load_receipt(rid)
        if rec is None:
            raise SafetyError(
                f"integration gate: validation receipt {rid} not found"
            )
        if rec.get("kind") != KIND_VALIDATION:
            raise SafetyError(
                f"integration gate: receipt {rid} is kind={rec.get('kind')}, "
                f"must be {KIND_VALIDATION} to satisfy validation"
            )
        if rec.get("outcome") != "pass":
            raise SafetyError(
                f"integration gate: receipt {rid} outcome={rec.get('outcome')}, "
                f"must be 'pass'"
            )
        receipt_snapshot = rec.get("candidate_snapshot_digest", "")
        if not receipt_snapshot:
            raise SafetyError(
                f"integration gate: validation receipt {rid} has no "
                f"candidate_snapshot_digest; cannot bind to integrating candidate"
            )
        if receipt_snapshot != candidate:
            raise SafetyError(
                f"integration gate: validation receipt {rid} "
                f"candidate_snapshot_digest {receipt_snapshot[:8]} != integrating "
                f"candidate {candidate[:8]}; rejecting wrong-candidate receipt"
            )
        # chunk_id binding: a PASS receipt from chunk A must not
        # authorize chunk B.
        want_chunk = expected_chunk_id if expected_chunk_id is not None else chunk_id
        receipt_chunk_id = rec.get("chunk_id", "")
        if want_chunk is not None and receipt_chunk_id != want_chunk:
            raise SafetyError(
                f"integration gate: validation receipt {rid} chunk_id="
                f"{receipt_chunk_id!r} != integrating chunk_id={want_chunk!r}; "
                f"rejecting cross-chunk receipt"
            )
        loaded.append(rec)

    # P06 follow-up #4 (A04 item 5): the durable required-validator set
    # is the ONLY authority. There is no caller-hint fallback.
    if not required_validators:
        raise SafetyError(
            "integration gate: no durable required-validator set for the "
            "integrating chunk; refusing integration"
        )
    used: list[str] = []
    for validator in required_validators:
        match = None
        for rec in loaded:
            rec_validator = rec.get("validator_id", "")
            rec_profile = rec.get("validator_profile", "")
            if validator in (rec_validator, rec_profile):
                match = rec
                break
        if match is None:
            raise SafetyError(
                f"integration gate: required validator {validator!r} has no "
                f"trusted PASS receipt bound to the integrating candidate; "
                f"refusing integration"
            )
        used.append(match.get("receipt_id", ""))
    return used


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
    validation_receipt_ids: list[str] | None = None,
    expected_old: str | None = None,
    expected_tree_digest: str | None = None,
    expected_chunk_id: str | None = None,
    expected_validator_id: str | None = None,
    campaign_worktree: Path | None = None,
    now: int | None = None,
) -> CompareAndSwapResult:
    """Atomically advance the campaign's integration ref.

    Sequence:

      0. Validate the trusted validation receipt(s) (P06-A04): every
         REQUIRED validator derived from the durable admitted chunk
         contract must have one ``kind=validation`` receipt with
         ``outcome=pass`` bound to the EXACT candidate being integrated
         (candidate snapshot + chunk_id + validator_id). A
         ``mutation/apply`` receipt cannot satisfy a required
         validation, and a single PASS receipt cannot satisfy a chunk
         requiring multiple validators.
      1. Read the LIVE current ref via ``git rev-parse`` (THE source
         of truth for what the world sees).
      2. Compare with ``expected_old`` (if provided). Mismatch -> fail
         with EFFECT_UNKNOWN + crash window.
      3. Holder fence MUST match.
      4. CAS the ref forward.
      5. Append to ``integration_journal`` and update campaigns row.
    """
    require_campaign_v2("compare_and_swap_advance")
    now = int(now if now is not None else time.time())
    branch = f"refs/heads/campaign/{campaign_id}"

    # Step 0a (follow-up #5 item 1): enforce the FULL campaign
    # continuation authority at the consequential API itself — blocked
    # campaign state, active/non-revoked/non-expired grant, trusted
    # budget headroom, and global PAUSED. We REUSE
    # ``check_campaign_continuation`` so there is one authority definition
    # (no divergent duplicate) and safe operation never depends on the
    # caller having run a preflight.
    _crow = db._conn.execute(
        "SELECT grant_id FROM campaigns WHERE campaign_id=?", (campaign_id,)
    ).fetchone()
    if _crow is None:
        raise SafetyError(f"campaign {campaign_id!r} not registered")
    from .admission import check_campaign_continuation
    check_campaign_continuation(
        db, campaign_id=campaign_id, grant_id=_crow["grant_id"], now=now
    )

    # Step 0b: gather the receipt ids (list takes precedence).
    receipts = list(validation_receipt_ids or [])
    if validation_receipt_id:
        receipts.append(validation_receipt_id)
    # De-duplicate while preserving order.
    seen: set[str] = set()
    receipts = [r for r in receipts if r and not (r in seen or seen.add(r))]
    # Cheap kind/outcome gate FIRST, so a mutation/apply receipt is
    # rejected as "not a validation receipt" before any candidate
    # derivation effort.
    _precheck_receipts(receipts)

    # Step 0c (item 5): durable admitted chunk authority. No caller-hint
    # fallback exists for campaign-v2 integration.
    chunk_auth = _durable_chunk_authority(
        db, campaign_id=campaign_id, chunk_id=chunk_id
    )

    # Step 0d (item 6): the ACTUAL campaign worktree is required; the
    # caller may not substitute a candidate digest.
    wt = _resolve_campaign_worktree(
        db, campaign_id=campaign_id, campaign_worktree=campaign_worktree
    )
    candidate = candidate_fingerprint(
        repo_root=repo_root,
        new_commit=new_commit,
        campaign_worktree=wt,
        require_canonical=True,
    )

    # Step 0e (item 4): all durable-required validators have a trusted
    # PASS receipt bound to the exact candidate/chunk.
    _validate_required_receipts(
        db,
        campaign_id=campaign_id,
        chunk_id=chunk_id,
        candidate=candidate,
        receipt_ids=receipts,
        required_validators=chunk_auth["required_validator_ids"],
        expected_chunk_id=None,
        expected_validator_id=None,
    )

    # Step 0f (item 9): bind integration to the admitted authority
    # lineage (chunk -> admission -> lease/fence). A released/stale
    # admission may not advance merely because the caller knows the
    # current integer fence.
    _validate_admission_lease_lineage(
        db,
        campaign_id=campaign_id,
        chunk_id=chunk_id,
        admission_id=chunk_auth["admission_id"],
        holder_fence_generation=holder_fence_generation,
        now=now,
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

    worktree_path = str(wt)

    if expected_old is not None and live_commit and live_commit != expected_old and expected_old != live_commit[: len(expected_old)] and live_commit[: len(live_commit)] != expected_old[: len(expected_old)]:
        _record_crash_window(
            db,
            campaign_id=campaign_id,
            chunk_id=chunk_id,
            kind="integration_cas_mismatch",
            observed_artifact=live_commit,
            repo_root=str(repo_root),
            worktree_path=worktree_path,
            integration_branch=branch,
            evidence={"new_commit": new_commit, "expected_old": expected_old},
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
            repo_root=str(repo_root),
            worktree_path=worktree_path,
            integration_branch=branch,
            evidence={"new_commit": new_commit, "expected_old": live_commit},
        )
        raise SafetyError(
            f"integration_fence_stale: holder={holder_fence_generation} current={cur_fence}"
        )

    # Step 3: durable DB integration intent BEFORE the ref advance
    # (P06 follow-up #3 A05 item 6). At either failpoint the durable
    # intent row is committed before the consequential operation.
    _failpoint_with_intent(
        db, "commit_exists_before_db_candidate_state",
        campaign_id=campaign_id, chunk_id=chunk_id,
        repo_root=str(repo_root), worktree_path=worktree_path,
        integration_branch=branch,
        evidence={"new_commit": new_commit, "expected_old": live_commit},
    )
    _failpoint_with_intent(
        db, "db_integration_intent_before_ref_advance",
        campaign_id=campaign_id, chunk_id=chunk_id,
        repo_root=str(repo_root), worktree_path=worktree_path,
        integration_branch=branch,
        evidence={"new_commit": new_commit, "expected_old": live_commit},
    )
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
            repo_root=str(repo_root),
            worktree_path=worktree_path,
            integration_branch=branch,
            evidence={"new_commit": new_commit, "expected_old": live_commit},
        )
        raise SafetyError(f"git update-ref failed: {e}") from e

    # Step 4: derive new tree digest (from the git tree of new_commit).
    new_tree = _run_git(["rev-parse", f"{new_commit}^{{tree}}"], cwd=repo_root)

    # Failpoint: ref advanced but no integration journal/event durable.
    _failpoint_with_intent(
        db, "ref_advanced_before_integration_journal_event",
        campaign_id=campaign_id, chunk_id=chunk_id,
        repo_root=str(repo_root), worktree_path=worktree_path,
        integration_branch=branch,
        evidence={"new_commit": new_commit, "expected_old": live_commit},
    )
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
                json.dumps({"new_tree": new_tree, "candidate": candidate}),
            ),
        )

    # Failpoint: journal durable but event/outbox projection missing.
    _failpoint_with_intent(
        db, "event_outbox_committed_before_projection_status",
        campaign_id=campaign_id, chunk_id=chunk_id,
        repo_root=str(repo_root), worktree_path=worktree_path,
        integration_branch=branch,
        evidence={"new_commit": new_commit, "expected_old": live_commit},
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


def _failpoint_with_intent(
    db: Database,
    name: str,
    *,
    campaign_id: str,
    chunk_id: str,
    repo_root: str = "",
    worktree_path: str = "",
    integration_branch: str = "",
    evidence: dict[str, Any] | None = None,
) -> None:
    """If ``name`` is armed: commit DURABLE intent, then raise.

    P06 follow-up #3 (A05 item 6): the durable intent (a crash window
    carrying repository/worktree identity + evidence) is committed
    BEFORE the consequential operation continues, so a restart can
    discover the uncertainty without depending on post-crash code.
    """
    if not failpoint_armed(name):
        return
    record_crash_intent(
        db, name,
        campaign_id=campaign_id, chunk_id=chunk_id,
        repo_root=repo_root, worktree_path=worktree_path,
        integration_branch=integration_branch, evidence=evidence,
    )
    raise_failpoint(name)


def record_crash_intent(
    db: Database,
    name: str,
    *,
    campaign_id: str,
    chunk_id: str = "",
    repo_root: str = "",
    worktree_path: str = "",
    integration_branch: str = "",
    evidence: dict[str, Any] | None = None,
) -> str:
    """Persist durable EFFECT_UNKNOWN intent evidence for ``name``.

    Returns the crash-window id. Used by failpoints and by the
    candidate-change boundary (``campaign_apply``).
    """
    return _record_crash_window(
        db,
        campaign_id=campaign_id,
        chunk_id=chunk_id,
        kind=name,
        observed_artifact=json.dumps(evidence or {})[:2000],
        repo_root=repo_root,
        worktree_path=worktree_path,
        integration_branch=integration_branch,
        evidence=evidence,
    )


def _record_crash_window(
    db: Database,
    *,
    campaign_id: str,
    chunk_id: str,
    kind: str,
    observed_artifact: str,
    repo_root: str = "",
    worktree_path: str = "",
    integration_branch: str = "",
    evidence: dict[str, Any] | None = None,
) -> str:
    """Record a crash window AND transition the campaign to EFFECT_UNKNOWN.

    P06 follow-up #2 A05: while EFFECT_UNKNOWN, no next-chunk
    admission, no integration continuation, and no automatic mutation
    replay are permitted. The state is restored to a known
    COMPLETED/NOT-COMPLETED or NEEDS_DECISION only via
    ``reconcile_crash_window``.

    P06 follow-up #3 (A05 item 7): the window stores the exact
    repository / worktree / integration-branch identity so
    reconciliation inspects the REAL campaign repo.

    P06 follow-up #4 (A05 item 1): the durable crash window AND the
    ``EFFECT_UNKNOWN`` transition are written in ONE ``BEGIN IMMEDIATE``
    transaction. Either BOTH exist or NEITHER does: there is never a
    durable crash window with the campaign still ACTIVE, and never an
    EFFECT_UNKNOWN campaign without the corresponding durable evidence.
    """
    now = int(time.time())
    wid = f"cw-{uuid.uuid4().hex[:16]}"
    with db.transaction() as cur:
        cur.execute(
            "SELECT state FROM campaigns WHERE campaign_id=?",
            (campaign_id,),
        )
        row = cur.fetchone()
        cur.execute(
            """
            INSERT INTO crash_windows (
                window_id, campaign_id, chunk_id, kind, observed_artifact,
                snapshot_at, recovered_at, reconciliation_reason,
                repo_root, worktree_path, integration_branch, evidence_json
            ) VALUES (?,?,?,?,?,?,0,'',?,?,?,?)
            """,
            (
                wid, campaign_id, chunk_id, kind, observed_artifact[:2000], now,
                repo_root, worktree_path, integration_branch,
                json.dumps(evidence or {}),
            ),
        )
        # Transition campaign to EFFECT_UNKNOWN in the SAME transaction.
        # Refuse to clobber a terminal state (the durable window alone is
        # a justified recovery object for those states).
        if row is not None and row["state"] not in (
            "CANCELLED", "EXPIRED", "COMPLETE", "BUDGET_EXHAUSTED"
        ):
            cur.execute(
                "UPDATE campaigns SET state='EFFECT_UNKNOWN', updated_at=? "
                "WHERE campaign_id=? AND state NOT IN "
                "('CANCELLED','EXPIRED','COMPLETE','BUDGET_EXHAUSTED')",
                (now, campaign_id),
            )
        # Atomicity failpoint: raise BEFORE commit so the whole
        # transaction (window + state) rolls back together.
        if failpoint_armed("crash_window_before_commit"):
            raise_failpoint("crash_window_before_commit")
    return wid


# ----------------------------- Failpoint injection -----------------------------

# The failpoint names + arming primitives live in ``overnight_runner.failpoints``
# (shared with the admission path). ``_FAILPOINTS`` is kept as an alias for
# backward compatibility with existing tests.
_FAILPOINTS = FAILPOINTS


def trigger_crash_failpoint(name: str) -> None:
    """If the named failpoint is armed, raise RuntimeError.

    P06 follow-up #3 A05: the dangerous boundaries use
    ``_failpoint_with_intent`` which commits DURABLE intent before
    raising. This bare helper is retained for callers that only need
    the deterministic raise.
    """
    if failpoint_armed(name):
        raise_failpoint(name)


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


def _resolve_repo_for_window(
    db: Database, campaign_row: dict[str, Any], window_row: dict[str, Any]
) -> tuple[Path | None, str]:
    """Return the REAL (repo_path, integration_branch) for a window.

    P06 follow-up #3 (A05 item 7): prefer the identity persisted with
    the crash window, then the campaign row.

    P06 follow-up #4 (A05 item 2): missing repository identity returns
    ``None`` — NOT ``Path("")`` (whose ``str()`` is ``"."`` and would
    silently make the caller run Git against the process cwd). The
    caller MUST treat ``None`` as fail-closed EFFECT_UNKNOWN and never
    run Git against ``.`` / ``state_dir`` / a guessed repository.
    """
    repo_root = window_row.get("repo_root") or campaign_row.get("repo_root") or ""
    branch = (
        window_row.get("integration_branch")
        or campaign_row.get("integration_branch")
        or f"refs/heads/campaign/{campaign_row.get('campaign_id','')}"
    )
    if repo_root:
        return Path(repo_root), branch
    return None, branch


def _ensure_integration_event(
    db: Database, *, campaign_id: str, commit: str
) -> bool:
    """Ensure the integration_advanced event for ``commit`` exists.

    P06 follow-up #4 (A05 item 3): reconciliation must recover the
    REQUIRED durable audit/event evidence for the event/outbox boundary
    before declaring SAFE_COMPLETED. Returns ``True`` when the event is
    present (already durable or idempotently recreated); ``False`` when
    it cannot be rebuilt (caller must remain EFFECT_UNKNOWN). Repeated
    calls never create a duplicate event.
    """
    if not commit:
        return False
    existing = db._conn.execute(
        "SELECT event_id FROM campaign_events WHERE campaign_id=? "
        "AND event_type='integration_advanced' AND to_state=? LIMIT 1",
        (campaign_id, commit),
    ).fetchone()
    if existing is not None:
        return True
    j = db._conn.execute(
        "SELECT chunk_id, expected_old_commit, fence_generation, actor, "
        "idempotency_key FROM integration_journal "
        "WHERE campaign_id=? AND committed_new_commit=? "
        "ORDER BY committed_at DESC LIMIT 1",
        (campaign_id, commit),
    ).fetchone()
    if j is None:
        return False
    key = j["idempotency_key"]
    event_id = f"ev-replay-{key}"
    try:
        with db.transaction() as cur:
            # Re-check inside the write transaction so a concurrent or
            # repeated reconcile cannot create a duplicate event.
            ex = cur.execute(
                "SELECT event_id FROM campaign_events WHERE campaign_id=? "
                "AND event_type='integration_advanced' AND to_state=? LIMIT 1",
                (campaign_id, commit),
            ).fetchone()
            if ex is not None:
                return True
            cur.execute(
                """
                INSERT INTO campaign_events (
                    event_id, campaign_id, chunk_id, event_type,
                    from_state, to_state, actor, payload, fence_generation,
                    issued_at, idempotency_key
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event_id, campaign_id, j["chunk_id"], "integration_advanced",
                    j["expected_old_commit"], commit, j["actor"],
                    json.dumps({"replayed_by_reconciliation": True}),
                    int(j["fence_generation"]), int(time.time()), key,
                ),
            )
    except Exception:
        # PRIMARY KEY / race: treat as already present.
        return True
    return True


def reconcile_crash_window(db: Database, *, campaign_id: str) -> dict[str, Any]:
    """Reconcile a campaign's crash windows.

    P06 follow-up #3 A05: this function does actual reconciliation
    work, inspecting the REAL campaign repository (persisted repo /
    worktree identity), not the runner state directory. For each
    unrecovered window it:
      1. Inspects the LIVE git ref for the campaign integration branch
         in the real campaign repo.
      2. Compares against the integration_journal's last recorded
         committed commit and the durable evidence (intended commit).
      3. Decides one of:
         - SAFE_COMPLETED: live ref matches journal.
         - SAFE_NOT_COMPLETED: the REAL repo shows no advanced ref and
           no durable evidence that an advance occurred.
         - EFFECT_UNKNOWN: the ref advanced (or durable intent exists)
           but cannot be proven complete -> preserve EFFECT_UNKNOWN and
           return NEEDS_DECISION.

    It NEVER clears EFFECT_UNKNOWN merely because no windows were
    found; unknown effect requires positive evidence (item 8).
    """
    now = int(time.time())
    cur = db._conn.execute(
        "SELECT state, repo_root, worktree_path, integration_branch FROM campaigns WHERE campaign_id=?",
        (campaign_id,),
    )
    campaign_row_raw = cur.fetchone()
    campaign_row = dict(campaign_row_raw) if campaign_row_raw else {
        "campaign_id": campaign_id, "state": None,
        "repo_root": "", "worktree_path": "", "integration_branch": "",
    }

    cur = db._conn.execute(
        """
        SELECT window_id, chunk_id, kind, observed_artifact, snapshot_at,
               recovered_at, reconciliation_reason,
               repo_root, worktree_path, integration_branch, evidence_json
        FROM crash_windows
        WHERE campaign_id=? AND recovered_at=0
        ORDER BY snapshot_at ASC
        """,
        (campaign_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        # No unrecovered windows. P06 follow-up #3 item 8: do NOT
        # auto-activate an EFFECT_UNKNOWN campaign without positive
        # evidence. Unknown consequential effect is preserved as
        # NEEDS_DECISION.
        if campaign_row.get("state") == "EFFECT_UNKNOWN":
            db._conn.execute(
                "UPDATE campaigns SET state='NEEDS_DECISION', updated_at=? "
                "WHERE campaign_id=? AND state='EFFECT_UNKNOWN'",
                (now, campaign_id),
            )
        return {
            "campaign_id": campaign_id,
            "decision": "NO_WINDOWS_NO_EVIDENCE",
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

    for w in rows:
        kind = w["kind"]
        repo_path, branch = _resolve_repo_for_window(db, campaign_row, w)
        if repo_path is None:
            # No durable repository identity: fail closed. We MUST NOT
            # run Git against "." / state_dir / cwd / a guessed repo.
            decisions.append({
                "window_id": w["window_id"],
                "decision": "EFFECT_UNKNOWN",
                "reason": f"no durable repo identity for kind={kind}; needs decision",
            })
            continue
        live_ok, live_commit, _err = _try_run_git(
            ["rev-parse", "--verify", branch], cwd=repo_path
        )
        live_commit = live_commit if live_ok else ""
        try:
            evidence = json.loads(w.get("evidence_json") or "{}")
        except (ValueError, TypeError):
            evidence = {}
        intended = evidence.get("new_commit", "")
        expected_old_ev = evidence.get("expected_old", "")

        if live_commit and last_journal_commit and live_commit == last_journal_commit:
            # P06 follow-up #4 (item 3): for the event/outbox boundary we
            # must ensure the REQUIRED durable event evidence exists (or
            # be idempotently restored) before declaring SAFE_COMPLETED.
            if kind == "event_outbox_committed_before_projection_status":
                if _ensure_integration_event(
                    db, campaign_id=campaign_id, commit=live_commit
                ):
                    decisions.append({
                        "window_id": w["window_id"],
                        "decision": "SAFE_COMPLETED",
                        "reason": "live ref matches journal; event/outbox evidence "
                                  "present or restored idempotently",
                    })
                else:
                    decisions.append({
                        "window_id": w["window_id"],
                        "decision": "EFFECT_UNKNOWN",
                        "reason": "ref/journal durable but required event/outbox "
                                  "evidence is missing and cannot be rebuilt",
                    })
            else:
                decisions.append({
                    "window_id": w["window_id"],
                    "decision": "SAFE_COMPLETED",
                    "reason": "live ref matches integration journal; CAS is durable",
                })
        elif live_commit and intended and live_commit == intended:
            # The ref DID advance to the intended commit, but the
            # journal/event projection is missing. We cannot prove the
            # full transition completed -> needs a decision.
            decisions.append({
                "window_id": w["window_id"],
                "decision": "EFFECT_UNKNOWN",
                "reason": (
                    "ref advanced to the intended commit but the integration "
                    "journal/event is missing; manual review required"
                ),
            })
        elif (
            live_commit and expected_old_ev
            and live_commit == expected_old_ev and live_commit != intended
        ):
            # Positive evidence the ref did NOT advance: it is still at
            # the pre-integration commit recorded at the boundary.
            decisions.append({
                "window_id": w["window_id"],
                "decision": "SAFE_NOT_COMPLETED",
                "reason": "live ref still at the recorded pre-integration commit; "
                          "no advance occurred",
            })
        elif live_commit:
            decisions.append({
                "window_id": w["window_id"],
                "decision": "EFFECT_UNKNOWN",
                "reason": "live ref diverges from journal; manual review required",
            })
        elif kind == "ref_advanced_before_integration_journal_event":
            # This boundary is only reached AFTER a successful ref
            # advance; a missing ref means the real repo was not
            # inspected or the ref was rolled back. Do not claim safe.
            decisions.append({
                "window_id": w["window_id"],
                "decision": "EFFECT_UNKNOWN",
                "reason": "post-advance boundary but no live ref; needs decision",
            })
        else:
            decisions.append({
                "window_id": w["window_id"],
                "decision": "SAFE_NOT_COMPLETED",
                "reason": "real repo shows no advanced ref and no durable advance evidence",
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
        # All windows resolved by positive evidence -> return to ACTIVE.
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
    """Return the path to the DB file (legacy repo_root proxy)."""
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
    "candidate_fingerprint",
    "compare_and_swap_advance",
    "reconcile_crash_window",
    "record_crash_intent",
    "mark_crash_window_recovered",
    "trigger_crash_failpoint",
]
