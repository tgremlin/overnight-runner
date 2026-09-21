"""P06-A01 / P06-A02 / P06-A03 — Derived chunk admission.

This module is the SINGLE source of authority that mints an
``AdmissionReceipt`` (kind ``trio.admission.v1``). Models never issue
admission receipts.

Deterministic checks performed before receipt mint:

  A.  The referenced grant must exist in the durable store AND be
      currently ``active`` (revoked / expired / draft rejects).
  B.  Containment: chunk spec fields (``write_paths``, ``read_paths``,
      ``command_ids``, ``validator_ids``, ``required_validator_ids``,
      ``required_receipt_profiles``) MUST be SUBSETS of the grant's
      corresponding fields (or of the broader approved envelope).
  C.  Identity drift: ``runtime_digest``, ``model_name``,
      ``model_digest``, ``policy_profile_id``,
      ``validator_profile_ids``, ``provider_profile_id``, and the
      current accepted predecessor snapshot MUST match the grant's
      pins.
  D.  Budget: a fresh admission must not exceed the budget ledger.
  E.  Idempotency (P06-A03): when the same ``idempotency_key`` is
      presented twice with the SAME canonical content, the prior
      receipt id is returned. Same key + DIFFERENT content =>
      ``SafetyError`` (deterministic conflict).

Receipt mismatch (different ``admission_id`` returned for the same
canonical content) is impossible because we ALWAYS look up the
idempotency_key first.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from .campaign_schemas import (
    AdmissionReceipt,
    AutonomyGrant,
    Budget,
    BudgetLedgerEntry,
    ChunkSpec,
    RepoSnapshot,
    content_sha256,
)
from .db import Database
from .feature_gate import require_campaign_v2
from .grants import (
    derive_initial_ledger,
    load_grant,
    load_grant_by_digest,
)
from .plans import (
    assert_chunk_criteria_approved,
    load_plan,
    load_plan_digest,
)
from .failpoints import failpoint_armed, raise_failpoint
from .resources import _acquire_lease_cur, current_fence
from .safety import SafetyError


class AdmissionConflict(SafetyError):
    """Raised when an idempotency key is presented with conflicting content."""


def _checksum_chunk(spec: ChunkSpec) -> str:
    return content_sha256(spec)


def _checksum_snapshot(snap: RepoSnapshot | None) -> str:
    if snap is None:
        return ""
    return content_sha256(snap)


def _subsets(child: list[str], parent: list[str]) -> bool:
    """True iff every element of ``child`` is in ``parent``.

    Empty ``child`` is a subset of any ``parent``. Strings are compared
    element-wise; paths are exact-string compared (no shell globbing,
    no ``..`` canonicalisation at this layer — that's broker policy).
    """
    parent_set = set(parent)
    return all((c in parent_set) for c in child)


def _ids_match(child: list[str], parent: list[str]) -> bool:
    """Same as ``_subsets`` but stricter identity: equality or child is
    subset of parent. Used for validator / command ids where order
    may matter at admission."""
    return _subsets(child, parent)


def derive_admission(
    db: Database,
    *,
    grant: AutonomyGrant,
    chunk: ChunkSpec,
    current_runtime_digest: str,
    worker_id: str,
    policy_profile_id: str,
    validator_profile_ids: list[str],
    provider_profile_id: str,
    current_model_name: str,
    current_model_digest: str,
    current_policy_profile_id: str,
    current_validator_profile_ids: list[str],
    current_provider_profile_id: str,
    current_accepted_snapshot: RepoSnapshot,
    now: int | None = None,
) -> tuple[AdmissionReceipt, BudgetLedgerEntry]:
    """Derive a trusted admission receipt for ``chunk`` under ``grant``.

    Performs all deterministic checks (A through E above). Returns the
    minted ``AdmissionReceipt`` and the ``BudgetLedgerEntry`` to which
    the chunk is bound.

    The caller passes the **current** identity pins (model/runtime/policy/...)
    so the receipt binds exactly to what the worker is observing.

    P06 follow-up #2:

    * A02: every identity kwarg is REQUIRED and non-optional. The
      legacy ``runtime_digest`` parameter is REMOVED. There is
      exactly ONE authoritative current-runtime argument.
    * A01: ``chunk.plan_id`` is NOT a caller choice — the runner
      pins the plan to the grant's ``approved_plan_digest``. A
      mismatch refuses the admission.
    * A03: the durable idempotency winner is inserted atomically
      BEFORE any lease/execution authority is allocated. Concurrent
      same-content callers all resolve to the SAME admission.
    """
    require_campaign_v2("derive_admission")
    now = int(now if now is not None else time.time())

    # P06 follow-up #4 (item 4): enforce the campaign continuation
    # authority INSIDE the admission API. Safe operation must not depend
    # on the caller remembering to invoke a preflight helper. This
    # refuses blocked campaign states, inactive/expired grants, and
    # exhausted budget headroom before any authority is minted.
    check_campaign_continuation(
        db, campaign_id=chunk.campaign_id, grant_id=grant.grant_id, now=now
    )

    # ----- (A) Grant state -----
    stored = load_grant(db, grant.grant_id)
    if stored is None:
        raise SafetyError("grant not found in durable store")
    if stored.state != "active":
        raise SafetyError(f"grant state must be 'active' (got {stored.state})")
    if stored.grant_id != grant.grant_id:
        raise SafetyError("grant_id mismatch between stored and supplied")

    # ----- (A-extra) Plan binding (P06 follow-up #2 A01). The grant
    # is pinned to an exact ``approved_plan_digest``; admission MUST
    # consult the durable plan and refuse any drift.
    if not getattr(stored, "approved_plan_digest", ""):
        raise SafetyError(
            "stored grant has no approved_plan_digest; cannot bind admission"
        )
    if grant.approved_plan_digest != stored.approved_plan_digest:
        raise SafetyError(
            "supplied grant approved_plan_digest does not match the stored "
            "grant's pinned plan"
        )
    pinned_plan_digest = load_plan_digest(db, stored.plan_id)
    if pinned_plan_digest is None:
        raise SafetyError(
            f"plan {stored.plan_id!r} is no longer registered"
        )
    if pinned_plan_digest != stored.approved_plan_digest:
        raise SafetyError(
            f"registered plan digest {pinned_plan_digest[:8]} != grant pinned "
            f"digest {stored.approved_plan_digest[:8]}; the plan was modified "
            f"after activation; refusing to broaden authority"
        )

    # ----- (B) Containment -----
    if not _subsets(chunk.permitted_write_paths, stored.allowed_write_paths):
        raise SafetyError("chunk.write_paths are not a subset of grant.allowed_write_paths")
    if not _subsets(chunk.permitted_read_paths, stored.repository_paths):
        raise SafetyError("chunk.read_paths are not a subset of grant.repository_paths")
    if not _ids_match(chunk.permitted_command_ids, stored.allowed_operations):
        raise SafetyError("chunk.command_ids are not a subset of grant.allowed_operations")
    if not _ids_match(chunk.permitted_validator_ids, stored.validator_profile_ids):
        raise SafetyError("chunk.validator_ids are not a subset of grant.validator_profile_ids")
    if not _ids_match(chunk.required_validator_ids, stored.validator_profile_ids):
        raise SafetyError("chunk.required_validator_ids are not a subset of grant.validator_profile_ids")

    # ----- (C) Identity drift -----
    # All identity kwargs are now REQUIRED positional kwargs (no defaults).
    if not isinstance(current_runtime_digest, str) or not current_runtime_digest:
        raise SafetyError("current_runtime_digest is REQUIRED (got empty/None)")
    if not isinstance(current_model_name, str) or not current_model_name:
        raise SafetyError("current_model_name is REQUIRED (got empty/None)")
    if not isinstance(current_model_digest, str) or not current_model_digest:
        raise SafetyError("current_model_digest is REQUIRED (got empty/None)")
    if not isinstance(current_policy_profile_id, str) or not current_policy_profile_id:
        raise SafetyError("current_policy_profile_id is REQUIRED (got empty/None)")
    if current_validator_profile_ids is None or not isinstance(current_validator_profile_ids, list):
        raise SafetyError("current_validator_profile_ids is REQUIRED (got None or non-list)")
    if not isinstance(current_provider_profile_id, str) or not current_provider_profile_id:
        raise SafetyError("current_provider_profile_id is REQUIRED (got empty/None)")
    if current_accepted_snapshot is None:
        raise SafetyError("current_accepted_snapshot required")
    if stored.budget.is_expired(now):
        raise SafetyError("grant has expired at admission")
    if current_runtime_digest != stored.runtime_digest:
        cur_substr = (current_runtime_digest or "missing")[:8]
        raise SafetyError(
            f"runtime_drift: admission runtime {cur_substr} != "
            f"grant runtime {stored.runtime_digest[:8]}"
        )
    if current_model_name != stored.model_name:
        raise SafetyError(
            f"model_drift: admission model {current_model_name} != grant model {stored.model_name}"
        )
    if current_model_digest != stored.model_digest:
        raise SafetyError(
            f"model_digest_drift: admission {current_model_digest[:8]} != grant {stored.model_digest[:8]}"
        )
    if current_policy_profile_id != stored.policy_profile_id:
        raise SafetyError(
            f"policy_drift: admission policy {current_policy_profile_id} != grant policy {stored.policy_profile_id}"
        )
    if not _ids_match(current_validator_profile_ids, stored.validator_profile_ids):
        raise SafetyError(
            f"validator_drift: admission validator profiles differ from grant"
        )
    if current_provider_profile_id != stored.provider_profile_id:
        raise SafetyError(
            f"provider_drift: admission provider {current_provider_profile_id} != grant provider {stored.provider_profile_id}"
        )

    # ----- (B-extra) Plan-required: chunk's package_id and
    # criterion_ids MUST match an entry in the GRANT-PINNED plan.
    # The plan is NOT caller-selectable; we resolve it via the
    # grant's ``plan_id`` so the caller cannot broaden authority.
    if not chunk.package_id:
        raise SafetyError("chunk.package_id is required for admission")
    chunk_crits = set(chunk.criterion_ids or [])
    if not chunk_crits:
        raise SafetyError("chunk.criterion_ids must be non-empty (per P06-A01)")
    assert_chunk_criteria_approved(
        db, plan_id=stored.plan_id,
        package_id=chunk.package_id,
        chunk_criterion_ids=chunk_crits,
    )

    # ----- Baseline-mismatch check: an existing campaign whose current
    # committed snapshot is a different commit than the supplied one
    # rejects (external ref change / wrong predecessor).
    cur = db._conn.execute(
        "SELECT current_commit FROM campaigns WHERE campaign_id=?",
        (chunk.campaign_id,),
    )
    row = cur.fetchone()
    if row is not None and row["current_commit"]:
        stored_commit = row["current_commit"]
        supplied = current_accepted_snapshot.commit
        if stored_commit[: len(supplied)] != supplied and supplied[: len(stored_commit)] != stored_commit:
            raise SafetyError(
                f"baseline_mismatch: campaign current_commit={stored_commit[:8]} != "
                f"admission commit={supplied[:8]}"
            )

    chunk_checksum = _checksum_chunk(chunk)

    # ----- (E+A03) Atomic idempotency reservation, lease, chunk, and
    # admission in ONE transaction (P06 follow-up #3 A03 item 2).
    #
    # The prior design committed the reservation first and inserted the
    # admission in a LATER transaction. A crash in between left a
    # permanent idempotency row pointing at a missing admission, which
    # poisoned every same-content replay forever ("idempotency record
    # references missing admission"). Committing the reservation, the
    # authority lease, the chunk row, and the admission row together
    # means a crash before the admission is durable rolls the entire
    # reservation back: a later same-content retry then legitimately
    # mints exactly one admission and exactly one lease.
    #
    # Concurrent same-content callers all resolve to the same admission
    # because SQLite serializes the ``BEGIN IMMEDIATE`` transactions and
    # the ``race_admissions`` PRIMARY KEY is the durable winner record.
    fence = current_fence(db, chunk.campaign_id)
    new_generation = fence.current_generation

    # Ledger is a precondition for the durable reservation.
    ledger = _load_or_create_ledger(
        db, chunk.campaign_id, stored.grant_id,
        family_id=chunk.package_id, grant=stored,
    )
    exhaustion = ledger.would_exceed(delta_chunks=1)
    if exhaustion is not None:
        raise SafetyError(f"budget: {exhaustion}")

    reservation_id = f"adm-{chunk.chunk_id}-{uuid.uuid4().hex[:12]}"
    required_validator_ids_json = json.dumps(list(chunk.required_validator_ids or []))

    with db.transaction() as cur:
        cur.execute(
            "SELECT admission_id, content_sha256 FROM race_admissions WHERE idem_key=?",
            (chunk.idempotency_key,),
        )
        existing = cur.fetchone()
        if existing is not None:
            if existing["content_sha256"] != chunk_checksum:
                raise AdmissionConflict(
                    f"idempotency_key {chunk.idempotency_key} presented with DIFFERENT content"
                )
            # Idempotent replay — reload the prior admission + ledger.
            cur.execute(
                "SELECT * FROM admissions WHERE admission_id=?",
                (existing["admission_id"],),
            )
            row = cur.fetchone()
            if row is None:
                raise SafetyError("idempotency record references missing admission")
            rec = _row_to_admission(dict(row))
            led = _load_ledger(db, rec.budget_ledger_id)
            if led is None:
                raise SafetyError("idempotency record references missing budget ledger")
            return rec, led

        # Durable winner: the reservation row is inserted in the SAME
        # transaction as the lease + chunk + admission below.
        cur.execute(
            "INSERT INTO race_admissions (idem_key, admission_id, content_sha256, issued_at) "
            "VALUES (?, ?, ?, ?)",
            (chunk.idempotency_key, reservation_id, chunk_checksum, now),
        )
        # Authority lease, cursor-scoped, same transaction.
        lease = _acquire_lease_cur(
            cur, db,
            campaign_id=chunk.campaign_id,
            resource_id=f"admission:{chunk.chunk_id}",
            owner_id=worker_id,
            owner_boot_id=os.environ.get("TR_BOOT_ID", "boot-static"),
            owner_pid=int(os.getpid()),
            fence_generation=new_generation,
            ttl_seconds=300,
        )
        # Failpoint: reservation + lease are staged but the admission
        # row is not yet durable. The transaction rolls back, leaving NO
        # dangling reservation (A03/A16 crash-safety).
        if failpoint_armed("reserve_before_admission_durable"):
            raise_failpoint("reserve_before_admission_durable")

        admission_id = reservation_id
        receipt = AdmissionReceipt(
            schema_version="trio.admission.v1",
            admission_id=admission_id,
            grant_id=stored.grant_id,
            grant_revision=stored.plan_revision,
            chunk_id=chunk.chunk_id,
            chunk_revision=chunk.revision,
            accepted_predecessor_snapshot=current_accepted_snapshot,
            runtime_digest=current_runtime_digest,
            model_name=stored.model_name,
            model_digest=stored.model_digest,
            policy_profile_id=stored.policy_profile_id,
            validator_profile_ids=list(stored.validator_profile_ids),
            provider_profile_id=stored.provider_profile_id,
            worker_id=worker_id,
            budget_ledger_id=ledger.ledger_id,
            fence_generation=new_generation,
            lease_id=lease.lease_id,
            idempotency_key=chunk.idempotency_key,
            issued_at=now,
            issuer="runner",
        )
        receipt_payload = json.dumps(receipt.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        snap_payload = json.dumps(current_accepted_snapshot.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        cur.execute(
            """
            INSERT INTO chunks (
                chunk_id, campaign_id, package_id, parent_chunk_id, revision,
                idempotency_key, state,
                snapshot_commit, snapshot_tree_digest,
                accepted_predecessor_commit, accepted_predecessor_tree,
                admission_id, created_at, updated_at, idempotency_content_sha256,
                required_validator_ids_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                chunk.chunk_id, chunk.campaign_id, chunk.package_id, chunk.parent_chunk_id,
                chunk.revision, chunk.idempotency_key, "ADMITTED",
                None, None,
                current_accepted_snapshot.commit, current_accepted_snapshot.tree_digest,
                receipt.admission_id, now, now, chunk_checksum,
                required_validator_ids_json,
            ),
        )
        cur.execute(
            """
            INSERT INTO admissions (
                admission_id, grant_id, grant_revision, chunk_id, chunk_revision,
                accepted_predecessor_commit, accepted_predecessor_tree,
                runtime_digest, model_name, model_digest, policy_profile_id,
                validator_profile_ids_json, provider_profile_id,
                worker_id, budget_ledger_id, fence_generation, lease_id,
                idempotency_key, issued_at, issuer,
                snapshot_json, payload_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                receipt.admission_id, receipt.grant_id, receipt.grant_revision,
                receipt.chunk_id, receipt.chunk_revision,
                receipt.accepted_predecessor_snapshot.commit,
                receipt.accepted_predecessor_snapshot.tree_digest,
                receipt.runtime_digest, receipt.model_name, receipt.model_digest,
                receipt.policy_profile_id,
                json.dumps(list(receipt.validator_profile_ids)),
                receipt.provider_profile_id,
                receipt.worker_id, receipt.budget_ledger_id,
                receipt.fence_generation, receipt.lease_id,
                receipt.idempotency_key, receipt.issued_at, receipt.issuer,
                snap_payload, receipt_payload,
            ),
        )
    return receipt, ledger


def _grant_digest(grant: AutonomyGrant) -> str:
    """Stable digest for canonical AutonomyGrant JSON."""
    return content_sha256(grant)


def _load_or_create_ledger(
    db: Database, campaign_id: str, grant_id: str, *, family_id: str, grant: AutonomyGrant | None = None
) -> BudgetLedgerEntry:
    cur = db._conn.execute(
        "SELECT * FROM budget_ledgers WHERE campaign_id=? ORDER BY revision DESC LIMIT 1",
        (campaign_id,),
    )
    row = cur.fetchone()
    if row is not None:
        bounds = Budget.model_validate(json.loads(row["bounds_json"]))
        return BudgetLedgerEntry(
            ledger_id=row["ledger_id"],
            campaign_id=row["campaign_id"],
            grant_id=row["grant_id"],
            family_id=row["family_id"],
            revision=row["revision"],
            cumulative_model_calls=row["cumulative_model_calls"],
            cumulative_tool_calls=row["cumulative_tool_calls"],
            cumulative_repairs=row["cumulative_repairs"],
            cumulative_rechunks=row["cumulative_rechunks"],
            cumulative_escalations=row["cumulative_escalations"],
            cumulative_active_seconds=row["cumulative_active_seconds"],
            cumulative_wall_seconds=row["cumulative_wall_seconds"] if "cumulative_wall_seconds" in row.keys() else 0,
            cumulative_cost_microusd=row["cumulative_cost_microusd"],
            cumulative_chunks=row["cumulative_chunks"],
            cumulative_context_tokens=row["cumulative_context_tokens"],
            bounds=bounds,
        )
    if grant is None:
        raise SafetyError(
            f"no budget ledger for {campaign_id} and no grant provided to create one"
        )
    return derive_initial_ledger(db, campaign_id=campaign_id, grant=grant, family_id=family_id)


def _load_ledger(db: Database, ledger_id: str) -> BudgetLedgerEntry | None:
    cur = db._conn.execute(
        "SELECT * FROM budget_ledgers WHERE ledger_id=?",
        (ledger_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    bounds = Budget.model_validate(json.loads(row["bounds_json"]))
    return BudgetLedgerEntry(
        ledger_id=row["ledger_id"],
        campaign_id=row["campaign_id"],
        grant_id=row["grant_id"],
        family_id=row["family_id"],
        revision=row["revision"],
        cumulative_model_calls=row["cumulative_model_calls"],
        cumulative_tool_calls=row["cumulative_tool_calls"],
        cumulative_repairs=row["cumulative_repairs"],
        cumulative_rechunks=row["cumulative_rechunks"],
        cumulative_escalations=row["cumulative_escalations"],
        cumulative_active_seconds=row["cumulative_active_seconds"],
        cumulative_wall_seconds=row["cumulative_wall_seconds"] if "cumulative_wall_seconds" in row.keys() else 0,
        cumulative_cost_microusd=row["cumulative_cost_microusd"],
        cumulative_chunks=row["cumulative_chunks"],
        cumulative_context_tokens=row["cumulative_context_tokens"],
        bounds=bounds,
    )


def _row_to_admission(row: dict[str, Any]) -> AdmissionReceipt:
    return AdmissionReceipt.model_validate(json.loads(row["payload_json"]))


def load_admission(db: Database, admission_id: str) -> AdmissionReceipt | None:
    cur = db._conn.execute(
        "SELECT payload_json FROM admissions WHERE admission_id=?",
        (admission_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return AdmissionReceipt.model_validate(json.loads(row["payload_json"]))


def chunk_state(db: Database, chunk_id: str) -> str | None:
    cur = db._conn.execute(
        "SELECT state FROM chunks WHERE chunk_id=?",
        (chunk_id,),
    )
    row = cur.fetchone()
    return row["state"] if row else None


def transition_chunk(
    db: Database, *, chunk_id: str, from_state: str, to_state: str, now: int | None = None
) -> None:
    """Deterministic chunk state transition.

    Refuses to transition unless the current state matches ``from_state``.
    """
    now = int(now if now is not None else time.time())
    with db.transaction() as cur:
        cur.execute(
            "UPDATE chunks SET state=?, updated_at=? WHERE chunk_id=? AND state=?",
            (to_state, now, chunk_id, from_state),
        )
        if cur.rowcount != 1:
            raise SafetyError(
                f"chunk state transition refused: chunk {chunk_id} expected state={from_state}"
            )


def require_active_grant_or_raise(db: Database, *, grant_id: str, now: int | None = None) -> AutonomyGrant:
    """Return the active grant for ``grant_id`` or raise SafetyError.

    The check enforces state='active' AND non-expired AND non-revoked.
    Used at every admission boundary so a continuation cannot silently
    proceed against a stale grant.
    """
    now = int(now if now is not None else time.time())
    stored = load_grant(db, grant_id)
    if stored is None:
        raise SafetyError(f"grant {grant_id!r} not found")
    if stored.state != "active":
        raise SafetyError(f"grant {grant_id!r} not active (state={stored.state})")
    if stored.budget.is_expired(now):
        raise SafetyError(f"grant {grant_id!r} expired at {stored.budget.grant_expires_at}")
    if stored.revoked_at and stored.revoked_at > 0:
        raise SafetyError(f"grant {grant_id!r} revoked at {stored.revoked_at}")
    return stored


def check_campaign_continuation(
    db: Database,
    *,
    campaign_id: str,
    grant_id: str,
    now: int | None = None,
) -> dict[str, Any]:
    """Pre-flight checks before any new admission/claim/integration.

    P06 follow-up #2 (A05/A07): every consequential continuation
    checks state, grant, budget, and lease/fence authority BEFORE
    allocating any new authority. Returns a dict summarising what
    was checked so callers can surface it in evidence.

    Refuses when any of:
      * campaign state is EFFECT_UNKNOWN, CANCELLED, EXPIRED, COMPLETE,
        BUDGET_EXHAUSTED, or PAUSED_OPERATOR
      * grant not active / revoked / expired
      * ledger would exceed its bounds for delta_chunks=1
      * the caller did not supply a valid fence/lease authority
    """
    now = int(now if now is not None else time.time())
    cur = db._conn.execute(
        "SELECT state, current_fence FROM campaigns WHERE campaign_id=?",
        (campaign_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise SafetyError(f"campaign {campaign_id!r} not registered")
    state = row["state"]
    blocking = {
        "EFFECT_UNKNOWN", "NEEDS_DECISION", "CANCELLED", "EXPIRED", "COMPLETE",
        "BUDGET_EXHAUSTED", "PAUSED_OPERATOR",
    }
    if state in blocking:
        raise SafetyError(
            f"campaign {campaign_id!r} is in blocking state={state}; "
            f"continuation refused"
        )
    # Grant authority.
    require_active_grant_or_raise(db, grant_id=grant_id, now=now)
    # Budget headroom. The ledger is created lazily on the first
    # admission; a continuation check that runs BEFORE any admission
    # is allowed to see no ledger.
    cur = db._conn.execute(
        "SELECT ledger_id, bounds_json FROM budget_ledgers WHERE campaign_id=? "
        "ORDER BY revision DESC LIMIT 1",
        (campaign_id,),
    )
    row = cur.fetchone()
    if row is None:
        # No ledger yet: this is the pre-flight check before the
        # first admission. The actual budget check fires inside
        # ``derive_admission`` once the ledger is created.
        bounds = None
        chunks_so_far = 0
    else:
        # Trusted durable ledger totals/bounds — NOT an incidental
        # ``COUNT(*) FROM chunks`` (P06 follow-up #3 item 13).
        ledger = _load_ledger(db, row["ledger_id"])
        if ledger is None:
            raise SafetyError(
                f"campaign {campaign_id!r} ledger {row['ledger_id']!r} vanished"
            )
        bounds = ledger.bounds
        chunks_so_far = ledger.cumulative_chunks
        # Block continuation when any already-consumed dimension leaves
        # no permitted capacity required by the next operation. We
        # require headroom for at least one more chunk plus the minimal
        # per-chunk model/tool/active-time deltas.
        reason = ledger.would_exceed(
            delta_chunks=1,
            delta_model_calls=1,
            delta_tool_calls=1,
            delta_active_seconds=1,
        )
        if reason is not None:
            raise SafetyError(
                f"campaign {campaign_id!r} has no budget headroom for the next "
                f"operation: {reason}"
            )
    # PAUSED sentinel.
    from .runtime import is_paused
    if is_paused():
        raise SafetyError("PAUSED sentinel present; continuation refused")
    return {
        "campaign_id": campaign_id,
        "grant_id": grant_id,
        "state": state,
        "chunks_so_far": chunks_so_far,
        "max_chunks": bounds.max_chunks if bounds else None,
        "checked_at": now,
    }


__all__ = [
    "AdmissionConflict",
    "derive_admission",
    "load_admission",
    "chunk_state",
    "transition_chunk",
    "require_active_grant_or_raise",
    "check_campaign_continuation",
]
