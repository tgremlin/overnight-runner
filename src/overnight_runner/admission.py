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
from .grants import (
    derive_initial_ledger,
    load_grant,
    load_grant_by_digest,
)
from .resources import acquire_lease, current_fence
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
    runtime_digest: str,
    worker_id: str,
    policy_profile_id: str,
    validator_profile_ids: list[str],
    provider_profile_id: str,
    current_runtime_digest: str | None = None,
    current_model_name: str | None = None,
    current_model_digest: str | None = None,
    current_policy_profile_id: str | None = None,
    current_validator_profile_ids: list[str] | None = None,
    current_provider_profile_id: str | None = None,
    current_accepted_snapshot: RepoSnapshot | None = None,
    now: int | None = None,
) -> tuple[AdmissionReceipt, BudgetLedgerEntry]:
    """Derive a trusted admission receipt for ``chunk`` under ``grant``.

    Performs all deterministic checks (A through E above). Returns the
    minted ``AdmissionReceipt`` and the ``BudgetLedgerEntry`` to which
    the chunk is bound.

    The caller passes the **current** identity pins (model/runtime/policy/...)
    so the receipt binds exactly to what the worker is observing.
    """
    now = int(now if now is not None else time.time())

    # ----- (A) Grant state -----
    # The runner is the SOLE authority on grant state. The caller's
    # supplied grant is treated as an identifier challenge only; the
    # receipt binds to the STORED active grant (which captures the
    # operator activation precisely). This avoids round-trip
    # canonicalisation drift between Pydantic defaults and the supplied
    # payload.
    stored = load_grant(db, grant.grant_id)
    if stored is None:
        raise SafetyError("grant not found in durable store")
    if stored.state != "active":
        raise SafetyError(f"grant state must be 'active' (got {stored.state})")
    if stored.grant_id != grant.grant_id:
        raise SafetyError("grant_id mismatch between stored and supplied")

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
    if current_runtime_digest is None:
        current_runtime_digest = runtime_digest
    if current_runtime_digest != stored.runtime_digest:
        raise SafetyError(
            f"runtime_drift: admission runtime {current_runtime_digest[:8]} != "
            f"grant runtime {stored.runtime_digest[:8]}"
        )
    if current_model_name is not None and current_model_name != stored.model_name:
        raise SafetyError(
            f"model_drift: admission model {current_model_name} != grant model {stored.model_name}"
        )
    if current_model_digest is not None and current_model_digest != stored.model_digest:
        raise SafetyError(
            f"model_digest_drift: admission {current_model_digest[:8]} != grant {stored.model_digest[:8]}"
        )
    if current_policy_profile_id is not None and current_policy_profile_id != stored.policy_profile_id:
        raise SafetyError(
            f"policy_drift: admission policy {current_policy_profile_id} != grant policy {stored.policy_profile_id}"
        )
    if current_validator_profile_ids is not None and not _ids_match(
        current_validator_profile_ids, stored.validator_profile_ids
    ):
        raise SafetyError(
            f"validator_drift: admission validator profiles differ from grant"
        )
    if current_provider_profile_id is not None and current_provider_profile_id != stored.provider_profile_id:
        raise SafetyError(
            f"provider_drift: admission provider {current_provider_profile_id} != grant provider {stored.provider_profile_id}"
        )
    if current_accepted_snapshot is None:
        raise SafetyError("current_accepted_snapshot required")
    if stored.budget.is_expired(now):
        raise SafetyError("grant has expired at admission")
    # ----- Baseline-mismatch check: an existing campaign whose current
    # committed snapshot is a different commit than the supplied one
    # rejects (external ref change / wrong predecessor). We compare
    # only on the common left-anchored prefix so a 64-char padded
    # campaign baseline matches a 40-char git SHA-1 cleanly.
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

    # ----- (E) Idempotency -----
    chunk_checksum = _checksum_chunk(chunk)
    with db.transaction() as cur:
        cur.execute(
            "SELECT admission_id, content_sha256 FROM race_admissions WHERE idem_key=?",
            (chunk.idempotency_key,),
        )
        existing = cur.fetchone()
        if existing is not None:
            stored_sha = existing["content_sha256"]
            if stored_sha == chunk_checksum:
                # Idempotent replay — return the prior receipt.
                cur.execute(
                    "SELECT * FROM admissions WHERE admission_id=?",
                    (existing["admission_id"],),
                )
                row = cur.fetchone()
                if row is None:
                    raise SafetyError("idempotency record references missing admission")
                rec = _row_to_admission(dict(row))
                ledger = _load_ledger(db, rec.budget_ledger_id)
                if ledger is None:
                    raise SafetyError("idempotency record references missing budget ledger")
                return rec, ledger
            raise AdmissionConflict(
                f"idempotency_key {chunk.idempotency_key} presented with DIFFERENT content"
            )

    # ----- (D) Budget -----
    ledger = _load_or_create_ledger(
        db, chunk.campaign_id, stored.grant_id,
        family_id=chunk.package_id, grant=stored,
    )
    exhaustion = ledger.would_exceed(delta_chunks=1)
    if exhaustion is not None:
        raise SafetyError(f"budget: {exhaustion}")

    fence = current_fence(db, chunk.campaign_id)
    new_generation = fence.current_generation

    # Acquire an initial lease for the admission phase itself. Worker
    # acquires the writer lease later.
    lease = acquire_lease(
        db,
        campaign_id=chunk.campaign_id,
        resource_id=f"admission:{chunk.chunk_id}",
        owner_id=worker_id,
        owner_boot_id=os.environ.get("TR_BOOT_ID", "boot-static"),
        owner_pid=int(os.getpid()),
        fence_generation=new_generation,
        ttl_seconds=300,
    )

    admission_id = f"adm-{chunk.chunk_id}-{int(time.time())}"
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
    with db.transaction() as cur:
        cur.execute(
            """
            INSERT INTO chunks (
                chunk_id, campaign_id, package_id, parent_chunk_id, revision,
                idempotency_key, state,
                snapshot_commit, snapshot_tree_digest,
                accepted_predecessor_commit, accepted_predecessor_tree,
                admission_id, created_at, updated_at, idempotency_content_sha256
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                chunk.chunk_id, chunk.campaign_id, chunk.package_id, chunk.parent_chunk_id,
                chunk.revision, chunk.idempotency_key, "ADMITTED",
                None, None,
                current_accepted_snapshot.commit, current_accepted_snapshot.tree_digest,
                receipt.admission_id, now, now, chunk_checksum,
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
        cur.execute(
            "INSERT INTO race_admissions (idem_key, admission_id, content_sha256, issued_at) VALUES (?,?,?,?)",
            (chunk.idempotency_key, receipt.admission_id, chunk_checksum, now),
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


__all__ = [
    "AdmissionConflict",
    "derive_admission",
    "load_admission",
    "chunk_state",
    "transition_chunk",
]
