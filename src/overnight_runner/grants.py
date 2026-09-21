"""P06-A01 — AutonomyGrant engine.

Grant authority originates ONLY from the protected runner/operator
approval channel in this module. A model-written grant, draft grant,
hash, or JSON document alone confers NO authority.

Identity of this engine:
  - Persists grants in the runner's durable store (DB schema v2).
  - ``activate_grant`` is the ONLY mint surface for a non-draft grant.
  - Stored grants are content-addressed via ``content_sha256``; the
    ``grants.grant_digest`` column is the durable reference.

Draft / forged / expired / revoked grants all fail closed at the
admission boundary. See ``admission.derive_admission``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .campaign_schemas import (
    AutonomyGrant,
    BudgetLedgerEntry,
    content_sha256,
)
from .db import Database
from .plans import load_plan_digest
from .protected_approvals import consume_protected_approval
from .safety import SafetyError


@dataclass(frozen=True)
class GrantActivationResult:
    grant_id: str
    activated_at: int
    operator_id: str
    operator_receipt_digest: str


def grant_payload(grant: AutonomyGrant) -> dict[str, Any]:
    """Canonical payload for grant activation/revocation records."""
    return grant.model_dump(mode="json", exclude_none=True)


def _grant_digest(grant: AutonomyGrant) -> str:
    return content_sha256(grant)


def activate_grant(
    db: Database,
    *,
    grant: AutonomyGrant,
    operator_id: str,
    approval_id: str,
    activated_at: int | None = None,
) -> GrantActivationResult:
    """Promote a DRAFT grant to ACTIVE.

    Authority comes ONLY from a pre-registered row in
    ``protected_approvals`` (recorded earlier via the trusted
    out-of-band channel by ``register_protected_approval``). The
    ``operator_id`` and ``approval_id`` supplied here are identifier
    challenges only — the runner does NOT accept a freshly-supplied
    dict as the receipt itself.

    P06 follow-up #2 (A01): the grant is pinned to the exact approved
    plan digest. ``grant.approved_plan_digest`` MUST equal the digest
    stored for ``grant.plan_id`` in ``approved_plans.plan_digest``.
    Mismatch -> SafetyError. The protected approval is also consulted
    AFTER this plan check so the trust order is: durable plan digest
    -> protected approval -> grant activation.
    """
    if grant.state != "draft":
        raise SafetyError(f"grant must be draft (state={grant.state})")
    if grant.plan_revision < 1:
        raise SafetyError("plan_revision must be >= 1")
    if not operator_id or not approval_id:
        raise SafetyError("operator_id and approval_id required")
    if not approval_id:
        raise SafetyError("approval_id is required")
    # (1) Plan digest pin: the grant MUST bind to an exact plan content.
    stored_plan_digest = load_plan_digest(db, grant.plan_id)
    if stored_plan_digest is None:
        raise SafetyError(
            f"plan {grant.plan_id!r} is not registered; refusing to activate a "
            f"grant against an unknown plan"
        )
    if stored_plan_digest != grant.approved_plan_digest:
        raise SafetyError(
            f"approved_plan_digest mismatch: grant pinned "
            f"{grant.approved_plan_digest[:8]} != registered plan "
            f"{stored_plan_digest[:8]} for plan_id={grant.plan_id!r}"
        )
    # (2) Resolve the pre-registered protected approval.
    from .protected_approvals import (
        load_protected_approval,
        consume_protected_approval,
    )
    ap = load_protected_approval(db, approval_id)
    if ap is None:
        raise SafetyError(
            f"approval {approval_id} not found in protected approvals table"
        )
    if ap.operator_id != operator_id:
        raise SafetyError(
            f"approval operator mismatch: supplied={operator_id} stored={ap.operator_id}"
        )
    if ap.operation != "activate_grant":
        raise SafetyError(
            f"approval operation mismatch: supplied={ap.operation} expected=activate_grant"
        )
    target_digest = _grant_digest(grant)
    if ap.grant_digest_target != target_digest:
        raise SafetyError(
            "approval grant_digest_target does not match the supplied grant digest"
        )
    # (3) All identity checks passed. Consume the approval AND insert
    # the active grant in ONE transaction (P06 follow-up #3 item 15):
    # a crash between consumption and grant insert must NOT burn the
    # one-shot approval without producing a grant.
    ts = int(activated_at if activated_at is not None else time.time())
    active = grant.model_copy(
        update={
            "state": "active",
            "activated_at": ts,
            "operator_id": operator_id,
            "operator_receipt_digest": ap.operator_receipt_digest,
        }
    )
    digest = _grant_digest(active)
    payload = json.dumps(grant_payload(active), sort_keys=True, separators=(",", ":"))
    from .failpoints import failpoint_armed, raise_failpoint
    with db.transaction() as cur:
        # Atomic one-shot consumption: refuse if already consumed.
        cur.execute(
            "UPDATE protected_approvals SET consumed_at=? "
            "WHERE approval_id=? AND consumed_at=0",
            (ts, approval_id),
        )
        if cur.rowcount == 0:
            raise SafetyError(
                f"approval {approval_id} already consumed; one-shot consumption enforced"
            )
        cur.execute(
            """
            INSERT INTO grants (
                grant_id, grant_digest, state, plan_id, plan_revision,
                operator_id, operator_receipt_digest,
                activated_at, revoked_at, revoked_reason,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                active.grant_id, digest, active.state,
                active.plan_id, active.plan_revision,
                active.operator_id, active.operator_receipt_digest,
                active.activated_at, active.revoked_at, active.revoked_reason,
                payload,
            ),
        )
        # Failpoint: crash after consume but before the transaction
        # commits. The whole transaction rolls back, so the approval is
        # NOT burned (transactional activation).
        if failpoint_armed("activate_grant_before_commit"):
            raise_failpoint("activate_grant_before_commit")
    return GrantActivationResult(
        grant_id=active.grant_id,
        activated_at=active.activated_at,
        operator_id=active.operator_id,
        operator_receipt_digest=active.operator_receipt_digest,
    )


def revoke_grant(db: Database, *, grant_id: str, reason: str) -> None:
    """Revoke a grant. Once revoked, derived admission is rejected.

    ``reason`` is bounded to 2000 chars by the schema.

    Refuses if the grant is not in a state that allows revocation.
    Updates BOTH ``grants.state`` and the canonical ``payload_json``
    so callers reading via ``load_grant`` see the revoked state.
    """
    existing = load_grant(db, grant_id)
    if existing is None:
        from .safety import SafetyError
        raise SafetyError(f"grant {grant_id} not found")
    if existing.state not in ("draft", "active"):
        from .safety import SafetyError
        raise SafetyError(f"grant {grant_id} not in a revokable state ({existing.state})")
    now = int(time.time())
    revoked = existing.model_copy(update={
        "state": "revoked",
        "revoked_at": now,
        "revoked_reason": str(reason)[:2000],
    })
    payload = json.dumps(grant_payload(revoked), sort_keys=True, separators=(",", ":"))
    with db.transaction() as cur:
        cur.execute(
            """
            UPDATE grants
            SET state='revoked',
                revoked_at=?,
                revoked_reason=?,
                payload_json=?
            WHERE grant_id=? AND state IN ('draft','active')
            """,
            (now, str(reason)[:2000], payload, grant_id),
        )
        if cur.rowcount == 0:
            from .safety import SafetyError
            raise SafetyError(
                f"grant {grant_id} not in a revokable state (already revoked, expired, or unknown)"
            )


def load_grant(db: Database, grant_id: str) -> AutonomyGrant | None:
    cur = db._conn.execute(
        "SELECT payload_json FROM grants WHERE grant_id=?",
        (grant_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return AutonomyGrant.model_validate(json.loads(row["payload_json"]))


def load_grant_by_digest(db: Database, digest: str) -> AutonomyGrant | None:
    cur = db._conn.execute(
        "SELECT payload_json FROM grants WHERE grant_digest=?",
        (digest,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return AutonomyGrant.model_validate(json.loads(row["payload_json"]))


def expire_orphan_grants(db: Database, *, now: int | None = None) -> list[str]:
    """Mark any ACTIVE grant with ``grant_expires_at`` past current
    time as ``expired``.

    Returns the list of grant ids transitioned.
    """
    now = int(now if now is not None else time.time())
    revoked: list[str] = []
    with db.transaction() as cur:
        cur.execute(
            "SELECT grant_id, payload_json FROM grants WHERE state='active'"
        )
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            try:
                g = AutonomyGrant.model_validate(json.loads(r["payload_json"]))
            except Exception:
                continue
            if g.budget.is_expired(now):
                cur.execute(
                    "UPDATE grants SET state='expired' WHERE grant_id=?",
                    (g.grant_id,),
                )
                revoked.append(g.grant_id)
    return revoked


def derive_initial_ledger(db: Database, *, campaign_id: str, grant: AutonomyGrant, family_id: str) -> BudgetLedgerEntry:
    """Create the immutable initial budget ledger for a campaign.

    P06 follow-up #3 item 12: V1 of campaign-v2 has exactly ONE
    authoritative budget ledger per campaign — the single campaign
    ledger IS the family ledger. We refuse to create a second ledger
    for the same campaign so an unbounded second family can never be
    minted; rechunks/revisions reuse the same ledger.
    """
    existing = db._conn.execute(
        "SELECT ledger_id FROM budget_ledgers WHERE campaign_id=? LIMIT 1",
        (campaign_id,),
    ).fetchone()
    if existing is not None:
        raise SafetyError(
            f"campaign {campaign_id!r} already has budget ledger "
            f"{existing['ledger_id']!r}; the single campaign ledger is the "
            f"authoritative family ledger (no second family)"
        )
    ledger = BudgetLedgerEntry(
        ledger_id=f"bl-{campaign_id}",
        campaign_id=campaign_id,
        grant_id=grant.grant_id,
        family_id=family_id,
        bounds=grant.budget,
    )
    with db.transaction() as cur:
        cur.execute(
            """
            INSERT INTO budget_ledgers (
                ledger_id, campaign_id, grant_id, family_id, revision,
                bounds_json
            ) VALUES (?, ?, ?, ?, 1, ?)
            """,
            (ledger.ledger_id, ledger.campaign_id, ledger.grant_id,
             ledger.family_id, json.dumps(ledger.bounds.model_dump(mode="json"))),
        )
    return ledger


def _digest_dict(d: dict[str, Any]) -> str:
    import hashlib
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


__all__ = [
    "GrantActivationResult",
    "grant_payload",
    "activate_grant",
    "revoke_grant",
    "load_grant",
    "load_grant_by_digest",
    "expire_orphan_grants",
    "derive_initial_ledger",
]
