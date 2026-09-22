"""P08 (A03) — runner-owned admission-lease renewal for a resumed obligation.

A resumed capacity obligation must be able to continue the SAME admission
after its original worker lease is gone. Historically the only way to do
that was to rewrite ``admissions.lease_id`` — which mutates the immutable
original authority lineage — and the cross-layer test harness did exactly
that with direct SQL. That is NOT autonomous resume.

This module adds the smallest additive runner authority operation:

  * the original ``AdmissionReceipt`` (and its original lease identity)
    stays IMMUTABLE root authority;
  * a renewal is persisted as an explicit DESCENDANT authority event in
    ``admission_lease_bindings``;
  * ``effective_admission_lease_id`` resolves which lease currently
    authorizes mutation for an admission.

Only the runner can mint this authority. Hermes, the model, and the worker
have no seam to call it, and there is no parameter through which a caller
can name an arbitrary lease.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from .db import Database
from .resources import acquire_lease, current_fence, load_lease_row
from .safety import SafetyError

ISSUER = "runner"
REASON_CAPACITY_RESUME = "capacity_resume"


def _binding_row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row)


def load_binding(db: Database, binding_id: str) -> dict[str, Any] | None:
    row = db._conn.execute(
        "SELECT * FROM admission_lease_bindings WHERE binding_id=?",
        (binding_id,),
    ).fetchone()
    return _binding_row_to_dict(row) if row is not None else None


def list_bindings(db: Database, admission_id: str) -> list[dict[str, Any]]:
    """Return the runner-issued renewal bindings, oldest first."""
    rows = db._conn.execute(
        "SELECT * FROM admission_lease_bindings WHERE admission_id=? "
        "ORDER BY issued_at ASC, rowid ASC",
        (admission_id,),
    ).fetchall()
    return [_binding_row_to_dict(r) for r in rows]


def effective_admission_lease_id(
    db: Database, admission_id: str
) -> tuple[str, str]:
    """Resolve the EFFECTIVE lease for an admission through runner authority.

    Returns ``(lease_id, source)`` where ``source`` is ``"original"`` when
    the immutable admission lease is still the authority, or ``"renewal"``
    when the latest runner-issued renewal binding is. Raises when the
    admission does not exist.

    A caller cannot substitute an arbitrary lease id: the only ids this
    function can return are the admission's own recorded lease or the
    ``replacement_lease_id`` of a runner-issued binding.
    """
    arow = db._conn.execute(
        "SELECT lease_id FROM admissions WHERE admission_id=?",
        (admission_id,),
    ).fetchone()
    if arow is None:
        raise SafetyError(f"admission {admission_id!r} not found")
    original = arow["lease_id"] or ""
    bindings = list_bindings(db, admission_id)
    if not bindings:
        return original, "original"
    return bindings[-1]["replacement_lease_id"] or original, "renewal"


def refresh_admission_lease_for_resume(
    db: Database,
    *,
    admission_id: str,
    wake_claim_id: str,
    reason: str = REASON_CAPACITY_RESUME,
    owner_id: str,
    owner_pid: int,
    ttl_seconds: int = 300,
    idempotency_key: str = "",
    owner_start_time: str | None = None,
    now: int | None = None,
) -> dict[str, Any]:
    """Issue a runner-owned replacement lease for a resumed admission.

    Authority is resolved entirely from durable runner state — the caller
    supplies the admission and the P07 resume/wake lineage, never a lease.

    Requires ALL of:

      * the admission exists and its durable chunk/campaign/grant resolve;
      * the ``wake_claim_id`` is a durable P07 claim for THIS campaign;
      * campaign continuation authority (state, active/non-revoked/
        non-expired grant, trusted budget headroom, not PAUSED);
      * the admission's original lease is NO LONGER valid for mutation
        (released or expired) — a live original lease is never replaced;
      * the current campaign fence equals the admission fence.

    Repeated calls with the same ``idempotency_key`` return the SAME
    binding and do NOT acquire a second lease.

    Returns the new effective lease authority.
    """
    now = int(now if now is not None else time.time())

    arow = db._conn.execute(
        "SELECT c.campaign_id AS campaign_id, a.chunk_id AS chunk_id, "
        "a.lease_id AS lease_id, a.fence_generation AS fence_generation, "
        "a.budget_ledger_id AS budget_ledger_id "
        "FROM admissions a JOIN chunks c ON c.chunk_id = a.chunk_id "
        "WHERE a.admission_id=?",
        (admission_id,),
    ).fetchone()
    if arow is None:
        raise SafetyError(
            f"resume renewal refused: admission {admission_id!r} not found"
        )
    campaign_id = arow["campaign_id"] or ""
    chunk_id = arow["chunk_id"] or ""
    prior_lease_id = arow["lease_id"] or ""
    admission_fence = int(arow["fence_generation"])

    # Idempotent replay: the same resume request never mints a second lease.
    if idempotency_key:
        ex = db._conn.execute(
            "SELECT * FROM admission_lease_bindings WHERE admission_id=? "
            "AND idempotency_key=?",
            (admission_id, idempotency_key),
        ).fetchone()
        if ex is not None:
            return _result(db, _binding_row_to_dict(ex), replay=True)

    # P07 resume lineage: the claim must be a durable claim for THIS campaign.
    if not wake_claim_id:
        raise SafetyError(
            "resume renewal refused: no P07 wake/resume lineage supplied"
        )
    wrow = db._conn.execute(
        "SELECT campaign_id FROM wake_claims WHERE claim_id=?",
        (wake_claim_id,),
    ).fetchone()
    if wrow is None:
        raise SafetyError(
            f"resume renewal refused: unknown wake claim {wake_claim_id!r}"
        )
    if (wrow["campaign_id"] or "") != campaign_id:
        raise SafetyError(
            f"resume renewal refused: wake claim {wake_claim_id!r} belongs to "
            f"campaign {wrow['campaign_id']!r}, not {campaign_id!r}"
        )

    crow = db._conn.execute(
        "SELECT grant_id FROM campaigns WHERE campaign_id=?",
        (campaign_id,),
    ).fetchone()
    if crow is None:
        raise SafetyError(f"campaign {campaign_id!r} not registered")
    grant_id = crow["grant_id"] or ""

    # Campaign continuation authority (state + active grant + trusted budget
    # headroom + not PAUSED): one definition, reused.
    from .admission import check_campaign_continuation, _load_ledger
    check_campaign_continuation(
        db, campaign_id=campaign_id, grant_id=grant_id, now=now
    )

    # The ledger recorded on the admission must still exist and be trusted.
    ledger_id = arow["budget_ledger_id"] or ""
    if ledger_id:
        led = _load_ledger(db, ledger_id)
        if led is None:
            raise SafetyError(
                f"resume renewal refused: budget ledger {ledger_id!r} vanished"
            )
        if (led.campaign_id or "") != campaign_id:
            raise SafetyError(
                "resume renewal refused: budget ledger belongs to another campaign"
            )

    # The admission's original lease must no longer authorize mutation.
    prior = load_lease_row(db, prior_lease_id) if prior_lease_id else None
    if prior is not None and int(prior["released_at"]) == 0:
        exp = int(prior["expires_at"])
        if exp == 0 or exp > now:
            raise SafetyError(
                f"resume renewal refused: admission lease {prior_lease_id!r} is "
                f"still valid for mutation; no renewal needed"
            )

    # Current campaign fence must match the admission's fence.
    cur_fence = current_fence(db, campaign_id)
    if cur_fence.current_generation != admission_fence:
        raise SafetyError(
            f"resume renewal refused: campaign fence="
            f"{cur_fence.current_generation} != admission fence={admission_fence}"
        )

    # Acquire the replacement lease through runner authority.
    lease = acquire_lease(
        db,
        campaign_id=campaign_id,
        resource_id=f"admission:{chunk_id}",
        owner_id=owner_id,
        owner_boot_id="",
        owner_pid=owner_pid,
        fence_generation=admission_fence,
        ttl_seconds=ttl_seconds,
        owner_start_time=owner_start_time,
    )

    binding_id = f"alb-{uuid.uuid4().hex[:16]}"
    try:
        with db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO admission_lease_bindings (
                    binding_id, admission_id, campaign_id, chunk_id,
                    prior_lease_id, replacement_lease_id, fence_generation,
                    wake_claim_id, reason, issued_at, issuer, idempotency_key
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    binding_id, admission_id, campaign_id, chunk_id,
                    prior_lease_id, lease.lease_id, admission_fence,
                    wake_claim_id, reason, now, ISSUER, idempotency_key,
                ),
            )
    except Exception:
        # Never leave a lease that no binding authorizes.
        from .resources import release_lease
        release_lease(db, lease_id=lease.lease_id)
        if idempotency_key:
            ex = db._conn.execute(
                "SELECT * FROM admission_lease_bindings WHERE admission_id=? "
                "AND idempotency_key=?",
                (admission_id, idempotency_key),
            ).fetchone()
            if ex is not None:
                return _result(db, _binding_row_to_dict(ex), replay=True)
        raise
    rec = load_binding(db, binding_id)
    assert rec is not None
    return _result(db, rec, replay=False)


def _result(db: Database, binding: dict[str, Any], *, replay: bool) -> dict[str, Any]:
    lease = load_lease_row(db, binding["replacement_lease_id"]) or {}
    return {
        "binding_id": binding["binding_id"],
        "admission_id": binding["admission_id"],
        "campaign_id": binding["campaign_id"],
        "chunk_id": binding["chunk_id"],
        "prior_lease_id": binding["prior_lease_id"],
        "effective_lease_id": binding["replacement_lease_id"],
        "fence_generation": int(binding["fence_generation"]),
        "wake_claim_id": binding["wake_claim_id"],
        "reason": binding["reason"],
        "issued_at": int(binding["issued_at"]),
        "issuer": binding["issuer"],
        "idempotent_replay": replay,
        "lease_expires_at": int(lease.get("expires_at", 0)),
    }


__all__ = [
    "ISSUER",
    "REASON_CAPACITY_RESUME",
    "refresh_admission_lease_for_resume",
    "effective_admission_lease_id",
    "load_binding",
    "list_bindings",
]
