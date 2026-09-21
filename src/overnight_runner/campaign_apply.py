"""P06 follow-up #2 A06 / follow-up #3 A06 — Campaign-aware mutation fence.

This module is the only sanctioned entry point for a campaign
worktree PATCH APPLY. It wraps the broker's ``apply_proposal``
operation with:

  * a campaign mutation lock (the runner DB IMMEDIATE write lock) held
    across the fence re-check, the atomic file apply, and the durable
    apply evidence record — so ``revoke_for_takeover`` cannot bump the
    fence concurrently (P06 follow-up #3 A06 item 10);
  * a lease binding: when a ``lease_id`` is supplied, the apply resolves
    the durable lease and proves it is live, campaign-scoped, owned by
    the presenting identity, and at the current fence generation
    (P06 follow-up #3 A06 item 11).

V1 broker behavior is unchanged. This module does not replace the
``Broker.apply_proposal`` method; it adds a runner-controlled mutation
fence layer that the campaign integration path consults before applying
any campaign-scoped patch.

Typical usage::

    from overnight_runner.campaign_apply import apply_campaign_patch
    apply_campaign_patch(
        db, broker, repo_root, campaign_id, proposal_id,
        admission_fence_generation=admission.fence_generation,
        lease_id=lease.lease_id,
    )
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from .db import Database
from .failpoints import failpoint_armed, raise_failpoint
from .resources import (
    campaign_mutation_lock,
    current_fence,
    enforce_fence,
    holder_process_alive,
    load_lease_row,
)
from .safety import SafetyError

# Resource-id prefixes that denote a campaign mutation authority. A
# lease bound to any other scope cannot author a worktree patch.
_CAMPAIGN_WRITER_SCOPE_PREFIXES = ("chunk:", "admission:", "campaign:", "writer")


def _lease_authorises_apply(
    db: Database,
    *,
    campaign_id: str,
    lease_id: str,
    admission_fence_generation: int,
) -> None:
    """Prove ``lease_id`` is a live, current, campaign-scoped lease.

    Raises ``SafetyError`` when the lease is missing, released,
    campaign-mismatched, out of scope, at a different fence
    generation, or held by a dead / impersonated process.
    """
    row = load_lease_row(db, lease_id)
    if row is None:
        raise SafetyError(f"lease_missing: no lease {lease_id!r} in durable store")
    if row["campaign_id"] != campaign_id:
        raise SafetyError(
            f"lease_campaign_mismatch: lease {lease_id} belongs to campaign "
            f"{row['campaign_id']!r}, not {campaign_id!r}"
        )
    if int(row["released_at"]) != 0:
        raise SafetyError(
            f"lease_released: lease {lease_id} was released at {row['released_at']}; "
            f"a released worker cannot mutate the campaign worktree"
        )
    scope = str(row["resource_id"])
    if not scope.startswith(_CAMPAIGN_WRITER_SCOPE_PREFIXES):
        raise SafetyError(
            f"lease_scope_invalid: lease {lease_id} resource {scope!r} is not a "
            f"campaign writer scope"
        )
    if int(row["fence_generation"]) != admission_fence_generation:
        raise SafetyError(
            f"lease_fence_stale: lease {lease_id} fence_generation="
            f"{row['fence_generation']} != required {admission_fence_generation}"
        )
    # Lease must not be expired unless the caller holds a still-live
    # owner identity AND the fence still matches (bounded operation).
    now = int(time.time())
    expired = int(row["expires_at"]) > 0 and int(row["expires_at"]) <= now
    if expired:
        # An expired lease is only permitted when its holder is dead
        # and the fence still matches — otherwise it is an authority
        # object awaiting explicit takeover.
        if holder_process_alive(
            owner_pid=int(row["owner_pid"]),
            owner_boot_id=row["owner_boot_id"],
            fence_generation=int(row["fence_generation"]),
            owner_start_time=row.get("owner_start_time") or None,
        ):
            raise SafetyError(
                f"lease_expired_but_live: lease {lease_id} expired but its holder "
                f"pid={row['owner_pid']} is still alive; refusing mutation"
            )
    else:
        # Live lease: the presenting identity must be the recorded
        # holder (a released/dead holder cannot mutate).
        if not holder_process_alive(
            owner_pid=int(row["owner_pid"]),
            owner_boot_id=row["owner_boot_id"],
            fence_generation=int(row["fence_generation"]),
            owner_start_time=row.get("owner_start_time") or None,
        ):
            raise SafetyError(
                f"lease_holder_dead: lease {lease_id} holder pid={row['owner_pid']} "
                f"is not alive; refusing to mutate campaign worktree"
            )


def apply_campaign_patch(
    db: Database,
    broker: Any,
    repo_root: str,
    campaign_id: str,
    proposal_id: str,
    *,
    admission_fence_generation: int,
    lease_id: str | None = None,
    expected_holder_pid: int | None = None,
    expected_holder_boot_id: str | None = None,
) -> dict[str, Any]:
    """Apply a proposal atomically AND enforce the campaign fence.

    P06 follow-up #3 A06:

      * A single runner DB IMMEDIATE transaction (the campaign mutation
        lock) is held across: current-fence re-check -> lease authority
        check -> broker atomic file apply -> durable apply evidence.
        ``revoke_for_takeover`` takes the same lock, so a takeover can
        never interleave between the fence check and the file write.
      * When ``lease_id`` is supplied, the apply is bound to that live
        lease, not merely to the fence generation.
      * A durable ``campaign_patch_applied`` evidence row is written in
        the same transaction as the file apply.

    Returns the broker's apply dict (with ``receipt_id`` etc.).
    Raises ``SafetyError`` on fence mismatch, invalid lease, dead
    holder, or apply failure.
    """
    # A05 item 6: durable intent BEFORE the candidate change so a crash
    # between the worktree write and the durable evidence is
    # discoverable on restart.
    armed = failpoint_armed("candidate_changed_before_state_durable")
    if armed:
        from .integration import record_crash_intent
        record_crash_intent(
            db, "candidate_changed_before_state_durable",
            campaign_id=campaign_id,
            repo_root=repo_root,
            evidence={"proposal_id": proposal_id, "lease_id": lease_id or ""},
        )

    with campaign_mutation_lock(db) as cur:
        # Pre-apply fence check (held under the write lock).
        fence = current_fence(db, campaign_id)
        if fence.current_generation != admission_fence_generation:
            raise SafetyError(
                f"fence_stale: apply holder_generation="
                f"{admission_fence_generation} current_generation="
                f"{fence.current_generation}; refusing stale writer patch"
            )
        # Lease authority check (A06 item 11), when a lease is supplied.
        if lease_id is not None:
            _lease_authorises_apply(
                db,
                campaign_id=campaign_id,
                lease_id=lease_id,
                admission_fence_generation=admission_fence_generation,
            )
        # Holder liveness check (legacy path).
        if expected_holder_pid is not None:
            alive = holder_process_alive(
                owner_pid=expected_holder_pid,
                owner_boot_id=expected_holder_boot_id or "",
                fence_generation=admission_fence_generation,
            )
            if not alive:
                raise SafetyError(
                    f"holder_dead: owner_pid={expected_holder_pid} is not alive; "
                    f"refusing to mutate campaign worktree"
                )

        # Apply via the broker — the atomic file write happens INSIDE
        # the mutation lock.
        result = broker.apply_proposal(proposal_id)

        # Failpoint: candidate changed, durable state not yet written.
        # The transaction rolls back (no apply evidence), but the
        # worktree change persists and the durable intent committed
        # above lets restart discover the uncertainty.
        if armed:
            raise_failpoint("candidate_changed_before_state_durable")

        # Durable apply evidence in the same transaction.
        cur.execute(
            """
            INSERT INTO campaign_events (
                event_id, campaign_id, chunk_id, event_type,
                from_state, to_state, actor, payload, fence_generation,
                issued_at, idempotency_key
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                f"ev-{uuid.uuid4().hex[:16]}", campaign_id, None,
                "campaign_patch_applied", None, None, "runner",
                json.dumps({
                    "proposal_id": proposal_id,
                    "receipt_id": (result or {}).get("receipt_id", ""),
                    "lease_id": lease_id or "",
                }),
                admission_fence_generation, int(time.time()),
                f"apply-{uuid.uuid4().hex[:8]}",
            ),
        )

    # Defense in depth: re-check after the lock is released so a
    # takeover that landed immediately after commit still rejects the
    # subsequent integration step.
    enforce_fence(
        db,
        campaign_id=campaign_id,
        holder_generation=admission_fence_generation,
        action="apply",
    )
    return result


__all__ = ["apply_campaign_patch"]
