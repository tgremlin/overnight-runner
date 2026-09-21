"""P06 follow-up #2 A06 — Campaign-aware mutation fence.

This module is the only sanctioned entry point for a campaign
worktree PATCH APPLY. It wraps the broker's ``apply_proposal``
operation with a fence check BEFORE the write happens, so a stale
writer cannot modify the campaign worktree after takeover.

V1 broker behavior is unchanged. This module does not replace the
``Broker.apply_proposal`` method; it adds a runner-controlled
mutation fence layer that the campaign integration path consults
before applying any campaign-scoped patch.

Typical usage::

    from overnight_runner.campaign_apply import apply_campaign_patch
    apply_campaign_patch(
        db, broker, repo_root, campaign_id, proposal_id,
        admission_fence_generation=admission.fence_generation,
    )
"""
from __future__ import annotations

import time
from typing import Any

from .db import Database
from .resources import current_fence, enforce_fence, holder_process_alive
from .safety import SafetyError


def apply_campaign_patch(
    db: Database,
    broker: Any,
    repo_root: str,
    campaign_id: str,
    proposal_id: str,
    *,
    admission_fence_generation: int,
    expected_holder_pid: int | None = None,
    expected_holder_boot_id: str | None = None,
) -> dict[str, Any]:
    """Apply a proposal atomically AND enforce the campaign fence.

    P06 follow-up #2 A06: before the patch is written, this entry
    point:
      * Re-checks the live campaign fence vs. the admission fence.
      * If a ``expected_holder_pid`` is supplied, verifies the
        owning process is still alive (an old owner cannot mutate
        after takeover).
      * Calls the broker to apply the proposal (single-file write).
      * Calls ``enforce_fence`` AFTER apply so a stale fence still
        rejects the integration step.

    Returns the broker's apply dict (with ``receipt_id`` etc.).
    Raises ``SafetyError`` on fence mismatch, expired holder, or
    apply failure.
    """
    # Pre-apply fence check.
    fence = current_fence(db, campaign_id)
    if fence.current_generation != admission_fence_generation:
        raise SafetyError(
            f"fence_stale: apply holder_generation="
            f"{admission_fence_generation} current_generation="
            f"{fence.current_generation}; refusing stale writer patch"
        )
    # Holder liveness check.
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

    # Apply via the broker.
    result = broker.apply_proposal(proposal_id)
    # Post-apply fence check (so a takeover between apply and
    # integration also rejects).
    enforce_fence(
        db,
        campaign_id=campaign_id,
        holder_generation=admission_fence_generation,
        action="apply",
    )
    return result


__all__ = ["apply_campaign_patch"]
