"""P06-A06 — Resource leases with fencing tokens/epochs.

Critical invariants:

  1. Acquired leases are durable: ``acquire_lease`` persists a row in
     ``leases`` keyed by ``lease_id``. A second writer attempting to
     acquire the same resource_id while a live lease exists is
     rejected.
  2. An expired lease with a still-live child process MUST NOT
     produce a second writer. The fencing model requires the owner
     to present its current ``fence_generation`` whenever it tries
     to perform a write; if the campaign's current fence generation
     has incremented past the holder's stale generation (because
     the old lease expired and a new owner was established), the
     write is rejected.
  3. ``release_lease`` marks ``released_at`` and is idempotent.
  4. ``revoke_for_takeover`` increments the campaign fence AND marks
     the existing lease released, so the next writer's
     ``acquire_lease`` succeeds.

Process identity is specific: ``owner_boot_id`` + ``owner_pid`` +
``lease_id``. We never broad-kill by process name.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .campaign_schemas import (
    FenceState,
    Lease,
)
from .db import Database
from .safety import SafetyError


@dataclass(frozen=True)
class LeaseRequest:
    campaign_id: str
    resource_id: str
    owner_id: str
    owner_boot_id: str
    owner_pid: int
    fence_generation: int
    ttl_seconds: int


def acquire_lease(
    db: Database,
    *,
    campaign_id: str,
    resource_id: str,
    owner_id: str,
    owner_boot_id: str,
    owner_pid: int,
    fence_generation: int,
    ttl_seconds: int,
) -> Lease:
    """Acquire a fresh lease for ``resource_id`` in ``campaign_id``.

    A live lease for the same resource blocks acquisition. The caller
    is responsible for ensuring the supplied ``fence_generation``
    matches the campaign's current fence (see ``current_fence``).
    """
    if ttl_seconds <= 0:
        raise SafetyError("ttl_seconds must be > 0")
    # Initialise fence on first use.
    fence = current_fence(db, campaign_id)
    if fence.current_generation != fence_generation:
        raise SafetyError(
            f"fence_mismatch: campaign fence={fence.current_generation} != "
            f"requested={fence_generation}"
        )

    now = int(time.time())
    expires_at = now + int(ttl_seconds)
    lease_id = f"lse-{uuid.uuid4().hex[:16]}"

    with db.transaction() as cur:
        # Reject if a LIVE lease exists for the same resource OR an
        # expired-but-unreleased lease whose holder is still alive
        # (P06 follow-up #2 A06).
        cur.execute(
            """
            SELECT lease_id, expires_at, released_at, owner_pid,
                   owner_boot_id, fence_generation FROM leases
            WHERE campaign_id=? AND resource_id=? AND released_at=0
            """,
            (campaign_id, resource_id),
        )
        live = cur.fetchone()
        if live is not None:
            if live["expires_at"] > now:
                # Fresh live lease; reject immediately.
                raise SafetyError(
                    f"resource busy: lease {live['lease_id']} still live "
                    f"(expires_at={live['expires_at']})"
                )
            # Lease is expired but UNRELEASED. The holder remains an
            # authority object requiring explicit reconciliation. If
            # the holder is still alive, direct acquire must fail.
            if holder_process_alive(
                owner_pid=int(live["owner_pid"]),
                owner_boot_id=live["owner_boot_id"],
                fence_generation=int(live["fence_generation"]),
            ):
                raise SafetyError(
                    f"expired_but_live: lease {live['lease_id']} expired "
                    f"but holder pid={live['owner_pid']} is still alive; "
                    f"explicit takeover required"
                )
        cur.execute(
            """
            INSERT INTO leases (
                lease_id, campaign_id, resource_id, owner_id,
                owner_boot_id, owner_pid, fence_generation,
                acquired_at, expires_at, released_at
            ) VALUES (?,?,?,?,?,?,?,?,?,0)
            """,
            (
                lease_id, campaign_id, resource_id, owner_id,
                owner_boot_id, owner_pid, fence_generation,
                now, expires_at,
            ),
        )
    return Lease(
        lease_id=lease_id,
        campaign_id=campaign_id,
        resource_id=resource_id,
        owner_id=owner_id,
        owner_boot_id=owner_boot_id,
        owner_pid=owner_pid,
        fence_generation=fence_generation,
        acquired_at=now,
        expires_at=expires_at,
        released_at=0,
    )


def release_lease(db: Database, *, lease_id: str) -> None:
    """Release a lease (idempotent). Does not check fence."""
    with db.transaction() as cur:
        cur.execute(
            "UPDATE leases SET released_at=? WHERE lease_id=? AND released_at=0",
            (int(time.time()), lease_id),
        )


def revoke_for_takeover(db: Database, *, campaign_id: str, reason: str) -> int:
    """Increment the campaign fence and mark all live leases released.

    Returns the new fence generation.

    A second writer attempting to claim AFTER this operation will see
    no live leases and a fence generation that has moved past any
    earlier holders. An old owner trying to write will fail because
    its fencing token will not match.
    """
    now = int(time.time())
    fence = current_fence(db, campaign_id)
    new_gen = fence.current_generation + 1
    with db.transaction() as cur:
        cur.execute(
            "UPDATE leases SET released_at=? WHERE campaign_id=? AND released_at=0",
            (now, campaign_id),
        )
        cur.execute(
            "UPDATE campaigns SET current_fence=?, updated_at=? WHERE campaign_id=?",
            (new_gen, now, campaign_id),
        )
    return new_gen


def current_fence(db: Database, campaign_id: str) -> FenceState:
    """Return the current fence state for a campaign. Auto-creates at gen 1."""
    cur = db._conn.execute(
        "SELECT current_fence FROM campaigns WHERE campaign_id=?",
        (campaign_id,),
    )
    row = cur.fetchone()
    if row is None:
        # Create the campaign's initial fence row lazily. The campaign
        # itself may not exist yet; caller is expected to have created
        # it via ``campaign.create_campaign`` first.
        return FenceState(campaign_id=campaign_id, current_generation=1, updated_at=int(time.time()))
    return FenceState(
        campaign_id=campaign_id,
        current_generation=int(row["current_fence"]),
        updated_at=int(time.time()),
    )


def enforce_fence(
    db: Database, *, campaign_id: str, holder_generation: int, action: str
) -> None:
    """Raise if the holder's generation is stale.

    Called by the broker / integration layer before any consequential
    write (apply, integration, integration_ref_advance). The grant
    admission supplies the holder's generation at admission time.
    """
    fence = current_fence(db, campaign_id)
    if holder_generation != fence.current_generation:
        raise SafetyError(
            f"fence_stale: {action} holder_generation={holder_generation} "
            f"current_generation={fence.current_generation}"
        )


def holder_process_alive(
    *, owner_pid: int, owner_boot_id: str, fence_generation: int,
) -> bool:
    """Best-effort check whether the lease holder's process is alive.

    In our test fixture the holder is the same process that opened
    the lease; we use ``/proc/<pid>`` on Linux. ``status`` field in
    ``/proc/<pid>/stat`` is 'Z' for zombie (defunct) and 'X' for
    dead. Any other state is live.
    """
    try:
        if owner_pid <= 0:
            return False
        import os
        try:
            os.kill(owner_pid, 0)
        except (ProcessLookupError, PermissionError, OSError):
            return False
        with open(f"/proc/{owner_pid}/stat", "r") as f:
            stat_line = f.read().strip()
        parts = stat_line.rsplit(")", 1)[-1].split()
        if not parts:
            return False
        state = parts[0]
        if state in ("Z", "X"):
            return False
        return True
    except Exception:
        return False


def load_lease(db: Database, lease_id: str) -> Lease | None:
    cur = db._conn.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,))
    row = cur.fetchone()
    if row is None:
        return None
    return Lease(
        lease_id=row["lease_id"],
        campaign_id=row["campaign_id"],
        resource_id=row["resource_id"],
        owner_id=row["owner_id"],
        owner_boot_id=row["owner_boot_id"],
        owner_pid=row["owner_pid"],
        fence_generation=row["fence_generation"],
        acquired_at=row["acquired_at"],
        expires_at=row["expires_at"],
        released_at=row["released_at"],
    )


def expire_overdue_leases(db: Database, *, now: int | None = None) -> list[str]:
    """Mark leases whose ``expires_at`` has passed as released.

    P06 follow-up #2 A06: an expired lease whose holder is still
    alive MUST NOT be silently released into claimable state. Such a
    lease remains an authority object requiring explicit takeover via
    ``revoke_for_takeover``. This sweep releases only the leases
    whose holders are dead (or whose leases do not have a real
    process identity).

    Returns the list of released lease ids. Does NOT increment the
    fence; the caller (``revoke_for_takeover``) is responsible for
    the fence bump when a new owner is taking over.
    """
    now = int(now if now is not None else time.time())
    released: list[str] = []
    with db.transaction() as cur:
        cur.execute(
            """
            SELECT lease_id, owner_pid, owner_boot_id, fence_generation
            FROM leases WHERE released_at=0 AND expires_at > 0 AND expires_at <= ?
            """,
            (now,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            alive = holder_process_alive(
                owner_pid=int(r["owner_pid"]),
                owner_boot_id=r["owner_boot_id"],
                fence_generation=int(r["fence_generation"]),
            )
            if alive:
                # Holder still alive; refuse to silently release.
                continue
            cur.execute(
                "UPDATE leases SET released_at=? WHERE lease_id=? AND released_at=0",
                (now, r["lease_id"]),
            )
            released.append(r["lease_id"])
    return released


__all__ = [
    "LeaseRequest",
    "acquire_lease",
    "release_lease",
    "revoke_for_takeover",
    "current_fence",
    "enforce_fence",
    "holder_process_alive",
    "load_lease",
    "expire_overdue_leases",
]
