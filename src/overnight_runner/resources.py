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
from contextlib import contextmanager
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


# ---------------------------------------------------------------------------
# Process identity (P06 follow-up #3 A06, item 9)
# ---------------------------------------------------------------------------

def host_boot_id() -> str:
    """Return the host boot id (``/proc/sys/kernel/random/boot_id``).

    Returns ``""`` when unavailable (non-Linux / restricted /proc).
    The boot id changes across a host reboot, so a stored boot id that
    no longer matches the live host proves the lease predates a reboot
    and can never be held by a live process.
    """
    try:
        with open("/proc/sys/kernel/random/boot_id", "r") as f:
            return f.read().strip()
    except OSError:
        return ""


def proc_start_time(pid: int) -> str:
    """Return field 22 (starttime) of ``/proc/<pid>/stat`` as a string.

    ``starttime`` is expressed in clock ticks since boot; combined with
    the boot id it is a stable process identity that survives PID
    reuse detection (a reused PID has a different starttime). Returns
    ``""`` when the process is gone or the field is unreadable.
    """
    if pid <= 0:
        return ""
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            line = f.read()
    except OSError:
        return ""
    # The comm field may contain spaces/parentheses; split after the
    # last ')'.
    try:
        rest = line.rsplit(")", 1)[-1].strip().split()
    except Exception:
        return ""
    # After comm, field 3 is state (index 0 here); starttime is field
    # 22 overall => index 22 - 3 + 1 = 20 in ``rest``.
    idx = 22 - 3
    if len(rest) <= idx:
        return ""
    return rest[idx]


def process_identity(pid: int) -> dict[str, str]:
    """Return ``{"boot_id":..., "start_time":...}`` for ``pid``."""
    return {"boot_id": host_boot_id(), "start_time": proc_start_time(pid)}


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
    owner_start_time: str | None = None,
) -> Lease:
    """Acquire a fresh lease for ``resource_id`` in ``campaign_id``.

    A live lease for the same resource blocks acquisition. The caller
    is responsible for ensuring the supplied ``fence_generation``
    matches the campaign's current fence (see ``current_fence``).

    P06 follow-up #3 (A06 item 9): the lease records the EXACT process
    identity. We prefer the live host boot id and the process
    start-time (from ``/proc``) so a reused PID cannot impersonate the
    old worker. Callers may still pass a legacy ``owner_boot_id``; when
    the host boot id is readable it is authoritative.
    """
    with db.transaction() as cur:
        return _acquire_lease_cur(
            cur, db,
            campaign_id=campaign_id,
            resource_id=resource_id,
            owner_id=owner_id,
            owner_boot_id=owner_boot_id,
            owner_pid=owner_pid,
            fence_generation=fence_generation,
            ttl_seconds=ttl_seconds,
            owner_start_time=owner_start_time,
        )


def _acquire_lease_cur(
    cur: Any,
    db: Database,
    *,
    campaign_id: str,
    resource_id: str,
    owner_id: str,
    owner_boot_id: str,
    owner_pid: int,
    fence_generation: int,
    ttl_seconds: int,
    owner_start_time: str | None = None,
) -> Lease:
    """Lease acquisition that runs on an EXISTING cursor/transaction.

    Used by the single-transaction derivation path
    (``admission.derive_admission``) so the reservation, lease, chunk,
    and admission rows commit atomically (P06 follow-up #3 A03 item 2).
    """
    if ttl_seconds <= 0:
        raise SafetyError("ttl_seconds must be > 0")
    fence = current_fence(db, campaign_id)
    if fence.current_generation != fence_generation:
        raise SafetyError(
            f"fence_mismatch: campaign fence={fence.current_generation} != "
            f"requested={fence_generation}"
        )
    real_boot = host_boot_id()
    stored_boot = real_boot or owner_boot_id
    stored_start = (
        owner_start_time if owner_start_time is not None
        else proc_start_time(owner_pid)
    )
    now = int(time.time())
    expires_at = now + int(ttl_seconds)
    lease_id = f"lse-{uuid.uuid4().hex[:16]}"
    # Reject if a LIVE lease exists for the same resource OR an
    # expired-but-unreleased lease whose holder is still alive
    # (P06 follow-up #2 A06).
    cur.execute(
        """
        SELECT lease_id, expires_at, released_at, owner_pid,
               owner_boot_id, owner_start_time, fence_generation
        FROM leases
        WHERE campaign_id=? AND resource_id=? AND released_at=0
        """,
        (campaign_id, resource_id),
    )
    live = cur.fetchone()
    if live is not None:
        if live["expires_at"] > now:
            raise SafetyError(
                f"resource busy: lease {live['lease_id']} still live "
                f"(expires_at={live['expires_at']})"
            )
        if holder_process_alive(
            owner_pid=int(live["owner_pid"]),
            owner_boot_id=live["owner_boot_id"],
            fence_generation=int(live["fence_generation"]),
            owner_start_time=_row_get(live, "owner_start_time") or None,
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
            acquired_at, expires_at, released_at, owner_start_time
        ) VALUES (?,?,?,?,?,?,?,?,?,0,?)
        """,
        (
            lease_id, campaign_id, resource_id, owner_id,
            stored_boot, owner_pid, fence_generation,
            now, expires_at, stored_start,
        ),
    )
    return Lease(
        lease_id=lease_id,
        campaign_id=campaign_id,
        resource_id=resource_id,
        owner_id=owner_id,
        owner_boot_id=stored_boot,
        owner_pid=owner_pid,
        fence_generation=fence_generation,
        acquired_at=now,
        expires_at=expires_at,
        released_at=0,
    )


def _row_get(row: Any, key: str) -> Any:
    """Return ``row[key]`` or ``None`` when the column is absent."""
    try:
        keys = row.keys()
    except AttributeError:
        return None
    if key not in keys:
        return None
    return row[key]


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
    *,
    owner_pid: int,
    owner_boot_id: str,
    fence_generation: int,
    owner_start_time: str | None = None,
) -> bool:
    """Best-effort check whether the lease holder's process is alive.

    Identity checks (P06 follow-up #3 A06 item 9), applied in order:

      1. ``owner_pid`` must be > 0 and signalable.
      2. ``/proc/<pid>/stat`` must exist and its state must not be
         ``Z`` (zombie) or ``X`` (dead).
      3. If ``owner_start_time`` is provided (a lease that recorded a
         real process identity), the live host boot id must equal
         ``owner_boot_id`` when both are readable, and the live
         ``starttime`` must equal ``owner_start_time``. A mismatch
         proves the PID was reused or the lease predates a reboot.

    Step 3 is the anti-PID-reuse guarantee. Callers that pass only a
    legacy synthetic boot id (no start time) retain the historical
    PID-liveness behaviour, since no authoritative host identity is
    available to compare against.

    We never broad-kill by process name; identity is specific.
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

        if owner_start_time is not None:
            real_boot = host_boot_id()
            if owner_boot_id and real_boot and owner_boot_id != real_boot:
                # Recorded identity predates a reboot (or the holder is
                # not the same host): not the lease holder.
                return False
            live_start = proc_start_time(owner_pid)
            if live_start and live_start != owner_start_time:
                # PID was reused by a different process.
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
            SELECT lease_id, owner_pid, owner_boot_id, owner_start_time,
                   fence_generation
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
                owner_start_time=r.get("owner_start_time") or None,
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


def load_lease_row(db: Database, lease_id: str) -> dict[str, Any] | None:
    """Return the raw lease row (including identity columns) or ``None``."""
    cur = db._conn.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,))
    row = cur.fetchone()
    return dict(row) if row is not None else None


def live_lease_for_campaign(
    db: Database,
    campaign_id: str,
    *,
    resource_id: str | None = None,
) -> dict[str, Any] | None:
    """Return the live (unreleased) lease row for a campaign, if any.

    P06 follow-up #3 (A06 item 11): ``apply_campaign_patch`` resolves
    the caller's lease against this durable record so a fence
    generation alone (a released worker's stale token) cannot author a
    campaign mutation.
    """
    if resource_id is not None:
        cur = db._conn.execute(
            "SELECT * FROM leases WHERE campaign_id=? AND resource_id=? AND released_at=0 "
            "ORDER BY acquired_at DESC LIMIT 1",
            (campaign_id, resource_id),
        )
    else:
        cur = db._conn.execute(
            "SELECT * FROM leases WHERE campaign_id=? AND released_at=0 "
            "ORDER BY acquired_at DESC LIMIT 1",
            (campaign_id,),
        )
    row = cur.fetchone()
    return dict(row) if row is not None else None


@contextmanager
def campaign_mutation_lock(db: Database):
    """Hold the runner DB write lock across a short consequential write.

    P06 follow-up #3 (A06 item 10): the campaign patch apply and the
    takeover/fence increment MUST be serialized so there is no ordering
    in which a stale writer mutates the worktree AFTER the new fence is
    established. SQLite ``BEGIN IMMEDIATE`` acquires the database write
    lock; ``revoke_for_takeover`` uses the same lock, so the two are
    mutually exclusive.

    The lock is deliberately short-lived: it must NOT span model
    inference or long validators — only the fence re-check + atomic
    file apply + durable apply evidence.
    """
    cur = db._conn.cursor()
    try:
        cur.execute("BEGIN IMMEDIATE")
        yield cur
        cur.execute("COMMIT")
    except Exception:
        cur.execute("ROLLBACK")
        raise


__all__ = [
    "LeaseRequest",
    "acquire_lease",
    "release_lease",
    "revoke_for_takeover",
    "current_fence",
    "enforce_fence",
    "holder_process_alive",
    "host_boot_id",
    "proc_start_time",
    "process_identity",
    "load_lease",
    "load_lease_row",
    "live_lease_for_campaign",
    "campaign_mutation_lock",
    "expire_overdue_leases",
]
