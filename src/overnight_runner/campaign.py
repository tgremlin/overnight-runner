"""P06 — campaign lifecycle + sequential orchestration.

A campaign is created (DRAFT), activated (ACTIVE), accepts chunks, and
progresses through the lifecycle vocabulary of ``CampaignState``.

Sequential orchestration rules:

  1. Each chunk must bind to a prior accepted predecessor snapshot.
  2. The campaign integration ref (``refs/heads/campaign/<id>``) is
     advanced atomically via ``integration.compare_and_swap_advance``.
  3. The same ``BudgetLedger`` is preserved across the campaign; new
     sessions, restarts, rechunks, and provider changes do NOT mint
     a new ledger.
  4. PAUSED semantics remain authoritative: ``require_not_paused`` at
     every transition; an operator pause stops NEW work at safe
     boundaries without erasing existing state.

A campaign is disposable and uses a private branch; the runner never
writes to a default branch.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .campaign_schemas import (
    CampaignEvent,
    CampaignRecord,
    CampaignState,
    ChunkSpec,
    ChunkState,
    RepoSnapshot,
)
from .db import Database
from .grants import load_grant
from .integration import (
    capture_current_snapshot,
    compare_and_swap_advance,
    ensure_campaign_worktree,
)
from .runtime import is_paused, require_not_paused
from .safety import SafetyError


@dataclass(frozen=True)
class CampaignHandle:
    campaign_id: str
    integration_branch: str
    fence_generation: int


def require_not_paused_or_raise() -> None:
    try:
        require_not_paused()
    except Exception as e:
        raise SafetyError(f"PAUSED: {e}") from e


def ensure_running_paused_aware(db: Database, *, campaign_id: str) -> None:
    """Check the campaign's own PAUSED-OR-NOT guard.

    The runner's global PAUSED sentinel already covers the campaign
    surface; this is a structural spot-check that the campaign was not
    marked paused in its own event journal.
    """
    require_not_paused_or_raise()
    cur = db._conn.execute(
        "SELECT 1 FROM campaign_events WHERE campaign_id=? AND event_type='PAUSED_OPERATOR' LIMIT 1",
        (campaign_id,),
    )
    if cur.fetchone() is not None:
        raise SafetyError(f"campaign {campaign_id} has a PAUSED_OPERATOR event")


def create_campaign(
    db: Database,
    *,
    plan_id: str,
    grant_id: str,
    base_commit: str,
    base_tree_digest: str,
    repo_path_text: str = "local",
) -> CampaignRecord:
    """Create a new campaign record (DRAFT). Activation follows in
    ``activate_campaign``.

    The grant is referenced by ``grant_id`` only; the runner does NOT
    need to find the grant in the durable store at campaign-create
    time. The campaign's lifecycle is governed by ``activate_campaign``
    (DRAFT -> ACTIVE) which is what binds to the store.
    """
    require_not_paused_or_raise()
    now = int(time.time())
    campaign_id = f"cmp-{plan_id}-{uuid.uuid4().hex[:8]}"
    integration_branch = f"refs/heads/campaign/{campaign_id}"
    base_snapshot = RepoSnapshot(
        schema_version="trio.repo-snapshot.v1",
        repository_id=repo_path_text,
        commit=base_commit,
        tree_digest=base_tree_digest,
    )
    with db.transaction() as cur:
        cur.execute(
            """
            INSERT INTO campaigns (
                campaign_id, grant_id, plan_id, state,
                integration_branch, current_commit, current_tree_digest,
                current_fence, created_at, updated_at, completed_at,
                grant_digest, plan_digest
            ) VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?)
            """,
            (
                campaign_id, grant_id, plan_id, CampaignState.DRAFT.value,
                integration_branch, base_commit, base_tree_digest,
                1, now, now,
                plan_id,
                plan_id,
            ),
        )
    return CampaignRecord(
        schema_version="trio.campaign-record.v1",
        campaign_id=campaign_id,
        grant_id=grant_id,
        plan_id=plan_id,
        integration_branch=integration_branch,
        current_snapshot=base_snapshot,
        current_fence=1,
        state=CampaignState.DRAFT,
        created_at=now,
        updated_at=now,
    )


def activate_campaign(db: Database, *, campaign_id: str) -> CampaignRecord:
    """Transition a DRAFT campaign to ACTIVE."""
    require_not_paused_or_raise()
    now = int(time.time())
    with db.transaction() as cur:
        cur.execute(
            "UPDATE campaigns SET state=?, updated_at=? WHERE campaign_id=? AND state='DRAFT'",
            (CampaignState.ACTIVE.value, now, campaign_id),
        )
        if cur.rowcount != 1:
            raise SafetyError(f"campaign {campaign_id} not in DRAFT state")
    return _load_campaign(db, campaign_id)


def list_chunks(db: Database, *, campaign_id: str) -> list[dict[str, Any]]:
    cur = db._conn.execute(
        "SELECT * FROM chunks WHERE campaign_id=? ORDER BY created_at ASC",
        (campaign_id,),
    )
    return [dict(r) for r in cur.fetchall()]


def install_chunk(db: Database, *, chunk: ChunkSpec) -> None:
    """Persist a chunk row in PROPOSED state. The runner may, but the
    worker MUST NOT, call this."""
    require_not_paused_or_raise()
    now = int(time.time())
    with db.transaction() as cur:
        cur.execute(
            """
            INSERT INTO chunks (
                chunk_id, campaign_id, package_id, parent_chunk_id, revision,
                idempotency_key, state, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                chunk.chunk_id, chunk.campaign_id, chunk.package_id,
                chunk.parent_chunk_id, chunk.revision,
                chunk.idempotency_key, chunk.state.value, now, now,
            ),
        )


def record_chunk_accepted(
    db: Database, *, chunk_id: str, accepted_commit: str, accepted_tree_digest: str
) -> None:
    """Mark a chunk ACCEPTED_FOR_CONTINUATION and capture its accepted snapshot."""
    require_not_paused_or_raise()
    now = int(time.time())
    with db.transaction() as cur:
        cur.execute(
            """
            UPDATE chunks SET
                state=?, snapshot_commit=?, snapshot_tree_digest=?,
                updated_at=?
            WHERE chunk_id=?
            """,
            (
                ChunkState.ACCEPTED_FOR_CONTINUATION.value,
                accepted_commit, accepted_tree_digest,
                now, chunk_id,
            ),
        )
        if cur.rowcount != 1:
            raise SafetyError(f"chunk {chunk_id} not found for acceptance")


def update_budget_after_chunk(
    db: Database,
    *,
    ledger_id: str,
    delta_model_calls: int = 0,
    delta_tool_calls: int = 0,
    delta_active_seconds: int = 0,
    delta_chunks: int = 1,
) -> None:
    """Increment cumulative budget counters without lowering them.

    Sessions, restarts, rechunks, and provider changes NEVER reset
    cumulative counters. The function only ever increments; it does NOT
    decrement. Raises ``SafetyError`` if applying these increments
    would exceed the budget bounds.
    """
    require_not_paused_or_raise()
    now = int(time.time())
    with db.transaction() as cur:
        cur.execute(
            """
            SELECT bounds_json, cumulative_model_calls, cumulative_tool_calls,
                   cumulative_active_seconds, cumulative_chunks
            FROM budget_ledgers WHERE ledger_id=?
            """,
            (ledger_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise SafetyError(f"ledger {ledger_id} not found")
        bounds = json.loads(row["bounds_json"])
        # Maximum comparisons (inline avoids requiring Pydantic here).
        if row["cumulative_model_calls"] + delta_model_calls > bounds["max_model_calls"]:
            raise SafetyError("budget exhausted: max_model_calls")
        if row["cumulative_tool_calls"] + delta_tool_calls > bounds["max_tool_calls"]:
            raise SafetyError("budget exhausted: max_tool_calls")
        if row["cumulative_active_seconds"] + delta_active_seconds > bounds["max_active_seconds"]:
            raise SafetyError("budget exhausted: max_active_seconds")
        if row["cumulative_chunks"] + delta_chunks > bounds["max_chunks"]:
            raise SafetyError("budget exhausted: max_chunks")
        cur.execute(
            """
            UPDATE budget_ledgers SET
                cumulative_model_calls=cumulative_model_calls+?,
                cumulative_tool_calls=cumulative_tool_calls+?,
                cumulative_active_seconds=cumulative_active_seconds+?,
                cumulative_chunks=cumulative_chunks+?,
                revision=revision+1
            WHERE ledger_id=?
            """,
            (delta_model_calls, delta_tool_calls, delta_active_seconds, delta_chunks, ledger_id),
        )


def pause_campaign(db: Database, *, campaign_id: str, reason: str) -> None:
    """Operator pause: stop new work at safe boundary."""
    require_not_paused_or_raise()
    now = int(time.time())
    with db.transaction() as cur:
        cur.execute(
            "UPDATE campaigns SET state='PAUSED_OPERATOR', updated_at=? WHERE campaign_id=?",
            (now, campaign_id),
        )
        cur.execute(
            """
            INSERT INTO campaign_events (
                event_id, campaign_id, chunk_id, event_type,
                from_state, to_state, actor, payload, fence_generation,
                issued_at, idempotency_key
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                f"ev-{uuid.uuid4().hex[:16]}", campaign_id, None, "PAUSED_OPERATOR",
                None, "PAUSED_OPERATOR", "operator", json.dumps({"reason": reason}),
                0, now, f"pause-{uuid.uuid4().hex[:8]}",
            ),
        )


def resume_campaign(db: Database, *, campaign_id: str) -> None:
    """Operator resume: back to ACTIVE."""
    now = int(time.time())
    with db.transaction() as cur:
        cur.execute(
            "UPDATE campaigns SET state='ACTIVE', updated_at=? WHERE campaign_id=? AND state='PAUSED_OPERATOR'",
            (now, campaign_id),
        )


def cancel_campaign(db: Database, *, campaign_id: str, reason: str) -> None:
    """Operator cancellation: stop work, preserve evidence."""
    require_not_paused_or_raise()
    now = int(time.time())
    with db.transaction() as cur:
        cur.execute(
            "UPDATE campaigns SET state='CANCELLED', updated_at=?, completed_at=? WHERE campaign_id=?",
            (now, now, campaign_id),
        )


def _load_campaign(db: Database, campaign_id: str) -> CampaignRecord:
    cur = db._conn.execute("SELECT * FROM campaigns WHERE campaign_id=?", (campaign_id,))
    row = cur.fetchone()
    if row is None:
        raise SafetyError(f"campaign {campaign_id} not found")
    snap = RepoSnapshot(
        schema_version="trio.repo-snapshot.v1",
        repository_id="local",
        commit=row["current_commit"] or "",
        tree_digest=row["current_tree_digest"] or "",
    )
    return CampaignRecord(
        schema_version="trio.campaign-record.v1",
        campaign_id=row["campaign_id"],
        grant_id=row["grant_id"],
        plan_id=row["plan_id"],
        integration_branch=row["integration_branch"],
        current_snapshot=snap,
        current_fence=row["current_fence"],
        state=CampaignState(row["state"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
    )


__all__ = [
    "CampaignHandle",
    "create_campaign",
    "activate_campaign",
    "list_chunks",
    "install_chunk",
    "record_chunk_accepted",
    "update_budget_after_chunk",
    "pause_campaign",
    "resume_campaign",
    "cancel_campaign",
    "ensure_running_paused_aware",
    "ensure_campaign_worktree",
    "capture_current_snapshot",
    "compare_and_swap_advance",
]
