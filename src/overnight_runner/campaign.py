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

import hashlib
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
    content_sha256,
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
    repo_root: str | None = None,
    worktree_path: str | None = None,
) -> CampaignRecord:
    """Create a new campaign record (DRAFT). Activation follows in
    ``activate_campaign``.

    P06 follow-up #2 (A01): ``campaigns.grant_digest`` MUST contain the
    actual grant digest (NOT a plan_id string), and
    ``campaigns.plan_digest`` MUST contain the actual approved plan
    digest (NOT a plan_id string). We resolve both by reading the
    durable stores at insert time; unknown grant/plan refuses the
    insert and raises ``SafetyError``.

    P06 follow-up #3 (A01 item 1 / A14 item 14): the campaign MUST be
    created with the grant's OWN plan. We require, before the insert:

      * the grant exists and its state is ``active``;
      * ``plan_id == stored_grant.plan_id``;
      * ``load_plan_digest(plan_id) == stored_grant.approved_plan_digest``.

    ``repo_root`` / ``worktree_path`` persist the durable repository
    identity so a later crash-window reconciliation inspects the real
    campaign repo (A05 item 7).
    """
    from .feature_gate import require_campaign_v2
    from .grants import load_grant
    from .plans import load_plan_digest
    require_not_paused_or_raise()
    require_campaign_v2("create_campaign")
    # Resolve grant_digest and plan_digest from the durable stores.
    # The campaign MUST NOT be created with placeholder plan_id strings
    # in those columns.
    stored_grant = load_grant(db, grant_id)
    if stored_grant is None:
        raise SafetyError(f"grant {grant_id!r} not registered in durable store")
    if stored_grant.state != "active":
        raise SafetyError(
            f"grant {grant_id!r} is not active (grant state={stored_grant.state}); "
            f"refusing to create a campaign"
        )
    # A01/A14: the campaign plan MUST be the grant-pinned plan.
    if plan_id != stored_grant.plan_id:
        raise SafetyError(
            f"campaign plan/grant mismatch: supplied plan_id={plan_id!r} != "
            f"grant-pinned plan_id={stored_grant.plan_id!r}"
        )
    stored_plan_digest = load_plan_digest(db, plan_id)
    if stored_plan_digest is None:
        raise SafetyError(f"plan {plan_id!r} not registered in durable store")
    if stored_plan_digest != stored_grant.approved_plan_digest:
        raise SafetyError(
            f"campaign plan digest mismatch: registered plan "
            f"{stored_plan_digest[:8]} != grant pinned "
            f"{stored_grant.approved_plan_digest[:8]}"
        )
    grant_digest = content_sha256(stored_grant)
    # The grant-pinned approved plan digest is the authoritative content
    # binding (NOT a recomputed value that could drift from the pin).
    plan_digest = stored_grant.approved_plan_digest
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
                grant_digest, plan_digest, repo_root, worktree_path
            ) VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?,?,?)
            """,
            (
                campaign_id, grant_id, plan_id, CampaignState.DRAFT.value,
                integration_branch, base_commit, base_tree_digest,
                1, now, now,
                grant_digest,
                plan_digest,
                repo_root or "",
                worktree_path or "",
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
    from .feature_gate import require_campaign_v2
    require_not_paused_or_raise()
    require_campaign_v2("activate_campaign")
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
    delta_repairs: int = 0,
    delta_rechunks: int = 0,
    delta_escalations: int = 0,
    delta_active_seconds: int = 0,
    delta_wall_seconds: int = 0,
    delta_cost_microusd: int = 0,
    delta_chunks: int = 0,
    delta_context_tokens: int = 0,
    now: int | None = None,
) -> dict[str, int]:
    """ONE trusted cumulative consume/update operation (P06 follow-up #2 A07).

    Covers every dimension defined in ``Budget`` and
    ``BudgetLedgerEntry``:

      * model calls
      * tool calls
      * repairs
      * rechunks/revisions
      * escalations
      * active seconds
      * cumulative wall/elapsed seconds
      * cost_microusd
      * chunks
      * context tokens

    The function only ever increments; sessions, restarts, rechunks,
    and provider changes NEVER reset cumulative counters. Raises
    ``SafetyError`` if applying these increments would exceed the
    budget bounds. The caller MUST use this trusted budget API;
    direct SQL UPDATE is reserved for migration/cleanup.

    Returns the new cumulative totals so callers can persist them
    into evidence.
    """
    require_not_paused_or_raise()
    now = int(now if now is not None else time.time())
    with db.transaction() as cur:
        cur.execute(
            """
            SELECT bounds_json,
                   cumulative_model_calls, cumulative_tool_calls,
                   cumulative_repairs, cumulative_rechunks,
                   cumulative_escalations,
                   cumulative_active_seconds, cumulative_wall_seconds,
                   cumulative_cost_microusd,
                   cumulative_chunks, cumulative_context_tokens
            FROM budget_ledgers WHERE ledger_id=?
            """,
            (ledger_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise SafetyError(f"ledger {ledger_id} not found")
        bounds = json.loads(row["bounds_json"])
        # Maximum comparisons.
        if row["cumulative_model_calls"] + delta_model_calls > bounds["max_model_calls"]:
            raise SafetyError(
                f"budget exhausted: max_model_calls "
                f"({row['cumulative_model_calls']+delta_model_calls}>{bounds['max_model_calls']})"
            )
        if row["cumulative_tool_calls"] + delta_tool_calls > bounds["max_tool_calls"]:
            raise SafetyError(
                f"budget exhausted: max_tool_calls "
                f"({row['cumulative_tool_calls']+delta_tool_calls}>{bounds['max_tool_calls']})"
            )
        if row["cumulative_repairs"] + delta_repairs > bounds["max_local_repairs"]:
            raise SafetyError(
                f"budget exhausted: max_local_repairs "
                f"({row['cumulative_repairs']+delta_repairs}>{bounds['max_local_repairs']})"
            )
        if row["cumulative_rechunks"] + delta_rechunks > bounds["max_rechunks"]:
            raise SafetyError(
                f"budget exhausted: max_rechunks "
                f"({row['cumulative_rechunks']+delta_rechunks}>{bounds['max_rechunks']})"
            )
        if row["cumulative_escalations"] + delta_escalations > bounds["max_frontier_escalations"]:
            raise SafetyError(
                f"budget exhausted: max_frontier_escalations "
                f"({row['cumulative_escalations']+delta_escalations}>{bounds['max_frontier_escalations']})"
            )
        if row["cumulative_active_seconds"] + delta_active_seconds > bounds["max_active_seconds"]:
            raise SafetyError(
                f"budget exhausted: max_active_seconds "
                f"({row['cumulative_active_seconds']+delta_active_seconds}>{bounds['max_active_seconds']})"
            )
        # Wall seconds: cumulative (no upper delta increment; we
        # always pass a delta and bound against max_wall_seconds).
        if delta_wall_seconds < 0:
            raise SafetyError("delta_wall_seconds must be >= 0")
        cur_wall = int(row["cumulative_wall_seconds"]) if "cumulative_wall_seconds" in row.keys() else 0
        if cur_wall + delta_wall_seconds > bounds["max_wall_seconds"]:
            raise SafetyError(
                f"budget exhausted: max_wall_seconds "
                f"({cur_wall+delta_wall_seconds}>{bounds['max_wall_seconds']})"
            )
        if row["cumulative_cost_microusd"] + delta_cost_microusd > bounds["max_cost_microusd"]:
            raise SafetyError(
                f"budget exhausted: max_cost_microusd "
                f"({row['cumulative_cost_microusd']+delta_cost_microusd}>{bounds['max_cost_microusd']})"
            )
        if row["cumulative_chunks"] + delta_chunks > bounds["max_chunks"]:
            raise SafetyError(
                f"budget exhausted: max_chunks "
                f"({row['cumulative_chunks']+delta_chunks}>{bounds['max_chunks']})"
            )
        if row["cumulative_context_tokens"] + delta_context_tokens > bounds["context_token_budget"]:
            raise SafetyError(
                f"budget exhausted: context_token_budget "
                f"({row['cumulative_context_tokens']+delta_context_tokens}>{bounds['context_token_budget']})"
            )
        cur.execute(
            """
            UPDATE budget_ledgers SET
                cumulative_model_calls=cumulative_model_calls+?,
                cumulative_tool_calls=cumulative_tool_calls+?,
                cumulative_repairs=cumulative_repairs+?,
                cumulative_rechunks=cumulative_rechunks+?,
                cumulative_escalations=cumulative_escalations+?,
                cumulative_active_seconds=cumulative_active_seconds+?,
                cumulative_wall_seconds=cumulative_wall_seconds+?,
                cumulative_cost_microusd=cumulative_cost_microusd+?,
                cumulative_chunks=cumulative_chunks+?,
                cumulative_context_tokens=cumulative_context_tokens+?,
                revision=revision+1
            WHERE ledger_id=?
            """,
            (
                delta_model_calls, delta_tool_calls,
                delta_repairs, delta_rechunks, delta_escalations,
                delta_active_seconds, delta_wall_seconds,
                delta_cost_microusd,
                delta_chunks, delta_context_tokens,
                ledger_id,
            ),
        )
        # Read back the new totals.
        cur.execute(
            """
            SELECT cumulative_model_calls, cumulative_tool_calls,
                   cumulative_repairs, cumulative_rechunks,
                   cumulative_escalations,
                   cumulative_active_seconds, cumulative_wall_seconds,
                   cumulative_cost_microusd,
                   cumulative_chunks, cumulative_context_tokens
            FROM budget_ledgers WHERE ledger_id=?
            """,
            (ledger_id,),
        )
        new = dict(cur.fetchone())
    return new


def read_budget_totals(db: Database, *, ledger_id: str) -> dict[str, int]:
    """Read the current cumulative totals from the durable ledger.

    Companion to ``update_budget_after_chunk``. The trusted budget
    API is the only sanctioned read surface for cumulative budget.
    """
    cur = db._conn.execute(
        """
        SELECT cumulative_model_calls, cumulative_tool_calls,
               cumulative_repairs, cumulative_rechunks,
               cumulative_escalations,
               cumulative_active_seconds, cumulative_wall_seconds,
               cumulative_cost_microusd,
               cumulative_chunks, cumulative_context_tokens
        FROM budget_ledgers WHERE ledger_id=?
        """,
        (ledger_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise SafetyError(f"ledger {ledger_id} not found")
    return dict(row)


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


# ---------------------------------------------------------------------------
# Human gate (C09 -> G01): the canonical Runner-owned transition
# ---------------------------------------------------------------------------
HUMAN_GATE_STATE = "AWAITING_HUMAN_AT_CLOSE"
#: Lifecycle states from which the human gate may be entered.
HUMAN_GATE_ELIGIBLE_STATES = ("ACTIVE",)


def record_human_gate(
    db: Database, *, campaign_id: str, gate_id: str, chunk_id: str = "",
    reason: str = "", now: int | None = None,
) -> dict[str, Any]:
    """Park an ACCEPTED campaign at the human acceptance gate.

    This is the canonical Runner-owned replacement for ad-hoc SQL that wrote
    ``campaigns.state = 'AWAITING_HUMAN_AT_CLOSE'`` from outside the Runner.

    Semantics:
      * the campaign must exist;
      * only ``HUMAN_GATE_ELIGIBLE_STATES`` may enter the gate;
      * the SAME gate request (same gate_id + chunk_id) replays idempotently;
      * a DIFFERENT gate/chunk while already parked is contradictory and
        rejected;
      * a correctly shaped ``campaign_events`` row is emitted;
      * budgets, acceptance history and C10 are untouched — the gate is never
        auto-approved.
    """
    now = int(now if now is not None else time.time())
    if not campaign_id:
        raise SafetyError("record_human_gate: campaign_id is required")
    if not gate_id:
        raise SafetyError("record_human_gate: gate_id is required")
    require_not_paused_or_raise()

    row = db._conn.execute(
        "SELECT state, completed_at FROM campaigns WHERE campaign_id=?",
        (campaign_id,),
    ).fetchone()
    if row is None:
        raise SafetyError(f"record_human_gate: campaign {campaign_id!r} not found")
    state = row["state"]

    with db.transaction() as cur:
        # Idempotency: the same gate request is a replay, not a new transition.
        existing = cur.execute(
            "SELECT event_id, chunk_id, payload FROM campaign_events "
            "WHERE campaign_id=? AND event_type=? ORDER BY issued_at DESC LIMIT 1",
            (campaign_id, HUMAN_GATE_STATE),
        ).fetchone()
        if state == HUMAN_GATE_STATE:
            if existing is None:
                raise SafetyError(
                    f"campaign {campaign_id!r} is {HUMAN_GATE_STATE} without a "
                    f"durable gate event; refusing to guess"
                )
            prior_chunk = existing["chunk_id"] or ""
            if prior_chunk != chunk_id:
                raise SafetyError(
                    f"record_human_gate: campaign {campaign_id!r} is already at the "
                    f"gate for chunk {prior_chunk!r}; refusing contradictory gate "
                    f"for chunk {chunk_id!r}"
                )
            return {"schema_version": "trio.human-gate.v1", "campaign_id": campaign_id,
                    "gate_id": gate_id, "chunk_id": chunk_id,
                    "state": HUMAN_GATE_STATE, "event_id": existing["event_id"],
                    "idempotent_replay": True}

        if state not in HUMAN_GATE_ELIGIBLE_STATES:
            raise SafetyError(
                f"record_human_gate: campaign {campaign_id!r} is in state {state!r}; "
                f"only {HUMAN_GATE_ELIGIBLE_STATES} may enter the human gate"
            )

        event_id = f"ev-{uuid.uuid4().hex[:16]}"
        cur.execute(
            """
            INSERT INTO campaign_events (
                event_id, campaign_id, chunk_id, event_type,
                from_state, to_state, actor, payload, fence_generation,
                issued_at, idempotency_key
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                event_id, campaign_id, chunk_id or None, HUMAN_GATE_STATE,
                state, HUMAN_GATE_STATE, "runner",
                json.dumps({"gate_id": gate_id, "reason": reason[:512]}),
                int(cur.execute("SELECT current_fence FROM campaigns WHERE campaign_id=?",
                                (campaign_id,)).fetchone()["current_fence"]),
                now, f"gate-{campaign_id}-{gate_id}-{chunk_id}",
            ),
        )
        cur.execute(
            "UPDATE campaigns SET state=?, updated_at=? WHERE campaign_id=? AND state=?",
            (HUMAN_GATE_STATE, now, campaign_id, state),
        )
        if cur.rowcount != 1:
            raise SafetyError(
                f"record_human_gate: campaign {campaign_id!r} state changed concurrently"
            )
    return {"schema_version": "trio.human-gate.v1", "campaign_id": campaign_id,
            "gate_id": gate_id, "chunk_id": chunk_id, "state": HUMAN_GATE_STATE,
            "event_id": event_id, "from_state": state, "issued_at": now,
            "idempotent_replay": False}


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
    "HUMAN_GATE_STATE",
    "HUMAN_GATE_ELIGIBLE_STATES",
    "record_human_gate",
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
