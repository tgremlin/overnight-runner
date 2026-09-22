"""P08 — autonomous progress + bounded failure handling (runner-owned glue).

P08 is an integration/verification phase. It adds only the runner-owned
glue needed on top of the accepted P00–P07 system:

  * a DURABLE context handoff bound to the same campaign/chunk/job/grant/
    budget identities, exposing only a bounded, permitted context slice;
  * an explicit PHASE-COMPLETION evaluation that is NOT satisfied by an
    empty runnable queue (required integration + explicit human gate).

No new execution authority is created here; the accepted runner remains
authoritative.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterable

from .db import Database
from .safety import SafetyError

PHASE_COMPLETION_VERSION = "trio.phase-completion.v1"
HANDOFF_VERSION = "trio.context-handoff.v1"

# The ONLY fields a continuation may receive across a handoff boundary.
BOUNDED_CONTEXT_FIELDS = (
    "campaign_id", "chunk_id", "job_id", "grant_id",
    "budget_ledger_id", "snapshot_commit",
)


def record_handoff(
    db: Database, *, campaign_id: str, chunk_id: str, job_id: str,
    grant_id: str, budget_ledger_id: str, snapshot_commit: str,
    reason: str = "", now: int | None = None,
) -> str:
    """Persist a durable context handoff (session/context boundary)."""
    now = int(now if now is not None else time.time())
    handoff_id = f"hoff-{uuid.uuid4().hex[:16]}"
    with db.transaction() as cur:
        cur.execute(
            """
            INSERT INTO context_handoffs (
                handoff_id, campaign_id, chunk_id, job_id, grant_id,
                budget_ledger_id, snapshot_commit, reason, created_at, state
            ) VALUES (?,?,?,?,?,?,?,?,?, 'OPEN')
            """,
            (handoff_id, campaign_id, chunk_id, job_id, grant_id,
             budget_ledger_id, snapshot_commit, reason[:512], now),
        )
    return handoff_id


def load_handoff(db: Database, handoff_id: str) -> dict[str, Any] | None:
    row = db._conn.execute("SELECT * FROM context_handoffs WHERE handoff_id=?",
                           (handoff_id,)).fetchone()
    return dict(row) if row is not None else None


def bound_context(handoff: dict[str, Any]) -> dict[str, Any]:
    """Return ONLY the permitted bounded context for a continuation."""
    return {k: handoff.get(k, "") for k in BOUNDED_CONTEXT_FIELDS}


def close_handoff(db: Database, handoff_id: str, *, state: str = "CONSUMED",
                  now: int | None = None) -> None:
    now = int(now if now is not None else time.time())
    with db.transaction() as cur:
        cur.execute("UPDATE context_handoffs SET state=?, created_at=? "
                    "WHERE handoff_id=?", (state, now, handoff_id))


def evaluate_phase_completion(
    db: Database, *, campaign_id: str, required_chunks: Iterable[str],
    human_gate_required: bool = False, human_gate_satisfied: bool = False,
) -> dict[str, Any]:
    """Decide phase completion from AUTHORITATIVE runner evidence.

    An empty runnable queue is NEVER sufficient. Completion requires:
      * every REQUIRED chunk integrated (integration_journal) AND accepted
        (chunk state ACCEPTED_FOR_CONTINUATION);
      * if a human acceptance gate is required, it must be satisfied.

    Returns a dict with ``complete``/``state``/``reason``.
    """
    required = list(dict.fromkeys(required_chunks))
    journal = {r["chunk_id"] for r in db._conn.execute(
        "SELECT chunk_id FROM integration_journal WHERE campaign_id=?",
        (campaign_id,)).fetchall()}
    accepted = {r["chunk_id"] for r in db._conn.execute(
        "SELECT chunk_id FROM chunks WHERE campaign_id=? "
        "AND state='ACCEPTED_FOR_CONTINUATION'", (campaign_id,)).fetchall()}
    missing = [c for c in required
               if c not in accepted or c not in journal]
    if missing:
        return {
            "schema_version": PHASE_COMPLETION_VERSION,
            "campaign_id": campaign_id,
            "complete": False,
            "state": "INCOMPLETE",
            "reason": "required_integration_criteria_missing",
            "missing_criteria": missing,
        }
    if human_gate_required and not human_gate_satisfied:
        return {
            "schema_version": PHASE_COMPLETION_VERSION,
            "campaign_id": campaign_id,
            "complete": False,
            "state": "AWAITING_HUMAN",
            "reason": "human_acceptance_gate_outstanding",
            "missing_criteria": [],
        }
    return {
        "schema_version": PHASE_COMPLETION_VERSION,
        "campaign_id": campaign_id,
        "complete": True,
        "state": "ELIGIBLE_FOR_PHASE_COMPLETION",
        "reason": "all_technical_criteria_met_and_gates_satisfied",
        "missing_criteria": [],
    }


__all__ = [
    "PHASE_COMPLETION_VERSION", "HANDOFF_VERSION", "BOUNDED_CONTEXT_FIELDS",
    "record_handoff", "load_handoff", "bound_context", "close_handoff",
    "evaluate_phase_completion",
]
