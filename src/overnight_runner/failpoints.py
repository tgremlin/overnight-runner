"""P06-A05 / follow-up #3 — deterministic crash failpoints.

Each boundary listed in ``FAILPOINTS`` is an operation boundary where a
process crash can leave the world in an ambiguous state. A test arms a
failpoint by setting ``TR_FAILPOINT_<NAME>=raise``; the next real
operation that reaches the boundary then (a) records enough DURABLE
intent that a restart can discover the uncertainty, and (b) raises so
the simulated crash terminates the operation.

P06 follow-up #3 (A05 item 6): a failpoint MUST NOT merely raise. For
dangerous crash points the durable intent is written BEFORE (or at) the
consequential operation, so recovery never depends on code that runs
after the simulated crash.

This module holds only the naming + arming primitives; the durable
intent record (a ``crash_windows`` row + ``EFFECT_UNKNOWN`` state) is
written by ``integration.record_crash_intent`` which owns the DB.
"""
from __future__ import annotations

import os

FAILPOINTS: dict[str, str] = {
    "candidate_changed_before_state_durable": (
        "candidate/worktree was changed before durable state commit"
    ),
    "validator_executed_before_receipt_state_durable": (
        "validator executed before receipt/state durable commit"
    ),
    "commit_exists_before_db_candidate_state": (
        "commit exists before DB candidate state"
    ),
    "db_integration_intent_before_ref_advance": (
        "DB integration intent exists before ref advance"
    ),
    "ref_advanced_before_integration_journal_event": (
        "ref advanced before integration journal/event"
    ),
    "event_outbox_committed_before_projection_status": (
        "event/outbox committed before projection/status"
    ),
}

# Failpoints that are internal to single-transaction boundaries (not one
# of the named six). They exist to prove atomicity/crash-safety; they do
# not require durable intent because the whole transaction rolls back.
INTERNAL_FAILPOINTS: dict[str, str] = {
    "reserve_before_admission_durable": (
        "idempotency reservation exists but admission row not yet durable"
    ),
    "crash_window_before_commit": (
        "crash window INSERT staged but EFFECT_UNKNOWN transition not yet committed"
    ),
    "activate_grant_before_commit": (
        "protected approval consumed but active grant not yet committed"
    ),
}


def failpoint_armed(name: str) -> bool:
    """True iff ``TR_FAILPOINT_<NAME>`` is set to ``raise``."""
    return os.environ.get(f"TR_FAILPOINT_{name}", "").lower() == "raise"


def raise_failpoint(name: str) -> None:
    """Raise the deterministic injected-crash error for ``name``."""
    desc = FAILPOINTS.get(name) or INTERNAL_FAILPOINTS.get(name, "?")
    raise RuntimeError(f"injected failpoint at boundary={name}: {desc}")


def trigger_crash_failpoint(name: str) -> None:
    """Raise iff ``name`` is armed.

    Kept for callers that only need the raise (no durable intent). The
    dangerous boundaries use ``integration``'s intent-recording helper
    instead, which arms + records durable intent + raises.
    """
    if failpoint_armed(name):
        raise_failpoint(name)


__all__ = [
    "FAILPOINTS",
    "INTERNAL_FAILPOINTS",
    "failpoint_armed",
    "raise_failpoint",
    "trigger_crash_failpoint",
]
