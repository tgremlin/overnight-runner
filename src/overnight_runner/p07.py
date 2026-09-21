"""P07 — Hermes foreman integration (runner-authoritative surfaces).

Hermes is a thin projection + conversation + bounded wake/control
surface. It is NOT a second lifecycle. The runner remains the sole
authority for campaigns, grants, admission, budgets, leases/fences,
worker lifecycle, mutations, validation receipts, and recovery.

This module exposes the narrow, schema-defined surfaces a thin Hermes
skill may call. None of them mint authority:

  * ``classify_capacity`` normalizes provider/capacity outcomes from
    trusted structured fields (HTTP status, provider fields, exit
    contracts) — never from LLM prose.
  * ``upsert_cooldown`` / ``cooldown_state`` persist provider/account
    cooldown in runner-owned state (scoped so one model/account does not
    poison unrelated work).
  * ``enter_capacity_wait`` / ``resolve_capacity_wait`` persist durable
    WAIT_CAPACITY (never consuming a repair merely for capacity).
  * ``resolve_fallback`` restricts fallback to a finite approved subset.
  * ``wake_tick`` re-evaluates authoritative durable state and performs
    at most the permitted next advancement; repeated wakes are safe.
  * ``request_control`` returns a durable job id (short control), while
    long execution stays runner-owned.
  * ``project_campaign_status`` projects runner truth read-only.

All functions take an already-opened :class:`~overnight_runner.db.Database`.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .db import Database
from .safety import SafetyError

CAPACITY_OUTCOME_VERSION = "trio.capacity-outcome.v1"
CONTROL_SCHEMA_VERSION = "trio.hermes-control.v1"
STATUS_SCHEMA_VERSION = "trio.runner-status.v1"


# ===========================================================================
# A02 — Normalized provider capacity outcomes
# ===========================================================================

class CapacityKind(str, Enum):
    OK = "ok"
    RATE_LIMITED = "rate_limited"
    OVERLOADED = "overloaded"
    QUOTA_EXHAUSTED_KNOWN_RESET = "quota_exhausted_known_reset"
    QUOTA_EXHAUSTED_NO_RESET = "quota_exhausted_no_reset"
    AUTH_FAILURE = "auth_failure"
    INVALID_PROVIDER_MODEL_PROFILE = "invalid_provider_model_profile"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    UNKNOWN_TRANSPORT = "unknown_transport"


# Kinds that warrant a runner-owned cooldown / capacity wait.
COOLING_KINDS = frozenset({
    CapacityKind.RATE_LIMITED,
    CapacityKind.OVERLOADED,
    CapacityKind.QUOTA_EXHAUSTED_KNOWN_RESET,
    CapacityKind.QUOTA_EXHAUSTED_NO_RESET,
    CapacityKind.PROVIDER_UNAVAILABLE,
})

# Kinds that must NOT retry without operator/authority change.
TERMINAL_AUTHORITY_KINDS = frozenset({
    CapacityKind.AUTH_FAILURE,
    CapacityKind.INVALID_PROVIDER_MODEL_PROFILE,
})

_QUOTA_MARKERS = (
    "quota", "insufficient_quota", "usage_limit", "usage limit",
    "token plan", "credits", "billing_hard_limit",
)
_QUOTA_RESET_MARKERS = (
    "reset", "retry_after", "resets_at", "reset_at", "window",
)


@dataclass(frozen=True)
class CapacityOutcome:
    schema_version: str
    kind: str
    provider: str
    model: str
    http_status: int | None
    retry_after_seconds: int | None
    reset_at: int | None
    source: str
    raw_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_capacity(
    *,
    provider: str = "",
    model: str = "",
    http_status: int | None = None,
    error_code: str = "",
    message: str = "",
    retry_after_seconds: int | None = None,
    reset_at: int | None = None,
    transport_error: str = "",
) -> CapacityOutcome:
    """Deterministically normalize a provider/capacity outcome.

    Uses ONLY trusted structured inputs (HTTP/status codes, provider
    error codes, explicit reset/retry metadata, transport error class).
    It never interprets model prose.
    """
    code = (error_code or "").strip().lower()
    msg = (message or "").lower()
    kind: CapacityKind

    def _outcome(k: CapacityKind, reason: str, source: str) -> CapacityOutcome:
        return CapacityOutcome(
            schema_version=CAPACITY_OUTCOME_VERSION,
            kind=k.value,
            provider=provider, model=model,
            http_status=http_status,
            retry_after_seconds=retry_after_seconds,
            reset_at=reset_at,
            source=source,
            raw_reason=reason[:512],
        )

    # Quota exhausted (explicit structured code / documented marker).
    if code in ("quota_exhausted", "insufficient_quota", "usage_limit_reached",
                "billing_hard_limit_reached") or any(m in msg for m in _QUOTA_MARKERS):
        known = bool(reset_at) or bool(retry_after_seconds) or code.endswith("_reset")
        if known or any(m in msg for m in _QUOTA_RESET_MARKERS):
            return _outcome(CapacityKind.QUOTA_EXHAUSTED_KNOWN_RESET,
                            code or "quota reset known", "quota_code")
        return _outcome(CapacityKind.QUOTA_EXHAUSTED_NO_RESET,
                        code or "quota reset unknown", "quota_code")

    # Auth / authorization.
    if code in ("invalid_api_key", "authentication_error", "unauthorized",
                "permission_denied", "insufficient_scope"):
        return _outcome(CapacityKind.AUTH_FAILURE, code, "error_code")
    if http_status in (401, 403):
        return _outcome(CapacityKind.AUTH_FAILURE, f"http {http_status}", "http_status")

    # Invalid provider/model/profile.
    if code in ("model_not_found", "invalid_model", "invalid_profile",
                "unknown_model", "invalid_request_error"):
        return _outcome(CapacityKind.INVALID_PROVIDER_MODEL_PROFILE, code, "error_code")
    if http_status in (400, 404, 422):
        return _outcome(CapacityKind.INVALID_PROVIDER_MODEL_PROFILE,
                        f"http {http_status}", "http_status")

    # Explicit rate limit.
    if http_status == 429 or code in ("rate_limit_exceeded", "too_many_requests"):
        return _outcome(CapacityKind.RATE_LIMITED, code or "http 429", "http_status")

    # Overload / temporary unavailability.
    if http_status in (503, 529) or code in ("overloaded", "server_overloaded",
                                             "temporarily_unavailable"):
        return _outcome(CapacityKind.OVERLOADED, code or f"http {http_status}",
                        "http_status")

    # Provider-side 5xx.
    if http_status in (500, 502, 504):
        return _outcome(CapacityKind.PROVIDER_UNAVAILABLE, f"http {http_status}",
                        "http_status")

    # Transport failures.
    if transport_error:
        te = transport_error.lower()
        if te in ("connectionerror", "connectionrefused", "timeout", "connecttimeout",
                  "readtimeout", "dns"):
            return _outcome(CapacityKind.PROVIDER_UNAVAILABLE, te, "transport")
        return _outcome(CapacityKind.UNKNOWN_TRANSPORT, te, "transport")

    return _outcome(CapacityKind.OK, "ok", "none")


# ===========================================================================
# A11 — Durable provider/account cooldown (runner-owned)
# ===========================================================================

def upsert_cooldown(
    db: Database,
    *,
    provider: str,
    kind: str,
    until_at: int,
    account: str = "",
    model: str = "",
    reason: str = "",
    now: int | None = None,
) -> None:
    """Persist a cooldown. Idempotent + restart-safe.

    Keyed on (provider, account, model) so cooling one model/account does
    not globally block unrelated providers.
    """
    now = int(now if now is not None else time.time())
    with db.transaction() as cur:
        cur.execute(
            """
            INSERT INTO provider_cooldowns
                (provider, account, model, kind, reason, until_at, updated_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(provider, account, model) DO UPDATE SET
                kind=excluded.kind,
                reason=excluded.reason,
                until_at=MAX(provider_cooldowns.until_at, excluded.until_at),
                updated_at=excluded.updated_at
            """,
            (provider, account, model, kind, reason[:512], int(until_at), now),
        )


def cooldown_state(
    db: Database, *, provider: str = "", account: str = "", model: str = "",
    now: int | None = None,
) -> dict[str, Any] | None:
    """Return the active cooldown row for the exact scope, else None."""
    now = int(now if now is not None else time.time())
    row = db._conn.execute(
        "SELECT * FROM provider_cooldowns WHERE provider=? AND account=? AND model=?",
        (provider, account, model),
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    if int(d["until_at"]) and int(d["until_at"]) <= now:
        return None
    d["cooling"] = True
    return d


def list_cooldowns(db: Database, *, now: int | None = None) -> list[dict[str, Any]]:
    now = int(now if now is not None else time.time())
    rows = db._conn.execute("SELECT * FROM provider_cooldowns").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["cooling"] = bool(int(d["until_at"]) == 0 or int(d["until_at"]) > now)
        out.append(d)
    return out


def apply_capacity_outcome(
    db: Database, outcome: CapacityOutcome, *, account: str = "",
    now: int | None = None,
) -> dict[str, Any] | None:
    """Persist a cooldown for a cooling outcome; return the cooldown row.

    Returns ``None`` for non-cooling outcomes (OK, auth, invalid).
    """
    if outcome.kind not in {k.value for k in COOLING_KINDS}:
        return None
    now = int(now if now is not None else time.time())
    until = 0
    if outcome.reset_at:
        until = int(outcome.reset_at)
    elif outcome.retry_after_seconds:
        until = now + int(outcome.retry_after_seconds)
    upsert_cooldown(db, provider=outcome.provider, account=account,
                    model=outcome.model, kind=outcome.kind, until_at=until,
                    reason=outcome.raw_reason, now=now)
    return cooldown_state(db, provider=outcome.provider, account=account,
                          model=outcome.model, now=now)


# ===========================================================================
# A03 — Durable capacity wait (WAIT_CAPACITY)
# ===========================================================================

def enter_capacity_wait(
    db: Database,
    *,
    campaign_id: str,
    reason_kind: str,
    grant_id: str = "",
    budget_ledger_id: str = "",
    chunk_id: str = "",
    obligation: str = "",
    provider: str = "",
    account: str = "",
    profile: str = "",
    next_eligible_at: int = 0,
    retry_provenance: str = "",
    now: int | None = None,
) -> str:
    """Persist a durable WAIT_CAPACITY obligation; return its wait id.

    Entering a capacity wait MUST NOT consume a repair attempt.
    """
    now = int(now if now is not None else time.time())
    wait_id = f"cwait-{uuid.uuid4().hex[:16]}"
    with db.transaction() as cur:
        cur.execute(
            """
            INSERT INTO capacity_waits (
                wait_id, campaign_id, chunk_id, obligation, provider, account,
                profile, reason_kind, entered_at, next_eligible_at,
                retry_provenance, grant_id, budget_ledger_id, wake_generation,
                active, resolved_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1,1,0)
            """,
            (wait_id, campaign_id, chunk_id, obligation, provider, account,
             profile, reason_kind, now, int(next_eligible_at),
             retry_provenance[:512], grant_id, budget_ledger_id),
        )
    return wait_id


def active_waits(db: Database, *, campaign_id: str = "") -> list[dict[str, Any]]:
    if campaign_id:
        rows = db._conn.execute(
            "SELECT * FROM capacity_waits WHERE active=1 AND campaign_id=?",
            (campaign_id,)).fetchall()
    else:
        rows = db._conn.execute("SELECT * FROM capacity_waits WHERE active=1").fetchall()
    return [dict(r) for r in rows]


def next_eligible_at(db: Database, *, campaign_id: str) -> int:
    """Earliest time the campaign may be reconsidered (0 = now)."""
    waits = active_waits(db, campaign_id=campaign_id)
    if not waits:
        return 0
    return max(int(w["next_eligible_at"]) for w in waits)


def resolve_capacity_wait(db: Database, *, wait_id: str, now: int | None = None) -> None:
    now = int(now if now is not None else time.time())
    with db.transaction() as cur:
        cur.execute(
            "UPDATE capacity_waits SET active=0, resolved_at=? WHERE wait_id=? AND active=1",
            (now, wait_id),
        )


# ===========================================================================
# A05 — Finite approved fallback routing
# ===========================================================================

@dataclass(frozen=True)
class FallbackProfile:
    profile_id: str
    role: str
    model_name: str
    provider: str
    runtime_digest: str
    egress_policy_id: str
    paid: bool = False
    cost_microusd: int = 0
    context_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FallbackPolicy:
    """Operator/architecture-approved fallback envelope."""
    role: str
    qualified_models: frozenset[str]
    approved_providers: frozenset[str]
    approved_profiles: frozenset[str]
    runtime_digest: str
    egress_policy_id: str
    allow_paid_spend: bool = False
    max_cost_microusd: int = 0
    max_context_tokens: int = 0


def resolve_fallback(
    primary: FallbackProfile,
    candidates: list[FallbackProfile],
    *,
    policy: FallbackPolicy,
    primary_unavailable: bool,
) -> FallbackProfile:
    """Return the first PERMITTED fallback or raise ``SafetyError``.

    Fallback is a deterministic subset of approved profiles. There is no
    "pick any available model": every dimension must be bound and
    approved, and paid spend is forbidden unless the policy explicitly
    permits it.
    """
    if not primary_unavailable:
        raise SafetyError("fallback_not_permitted: primary is available")

    for cand in candidates:
        if cand.role != policy.role:
            raise SafetyError(f"fallback_role_mismatch: {cand.profile_id}")
        if cand.profile_id not in policy.approved_profiles:
            raise SafetyError(f"fallback_profile_not_approved: {cand.profile_id}")
        if cand.model_name not in policy.qualified_models:
            raise SafetyError(f"fallback_model_not_qualified: {cand.model_name}")
        if cand.provider not in policy.approved_providers:
            raise SafetyError(f"fallback_provider_not_approved: {cand.provider}")
        if cand.runtime_digest != policy.runtime_digest:
            raise SafetyError(f"fallback_runtime_digest_mismatch: {cand.profile_id}")
        if cand.egress_policy_id != policy.egress_policy_id:
            raise SafetyError(f"fallback_egress_not_permitted: {cand.profile_id}")
        if cand.paid and not policy.allow_paid_spend:
            raise SafetyError(f"fallback_paid_not_permitted: {cand.profile_id}")
        if cand.cost_microusd > policy.max_cost_microusd:
            raise SafetyError(f"fallback_cost_exceeds_allowance: {cand.profile_id}")
        if cand.context_tokens > policy.max_context_tokens:
            raise SafetyError(f"fallback_context_exceeds_budget: {cand.profile_id}")
        return cand
    raise SafetyError("fallback_none_permitted: no approved fallback profile")


# ===========================================================================
# A12 — Wake / tick (bounded, idempotent)
# ===========================================================================

WAKE_OPERATION = "wake"

# Campaign states from which a wake must not advance work.
_WAKE_BLOCKED_STATES = frozenset({
    "EFFECT_UNKNOWN", "NEEDS_DECISION", "CANCELLED", "EXPIRED",
    "COMPLETE", "BUDGET_EXHAUSTED", "PAUSED_OPERATOR",
})


def _campaign_row(db: Database, campaign_id: str) -> dict[str, Any]:
    row = db._conn.execute(
        "SELECT * FROM campaigns WHERE campaign_id=?", (campaign_id,)
    ).fetchone()
    if row is None:
        raise SafetyError(f"campaign {campaign_id!r} not registered")
    return dict(row)


def _budget_headroom_reason(db: Database, campaign_id: str) -> str | None:
    from .admission import _load_ledger
    row = db._conn.execute(
        "SELECT ledger_id FROM budget_ledgers WHERE campaign_id=? "
        "ORDER BY revision DESC LIMIT 1", (campaign_id,)).fetchone()
    if row is None:
        return None
    ledger = _load_ledger(db, row["ledger_id"])
    if ledger is None:
        return "ledger missing"
    return ledger.would_exceed(delta_chunks=1, delta_model_calls=1,
                               delta_tool_calls=1, delta_active_seconds=1)


def wake_tick(
    db: Database,
    *,
    campaign_id: str,
    window_key: str,
    now: int | None = None,
) -> dict[str, Any]:
    """Re-evaluate authoritative durable state; perform at most one
    permitted advancement.

    Idempotent: duplicate wake requests within the same ``window_key`` do
    not produce a second advancement. A wake NEVER bypasses PAUSED,
    grant revocation/expiry, budget, blocked campaign state, capacity
    cooldown, ``next_eligible_at``, or already-running work.
    """
    now = int(now if now is not None else time.time())
    from .runtime import is_paused

    decision = "NOOP"
    reason = "nothing_due"
    advanced = 0
    job_id = f"wake-{campaign_id}-{window_key}"

    # Idempotency: one durable advancement per (campaign, window).
    existing = db._conn.execute(
        "SELECT job_id FROM wake_jobs WHERE job_id=?", (job_id,)).fetchone()
    if existing is not None:
        return {"decision": "DUPLICATE", "advanced": 0,
                "reason": "already_advanced_in_window", "job_id": job_id}

    try:
        row = _campaign_row(db, campaign_id)
    except SafetyError as e:
        return {"decision": "REFUSED", "advanced": 0, "reason": str(e),
                "job_id": job_id}

    # Authority gates (never bypassed by a wake).
    if is_paused():
        decision, reason = "REFUSED", "paused"
    elif row["state"] in _WAKE_BLOCKED_STATES:
        decision, reason = "REFUSED", f"blocked_state:{row['state']}"
    else:
        grant_id = row.get("grant_id", "")
        try:
            from .admission import require_active_grant_or_raise
            require_active_grant_or_raise(db, grant_id=grant_id, now=now)
        except SafetyError as e:
            decision, reason = "REFUSED", f"grant:{e}"
        else:
            head = _budget_headroom_reason(db, campaign_id)
            if head is not None:
                decision, reason = "REFUSED", f"budget:{head}"
            else:
                nxt = next_eligible_at(db, campaign_id=campaign_id)
                if nxt and now < nxt:
                    decision, reason = "WAIT", f"not_eligible_until:{nxt}"
                else:
                    live = db._conn.execute(
                        "SELECT COUNT(*) AS n FROM leases WHERE campaign_id=? "
                        "AND released_at=0 AND (expires_at=0 OR expires_at>?)",
                        (campaign_id, now)).fetchone()["n"]
                    if int(live) > 0:
                        decision, reason = "WAIT", "work_already_running"
                    else:
                        decision, reason = "DISPATCH_ELIGIBLE", "next_advancement"

    with db.transaction() as cur:
        # Re-check under the write lock so concurrent duplicate wakes
        # cannot both advance.
        again = cur.execute("SELECT job_id FROM wake_jobs WHERE job_id=?",
                            (job_id,)).fetchone()
        if again is not None:
            return {"decision": "DUPLICATE", "advanced": 0,
                    "reason": "already_advanced_in_window", "job_id": job_id}
        cur.execute(
            "INSERT INTO wake_jobs (job_id, operation, campaign_id, requested_at, "
            "state, detail, wake_generation) VALUES (?,?,?,?,?,?,1)",
            (job_id, WAKE_OPERATION, campaign_id, now, decision, reason),
        )
    if decision == "DISPATCH_ELIGIBLE":
        advanced = 1
    return {"decision": decision, "advanced": advanced, "reason": reason,
            "job_id": job_id}


# ===========================================================================
# A07 — Short control returns a durable job identity
# ===========================================================================

def request_control(
    db: Database,
    *,
    operation: str,
    campaign_id: str = "",
    window_key: str = "",
    now: int | None = None,
) -> str:
    """Record a short-control request and return a DURABLE job id.

    A short-control "success" means the control was ACCEPTED, never that
    implementation completed. Long execution remains runner-owned and is
    observed via authoritative runner status.
    """
    now = int(now if now is not None else time.time())
    job_id = f"ctl-{operation}-{campaign_id}-{window_key or now}-{uuid.uuid4().hex[:8]}"
    with db.transaction() as cur:
        cur.execute(
            "INSERT INTO wake_jobs (job_id, operation, campaign_id, requested_at, "
            "state, detail, wake_generation) VALUES (?,?,?,?,?,?,1)",
            (job_id, operation, campaign_id, now, "ACCEPTED", "control_accepted"),
        )
    return job_id


def job_state(db: Database, job_id: str) -> dict[str, Any] | None:
    row = db._conn.execute("SELECT * FROM wake_jobs WHERE job_id=?", (job_id,)).fetchone()
    return dict(row) if row is not None else None


def complete_job(db: Database, job_id: str, *, state: str, detail: str = "") -> None:
    """Runner-authoritative completion. Only the runner calls this."""
    if state not in ("COMPLETED", "NOOP", "REFUSED"):
        raise SafetyError(f"invalid job terminal state: {state!r}")
    with db.transaction() as cur:
        cur.execute(
            "UPDATE wake_jobs SET state=?, detail=? WHERE job_id=?",
            (state, detail[:512], job_id),
        )


# ===========================================================================
# A10 / A01 — Status projection (read-only runner truth)
# ===========================================================================

def project_campaign_status(db: Database, *, campaign_id: str,
                            now: int | None = None) -> dict[str, Any]:
    """Project authoritative runner truth. Creates no authority."""
    now = int(now if now is not None else time.time())
    row = db._conn.execute(
        "SELECT * FROM campaigns WHERE campaign_id=?", (campaign_id,)).fetchone()
    if row is None:
        raise SafetyError(f"campaign {campaign_id!r} not registered")
    camp = dict(row)

    chunks = db._conn.execute(
        "SELECT chunk_id, state, snapshot_commit FROM chunks "
        "WHERE campaign_id=? ORDER BY created_at DESC LIMIT 1", (campaign_id,)
    ).fetchone()
    last_accepted = None
    ja = db._conn.execute(
        "SELECT committed_new_commit FROM integration_journal WHERE campaign_id=? "
        "ORDER BY committed_at DESC LIMIT 1", (campaign_id,)).fetchone()
    if ja is not None:
        last_accepted = ja["committed_new_commit"]

    grant = None
    gid = camp.get("grant_id", "")
    if gid:
        g = db._conn.execute("SELECT state, payload_json FROM grants WHERE grant_id=?",
                             (gid,)).fetchone()
        if g is not None:
            try:
                payload = json.loads(g["payload_json"])
                grant = {
                    "grant_id": gid, "state": g["state"],
                    "grant_expires_at": payload.get("budget", {}).get("grant_expires_at", 0),
                }
            except Exception:
                grant = {"grant_id": gid, "state": g["state"], "grant_expires_at": 0}

    remaining = None
    led = db._conn.execute(
        "SELECT * FROM budget_ledgers WHERE campaign_id=? ORDER BY revision DESC LIMIT 1",
        (campaign_id,)).fetchone()
    if led is not None:
        b = json.loads(led["bounds_json"])
        remaining = {
            "chunks": max(0, b["max_chunks"] - led["cumulative_chunks"]),
            "model_calls": max(0, b["max_model_calls"] - led["cumulative_model_calls"]),
            "tool_calls": max(0, b["max_tool_calls"] - led["cumulative_tool_calls"]),
            "wall_seconds": max(0, b["max_wall_seconds"] - led["cumulative_wall_seconds"]),
        }

    waits = active_waits(db, campaign_id=campaign_id)
    wait = None
    if waits:
        w = sorted(waits, key=lambda x: int(x["entered_at"]))[-1]
        wait = {"reason": w["reason_kind"], "next_eligible_at": int(w["next_eligible_at"]),
                "provider": w["provider"], "profile": w["profile"]}

    safe_next = {
        "ACTIVE": "wake or admit the next eligible chunk",
        "WAIT_CAPACITY": "wait until next_eligible_at, then wake",
        "EFFECT_UNKNOWN": "operator reconciliation required",
        "NEEDS_DECISION": "operator decision required",
        "PAUSED_OPERATOR": "operator resume required",
    }.get(camp["state"], "inspect runner status")

    human_action = camp["state"] in ("EFFECT_UNKNOWN", "NEEDS_DECISION", "PAUSED_OPERATOR")

    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "state": camp["state"],
        "current_chunk": chunks["chunk_id"] if chunks else None,
        "current_chunk_state": chunks["state"] if chunks else None,
        "last_accepted_snapshot": last_accepted,
        "grant": grant,
        "budget_remaining": remaining,
        "capacity_wait": wait,
        "next_eligible_at": wait["next_eligible_at"] if wait else 0,
        "provider_profile": camp.get("plan_id", ""),
        "human_action_required": human_action,
        "safe_next_action": safe_next,
        "authoritative_runner_job_id": None,
        "projection_source": "runner",
    }


def project_hermes_cards(db: Database, *, campaign_id: str) -> list[dict[str, Any]]:
    """Rebuildable, NON-authoritative projection cards referencing runner IDs.

    Hermes cards carry runner IDs as their only identity. Deleting and
    rebuilding this projection always yields the same result from runner
    state; duplicate cards cannot create duplicate runner chunks.
    """
    st = project_campaign_status(db, campaign_id=campaign_id)
    cards = [{
        "card_id": f"runner:{campaign_id}:campaign",
        "kind": "campaign",
        "runner_id": campaign_id,
        "authoritative": False,
        "state": st["state"],
        "human_action_required": st["human_action_required"],
    }]
    if st["current_chunk"]:
        cards.append({
            "card_id": f"runner:{campaign_id}:chunk:{st['current_chunk']}",
            "kind": "chunk",
            "runner_id": st["current_chunk"],
            "campaign_id": campaign_id,
            "authoritative": False,
            "state": st["current_chunk_state"],
            "human_action_required": st["human_action_required"],
        })
    return cards


# ===========================================================================
# Narrow schema-defined adapter surface (what thin Hermes skills call)
# ===========================================================================

ADAPTER_SCHEMA: dict[str, Any] = {
    "version": CONTROL_SCHEMA_VERSION,
    "surfaces": {
        "status": {"kind": "read", "returns": STATUS_SCHEMA_VERSION,
                   "authority": "none"},
        "cards": {"kind": "read", "returns": "cards", "authority": "none"},
        "capacity": {"kind": "read", "returns": "cooldowns", "authority": "none"},
        "wake": {"kind": "control", "returns": "wake_jobs", "authority": "bounded"},
        "control": {"kind": "control", "returns": "wake_jobs", "authority": "bounded"},
    },
    "forbidden": [
        "direct_sqlite", "repository_filesystem_write", "arbitrary_shell",
        "git_mutation", "grant_or_approval_mint", "independent_worker_launch",
        "mark_code_accepted", "forge_validation_receipts",
    ],
}


__all__ = [
    "CAPACITY_OUTCOME_VERSION", "CONTROL_SCHEMA_VERSION", "STATUS_SCHEMA_VERSION",
    "CapacityKind", "COOLING_KINDS", "TERMINAL_AUTHORITY_KINDS",
    "CapacityOutcome", "classify_capacity",
    "upsert_cooldown", "cooldown_state", "list_cooldowns", "apply_capacity_outcome",
    "enter_capacity_wait", "active_waits", "next_eligible_at", "resolve_capacity_wait",
    "FallbackProfile", "FallbackPolicy", "resolve_fallback",
    "wake_tick", "WAKE_OPERATION",
    "request_control", "job_state", "complete_job",
    "project_campaign_status", "project_hermes_cards",
    "ADAPTER_SCHEMA",
]
