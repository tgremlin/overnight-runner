"""P07 — Hermes foreman integration (runner-authoritative surfaces).

Hermes is a thin projection + conversation + bounded wake/control
surface. It is NOT a second lifecycle/queue/scheduler/approval/mutation
authority. The runner remains authoritative.

Review corrections applied here (P07 follow-up):
  * A02: EVERY normalized capacity outcome is persisted (append-only);
    classification uses ONLY trusted structured fields (HTTP status,
    documented provider error codes via provider adapters, Retry-After,
    structured reset, transport class). Free-form provider prose is
    bounded evidence only, never authority.
  * A03: wake consults the runner-owned cooldown authority (not only
    ``capacity_waits``); known-reset waits resume at/after reset;
    no-reset quota and auth/invalid-profile do NOT auto-resume.
  * A06: wake claims a REAL runner obligation with a runner-owned
    idempotency key derived from authoritative state; different external
    trigger ids converge on ONE claim.
  * A05: fallback authority is derived from runner-owned state (grant +
    approved-fallback registry), never a caller-fabricated policy.
  * A07: public job polling via ``job_state`` + CLI ``hermes-job``;
    bounded control-operation allowlist.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from .db import Database
from .safety import SafetyError

CAPACITY_OUTCOME_VERSION = "trio.capacity-outcome.v1"
CONTROL_SCHEMA_VERSION = "trio.hermes-control.v1"
STATUS_SCHEMA_VERSION = "trio.runner-status.v1"
FALLBACK_POLICY_VERSION = "trio.fallback-policy.v1"


# ===========================================================================
# A02 — Normalized provider capacity outcomes (structured inputs only)
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


COOLING_KINDS = frozenset({
    CapacityKind.RATE_LIMITED, CapacityKind.OVERLOADED,
    CapacityKind.QUOTA_EXHAUSTED_KNOWN_RESET,
    CapacityKind.QUOTA_EXHAUSTED_NO_RESET,
    CapacityKind.PROVIDER_UNAVAILABLE,
})

# Kinds that must never be treated as a transient retry.
TERMINAL_AUTHORITY_KINDS = frozenset({
    CapacityKind.AUTH_FAILURE,
    CapacityKind.INVALID_PROVIDER_MODEL_PROFILE,
    CapacityKind.QUOTA_EXHAUSTED_NO_RESET,
})

# Provider-specific adapters: documented EXACT provider error
# ``type``/``code`` values -> canonical code. This keeps classification
# deterministic without scanning free-form prose.
PROVIDER_ERROR_ADAPTERS: dict[str, dict[str, str]] = {
    "anthropic": {
        "rate_limit_error": "rate_limit_exceeded",
        "authentication_error": "authentication_error",
        "permission_error": "permission_denied",
        "not_found_error": "model_not_found",
        "invalid_request_error": "invalid_request_error",
        "overloaded_error": "overloaded",
        "api_error": "provider_unavailable",
    },
    "minimax": {
        "rate_limit_error": "rate_limit_exceeded",
        "authentication_error": "authentication_error",
        "permission_error": "permission_denied",
        "not_found_error": "model_not_found",
        "invalid_request_error": "invalid_request_error",
        "overloaded_error": "overloaded",
        "api_error": "provider_unavailable",
        "insufficient_quota": "insufficient_quota",
    },
    "openai": {
        "rate_limit_exceeded": "rate_limit_exceeded",
        "insufficient_quota": "insufficient_quota",
        "invalid_api_key": "invalid_api_key",
        "model_not_found": "model_not_found",
        "invalid_request_error": "invalid_request_error",
        "server_error": "provider_unavailable",
        "temporarily_unavailable": "temporarily_unavailable",
    },
}

# Structured canonical codes that mean "quota exhausted".
_QUOTA_CODES = frozenset({
    "quota_exhausted", "insufficient_quota", "usage_limit_reached",
    "billing_hard_limit_reached",
})
_QUOTA_KNOWN_RESET_CODES = frozenset({
    "quota_exhausted_known_reset", "usage_limit_reset_known",
})


def adapt_provider_error(provider: str, provider_error_type: str) -> str:
    """Map a documented provider error type to a canonical code.

    Returns ``""`` when the provider/type is unknown so the caller does
    not get accidental authority from an unrecognized field.
    """
    table = PROVIDER_ERROR_ADAPTERS.get((provider or "").lower(), {})
    return table.get((provider_error_type or "").lower(), "")


@dataclass(frozen=True)
class CapacityOutcome:
    schema_version: str
    kind: str
    provider: str
    model: str
    http_status: int | None
    error_code: str
    retry_after_seconds: int | None
    reset_at: int | None
    transport_class: str
    source: str
    classification_version: str
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_capacity(
    *,
    provider: str = "",
    model: str = "",
    http_status: int | None = None,
    error_code: str = "",
    provider_error_type: str = "",
    retry_after_seconds: int | None = None,
    reset_at: int | None = None,
    transport_error: str = "",
    message: str = "",
) -> CapacityOutcome:
    """Deterministically normalize a provider/capacity outcome.

    Authoritative inputs are ONLY: HTTP status, a documented provider
    error ``type`` (through the provider adapter) or an already-canonical
    ``error_code``, explicit ``Retry-After``, a structured ``reset_at``,
    and the transport exception class.

    ``message`` is retained as bounded EVIDENCE only and is NEVER scanned
    for classification.
    """
    code = (provider_error_type or "").strip().lower()
    if code and provider:
        canonical = adapt_provider_error(provider, code)
        if canonical:
            code = canonical
    if not code:
        code = (error_code or "").strip().lower()

    def _out(k: CapacityKind, source: str) -> CapacityOutcome:
        return CapacityOutcome(
            schema_version=CAPACITY_OUTCOME_VERSION, kind=k.value,
            provider=provider, model=model, http_status=http_status,
            error_code=code, retry_after_seconds=retry_after_seconds,
            reset_at=reset_at,
            transport_class=(transport_error or "").lower(),
            source=source, classification_version=CAPACITY_OUTCOME_VERSION,
            evidence=(message or "")[:512],
        )

    # Quota (structured code only).
    if code in _QUOTA_KNOWN_RESET_CODES:
        return _out(CapacityKind.QUOTA_EXHAUSTED_KNOWN_RESET, "provider_error_code")
    if code in _QUOTA_CODES:
        if reset_at or retry_after_seconds:
            return _out(CapacityKind.QUOTA_EXHAUSTED_KNOWN_RESET, "provider_error_code")
        return _out(CapacityKind.QUOTA_EXHAUSTED_NO_RESET, "provider_error_code")

    # Auth.
    if code in ("invalid_api_key", "authentication_error", "unauthorized",
                "permission_denied", "insufficient_scope"):
        return _out(CapacityKind.AUTH_FAILURE, "provider_error_code")
    if http_status in (401, 403):
        return _out(CapacityKind.AUTH_FAILURE, "http_status")

    # Invalid provider/model/profile.
    if code in ("model_not_found", "invalid_model", "invalid_profile",
                "unknown_model", "invalid_request_error"):
        return _out(CapacityKind.INVALID_PROVIDER_MODEL_PROFILE, "provider_error_code")
    if http_status in (400, 404, 422):
        return _out(CapacityKind.INVALID_PROVIDER_MODEL_PROFILE, "http_status")

    # Explicit rate limit.
    if http_status == 429 or code in ("rate_limit_exceeded", "too_many_requests"):
        return _out(CapacityKind.RATE_LIMITED, "http_status" if http_status == 429
                    else "provider_error_code")

    # Overload / temporary unavailability.
    if http_status in (503, 529) or code in ("overloaded", "server_overloaded",
                                             "temporarily_unavailable"):
        return _out(CapacityKind.OVERLOADED, "http_status" if http_status in (503, 529)
                    else "provider_error_code")

    # Provider-side 5xx.
    if http_status in (500, 502, 504) or code == "provider_unavailable":
        return _out(CapacityKind.PROVIDER_UNAVAILABLE, "http_status")

    # Transport class.
    if transport_error:
        te = transport_error.lower()
        if te in ("connectionerror", "connectionrefused", "timeout", "connecttimeout",
                  "readtimeout", "dns"):
            return _out(CapacityKind.PROVIDER_UNAVAILABLE, "transport")
        return _out(CapacityKind.UNKNOWN_TRANSPORT, "transport")

    return _out(CapacityKind.OK, "none")


def record_capacity_outcome(
    db: Database,
    outcome: CapacityOutcome,
    *,
    account: str = "",
    campaign_id: str = "",
    chunk_id: str = "",
    obligation: str = "",
    now: int | None = None,
) -> str:
    """Persist a normalized outcome (append-only). Returns the outcome id."""
    now = int(now if now is not None else time.time())
    outcome_id = f"cap-{uuid.uuid4().hex[:16]}"
    with db.transaction() as cur:
        cur.execute(
            """
            INSERT INTO capacity_outcomes (
                outcome_id, schema_version, provider, account, model, kind,
                http_status, error_code, retry_after_seconds, reset_at,
                transport_class, occurred_at, campaign_id, chunk_id, obligation,
                evidence_ref, classification_source, classification_version
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (outcome_id, outcome.schema_version, outcome.provider, account,
             outcome.model, outcome.kind, outcome.http_status, outcome.error_code,
             outcome.retry_after_seconds, outcome.reset_at, outcome.transport_class,
             now, campaign_id, chunk_id, obligation, outcome.evidence,
             outcome.source, outcome.classification_version),
        )
    return outcome_id


def capacity_outcome_history(
    db: Database, *, kind: str = "", campaign_id: str = "",
) -> list[dict[str, Any]]:
    q = "SELECT * FROM capacity_outcomes"
    clauses, params = [], []
    if kind:
        clauses.append("kind=?")
        params.append(kind)
    if campaign_id:
        clauses.append("campaign_id=?")
        params.append(campaign_id)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY occurred_at ASC"
    return [dict(r) for r in db._conn.execute(q, tuple(params)).fetchall()]


# ===========================================================================
# A11 — Durable provider/account cooldown (projection for cooling kinds)
# ===========================================================================

def upsert_cooldown(
    db: Database, *, provider: str, kind: str, until_at: int,
    account: str = "", model: str = "", reason: str = "", now: int | None = None,
) -> None:
    now = int(now if now is not None else time.time())
    with db.transaction() as cur:
        cur.execute(
            """
            INSERT INTO provider_cooldowns
                (provider, account, model, kind, reason, until_at, updated_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(provider, account, model) DO UPDATE SET
                kind=excluded.kind, reason=excluded.reason,
                until_at=MAX(provider_cooldowns.until_at, excluded.until_at),
                updated_at=excluded.updated_at
            """,
            (provider, account, model, kind, reason[:512], int(until_at), now),
        )


def cooldown_state(
    db: Database, *, provider: str = "", account: str = "", model: str = "",
    now: int | None = None,
) -> dict[str, Any] | None:
    now = int(now if now is not None else time.time())
    row = db._conn.execute(
        "SELECT * FROM provider_cooldowns WHERE provider=? AND account=? AND model=?",
        (provider, account, model)).fetchone()
    if row is None:
        return None
    d = dict(row)
    if int(d["until_at"]) and int(d["until_at"]) <= now:
        return None
    d["cooling"] = True
    return d


def active_cooldown_for(
    db: Database, *, provider: str, account: str = "", model: str = "",
    now: int | None = None,
) -> dict[str, Any] | None:
    """Runner cooldown authority for the exact scope.

    ``until_at == 0`` means an INDEFINITE (unknown-reset) cooldown and is
    treated as active until explicitly revised.
    """
    return cooldown_state(db, provider=provider, account=account, model=model, now=now)


def list_cooldowns(db: Database, *, now: int | None = None) -> list[dict[str, Any]]:
    now = int(now if now is not None else time.time())
    out = []
    for r in db._conn.execute("SELECT * FROM provider_cooldowns").fetchall():
        d = dict(r)
        d["cooling"] = bool(int(d["until_at"]) == 0 or int(d["until_at"]) > now)
        out.append(d)
    return out


def apply_capacity_outcome(
    db: Database, outcome: CapacityOutcome, *, account: str = "",
    campaign_id: str = "", chunk_id: str = "", obligation: str = "",
    now: int | None = None,
) -> dict[str, Any]:
    """Persist EVERY outcome; create a cooldown only for cooling kinds.

    Returns ``{"outcome_id":..., "cooldown": row|None}``.
    """
    now = int(now if now is not None else time.time())
    outcome_id = record_capacity_outcome(
        db, outcome, account=account, campaign_id=campaign_id, chunk_id=chunk_id,
        obligation=obligation, now=now)
    cooldown = None
    if outcome.kind in {k.value for k in COOLING_KINDS}:
        until = 0
        if outcome.reset_at:
            until = int(outcome.reset_at)
        elif outcome.retry_after_seconds:
            until = now + int(outcome.retry_after_seconds)
        upsert_cooldown(db, provider=outcome.provider, account=account,
                        model=outcome.model, kind=outcome.kind, until_at=until,
                        reason=outcome.evidence, now=now)
        cooldown = cooldown_state(db, provider=outcome.provider, account=account,
                                  model=outcome.model, now=now)
    return {"outcome_id": outcome_id, "cooldown": cooldown}


# ===========================================================================
# A03 — Durable capacity wait (WAIT_CAPACITY)
# ===========================================================================

def enter_capacity_wait(
    db: Database, *, campaign_id: str, reason_kind: str, grant_id: str = "",
    budget_ledger_id: str = "", chunk_id: str = "", obligation: str = "",
    provider: str = "", account: str = "", profile: str = "",
    next_eligible_at: int = 0, retry_provenance: str = "", now: int | None = None,
) -> str:
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
    waits = active_waits(db, campaign_id=campaign_id)
    if not waits:
        return 0
    return max(int(w["next_eligible_at"]) for w in waits)


def resolve_capacity_wait(db: Database, *, wait_id: str, now: int | None = None) -> None:
    now = int(now if now is not None else time.time())
    with db.transaction() as cur:
        cur.execute(
            "UPDATE capacity_waits SET active=0, resolved_at=? WHERE wait_id=? AND active=1",
            (now, wait_id))


# ===========================================================================
# A05 — Finite approved fallback with runner-owned provenance
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
    """Runner-owned fallback envelope. ``provenance`` MUST be "runner";
    a caller-fabricated policy (provenance "caller") is not authoritative."""
    role: str
    qualified_models: frozenset[str]
    approved_providers: frozenset[str]
    approved_profiles: frozenset[str]
    runtime_digest: str
    egress_policy_id: str
    allow_paid_spend: bool
    max_cost_microusd: int
    max_context_tokens: int
    policy_version: str = FALLBACK_POLICY_VERSION
    provenance: str = "caller"


def register_approved_fallback(
    db: Database, *, role: str, profile_id: str, model_name: str, provider: str,
    runtime_digest: str, egress_policy_id: str, paid: bool = False,
    cost_microusd: int = 0, context_tokens: int = 0,
) -> None:
    """Trusted operator surface: register a qualified approved fallback."""
    with db.transaction() as cur:
        cur.execute(
            """
            INSERT INTO approved_fallbacks (
                role, profile_id, model_name, provider, runtime_digest,
                egress_policy_id, paid, cost_microusd, context_tokens
            ) VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(role, profile_id) DO UPDATE SET
                model_name=excluded.model_name, provider=excluded.provider,
                runtime_digest=excluded.runtime_digest,
                egress_policy_id=excluded.egress_policy_id, paid=excluded.paid,
                cost_microusd=excluded.cost_microusd,
                context_tokens=excluded.context_tokens
            """,
            (role, profile_id, model_name, provider, runtime_digest,
             egress_policy_id, int(paid), int(cost_microusd), int(context_tokens)),
        )


def load_fallback_policy(db: Database, *, grant_id: str, role: str) -> FallbackPolicy:
    """Build the fallback envelope from RUNNER-OWNED state only."""
    from .admission import require_active_grant_or_raise
    grant = require_active_grant_or_raise(db, grant_id=grant_id)
    rows = db._conn.execute(
        "SELECT * FROM approved_fallbacks WHERE role=?", (role,)).fetchall()
    profiles = {r["profile_id"] for r in rows}
    models = {r["model_name"] for r in rows}
    providers = {r["provider"] for r in rows}
    # Remaining cost/context headroom from the trusted ledger, if present.
    led = db._conn.execute(
        "SELECT * FROM budget_ledgers WHERE grant_id=? ORDER BY revision DESC LIMIT 1",
        (grant_id,)).fetchone()
    if led is not None:
        bounds = json.loads(led["bounds_json"])
        remaining_cost = max(0, bounds["max_cost_microusd"] - led["cumulative_cost_microusd"])
        remaining_ctx = max(0, bounds["context_token_budget"] - led["cumulative_context_tokens"])
    else:
        remaining_cost = grant.budget.max_cost_microusd
        remaining_ctx = grant.budget.context_token_budget
    # Paid spend is permitted only when the grant carries an explicit
    # spend allowance (max_cost_microusd > 0).
    allow_paid = grant.budget.max_cost_microusd > 0
    return FallbackPolicy(
        role=role, qualified_models=frozenset(models),
        approved_providers=frozenset(providers), approved_profiles=frozenset(profiles),
        runtime_digest=grant.runtime_digest, egress_policy_id=grant.egress_policy_id,
        allow_paid_spend=allow_paid, max_cost_microusd=remaining_cost,
        max_context_tokens=remaining_ctx, provenance="runner",
    )


def _resolve_with_policy(
    primary: FallbackProfile, candidates: list[FallbackProfile], *,
    policy: FallbackPolicy, primary_unavailable: bool,
) -> FallbackProfile:
    if policy.provenance != "runner":
        raise SafetyError("fallback_policy_untrusted: policy was not derived "
                          "from runner-owned state")
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


def resolve_fallback(
    db: Database, *, grant_id: str, role: str, primary: FallbackProfile,
    candidates: list[FallbackProfile], primary_unavailable: bool,
) -> FallbackProfile:
    """Resolve fallback using a RUNNER-DERIVED policy. The caller cannot
    supply the policy object."""
    policy = load_fallback_policy(db, grant_id=grant_id, role=role)
    return _resolve_with_policy(primary, candidates, policy=policy,
                                primary_unavailable=primary_unavailable)


# ===========================================================================
# A06 — Wake / tick (real claim, runner-owned idempotency)
# ===========================================================================

WAKE_OPERATION = "wake"

_WAKE_BLOCKED_STATES = frozenset({
    "EFFECT_UNKNOWN", "NEEDS_DECISION", "CANCELLED", "EXPIRED",
    "COMPLETE", "BUDGET_EXHAUSTED", "PAUSED_OPERATOR",
})

# Controls the adapter may accept. Anything else is rejected.
ALLOWED_CONTROL_OPERATIONS = frozenset({"wake", "tick", "status-refresh"})


def _campaign_row(db: Database, campaign_id: str) -> dict[str, Any]:
    row = db._conn.execute("SELECT * FROM campaigns WHERE campaign_id=?",
                           (campaign_id,)).fetchone()
    if row is None:
        raise SafetyError(f"campaign {campaign_id!r} not registered")
    return dict(row)


def obligation_identity(camp: dict[str, Any]) -> tuple[str, int]:
    """Runner-owned obligation identity derived from authoritative state."""
    gen = int(camp.get("current_fence", 1))
    obligation_id = f"{camp['campaign_id']}:{camp.get('current_commit') or 'none'}:{gen}"
    return obligation_id, gen


def _routing_identity(db: Database, camp: dict[str, Any],
                      waits: list[dict[str, Any]]) -> tuple[str, str, str]:
    for w in waits:
        if w.get("provider"):
            return w["provider"], w.get("account", ""), w.get("profile", "")
    gid = camp.get("grant_id", "")
    row = db._conn.execute("SELECT payload_json FROM grants WHERE grant_id=?",
                           (gid,)).fetchone()
    if row is not None:
        try:
            payload = json.loads(row["payload_json"])
            return (payload.get("provider_profile_id", ""), "",
                    payload.get("model_name", ""))
        except Exception:
            pass
    return "", "", ""


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


def _wake_decision(db: Database, camp: dict[str, Any], now: int) -> tuple[str, str]:
    from .runtime import is_paused
    cid = camp["campaign_id"]
    if is_paused():
        return "REFUSED", "paused"
    if camp["state"] in _WAKE_BLOCKED_STATES:
        return "REFUSED", f"blocked_state:{camp['state']}"
    try:
        from .admission import require_active_grant_or_raise
        require_active_grant_or_raise(db, grant_id=camp.get("grant_id", ""), now=now)
    except SafetyError as e:
        return "REFUSED", f"grant:{e}"

    waits = active_waits(db, campaign_id=cid)
    # No-reset quota / auth / invalid-profile waits never auto-resume.
    for w in waits:
        if w["reason_kind"] in ("quota_exhausted_no_reset",):
            return "REFUSED", f"capacity_parked_no_reset:{w['reason_kind']}"
        if w["reason_kind"] in ("auth_failure", "invalid_provider_model_profile"):
            return "REFUSED", f"capacity_authority_required:{w['reason_kind']}"

    # Cooldown authority (independent of capacity_waits.next_eligible_at).
    provider, account, model = _routing_identity(db, camp, waits)
    cool = active_cooldown_for(db, provider=provider, account=account, model=model, now=now)
    if cool is not None:
        kind = cool["kind"]
        until = int(cool["until_at"])
        if kind in ("auth_failure", "invalid_provider_model_profile"):
            return "REFUSED", f"cooldown_authority_required:{kind}"
        if kind == "quota_exhausted_no_reset" or until == 0:
            return "REFUSED", f"cooldown_no_reset:{kind}"
        if now < until:
            return "WAIT", f"cooldown_until:{until}"
        # expired cooldown -> fall through

    nxt = next_eligible_at(db, campaign_id=cid)
    if nxt and now < nxt:
        return "WAIT", f"not_eligible_until:{nxt}"

    head = _budget_headroom_reason(db, cid)
    if head is not None:
        return "REFUSED", f"budget:{head}"

    live = db._conn.execute(
        "SELECT COUNT(*) AS n FROM leases WHERE campaign_id=? AND released_at=0 "
        "AND (expires_at=0 OR expires_at>?)", (cid, now)).fetchone()["n"]
    if int(live) > 0:
        return "WAIT", "work_already_running"

    return "DISPATCH_ELIGIBLE", "next_advancement"


def wake_tick(
    db: Database, *, campaign_id: str, trigger_id: str = "", now: int | None = None,
) -> dict[str, Any]:
    """Re-evaluate authoritative state; at most one REAL claim per
    obligation. Idempotency is keyed on runner state, so competing
    external trigger ids converge on one claim."""
    now = int(now if now is not None else time.time())
    camp = _campaign_row(db, campaign_id)
    obligation_id, gen = obligation_identity(camp)
    claim_id = f"wclaim-{obligation_id}"
    job_id = f"wake-{obligation_id}"

    if db._conn.execute("SELECT claim_id FROM wake_claims WHERE claim_id=?",
                        (claim_id,)).fetchone() is not None:
        return {"decision": "DUPLICATE", "advanced": 0,
                "reason": "obligation_already_claimed", "job_id": job_id,
                "claim_id": claim_id, "obligation_id": obligation_id}

    decision, reason = _wake_decision(db, camp, now)
    advanced = 0
    with db.transaction() as cur:
        # Re-check under the write lock (concurrent duplicate triggers).
        if cur.execute("SELECT claim_id FROM wake_claims WHERE claim_id=?",
                       (claim_id,)).fetchone() is not None:
            return {"decision": "DUPLICATE", "advanced": 0,
                    "reason": "obligation_already_claimed", "job_id": job_id,
                    "claim_id": claim_id, "obligation_id": obligation_id}
        cur.execute(
            """
            INSERT INTO wake_jobs (job_id, operation, campaign_id, requested_at,
                state, detail, wake_generation) VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(job_id) DO UPDATE SET
                state=excluded.state, detail=excluded.detail,
                requested_at=excluded.requested_at
            """,
            (job_id, WAKE_OPERATION, campaign_id, now, decision, reason, gen),
        )
        if decision == "DISPATCH_ELIGIBLE":
            cur.execute(
                """
                INSERT INTO wake_claims (claim_id, campaign_id, obligation_id,
                    generation, claimed_at, state, caller_trigger_id)
                VALUES (?,?,?,?,?,?,?)
                """,
                (claim_id, campaign_id, obligation_id, gen, now, "CLAIMED",
                 (trigger_id or "")[:256]),
            )
            # Atomically resolve the stale waits with the claim.
            cur.execute(
                "UPDATE capacity_waits SET active=0, resolved_at=? "
                "WHERE campaign_id=? AND active=1", (now, campaign_id))
            advanced = 1
    return {"decision": decision, "advanced": advanced, "reason": reason,
            "job_id": job_id, "claim_id": claim_id, "obligation_id": obligation_id}


def hermes_tick(db: Database, *, trigger_id: str = "", now: int | None = None) -> dict[str, Any]:
    """Runner-owned due-work tick over ALL active campaigns.

    Discovers due obligations from runner state; Hermes never needs to
    know campaign ids or window keys.
    """
    now = int(now if now is not None else time.time())
    rows = db._conn.execute("SELECT campaign_id FROM campaigns WHERE state='ACTIVE'").fetchall()
    results = []
    for r in rows:
        results.append(wake_tick(db, campaign_id=r["campaign_id"],
                                 trigger_id=trigger_id, now=now))
    return {"schema_version": CONTROL_SCHEMA_VERSION,
            "claims": sum(1 for x in results if x.get("advanced") == 1),
            "results": results}


# ===========================================================================
# A07 — Short control + public job polling
# ===========================================================================

def request_control(
    db: Database, *, operation: str, campaign_id: str = "",
    window_key: str = "", now: int | None = None,
) -> str:
    """Record a bounded short-control request; return a durable job id."""
    if operation not in ALLOWED_CONTROL_OPERATIONS:
        raise SafetyError(
            f"control_operation_not_allowed: {operation!r}; allowed="
            f"{sorted(ALLOWED_CONTROL_OPERATIONS)}")
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
    if state not in ("COMPLETED", "NOOP", "REFUSED"):
        raise SafetyError(f"invalid job terminal state: {state!r}")
    with db.transaction() as cur:
        cur.execute("UPDATE wake_jobs SET state=?, detail=? WHERE job_id=?",
                    (state, detail[:512], job_id))


# ===========================================================================
# A10 / A01 — Status projection (read-only runner truth)
# ===========================================================================

def project_campaign_status(db: Database, *, campaign_id: str,
                            now: int | None = None) -> dict[str, Any]:
    now = int(now if now is not None else time.time())
    camp = _campaign_row(db, campaign_id)

    chunk = db._conn.execute(
        "SELECT chunk_id, state FROM chunks WHERE campaign_id=? "
        "ORDER BY created_at DESC LIMIT 1", (campaign_id,)).fetchone()
    ja = db._conn.execute(
        "SELECT committed_new_commit FROM integration_journal WHERE campaign_id=? "
        "ORDER BY committed_at DESC LIMIT 1", (campaign_id,)).fetchone()
    last_accepted = ja["committed_new_commit"] if ja else None

    grant = None
    routing = {"provider": "", "model": "", "runtime_digest": ""}
    gid = camp.get("grant_id", "")
    if gid:
        g = db._conn.execute("SELECT state, payload_json FROM grants WHERE grant_id=?",
                             (gid,)).fetchone()
        if g is not None:
            try:
                payload = json.loads(g["payload_json"])
                routing = {
                    "provider": payload.get("provider_profile_id", ""),
                    "model": payload.get("model_name", ""),
                    "runtime_digest": payload.get("runtime_digest", ""),
                }
                grant = {"grant_id": gid, "state": g["state"],
                         "grant_expires_at": payload.get("budget", {}).get("grant_expires_at", 0)}
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
    cooldown = None
    if routing["provider"]:
        cool = active_cooldown_for(db, provider=routing["provider"],
                                   model=routing["model"], now=now)
        if cool is not None:
            cooldown = {"provider": cool["provider"], "model": cool["model"],
                        "kind": cool["kind"], "until_at": int(cool["until_at"])}

    # Authoritative latest runner job.
    jb = db._conn.execute(
        "SELECT job_id, state FROM wake_jobs WHERE campaign_id=? "
        "ORDER BY requested_at DESC LIMIT 1", (campaign_id,)).fetchone()
    runner_job_id = jb["job_id"] if jb else None

    human_action = camp["state"] in ("EFFECT_UNKNOWN", "NEEDS_DECISION", "PAUSED_OPERATOR")
    safe_next = _safe_next_action(camp["state"], wait, cooldown, now, human_action)

    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "state": camp["state"],
        "current_chunk": chunk["chunk_id"] if chunk else None,
        "current_chunk_state": chunk["state"] if chunk else None,
        "last_accepted_snapshot": last_accepted,
        "grant": grant,
        "provider_profile": routing["provider"],
        "model": routing["model"],
        "runtime_digest": routing["runtime_digest"],
        "budget_remaining": remaining,
        "capacity_wait": wait,
        "provider_cooldown": cooldown,
        "next_eligible_at": wait["next_eligible_at"] if wait else 0,
        "human_action_required": human_action,
        "safe_next_action": safe_next,
        "authoritative_runner_job_id": runner_job_id,
        "projection_source": "runner",
    }


def _safe_next_action(state: str, wait, cooldown, now: int, human_action: bool) -> str:
    if human_action:
        return {
            "EFFECT_UNKNOWN": "operator reconciliation required",
            "NEEDS_DECISION": "operator decision required",
            "PAUSED_OPERATOR": "operator resume required",
        }.get(state, "operator action required")
    if wait is not None:
        kind = wait["reason"]
        nxt = int(wait["next_eligible_at"])
        if kind in ("auth_failure", "invalid_provider_model_profile"):
            return "authority/provider correction required"
        if kind == "quota_exhausted_no_reset":
            return "parked; operator/fallback/provider evidence required"
        if nxt and now < nxt:
            return f"wait until {nxt}, then wake"
    if cooldown is not None:
        if cooldown["kind"] in ("auth_failure", "invalid_provider_model_profile"):
            return "authority/provider correction required"
        if cooldown["kind"] == "quota_exhausted_no_reset" or int(cooldown["until_at"]) == 0:
            return "parked; operator/fallback/provider evidence required"
        if now < int(cooldown["until_at"]):
            return f"wait until {cooldown['until_at']} (provider cooldown), then wake"
    if state == "ACTIVE":
        return "wake or admit the next eligible chunk"
    return "inspect runner status"


def project_hermes_cards(db: Database, *, campaign_id: str) -> list[dict[str, Any]]:
    st = project_campaign_status(db, campaign_id=campaign_id)
    cards = [{
        "card_id": f"runner:{campaign_id}:campaign",
        "kind": "campaign", "runner_id": campaign_id, "authoritative": False,
        "state": st["state"], "human_action_required": st["human_action_required"],
    }]
    if st["current_chunk"]:
        cards.append({
            "card_id": f"runner:{campaign_id}:chunk:{st['current_chunk']}",
            "kind": "chunk", "runner_id": st["current_chunk"],
            "campaign_id": campaign_id, "authoritative": False,
            "state": st["current_chunk_state"],
            "human_action_required": st["human_action_required"],
        })
    return cards


# ===========================================================================
# Narrow schema-defined adapter surface
# ===========================================================================

ADAPTER_SCHEMA: dict[str, Any] = {
    "version": CONTROL_SCHEMA_VERSION,
    "surfaces": {
        "status": {"kind": "read", "returns": STATUS_SCHEMA_VERSION, "authority": "none"},
        "cards": {"kind": "read", "returns": "cards", "authority": "none"},
        "capacity": {"kind": "read", "returns": "cooldowns", "authority": "none"},
        "job": {"kind": "read", "returns": "wake_jobs", "authority": "none"},
        "tick": {"kind": "control", "returns": "wake_claims", "authority": "bounded"},
        "wake": {"kind": "control", "returns": "wake_claims", "authority": "bounded"},
        "control": {"kind": "control", "returns": "wake_jobs", "authority": "bounded",
                    "allowed_operations": sorted(ALLOWED_CONTROL_OPERATIONS)},
    },
    "forbidden": [
        "direct_sqlite", "repository_filesystem_write", "arbitrary_shell",
        "git_mutation", "grant_or_approval_mint", "independent_worker_launch",
        "mark_code_accepted", "forge_validation_receipts", "second_scheduler",
        "second_queue",
    ],
}


__all__ = [
    "CAPACITY_OUTCOME_VERSION", "CONTROL_SCHEMA_VERSION", "STATUS_SCHEMA_VERSION",
    "FALLBACK_POLICY_VERSION",
    "CapacityKind", "COOLING_KINDS", "TERMINAL_AUTHORITY_KINDS",
    "PROVIDER_ERROR_ADAPTERS", "adapt_provider_error",
    "CapacityOutcome", "classify_capacity",
    "record_capacity_outcome", "capacity_outcome_history", "apply_capacity_outcome",
    "upsert_cooldown", "cooldown_state", "active_cooldown_for", "list_cooldowns",
    "enter_capacity_wait", "active_waits", "next_eligible_at", "resolve_capacity_wait",
    "FallbackProfile", "FallbackPolicy", "register_approved_fallback",
    "load_fallback_policy", "resolve_fallback",
    "WAKE_OPERATION", "ALLOWED_CONTROL_OPERATIONS", "obligation_identity",
    "wake_tick", "hermes_tick",
    "request_control", "job_state", "complete_job",
    "project_campaign_status", "project_hermes_cards",
    "ADAPTER_SCHEMA",
]
