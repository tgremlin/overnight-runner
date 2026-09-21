"""P06 — campaign-v2 strict Pydantic schemas.

These are the v2 surface models (Campaign, WorkPackage, ChunkSpec,
AutonomyGrant, AdmissionReceipt, Budget, Lease, CampaignEvent). They
extend the V1 surface with new additive concepts only; existing V1
``TaskManifest`` and ``Disposition`` enums are preserved.

Additive / opt-in: V1 callers do NOT see any schema change. campaign-v2
state is only minted when explicitly constructed (e.g. by the
campaign dispatcher, a CLI short-control operation, or a test).

A receipt created by a worker is NOT a campaign authority; the
runner-owned mint surface is the ``overnight_runner.receipts``
module. campaign-v2 entries merely REFERENCE receipt ids.
"""
from __future__ import annotations

import hashlib
import time
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated


SCHEMA_VERSION = "trio.campaign.v1"
ID = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$", min_length=1, max_length=128)]
# Accept SHA1 (40 chars) and SHA256 (64 chars) since git commits are SHA1
# while our receipt digests are SHA256. The pattern is lowercase hex.
SHA256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{40,64}$", min_length=40, max_length=64)]

# ---------------------------------------------------------------------------
# Class registry: lifecycle vocabulary used by ``Campaign.state`` and
# ``Chunk.state``. These are versioned, additive, and never widen V1 enums.
# ---------------------------------------------------------------------------


class CampaignState(str, Enum):
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    READY = "READY"
    RUNNING = "RUNNING"
    WAIT_CAPACITY = "WAIT_CAPACITY"
    WAIT_RESOURCE = "WAIT_RESOURCE"
    PAUSED_OPERATOR = "PAUSED_OPERATOR"
    NEEDS_DECISION = "NEEDS_DECISION"
    EFFECT_UNKNOWN = "EFFECT_UNKNOWN"
    INTEGRATION_REVIEW = "INTEGRATION_REVIEW"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class ChunkState(str, Enum):
    PROPOSED = "PROPOSED"
    VALIDATED = "VALIDATED"
    ADMITTED = "ADMITTED"
    CLAIMED = "CLAIMED"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    SEMANTIC_REVIEW = "SEMANTIC_REVIEW"
    ACCEPTED_FOR_CONTINUATION = "ACCEPTED_FOR_CONTINUATION"
    NEEDS_CONTEXT = "NEEDS_CONTEXT"
    REPAIR_PENDING = "REPAIR_PENDING"
    RECHUNK_PENDING = "RECHUNK_PENDING"
    BLOCKED = "BLOCKED"
    EFFECT_UNKNOWN = "EFFECT_UNKNOWN"
    SUPERSEDED = "SUPERSEDED"
    WAIT_CAPACITY = "WAIT_CAPACITY"


class WorkPackageState(str, Enum):
    PROPOSED = "PROPOSED"
    ELIGIBLE = "ELIGIBLE"
    ACTIVE = "ACTIVE"
    BLOCKED = "BLOCKED"
    INTEGRATION_REVIEW = "INTEGRATION_REVIEW"
    COMPLETE = "COMPLETE"


# Versioning: id format `cr-grant-{...}` prefixed by kind for clarity.
GRANT_KIND = "trio.grant.v1"
ADMISSION_KIND = "trio.admission.v1"
EVENT_KIND = "trio.campaign-event.v1"


# ---------------------------------------------------------------------------
# Strict base
# ---------------------------------------------------------------------------


class StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False, str_strip_whitespace=False)


# ---------------------------------------------------------------------------
# Repo snapshot
# ---------------------------------------------------------------------------


class RepoSnapshot(StrictBase):
    """An exact (commit + tree) reference to a repository state.

    Used in admission receipts and chunk specs to anchor exactly what
    is being admitted. ``commit`` is the literal git commit; ``tree_digest``
    is a content fingerprint of the WORKING TREE (captures untracked /
    staged-but-uncommitted state for runner-internal purposes).
    """

    schema_version: Literal["trio.repo-snapshot.v1"] = "trio.repo-snapshot.v1"
    repository_id: ID
    commit: SHA256
    tree_digest: SHA256


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class Budget(StrictBase):
    """Cumulative campaign budget. Bound checks happen at admission and
    on each successful chunk completion. Sessions, restarts, rechunks,
    and provider changes do NOT mint new budget — the same ledger is
    preserved across them."""

    schema_version: Literal["trio.budget.v1"] = "trio.budget.v1"
    max_model_calls: int = Field(default=6, ge=0, le=10_000)
    max_tool_calls: int = Field(default=20, ge=0, le=100_000)
    max_local_repairs: int = Field(default=2, ge=0, le=100)
    max_rechunks: int = Field(default=1, ge=0, le=100)
    max_frontier_escalations: int = Field(default=0, ge=0, le=100)
    max_active_seconds: int = Field(default=900, ge=10, le=28_800)
    # Cumulative wall budget; a session restart does NOT extend this.
    max_wall_seconds: int = Field(default=28_800, ge=60, le=2_592_000)
    # Provider cost cap (micro-units; integer to avoid float drift).
    max_cost_microusd: int = Field(default=0, ge=0, le=10**12)
    grant_expires_at: int = Field(default=0, ge=0, description="epoch seconds; 0 = no expiry")
    # Campaign-specific bounded counters.
    max_chunks: int = Field(default=3, ge=0, le=1000)
    max_families: int = Field(default=1, ge=0, le=100)
    context_token_budget: int = Field(default=8192, ge=256, le=262_144)

    def is_expired(self, now: int) -> bool:
        return self.grant_expires_at > 0 and now > self.grant_expires_at


# ---------------------------------------------------------------------------
# AutonomyGrant
# ---------------------------------------------------------------------------


class AutonomyGrant(StrictBase):
    """A finite, operator-approved plan envelope.

    ``state`` is the lifecycle class of the grant itself (draft/active/revoked/expired).
    A grant in state ``active`` is admitting-eligible. Revoked or expired
    grants MUST be rejected by derived admission.

    Grant authority originates ONLY from the protected runner/operator
    approval channel (``grants.activate_grant``). A model-written grant,
    draft grant, hash, or JSON document alone confers NO authority.
    """

    schema_version: Literal["trio.grant.v1"] = GRANT_KIND
    grant_id: ID
    state: Literal["draft", "active", "revoked", "expired"]
    plan_id: ID
    plan_revision: int = Field(ge=1, le=1_000_000)
    repository_paths: list[str] = Field(default_factory=list)
    allowed_write_paths: list[str] = Field(default_factory=list)
    protected_paths: list[str] = Field(default_factory=list)
    allowed_operations: list[str] = Field(default_factory=list)
    # Identity pins. Admission fails closed when any of these change.
    runtime_digest: SHA256
    model_name: str = Field(min_length=1, max_length=200)
    model_digest: SHA256
    policy_profile_id: ID
    validator_profile_ids: list[ID] = Field(default_factory=list)
    provider_profile_id: ID
    egress_policy_id: ID
    # Operator-side receipts that authorise the grant.
    operator_id: ID
    operator_receipt_digest: SHA256
    # Activation envelope: epoch seconds; 0 = not yet activated.
    activated_at: int = Field(default=0, ge=0)
    revoked_at: int = Field(default=0, ge=0)
    revoked_reason: str = Field(default="", max_length=2000)
    # Budget.
    budget: Budget = Field(default_factory=Budget)

    @model_validator(mode="after")
    def _cross(self) -> "AutonomyGrant":
        if self.state == "active" and self.activated_at == 0:
            raise ValueError("activated_at must be set when state=active")
        if self.state == "revoked" and self.revoked_at == 0:
            raise ValueError("revoked_at must be set when state=revoked")
        return self


# ---------------------------------------------------------------------------
# WorkPackage
# ---------------------------------------------------------------------------


class WorkPackage(StrictBase):
    schema_version: Literal["trio.work-package.v1"] = "trio.work-package.v1"
    package_id: ID
    campaign_id: ID
    title: str = Field(min_length=1, max_length=200)
    goal: str = Field(min_length=1, max_length=8000)
    depends_on: list[ID] = Field(default_factory=list)
    criterion_ids: list[ID] = Field(default_factory=list)
    invariant_ids: list[ID] = Field(default_factory=list)
    state: WorkPackageState = WorkPackageState.PROPOSED
    revision: int = Field(default=1, ge=1, le=1_000_000)
    supersedes: Optional[SHA256] = None


# ---------------------------------------------------------------------------
# ChunkSpec
# ---------------------------------------------------------------------------


class ChunkSpec(StrictBase):
    """A bounded chunk within an in-progress campaign.

    Binds to the exact (post-apply) accepted predecessor snapshot so the
    next chunk's source material is reconstructed from the actual state
    of the campaign integration ref, NOT from a guessed future commit.
    """

    schema_version: Literal["trio.chunk.v1"] = "trio.chunk.v1"
    chunk_id: ID
    campaign_id: ID
    package_id: ID
    parent_chunk_id: Optional[ID] = None
    revision: int = Field(default=1, ge=1, le=1_000_000)
    title: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=8000)
    non_goals: list[str] = Field(default_factory=list, max_length=64)
    accepted_predecessor_snapshot: Optional[RepoSnapshot] = None
    permitted_signature_paths: list[str] = Field(default_factory=list, max_length=4096)
    permitted_write_paths: list[str] = Field(default_factory=list, max_length=4096)
    permitted_read_paths: list[str] = Field(default_factory=list, max_length=4096)
    permitted_command_ids: list[ID] = Field(default_factory=list)
    permitted_validator_ids: list[ID] = Field(default_factory=list)
    required_validator_ids: list[ID] = Field(default_factory=list)
    required_receipt_profiles: list[ID] = Field(default_factory=list)
    criterion_ids: list[ID] = Field(default_factory=list)
    idempotency_key: ID
    state: ChunkState = ChunkState.PROPOSED
    supersedes: Optional[SHA256] = None


# ---------------------------------------------------------------------------
# AdmissionReceipt
# ---------------------------------------------------------------------------


class AdmissionReceipt(StrictBase):
    """Trusted runner-issued admission for a derived chunk.

    Models NEVER issue admission receipts. The runner mints them through
    ``overnight_runner.admission.derive_admission()`` after all
    containment and identity bindings succeed.

    ``fence_generation`` is monotonic per campaign. Subsequent admission
    receipts increment it. A broker write must match the current fence or
    the write is rejected by ``Resources``/``Leases``.
    """

    schema_version: Literal["trio.admission.v1"] = ADMISSION_KIND
    admission_id: ID
    grant_id: ID
    grant_revision: int = Field(ge=1, le=1_000_000)
    chunk_id: ID
    chunk_revision: int = Field(ge=1, le=1_000_000)
    accepted_predecessor_snapshot: RepoSnapshot
    runtime_digest: SHA256
    model_name: str = Field(min_length=1, max_length=200)
    model_digest: SHA256
    policy_profile_id: ID
    validator_profile_ids: list[ID] = Field(default_factory=list)
    provider_profile_id: ID
    worker_id: ID
    budget_ledger_id: ID
    fence_generation: int = Field(ge=1, le=1_000_000_000)
    lease_id: ID
    idempotency_key: ID
    issued_at: int = Field(ge=0)
    issuer: Literal["runner"] = "runner"

    def content_fields(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


# ---------------------------------------------------------------------------
# BudgetLedgerEntry (one per chunk completion)
# ---------------------------------------------------------------------------


class BudgetLedgerEntry(StrictBase):
    """Authoritative cumulative budget ledger for a campaign.

    A single ``BudgetLedger`` (one per campaign) is preserved across
    sessions, restarts, rechunks, and provider changes. The runner
    is the SOLE authoritative source; P04 ``revisions_requested`` and
    worker-reported counters are requests/observations, NOT authority.
    """

    schema_version: Literal["trio.budget-ledger.v1"] = "trio.budget-ledger.v1"
    ledger_id: ID
    campaign_id: ID
    grant_id: ID
    family_id: ID = Field(description="Stable per (plan, work_package) since rechunks should not mint new ledger")
    revision: int = Field(default=1, ge=1, le=1_000_000)
    cumulative_model_calls: int = Field(default=0, ge=0)
    cumulative_tool_calls: int = Field(default=0, ge=0)
    cumulative_repairs: int = Field(default=0, ge=0)
    cumulative_rechunks: int = Field(default=0, ge=0)
    cumulative_escalations: int = Field(default=0, ge=0)
    cumulative_active_seconds: int = Field(default=0, ge=0)
    cumulative_cost_microusd: int = Field(default=0, ge=0)
    cumulative_chunks: int = Field(default=0, ge=0)
    cumulative_context_tokens: int = Field(default=0, ge=0)
    bounds: Budget = Field(default_factory=Budget)

    def would_exceed(self, delta_model_calls: int = 0, delta_tool_calls: int = 0,
                     delta_repair: int = 0, delta_rechunk: int = 0,
                     delta_escalation: int = 0, delta_active_seconds: int = 0,
                     delta_cost_microusd: int = 0, delta_chunks: int = 0,
                     delta_context_tokens: int = 0) -> Optional[str]:
        """Return reason if applying these deltas would exceed bounds.

        Returns ``None`` when within bounds (no exhaustion).
        """
        b = self.bounds
        if self.cumulative_model_calls + delta_model_calls > b.max_model_calls:
            return f"max_model_calls exceeded ({self.cumulative_model_calls + delta_model_calls}>{b.max_model_calls})"
        if self.cumulative_tool_calls + delta_tool_calls > b.max_tool_calls:
            return f"max_tool_calls exceeded ({self.cumulative_tool_calls + delta_tool_calls}>{b.max_tool_calls})"
        if self.cumulative_repairs + delta_repair > b.max_local_repairs:
            return f"max_local_repairs exceeded ({self.cumulative_repairs + delta_repair}>{b.max_local_repairs})"
        if self.cumulative_rechunks + delta_rechunk > b.max_rechunks:
            return f"max_rechunks exceeded ({self.cumulative_rechunks + delta_rechunk}>{b.max_rechunks})"
        if self.cumulative_escalations + delta_escalation > b.max_frontier_escalations:
            return f"max_frontier_escalations exceeded ({self.cumulative_escalations + delta_escalation}>{b.max_frontier_escalations})"
        if self.cumulative_active_seconds + delta_active_seconds > b.max_active_seconds:
            return f"max_active_seconds exceeded ({self.cumulative_active_seconds + delta_active_seconds}>{b.max_active_seconds})"
        if self.cumulative_cost_microusd + delta_cost_microusd > b.max_cost_microusd:
            return f"max_cost_microusd exceeded ({self.cumulative_cost_microusd + delta_cost_microusd}>{b.max_cost_microusd})"
        if self.cumulative_chunks + delta_chunks > b.max_chunks:
            return f"max_chunks exceeded ({self.cumulative_chunks + delta_chunks}>{b.max_chunks})"
        if self.cumulative_context_tokens + delta_context_tokens > b.context_token_budget:
            return f"context_token_budget exceeded ({self.cumulative_context_tokens + delta_context_tokens}>{b.context_token_budget})"
        return None


# ---------------------------------------------------------------------------
# Lease + FenceState
# ---------------------------------------------------------------------------


class Lease(StrictBase):
    """A short-lived, durable lease for a campaign writer.

    Identity is specific (worker process + boot id + lease epoch).
    A second writer attempting to claim while the lease is live MUST
    be rejected. An expired lease that still has a live child MUST
    NOT produce a second writer (P06-A06).
    """

    schema_version: Literal["trio.lease.v1"] = "trio.lease.v1"
    lease_id: ID
    campaign_id: ID
    resource_id: ID = Field(description="e.g. 'campaign_integration_branch' or 'chunk_writer'")
    owner_id: ID = Field(description="Specific owner identity (worker pid + boot id + epoch)")
    owner_boot_id: str = Field(min_length=1, max_length=128)
    owner_pid: int = Field(ge=0)
    fence_generation: int = Field(ge=1)
    acquired_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)
    released_at: int = Field(default=0, ge=0)

    @property
    def is_expired(self) -> bool:
        now = int(time.time())
        return self.expires_at > 0 and now > self.expires_at and self.released_at == 0

    @property
    def is_live(self) -> bool:
        return self.released_at == 0 and not self.is_expired


class FenceState(StrictBase):
    """The campaign-wide fence state. Monotonically incremented when leases
    are revoked and re-issued after expiry. A holder of an older fence
    cannot perform write/integration."""

    schema_version: Literal["trio.fence.v1"] = "trio.fence.v1"
    campaign_id: ID
    current_generation: int = Field(ge=1)
    last_increment_reason: str = Field(default="", max_length=2000)
    updated_at: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Campaign event outbox
# ---------------------------------------------------------------------------


class CampaignEvent(StrictBase):
    """Durable, idempotent outbox record for a campaign state change.

    Repeated delivery of the same event_id is idempotent. Event ids are
    issued by the runner; clients can retry by id without duplication.
    """

    schema_version: Literal["trio.campaign-event.v1"] = EVENT_KIND
    event_id: ID
    campaign_id: ID
    chunk_id: Optional[ID] = None
    event_type: str = Field(min_length=1, max_length=128)
    from_state: Optional[str] = None
    to_state: Optional[str] = None
    actor: str = Field(default="runner", min_length=1, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)
    fence_generation: int = Field(default=1, ge=1)
    issued_at: int = Field(ge=0)
    idempotency_key: ID


# ---------------------------------------------------------------------------
# Campaign + Package record (durable records minted by the runner)
# ---------------------------------------------------------------------------


class CampaignRecord(StrictBase):
    schema_version: Literal["trio.campaign-record.v1"] = "trio.campaign-record.v1"
    campaign_id: ID
    grant_id: ID
    plan_id: ID
    integration_branch: str = Field(min_length=1, max_length=512,
                                    description="e.g. refs/heads/campaign/<id>")
    current_snapshot: Optional[RepoSnapshot] = None
    current_fence: int = Field(default=1, ge=1)
    state: CampaignState = CampaignState.DRAFT
    created_at: int = Field(ge=0)
    updated_at: int = Field(ge=0)
    completed_at: int = Field(default=0, ge=0)


def content_sha256(model: BaseModel) -> str:
    """SHA-256 of canonicalised Pydantic model.

    Uses ``exclude_none=True`` so optional unset fields do not
    contribute to the digest and the same model always produces the
    same digest regardless of which explicit-vs-default fields the
    caller supplied.
    """
    canonical = model.model_dump(mode="json", exclude_none=True)
    return hashlib.sha256(
        __import__("json").dumps(
            canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
            default=lambda o: o.value if hasattr(o, "value") else str(o),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "CampaignState", "ChunkState", "WorkPackageState",
    "GRANT_KIND", "ADMISSION_KIND", "EVENT_KIND",
    "ID", "SHA256",
    "StrictBase",
    "RepoSnapshot",
    "Budget", "BudgetLedgerEntry",
    "AutonomyGrant",
    "WorkPackage", "ChunkSpec",
    "AdmissionReceipt",
    "Lease", "FenceState",
    "CampaignEvent",
    "CampaignRecord",
    "content_sha256",
]
