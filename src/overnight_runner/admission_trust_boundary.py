"""OV5-1/4 — the versioned trust-boundary flag and the completion gate.

OV5-1: the flag is FAIL-CLOSED. Absent (or empty) = off; the exact versioned
string = on; ANY other value is refused with `UNKNOWN_TRUST_BOUNDARY_FLAG` rather
than being treated as off. Supplying boundary inputs without the flag is refused
too.

OV5-4: the SCOPE and EVIDENCE intake now belong to the COMPLETION/acceptance path
(`check_completion`), not to admission. Execution-class admission stays at
admission time (it is a property of the chunk, not of the finished work).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from .baseline import RunnerBaseline, is_runner_issued
from .execution_class import ExecutionClass, admit_execution_class
from .intake import intake_scope_gate, intake_validator_evidence
from .safety import SafetyError
from .workspace_snapshot import candidate_tree_digest

__all__ = [
    "TRUST_BOUNDARY_ENV",
    "TRUST_BOUNDARY_FLAG",
    "TrustBoundaryDecision",
    "UnknownTrustBoundaryFlag",
    "apply_trust_boundary",
    "check_completion",
    "resolve_trust_boundary",
    "trust_boundary_enabled",
]

TRUST_BOUNDARY_FLAG = "trio.admission-trust-boundary.v1"
TRUST_BOUNDARY_ENV = "TRIO_ADMISSION_TRUST_BOUNDARY"


class UnknownTrustBoundaryFlag(SafetyError):
    """A flag value that is neither absent nor the exact versioned string."""

    code = "UNKNOWN_TRUST_BOUNDARY_FLAG"

    def __init__(self, value: Any) -> None:
        super().__init__(f"UNKNOWN_TRUST_BOUNDARY_FLAG: {value!r} is not {TRUST_BOUNDARY_FLAG!r}; refusing rather than treating it as off")
        self.value = value


@dataclass(frozen=True)
class TrustBoundaryDecision:
    ok: bool
    applied: bool = False
    code: str = ""
    detail: str = ""
    execution_class: Any = None
    scope: Any = None
    evidence: Any = None


def resolve_trust_boundary(flag: str | None = None) -> str:
    """Return 'on' or 'off'; raise on any other non-empty value (fail-closed)."""
    value = flag if flag is not None else os.environ.get(TRUST_BOUNDARY_ENV)
    if value is None or value == "":
        return "off"
    if value == TRUST_BOUNDARY_FLAG:
        return "on"
    raise UnknownTrustBoundaryFlag(value)


def trust_boundary_enabled(flag: str | None = None) -> bool:
    """True only for the exact versioned flag; a foreign value RAISES."""
    return resolve_trust_boundary(flag) == "on"


def apply_trust_boundary(
    *,
    execution_class: Any = None,
    flag: str | None = None,
    inputs_supplied: bool = False,
) -> TrustBoundaryDecision:
    """Execution-class admission at ADMISSION time (OV5-4 keeps only this part)."""
    if inputs_supplied and (flag is None or flag == ""):
        return TrustBoundaryDecision(False, applied=False, code="UNKNOWN_TRUST_BOUNDARY_FLAG", detail="boundary inputs were supplied without the versioned flag")
    try:
        enabled = trust_boundary_enabled(flag)
    except UnknownTrustBoundaryFlag as exc:
        return TrustBoundaryDecision(False, applied=False, code=exc.code, detail=str(exc))
    if not enabled:
        return TrustBoundaryDecision(True, applied=False)
    if execution_class is None:
        return TrustBoundaryDecision(False, applied=True, code="UNWIRED_EXECUTION_CLASS", detail="the trust boundary needs the chunk's execution class")
    try:
        cls = execution_class if isinstance(execution_class, ExecutionClass) else ExecutionClass(execution_class)
    except ValueError:
        return TrustBoundaryDecision(False, applied=True, code="UNKNOWN_EXECUTION_CLASS", detail=f"unknown execution class {execution_class!r}")
    class_decision = admit_execution_class(cls)
    if not class_decision.ok:
        return TrustBoundaryDecision(False, applied=True, code=class_decision.code, detail=class_decision.detail, execution_class=class_decision)
    return TrustBoundaryDecision(True, applied=True, execution_class=class_decision)


def check_completion(
    *,
    workspace_dir: str | None = None,
    baseline: RunnerBaseline | None = None,
    chunk_spec: Any = None,
    grant: Any = None,
    submitted_scope: Any = None,
    submitted_evidence: Any = None,
    runner_evidence: Any = None,
    recomputed_evidence: Mapping[str, Any] | None = None,
    contract_override: Mapping[str, Any] | None = None,
    accepted_tree_digest: str | None = None,
    flag: str | None = None,
) -> TrustBoundaryDecision:
    """The COMPLETION gate: scope + validator-evidence intake (OV5-4).

    Runs only when the versioned flag is on; with the flag absent nothing changes.
    """
    try:
        enabled = trust_boundary_enabled(flag)
    except UnknownTrustBoundaryFlag as exc:
        return TrustBoundaryDecision(False, applied=False, code=exc.code, detail=str(exc))
    if not enabled:
        return TrustBoundaryDecision(True, applied=False)

    if workspace_dir is None or chunk_spec is None or grant is None:
        return TrustBoundaryDecision(False, applied=True, code="UNWIRED_WORKSPACE", detail="completion needs a workspace, the stored ChunkSpec and the grant")
    if not isinstance(baseline, RunnerBaseline) or not is_runner_issued(baseline):
        return TrustBoundaryDecision(False, applied=True, code="UNWIRED_BASELINE", detail="completion needs a Runner-issued baseline")
    # §OV6-2: the Runner computes the tree digest from the workspace IT holds and
    # requires it to equal the accepted digest before anything else is believed.
    computed_tree = candidate_tree_digest(workspace_dir)
    if accepted_tree_digest is not None and computed_tree != accepted_tree_digest:
        return TrustBoundaryDecision(False, applied=True, code="TREE_DIGEST_MISMATCH", detail=f"computed tree {computed_tree[:12]} != accepted {accepted_tree_digest[:12]}")
    scope = intake_scope_gate(
        submitted=submitted_scope, workspace_dir=workspace_dir, baseline=baseline,
        chunk_spec=chunk_spec, grant=grant, contract_override=contract_override,
    )
    if not scope.ok:
        return TrustBoundaryDecision(False, applied=True, code=scope.code, detail=scope.detail, scope=scope)
    evidence = intake_validator_evidence(submitted=submitted_evidence, recomputed=recomputed_evidence, runner_evidence=runner_evidence, current_tree_digest=computed_tree)
    if not evidence.ok:
        return TrustBoundaryDecision(False, applied=True, code=evidence.code, detail=evidence.detail, scope=scope, evidence=evidence)
    return TrustBoundaryDecision(True, applied=True, scope=scope, evidence=evidence)
