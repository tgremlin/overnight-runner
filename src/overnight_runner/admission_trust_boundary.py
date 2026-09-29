"""OV4-3 — the admission trust boundary, behind a versioned default-safe flag.

Wires (c) execution-class admission and (e) scope/evidence intake into the REAL
admission path (`admission.derive_admission`). The flag is versioned and DEFAULT
OFF: with the flag absent the Runner behaves exactly as before (no new required
arguments, no new refusals). With the flag ON, an input the boundary cannot
verify is refused rather than admitted.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

from .execution_class import ExecutionClass, admit_execution_class
from .intake import intake_scope_gate, intake_validator_evidence
from .workspace_snapshot import SnapshotEntry

__all__ = [
    "TRUST_BOUNDARY_FLAG",
    "TRUST_BOUNDARY_ENV",
    "TrustBoundaryDecision",
    "apply_trust_boundary",
    "trust_boundary_enabled",
]

TRUST_BOUNDARY_FLAG = "trio.admission-trust-boundary.v1"
TRUST_BOUNDARY_ENV = "TRIO_ADMISSION_TRUST_BOUNDARY"


@dataclass(frozen=True)
class TrustBoundaryDecision:
    ok: bool
    applied: bool = False
    code: str = ""
    detail: str = ""
    execution_class: Any = None
    scope: Any = None
    evidence: Any = None


def trust_boundary_enabled(flag: str | None = None) -> bool:
    """True ONLY for the exact versioned flag value. Default-safe: absent = off."""
    value = flag if flag is not None else os.environ.get(TRUST_BOUNDARY_ENV)
    return value == TRUST_BOUNDARY_FLAG


def apply_trust_boundary(
    *,
    execution_class: Any = None,
    workspace_dir: str | None = None,
    pre_snapshot: Mapping[str, SnapshotEntry] | None = None,
    contract: Mapping[str, Any] | None = None,
    submitted_scope: Any = None,
    submitted_evidence: Any = None,
    recomputed_evidence: Mapping[str, Any] | None = None,
    flag: str | None = None,
) -> TrustBoundaryDecision:
    """Run (c) and (e) when enabled; otherwise do nothing at all."""
    if not trust_boundary_enabled(flag):
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

    if workspace_dir is None or pre_snapshot is None or contract is None:
        return TrustBoundaryDecision(False, applied=True, code="UNWIRED_WORKSPACE", detail="the trust boundary needs a workspace, its pre-admission snapshot and the contract", execution_class=class_decision)

    scope = intake_scope_gate(submitted=submitted_scope, workspace_dir=workspace_dir, pre_snapshot=pre_snapshot, contract=contract)
    if not scope.ok:
        return TrustBoundaryDecision(False, applied=True, code=scope.code, detail=scope.detail, execution_class=class_decision, scope=scope)

    if submitted_evidence is None:
        return TrustBoundaryDecision(False, applied=True, code="UNWIRED_VALIDATOR_EVIDENCE", detail="the trust boundary needs submitted validator evidence to recompute", execution_class=class_decision, scope=scope)
    evidence = intake_validator_evidence(submitted=submitted_evidence, recomputed=recomputed_evidence)
    if not evidence.ok:
        return TrustBoundaryDecision(False, applied=True, code=evidence.code, detail=evidence.detail, execution_class=class_decision, scope=scope, evidence=evidence)

    return TrustBoundaryDecision(True, applied=True, execution_class=class_decision, scope=scope, evidence=evidence)
