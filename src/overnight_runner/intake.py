"""OV5-2/3 — intake with Runner-HELD baseline and contract.

The Runner computes the change set from a workspace it holds, against a baseline
IT issued (`baseline.py`), under a contract IT derives from the stored `ChunkSpec`
and `AutonomyGrant` — never from a caller. A caller-supplied snapshot, contract or
change list is either impossible (no such parameter) or explicitly ignored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

INTAKE_SCHEMA_VERSION = "trio.runner-intake.v1"


@dataclass(frozen=True)
class IntakeDecision:
    ok: bool
    code: str = ""
    detail: str = ""
    recomputed: Any = None
    digest: str = ""

from .baseline import RunnerBaseline, is_runner_issued
from .evidence_util import digest_of as _digest_of
from .evidence_util import usable_digest as _usable_digest
from .scope_gate import evaluate_scope
from .workspace_snapshot import SnapshotEntry, diff_snapshots, snapshot_workspace

__all__ = [
    "INTAKE_SCHEMA_VERSION",
    "IntakeDecision",
    "intake_scope_gate",
    "intake_validator_evidence",
    "recompute_scope_change_set",
    "scope_contract_from",
    "submitted_verdict_digest",
]


def scope_contract_from(*, chunk_spec: Any, grant: Any) -> dict[str, Any]:
    """Derive the scope contract from the STORED chunk spec and grant.

    The allowed writes are the INTERSECTION of the grant's and the chunk's
    permitted writes (a chunk may never widen the grant), and the protections come
    from the grant. A caller cannot add or widen a path here.
    """
    grant_allowed = set(getattr(grant, "allowed_write_paths", []) or [])
    chunk_permitted = list(getattr(chunk_spec, "permitted_write_paths", []) or [])
    allowed = [p for p in chunk_permitted if p in grant_allowed]
    return {
        "allowedWritePaths": allowed,
        "protectedPaths": list(getattr(grant, "protected_paths", []) or []),
        "forbiddenScope": [],
        "licensedPathPrefixes": None,
        "caseSensitive": True,
    }


def _usable_digest_or_none(value: Any) -> bool:
    return _usable_digest(value)


def recompute_scope_change_set(baseline: RunnerBaseline, workspace_dir: str) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    """The Runner's OWN change set, against the baseline IT issued."""
    after = snapshot_workspace(workspace_dir)
    changes = [
        {"kind": c.kind, "path": c.path, "reasons": list(c.reasons), **({"from": c.from_path} if c.from_path else {})}
        for c in diff_snapshots(dict(baseline.entries), after)
    ]
    return changes, after


def submitted_verdict_digest(decision_ok: bool, codes: Iterable[str], changes: int) -> str:
    """The digest a truthful submitter would produce (the Runner recomputes it)."""
    return _digest_of({"ok": decision_ok, "codes": sorted(set(codes)), "changes": changes})


def intake_scope_gate(
    *,
    submitted: Any,
    workspace_dir: str,
    baseline: RunnerBaseline,
    chunk_spec: Any,
    grant: Any,
    contract_override: Mapping[str, Any] | None = None,
) -> IntakeDecision:
    """Recompute the scope decision against a Runner-held baseline and contract.

    `contract_override` exists only to be REFUSED as an attention check: a caller
    cannot widen the contract, so supplying one is an error, not a silent ignore.
    """
    if contract_override is not None:
        return IntakeDecision(False, "CONTRACT_OVERRIDE_REFUSED", "the scope contract is derived from the stored ChunkSpec and grant; a caller may not supply one")
    if not isinstance(baseline, RunnerBaseline) or not is_runner_issued(baseline):
        return IntakeDecision(False, "BASELINE_NOT_TRUSTED", "the baseline was not issued by the Runner")
    if not isinstance(submitted, Mapping):
        return IntakeDecision(False, "SCOPE_EVIDENCE_MISSING", "no submitted scope verdict")
    if not _usable_digest_or_none(submitted.get("digest")):
        return IntakeDecision(False, "EVIDENCE_DIGEST_MISSING", "the submitted verdict carries no digest")

    contract = scope_contract_from(chunk_spec=chunk_spec, grant=grant)
    changes, after = recompute_scope_change_set(baseline, workspace_dir)
    ok, refusals = evaluate_scope(contract, changes, after)
    codes = sorted({r.code for r in refusals})
    # the digest covers the DECISION only; metadata (the baseline source) is not
    # part of what a submitter signs, so both sides can compute it identically.
    body = {"ok": ok, "codes": codes, "changes": len(changes)}
    recomputed = {**body, "baseline_source": baseline.source, "digest": _digest_of(body)}
    if not _usable_digest_or_none(recomputed["digest"]):
        return IntakeDecision(False, "EVIDENCE_DIGEST_MISSING", "the recomputation produced no digest")

    if submitted.get("digest") != recomputed["digest"]:
        return IntakeDecision(False, "SCOPE_EVIDENCE_DISAGREES", "the submitted verdict digest does not equal the Runner's recomputation", recomputed=recomputed, digest=recomputed["digest"])
    if not ok:
        return IntakeDecision(False, codes[0] if codes else "SCOPE_REFUSED", "the Runner's recomputation refuses this change set", recomputed=recomputed, digest=recomputed["digest"])
    if submitted.get("allowed") is not True:
        return IntakeDecision(False, "SCOPE_SUBMITTED_REFUSAL", "the submitted verdict refuses; a refusal is never ok=True", recomputed=recomputed, digest=recomputed["digest"])
    return IntakeDecision(True, recomputed=recomputed, digest=recomputed["digest"])


def intake_validator_evidence(
    *,
    submitted: Any,
    recomputed: Mapping[str, Any] | None = None,
    required_fields: Iterable[str] = ("validator_results",),
    runner_evidence: Any = None,
    current_tree_digest: str | None = None,
) -> IntakeDecision:
    """Compare submitted evidence with the Runner's recomputation.

    When `runner_evidence` is supplied the Runner's OWN validator run is
    authoritative (OV5-5) and the caller's `recomputed` is ignored entirely.
    """
    if runner_evidence is not None:
        from .validator_evidence import intake_runner_evidence

        return intake_runner_evidence(submitted=submitted, runner_evidence=runner_evidence, current_tree_digest=current_tree_digest)
    if not isinstance(submitted, Mapping):
        return IntakeDecision(False, "EVIDENCE_MISSING", "no submitted validator evidence")
    if not isinstance(recomputed, Mapping):
        return IntakeDecision(False, "EVIDENCE_DIGEST_MISSING", "the Runner has no recomputation to compare against")
    missing = [f for f in required_fields if f not in submitted]
    if missing:
        return IntakeDecision(False, "EVIDENCE_INCOMPLETE", f"missing required fields: {missing}")
    if not _usable_digest(submitted.get("digest")) or not _usable_digest(recomputed.get("digest")):
        return IntakeDecision(False, "EVIDENCE_DIGEST_MISSING", "a digest is missing on one side of the comparison")
    if submitted.get("digest") != recomputed.get("digest"):
        return IntakeDecision(False, "EVIDENCE_DISAGREES", "submitted evidence digest does not equal the Runner's recomputation", digest=str(recomputed.get("digest")))
    if submitted.get("passed") is True and recomputed.get("passed") is not True:
        return IntakeDecision(False, "EVIDENCE_DISAGREES", "submitted a PASS the Runner's recomputation does not support", digest=str(recomputed.get("digest")))
    if submitted.get("passed") is not True:
        return IntakeDecision(False, "EVIDENCE_SUBMITTED_REFUSAL", "the submitted evidence refuses; a refusal is never ok=True", digest=str(recomputed.get("digest")))
    return IntakeDecision(True, digest=str(recomputed.get("digest")))
