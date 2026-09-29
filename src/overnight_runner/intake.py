"""OV4-2 — scope-gate and validator-evidence intake with REAL recomputation.

The Runner computes the change set ITSELF, from a workspace it holds, using the
trusted snapshot/diff port (`workspace_snapshot.py`) and its own scope gate
(`scope_gate.py`). A caller-supplied change list or verdict is never authority.

Fail-closed rules:
  * a submitted verdict or a recomputation with a missing/None digest is a failure
    (`EVIDENCE_DIGEST_MISSING`), on EITHER side;
  * a recomputed refusal is returned as not-ok;
  * a submitted REFUSAL is also returned as not-ok — a refusal is not an
    authorisation, so `ok=True` is never returned for one;
  * a digest disagreement is refused.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .scope_gate import evaluate_scope
from .workspace_snapshot import SnapshotEntry, diff_snapshots, snapshot_workspace

__all__ = [
    "INTAKE_SCHEMA_VERSION",
    "IntakeDecision",
    "intake_scope_gate",
    "intake_validator_evidence",
    "recompute_scope_change_set",
]

INTAKE_SCHEMA_VERSION = "trio.runner-intake.v1"


@dataclass(frozen=True)
class IntakeDecision:
    ok: bool
    code: str = ""
    detail: str = ""
    recomputed: Any = None
    digest: str = ""


def _digest_of(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _usable_digest(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != "" and value not in ("None", "null")


def recompute_scope_change_set(pre: Mapping[str, SnapshotEntry], workspace_dir: str) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    """The Runner's OWN change set: snapshot the workspace it holds and diff it."""
    after = snapshot_workspace(workspace_dir)
    changes = [
        {"kind": c.kind, "path": c.path, "reasons": list(c.reasons), **({"from": c.from_path} if c.from_path else {})}
        for c in diff_snapshots(dict(pre), after)
    ]
    return changes, after


def intake_scope_gate(
    *,
    submitted: Any,
    workspace_dir: str,
    pre_snapshot: Mapping[str, SnapshotEntry],
    contract: Mapping[str, Any],
) -> IntakeDecision:
    """Recompute the scope decision from the workspace; never trust the caller."""
    if not isinstance(submitted, Mapping):
        return IntakeDecision(False, "SCOPE_EVIDENCE_MISSING", "no submitted scope verdict")
    if not _usable_digest(submitted.get("digest")):
        return IntakeDecision(False, "EVIDENCE_DIGEST_MISSING", "the submitted verdict carries no digest")

    changes, after = recompute_scope_change_set(pre_snapshot, workspace_dir)
    ok, refusals = evaluate_scope(contract, changes, after)
    codes = sorted({r.code for r in refusals})
    recomputed = {"ok": ok, "codes": codes, "changes": len(changes)}
    recomputed["digest"] = _digest_of(recomputed)
    if not _usable_digest(recomputed["digest"]):
        return IntakeDecision(False, "EVIDENCE_DIGEST_MISSING", "the recomputation produced no digest")

    if submitted.get("digest") != recomputed["digest"]:
        return IntakeDecision(False, "SCOPE_EVIDENCE_DISAGREES", "the submitted verdict digest does not equal the Runner's recomputation", recomputed=recomputed, digest=recomputed["digest"])
    if not ok:
        return IntakeDecision(False, codes[0] if codes else "SCOPE_REFUSED", "the Runner's recomputation refuses this change set", recomputed=recomputed, digest=recomputed["digest"])
    # §OV4-2d: a submitted REFUSAL is not an authorisation.
    if submitted.get("allowed") is not True:
        return IntakeDecision(False, "SCOPE_SUBMITTED_REFUSAL", "the submitted verdict refuses; a refusal is never an ok=True", recomputed=recomputed, digest=recomputed["digest"])
    return IntakeDecision(True, recomputed=recomputed, digest=recomputed["digest"])


def intake_validator_evidence(
    *,
    submitted: Any,
    recomputed: Mapping[str, Any] | None,
    required_fields: Iterable[str] = ("validator_results",),
) -> IntakeDecision:
    """Compare submitted evidence with the Runner's recomputation."""
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
