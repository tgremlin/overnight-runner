"""M5/H1 proposal (e) — scope-gate and validator-evidence intake.

ADDITIVE new module. The Runner RECOMPUTES; it never trusts submitted evidence:

  * ``intake_scope_gate`` recomputes the path rule itself from the change set and
    the grant's ``allowed_write_paths``. A submitted PASS that the recomputation
    contradicts is refused (`SCOPE_EVIDENCE_DISAGREES`) — a submitted verdict can
    never widen authority.
  * ``intake_validator_evidence`` requires the recomputed digests to match the
    submitted ones; a disagreement or a missing required field is refused.

Both are fail-closed: an absent or unparsable submission is a refusal, not a
default-allow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

__all__ = ["IntakeDecision", "intake_scope_gate", "intake_validator_evidence"]


@dataclass(frozen=True)
class IntakeDecision:
    ok: bool
    code: str = ""
    detail: str = ""
    recomputed: Any = None


def _recompute_out_of_scope(
    changes: Iterable[Mapping[str, Any]], allowed_write_paths: Iterable[str]
) -> list[str]:
    """The Runner's own path rule: every non-delete change must be allowed."""
    allowed = set(allowed_write_paths)
    out: list[str] = []
    for change in changes:
        path = change.get("path")
        kind = change.get("kind")
        if not isinstance(path, str):
            out.append("<malformed>")
            continue
        if kind in ("added", "modified", "renamed") and path not in allowed:
            out.append(path)
        if kind == "deleted" and path not in allowed:
            out.append(path)
    return sorted(set(out))


def intake_scope_gate(
    *,
    submitted: Any,
    changes: Iterable[Mapping[str, Any]],
    allowed_write_paths: Iterable[str],
) -> IntakeDecision:
    """Recompute the scope decision; a submitted PASS cannot override it."""
    if not isinstance(submitted, Mapping):
        return IntakeDecision(False, "SCOPE_EVIDENCE_MISSING", "no submitted scope verdict")
    if submitted.get("allowed") is not True:
        # a submitted refusal is fine (the Runner only ever tightens)
        return IntakeDecision(True, recomputed={"out_of_scope": []})
    out_of_scope = _recompute_out_of_scope(changes, allowed_write_paths)
    if out_of_scope:
        return IntakeDecision(
            False, "SCOPE_EVIDENCE_DISAGREES",
            f"submitted a PASS but the Runner recomputes out-of-scope writes: {out_of_scope}",
            recomputed={"out_of_scope": out_of_scope},
        )
    return IntakeDecision(True, recomputed={"out_of_scope": []})


def intake_validator_evidence(
    *,
    submitted: Any,
    recomputed: Mapping[str, Any],
    required_fields: Iterable[str] = ("validator_results",),
) -> IntakeDecision:
    """Compare submitted evidence with the Runner's recomputation."""
    if not isinstance(submitted, Mapping):
        return IntakeDecision(False, "EVIDENCE_MISSING", "no submitted validator evidence")
    missing = [f for f in required_fields if f not in submitted]
    if missing:
        return IntakeDecision(False, "EVIDENCE_INCOMPLETE", f"missing required fields: {missing}")
    if submitted.get("digest") != recomputed.get("digest"):
        return IntakeDecision(
            False, "EVIDENCE_DISAGREES",
            "submitted evidence digest does not equal the Runner's recomputation",
            recomputed=dict(recomputed),
        )
    if submitted.get("passed") is True and recomputed.get("passed") is not True:
        return IntakeDecision(
            False, "EVIDENCE_DISAGREES",
            "submitted a PASS the Runner's recomputation does not support",
            recomputed=dict(recomputed),
        )
    return IntakeDecision(True, recomputed=dict(recomputed))
