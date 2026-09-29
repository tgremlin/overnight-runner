"""M5/H1 proposal (b) + (f) — the trusted handler entry point.

ADDITIVE new module. Consumes a `trio.chunk-extension.v1` sidecar and the
versioned `runner-mapping.v1.json` table, and refuses anything the table does not
authorize:

  * the sidecar must re-derive (`verify_extension`) against the Runner-registered
    ``approved_artifact_id`` and the chunk's ``ChunkSpec``;
  * every host action the sidecar names must be in the mapping's ``hostActions``
    (host actions are extension-only: they are never model commands);
  * every validator id it names must be in the mapping's ``validators``.

A verification failure is an UNKNOWN EFFECT, and ``(f)`` is wired here for real:
``unknown_effect_action()`` is what the caller must do (stop), so the error-class
vocabulary is not inert.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from .execution_state import unknown_effect_action
from .plan_verify import verify_extension

__all__ = [
    "HandlerDecision",
    "MAPPING_VERSION",
    "admit_chunk_extension",
    "load_mapping",
]

MAPPING_VERSION = "trio.runner-mapping.v1"


@dataclass(frozen=True)
class HandlerDecision:
    ok: bool
    code: str = ""
    detail: str = ""
    #: What the caller must do next when the decision is not ok (from (f)).
    action: str = ""


def load_mapping(path: str) -> dict[str, Any]:
    """Load and validate the versioned mapping table. Raises on any surprise."""
    with open(path, "r", encoding="utf-8") as fh:
        mapping = json.load(fh)
    if not isinstance(mapping, Mapping) or mapping.get("version") != MAPPING_VERSION:
        raise ValueError(f"not a {MAPPING_VERSION} mapping table: {path}")
    for key in ("hostActions", "validators"):
        if not isinstance(mapping.get(key), list):
            raise ValueError(f"mapping table lacks a {key} list")
    return dict(mapping)


def admit_chunk_extension(
    *,
    sidecar: Any,
    chunk_spec: Any,
    registered_approved_artifact_id: str,
    mapping: Mapping[str, Any],
) -> HandlerDecision:
    """Re-derive the extension and allowlist every effect it names."""
    verdict = verify_extension(sidecar, chunk_spec, registered_approved_artifact_id)
    if not verdict.ok:
        # (f): an unverifiable extension is an unknown effect → stop.
        return HandlerDecision(False, verdict.code, verdict.detail, unknown_effect_action())

    host_actions = sidecar.get("host_actions") or []
    unknown_actions = [a for a in host_actions if a not in set(mapping.get("hostActions", []))]
    if unknown_actions:
        return HandlerDecision(
            False, "HOST_ACTION_NOT_ALLOWED",
            f"host actions not in the mapping table: {unknown_actions}", "stop",
        )

    validators = sidecar.get("validator_ids") or []
    unknown_validators = [v for v in validators if v not in set(mapping.get("validators", []))]
    if unknown_validators:
        return HandlerDecision(
            False, "VALIDATOR_NOT_ALLOWED",
            f"validator ids not in the mapping table: {unknown_validators}", "stop",
        )

    return HandlerDecision(True)
