# M5/H1 proposal: vendored compiler→Runner projection verification.
#
# Copied VERBATIM from trio-game-forge `python/trio-workers/trio_workers/runner_projection.py`
# (sha256 421f962bad35be20f88ef5f914bf9331c3f9a4c9ccf41ac5b06af6d99df9850a) so the Runner patch series is self-contained: a Runner
# patch cannot import from the compiler repo. The pin above is the audit trail.

"""§C3-B3 / §C3-C — pure compiler→Runner projection helpers.

No IO, no clock, no Runner import, no durable state. Two responsibilities:

* `runner_plan_digest` reproduces the Runner's `plans.register_plan` digest
  EXACTLY (the same canonicalization, key order, separators and criterion-id
  ordering), so the TS adapter and the Runner can be compared three ways.
* `verify_extension` / `verify_projection` are the handler-side re-derivation
  checks: a trusted generic handler (M5/H1) calls them at admission. Any
  mismatch or missing sidecar is an UNKNOWN EFFECT → the returned
  `VerifyResult.ok` is False with a specific `code`; there is never a silent
  default.

`content_digest` mirrors the compiler's `canonical()`/`digestOf()`
(`packages/plan-compiler/src/json.ts`): sorted object keys, no whitespace,
non-ASCII preserved. Record values are JSON scalars/arrays/objects produced by
the compiler (integers only, no floats), so Python `json.dumps` matches
JavaScript `JSON.stringify` for the values these records contain.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "VerifyResult",
    "content_digest",
    "runner_plan_digest",
    "verify_extension",
    "verify_projection",
]


def _canonical(value: Any) -> str:
    if value is None or not isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(_canonical(v) for v in value) + "]"
    keys = sorted(value.keys())
    return "{" + ",".join(f"{json.dumps(k, ensure_ascii=False)}:{_canonical(value[k])}" for k in keys) + "}"


def content_digest(record: Mapping[str, Any]) -> str:
    """sha256 over the compiler-canonical JSON of `record`."""
    return hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()


def runner_plan_digest(plan_id: str, work_package_criterion_ids: Mapping[str, Any]) -> str:
    """Exact reproduction of `overnight_runner.plans.register_plan`'s digest.

    `approved_artifact_id` is NOT part of the payload.
    """
    canonical = {
        "plan_id": plan_id,
        "work_packages": [
            {"package_id": pkg, "criterion_ids": sorted(crits)}
            for pkg, crits in sorted(work_package_criterion_ids.items())
        ],
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    code: str = ""
    detail: str = ""


_OK = VerifyResult(True)


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def verify_extension(sidecar: Any, chunk_spec: Any, registered_approved_artifact_id: str) -> VerifyResult:
    """Re-derive and cross-check a `trio.chunk-extension.v1` sidecar.

    Requires: the sidecar is present and well-formed; its `digest` recomputes
    over the record without `digest`; `candidate_fingerprint` equals the
    Runner-registered `approved_artifact_id`; and its chunk/contract/idempotency
    identities match the `ChunkSpec`. Any failure is an unknown effect.
    """
    if not isinstance(sidecar, Mapping):
        return VerifyResult(False, "EXTENSION_MISSING", "no sidecar for this chunk")
    digest = sidecar.get("digest")
    if not isinstance(digest, str) or digest == "":
        return VerifyResult(False, "EXTENSION_MALFORMED", "sidecar carries no digest")
    without_digest = {k: v for k, v in sidecar.items() if k != "digest"}
    if content_digest(without_digest) != digest:
        return VerifyResult(False, "EXTENSION_DIGEST_MISMATCH", "sidecar digest does not recompute")
    if sidecar.get("candidate_fingerprint") != registered_approved_artifact_id:
        return VerifyResult(
            False, "EXTENSION_ARTIFACT_MISMATCH",
            "sidecar candidate_fingerprint does not equal the registered approved_artifact_id",
        )
    chunk_id = _field(chunk_spec, "chunk_id")
    if sidecar.get("chunk_id") != chunk_id:
        return VerifyResult(False, "EXTENSION_CHUNK_MISMATCH", "sidecar chunk_id does not match the ChunkSpec")
    if sidecar.get("contract_id") != chunk_id:
        return VerifyResult(False, "EXTENSION_CONTRACT_MISMATCH", "sidecar contract_id does not match the ChunkSpec chunk_id")
    if sidecar.get("idempotency_key") != _field(chunk_spec, "idempotency_key"):
        return VerifyResult(False, "EXTENSION_IDEMPOTENCY_MISMATCH", "sidecar idempotency_key does not match the ChunkSpec")
    return _OK


def verify_projection(candidate_projection: Any, grant_approved_plan_digest: str) -> VerifyResult:
    """Re-derive the plan digest from the anchored candidate's projection and
    require it to equal the grant's pinned `approved_plan_digest`.

    This closes the Runner's unchecked-projection gap: `register_plan` does not
    verify the projection against `approved_artifact_id`, so the handler
    re-derives the digest here. A mismatch is an unknown effect.
    """
    if not isinstance(candidate_projection, Mapping):
        return VerifyResult(False, "PROJECTION_MISSING", "no candidate projection supplied")
    plan_id = candidate_projection.get("plan_id")
    wpci = candidate_projection.get("work_package_criterion_ids")
    if not isinstance(plan_id, str) or not isinstance(wpci, Mapping):
        return VerifyResult(False, "PROJECTION_MALFORMED", "projection lacks plan_id/work_package_criterion_ids")
    derived = runner_plan_digest(plan_id, wpci)
    if derived != grant_approved_plan_digest:
        return VerifyResult(
            False, "PROJECTION_DIGEST_MISMATCH",
            "re-derived plan digest does not equal the grant's approved_plan_digest",
        )
    return _OK
