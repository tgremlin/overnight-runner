"""Runner-owned trusted RECEIPTS module.

This module is the SOLE authority that mints and verifies receipts at the
overnight-runner boundary. There are two receipt kinds, and they are NOT
interchangeable:

  - ``mutation/apply`` receipts are minted at ``Broker.apply_proposal``
    time. They bind to the per-file pre/post sha256 (a single-file
    precondition). They are USEFUL mutation evidence but are NOT the
    P05-A06 trusted VALIDATION receipt.

  - ``validation`` receipts are minted at the runner validator boundary
    (``Worker._finalise``) AFTER a required validator actually ran
    against the ACTUAL post-apply candidate. They bind to the candidate
    snapshot/tree identity (not the per-file pre-mutation hash). These
    ARE the P05-A06 trusted validator receipts.

Receipts live at ``<state_dir>/receipts/<kind>/<receipt_id>.json`` and
are content-addressed via the opaque ``receipt_id``. Verify rejects
unknown / tampered / kind-mismatched ids without crashing.

The worker (P05 trio-workers) holds NO secret and NO durable-store
write capability. It only forwards the opaque ``receipt_id``. This
module is the only entry point that creates valid receipt records.

The single dispatch is ``mint_receipt(kind, ...)`` / ``verify_receipt``;
they share the kind-specific schema below.

Additive / opt-in: existing callers must continue to work without
verification or minting. Both halves check ``OVERNIGHT_RECEIPTS=1`` OR an
explicit ``receipts_dir`` argument, and otherwise are no-ops that DO NOT
produce observable side effects.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Union

from .runtime import state_dir


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

RECEIPT_KIND_VERSION = "overnight_runner.receipts.v1"

KIND_MUTATION_APPLY = "mutation/apply"
KIND_VALIDATION = "validation"
_VALID_KINDS = {KIND_MUTATION_APPLY, KIND_VALIDATION}


# Receipt outcomes are closed for the validation kind.
ValidatorOutcome = Literal["pass", "fail", "blocked", "error", "timeout"]

# Runtime will fall back to env-var only if no dir is supplied. Receipt mint is
# opt-in: ``OVERNIGHT_RECEIPTS=1`` enables; default is disabled (V1 preserved).


# ---------------------------------------------------------------------------
# Schemas (typed; frozen)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MutationReceiptPayload:
    """Captured at ``Broker.apply_proposal`` time (single-file evidence)."""
    schema_version: str = RECEIPT_KIND_VERSION
    kind: Literal["mutation/apply"] = KIND_MUTATION_APPLY
    receipt_id: str = ""
    proposal_id: str = ""
    op: str = ""                           # replace_exact|replace_file|create_file
    path: str = ""
    pre_sha256: str = ""                   # single-file pre-mutation hash
    post_sha256: str = ""                  # single-file post-mutation hash
    bytes_written: int = 0
    candidate_snapshot_digest: str = ""    # captured AFTER apply (tree-wide)
    issued_at: int = 0                     # epoch seconds
    issuer: str = "overnight_runner.broker"


@dataclass(frozen=True)
class ValidationReceiptPayload:
    """Captured at the runner validator boundary (tree-wide)."""
    schema_version: str = RECEIPT_KIND_VERSION
    kind: Literal["validation"] = KIND_VALIDATION
    receipt_id: str = ""
    validator_id: str = ""
    validator_command: str = ""            # command_id (e.g. "pytest_runner_tests")
    validator_profile: str = ""            # profile id (may equal command_id)
    candidate_snapshot_digest: str = ""    # git_worktree_sha of the candidate
    candidate_tree_state: str = ""         # "post-apply" | "pre-apply"
    proposal_id: str = ""
    chunk_id: str = ""
    request_id: str = ""
    patch_ref: str = ""                    # relative path to patch artifact
    patch_sha256: str = ""                 # digest of patch text (if available)
    exit_code: int = 0
    signal_name: str = ""
    timed_out: bool = False
    outcome: ValidatorOutcome = "error"    # pass|fail|blocked|error|timeout
    detail: str = ""
    raw_artifact_ref: str = ""             # path under artifact_dir for stdout/stderr
    env_digest: str = ""                   # runtime fingerprint
    profile_digest: str = ""
    issued_at: int = 0
    issuer: str = "overnight_runner.validator_boundary"


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_ENV_ENABLED = "OVERNIGHT_RECEIPTS"

# Hard cap on receipt size to avoid runaway detail strings (defence in depth).
_MAX_DETAIL_BYTES = 4096
_MAX_PATH_LEN = 1024
_MAX_ID_LEN = 256


def _receipts_root(receipts_dir: Path | None = None) -> Path | None:
    """Where durable receipt records live. ``None`` => receipts disabled."""
    if receipts_dir is not None:
        return receipts_dir
    if os.environ.get(_ENV_ENABLED) == "1":
        return state_dir() / "receipts"
    return None


def _enabled(receipts_dir: Path | None = None) -> bool:
    return _receipts_root(receipts_dir) is not None


def _slug(prefix: str) -> str:
    """Opaque, attacker-uncontrollable id. ``prefix-<token>``."""
    safe = "".join(ch for ch in prefix if ch.isalnum() or ch in "_-")
    if not safe:
        safe = "rec"
    return f"{safe}-{secrets.token_urlsafe(18)}"


def _safe_id(receipt_id: str) -> Path | None:
    """Refuse path traversal or shell-flavoured chars in an id.

    Returns ``receipt_id`` on accept, ``None`` on reject.
    """
    if not receipt_id or len(receipt_id) > _MAX_ID_LEN:
        return None
    if "/" in receipt_id or "\\" in receipt_id or ".." in receipt_id or receipt_id.startswith("."):
        return None
    for ch in receipt_id:
        if not (ch.isalnum() or ch in "_-."):
            return None
    return receipt_id


def _truncate(s: str, n: int) -> str:
    if s is None:
        return ""
    if len(s) <= n:
        return s
    return s[: max(0, n - 3)] + "..."


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON to a tmp file in the same dir, then os.replace."""
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".overnight-rec-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Mint (entry point for the runner boundary)
# ---------------------------------------------------------------------------

def mint_receipt(
    *,
    kind: str,
    receipts_dir: Path | None = None,
    payload: dict[str, Any],
) -> str:
    """Persist a new receipt of the given kind.

    Returns the opaque ``receipt_id`` if minting is enabled; returns
    ``""`` (no observable effect) when disabled so callers can be
    unconditional without changing V1 behavior.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"unknown receipt kind: {kind!r}")
    root = _receipts_root(receipts_dir)
    if root is None:
        return ""

    payload = dict(payload)
    payload["schema_version"] = payload.get("schema_version", RECEIPT_KIND_VERSION)
    payload["kind"] = kind
    payload["issued_at"] = int(payload.get("issued_at") or time.time())
    payload.setdefault("issuer",
                       "overnight_runner.broker" if kind == KIND_MUTATION_APPLY
                       else "overnight_runner.validator_boundary")
    if not payload.get("receipt_id"):
        payload["receipt_id"] = _slug("rec-mut" if kind == KIND_MUTATION_APPLY else "rec-val")

    # Bound dangerous string fields.
    for key in ("detail",):
        if key in payload and isinstance(payload[key], str):
            payload[key] = _truncate(payload[key], _MAX_DETAIL_BYTES)
    if "path" in payload and isinstance(payload["path"], str):
        payload["path"] = _truncate(payload["path"], _MAX_PATH_LEN)

    rid = payload["receipt_id"]
    if _safe_id(rid) is None:
        # Defensive: reforge a safe id if the caller passed something we
        # would refuse to verify. We choose not to raise here so the
        # boundary is forgiving and the durable id is always safe.
        payload["receipt_id"] = _slug("rec-mut" if kind == KIND_MUTATION_APPLY else "rec-val")
        rid = payload["receipt_id"]

    kind_dir = root / kind
    out_path = kind_dir / f"{rid}.json"
    # Refuse to clobber an existing record (immutable ids).
    if out_path.exists():
        raise FileExistsError(f"receipt already exists: {rid}")
    _atomic_write_json(out_path, payload)
    return rid


# ---------------------------------------------------------------------------
# Convenience wrappers (kind-specific typed APIs)
# ---------------------------------------------------------------------------

def mint_mutation_receipt(
    *,
    proposal_id: str,
    path: str,
    op: str,
    pre_sha256: str,
    post_sha256: str,
    bytes_written: int,
    candidate_snapshot_digest: str = "",
    receipts_dir: Path | None = None,
) -> str:
    payload: dict[str, Any] = {
        "proposal_id": proposal_id,
        "path": path,
        "op": op,
        "pre_sha256": pre_sha256,
        "post_sha256": post_sha256,
        "bytes_written": bytes_written,
        "candidate_snapshot_digest": candidate_snapshot_digest,
    }
    return mint_receipt(kind=KIND_MUTATION_APPLY, receipts_dir=receipts_dir, payload=payload)


def mint_validation_receipt(
    *,
    validator_id: str,
    validator_command: str,
    validator_profile: str = "",
    candidate_snapshot_digest: str,
    candidate_tree_state: str = "post-apply",
    proposal_id: str = "",
    chunk_id: str = "",
    request_id: str = "",
    patch_ref: str = "",
    patch_sha256: str = "",
    exit_code: int = 0,
    signal_name: str = "",
    timed_out: bool = False,
    outcome: ValidatorOutcome = "pass",
    detail: str = "",
    raw_artifact_ref: str = "",
    env_digest: str = "",
    profile_digest: str = "",
    receipts_dir: Path | None = None,
) -> str:
    if outcome not in ("pass", "fail", "blocked", "error", "timeout"):
        raise ValueError(f"invalid validation outcome: {outcome!r}")
    if not candidate_snapshot_digest:
        raise ValueError("validation receipt requires candidate_snapshot_digest")
    payload: dict[str, Any] = {
        "validator_id": validator_id,
        "validator_command": validator_command,
        "validator_profile": validator_profile or validator_command,
        "candidate_snapshot_digest": candidate_snapshot_digest,
        "candidate_tree_state": candidate_tree_state,
        "proposal_id": proposal_id,
        "chunk_id": chunk_id,
        "request_id": request_id,
        "patch_ref": patch_ref,
        "patch_sha256": patch_sha256,
        "exit_code": int(exit_code),
        "signal_name": signal_name,
        "timed_out": bool(timed_out),
        "outcome": outcome,
        "detail": detail,
        "raw_artifact_ref": raw_artifact_ref,
        "env_digest": env_digest,
        "profile_digest": profile_digest,
    }
    return mint_receipt(kind=KIND_VALIDATION, receipts_dir=receipts_dir, payload=payload)


# ---------------------------------------------------------------------------
# Verify (entry point for the runner boundary)
# ---------------------------------------------------------------------------

def verify_receipt(
    receipt_id: str,
    *,
    receipts_dir: Path | None = None,
    expected_kind: str | None = None,
    snapshot_digest: str | None = None,
    candidate_snapshot_digest: str | None = None,
    validator_id: str | None = None,
    validator_command: str | None = None,
    chunk_id: str | None = None,
    request_id: str | None = None,
    proposal_id: str | None = None,
    patch_sha256: str | None = None,
    outcome: ValidatorOutcome | None = None,
) -> bool:
    """Runner-owned strict verification.

    Strict verification semantics:
      - Unknown / tampered / kind-mismatched ids return ``False`` (no crash).
      - If ``expected_kind`` is supplied, the record must match.
      - All non-None binding fields must match exactly.

    Returns ``False`` when receipts are disabled (a worker trying to
    forward an id from outside the runtime must never be told the id
    is valid against an empty store).
    """
    if _safe_id(receipt_id) is None:
        return False
    root = _receipts_root(receipts_dir)
    if root is None:
        # Receipts disabled: no record can be verified, even if the id
        # looks right. This protects against worker-side forgery attempts.
        return False

    if expected_kind is not None and expected_kind not in _VALID_KINDS:
        return False

    # Search both kind dirs; the opaque id alone doesn't carry the kind.
    record: dict[str, Any] | None = None
    found_kind: str | None = None
    for kind in (KIND_VALIDATION, KIND_MUTATION_APPLY):
        p = root / kind / f"{receipt_id}.json"
        r = _read_json(p)
        if r is not None:
            record = r
            found_kind = kind
            break

    if record is None:
        return False

    if expected_kind is not None and found_kind != expected_kind:
        # Critical: a mutation/apply receipt MUST NOT satisfy a
        # required validation-receipt verification (and vice versa).
        return False

    if found_kind == KIND_VALIDATION:
        if validator_id is not None and record.get("validator_id") != validator_id:
            return False
        if validator_command is not None and record.get("validator_command") != validator_command:
            return False
        if candidate_snapshot_digest is not None:
            if record.get("candidate_snapshot_digest") != candidate_snapshot_digest:
                return False
        # For backward-compat the forge worker sometimes passes the
        # ``snapshot_digest=`` keyword: in validation context that maps to
        # ``candidate_snapshot_digest``.
        if snapshot_digest is not None:
            if record.get("candidate_snapshot_digest") != snapshot_digest:
                return False
        if chunk_id is not None and record.get("chunk_id") != chunk_id:
            return False
        if request_id is not None and record.get("request_id") != request_id:
            return False
        if proposal_id is not None and record.get("proposal_id") != proposal_id:
            return False
        if patch_sha256 is not None and record.get("patch_sha256") != patch_sha256:
            return False
        if outcome is not None and record.get("outcome") != outcome:
            return False
    elif found_kind == KIND_MUTATION_APPLY:
        # For mutation/apply records, only the ``proposal_id`` and the
        # single-file ``snapshot_digest`` (pre_sha256) binding make sense.
        if proposal_id is not None and record.get("proposal_id") != proposal_id:
            return False
        if snapshot_digest is not None and record.get("pre_sha256") != snapshot_digest:
            return False
        if chunk_id is not None:
            # worker-supplied chunk_id MUST NOT satisfy a mutation/apply
            # record (it carries no chunk_id field).
            return False
        if request_id is not None:
            return False
        if candidate_snapshot_digest is not None:
            if record.get("candidate_snapshot_digest") != candidate_snapshot_digest:
                return False
        if validator_id is not None or validator_command is not None:
            return False
        if outcome is not None:
            return False
    return True


def load_receipt(
    receipt_id: str,
    *,
    receipts_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Load a record (no verification). Returns ``None`` when missing."""
    if _safe_id(receipt_id) is None:
        return None
    root = _receipts_root(receipts_dir)
    if root is None:
        return None
    for kind in (KIND_VALIDATION, KIND_MUTATION_APPLY):
        p = root / kind / f"{receipt_id}.json"
        r = _read_json(p)
        if r is not None:
            r["_kind"] = kind
            return r
    return None


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def receipts_enabled(receipts_dir: Path | None = None) -> bool:
    return _enabled(receipts_dir)


__all__ = [
    "RECEIPT_KIND_VERSION",
    "KIND_MUTATION_APPLY", "KIND_VALIDATION",
    "ValidatorOutcome",
    "MutationReceiptPayload", "ValidationReceiptPayload",
    "mint_receipt", "mint_mutation_receipt", "mint_validation_receipt",
    "verify_receipt", "load_receipt", "receipts_enabled",
]
