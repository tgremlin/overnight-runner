"""Runner-owned trusted receipt authority (P05-A06).

Trusted receipts originate here, at the RUNNER boundary, not from
worker-controlled fields. `issued_by` is a label only — it is NOT the trust
signal. The trust comes from the runner's DURABLE EVIDENCE STORE:

  - The runner issues an opaque, non-guessable `receipt_id` per successful
    mutation and persists a durable evidence record under
    `OVERNIGHT_STATE_DIR/receipts/<receipt_id>.json` (the same state dir the
    runner already uses for its lock / PAUSED sentinel).
  - Verification re-reads that durable record and compares it against the
    caller-supplied identity fields. If no durable record exists (a worker
    fabricated a receipt-shaped object with correct-looking hashes) the
    verification FAILS.

The worker NEVER holds any secret and NEVER writes to the durable store:
  - minting is exclusive to runner-owned code (called by the runner at apply
    time);
  - the worker only forwards the opaque receipt reference and asks the
    runner-owned verifier to confirm it against the evidence store.

This satisfies: "do not let the worker process possess the trust key" while
reusing the runner's existing evidence architecture (no new authority, no
secret handed to the worker).
"""
from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path

from .runtime import state_dir


def receipts_dir() -> Path:
    d = state_dir() / "receipts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mint_receipt(
    *,
    proposal_id: str,
    path: str,
    snapshot_digest: str,
    chunk_id: str,
    request_id: str,
    result: dict,
) -> str:
    """Runner-owned mint: issue an opaque receipt ID and persist durable
    evidence. Called only by the runner at apply time. Returns the opaque
    reference the worker may forward (no secret, no capability)."""
    receipt_id = "rec-" + secrets.token_urlsafe(24)
    record = {
        "receipt_id": receipt_id,
        "proposal_id": proposal_id,
        "path": path,
        "snapshot_digest": snapshot_digest,
        "chunk_id": chunk_id,
        "request_id": request_id,
        "issued_at": time.time(),
        "result": {
            "applied": bool(result.get("applied")),
            "op": result.get("op"),
            "pre_sha256": result.get("pre_sha256"),
            "post_sha256": result.get("post_sha256"),
            "bytes_written": result.get("bytes_written"),
        },
    }
    (receipts_dir() / f"{receipt_id}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True)
    )
    return receipt_id


def verify_receipt(
    receipt_id: str,
    *,
    snapshot_digest: str | None = None,
    chunk_id: str | None = None,
    request_id: str | None = None,
    proposal_id: str | None = None,
) -> bool:
    """Runner-owned verification against the durable evidence store. True only
    if the runner actually issued that receipt (a durable record exists) AND
    every supplied identity field matches. A worker-fabricated receipt-shaped
    object fails because no durable record was written by the runner."""
    if not receipt_id or not isinstance(receipt_id, str):
        return False
    if ".." in receipt_id or "/" in receipt_id:
        return False  # path-traversal guard on the opaque id
    p = receipts_dir() / f"{receipt_id}.json"
    if not p.is_file():
        return False
    try:
        rec = json.loads(p.read_text())
    except Exception:
        return False
    checks = [
        (snapshot_digest is not None and rec.get("snapshot_digest") != snapshot_digest),
        (chunk_id is not None and rec.get("chunk_id") != chunk_id),
        (request_id is not None and rec.get("request_id") != request_id),
        (proposal_id is not None and rec.get("proposal_id") != proposal_id),
    ]
    return not any(checks)


__all__ = ["mint_receipt", "verify_receipt", "receipts_dir"]