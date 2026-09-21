"""P06-A01 — Protected operator approval surface.

This module is the ONLY path that ``activate_grant`` consults to decide
whether a promotion from ``draft`` to ``active`` is allowed. A
caller-supplied dict (or any model-supplied structure) is treated as
ONLY AN IDENTIFIER CHALLENGE; the actual authority comes from a row
in the ``protected_approvals`` table that was placed there via a
trusted out-of-band channel (e.g. an operator CLI invocation whose
terminal session and OS credentials are independently authenticated).

Why this matters:

The foreman / runner / scheduler / worker MUST NOT be allowed to
construct an "operator approval" themselves. The trust boundary is
here. In a real deployment, the protected surface would be backed by
GPG-signed approval tokens, an HSM, or an operator ACL on a separate
daemon. The fixture here preserves that boundary in the runner's
durable store so the system fails closed when a caller-supplied dict
is presented on its own.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from .campaign_schemas import AutonomyGrant, content_sha256
from .db import Database
from .safety import SafetyError


@dataclass(frozen=True)
class ProtectedApproval:
    approval_id: str
    operation: str
    grant_digest_target: str
    operator_id: str
    operator_receipt_digest: str
    issued_at: int
    consumed_at: int
    payload: dict[str, Any]

    @property
    def is_consumed(self) -> bool:
        """True iff this approval has already been consumed (one-shot)."""
        return self.consumed_at > 0


def register_protected_approval(
    db: Database,
    *,
    approval_id: str,
    operation: str,
    grant_digest_target: str,
    operator_id: str,
    operator_receipt: dict[str, Any],
    issued_at: int | None = None,
) -> ProtectedApproval:
    """Persist a protected approval record. SINGLE-WRITER on the
    ``protected_approvals`` table; ``grant`` activation only consumes
    pre-existing rows. Re-registering an existing ``approval_id``
    raises ``SafetyError``.
    """
    digest = _digest_dict(operator_receipt)
    ts = int(issued_at if issued_at is not None else time.time())
    payload = json.dumps(operator_receipt or {}, sort_keys=True, separators=(",", ":"))
    with db.transaction() as cur:
        cur.execute(
            """
            INSERT INTO protected_approvals (
                approval_id, operation, grant_digest_target,
                operator_id, operator_receipt_digest,
                issued_at, consumed_at, payload
            ) VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (approval_id, operation, grant_digest_target,
             operator_id, digest, ts, payload),
        )
    return ProtectedApproval(
        approval_id=approval_id,
        operation=operation,
        grant_digest_target=grant_digest_target,
        operator_id=operator_id,
        operator_receipt_digest=digest,
        issued_at=ts,
        consumed_at=0,
        payload=operator_receipt or {},
    )


def load_protected_approval(
    db: Database, approval_id: str
) -> ProtectedApproval | None:
    cur = db._conn.execute(
        "SELECT * FROM protected_approvals WHERE approval_id=?",
        (approval_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return ProtectedApproval(
        approval_id=row["approval_id"],
        operation=row["operation"],
        grant_digest_target=row["grant_digest_target"],
        operator_id=row["operator_id"],
        operator_receipt_digest=row["operator_receipt_digest"],
        issued_at=row["issued_at"],
        consumed_at=row["consumed_at"],
        payload=json.loads(row["payload"]) if row["payload"] else {},
    )


def consume_protected_approval(
    db: Database, approval_id: str
) -> ProtectedApproval:
    """Mark an approval as consumed.

    P06 follow-up #2 (A11): one-shot semantics. Re-consuming an
    already-consumed approval raises ``SafetyError``. This prevents
    an attacker who has captured a single approval envelope from
    re-using it to mint multiple grants. The function returns the
    updated approval record.
    """
    now = int(time.time())
    with db.transaction() as cur:
        # Atomic compare-and-set: refuse if consumed_at > 0.
        cur.execute(
            "UPDATE protected_approvals SET consumed_at=? "
            "WHERE approval_id=? AND consumed_at=0",
            (now, approval_id),
        )
        if cur.rowcount == 0:
            existing = load_protected_approval(db, approval_id)
            if existing is None:
                raise SafetyError(f"approval {approval_id} not found")
            raise SafetyError(
                f"approval {approval_id} already consumed at "
                f"{existing.consumed_at}; one-shot consumption enforced"
            )
    ap = load_protected_approval(db, approval_id)
    if ap is None:
        raise SafetyError(f"approval {approval_id} disappeared")
    return ap


def _digest_dict(d: dict[str, Any]) -> str:
    import hashlib
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ProtectedApproval",
    "register_protected_approval",
    "load_protected_approval",
    "consume_protected_approval",
]
