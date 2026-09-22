"""P06-A01 — Approved plan registration and chunk authority.

A plan is a deterministic, content-addressed structure that registers
``work_packages`` and ``criteria``. Admission of a chunk requires
that:

  * the plan exists in the durable ``approved_plans`` table;
  * the chunk's ``package_id`` matches one of the plan's
    ``work_packages``;
  * each of the chunk's ``criterion_ids`` is a subset of the union
    of all criterion ids referenced by the matching work package
    (or, if registered as plan-level criteria, the union of plan
    criteria).

This module's ``lookup_plan_criteria`` is the single authority for
chunk-to-plan mapping. A forged, missing, or unknown plan rejects;
foreign/unapproved criteria reject.

P06 follow-up #2 (A12): ``register_plan`` is a TRUSTED OPERATOR /
PLANNER ingestion surface. It is NOT exposed to worker/model tool
surfaces; a worker cannot register or modify plans. The function
returns the deterministic ``plan_digest`` so callers can bind a
grant to the exact plan content.
"""
from __future__ import annotations

import json
from typing import Any

from .db import Database
from .safety import SafetyError

# Module-level marker: this is the only plan-registration surface.
# Worker-facing surfaces MUST NOT call this.
PLAN_TRUSTED_OPERATOR_SURFACE = True


def register_plan(
    db: Database,
    *,
    plan_id: str,
    approved_artifact_id: str,
    work_package_criterion_ids: dict[str, set[str]],
) -> str:
    """Persist a plan with the union of criteria per package.

    ``work_package_criterion_ids`` maps ``package_id`` -> set of
    criterion ids. The plan's ``plan_digest`` is the SHA-256 of the
    canonical payload.

    P06 follow-up #2: This is a TRUSTED operator-side surface. The
    function returns the deterministic ``plan_digest`` so the grant
    activation layer can bind a grant to the exact plan digest.
    Re-registering an existing ``plan_id`` raises ``SafetyError`` to
    preserve content-addressed immutability.
    """
    # Canonicalize to a sorted JSON-serializable form.
    canonical = {
        "plan_id": plan_id,
        "work_packages": [
            {
                "package_id": pkg,
                "criterion_ids": sorted(sorted(crits)),
            }
            for pkg, crits in sorted(work_package_criterion_ids.items())
        ],
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    import hashlib
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    # Refuse to re-register an existing plan_id (content-addressed immutability).
    existing = load_plan_digest(db, plan_id)
    if existing is not None:
        if existing == digest:
            # Idempotent re-register with identical content: return existing digest.
            return digest
        raise SafetyError(
            f"plan {plan_id!r} already registered with a different digest "
            f"(existing={existing[:8]}, attempted={digest[:8]}); refusing to widen"
        )
    with db.transaction() as cur:
        cur.execute(
            """
            INSERT INTO approved_plans (
                plan_id, approved_artifact_id, plan_digest, payload, created_at
            ) VALUES (?, ?, ?, ?, strftime('%s','now'))
            """,
            (plan_id, approved_artifact_id, digest, payload),
        )
    return digest


def load_plan_digest(db: Database, plan_id: str) -> str | None:
    """Return the canonical ``plan_digest`` for ``plan_id`` or ``None``.

    This is the authoritative content binding. Admission consults this
    to refuse any plan whose digest has drifted away from the grant's
    pinned ``approved_plan_digest``.
    """
    cur = db._conn.execute(
        "SELECT plan_digest FROM approved_plans WHERE plan_id=?",
        (plan_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return row["plan_digest"]


def load_plan_artifact_id(db: Database, plan_id: str) -> str | None:
    """Return the APPROVED SOURCE ARTIFACT identity for ``plan_id``.

    P08 (A01): ``approved_artifact_id`` is how the runner records WHICH
    external plan artifact (e.g. the canonical P04 master-plan artifact)
    the reduced package/criterion projection was derived from. It is
    deliberately separate from ``plan_digest``, which is the runner's own
    projection digest and is NOT the full source-plan digest.
    """
    cur = db._conn.execute(
        "SELECT approved_artifact_id FROM approved_plans WHERE plan_id=?",
        (plan_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return row["approved_artifact_id"]


def load_plan(db: Database, plan_id: str) -> dict[str, Any] | None:
    cur = db._conn.execute(
        "SELECT payload FROM approved_plans WHERE plan_id=?",
        (plan_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return json.loads(row["payload"])


def lookup_plan_criteria(
    db: Database, plan_id: str, package_id: str
) -> set[str] | None:
    """Return the criterion ids in ``plan_id``'s ``package_id``.

    Returns ``None`` if the plan or the package is unknown.
    """
    plan = load_plan(db, plan_id)
    if plan is None:
        return None
    for wp in plan["work_packages"]:
        if wp["package_id"] == package_id:
            return set(wp["criterion_ids"])
    return None


def plan_criterion_ids_for_package(
    db: Database, plan_id: str, package_id: str
) -> set[str]:
    """Strict variant that raises on unknown plan/package.

    The admission layer uses this to fail closed on forged plans or
    package_ids the runner has never seen.
    """
    out = lookup_plan_criteria(db, plan_id, package_id)
    if out is None:
        raise SafetyError(
            f"unknown plan_id={plan_id!r} or package_id={package_id!r}"
        )
    return out


def assert_chunk_criteria_approved(
    db: Database, plan_id: str, package_id: str, chunk_criterion_ids: set[str]
) -> None:
    """Verify that every chunk criterion is in the matching plan/package.

    Fail-closed: missing plan/package, OR any foreign criterion,
    raises ``SafetyError``.
    """
    approved = plan_criterion_ids_for_package(db, plan_id, package_id)
    foreign = chunk_criterion_ids - approved
    if foreign:
        raise SafetyError(
            f"unknown package criterion ids: {sorted(foreign)} not approved by plan {plan_id!r} package {package_id!r}"
        )


__all__ = [
    "PLAN_TRUSTED_OPERATOR_SURFACE",
    "register_plan",
    "load_plan",
    "load_plan_digest",
    "load_plan_artifact_id",
    "lookup_plan_criteria",
    "plan_criterion_ids_for_package",
    "assert_chunk_criteria_approved",
]
