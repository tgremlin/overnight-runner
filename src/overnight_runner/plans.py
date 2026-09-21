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
"""
from __future__ import annotations

import json
from typing import Any

from .db import Database
from .safety import SafetyError


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
    "register_plan",
    "load_plan",
    "lookup_plan_criteria",
    "plan_criterion_ids_for_package",
    "assert_chunk_criteria_approved",
]
