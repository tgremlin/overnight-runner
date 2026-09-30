"""§PIPEGIT2 G8 — the ACCEPTED-COMMIT LINEAGE check (additive; versioned).

`record_chunk_accepted` stores `accepted_commit`. Until now nothing checked that a
campaign's accepted commits form a CHAIN on the campaign branch. This module adds that
check: a newly accepted commit must be a DESCENDANT of the previous accepted commit of
the SAME campaign, verified with `git merge-base --is-ancestor` run against the pipeline
mirror. A break raises `ACCEPT_LINEAGE_BROKEN`.

It is opt-in and additive: with no `lineage_mirror` supplied nothing changes, so every
existing caller and test behaves exactly as before. The schema version travels with the
result so a caller can record which rule produced it.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

from .safety import SafetyError

__all__ = [
    "ACCEPT_LINEAGE_BROKEN",
    "LINEAGE_SCHEMA_VERSION",
    "AcceptLineageError",
    "LineageResult",
    "campaign_of",
    "previous_accepted_commit",
    "verify_accept_lineage",
]

LINEAGE_SCHEMA_VERSION = "trio.accept-lineage.v1"
ACCEPT_LINEAGE_BROKEN = "ACCEPT_LINEAGE_BROKEN"

#: The state a chunk reaches once its completion gate accepted it.
ACCEPTED_STATE = "ACCEPTED_FOR_CONTINUATION"


class AcceptLineageError(SafetyError):
    """The new accepted commit does not descend from the previous accepted commit."""

    code = ACCEPT_LINEAGE_BROKEN

    def __init__(self, detail: str, *, previous: str | None = None, accepted: str = "") -> None:
        super().__init__(f"{ACCEPT_LINEAGE_BROKEN}: {detail}")
        self.detail = detail
        self.previous = previous
        self.accepted = accepted


@dataclass(frozen=True)
class LineageResult:
    ok: bool
    schema_version: str
    previous: str | None
    accepted: str


def previous_accepted_commit(db: Any, *, campaign_id: str, exclude_chunk_id: str) -> str | None:
    """The most recent OTHER accepted commit in this campaign, or None."""
    with db.transaction() as cur:
        cur.execute(
            """
            SELECT snapshot_commit FROM chunks
            WHERE campaign_id = ? AND state = ? AND chunk_id <> ?
              AND snapshot_commit IS NOT NULL AND snapshot_commit <> ''
            ORDER BY updated_at DESC LIMIT 1
            """,
            (campaign_id, ACCEPTED_STATE, exclude_chunk_id),
        )
        row = cur.fetchone()
    if row is None:
        return None
    value = row[0] if not isinstance(row, dict) else row.get("snapshot_commit")
    return str(value) if value else None


def _is_ancestor(mirror: str, ancestor: str, descendant: str) -> bool:
    proc = subprocess.run(
        ["git", "--git-dir", mirror, "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True, text=True, check=False,
    )
    return proc.returncode == 0


def campaign_of(db: Any, chunk_id: str) -> str | None:
    """The campaign a chunk belongs to."""
    with db.transaction() as cur:
        cur.execute("SELECT campaign_id FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cur.fetchone()
    if row is None:
        return None
    value = row[0] if not isinstance(row, dict) else row.get("campaign_id")
    return str(value) if value else None


def verify_accept_lineage(
    *,
    db: Any,
    campaign_id: str,
    chunk_id: str,
    accepted_commit: str,
    mirror: str,
) -> LineageResult:
    """Verify the chain, or raise `AcceptLineageError`.

    The FIRST accepted chunk of a campaign has no predecessor and passes. Every later one
    must descend from the previous accepted commit ON THE CAMPAIGN BRANCH.
    """
    previous = previous_accepted_commit(db, campaign_id=campaign_id, exclude_chunk_id=chunk_id)
    if previous is None:
        return LineageResult(ok=True, schema_version=LINEAGE_SCHEMA_VERSION, previous=None, accepted=accepted_commit)
    if previous == accepted_commit:
        # the same commit accepted twice (an idempotent replay) is not a break
        return LineageResult(ok=True, schema_version=LINEAGE_SCHEMA_VERSION, previous=previous, accepted=accepted_commit)
    if not _is_ancestor(mirror, previous, accepted_commit):
        raise AcceptLineageError(
            "the accepted commit does not descend from the previous accepted commit of "
            f"this campaign ({previous[:12]} is not an ancestor of {accepted_commit[:12]})",
            previous=previous,
            accepted=accepted_commit,
        )
    return LineageResult(ok=True, schema_version=LINEAGE_SCHEMA_VERSION, previous=previous, accepted=accepted_commit)
