"""§PIPEGIT2 G8 — the ACCEPTED-COMMIT LINEAGE rows.

A newly accepted commit must descend from the previous accepted commit of the same
campaign (verified against the pipeline mirror with `git merge-base --is-ancestor`);
otherwise `ACCEPT_LINEAGE_BROKEN`.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from overnight_runner.accept_lineage import (
    ACCEPT_LINEAGE_BROKEN,
    LINEAGE_SCHEMA_VERSION,
    AcceptLineageError,
    campaign_of,
    previous_accepted_commit,
    verify_accept_lineage,
)
from overnight_runner.db import Database


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@e", "-c", "user.name=T", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _mirror(tmp_path: Path) -> tuple[Path, list[str]]:
    """A real bare repo with three linear commits; returns the mirror and their SHAs."""
    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    shas: list[str] = []
    for i in range(3):
        (work / f"f{i}.txt").write_text(f"{i}\n")
        _git(work, "add", ".")
        _git(work, "commit", "-q", "-m", f"c{i}")
        shas.append(_git(work, "rev-parse", "HEAD"))
    mirror = tmp_path / "mirror.git"
    subprocess.run(["git", "clone", "--bare", "-q", str(work), str(mirror)], check=True)
    return mirror, shas


def _campaign(tmp_path: Path) -> tuple[Database, str, list[str]]:
    db = Database(tmp_path / "state.db")
    with db.transaction() as cur:
        cur.execute(
            "INSERT INTO campaigns (campaign_id, grant_id, plan_id, state, integration_branch,"
            " current_commit, current_tree_digest, created_at, updated_at, grant_digest, plan_digest)"
            " VALUES ('C1','G','P','ACTIVE','campaign/C1','a'*40,'b'*40,?,?,'d'*64,'e'*64)",
            (int(time.time()), int(time.time())),
        )
    for i, cid in enumerate(["chunk-1", "chunk-2"], start=1):
        with db.transaction() as cur:
            cur.execute(
                "INSERT INTO chunks (chunk_id, campaign_id, package_id, revision, idempotency_key, state, created_at, updated_at)"
                " VALUES (?, 'C1', 'pkg', 1, ?, 'ADMITTED', ?, ?)",
                (cid, f"idem-{i}", int(time.time()) + i, int(time.time()) + i),
            )
    return db, "C1", ["chunk-1", "chunk-2"]


def test_the_first_accepted_chunk_has_no_predecessor(tmp_path):
    db, campaign, _ = _campaign(tmp_path)
    mirror, shas = _mirror(tmp_path)
    result = verify_accept_lineage(db=db, campaign_id=campaign, chunk_id="chunk-1", accepted_commit=shas[0], mirror=str(mirror))
    assert result.ok and result.previous is None
    assert result.schema_version == LINEAGE_SCHEMA_VERSION


def test_a_descendant_is_accepted_and_reports_its_predecessor(tmp_path):
    db, campaign, _ = _campaign(tmp_path)
    mirror, shas = _mirror(tmp_path)
    with db.transaction() as cur:
        cur.execute("UPDATE chunks SET state='ACCEPTED_FOR_CONTINUATION', snapshot_commit=?, updated_at=? WHERE chunk_id='chunk-1'", (shas[0], int(time.time())))
    assert previous_accepted_commit(db, campaign_id=campaign, exclude_chunk_id="chunk-2") == shas[0]
    result = verify_accept_lineage(db=db, campaign_id=campaign, chunk_id="chunk-2", accepted_commit=shas[2], mirror=str(mirror))
    assert result.ok and result.previous == shas[0]


def test_a_NON_descendant_is_ACCEPT_LINEAGE_BROKEN(tmp_path):
    """The commit exists, but on a DIFFERENT line: refused with the typed code."""
    db, campaign, _ = _campaign(tmp_path)
    mirror, shas = _mirror(tmp_path)
    work = tmp_path / "work"
    _git(work, "checkout", "-q", "-b", "side", shas[0])
    (work / "side.txt").write_text("side\n")
    _git(work, "add", ".")
    _git(work, "commit", "-q", "-m", "side")
    side = _git(work, "rev-parse", "HEAD")
    subprocess.run(["git", "--git-dir", str(mirror), "fetch", "-q", str(work), f"{side}:refs/heads/side"], check=True)
    with db.transaction() as cur:
        cur.execute("UPDATE chunks SET state='ACCEPTED_FOR_CONTINUATION', snapshot_commit=?, updated_at=? WHERE chunk_id='chunk-1'", (shas[2], int(time.time())))
    with pytest.raises(AcceptLineageError) as excinfo:
        verify_accept_lineage(db=db, campaign_id=campaign, chunk_id="chunk-2", accepted_commit=side, mirror=str(mirror))
    assert excinfo.value.code == ACCEPT_LINEAGE_BROKEN
    assert excinfo.value.previous == shas[2]
    assert excinfo.value.accepted == side


def test_an_idempotent_replay_of_the_same_commit_is_not_a_break(tmp_path):
    db, campaign, _ = _campaign(tmp_path)
    mirror, shas = _mirror(tmp_path)
    with db.transaction() as cur:
        cur.execute("UPDATE chunks SET state='ACCEPTED_FOR_CONTINUATION', snapshot_commit=?, updated_at=? WHERE chunk_id='chunk-1'", (shas[1], int(time.time())))
    result = verify_accept_lineage(db=db, campaign_id=campaign, chunk_id="chunk-2", accepted_commit=shas[1], mirror=str(mirror))
    assert result.ok and result.previous == shas[1]


def test_the_campaign_is_resolved_from_the_chunk(tmp_path):
    db, campaign, _ = _campaign(tmp_path)
    assert campaign_of(db, "chunk-1") == campaign
    assert campaign_of(db, "nope") is None
