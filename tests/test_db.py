"""SQLite persistence tests (Phase 2)."""
import json
from pathlib import Path

import pytest

from overnight_runner.db import Database
from overnight_runner.schemas import TaskManifest, TaskStatus, canonical_json


def _tmp_db(tmp_path: Path) -> Database:
    return Database(tmp_path / "state.db")


def _manifest() -> TaskManifest:
    raw = {
        "schema_version": "1.0",
        "task_id": "db-test-1",
        "title": "t",
        "execution_class": "read_only",
        "objective": "o",
        "repo": {"path": "/tmp/x"},
    }
    return TaskManifest.model_validate(raw)


def test_import_and_status(tmp_path: Path):
    db = _tmp_db(tmp_path)
    try:
        m = _manifest()
        sha = "abc123"
        db.upsert_task(m.task_id, sha, canonical_json(m).decode(), TaskStatus.PENDING_APPROVAL)
        rows = db.list_tasks()
        assert len(rows) == 1
        assert rows[0]["task_id"] == "db-test-1"
        assert rows[0]["status"] == "PENDING_APPROVAL"
    finally:
        db.close()


def test_approve_binds_envelope(tmp_path: Path):
    db = _tmp_db(tmp_path)
    try:
        m = _manifest()
        db.upsert_task(m.task_id, "sha", canonical_json(m).decode(), TaskStatus.PENDING_APPROVAL)
        env = {"manifest_sha256": "sha", "approved_repo_head": "HEAD", "approved_runtime_sha256": "rsha"}
        db.approve_task(m.task_id, approved_by="tester", approval_envelope=env)
        t = db.get_task(m.task_id)
        assert t["status"] == "APPROVED"
        assert t["approved_repo_head"] == "HEAD"
        assert t["approved_by"] == "tester"
    finally:
        db.close()


def test_event_log(tmp_path: Path):
    db = _tmp_db(tmp_path)
    try:
        db.emit_event("sess1", "task_imported", task_id="t", details={"k": 1})
        cur = db._conn.execute("SELECT COUNT(*) AS n FROM events")
        assert cur.fetchone()["n"] == 1
    finally:
        db.close()
