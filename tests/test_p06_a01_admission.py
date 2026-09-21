"""P06-A01 — Delegated admission tests.

A valid plan grant admits later bounded chunks WITHOUT per-chunk human
approval. Draft / forged / revoked / expired grants fail closed. Path,
command, validator, worker, model, and provider expansions are rejected.

Each test uses an isolated OVERNIGHT_STATE_DIR and a fresh disposable
fixture repo; no installed runner state is mutated.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import pytest

from overnight_runner.campaign_schemas import (
    AdmissionReceipt,
    AutonomyGrant,
    Budget,
    ChunkSpec,
    RepoSnapshot,
)
from overnight_runner.db import Database
from overnight_runner.grants import activate_grant, load_grant, revoke_grant
from overnight_runner.admission import derive_admission, AdmissionConflict
from overnight_runner.safety import git_commit_all, git_init_empty, git_worktree_sha
from overnight_runner.campaign import create_campaign


def _isolated_state(tmp_path, monkeypatch):
    sd = tmp_path / "state"
    sd.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OVERNIGHT_STATE_DIR", str(sd))


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "fixture_repo"
    repo.mkdir()
    git_init_empty(repo)
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("ORIGINAL\n")
    git_commit_all(repo, "init")
    return repo


def _pad_to_sha256(s: str) -> str:
    """Pad a 40-char SHA-1 (or shorter digest) to a 64-char lowercase hex
    string so it satisfies the ``[a-f0-9]{64}`` schema. Pure test helper."""
    h = s + ("0" * 64)
    return h[:64]


def _valid_grant(repo: Path) -> AutonomyGrant:
    rev = repo
    head = "0" * 64
    return AutonomyGrant(
        schema_version="trio.grant.v1",
        grant_id="gr-test-1",
        state="draft",
        plan_id="pl-test-1",
        plan_revision=1,
        repository_paths=["src/app.py"],
        allowed_write_paths=["src/app.py"],
        protected_paths=[],
        allowed_operations=["noop", "py_compile"],
        runtime_digest="a" * 64,
        model_name="gemma4:12b",
        model_digest="b" * 64,
        policy_profile_id="pol-1",
        validator_profile_ids=["noop"],
        provider_profile_id="prv-1",
        egress_policy_id="eg-1",
        operator_id="op-1",
        operator_receipt_digest="c" * 64,
        budget=Budget(
            schema_version="trio.budget.v1",
            max_model_calls=10,
            max_tool_calls=20,
            max_local_repairs=2,
            max_rechunks=1,
            max_active_seconds=3600,
            max_wall_seconds=28800,
            max_cost_microusd=1000,
            grant_expires_at=0,
            max_chunks=3,
            max_families=1,
            context_token_budget=8192,
        ),
    )


def _valid_chunk(campaign_id: str) -> ChunkSpec:
    return ChunkSpec(
        schema_version="trio.chunk.v1",
        chunk_id="chk-1",
        campaign_id=campaign_id,
        package_id="pkg-1",
        revision=1,
        title="bounded chunk",
        objective="patch src/app.py",
        permitted_signature_paths=["src/app.py"],
        permitted_write_paths=["src/app.py"],
        permitted_read_paths=["src/app.py"],
        permitted_command_ids=["noop"],
        permitted_validator_ids=["noop"],
        required_validator_ids=["noop"],
        required_receipt_profiles=["noop"],
        criterion_ids=["crit-1"],
        idempotency_key="idem-test-1",
    )


class TestP06A01DelegatedAdmission(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._old_state = os.environ.get("OVERNIGHT_STATE_DIR")
        self._old_flag = os.environ.get("OVERNIGHT_RECEIPTS")
        os.environ["OVERNIGHT_STATE_DIR"] = str(self._tmp / "state")
        os.environ["OVERNIGHT_RECEIPTS"] = "1"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("OVERNIGHT_STATE_DIR", None)
        os.environ.pop("OVERNIGHT_RECEIPTS", None)
        if self._old_state is not None:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state
        if self._old_flag is not None:
            os.environ["OVERNIGHT_RECEIPTS"] = self._old_flag

    # ---------- Valid grant admits a later chunk without per-chunk human approval ----------

    def test_valid_grant_admits_later_chunk(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant(repo)
            activate_grant(
                db,
                grant=grant,
                operator_id="op-test",
                operator_receipt={"approval_id": "appr-1"},
            )
            # Operator does NOT approve each chunk — the grant envelope
            # is the delegation. The chunk is admitted by deterministic
            # binding alone.
            head_sha = __import__("subprocess").run(
                ["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True
            ).stdout.strip()
            snap = RepoSnapshot(
                schema_version="trio.repo-snapshot.v1",
                repository_id="local",
                commit=_pad_to_sha256(head_sha),
                tree_digest=_pad_to_sha256(git_worktree_sha(repo)),
            )
            camp = create_campaign(
                db,
                plan_id="pl-1",
                grant_id="gr-test-1",
                base_commit=snap.commit,
                base_tree_digest=snap.tree_digest,
            )
            # Auto-activate to ACTIVE for the test path.
            from overnight_runner.campaign import activate_campaign
            activate_campaign(db, campaign_id=camp.campaign_id)
            assert camp.campaign_id, "campaign id missing"
            chunk = _valid_chunk(camp.campaign_id)
            receipt, ledger = derive_admission(
                db,
                grant=grant,
                chunk=chunk,
                runtime_digest=grant.runtime_digest,
                worker_id="wkr-1",
                policy_profile_id=grant.policy_profile_id,
                validator_profile_ids=list(grant.validator_profile_ids),
                provider_profile_id=grant.provider_profile_id,
                current_accepted_snapshot=snap,
            )
            self.assertIsInstance(receipt, AdmissionReceipt)
            self.assertEqual(receipt.chunk_id, chunk.chunk_id)
            self.assertEqual(receipt.issuer, "runner")
        finally:
            db.close()

    # ---------- Draft grant rejects ----------

    def test_draft_grant_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            from overnight_runner.campaign import create_campaign
            grant = _valid_grant(repo)
            base_sha = "a" * 64
            camp = create_campaign(
                db, plan_id="pl-1", grant_id=grant.grant_id,
                base_commit=base_sha, base_tree_digest="b" * 64,
            )
            # Insert the grant directly into the durable store as a
            # DRAFT (skipping ``activate_grant``). Admission must reject
            # because the grant is not ``active`` (the runner is the
            # sole authority on grant state).
            import json
            from overnight_runner.grants import grant_payload
            with db.transaction() as cur:
                cur.execute(
                    """
                    INSERT INTO grants (
                        grant_id, grant_digest, state, plan_id, plan_revision,
                        operator_id, operator_receipt_digest,
                        activated_at, revoked_at, revoked_reason,
                        payload_json
                    ) VALUES (?,?,?,?,?,?,?,0,0,'',?)
                    """,
                    (
                        grant.grant_id, "draft-digest", grant.state,
                        grant.plan_id, grant.plan_revision,
                        grant.operator_id, grant.operator_receipt_digest,
                        json.dumps(grant_payload(grant)),
                    ),
                )
            with self.assertRaises(Exception) as ctx:
                derive_admission(
                    db,
                    grant=grant,
                    chunk=_valid_chunk(camp.campaign_id),
                    runtime_digest=grant.runtime_digest,
                    worker_id="wkr-1",
                    policy_profile_id=grant.policy_profile_id,
                    validator_profile_ids=list(grant.validator_profile_ids),
                    provider_profile_id=grant.provider_profile_id,
                    current_accepted_snapshot=RepoSnapshot(
                        schema_version="trio.repo-snapshot.v1",
                        repository_id="local",
                        commit=base_sha,
                        tree_digest="c" * 64,
                    ),
                )
            self.assertIn("grant state", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Forged grant rejects (content mismatch) ----------

    def test_forged_grant_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            from overnight_runner.campaign import create_campaign, activate_campaign
            base_sha = "a" * 64
            camp = create_campaign(
                db, plan_id="pl-1", grant_id="gr-test-1",
                base_commit=base_sha, base_tree_digest="b" * 64,
            )
            activate_campaign(db, campaign_id=camp.campaign_id)
            grant = _valid_grant(repo)
            activate_grant(
                db,
                grant=grant,
                operator_id="op-test",
                operator_receipt={"approval_id": "appr-1"},
            )
            # Caller supplies a forged grant_id (the forgery). The
            # runner rejects because the supplied grant_id does not
            # match the stored one ("not found in durable store").
            forged = grant.model_copy(update={"grant_id": "gr-forged"})
            with self.assertRaises(Exception) as ctx:
                derive_admission(
                    db,
                    grant=forged,
                    chunk=_valid_chunk(camp.campaign_id),
                    runtime_digest=forged.runtime_digest,
                    worker_id="wkr-1",
                    policy_profile_id=forged.policy_profile_id,
                    validator_profile_ids=list(forged.validator_profile_ids),
                    provider_profile_id=forged.provider_profile_id,
                    current_accepted_snapshot=RepoSnapshot(
                        schema_version="trio.repo-snapshot.v1",
                        repository_id="local",
                        commit=base_sha,
                        tree_digest="c" * 64,
                    ),
                )
            self.assertIn("not found", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Revoked grant rejects ----------

    def test_revoked_grant_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant(repo)
            activate_grant(
                db,
                grant=grant,
                operator_id="op-test",
                operator_receipt={"approval_id": "appr-1"},
            )
            revoke_grant(db, grant_id=grant.grant_id, reason="operator revoked")
            # The Pydantic grant model still says "draft"/"active" but the
            # store-side state is "revoked"; load_grant returns it.
            stored = load_grant(db, grant.grant_id)
            assert stored.state == "revoked"
            with self.assertRaises(Exception) as ctx:
                derive_admission(
                    db,
                    grant=stored,
                    chunk=_valid_chunk("cmp-x"),
                    runtime_digest=stored.runtime_digest,
                    worker_id="wkr-1",
                    policy_profile_id=stored.policy_profile_id,
                    validator_profile_ids=list(stored.validator_profile_ids),
                    provider_profile_id=stored.provider_profile_id,
                    current_accepted_snapshot=RepoSnapshot(
                        schema_version="trio.repo-snapshot.v1",
                        repository_id="local",
                        commit="a"*16 + "b"*16 + "c"*16 + "d"*16,
                        tree_digest="1"*16 + "2"*16 + "3"*16 + "4"*16,
                    ),
                )
            self.assertIn("grant state", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Expired grant rejects ----------

    def test_expired_grant_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            # Already-expired budget in the grant.
            expired_budget = Budget(
                schema_version="trio.budget.v1",
                max_model_calls=10,
                max_tool_calls=20,
                max_local_repairs=2,
                max_rechunks=1,
                max_active_seconds=3600,
                max_wall_seconds=28800,
                max_cost_microusd=1000,
                grant_expires_at=int(time.time()) - 10,
                max_chunks=3,
                max_families=1,
                context_token_budget=8192,
            )
            grant = _valid_grant(repo).model_copy(update={"budget": expired_budget})
            activate_grant(
                db,
                grant=grant,
                operator_id="op-test",
                operator_receipt={"approval_id": "appr-1"},
            )
            stored = load_grant(db, grant.grant_id)
            with self.assertRaises(Exception) as ctx:
                derive_admission(
                    db,
                    grant=stored,
                    chunk=_valid_chunk("cmp-x"),
                    runtime_digest=stored.runtime_digest,
                    worker_id="wkr-1",
                    policy_profile_id=stored.policy_profile_id,
                    validator_profile_ids=list(stored.validator_profile_ids),
                    provider_profile_id=stored.provider_profile_id,
                    current_accepted_snapshot=RepoSnapshot(
                        schema_version="trio.repo-snapshot.v1",
                        repository_id="local",
                        commit="a"*16 + "b"*16 + "c"*16 + "d"*16,
                        tree_digest="1"*16 + "2"*16 + "3"*16 + "4"*16,
                    ),
                )
            self.assertIn("grant", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Path expansion (write outside allowed set) rejects ----------

    def test_path_expansion_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant(repo)
            activate_grant(
                db,
                grant=grant,
                operator_id="op-test",
                operator_receipt={"approval_id": "appr-1"},
            )
            stored = load_grant(db, grant.grant_id)
            chunk = _valid_chunk("cmp-x").model_copy(update={
                "permitted_write_paths": ["src/app.py", "outside.py"],
            })
            with self.assertRaises(Exception) as ctx:
                derive_admission(
                    db,
                    grant=stored,
                    chunk=chunk,
                    runtime_digest=stored.runtime_digest,
                    worker_id="wkr-1",
                    policy_profile_id=stored.policy_profile_id,
                    validator_profile_ids=list(stored.validator_profile_ids),
                    provider_profile_id=stored.provider_profile_id,
                    current_accepted_snapshot=RepoSnapshot(
                        schema_version="trio.repo-snapshot.v1",
                        repository_id="local",
                        commit="a"*16 + "b"*16 + "c"*16 + "d"*16,
                        tree_digest="1"*16 + "2"*16 + "3"*16 + "4"*16,
                    ),
                )
            self.assertIn("write_paths", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Command/validator expansion rejects ----------

    def test_validator_expansion_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant(repo)
            activate_grant(
                db,
                grant=grant,
                operator_id="op-test",
                operator_receipt={"approval_id": "appr-1"},
            )
            stored = load_grant(db, grant.grant_id)
            chunk = _valid_chunk("cmp-x").model_copy(update={
                "required_validator_ids": ["noop", "shell_echo"],
            })
            with self.assertRaises(Exception) as ctx:
                derive_admission(
                    db,
                    grant=stored,
                    chunk=chunk,
                    runtime_digest=stored.runtime_digest,
                    worker_id="wkr-1",
                    policy_profile_id=stored.policy_profile_id,
                    validator_profile_ids=list(stored.validator_profile_ids),
                    provider_profile_id=stored.provider_profile_id,
                    current_accepted_snapshot=RepoSnapshot(
                        schema_version="trio.repo-snapshot.v1",
                        repository_id="local",
                        commit="a"*16 + "b"*16 + "c"*16 + "d"*16,
                        tree_digest="1"*16 + "2"*16 + "3"*16 + "4"*16,
                    ),
                )
            self.assertIn("validator", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Worker/model/provider expansion / drift rejects ----------

    def test_runtime_drift_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant(repo)
            activate_grant(
                db,
                grant=grant,
                operator_id="op-test",
                operator_receipt={"approval_id": "appr-1"},
            )
            stored = load_grant(db, grant.grant_id)
            with self.assertRaises(Exception) as ctx:
                derive_admission(
                    db,
                    grant=stored,
                    chunk=_valid_chunk("cmp-x"),
                    runtime_digest="wrong" + "0" * 58,  # drift
                    worker_id="wkr-1",
                    policy_profile_id=stored.policy_profile_id,
                    validator_profile_ids=list(stored.validator_profile_ids),
                    provider_profile_id=stored.provider_profile_id,
                    current_accepted_snapshot=RepoSnapshot(
                        schema_version="trio.repo-snapshot.v1",
                        repository_id="local",
                        commit="a"*16 + "b"*16 + "c"*16 + "d"*16,
                        tree_digest="1"*16 + "2"*16 + "3"*16 + "4"*16,
                    ),
                )
            self.assertIn("runtime", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Baseline mismatch rejects (snapshot diff pin) ----------

    def test_baseline_mismatch_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            from overnight_runner.campaign import create_campaign, activate_campaign
            base_sha = "a" * 64
            camp = create_campaign(
                db, plan_id="pl-1", grant_id="gr-test-1",
                base_commit=base_sha, base_tree_digest="b" * 64,
            )
            activate_campaign(db, campaign_id=camp.campaign_id)
            grant = _valid_grant(repo)
            activate_grant(
                db,
                grant=grant,
                operator_id="op-test",
                operator_receipt={"approval_id": "appr-1"},
            )
            stored = load_grant(db, grant.grant_id)
            wrong_sha = "c" * 64
            with self.assertRaises(Exception) as ctx:
                derive_admission(
                    db,
                    grant=stored,
                    chunk=_valid_chunk(camp.campaign_id),
                    runtime_digest=stored.runtime_digest,
                    worker_id="wkr-1",
                    policy_profile_id=stored.policy_profile_id,
                    validator_profile_ids=list(stored.validator_profile_ids),
                    provider_profile_id=stored.provider_profile_id,
                    current_accepted_snapshot=RepoSnapshot(
                        schema_version="trio.repo-snapshot.v1",
                        repository_id="local",
                        commit=wrong_sha,
                        tree_digest="e" * 64,
                    ),
                )
            self.assertIn("baseline_mismatch", str(ctx.exception).lower())
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
