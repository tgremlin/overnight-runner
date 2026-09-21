"""P06 follow-up #5 — very narrow final authority checks.

  1. compare_and_swap_advance enforces the FULL campaign continuation
     authority (active/non-revoked/non-expired grant + trusted budget
     headroom + blocked state + PAUSED) by reusing
     check_campaign_continuation.
  2. the admission lease must be campaign-scoped, unreleased, and
     UNEXPIRED, with lease fence == admission fence == campaign fence.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from overnight_runner.admission import derive_admission
from overnight_runner.campaign import (
    activate_campaign,
    create_campaign,
    update_budget_after_chunk,
)
from overnight_runner.campaign_schemas import (
    AutonomyGrant,
    Budget,
    ChunkSpec,
    RepoSnapshot,
    content_sha256,
)
from overnight_runner.db import Database
from overnight_runner.grants import activate_grant, load_grant, revoke_grant
from overnight_runner.integration import (
    compare_and_swap_advance,
    ensure_campaign_worktree,
)
from overnight_runner.plans import load_plan_digest, register_plan
from overnight_runner.protected_approvals import register_protected_approval
from overnight_runner.receipts import _receipts_root, mint_validation_receipt
from overnight_runner.safety import (
    SafetyError,
    git_commit_all,
    git_head,
    git_init_empty,
    git_worktree_sha,
)


def _isolated_setup(tmp_path: Path) -> Path:
    sd = tmp_path / "state"
    sd.mkdir(parents=True, exist_ok=True)
    os.environ["OVERNIGHT_STATE_DIR"] = str(sd)
    os.environ["OVERNIGHT_RECEIPTS"] = "1"
    return sd / "state.db"


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "fixture_repo"
    repo.mkdir()
    git_init_empty(repo)
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("BASE\n")
    git_commit_all(repo, "init")
    return repo


def _grant(db, *, plan_id, grant_id, grant_expires_at=0, max_chunks=3):
    packages = {"pkg-1": {"crit-1", "crit-2"}}
    grant = AutonomyGrant(
        schema_version="trio.grant.v1",
        grant_id=grant_id, state="draft",
        plan_id=plan_id, plan_revision=1, approved_plan_digest="0" * 64,
        repository_paths=["src/app.py"], allowed_write_paths=["src/app.py"],
        protected_paths=[], allowed_operations=["noop"],
        runtime_digest="a" * 64, model_name="gemma", model_digest="b" * 64,
        policy_profile_id="pol-1", validator_profile_ids=["noop"],
        provider_profile_id="prv-1", egress_policy_id="eg-1",
        operator_id="op-1", operator_receipt_digest="c" * 64,
        budget=Budget(schema_version="trio.budget.v1", max_chunks=max_chunks,
                      max_model_calls=10, max_tool_calls=20, max_local_repairs=2,
                      max_rechunks=1, max_active_seconds=3600,
                      max_wall_seconds=28800, max_cost_microusd=1000,
                      grant_expires_at=grant_expires_at, context_token_budget=8192),
    )
    register_plan(db, plan_id=plan_id, approved_artifact_id=plan_id,
                  work_package_criterion_ids=packages)
    grant = grant.model_copy(update={"approved_plan_digest": load_plan_digest(db, plan_id)})
    register_protected_approval(
        db, approval_id=f"appr-{grant_id}", operation="activate_grant",
        grant_digest_target=content_sha256(grant), operator_id="op-1",
        operator_receipt={"approval_id": f"appr-{grant_id}"})
    activate_grant(db, grant=grant, operator_id="op-1", approval_id=f"appr-{grant_id}")
    return load_grant(db, grant_id)


def _id_kwargs(grant) -> dict:
    return dict(
        current_runtime_digest=grant.runtime_digest,
        current_model_name=grant.model_name,
        current_model_digest=grant.model_digest,
        current_policy_profile_id=grant.policy_profile_id,
        current_validator_profile_ids=list(grant.validator_profile_ids),
        current_provider_profile_id=grant.provider_profile_id,
    )


def _snap(commit="a" * 64, tree="b" * 64) -> RepoSnapshot:
    return RepoSnapshot(schema_version="trio.repo-snapshot.v1",
                        repository_id="local", commit=commit, tree_digest=tree)


def _chunk(campaign_id, *, chunk_id="chk-1"):
    return ChunkSpec(
        schema_version="trio.chunk.v1",
        chunk_id=chunk_id, campaign_id=campaign_id, package_id="pkg-1",
        revision=1, title="c", objective="c",
        permitted_signature_paths=["src/app.py"],
        permitted_write_paths=["src/app.py"],
        permitted_read_paths=["src/app.py"],
        permitted_command_ids=["noop"], permitted_validator_ids=["noop"],
        required_validator_ids=["noop"], required_receipt_profiles=["noop"],
        criterion_ids=["crit-1"], idempotency_key=f"idem-{chunk_id}")


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))
        self._repo = _make_repo(self._tmp)
        self._base = git_head(self._repo) + "0" * 24

    def tearDown(self):
        for name in list(os.environ):
            if name.startswith("TR_FAILPOINT_"):
                os.environ.pop(name, None)
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _campaign_and_admit(self, *, plan_id, grant_id, chunk_id="chk-1",
                            grant_expires_at=0, max_chunks=3):
        grant = _grant(self._db, plan_id=plan_id, grant_id=grant_id,
                       grant_expires_at=grant_expires_at, max_chunks=max_chunks)
        camp = create_campaign(self._db, plan_id=grant.plan_id,
                               grant_id=grant.grant_id, base_commit=self._base,
                               base_tree_digest="b" * 64, repo_root=str(self._repo))
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        wt = ensure_campaign_worktree(repo_root=self._repo,
                                      campaign_id=camp.campaign_id,
                                      base_commit=self._base, db=self._db)
        cur_commit = self._db._conn.execute(
            "SELECT current_commit FROM campaigns WHERE campaign_id=?",
            (camp.campaign_id,)).fetchone()["current_commit"]
        receipt, _ = derive_admission(
            self._db, grant=grant, chunk=_chunk(camp.campaign_id, chunk_id=chunk_id),
            worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            current_accepted_snapshot=_snap(commit=cur_commit),
            **_id_kwargs(grant))
        return grant, camp, wt, receipt

    def _commit(self, wt):
        (wt / "src" / "app.py").write_text("CAND\n")
        subprocess.run(["git", "add", "-A"], cwd=str(wt), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "cand"], cwd=str(wt), check=True,
                       capture_output=True)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(wt),
                              capture_output=True, text=True).stdout.strip()
        return head, git_worktree_sha(wt)

    def _receipt(self, chunk_id, candidate):
        return mint_validation_receipt(
            validator_id="noop", validator_command="noop", validator_profile="noop",
            candidate_snapshot_digest=candidate, candidate_tree_state="post-apply",
            chunk_id=chunk_id, outcome="pass", detail="t", receipts_dir=_receipts_root())

    def _cas(self, camp, wt, new_commit, rid, *, fence=1, now=None):
        kwargs = {}
        if now is not None:
            kwargs["now"] = now
        return compare_and_swap_advance(
            self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
            chunk_id="chk-1", new_commit=new_commit, holder_fence_generation=fence,
            actor="runner", idempotency_key="i5", validation_receipt_ids=[rid],
            campaign_worktree=wt, **kwargs)


# ============================================================
# 1 — CAS enforces the full campaign continuation authority
# ============================================================

class TestCasContinuationAuthority(_Base):
    def test_revoked_grant_rejects(self):
        grant, camp, wt, _r = self._campaign_and_admit(plan_id="pl-rev", grant_id="gr-rev")
        new_commit, candidate = self._commit(wt)
        rid = self._receipt("chk-1", candidate)
        revoke_grant(self._db, grant_id=grant.grant_id, reason="operator revoked")
        before = (wt / "src" / "app.py").read_text()
        with self.assertRaises(SafetyError) as ctx:
            self._cas(camp, wt, new_commit, rid)
        self.assertIn("grant", str(ctx.exception).lower())
        self.assertEqual((wt / "src" / "app.py").read_text(), before)

    def test_expired_grant_rejects(self):
        expiry = int(time.time()) + 100_000
        grant, camp, wt, _r = self._campaign_and_admit(
            plan_id="pl-exp", grant_id="gr-exp", grant_expires_at=expiry)
        new_commit, candidate = self._commit(wt)
        rid = self._receipt("chk-1", candidate)
        with self.assertRaises(SafetyError) as ctx:
            self._cas(camp, wt, new_commit, rid, now=expiry + 1)
        self.assertIn("expired", str(ctx.exception).lower())

    def test_exhausted_budget_rejects(self):
        grant, camp, wt, _r = self._campaign_and_admit(
            plan_id="pl-bud", grant_id="gr-bud", max_chunks=1)
        new_commit, candidate = self._commit(wt)
        rid = self._receipt("chk-1", candidate)
        # Consume the trusted chunk budget before integration.
        update_budget_after_chunk(self._db, ledger_id=f"bl-{camp.campaign_id}",
                                  delta_chunks=1)
        with self.assertRaises(SafetyError) as ctx:
            self._cas(camp, wt, new_commit, rid)
        self.assertIn("budget", str(ctx.exception).lower())

    def test_active_grant_and_budget_passes(self):
        grant, camp, wt, _r = self._campaign_and_admit(plan_id="pl-ok", grant_id="gr-ok")
        new_commit, candidate = self._commit(wt)
        rid = self._receipt("chk-1", candidate)
        result = self._cas(camp, wt, new_commit, rid)
        self.assertEqual(result.committed_new_commit, new_commit)


# ============================================================
# 2 — admission lease must be unexpired + campaign-scoped
# ============================================================

class TestAdmissionLeaseAuthority(_Base):
    def _admission_lease_id(self, receipt):
        return receipt.lease_id

    def test_active_admission_lease_passes(self):
        grant, camp, wt, receipt = self._campaign_and_admit(plan_id="pl-al", grant_id="gr-al")
        new_commit, candidate = self._commit(wt)
        rid = self._receipt("chk-1", candidate)
        result = self._cas(camp, wt, new_commit, rid)
        self.assertEqual(result.committed_new_commit, new_commit)

    def test_expired_admission_lease_rejects(self):
        grant, camp, wt, receipt = self._campaign_and_admit(plan_id="pl-el", grant_id="gr-el")
        new_commit, candidate = self._commit(wt)
        rid = self._receipt("chk-1", candidate)
        with self._db.transaction() as cur:
            cur.execute("UPDATE leases SET expires_at=? WHERE lease_id=?",
                        (int(time.time()) - 5, receipt.lease_id))
        with self.assertRaises(SafetyError) as ctx:
            self._cas(camp, wt, new_commit, rid)
        self.assertIn("expired", str(ctx.exception).lower())

    def test_cross_campaign_admission_lease_rejects(self):
        grant, camp, wt, receipt = self._campaign_and_admit(plan_id="pl-xc", grant_id="gr-xc")
        new_commit, candidate = self._commit(wt)
        rid = self._receipt("chk-1", candidate)
        # Re-point the admission lease at a different campaign.
        other = create_campaign(self._db, plan_id=grant.plan_id,
                                grant_id=grant.grant_id, base_commit=self._base,
                                base_tree_digest="b" * 64)
        with self._db.transaction() as cur:
            cur.execute("UPDATE leases SET campaign_id=? WHERE lease_id=?",
                        (other.campaign_id, receipt.lease_id))
        with self.assertRaises(SafetyError) as ctx:
            self._cas(camp, wt, new_commit, rid)
        self.assertIn("campaign", str(ctx.exception).lower())

    def test_released_admission_lease_rejects(self):
        from overnight_runner.resources import release_lease
        grant, camp, wt, receipt = self._campaign_and_admit(plan_id="pl-rl", grant_id="gr-rl")
        new_commit, candidate = self._commit(wt)
        rid = self._receipt("chk-1", candidate)
        release_lease(self._db, lease_id=receipt.lease_id)
        with self.assertRaises(SafetyError) as ctx:
            self._cas(camp, wt, new_commit, rid)
        self.assertIn("released", str(ctx.exception).lower())

    def test_stale_lease_fence_rejects(self):
        grant, camp, wt, receipt = self._campaign_and_admit(plan_id="pl-sf", grant_id="gr-sf")
        new_commit, candidate = self._commit(wt)
        rid = self._receipt("chk-1", candidate)
        with self.assertRaises(SafetyError) as ctx:
            self._cas(camp, wt, new_commit, rid, fence=999)
        self.assertIn("fence", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
