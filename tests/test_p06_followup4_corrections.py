"""P06 follow-up #4 — narrow final correction pass.

Covers the follow-up #4 correction list:

  1. crash window + EFFECT_UNKNOWN are atomic.
  2. missing repo identity is fail-closed (never run Git in "." / cwd).
  3. required event/outbox evidence is recovered (idempotently) before
     SAFE_COMPLETED.
  4. campaign continuation is enforced INSIDE the authority APIs.
  5. campaign-v2 integration requires a durable admitted chunk.
  6. campaign-v2 integration requires the canonical campaign worktree.
  7. an expired lease never authorizes a patch apply.
  8. patch apply is bound to the presenting owner identity.
  9. integration is bound to the admission/lease lineage.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from overnight_runner.admission import derive_admission
from overnight_runner.broker import Broker, CommandRegistry, CommandSpec, Proposal
from overnight_runner.campaign import activate_campaign, create_campaign
from overnight_runner.campaign_apply import apply_campaign_patch
from overnight_runner.campaign_schemas import (
    AutonomyGrant,
    Budget,
    ChunkSpec,
    RepoSnapshot,
    content_sha256,
)
from overnight_runner.grants import load_grant
from overnight_runner.db import Database
from overnight_runner.grants import activate_grant
from overnight_runner.integration import (
    compare_and_swap_advance,
    ensure_campaign_worktree,
    record_crash_intent,
    reconcile_crash_window,
)
from overnight_runner.plans import load_plan_digest, register_plan
from overnight_runner.protected_approvals import register_protected_approval
from overnight_runner.receipts import _receipts_root, mint_validation_receipt
from overnight_runner.resources import (
    acquire_lease,
    current_fence,
    process_identity,
    release_lease,
)
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


def _make_repo(tmp_path: Path, name: str = "fixture_repo") -> Path:
    repo = tmp_path / name
    repo.mkdir()
    git_init_empty(repo)
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("BASE\n")
    git_commit_all(repo, "init")
    return repo


def _grant(db, *, plan_id="pl-1", grant_id="gr-1") -> AutonomyGrant:
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
        budget=Budget(schema_version="trio.budget.v1", max_chunks=3,
                      max_model_calls=10, max_tool_calls=20, max_local_repairs=2,
                      max_rechunks=1, max_active_seconds=3600,
                      max_wall_seconds=28800, max_cost_microusd=1000,
                      context_token_budget=8192),
    )
    register_plan(db, plan_id=plan_id, approved_artifact_id=plan_id,
                  work_package_criterion_ids=packages)
    grant = grant.model_copy(update={"approved_plan_digest": load_plan_digest(db, plan_id)})
    register_protected_approval(
        db, approval_id=f"appr-{grant_id}", operation="activate_grant",
        grant_digest_target=content_sha256(grant),
        operator_id="op-1", operator_receipt={"approval_id": f"appr-{grant_id}"})
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


def _chunk(campaign_id, *, chunk_id="chk-1", idem=None, validators=("noop",)) -> ChunkSpec:
    return ChunkSpec(
        schema_version="trio.chunk.v1",
        chunk_id=chunk_id, campaign_id=campaign_id, package_id="pkg-1",
        revision=1, title="c", objective="c",
        permitted_signature_paths=["src/app.py"],
        permitted_write_paths=["src/app.py"],
        permitted_read_paths=["src/app.py"],
        permitted_command_ids=list(validators),
        permitted_validator_ids=list(validators),
        required_validator_ids=list(validators),
        required_receipt_profiles=list(validators),
        criterion_ids=["crit-1"],
        idempotency_key=idem or f"idem-{chunk_id}",
    )


def _admit(db, grant, camp, *, chunk_id="chk-1", idem=None, validators=("noop",)):
    cur_commit = db._conn.execute(
        "SELECT current_commit FROM campaigns WHERE campaign_id=?",
        (camp.campaign_id,)).fetchone()["current_commit"]
    receipt, _ = derive_admission(
        db, grant=grant, chunk=_chunk(camp.campaign_id, chunk_id=chunk_id, idem=idem,
                                      validators=validators),
        worker_id="wkr-1",
        policy_profile_id=grant.policy_profile_id,
        validator_profile_ids=list(grant.validator_profile_ids),
        provider_profile_id=grant.provider_profile_id,
        current_accepted_snapshot=_snap(commit=cur_commit),
        **_id_kwargs(grant))
    return receipt


def _commit(wt: Path, content="CAND\n"):
    (wt / "src" / "app.py").write_text(content)
    subprocess.run(["git", "add", "-A"], cwd=str(wt), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "cand"], cwd=str(wt), check=True, capture_output=True)
    new_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(wt),
                                capture_output=True, text=True).stdout.strip()
    return new_commit, git_worktree_sha(wt)


def _mint(chunk_id, candidate, validator_id="noop"):
    return mint_validation_receipt(
        validator_id=validator_id, validator_command=validator_id,
        validator_profile=validator_id, candidate_snapshot_digest=candidate,
        candidate_tree_state="post-apply", chunk_id=chunk_id, outcome="pass",
        detail="test", receipts_dir=_receipts_root())


def _broker(wt: Path, artifact_dir: Path, text="NEW\n"):
    reg = CommandRegistry()
    reg.register(CommandSpec("noop", ["true"], "repo", 5, "read"))
    b = Broker(repo_root=wt, registry=reg,
               allowed_write_paths=["src/app.py"], allowed_create_paths=[],
               allowed_read_paths=["src/app.py"], allowed_protected_read_paths=[],
               model_allowed_command_ids=["noop"], required_validator_ids=["noop"],
               artifact_dir=artifact_dir)
    before = (wt / "src" / "app.py").read_text()
    b.proposals["prop-x"] = Proposal(
        proposal_id="prop-x", op="replace_file", path="src/app.py",
        abs_path=wt / "src" / "app.py", before_text=before, proposed_text=text,
        preview_diff="", changed_lines=1, proposed_bytes=len(text))
    return b


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

    def _campaign(self, *, plan_id="pl-c", grant_id="gr-c", activate=True,
                  worktree=False, repo_root=True):
        grant = _grant(self._db, plan_id=plan_id, grant_id=grant_id)
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit=self._base, base_tree_digest="b" * 64,
            repo_root=str(self._repo) if repo_root else None)
        if activate:
            activate_campaign(self._db, campaign_id=camp.campaign_id)
        wt = None
        if worktree:
            wt = ensure_campaign_worktree(repo_root=self._repo,
                                          campaign_id=camp.campaign_id,
                                          base_commit=self._base, db=self._db)
        return grant, camp, wt

    def _set_state(self, camp, state):
        with self._db.transaction() as cur:
            cur.execute("UPDATE campaigns SET state=? WHERE campaign_id=?",
                        (state, camp.campaign_id))


# ============================================================
# 1 — crash window + EFFECT_UNKNOWN atomicity
# ============================================================

class TestA05CrashWindowAtomicity(_Base):
    def test_window_and_state_are_atomic(self):
        grant, camp, _ = self._campaign(plan_id="pl-atomic", grant_id="gr-atomic")
        os.environ["TR_FAILPOINT_crash_window_before_commit"] = "raise"
        with self.assertRaises(RuntimeError):
            record_crash_intent(self._db, "integration_cas_mismatch",
                                campaign_id=camp.campaign_id,
                                repo_root=str(self._repo))
        os.environ.pop("TR_FAILPOINT_crash_window_before_commit")
        # Neither the durable window NOR EFFECT_UNKNOWN exists.
        n = self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM crash_windows WHERE campaign_id=?",
            (camp.campaign_id,)).fetchone()["n"]
        self.assertEqual(n, 0)
        state = self._db._conn.execute(
            "SELECT state FROM campaigns WHERE campaign_id=?",
            (camp.campaign_id,)).fetchone()["state"]
        self.assertEqual(state, "ACTIVE")
        # Both are written together when not interrupted.
        record_crash_intent(self._db, "integration_cas_mismatch",
                            campaign_id=camp.campaign_id,
                            repo_root=str(self._repo))
        n = self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM crash_windows WHERE campaign_id=?",
            (camp.campaign_id,)).fetchone()["n"]
        self.assertEqual(n, 1)
        state = self._db._conn.execute(
            "SELECT state FROM campaigns WHERE campaign_id=?",
            (camp.campaign_id,)).fetchone()["state"]
        self.assertEqual(state, "EFFECT_UNKNOWN")


# ============================================================
# 2 — missing repo identity is fail-closed
# ============================================================

class TestA05MissingRepoFailClosed(_Base):
    def test_missing_identity_is_effect_unknown(self):
        grant, camp, _ = self._campaign(plan_id="pl-noid", grant_id="gr-noid",
                                        repo_root=False)
        record_crash_intent(self._db, "integration_cas_mismatch",
                            campaign_id=camp.campaign_id, repo_root="")
        result = reconcile_crash_window(self._db, campaign_id=camp.campaign_id)
        decisions = [d["decision"] for d in result["window_decisions"]]
        self.assertEqual(decisions, ["EFFECT_UNKNOWN"])
        self.assertNotEqual(result["decision"], "RESOLVED_TO_ACTIVE")

    def test_does_not_inspect_a_different_repo_from_cwd(self):
        # A second repo with a campaign/<id> ref that HAS advanced.
        other = _make_repo(self._tmp, "other_repo")
        (other / "src" / "app.py").write_text("OTHER\n")
        subprocess.run(["git", "add", "-A"], cwd=str(other), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "other"], cwd=str(other), check=True, capture_output=True)
        other_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(other),
                                      capture_output=True, text=True).stdout.strip()

        grant, camp, _ = self._campaign(plan_id="pl-cwd", grant_id="gr-cwd",
                                        repo_root=False)
        # Create a matching campaign branch in the OTHER repo.
        subprocess.run(["git", "branch", "-f", f"campaign/{camp.campaign_id}", other_commit],
                       cwd=str(other), check=True, capture_output=True)
        record_crash_intent(self._db, "integration_cas_mismatch",
                            campaign_id=camp.campaign_id, repo_root="")

        old = os.getcwd()
        os.chdir(other)
        try:
            result = reconcile_crash_window(self._db, campaign_id=camp.campaign_id)
        finally:
            os.chdir(old)
        # Fail-closed: must NOT find the other repo's ref and declare safe.
        decisions = [d["decision"] for d in result["window_decisions"]]
        self.assertEqual(decisions, ["EFFECT_UNKNOWN"])


# ============================================================
# 3 — required event evidence recovered before SAFE_COMPLETED
# ============================================================

class TestA05EventEvidenceRecovery(_Base):
    def test_event_recovered_idempotently(self):
        grant, camp, wt = self._campaign(plan_id="pl-ev", grant_id="gr-ev",
                                         worktree=True)
        _admit(self._db, grant, camp, chunk_id="chk-ev")
        new_commit, candidate = _commit(wt)
        rid = _mint("chk-ev", candidate)
        os.environ["TR_FAILPOINT_event_outbox_committed_before_projection_status"] = "raise"
        try:
            with self.assertRaises(RuntimeError):
                compare_and_swap_advance(
                    self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
                    chunk_id="chk-ev", new_commit=new_commit,
                    holder_fence_generation=1, actor="runner", idempotency_key="iev",
                    validation_receipt_ids=[rid], campaign_worktree=wt)
        finally:
            os.environ.pop("TR_FAILPOINT_event_outbox_committed_before_projection_status")
        # Journal durable, event missing.
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM integration_journal WHERE campaign_id=?",
            (camp.campaign_id,)).fetchone()["n"], 1)
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM campaign_events WHERE campaign_id=? "
            "AND event_type='integration_advanced'",
            (camp.campaign_id,)).fetchone()["n"], 0)
        # Restart.
        db_path = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db"
        self._db.close()
        self._db = Database(db_path)
        result = reconcile_crash_window(self._db, campaign_id=camp.campaign_id)
        self.assertEqual(result["decision"], "RESOLVED_TO_ACTIVE")
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM campaign_events WHERE campaign_id=? "
            "AND event_type='integration_advanced'",
            (camp.campaign_id,)).fetchone()["n"], 1)
        state = self._db._conn.execute(
            "SELECT state FROM campaigns WHERE campaign_id=?",
            (camp.campaign_id,)).fetchone()["state"]
        self.assertEqual(state, "ACTIVE")
        # Repeat reconciliation: no duplicate event.
        reconcile_crash_window(self._db, campaign_id=camp.campaign_id)
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM campaign_events WHERE campaign_id=? "
            "AND event_type='integration_advanced'",
            (camp.campaign_id,)).fetchone()["n"], 1)


# ============================================================
# 4 — continuation enforced inside the authority APIs
# ============================================================

class TestContinuationInsideAuthorityAPIs(_Base):
    def test_derive_admission_refuses_effect_unknown(self):
        grant, camp, _ = self._campaign(plan_id="pl-adm-blk", grant_id="gr-adm-blk")
        _admit(self._db, grant, camp, chunk_id="chk-1")
        self._set_state(camp, "EFFECT_UNKNOWN")
        with self.assertRaises(SafetyError) as ctx:
            _admit(self._db, grant, camp, chunk_id="chk-2", idem="idem-2")
        self.assertIn("effect_unknown", str(ctx.exception).lower())

    def test_cas_refuses_effect_unknown(self):
        grant, camp, wt = self._campaign(plan_id="pl-cas-eu", grant_id="gr-cas-eu",
                                         worktree=True)
        _admit(self._db, grant, camp, chunk_id="chk-1")
        new_commit, candidate = _commit(wt)
        rid = _mint("chk-1", candidate)
        self._set_state(camp, "EFFECT_UNKNOWN")
        with self.assertRaises(SafetyError) as ctx:
            compare_and_swap_advance(
                self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
                chunk_id="chk-1", new_commit=new_commit, holder_fence_generation=1,
                actor="runner", idempotency_key="i", validation_receipt_ids=[rid],
                campaign_worktree=wt)
        self.assertIn("blocking state", str(ctx.exception).lower())

    def test_cas_refuses_cancelled(self):
        grant, camp, wt = self._campaign(plan_id="pl-cas-can", grant_id="gr-cas-can",
                                         worktree=True)
        _admit(self._db, grant, camp, chunk_id="chk-1")
        new_commit, candidate = _commit(wt)
        rid = _mint("chk-1", candidate)
        self._set_state(camp, "CANCELLED")
        with self.assertRaises(SafetyError):
            compare_and_swap_advance(
                self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
                chunk_id="chk-1", new_commit=new_commit, holder_fence_generation=1,
                actor="runner", idempotency_key="i", validation_receipt_ids=[rid],
                campaign_worktree=wt)

    def test_normal_active_path_green(self):
        grant, camp, wt = self._campaign(plan_id="pl-cas-ok", grant_id="gr-cas-ok",
                                         worktree=True)
        _admit(self._db, grant, camp, chunk_id="chk-1")
        new_commit, candidate = _commit(wt)
        rid = _mint("chk-1", candidate)
        result = compare_and_swap_advance(
            self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
            chunk_id="chk-1", new_commit=new_commit, holder_fence_generation=1,
            actor="runner", idempotency_key="i", validation_receipt_ids=[rid],
            campaign_worktree=wt)
        self.assertEqual(result.committed_new_commit, new_commit)


# ============================================================
# 5 — durable admitted chunk authority required
# ============================================================

class TestA04DurableChunkAuthority(_Base):
    def test_missing_chunk_blocks(self):
        grant, camp, wt = self._campaign(plan_id="pl-nc", grant_id="gr-nc",
                                         worktree=True)
        new_commit, candidate = _commit(wt)
        rid = _mint("ghost", candidate)
        with self.assertRaises(SafetyError) as ctx:
            compare_and_swap_advance(
                self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
                chunk_id="ghost", new_commit=new_commit, holder_fence_generation=1,
                actor="runner", idempotency_key="i", validation_receipt_ids=[rid],
                campaign_worktree=wt)
        self.assertIn("no durable admitted chunk", str(ctx.exception).lower())

    def test_chunk_of_other_campaign_blocks(self):
        grant_a, camp_a, _ = self._campaign(plan_id="pl-a", grant_id="gr-a")
        _admit(self._db, grant_a, camp_a, chunk_id="chk-a")
        grant_b, camp_b, wt = self._campaign(plan_id="pl-b", grant_id="gr-b",
                                             worktree=True)
        new_commit, candidate = _commit(wt)
        rid = _mint("chk-a", candidate)
        with self.assertRaises(SafetyError) as ctx:
            compare_and_swap_advance(
                self._db, repo_root=self._repo, campaign_id=camp_b.campaign_id,
                chunk_id="chk-a", new_commit=new_commit, holder_fence_generation=1,
                actor="runner", idempotency_key="i", validation_receipt_ids=[rid],
                campaign_worktree=wt)
        self.assertIn("belongs to campaign", str(ctx.exception).lower())

    def test_missing_admission_id_blocks(self):
        grant, camp, wt = self._campaign(plan_id="pl-ma", grant_id="gr-ma",
                                         worktree=True)
        with self._db.transaction() as cur:
            cur.execute(
                "INSERT INTO chunks (chunk_id, campaign_id, package_id, revision, "
                "idempotency_key, state, admission_id, created_at, updated_at, "
                "required_validator_ids_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                ("chk-noadm", camp.campaign_id, "pkg-1", 1, "idem-noadm",
                 "ADMITTED", None, 0, 0, json.dumps(["noop"])))
        new_commit, candidate = _commit(wt)
        rid = _mint("chk-noadm", candidate)
        with self.assertRaises(SafetyError) as ctx:
            compare_and_swap_advance(
                self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
                chunk_id="chk-noadm", new_commit=new_commit, holder_fence_generation=1,
                actor="runner", idempotency_key="i", validation_receipt_ids=[rid],
                campaign_worktree=wt)
        self.assertIn("admission_id", str(ctx.exception).lower())


# ============================================================
# 6 — canonical campaign worktree required
# ============================================================

class TestA04CanonicalWorktreeRequired(_Base):
    def test_caller_digest_cannot_substitute(self):
        grant, camp, wt = self._campaign(plan_id="pl-cw", grant_id="gr-cw",
                                         worktree=True)
        _admit(self._db, grant, camp, chunk_id="chk-1")
        new_commit, _candidate = _commit(wt)
        # Receipt claims a digest the caller also supplies as
        # expected_tree_digest — but the canonical worktree fingerprint
        # is different, so integration must reject.
        rid = _mint("chk-1", "e" * 64)
        with self.assertRaises(SafetyError) as ctx:
            compare_and_swap_advance(
                self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
                chunk_id="chk-1", new_commit=new_commit, holder_fence_generation=1,
                actor="runner", idempotency_key="i", validation_receipt_ids=[rid],
                expected_tree_digest="e" * 64)
        # The canonical worktree fingerprint is used; the caller digest
        # and the receipt digest are both ignored/rejected.
        self.assertIn("candidate", str(ctx.exception).lower())

    def test_missing_worktree_identity_blocks(self):
        grant, camp, _ = self._campaign(plan_id="pl-nowt", grant_id="gr-nowt",
                                        worktree=False)
        _admit(self._db, grant, camp, chunk_id="chk-1")
        rid = _mint("chk-1", "e" * 64)
        with self.assertRaises(SafetyError) as ctx:
            compare_and_swap_advance(
                self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
                chunk_id="chk-1", new_commit="0" * 40, holder_fence_generation=1,
                actor="runner", idempotency_key="i", validation_receipt_ids=[rid])
        self.assertIn("worktree", str(ctx.exception).lower())


# ============================================================
# 7 + 8 — expired lease + presenting owner identity
# ============================================================

class TestA06ExpiredLeaseAndOwnerBinding(_Base):
    def _lease(self, camp, owner_id="wkr-A"):
        gen = current_fence(self._db, camp.campaign_id).current_generation
        lease = acquire_lease(self._db, campaign_id=camp.campaign_id,
                              resource_id="chunk:c", owner_id=owner_id,
                              owner_boot_id="b", owner_pid=os.getpid(),
                              fence_generation=gen, ttl_seconds=300)
        return gen, lease

    def _apply(self, camp, wt, *, lease, owner_id="wkr-A", pid=None, start=None):
        ident = process_identity(os.getpid())
        return apply_campaign_patch(
            self._db, _broker(wt, self._tmp / "art"), str(self._repo),
            camp.campaign_id, "prop-x",
            admission_fence_generation=current_fence(self._db, camp.campaign_id).current_generation,
            lease_id=lease.lease_id, owner_id=owner_id,
            owner_pid=os.getpid() if pid is None else pid,
            owner_start_time=ident["start_time"] if start is None else start)

    def test_fresh_current_lease_succeeds(self):
        grant, camp, wt = self._campaign(plan_id="pl-fresh", grant_id="gr-fresh",
                                         worktree=True)
        _gen, lease = self._lease(camp)
        self._apply(camp, wt, lease=lease)

    def test_expired_lease_with_dead_owner_rejects(self):
        grant, camp, wt = self._campaign(plan_id="pl-exp-dead", grant_id="gr-exp-dead",
                                         worktree=True)
        _gen, lease = self._lease(camp)
        with self._db.transaction() as cur:
            cur.execute("UPDATE leases SET expires_at=? WHERE lease_id=?",
                        (int(time.time()) - 10, lease.lease_id))
        before = (wt / "src" / "app.py").read_text()
        with self.assertRaises(SafetyError) as ctx:
            self._apply(camp, wt, lease=lease)
        self.assertIn("expired", str(ctx.exception).lower())
        self.assertEqual((wt / "src" / "app.py").read_text(), before)

    def test_expired_lease_with_live_owner_rejects(self):
        grant, camp, wt = self._campaign(plan_id="pl-exp-live", grant_id="gr-exp-live",
                                         worktree=True)
        _gen, lease = self._lease(camp)
        with self._db.transaction() as cur:
            cur.execute("UPDATE leases SET expires_at=? WHERE lease_id=?",
                        (int(time.time()) - 1, lease.lease_id))
        with self.assertRaises(SafetyError) as ctx:
            self._apply(camp, wt, lease=lease)
        self.assertIn("expired", str(ctx.exception).lower())

    def test_wrong_owner_id_rejects(self):
        grant, camp, wt = self._campaign(plan_id="pl-own", grant_id="gr-own",
                                         worktree=True)
        _gen, lease = self._lease(camp, owner_id="wkr-A")
        with self.assertRaises(SafetyError) as ctx:
            self._apply(camp, wt, lease=lease, owner_id="wkr-B")
        self.assertIn("owner", str(ctx.exception).lower())

    def test_wrong_pid_identity_rejects(self):
        grant, camp, wt = self._campaign(plan_id="pl-pid", grant_id="gr-pid",
                                         worktree=True)
        _gen, lease = self._lease(camp)
        with self.assertRaises(SafetyError) as ctx:
            self._apply(camp, wt, lease=lease, pid=os.getpid() + 4242)
        self.assertIn("pid", str(ctx.exception).lower())

    def test_wrong_start_time_rejects(self):
        grant, camp, wt = self._campaign(plan_id="pl-start", grant_id="gr-start",
                                         worktree=True)
        _gen, lease = self._lease(camp)
        with self.assertRaises(SafetyError) as ctx:
            self._apply(camp, wt, lease=lease, start="0")
        self.assertIn("start-time", str(ctx.exception).lower())


# ============================================================
# 9 — integration bound to admission/lease lineage
# ============================================================

class TestIntegrationAdmissionLeaseLineage(_Base):
    def test_released_admission_lease_blocks(self):
        grant, camp, wt = self._campaign(plan_id="pl-lin", grant_id="gr-lin",
                                         worktree=True)
        receipt = _admit(self._db, grant, camp, chunk_id="chk-1")
        new_commit, candidate = _commit(wt)
        rid = _mint("chk-1", candidate)
        # Release the admission's own lease.
        release_lease(self._db, lease_id=receipt.lease_id)
        with self.assertRaises(SafetyError) as ctx:
            compare_and_swap_advance(
                self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
                chunk_id="chk-1", new_commit=new_commit, holder_fence_generation=1,
                actor="runner", idempotency_key="i", validation_receipt_ids=[rid],
                campaign_worktree=wt)
        self.assertIn("released", str(ctx.exception).lower())

    def test_stale_admission_fence_blocks(self):
        grant, camp, wt = self._campaign(plan_id="pl-linf", grant_id="gr-linf",
                                         worktree=True)
        _admit(self._db, grant, camp, chunk_id="chk-1")
        new_commit, candidate = _commit(wt)
        rid = _mint("chk-1", candidate)
        with self.assertRaises(SafetyError) as ctx:
            compare_and_swap_advance(
                self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
                chunk_id="chk-1", new_commit=new_commit, holder_fence_generation=999,
                actor="runner", idempotency_key="i", validation_receipt_ids=[rid],
                campaign_worktree=wt)
        self.assertIn("fence", str(ctx.exception).lower())


# ============================================================
# 11 — failpoint acceptance: restart + reconcile + no blind replay
# ============================================================

class TestA05FailpointAcceptance(_Base):
    def _drive(self, name: str, suffix: str, plan_id: str):
        grant, camp, wt = self._campaign(plan_id=plan_id,
                                         grant_id=f"gr-{plan_id}", worktree=True)
        _admit(self._db, grant, camp, chunk_id=f"chk-{suffix}")
        new_commit, candidate = _commit(wt)
        rid = _mint(f"chk-{suffix}", candidate)
        os.environ[f"TR_FAILPOINT_{name}"] = "raise"
        try:
            with self.assertRaises(RuntimeError):
                compare_and_swap_advance(
                    self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
                    chunk_id=f"chk-{suffix}", new_commit=new_commit,
                    holder_fence_generation=1, actor="runner",
                    idempotency_key=f"i-{suffix}", validation_receipt_ids=[rid],
                    campaign_worktree=wt)
        finally:
            os.environ.pop(f"TR_FAILPOINT_{name}")
        return camp, new_commit

    def _counts(self):
        return {
            t: self._db._conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]
            for t in ("integration_journal", "campaign_events", "admissions", "leases")
        }

    def test_cas_failpoints_reach_intent_restart_reconcile(self):
        cases = [
            ("commit_exists_before_db_candidate_state", "ce"),
            ("db_integration_intent_before_ref_advance", "di"),
            ("ref_advanced_before_integration_journal_event", "ra"),
            ("event_outbox_committed_before_projection_status", "ev"),
        ]
        for name, suf in cases:
            with self.subTest(failpoint=name):
                camp, new_commit = self._drive(name, suf, f"pl-fp-{suf}")
                # Real path reached: durable intent + EFFECT_UNKNOWN.
                kinds = [r["kind"] for r in self._db._conn.execute(
                    "SELECT kind FROM crash_windows WHERE campaign_id=?",
                    (camp.campaign_id,)).fetchall()]
                self.assertIn(name, kinds)
                self.assertEqual(self._db._conn.execute(
                    "SELECT state FROM campaigns WHERE campaign_id=?",
                    (camp.campaign_id,)).fetchone()["state"], "EFFECT_UNKNOWN")
                # Restart.
                db_path = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db"
                self._db.close()
                self._db = Database(db_path)
                before = self._counts()
                result = reconcile_crash_window(self._db, campaign_id=camp.campaign_id)
                after = self._counts()
                # No blind replay of journal/admissions/leases.
                self.assertEqual(after["integration_journal"], before["integration_journal"])
                self.assertEqual(after["admissions"], before["admissions"])
                self.assertEqual(after["leases"], before["leases"])
                if name == "ref_advanced_before_integration_journal_event":
                    # Ref DID advance but journal is absent -> needs a
                    # decision, never a false SAFE_COMPLETED.
                    self.assertEqual(result["decision"], "EFFECT_UNKNOWN_PRESERVED")
                elif name == "event_outbox_committed_before_projection_status":
                    # Required event evidence restored idempotently ->
                    # only THEN ACTIVE (exactly one event).
                    self.assertEqual(result["decision"], "RESOLVED_TO_ACTIVE")
                    self.assertEqual(after["campaign_events"],
                                     before["campaign_events"] + 1)
                else:
                    # No ref advance and no durable advance evidence -> safe.
                    self.assertEqual(result["decision"], "RESOLVED_TO_ACTIVE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
