"""P08 (A03) — runner-owned admission-lease renewal for resumed obligations.

Proves that a resumed capacity obligation continues the SAME admission
through RUNNER AUTHORITY, never through test-supplied SQL:

  A. a caller-acquired unrelated lease cannot substitute for authority
  B. a renewal referencing another campaign's wake claim is rejected
  C. a binding that does not bind the admission to its own chunk is rejected
  D. a stale fence is rejected
  E. an expired replacement lease has no integration authority
  F. a valid capacity-resume renewal authorizes integration (PASS)
  G. a repeated resume request is idempotent (no duplicate lease authority)

Plus: a still-valid original lease blocks renewal, and an unknown wake claim
is rejected.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from overnight_runner.admission import derive_admission
from overnight_runner.admission_lease import (
    effective_admission_lease_id,
    list_bindings,
    refresh_admission_lease_for_resume,
)
from overnight_runner.broker import Broker, CommandRegistry, CommandSpec, Proposal
from overnight_runner.campaign import (
    activate_campaign,
    create_campaign,
    read_budget_totals,
    update_budget_after_chunk,
)
from overnight_runner.campaign_apply import apply_campaign_patch
from overnight_runner.campaign_schemas import (
    AutonomyGrant,
    Budget,
    ChunkSpec,
    RepoSnapshot,
    content_sha256,
)
from overnight_runner.db import Database
from overnight_runner.grants import activate_grant, load_grant
from overnight_runner.integration import (
    compare_and_swap_advance,
    ensure_campaign_worktree,
)
from overnight_runner.p07 import (
    classify_capacity,
    handle_provider_result,
    start_wake_claim_execution,
    wake_tick,
)
from overnight_runner.plans import load_plan_digest, register_plan
from overnight_runner.protected_approvals import register_protected_approval
from overnight_runner.receipts import mint_validation_receipt
from overnight_runner.resources import (
    acquire_lease,
    current_fence,
    process_identity,
    release_lease,
    revoke_for_takeover,
)
from overnight_runner.safety import (
    SafetyError,
    git_commit_all,
    git_head,
    git_init_empty,
    git_worktree_sha,
)
from overnight_runner.schemas import Disposition, ExecutionClass, TaskManifest
from overnight_runner.worker import _finalise


class RenewalBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="p08-alb-"))
        state = self.tmp / "state"
        state.mkdir(parents=True, exist_ok=True)
        os.environ["OVERNIGHT_STATE_DIR"] = str(state)
        os.environ["OVERNIGHT_RECEIPTS"] = "1"
        os.environ["TR_P06_CAMPAIGN_V2"] = "1"
        self.db = Database(state / "state.db")
        self.repo = self.tmp / "pilot"
        self.repo.mkdir()
        git_init_empty(self.repo)
        (self.repo / "src").mkdir()
        (self.repo / "src" / "app.py").write_text("BASE\n")
        git_commit_all(self.repo, "init")
        self.base = git_head(self.repo)
        self.grant = self._grant("pl-1", "gr-1")
        self.camp = self._campaign(self.grant, "camp")
        self.wt = ensure_campaign_worktree(
            repo_root=self.repo, campaign_id=self.camp.campaign_id,
            base_commit=self.base, db=self.db)
        self.ledger = f"bl-{self.camp.campaign_id}"

    def tearDown(self):
        try:
            self.db.close()
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------- fixture helpers -------------------------

    def _grant(self, plan_id: str, grant_id: str) -> AutonomyGrant:
        register_plan(self.db, plan_id=plan_id, approved_artifact_id=plan_id,
                      work_package_criterion_ids={"pkg-1": {"crit-1"}})
        g = AutonomyGrant(
            schema_version="trio.grant.v1", grant_id=grant_id, state="draft",
            plan_id=plan_id, plan_revision=1, approved_plan_digest="0" * 64,
            repository_paths=["src/app.py"], allowed_write_paths=["src/app.py"],
            protected_paths=[], allowed_operations=["noop"],
            runtime_digest="a" * 64, model_name="gemma", model_digest="b" * 64,
            policy_profile_id="pol-1", validator_profile_ids=["noop"],
            provider_profile_id="prv-1", egress_policy_id="eg-1",
            operator_id="op-1", operator_receipt_digest="c" * 64,
            budget=Budget(schema_version="trio.budget.v1", max_chunks=5,
                          max_model_calls=50, max_tool_calls=100,
                          max_local_repairs=2, max_rechunks=1,
                          max_active_seconds=3600, max_wall_seconds=28800,
                          max_cost_microusd=0, context_token_budget=8192),
        )
        g = g.model_copy(update={
            "approved_plan_digest": load_plan_digest(self.db, plan_id)})
        register_protected_approval(
            self.db, approval_id=f"appr-{grant_id}", operation="activate_grant",
            grant_digest_target=content_sha256(g), operator_id="op-1",
            operator_receipt={"approval_id": f"appr-{grant_id}"})
        activate_grant(self.db, grant=g, operator_id="op-1",
                       approval_id=f"appr-{grant_id}")
        return load_grant(self.db, grant_id)

    def _campaign(self, grant, tag: str):
        camp = create_campaign(self.db, plan_id=grant.plan_id,
                               grant_id=grant.grant_id, base_commit=self.base,
                               base_tree_digest="b" * 64,
                               repo_root=str(self.repo))
        activate_campaign(self.db, campaign_id=camp.campaign_id)
        return camp

    def _admit(self, campaign, *, chunk_id: str):
        chunk = ChunkSpec(
            schema_version="trio.chunk.v1", chunk_id=chunk_id,
            campaign_id=campaign.campaign_id, package_id="pkg-1", revision=1,
            title="c", objective="c", permitted_signature_paths=["src/app.py"],
            permitted_write_paths=["src/app.py"],
            permitted_read_paths=["src/app.py"],
            permitted_command_ids=["noop"], permitted_validator_ids=["noop"],
            required_validator_ids=["noop"], required_receipt_profiles=["noop"],
            criterion_ids=["crit-1"], idempotency_key=f"idem-{chunk_id}")
        grant = load_grant(self.db, campaign.grant_id)
        cur = self.db._conn.execute(
            "SELECT current_commit FROM campaigns WHERE campaign_id=?",
            (campaign.campaign_id,)).fetchone()
        snap = RepoSnapshot(schema_version="trio.repo-snapshot.v1",
                            repository_id="local",
                            commit=cur["current_commit"] or self.base,
                            tree_digest="b" * 64)
        receipt, _ = derive_admission(
            self.db, grant=grant, chunk=chunk, worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            current_accepted_snapshot=snap,
            current_runtime_digest=grant.runtime_digest,
            current_model_name=grant.model_name,
            current_model_digest=grant.model_digest,
            current_policy_profile_id=grant.policy_profile_id,
            current_validator_profile_ids=list(grant.validator_profile_ids),
            current_provider_profile_id=grant.provider_profile_id)
        return chunk, receipt

    def _release_all(self, campaign_id: str):
        for r in self.db._conn.execute(
                "SELECT lease_id FROM leases WHERE campaign_id=? AND released_at=0",
                (campaign_id,)).fetchall():
            release_lease(self.db, lease_id=r["lease_id"])

    def _capacity_wait_and_wake(self, campaign_id: str) -> dict:
        """Simulated 429 -> durable wait -> restart -> one wake claim."""
        out = classify_capacity(provider="prv-1", http_status=429, model="gemma",
                                retry_after_seconds=60)
        handle_provider_result(self.db, out, campaign_id=campaign_id, now=1000)
        self.db.close()
        self.db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        self._release_all(campaign_id)
        before = wake_tick(self.db, campaign_id=campaign_id, trigger_id="pre",
                           now=1010)
        self.assertEqual(before["decision"], "WAIT")
        return wake_tick(self.db, campaign_id=campaign_id, trigger_id="go",
                         now=1200)

    def _apply_and_commit(self, campaign_id: str, admission_id: str, lease_id: str):
        reg = CommandRegistry()
        reg.register(CommandSpec("noop", ["true"], "repo", 5, "read"))
        broker = Broker(repo_root=self.wt, registry=reg,
                        allowed_write_paths=["src/app.py"],
                        allowed_create_paths=[], allowed_read_paths=["src/app.py"],
                        allowed_protected_read_paths=[],
                        model_allowed_command_ids=["noop"],
                        required_validator_ids=[], approved_repo_head=None,
                        artifact_dir=self.tmp / "art")
        before = (self.wt / "src" / "app.py").read_text()
        broker.proposals["prop-1"] = Proposal(
            proposal_id="prop-1", op="replace_file", path="src/app.py",
            abs_path=self.wt / "src" / "app.py", before_text=before,
            proposed_text="CHANGED\n", preview_diff="", changed_lines=1,
            proposed_bytes=8)
        ident = process_identity(os.getpid())
        lease = self.db._conn.execute(
            "SELECT lease_id, fence_generation FROM leases WHERE lease_id=?",
            (lease_id,)).fetchone()
        apply_campaign_patch(
            self.db, broker, str(self.repo), campaign_id, "prop-1",
            admission_fence_generation=int(lease["fence_generation"]),
            lease_id=lease_id, owner_id="wkr-1", owner_pid=os.getpid(),
            owner_start_time=ident["start_time"])
        subprocess.run(["git", "add", "-A"], cwd=str(self.wt), check=True,
                       capture_output=True)
        subprocess.run(["git", "commit", "-m", "chunk"], cwd=str(self.wt),
                       check=True, capture_output=True)
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(self.wt),
                              capture_output=True, text=True).stdout.strip()

    def _validate(self, chunk_id: str):
        ids: list[str] = []

        def _mint(payload):
            rid = mint_validation_receipt(
                validator_id=payload["validator_id"],
                validator_command=payload["validator_command"],
                validator_profile=payload.get("validator_profile")
                or payload["validator_command"],
                candidate_snapshot_digest=payload["candidate_snapshot_digest"],
                chunk_id=payload.get("chunk_id", ""), outcome=payload["outcome"],
                detail=payload.get("detail", ""),
                receipts_dir=Path(os.environ["OVERNIGHT_STATE_DIR"]) / "receipts")
            ids.append(rid)
            return rid

        for pc in self.wt.rglob("__pycache__"):
            shutil.rmtree(pc, ignore_errors=True)
        manifest = TaskManifest(
            task_id=chunk_id, title="c", objective="c",
            execution_class=ExecutionClass.SOURCE_MUTATION,
            repo={"path": str(self.wt)},
            paths={"write_paths": ["src/app.py"], "create_paths": [],
                   "read_paths": ["src/app.py"], "protected_read_paths": []},
            commands={"model_allowed_command_ids": ["noop"],
                      "required_validator_ids": ["noop"],
                      "allow_no_mutation": True},
            model_profile={"model_name": "gemma", "temperature": 0.0},
            context_budget={"max_read_bytes": 4096, "max_files_read": 4,
                            "max_files_written": 2},
            limits={"max_model_turns": 1, "max_tool_calls": 4,
                    "max_changed_files": 2, "max_diff_lines": 400,
                    "max_written_bytes": 65536, "task_timeout_seconds": 60,
                    "max_tool_result_bytes": 24576},
            acceptance_criteria=[])
        reg = CommandRegistry()
        reg.register(CommandSpec("noop", ["true"], "repo", 5, "read"))
        broker = Broker(repo_root=self.wt, registry=reg,
                        allowed_write_paths=["src/app.py"],
                        allowed_create_paths=[], allowed_read_paths=["src/app.py"],
                        allowed_protected_read_paths=[],
                        model_allowed_command_ids=["noop"],
                        required_validator_ids=[], approved_repo_head=None,
                        artifact_dir=self.tmp / "art-v")
        status, _code, text, rids = _finalise(
            manifest=manifest, broker=broker, disposition=Disposition.DONE,
            artifact_dir=self.tmp / "art-v", applied_proposals=[],
            on_validation_receipt=_mint, env_digest="e", profile_digest="p")
        return status, rids, text

    def _integrate(self, campaign_id: str, chunk_id: str, commit: str, rids,
                   *, fence=1, now=None, expected_old=None):
        return compare_and_swap_advance(
            self.db, repo_root=self.repo, campaign_id=campaign_id,
            chunk_id=chunk_id, new_commit=commit, holder_fence_generation=fence,
            actor="runner", idempotency_key=f"i-{chunk_id}",
            validation_receipt_ids=list(rids), expected_old=expected_old,
            campaign_worktree=self.wt, now=now)

    # ============================== negatives ==============================

    def test_a_arbitrary_caller_lease_cannot_substitute(self):
        chunk, receipt = self._admit(self.camp, chunk_id="chk-1")
        self._release_all(self.camp.campaign_id)
        # A caller acquires a lease of its own, entirely outside admission
        # renewal authority.
        stray = acquire_lease(
            self.db, campaign_id=self.camp.campaign_id, resource_id="worker:rogue",
            owner_id="rogue", owner_boot_id="", owner_pid=os.getpid(),
            fence_generation=current_fence(self.db, self.camp.campaign_id).current_generation,
            ttl_seconds=600)
        eff, source = effective_admission_lease_id(self.db, receipt.admission_id)
        self.assertNotEqual(eff, stray.lease_id)
        self.assertEqual(source, "original")
        # The unrelated lease cannot author a campaign mutation at all.
        with self.assertRaises(SafetyError) as ctx:
            self._apply_and_commit(self.camp.campaign_id,
                                   receipt.admission_id, stray.lease_id)
        self.assertIn("lease_scope_invalid", str(ctx.exception))
        # And it cannot become the admission's effective lease either.
        eff2, source2 = effective_admission_lease_id(self.db, receipt.admission_id)
        self.assertEqual(eff2, receipt.lease_id)
        self.assertEqual(source2, "original")

    def test_b_renewal_wake_claim_must_belong_to_the_campaign(self):
        _, receipt = self._admit(self.camp, chunk_id="chk-1")
        other_grant = self._grant("pl-2", "gr-2")
        other = self._campaign(other_grant, "other")
        other_claim = wake_tick(self.db, campaign_id=other.campaign_id,
                                trigger_id="x", now=1000)
        self.assertEqual(other_claim["decision"], "DISPATCH_ELIGIBLE")
        self._release_all(self.camp.campaign_id)
        with self.assertRaises(SafetyError) as ctx:
            refresh_admission_lease_for_resume(
                self.db, admission_id=receipt.admission_id,
                wake_claim_id=other_claim["claim_id"], owner_id="wkr-1",
                owner_pid=os.getpid(), now=1200)
        self.assertIn("belongs to campaign", str(ctx.exception))

    def test_b2_unknown_wake_claim_rejected(self):
        _, receipt = self._admit(self.camp, chunk_id="chk-1")
        self._release_all(self.camp.campaign_id)
        with self.assertRaises(SafetyError):
            refresh_admission_lease_for_resume(
                self.db, admission_id=receipt.admission_id,
                wake_claim_id="claim-does-not-exist", owner_id="wkr-1",
                owner_pid=os.getpid(), now=1200)
        with self.assertRaises(SafetyError):
            refresh_admission_lease_for_resume(
                self.db, admission_id=receipt.admission_id, wake_claim_id="",
                owner_id="wkr-1", owner_pid=os.getpid(), now=1200)

    def test_c_binding_must_bind_the_admission_to_its_own_chunk(self):
        chunk, receipt = self._admit(self.camp, chunk_id="chk-1")
        self._release_all(self.camp.campaign_id)
        # Issue a VALID renewal, then forge a binding that points the
        # admission at a different chunk (simulating a substitution attempt).
        claim = self._capacity_wait_and_wake(self.camp.campaign_id)
        res = refresh_admission_lease_for_resume(
            self.db, admission_id=receipt.admission_id,
            wake_claim_id=claim["claim_id"], owner_id="wkr-1",
            owner_pid=os.getpid(), now=1200)
        with self.db.transaction() as cur:
            cur.execute(
                "UPDATE admission_lease_bindings SET chunk_id='chk-other' "
                "WHERE binding_id=?", (res["binding_id"],))
        commit = self._apply_and_commit(self.camp.campaign_id,
                                        receipt.admission_id,
                                        res["effective_lease_id"])
        status, rids, text = self._validate(chunk.chunk_id)
        self.assertEqual(status, "PASSED", text)
        with self.assertRaises(SafetyError) as ctx:
            self._integrate(self.camp.campaign_id, chunk.chunk_id, commit, rids)
        self.assertIn("does not bind admission", str(ctx.exception))

    def test_d_stale_fence_rejected(self):
        _, receipt = self._admit(self.camp, chunk_id="chk-1")
        self._release_all(self.camp.campaign_id)
        claim = self._capacity_wait_and_wake(self.camp.campaign_id)
        revoke_for_takeover(self.db, campaign_id=self.camp.campaign_id,
                            reason="operator takeover")
        with self.assertRaises(SafetyError) as ctx:
            refresh_admission_lease_for_resume(
                self.db, admission_id=receipt.admission_id,
                wake_claim_id=claim["claim_id"], owner_id="wkr-1",
                owner_pid=os.getpid(), now=1200)
        self.assertIn("fence", str(ctx.exception))

    def test_e_expired_replacement_lease_has_no_authority(self):
        chunk, receipt = self._admit(self.camp, chunk_id="chk-1")
        self._release_all(self.camp.campaign_id)
        claim = self._capacity_wait_and_wake(self.camp.campaign_id)
        res = refresh_admission_lease_for_resume(
            self.db, admission_id=receipt.admission_id,
            wake_claim_id=claim["claim_id"], owner_id="wkr-1",
            owner_pid=os.getpid(), ttl_seconds=30, now=1200)
        commit = self._apply_and_commit(self.camp.campaign_id,
                                        receipt.admission_id,
                                        res["effective_lease_id"])
        status, rids, text = self._validate(chunk.chunk_id)
        self.assertEqual(status, "PASSED", text)
        with self.assertRaises(SafetyError) as ctx:
            self._integrate(self.camp.campaign_id, chunk.chunk_id, commit, rids,
                            now=res["lease_expires_at"] + 1)
        self.assertIn("expired", str(ctx.exception))

    def test_h_renewal_refused_while_original_lease_still_valid(self):
        _, receipt = self._admit(self.camp, chunk_id="chk-1")
        # A durable P07 claim row is inserted as FIXTURE data only (no
        # authority is manufactured); the original admission lease is left
        # live, so the renewal guard must refuse.
        with self.db.transaction() as cur:
            cur.execute(
                "INSERT INTO wake_claims (claim_id, campaign_id, obligation_id, "
                "generation, claimed_at, state, caller_trigger_id) "
                "VALUES ('claim-fixture',?,?,1,900,'CLAIMED','fix')",
                (self.camp.campaign_id, "obl-fixture"))
        with self.assertRaises(SafetyError) as ctx:
            refresh_admission_lease_for_resume(
                self.db, admission_id=receipt.admission_id,
                wake_claim_id="claim-fixture", owner_id="wkr-1",
                owner_pid=os.getpid(), now=1200)
        self.assertIn("still valid", str(ctx.exception))

    # ============================== positives ==============================

    def test_f_valid_capacity_resume_renewal_authorizes_integration(self):
        chunk, receipt = self._admit(self.camp, chunk_id="chk-1")
        claim = self._capacity_wait_and_wake(self.camp.campaign_id)
        self.assertEqual(claim["decision"], "DISPATCH_ELIGIBLE")
        # One owned resumed execution, consumed by the runner.
        run = start_wake_claim_execution(
            self.db, claim_id=claim["claim_id"],
            argv=[sys.executable, "-c", "pass"], cwd=str(self.tmp),
            now=1201, result_dir=str(self.tmp / "res"))
        self.assertTrue(run["run_id"])
        res = refresh_admission_lease_for_resume(
            self.db, admission_id=receipt.admission_id,
            wake_claim_id=claim["claim_id"], owner_id="wkr-1",
            owner_pid=os.getpid(), idempotency_key="resume-1", now=1202)
        self.assertEqual(res["issuer"], "runner")
        self.assertEqual(res["admission_id"], receipt.admission_id)
        self.assertEqual(res["chunk_id"], chunk.chunk_id)
        self.assertEqual(res["campaign_id"], self.camp.campaign_id)
        self.assertEqual(res["prior_lease_id"], receipt.lease_id)
        self.assertEqual(res["wake_claim_id"], claim["claim_id"])
        eff, source = effective_admission_lease_id(self.db, receipt.admission_id)
        self.assertEqual(eff, res["effective_lease_id"])
        self.assertEqual(source, "renewal")
        # The ORIGINAL admission receipt lineage is untouched.
        self.assertEqual(
            self.db._conn.execute("SELECT lease_id FROM admissions WHERE admission_id=?",
                                  (receipt.admission_id,)).fetchone()["lease_id"],
            receipt.lease_id)
        commit = self._apply_and_commit(self.camp.campaign_id,
                                        receipt.admission_id,
                                        res["effective_lease_id"])
        status, rids, text = self._validate(chunk.chunk_id)
        self.assertEqual(status, "PASSED", text)
        result = self._integrate(self.camp.campaign_id, chunk.chunk_id, commit, rids)
        self.assertEqual(result.committed_new_commit, commit)
        # SAME ledger; no repair charged.
        update_budget_after_chunk(self.db, ledger_id=self.ledger, delta_chunks=1)
        self.assertEqual(
            read_budget_totals(self.db, ledger_id=self.ledger)["cumulative_repairs"], 0)

    def test_g_repeated_resume_request_is_idempotent(self):
        _, receipt = self._admit(self.camp, chunk_id="chk-1")
        claim = self._capacity_wait_and_wake(self.camp.campaign_id)
        first = refresh_admission_lease_for_resume(
            self.db, admission_id=receipt.admission_id,
            wake_claim_id=claim["claim_id"], owner_id="wkr-1",
            owner_pid=os.getpid(), idempotency_key="resume-1", now=1202)
        second = refresh_admission_lease_for_resume(
            self.db, admission_id=receipt.admission_id,
            wake_claim_id=claim["claim_id"], owner_id="wkr-1",
            owner_pid=os.getpid(), idempotency_key="resume-1", now=1250)
        self.assertEqual(first["binding_id"], second["binding_id"])
        self.assertEqual(first["effective_lease_id"], second["effective_lease_id"])
        self.assertTrue(second["idempotent_replay"])
        self.assertEqual(len(list_bindings(self.db, receipt.admission_id)), 1)
        # Exactly one live lease for the admission resource.
        n = self.db._conn.execute(
            "SELECT COUNT(*) AS n FROM leases WHERE campaign_id=? AND "
            "resource_id='admission:chk-1' AND released_at=0",
            (self.camp.campaign_id,)).fetchone()["n"]
        self.assertEqual(n, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
