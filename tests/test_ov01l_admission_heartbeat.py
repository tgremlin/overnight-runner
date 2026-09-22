"""OV-01L framework defect B — bounded admission-lease heartbeat.

Normal execution (real Qwen latency + validators + Jev) can legitimately
exceed the initial 300s admission lease. The correct fix is a runner-owned
heartbeat that extends the CURRENT effective lease for the SAME holder — not
a bigger constant, not capacity-resume authority, and never a caller-named
lease.

A. same holder, live lease, valid fence        -> extension succeeds
B. different PID                               -> reject
C. same PID, different process start time       -> reject
D. different boot id                            -> reject
E. stale fence                                  -> reject
F. released lease                               -> reject
G. already expired lease                        -> reject
H. paused campaign                              -> reject
I. revoked grant                                -> reject
J. budget exhausted                             -> reject
K. repeated heartbeat                           -> monotonic; one authority
L. integration after a legitimate heartbeat     -> PASS
M. original admission lineage stays visible      -> PASS
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from overnight_runner.admission import derive_admission
from overnight_runner.admission_lease import (
    MAX_EFFECTIVE_SECONDS,
    heartbeat_admission_lease,
    list_heartbeats,
)
from overnight_runner.broker import Broker, CommandRegistry, CommandSpec, Proposal
from overnight_runner.campaign import (
    activate_campaign, create_campaign, read_budget_totals,
    update_budget_after_chunk,
)
from overnight_runner.campaign_apply import apply_campaign_patch
from overnight_runner.campaign_schemas import (
    AutonomyGrant, Budget, ChunkSpec, RepoSnapshot, content_sha256,
)
from overnight_runner.db import Database
from overnight_runner.grants import activate_grant, load_grant, revoke_grant
from overnight_runner.integration import (
    compare_and_swap_advance, ensure_campaign_worktree,
)
from overnight_runner.plans import load_plan_digest, register_plan
from overnight_runner.protected_approvals import register_protected_approval
from overnight_runner.receipts import mint_validation_receipt
from overnight_runner.resources import (
    acquire_lease, current_fence, load_lease_row, process_identity, release_lease,
    revoke_for_takeover,
)
from overnight_runner.runtime import paused_path
from overnight_runner.safety import (
    SafetyError, git_commit_all, git_head, git_init_empty, git_worktree_sha,
)
from overnight_runner.schemas import Disposition, ExecutionClass, TaskManifest
from overnight_runner.worker import _finalise


class HeartbeatBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ov01l-hb-"))
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
        self.grant = self._grant()
        self.camp = create_campaign(
            self.db, plan_id="pl-hb", grant_id="gr-hb", base_commit=self.base,
            base_tree_digest="b" * 64, repo_root=str(self.repo))
        activate_campaign(self.db, campaign_id=self.camp.campaign_id)
        self.wt = ensure_campaign_worktree(
            repo_root=self.repo, campaign_id=self.camp.campaign_id,
            base_commit=self.base, db=self.db)
        self.ledger = f"bl-{self.camp.campaign_id}"
        self.owner_id = "wkr-hb"
        self.ident = process_identity(os.getpid())
        self.admission, _ = self._admit()
        self.lease_id = self.admission.lease_id

    def tearDown(self):
        try:
            self.db.close()
        except Exception:
            pass
        try:
            paused_path().unlink()
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _grant(self, *, max_chunks=5, max_cost_microusd=0):
        register_plan(self.db, plan_id="pl-hb", approved_artifact_id="pl-hb",
                      work_package_criterion_ids={"pkg-1": {"crit-1"}})
        g = AutonomyGrant(
            schema_version="trio.grant.v1", grant_id="gr-hb", state="draft",
            plan_id="pl-hb", plan_revision=1, approved_plan_digest="0" * 64,
            repository_paths=["src/app.py"], allowed_write_paths=["src/app.py"],
            protected_paths=[], allowed_operations=["noop"],
            runtime_digest="a" * 64, model_name="gemma", model_digest="b" * 64,
            policy_profile_id="pol-1", validator_profile_ids=["noop"],
            provider_profile_id="prv-1", egress_policy_id="eg-1",
            operator_id="op-1", operator_receipt_digest="c" * 64,
            budget=Budget(schema_version="trio.budget.v1", max_chunks=max_chunks,
                          max_model_calls=50, max_tool_calls=100,
                          max_local_repairs=2, max_rechunks=1,
                          max_active_seconds=3600, max_wall_seconds=28800,
                          max_cost_microusd=max_cost_microusd,
                          context_token_budget=8192),
        )
        g = g.model_copy(update={
            "approved_plan_digest": load_plan_digest(self.db, "pl-hb")})
        register_protected_approval(
            self.db, approval_id="appr-hb", operation="activate_grant",
            grant_digest_target=content_sha256(g), operator_id="op-1",
            operator_receipt={"approval_id": "appr-hb"})
        activate_grant(self.db, grant=g, operator_id="op-1", approval_id="appr-hb")
        return load_grant(self.db, "gr-hb")

    def _admit(self, chunk_id="chk-1"):
        chunk = ChunkSpec(
            schema_version="trio.chunk.v1", chunk_id=chunk_id,
            campaign_id=self.camp.campaign_id, package_id="pkg-1", revision=1,
            title="c", objective="c", permitted_signature_paths=["src/app.py"],
            permitted_write_paths=["src/app.py"], permitted_read_paths=["src/app.py"],
            permitted_command_ids=["noop"], permitted_validator_ids=["noop"],
            required_validator_ids=["noop"], required_receipt_profiles=["noop"],
            criterion_ids=["crit-1"], idempotency_key=f"idem-{chunk_id}")
        cur = self.db._conn.execute(
            "SELECT current_commit FROM campaigns WHERE campaign_id=?",
            (self.camp.campaign_id,)).fetchone()
        snap = RepoSnapshot(schema_version="trio.repo-snapshot.v1",
                            repository_id="local",
                            commit=(cur["current_commit"] or self.base),
                            tree_digest="b" * 64)
        grant = load_grant(self.db, "gr-hb")
        return derive_admission(
            self.db, grant=grant, chunk=chunk, worker_id=self.owner_id,
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

    def _hb(self, **kw):
        args = dict(admission_id=self.admission.admission_id, owner_id=self.owner_id,
                    owner_pid=os.getpid(),
                    owner_start_time=self.ident["start_time"],
                    owner_boot_id=self.ident["boot_id"],
                    extend_seconds=300)
        args.update(kw)
        return heartbeat_admission_lease(self.db, **args)

    # ------------------------------------------------------------------ A
    def test_a_same_holder_extends(self):
        before = int(load_lease_row(self.db, self.lease_id)["expires_at"])
        res = self._hb()
        self.assertEqual(res["issuer"], "runner")
        self.assertEqual(res["lease_id"], self.lease_id)
        self.assertEqual(res["prior_expires_at"], before)
        self.assertGreater(res["new_expires_at"], before)
        self.assertEqual(int(load_lease_row(self.db, self.lease_id)["expires_at"]),
                         res["new_expires_at"])

    # ------------------------------------------------------------------ B
    def test_b_different_pid_rejected(self):
        with self.assertRaises(SafetyError) as ctx:
            self._hb(owner_pid=os.getpid() + 1)
        self.assertIn("owner_pid mismatch", str(ctx.exception))

    # ------------------------------------------------------------------ C
    def test_c_different_process_start_time_rejected(self):
        with self.assertRaises(SafetyError) as ctx:
            self._hb(owner_start_time="999999")
        self.assertIn("start time does not match the stored lease identity", str(ctx.exception))

    # ------------------------------------------------------------------ D
    def test_d_different_boot_id_rejected(self):
        with self.assertRaises(SafetyError) as ctx:
            self._hb(owner_boot_id="boot-other")
        self.assertIn("boot id does not match the stored lease identity", str(ctx.exception))

    # ------------------------------------------------------------------ E
    def test_e_stale_fence_rejected(self):
        # A fence bump WITHOUT releasing the lease: the lease is still live,
        # so the stale-fence guard (not the released guard) must fire.
        with self.db.transaction() as cur:
            cur.execute("UPDATE campaigns SET current_fence=current_fence+1 "
                        "WHERE campaign_id=?", (self.camp.campaign_id,))
        with self.assertRaises(SafetyError) as ctx:
            self._hb()
        self.assertIn("fence", str(ctx.exception))

    # ------------------------------------------------------------------ F
    def test_f_released_lease_rejected(self):
        release_lease(self.db, lease_id=self.lease_id)
        with self.assertRaises(SafetyError) as ctx:
            self._hb()
        self.assertIn("released", str(ctx.exception))

    # ------------------------------------------------------------------ G
    def test_g_expired_lease_rejected(self):
        exp = int(load_lease_row(self.db, self.lease_id)["expires_at"])
        with self.assertRaises(SafetyError) as ctx:
            self._hb(now=exp + 10)
        self.assertIn("already expired", str(ctx.exception))

    # ------------------------------------------------------------------ H
    def test_h_paused_campaign_rejected(self):
        p = paused_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("paused")
        with self.assertRaises(SafetyError) as ctx:
            self._hb()
        self.assertIn("PAUSED", str(ctx.exception))

    # ------------------------------------------------------------------ I
    def test_i_revoked_grant_rejected(self):
        revoke_grant(self.db, grant_id="gr-hb", reason="operator revoke")
        with self.assertRaises(SafetyError) as ctx:
            self._hb()
        self.assertIn("revoked", str(ctx.exception))

    # ------------------------------------------------------------------ J
    def test_j_budget_exhausted_rejected(self):
        update_budget_after_chunk(self.db, ledger_id=self.ledger, delta_chunks=5)
        with self.assertRaises(SafetyError) as ctx:
            self._hb()
        self.assertIn("budget", str(ctx.exception).lower())

    # ------------------------------------------------------------------ K
    def test_k_repeated_heartbeat_is_monotonic_and_single_authority(self):
        first = self._hb(extend_seconds=60)
        second = self._hb(extend_seconds=60)
        self.assertGreaterEqual(second["new_expires_at"], first["new_expires_at"])
        self.assertEqual(second["prior_expires_at"], first["new_expires_at"])
        # One lease row, one authority.
        n = self.db._conn.execute(
            "SELECT COUNT(*) AS n FROM leases WHERE lease_id=?", (self.lease_id,)
        ).fetchone()["n"]
        self.assertEqual(n, 1)
        self.assertEqual(len(list_heartbeats(self.db, self.admission.admission_id)), 2)
        # Idempotent replay with the same key does not add an event.
        r1 = self._hb(extend_seconds=60, idempotency_key="hb-1")
        r2 = self._hb(extend_seconds=60, idempotency_key="hb-1")
        self.assertEqual(r1["heartbeat_id"], r2["heartbeat_id"])
        self.assertTrue(r2["idempotent_replay"])
        self.assertEqual(len(list_heartbeats(self.db, self.admission.admission_id)), 3)

    # ------------------------------------------------------------------ L/M
    def test_l_integration_after_heartbeat_passes_and_lineage_visible(self):
        admission_lease_before = self.db._conn.execute(
            "SELECT lease_id FROM admissions WHERE admission_id=?",
            (self.admission.admission_id,)).fetchone()["lease_id"]
        acquired_before = int(load_lease_row(self.db, self.lease_id)["acquired_at"])
        hb = self._hb()
        self.assertGreater(hb["new_expires_at"], hb["prior_expires_at"])
        # M. original lineage visible + unchanged.
        self.assertEqual(
            self.db._conn.execute(
                "SELECT lease_id FROM admissions WHERE admission_id=?",
                (self.admission.admission_id,)).fetchone()["lease_id"],
            admission_lease_before)
        self.assertEqual(int(load_lease_row(self.db, self.lease_id)["acquired_at"]),
                         acquired_before)
        self.assertEqual(len(list_heartbeats(self.db, self.admission.admission_id)), 1)
        # L. integration succeeds with the heartbeat-extended lease.
        reg = CommandRegistry()
        reg.register(CommandSpec("noop", ["true"], "repo", 5, "read"))
        broker = Broker(repo_root=self.wt, registry=reg,
                        allowed_write_paths=["src/app.py"], allowed_create_paths=[],
                        allowed_read_paths=["src/app.py"],
                        allowed_protected_read_paths=[],
                        model_allowed_command_ids=["noop"],
                        required_validator_ids=[], approved_repo_head=None,
                        artifact_dir=self.tmp / "art")
        before_text = (self.wt / "src" / "app.py").read_text()
        broker.proposals["prop-1"] = Proposal(
            proposal_id="prop-1", op="replace_file", path="src/app.py",
            abs_path=self.wt / "src" / "app.py", before_text=before_text,
            proposed_text="CHANGED\n", preview_diff="", changed_lines=1,
            proposed_bytes=8)
        apply_campaign_patch(
            self.db, broker, str(self.repo), self.camp.campaign_id, "prop-1",
            admission_fence_generation=1, lease_id=self.lease_id,
            owner_id=self.owner_id, owner_pid=os.getpid(),
            owner_start_time=self.ident["start_time"])
        subprocess.run(["git", "add", "-A"], cwd=str(self.wt), check=True,
                       capture_output=True)
        subprocess.run(["git", "commit", "-m", "chunk"], cwd=str(self.wt),
                       check=True, capture_output=True)
        new_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(self.wt),
                                    capture_output=True, text=True).stdout.strip()
        for pc in self.wt.rglob("__pycache__"):
            shutil.rmtree(pc, ignore_errors=True)
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

        manifest = TaskManifest(
            task_id="chk-1", title="c", objective="c",
            execution_class=ExecutionClass.SOURCE_MUTATION,
            repo={"path": str(self.wt)},
            paths={"write_paths": ["src/app.py"], "create_paths": [],
                   "read_paths": ["src/app.py"], "protected_read_paths": []},
            commands={"model_allowed_command_ids": ["noop"],
                      "required_validator_ids": ["noop"], "allow_no_mutation": True},
            model_profile={"model_name": "gemma", "temperature": 0.0},
            context_budget={"max_read_bytes": 4096, "max_files_read": 4,
                            "max_files_written": 2},
            limits={"max_model_turns": 1, "max_tool_calls": 4,
                    "max_changed_files": 2, "max_diff_lines": 400,
                    "max_written_bytes": 65536, "task_timeout_seconds": 60,
                    "max_tool_result_bytes": 24576},
            acceptance_criteria=[])
        status, _c, text, rids = _finalise(
            manifest=manifest, broker=broker, disposition=Disposition.DONE,
            artifact_dir=self.tmp / "art", applied_proposals=[],
            on_validation_receipt=_mint, env_digest="e", profile_digest="p")
        self.assertEqual(status, "PASSED", text)
        compare_and_swap_advance(
            self.db, repo_root=self.repo, campaign_id=self.camp.campaign_id,
            chunk_id="chk-1", new_commit=new_commit, holder_fence_generation=1,
            actor="runner", idempotency_key="i-1", validation_receipt_ids=list(rids),
            campaign_worktree=self.wt)

    def test_n_extension_is_bounded_by_max_effective(self):
        acquired = int(load_lease_row(self.db, self.lease_id)["acquired_at"])
        res = self._hb(extend_seconds=100000)
        self.assertLessEqual(res["new_expires_at"], acquired + MAX_EFFECTIVE_SECONDS)
        with self.assertRaises(SafetyError) as ctx:
            self._hb(extend_seconds=100000)
        self.assertIn("bounded maximum", str(ctx.exception))

    # --------------------------------------------------- independent identity
    def test_n_stored_identity_matches_caller_but_not_live_proc(self):
        """Stored == caller text, but the LIVE /proc identity differs -> reject."""
        with self.db.transaction() as cur:
            cur.execute("UPDATE leases SET owner_start_time='424242' WHERE lease_id=?",
                        (self.lease_id,))
        with self.assertRaises(SafetyError) as ctx:
            self._hb(owner_start_time="424242")  # caller repeats the stored value
        self.assertIn("not alive or no longer carries", str(ctx.exception))
        self.assertEqual(len(list_heartbeats(self.db, self.admission.admission_id)), 0)

    def test_o_dead_holder_pid_rejected(self):
        dead = 999999
        with self.db.transaction() as cur:
            cur.execute("UPDATE leases SET owner_pid=?, owner_start_time='1' "
                        "WHERE lease_id=?", (dead, self.lease_id))
        with self.assertRaises(SafetyError) as ctx:
            self._hb(owner_pid=dead, owner_start_time="1")
        self.assertIn("not alive", str(ctx.exception))
        self.assertEqual(len(list_heartbeats(self.db, self.admission.admission_id)), 0)

    def test_p_omitting_identity_does_not_disable_verification(self):
        # Omission still succeeds ONLY because the live identity matches...
        res = self._hb(owner_start_time=None, owner_boot_id="")
        self.assertEqual(res["issuer"], "runner")
        # ...and omission cannot rescue a lease whose stored identity is stale.
        with self.db.transaction() as cur:
            cur.execute("UPDATE leases SET owner_start_time='31337' WHERE lease_id=?",
                        (self.lease_id,))
        with self.assertRaises(SafetyError):
            self._hb(owner_start_time=None, owner_boot_id="")

    def test_q_fence_change_during_extension_rejected(self):
        """A fence move AFTER the pre-checks but INSIDE the write transaction."""
        import unittest.mock as mock
        import overnight_runner.runtime as rt
        calls = {"n": 0}

        def paused_twice():
            calls["n"] += 1
            if calls["n"] == 2:  # the in-transaction re-check
                self.db._conn.execute(
                    "UPDATE campaigns SET current_fence=current_fence+1 "
                    "WHERE campaign_id=?", (self.camp.campaign_id,))
            return False

        with mock.patch.object(rt, "is_paused", paused_twice):
            with self.assertRaises(SafetyError) as ctx:
                self._hb()
        self.assertIn("fence moved", str(ctx.exception))
        self.assertEqual(len(list_heartbeats(self.db, self.admission.admission_id)), 0)

    def test_r_grant_revoked_before_transactional_extension(self):
        revoke_grant(self.db, grant_id="gr-hb", reason="revoke")
        with self.assertRaises(SafetyError):
            self._hb()
        self.assertEqual(len(list_heartbeats(self.db, self.admission.admission_id)), 0)

    def test_s_release_race_rejected(self):
        import unittest.mock as mock
        import overnight_runner.admission as adm
        real = adm.check_campaign_continuation

        def release_then_check(db, *, campaign_id, grant_id, now=None):
            db._conn.execute("UPDATE leases SET released_at=1 WHERE lease_id=?",
                             (self.lease_id,))
            return real(db, campaign_id=campaign_id, grant_id=grant_id, now=now)

        with mock.patch.object(adm, "check_campaign_continuation", release_then_check):
            with self.assertRaises(SafetyError) as ctx:
                self._hb()
        self.assertTrue(
            "released concurrently" in str(ctx.exception)
            or "changed concurrently" in str(ctx.exception), str(ctx.exception))
        self.assertEqual(len(list_heartbeats(self.db, self.admission.admission_id)), 0)
        # No duplicate authority was created.
        n = self.db._conn.execute(
            "SELECT COUNT(*) AS n FROM leases WHERE campaign_id=?",
            (self.camp.campaign_id,)).fetchone()["n"]
        self.assertEqual(n, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
