"""P06 follow-up #3 — independent authority / recovery corrections.

Covers the follow-up #3 correction list (items 1..16):

  * A01/A14 — campaign creation MUST use the grant-pinned plan.
  * A03/A16 — idempotency reservation crash-safety.
  * A04     — one canonical candidate snapshot + required validators
              derived from durable chunk authority.
  * A05     — six wired failpoints with durable EFFECT_UNKNOWN intent,
              real-repo reconciliation, no auto-activation.
  * A06     — boot/process identity, serialized takeover vs apply,
              patch apply bound to a live lease.
  * A07     — trusted budget acceptance evidence + single family ledger.
  * item 13 — continuation uses the trusted ledger, not chunk counts.
  * item 15 — protected approval activation is transactional.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from overnight_runner.admission import (
    check_campaign_continuation,
    derive_admission,
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
from overnight_runner.grants import activate_grant, derive_initial_ledger, load_grant
from overnight_runner.integration import (
    compare_and_swap_advance,
    ensure_campaign_worktree,
    record_crash_intent,
    reconcile_crash_window,
)
from overnight_runner.plans import load_plan_digest, register_plan
from overnight_runner.protected_approvals import (
    load_protected_approval,
    register_protected_approval,
)
from overnight_runner.receipts import (
    KIND_VALIDATION,
    _receipts_root,
    mint_validation_receipt,
)
from overnight_runner.resources import (
    acquire_lease,
    current_fence,
    holder_process_alive,
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


# ----------------------------- helpers -----------------------------

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


def _grant(db, *, plan_id="pl-1", grant_id="gr-1", packages=None,
           validator_profile_ids=None, max_frontier_escalations=0,
           max_wall_seconds=28800, max_chunks=3, grant_expires_at=0) -> AutonomyGrant:
    packages = packages or {"pkg-1": {"crit-1", "crit-2"}}
    validator_profile_ids = validator_profile_ids or ["noop"]
    grant = AutonomyGrant(
        schema_version="trio.grant.v1",
        grant_id=grant_id, state="draft",
        plan_id=plan_id, plan_revision=1, approved_plan_digest="0" * 64,
        repository_paths=["src/app.py"], allowed_write_paths=["src/app.py"],
        protected_paths=[], allowed_operations=["noop"],
        runtime_digest="a" * 64, model_name="gemma", model_digest="b" * 64,
        policy_profile_id="pol-1", validator_profile_ids=list(validator_profile_ids),
        provider_profile_id="prv-1", egress_policy_id="eg-1",
        operator_id="op-1", operator_receipt_digest="c" * 64,
        budget=Budget(
            schema_version="trio.budget.v1", max_chunks=max_chunks,
            max_model_calls=10, max_tool_calls=20, max_local_repairs=2,
            max_rechunks=1, max_frontier_escalations=max_frontier_escalations,
            max_active_seconds=3600, max_wall_seconds=max_wall_seconds,
            max_cost_microusd=1000, grant_expires_at=grant_expires_at,
            context_token_budget=8192,
        ),
    )
    register_plan(db, plan_id=plan_id, approved_artifact_id=plan_id,
                  work_package_criterion_ids=packages)
    digest = load_plan_digest(db, plan_id)
    grant = grant.model_copy(update={"approved_plan_digest": digest})
    register_protected_approval(
        db, approval_id=f"appr-{grant_id}", operation="activate_grant",
        grant_digest_target=content_sha256(grant), operator_id="op-1",
        operator_receipt={"approval_id": f"appr-{grant_id}"})
    activate_grant(db, grant=grant, operator_id="op-1", approval_id=f"appr-{grant_id}")
    return load_grant(db, grant_id)


def _id_kwargs(grant: AutonomyGrant) -> dict:
    return dict(
        current_runtime_digest=grant.runtime_digest,
        current_model_name=grant.model_name,
        current_model_digest=grant.model_digest,
        current_policy_profile_id=grant.policy_profile_id,
        current_validator_profile_ids=list(grant.validator_profile_ids),
        current_provider_profile_id=grant.provider_profile_id,
    )


def _snap(commit: str = "a" * 64, tree: str = "b" * 64) -> RepoSnapshot:
    return RepoSnapshot(schema_version="trio.repo-snapshot.v1",
                        repository_id="local", commit=commit, tree_digest=tree)


def _chunk(campaign_id: str, *, chunk_id="chk-1", idem=None,
           required_validators=None) -> ChunkSpec:
    rv = required_validators or ["noop"]
    return ChunkSpec(
        schema_version="trio.chunk.v1",
        chunk_id=chunk_id, campaign_id=campaign_id, package_id="pkg-1",
        revision=1, title="c", objective="c",
        permitted_signature_paths=["src/app.py"],
        permitted_write_paths=["src/app.py"],
        permitted_read_paths=["src/app.py"],
        permitted_command_ids=["noop"],
        permitted_validator_ids=list(rv),
        required_validator_ids=list(rv),
        required_receipt_profiles=list(rv),
        criterion_ids=["crit-1"],
        idempotency_key=idem or f"idem-{chunk_id}",
    )


def _receipts_dir() -> Path:
    return _receipts_root()


def _mint(chunk_id: str, candidate: str, validator_id: str = "noop",
          outcome: str = "pass") -> str:
    return mint_validation_receipt(
        validator_id=validator_id, validator_command=validator_id,
        validator_profile=validator_id,
        candidate_snapshot_digest=candidate, candidate_tree_state="post-apply",
        chunk_id=chunk_id, outcome=outcome, detail="test",
        receipts_dir=_receipts_dir(),
    )


def _admit_chunk(db, grant, camp, *, chunk_id, validators=None, idem=None):
    """Admit a chunk (durable authority) for integration tests."""
    validators = validators or ["noop"]
    cur_commit = db._conn.execute(
        "SELECT current_commit FROM campaigns WHERE campaign_id=?",
        (camp.campaign_id,)).fetchone()["current_commit"]
    receipt, _ = derive_admission(
        db, grant=grant,
        chunk=_chunk(camp.campaign_id, chunk_id=chunk_id, idem=idem,
                     required_validators=validators),
        worker_id="wkr-1",
        policy_profile_id=grant.policy_profile_id,
        validator_profile_ids=list(grant.validator_profile_ids),
        provider_profile_id=grant.provider_profile_id,
        current_accepted_snapshot=_snap(commit=cur_commit),
        **_id_kwargs(grant))
    return receipt


def _commit_in(worktree: Path, content: str, msg: str) -> str:
    (worktree / "src" / "app.py").write_text(content)
    subprocess.run(["git", "add", "-A"], cwd=str(worktree), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", msg], cwd=str(worktree), check=True, capture_output=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(worktree),
                          capture_output=True, text=True).stdout.strip()


# ============================================================
# A01/A14 — create_campaign uses the grant-pinned plan
# ============================================================

class TestA01CampaignPlanConsistency(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_grant_plan_a_campaign_plan_b_rejects(self):
        grant = _grant(self._db, plan_id="pl-A", grant_id="gr-A")
        # Register a second plan B with a different digest.
        register_plan(self._db, plan_id="pl-B", approved_artifact_id="pl-B",
                      work_package_criterion_ids={"pkg-1": {"crit-1"}})
        with self.assertRaises(SafetyError) as ctx:
            create_campaign(self._db, plan_id="pl-B", grant_id=grant.grant_id,
                            base_commit="a" * 64, base_tree_digest="b" * 64)
        self.assertIn("plan", str(ctx.exception).lower())

    def test_exact_grant_pinned_plan_succeeds(self):
        grant = _grant(self._db, plan_id="pl-ok", grant_id="gr-ok")
        camp = create_campaign(self._db, plan_id=grant.plan_id,
                               grant_id=grant.grant_id,
                               base_commit="a" * 64, base_tree_digest="b" * 64)
        self.assertEqual(camp.plan_id, grant.plan_id)
        # Durable metadata stores the real grant+plan digests.
        row = self._db._conn.execute(
            "SELECT grant_digest, plan_digest FROM campaigns WHERE campaign_id=?",
            (camp.campaign_id,),
        ).fetchone()
        self.assertEqual(row["grant_digest"], content_sha256(grant))
        self.assertEqual(row["plan_digest"], grant.approved_plan_digest)

    def test_same_plan_id_wrong_digest_rejects(self):
        # Activate a grant pinned to plan A, then re-register the SAME
        # plan_id with different content is impossible (content
        # addressed). Instead, simulate drift by tampering the plan row.
        grant = _grant(self._db, plan_id="pl-drift", grant_id="gr-drift")
        with self._db.transaction() as cur:
            cur.execute(
                "UPDATE approved_plans SET plan_digest=? WHERE plan_id=?",
                ("d" * 64, grant.plan_id),
            )
        with self.assertRaises(SafetyError):
            create_campaign(self._db, plan_id=grant.plan_id,
                            grant_id=grant.grant_id,
                            base_commit="a" * 64, base_tree_digest="b" * 64)


# ============================================================
# A03/A16 — idempotency reservation crash-safety
# ============================================================

class TestA03ReservationCrashSafety(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        os.environ.pop("TR_FAILPOINT_reserve_before_admission_durable", None)
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_crash_before_admission_durable_never_poisons(self):
        grant = _grant(self._db, plan_id="pl-a3", grant_id="gr-a3")
        camp = create_campaign(self._db, plan_id=grant.plan_id,
                               grant_id=grant.grant_id,
                               base_commit="a" * 64, base_tree_digest="b" * 64)
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        chunk = _chunk(camp.campaign_id, idem="idem-crash")

        os.environ["TR_FAILPOINT_reserve_before_admission_durable"] = "raise"
        with self.assertRaises(RuntimeError):
            derive_admission(self._db, grant=grant, chunk=chunk, worker_id="wkr-1",
                             policy_profile_id=grant.policy_profile_id,
                             validator_profile_ids=list(grant.validator_profile_ids),
                             provider_profile_id=grant.provider_profile_id,
                             current_accepted_snapshot=_snap(),
                             **_id_kwargs(grant))
        os.environ.pop("TR_FAILPOINT_reserve_before_admission_durable", None)

        # Reopen DB (restart) — no dangling reservation, no admission, no lease.
        self._db.close()
        self._db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        self.assertEqual(
            self._db._conn.execute("SELECT COUNT(*) AS n FROM race_admissions").fetchone()["n"], 0)
        self.assertEqual(
            self._db._conn.execute("SELECT COUNT(*) AS n FROM admissions").fetchone()["n"], 0)
        self.assertEqual(
            self._db._conn.execute("SELECT COUNT(*) AS n FROM leases").fetchone()["n"], 0)

        # Same-content retry resolves deterministically.
        grant = load_grant(self._db, grant.grant_id)
        receipt, _ = derive_admission(
            self._db, grant=grant, chunk=chunk, worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            current_accepted_snapshot=_snap(), **_id_kwargs(grant))
        self.assertEqual(receipt.chunk_id, "chk-1")
        self.assertEqual(
            self._db._conn.execute("SELECT COUNT(*) AS n FROM admissions").fetchone()["n"], 1)
        self.assertEqual(
            self._db._conn.execute("SELECT COUNT(*) AS n FROM leases").fetchone()["n"], 1)

        # A second same-content call is idempotent (still exactly one).
        derive_admission(self._db, grant=grant, chunk=chunk, worker_id="wkr-2",
                         policy_profile_id=grant.policy_profile_id,
                         validator_profile_ids=list(grant.validator_profile_ids),
                         provider_profile_id=grant.provider_profile_id,
                         current_accepted_snapshot=_snap(), **_id_kwargs(grant))
        self.assertEqual(
            self._db._conn.execute("SELECT COUNT(*) AS n FROM admissions").fetchone()["n"], 1)
        self.assertEqual(
            self._db._conn.execute("SELECT COUNT(*) AS n FROM leases").fetchone()["n"], 1)


# ============================================================
# A04 — canonical candidate snapshot + durable required validators
# ============================================================

class TestA04CanonicalCandidateAndRequiredValidators(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))
        self._repo = _make_repo(self._tmp)

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _campaign(self, *, validator_profile_ids, required_validators,
                  plan_id="pl-a4", grant_id="gr-a4"):
        grant = _grant(self._db, plan_id=plan_id, grant_id=grant_id,
                       validator_profile_ids=validator_profile_ids)
        base = git_head(self._repo)
        padded = base + "0" * 24
        camp = create_campaign(self._db, plan_id=grant.plan_id,
                               grant_id=grant.grant_id,
                               base_commit=padded, base_tree_digest="b" * 64,
                               repo_root=str(self._repo))
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        wt = ensure_campaign_worktree(repo_root=self._repo,
                                      campaign_id=camp.campaign_id,
                                      base_commit=padded, db=self._db)
        return grant, camp, wt

    def _admit(self, grant, camp, *, chunk_id, required_validators, idem=None):
        chunk = _chunk(camp.campaign_id, chunk_id=chunk_id, idem=idem,
                       required_validators=required_validators)
        cur_commit = self._db._conn.execute(
            "SELECT current_commit FROM campaigns WHERE campaign_id=?",
            (camp.campaign_id,)).fetchone()["current_commit"]
        receipt, _ = derive_admission(
            self._db, grant=grant, chunk=chunk, worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            current_accepted_snapshot=_snap(commit=cur_commit),
            **_id_kwargs(grant))
        return receipt

    def test_worktree_head_mismatch_rejects(self):
        grant, camp, wt = self._campaign(
            validator_profile_ids=["noop"], required_validators=["noop"])
        self._admit(grant, camp, chunk_id="chk-1", required_validators=["noop"])
        new_commit = _commit_in(wt, "A\n", "c1")
        candidate = git_worktree_sha(wt)
        rid = _mint("chk-1", candidate)
        # new_commit that is NOT the worktree HEAD must reject.
        with self.assertRaises(SafetyError) as ctx:
            compare_and_swap_advance(
                self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
                chunk_id="chk-1", new_commit="0" * 40,
                holder_fence_generation=1, actor="runner", idempotency_key="i1",
                validation_receipt_ids=[rid], campaign_worktree=wt,
            )
        self.assertIn("head", str(ctx.exception).lower())

    def test_wrong_candidate_receipt_rejects(self):
        grant, camp, wt = self._campaign(
            validator_profile_ids=["noop"], required_validators=["noop"])
        self._admit(grant, camp, chunk_id="chk-1", required_validators=["noop"])
        new_commit = _commit_in(wt, "A\n", "c1")
        rid = _mint("chk-1", "f" * 64)  # wrong candidate snapshot
        with self.assertRaises(SafetyError) as ctx:
            compare_and_swap_advance(
                self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
                chunk_id="chk-1", new_commit=new_commit,
                holder_fence_generation=1, actor="runner", idempotency_key="i1",
                validation_receipt_ids=[rid], campaign_worktree=wt,
            )
        self.assertIn("candidate", str(ctx.exception).lower())

    def test_two_required_validators_one_receipt_rejects(self):
        grant, camp, wt = self._campaign(
            validator_profile_ids=["vA", "vB"], required_validators=["vA", "vB"])
        self._admit(grant, camp, chunk_id="chk-1", required_validators=["vA", "vB"])
        new_commit = _commit_in(wt, "A\n", "c1")
        candidate = git_worktree_sha(wt)
        ra = _mint("chk-1", candidate, validator_id="vA")
        with self.assertRaises(SafetyError) as ctx:
            compare_and_swap_advance(
                self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
                chunk_id="chk-1", new_commit=new_commit,
                holder_fence_generation=1, actor="runner", idempotency_key="i1",
                validation_receipt_ids=[ra], campaign_worktree=wt,
            )
        self.assertIn("vB", str(ctx.exception))

    def test_both_required_receipts_pass(self):
        grant, camp, wt = self._campaign(
            validator_profile_ids=["vA", "vB"], required_validators=["vA", "vB"])
        self._admit(grant, camp, chunk_id="chk-1", required_validators=["vA", "vB"])
        new_commit = _commit_in(wt, "A\n", "c1")
        candidate = git_worktree_sha(wt)
        ra = _mint("chk-1", candidate, validator_id="vA")
        rb = _mint("chk-1", candidate, validator_id="vB")
        result = compare_and_swap_advance(
            self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
            chunk_id="chk-1", new_commit=new_commit,
            holder_fence_generation=1, actor="runner", idempotency_key="i1",
            validation_receipt_ids=[ra, rb], campaign_worktree=wt,
        )
        self.assertEqual(result.committed_new_commit, new_commit)

    def test_duplicate_receipt_does_not_satisfy_second_validator(self):
        grant, camp, wt = self._campaign(
            validator_profile_ids=["vA", "vB"], required_validators=["vA", "vB"])
        self._admit(grant, camp, chunk_id="chk-1", required_validators=["vA", "vB"])
        new_commit = _commit_in(wt, "A\n", "c1")
        candidate = git_worktree_sha(wt)
        ra1 = _mint("chk-1", candidate, validator_id="vA")
        ra2 = _mint("chk-1", candidate, validator_id="vA")
        with self.assertRaises(SafetyError):
            compare_and_swap_advance(
                self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
                chunk_id="chk-1", new_commit=new_commit,
                holder_fence_generation=1, actor="runner", idempotency_key="i1",
                validation_receipt_ids=[ra1, ra2], campaign_worktree=wt,
            )

    def test_validator_a_receipt_cannot_satisfy_required_b(self):
        grant, camp, wt = self._campaign(
            validator_profile_ids=["vA", "vB"], required_validators=["vB"])
        self._admit(grant, camp, chunk_id="chk-1", required_validators=["vB"])
        new_commit = _commit_in(wt, "A\n", "c1")
        candidate = git_worktree_sha(wt)
        ra = _mint("chk-1", candidate, validator_id="vA")
        with self.assertRaises(SafetyError):
            compare_and_swap_advance(
                self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
                chunk_id="chk-1", new_commit=new_commit,
                holder_fence_generation=1, actor="runner", idempotency_key="i1",
                validation_receipt_ids=[ra], campaign_worktree=wt,
            )

    def test_chunk_a_receipt_cannot_integrate_chunk_b(self):
        grant, camp, wt = self._campaign(
            validator_profile_ids=["noop"], required_validators=["noop"])
        self._admit(grant, camp, chunk_id="chk-A", required_validators=["noop"],
                    idem="idem-A")
        self._admit(grant, camp, chunk_id="chk-B", required_validators=["noop"],
                    idem="idem-B")
        new_commit = _commit_in(wt, "A\n", "c1")
        candidate = git_worktree_sha(wt)
        ra = _mint("chk-A", candidate)  # bound to chunk A
        with self.assertRaises(SafetyError) as ctx:
            compare_and_swap_advance(
                self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
                chunk_id="chk-B", new_commit=new_commit,
                holder_fence_generation=1, actor="runner", idempotency_key="iB",
                validation_receipt_ids=[ra], campaign_worktree=wt,
            )
        self.assertIn("chunk", str(ctx.exception).lower())


# ============================================================
# A05 — failpoints create durable EFFECT_UNKNOWN evidence
# ============================================================

class TestA05FailpointsDurableIntent(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))
        self._repo = _make_repo(self._tmp)

    def tearDown(self):
        for name in (
            "commit_exists_before_db_candidate_state",
            "db_integration_intent_before_ref_advance",
            "ref_advanced_before_integration_journal_event",
            "event_outbox_committed_before_projection_status",
            "candidate_changed_before_state_durable",
            "validator_executed_before_receipt_state_durable",
        ):
            os.environ.pop(f"TR_FAILPOINT_{name}", None)
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _campaign(self, plan_id="pl-fp", grant_id="gr-fp"):
        grant = _grant(self._db, plan_id=plan_id, grant_id=grant_id)
        base = git_head(self._repo)
        padded = base + "0" * 24
        camp = create_campaign(self._db, plan_id=grant.plan_id,
                               grant_id=grant.grant_id,
                               base_commit=padded, base_tree_digest="b" * 64,
                               repo_root=str(self._repo))
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        wt = ensure_campaign_worktree(repo_root=self._repo,
                                      campaign_id=camp.campaign_id,
                                      base_commit=padded, db=self._db)
        return grant, camp, wt

    def _window_kinds(self, campaign_id):
        rows = self._db._conn.execute(
            "SELECT kind FROM crash_windows WHERE campaign_id=?", (campaign_id,)
        ).fetchall()
        return [r["kind"] for r in rows]

    def _cas(self, grant, camp, wt, *, chunk_id="chk-fp", idem=None,
             receipt_validator="noop"):
        _admit_chunk(self._db, grant, camp, chunk_id=chunk_id, idem=idem)
        new_commit = _commit_in(wt, "FP\n", "fp")
        candidate = git_worktree_sha(wt)
        rid = _mint(chunk_id, candidate, validator_id=receipt_validator)
        return compare_and_swap_advance(
            self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
            chunk_id=chunk_id, new_commit=new_commit,
            holder_fence_generation=1, actor="runner", idempotency_key="ifp",
            validation_receipt_ids=[rid], campaign_worktree=wt,
        ), new_commit

    def test_all_four_cas_failpoints_reach_and_record(self):
        for name in (
            "commit_exists_before_db_candidate_state",
            "db_integration_intent_before_ref_advance",
            "ref_advanced_before_integration_journal_event",
            "event_outbox_committed_before_projection_status",
        ):
            with self.subTest(failpoint=name):
                # Fresh campaign each time.
                grant, camp, wt = self._campaign(
                    plan_id=f"pl-{name[:6]}", grant_id=f"gr-{name[:6]}")
                os.environ[f"TR_FAILPOINT_{name}"] = "raise"
                try:
                    with self.assertRaises(RuntimeError):
                        self._cas(grant, camp, wt, chunk_id=f"chk-{name}",
                                  idem=f"idem-{name}")
                finally:
                    os.environ.pop(f"TR_FAILPOINT_{name}")
                # Durable EFFECT_UNKNOWN intent row with the failpoint name.
                self.assertIn(name, self._window_kinds(camp.campaign_id))
                row = self._db._conn.execute(
                    "SELECT state FROM campaigns WHERE campaign_id=?",
                    (camp.campaign_id,),
                ).fetchone()
                self.assertEqual(row["state"], "EFFECT_UNKNOWN")

    def test_ref_advance_failpoint_actually_advances_ref(self):
        grant, camp, wt = self._campaign(plan_id="pl-ref", grant_id="gr-ref")
        os.environ["TR_FAILPOINT_ref_advanced_before_integration_journal_event"] = "raise"
        try:
            with self.assertRaises(RuntimeError):
                result, new_commit = self._cas(grant, camp, wt)
        finally:
            os.environ.pop("TR_FAILPOINT_ref_advanced_before_integration_journal_event")
        # The real campaign ref DID advance.
        live = subprocess.run(
            ["git", "rev-parse", f"refs/heads/campaign/{camp.campaign_id}"],
            cwd=str(self._repo), capture_output=True, text=True,
        ).stdout.strip()
        self.assertTrue(live)
        # ... but the journal is empty (crash before journal).
        n = self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM integration_journal WHERE campaign_id=?",
            (camp.campaign_id,),
        ).fetchone()["n"]
        self.assertEqual(n, 0)

    def test_candidate_change_failpoint_records_intent(self):
        grant, camp, wt = self._campaign(plan_id="pl-cand", grant_id="gr-cand")
        reg = CommandRegistry()
        reg.register(CommandSpec("noop", ["true"], "repo", 5, "read"))
        broker = Broker(repo_root=wt, registry=reg,
                        allowed_write_paths=["src/app.py"], allowed_create_paths=[],
                        allowed_read_paths=["src/app.py"],
                        allowed_protected_read_paths=[],
                        model_allowed_command_ids=["noop"],
                        required_validator_ids=["noop"],
                        artifact_dir=self._tmp / "art")
        before = (wt / "src" / "app.py").read_text()
        broker.proposals["prop-x"] = Proposal(
            proposal_id="prop-x", op="replace_file", path="src/app.py",
            abs_path=wt / "src" / "app.py", before_text=before,
            proposed_text="CHANGED\n", preview_diff="", changed_lines=1,
            proposed_bytes=8)
        fence = current_fence(self._db, camp.campaign_id)
        lease = acquire_lease(self._db, campaign_id=camp.campaign_id,
                              resource_id="chunk:chk-fp", owner_id="wkr",
                              owner_boot_id="boot", owner_pid=os.getpid(),
                              fence_generation=fence.current_generation,
                              ttl_seconds=300)
        ident = process_identity(os.getpid())
        os.environ["TR_FAILPOINT_candidate_changed_before_state_durable"] = "raise"
        try:
            with self.assertRaises(RuntimeError):
                apply_campaign_patch(
                    self._db, broker, str(self._repo), camp.campaign_id, "prop-x",
                    admission_fence_generation=fence.current_generation,
                    lease_id=lease.lease_id, owner_id="wkr",
                    owner_pid=os.getpid(), owner_start_time=ident["start_time"])
        finally:
            os.environ.pop("TR_FAILPOINT_candidate_changed_before_state_durable")
        self.assertIn("candidate_changed_before_state_durable",
                      self._window_kinds(camp.campaign_id))

    def test_validator_failpoint_records_intent(self):
        from overnight_runner.schemas import TaskManifest, ExecutionClass, Disposition
        from overnight_runner.worker import _finalise
        grant, camp, wt = self._campaign(plan_id="pl-val", grant_id="gr-val")
        reg = CommandRegistry()
        reg.register(CommandSpec("noop", ["true"], "repo", 5, "read"))
        broker = Broker(repo_root=wt, registry=reg,
                        allowed_write_paths=["src/app.py"], allowed_create_paths=[],
                        allowed_read_paths=["src/app.py"],
                        allowed_protected_read_paths=[],
                        model_allowed_command_ids=["noop"],
                        required_validator_ids=["noop"],
                        artifact_dir=self._tmp / "art2")
        manifest = TaskManifest(
            task_id="chk-fp", title="t", objective="o",
            execution_class=ExecutionClass.SOURCE_MUTATION,
            repo={"path": str(wt)},
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
            acceptance_criteria=[],
        )
        intent_calls: list[dict] = []

        def _intent(payload):
            intent_calls.append(payload)
            record_crash_intent(
                self._db, payload["boundary"], campaign_id=camp.campaign_id,
                chunk_id=payload.get("chunk_id", ""),
                repo_root=str(self._repo))

        os.environ["TR_FAILPOINT_validator_executed_before_receipt_state_durable"] = "raise"
        try:
            with self.assertRaises(RuntimeError):
                _finalise(manifest=manifest, broker=broker, disposition=Disposition.DONE,
                          artifact_dir=self._tmp / "art3", applied_proposals=[],
                          on_validation_receipt=lambda p: "",
                          on_crash_intent=_intent)
        finally:
            os.environ.pop("TR_FAILPOINT_validator_executed_before_receipt_state_durable")
        self.assertTrue(intent_calls)
        self.assertIn("validator_executed_before_receipt_state_durable",
                      self._window_kinds(camp.campaign_id))


# ============================================================
# A05 — real-repo reconciliation + no auto-activation
# ============================================================

class TestA05Reconciliation(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))
        self._repo = _make_repo(self._tmp)

    def tearDown(self):
        os.environ.pop("TR_FAILPOINT_ref_advanced_before_integration_journal_event", None)
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_reconcile_detects_real_ref_advance(self):
        grant = _grant(self._db, plan_id="pl-rec", grant_id="gr-rec")
        base = git_head(self._repo)
        padded = base + "0" * 24
        camp = create_campaign(self._db, plan_id=grant.plan_id,
                               grant_id=grant.grant_id,
                               base_commit=padded, base_tree_digest="b" * 64,
                               repo_root=str(self._repo))
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        wt = ensure_campaign_worktree(repo_root=self._repo,
                                      campaign_id=camp.campaign_id,
                                      base_commit=padded, db=self._db)
        _admit_chunk(self._db, grant, camp, chunk_id="chk-rec")
        new_commit = _commit_in(wt, "REC\n", "rec")
        candidate = git_worktree_sha(wt)
        rid = _mint("chk-rec", candidate)
        os.environ["TR_FAILPOINT_ref_advanced_before_integration_journal_event"] = "raise"
        try:
            with self.assertRaises(RuntimeError):
                compare_and_swap_advance(
                    self._db, repo_root=self._repo, campaign_id=camp.campaign_id,
                    chunk_id="chk-rec", new_commit=new_commit,
                    holder_fence_generation=1, actor="runner", idempotency_key="irec",
                    validation_receipt_ids=[rid], campaign_worktree=wt)
        finally:
            os.environ.pop("TR_FAILPOINT_ref_advanced_before_integration_journal_event")
        # Restart: close DB, reopen.
        db_path = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db"
        self._db.close()
        self._db = Database(db_path)
        result = reconcile_crash_window(self._db, campaign_id=camp.campaign_id)
        # Must detect the advance (EFFECT_UNKNOWN) — never SAFE_NOT_COMPLETED.
        self.assertNotEqual(result["decision"], "RESOLVED_TO_ACTIVE")
        decisions = [d["decision"] for d in result["window_decisions"]]
        self.assertIn("EFFECT_UNKNOWN", decisions)
        for d in result["window_decisions"]:
            self.assertNotEqual(d["decision"], "SAFE_NOT_COMPLETED")

    def test_no_windows_does_not_auto_activate(self):
        grant = _grant(self._db, plan_id="pl-noact", grant_id="gr-noact")
        camp = create_campaign(self._db, plan_id=grant.plan_id,
                               grant_id=grant.grant_id,
                               base_commit="a" * 64, base_tree_digest="b" * 64)
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        with self._db.transaction() as cur:
            cur.execute("UPDATE campaigns SET state='EFFECT_UNKNOWN' WHERE campaign_id=?",
                        (camp.campaign_id,))
        result = reconcile_crash_window(self._db, campaign_id=camp.campaign_id)
        self.assertEqual(result["decision"], "NO_WINDOWS_NO_EVIDENCE")
        state = self._db._conn.execute(
            "SELECT state FROM campaigns WHERE campaign_id=?", (camp.campaign_id,)
        ).fetchone()["state"]
        self.assertNotEqual(state, "ACTIVE")


# ============================================================
# A06 — process identity
# ============================================================

class TestA06ProcessIdentity(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_identity_semantics(self):
        real_boot = process_identity(os.getpid())["boot_id"]
        if not real_boot:
            self.skipTest("no /proc boot id on this platform")
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            ident = process_identity(proc.pid)
            self.assertTrue(holder_process_alive(
                owner_pid=proc.pid, owner_boot_id=ident["boot_id"],
                fence_generation=1, owner_start_time=ident["start_time"]))
            # Same PID, wrong boot identity -> not the lease holder.
            self.assertFalse(holder_process_alive(
                owner_pid=proc.pid,
                owner_boot_id="00000000-0000-0000-0000-000000000000",
                fence_generation=1, owner_start_time=ident["start_time"]))
            # Same PID + boot, WRONG start time (PID reuse) -> not holder.
            self.assertFalse(holder_process_alive(
                owner_pid=proc.pid, owner_boot_id=ident["boot_id"],
                fence_generation=1, owner_start_time="0"))
        finally:
            proc.terminate()
            proc.wait(timeout=5)
        # Dead child -> dead.
        self.assertFalse(holder_process_alive(
            owner_pid=proc.pid, owner_boot_id=real_boot, fence_generation=1,
            owner_start_time="0"))

    def test_lease_stores_real_identity(self):
        grant = _grant(self._db, plan_id="pl-id", grant_id="gr-id")
        camp = create_campaign(self._db, plan_id=grant.plan_id,
                               grant_id=grant.grant_id,
                               base_commit="a" * 64, base_tree_digest="b" * 64)
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        lease = acquire_lease(self._db, campaign_id=camp.campaign_id,
                              resource_id="chunk:c", owner_id="w", owner_boot_id="synth",
                              owner_pid=os.getpid(), fence_generation=1, ttl_seconds=300)
        row = self._db._conn.execute(
            "SELECT owner_boot_id, owner_start_time FROM leases WHERE lease_id=?",
            (lease.lease_id,)).fetchone()
        real_boot = process_identity(os.getpid())["boot_id"]
        if real_boot:
            self.assertEqual(row["owner_boot_id"], real_boot)
        self.assertTrue(row["owner_start_time"])


# ============================================================
# A06 — serialize takeover vs apply + live-lease binding
# ============================================================

class TestA06SerializationAndLeaseBinding(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))
        self._repo = _make_repo(self._tmp)

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _setup_campaign(self, plan_id="pl-ser", grant_id="gr-ser"):
        grant = _grant(self._db, plan_id=plan_id, grant_id=grant_id)
        base = git_head(self._repo)
        padded = base + "0" * 24
        camp = create_campaign(self._db, plan_id=grant.plan_id,
                               grant_id=grant.grant_id,
                               base_commit=padded, base_tree_digest="b" * 64,
                               repo_root=str(self._repo))
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        wt = ensure_campaign_worktree(repo_root=self._repo,
                                      campaign_id=camp.campaign_id,
                                      base_commit=padded, db=self._db)
        return grant, camp, wt

    def _broker(self, wt, observed: list | None = None):
        import sqlite3
        reg = CommandRegistry()
        reg.register(CommandSpec("noop", ["true"], "repo", 5, "read"))
        broker = Broker(repo_root=wt, registry=reg,
                        allowed_write_paths=["src/app.py"], allowed_create_paths=[],
                        allowed_read_paths=["src/app.py"],
                        allowed_protected_read_paths=[],
                        model_allowed_command_ids=["noop"],
                        required_validator_ids=["noop"],
                        artifact_dir=self._tmp / "art")
        orig = broker.apply_proposal
        dbfile = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db"

        def _wrap_apply(proposal_id):
            if observed is not None:
                # Read the CURRENT fence via a raw READ-ONLY connection
                # (no migrations -> no write lock contention). If this
                # ever shows the takeover generation, a stale writer
                # mutated AFTER the takeover.
                con = sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True, timeout=2)
                try:
                    row = con.execute(
                        "SELECT current_fence FROM campaigns WHERE campaign_id=?",
                        (self._camp_id,)).fetchone()
                    observed.append(int(row[0]) if row else -1)
                finally:
                    con.close()
            return orig(proposal_id)

        broker.apply_proposal = _wrap_apply  # type: ignore[assignment]
        return broker

    def _proposal(self, broker, wt, text="NEW\n"):
        before = (wt / "src" / "app.py").read_text()
        broker.proposals["prop-x"] = Proposal(
            proposal_id="prop-x", op="replace_file", path="src/app.py",
            abs_path=wt / "src" / "app.py", before_text=before,
            proposed_text=text, preview_diff="", changed_lines=1,
            proposed_bytes=len(text))
        return broker

    def test_stale_lease_cannot_apply_after_takeover(self):
        grant, camp, wt = self._setup_campaign(plan_id="pl-stale", grant_id="gr-stale")
        self._camp_id = camp.campaign_id
        broker = self._proposal(self._broker(wt), wt)
        old_gen = current_fence(self._db, camp.campaign_id).current_generation
        lease = acquire_lease(self._db, campaign_id=camp.campaign_id,
                              resource_id="chunk:c", owner_id="w", owner_boot_id="b",
                              owner_pid=os.getpid(), fence_generation=old_gen,
                              ttl_seconds=300)
        # Takeover first (deterministic stale case).
        revoke_for_takeover(self._db, campaign_id=camp.campaign_id, reason="t")
        before = (wt / "src" / "app.py").read_text()
        ident = process_identity(os.getpid())
        with self.assertRaises(SafetyError) as ctx:
            apply_campaign_patch(self._db, broker, str(self._repo), camp.campaign_id,
                                 "prop-x", admission_fence_generation=old_gen,
                                 lease_id=lease.lease_id, owner_id="w",
                                 owner_pid=os.getpid(),
                                 owner_start_time=ident["start_time"])
        self.assertIn("fence_stale", str(ctx.exception).lower())
        # No stale mutation happened.
        self.assertEqual((wt / "src" / "app.py").read_text(), before)

    def test_released_lease_cannot_apply(self):
        grant, camp, wt = self._setup_campaign(plan_id="pl-rel", grant_id="gr-rel")
        self._camp_id = camp.campaign_id
        broker = self._proposal(self._broker(wt), wt)
        gen = current_fence(self._db, camp.campaign_id).current_generation
        lease = acquire_lease(self._db, campaign_id=camp.campaign_id,
                              resource_id="chunk:c", owner_id="w", owner_boot_id="b",
                              owner_pid=os.getpid(), fence_generation=gen, ttl_seconds=300)
        release_lease(self._db, lease_id=lease.lease_id)
        ident = process_identity(os.getpid())
        with self.assertRaises(SafetyError) as ctx:
            apply_campaign_patch(self._db, broker, str(self._repo), camp.campaign_id,
                                 "prop-x", admission_fence_generation=gen,
                                 lease_id=lease.lease_id, owner_id="w",
                                 owner_pid=os.getpid(),
                                 owner_start_time=ident["start_time"])
        self.assertIn("released", str(ctx.exception).lower())

    def test_concurrent_takeover_never_allows_stale_mutation(self):
        """Real concurrency: no ordering lets a stale writer mutate the
        campaign worktree AFTER a takeover established the new fence."""
        for i in range(6):
            grant, camp, wt = self._setup_campaign(
                plan_id=f"pl-con-{i}", grant_id=f"gr-con-{i}")
            self._camp_id = camp.campaign_id
            observed: list[int] = []
            broker = self._proposal(self._broker(wt, observed), wt,
                                    text=f"R{i}\n")
            old_gen = current_fence(self._db, camp.campaign_id).current_generation
            lease = acquire_lease(self._db, campaign_id=camp.campaign_id,
                                  resource_id="chunk:c", owner_id="w", owner_boot_id="b",
                                  owner_pid=os.getpid(), fence_generation=old_gen,
                                  ttl_seconds=300)
            dbfile = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db"
            barrier = threading.Barrier(2)
            outcomes: dict[str, str] = {}

            ident = process_identity(os.getpid())

            def _apply():
                db2 = Database(dbfile)
                try:
                    barrier.wait(timeout=10)
                    apply_campaign_patch(db2, broker, str(self._repo),
                                         camp.campaign_id, "prop-x",
                                         admission_fence_generation=old_gen,
                                         lease_id=lease.lease_id, owner_id="w",
                                         owner_pid=os.getpid(),
                                         owner_start_time=ident["start_time"])
                    outcomes["apply"] = "ok"
                except Exception as e:  # noqa: BLE001
                    outcomes["apply"] = f"err:{type(e).__name__}"
                finally:
                    db2.close()

            def _takeover():
                db2 = Database(dbfile)
                try:
                    barrier.wait(timeout=10)
                    revoke_for_takeover(db2, campaign_id=camp.campaign_id,
                                        reason="race")
                    outcomes["takeover"] = "ok"
                finally:
                    db2.close()

            ta = threading.Thread(target=_apply)
            tb = threading.Thread(target=_takeover)
            ta.start(); tb.start()
            ta.join(timeout=20); tb.join(timeout=20)

            # INVARIANT: no write observed the post-takeover generation.
            new_gen = current_fence(self._db, camp.campaign_id).current_generation
            self.assertEqual(new_gen, old_gen + 1)
            for g in observed:
                self.assertEqual(
                    g, old_gen,
                    "stale writer mutated the worktree after takeover (fence=%s)" % g)
            # If the write was rejected WITHOUT ever reaching apply, the
            # file was not mutated. (A rejection AFTER a write is still
            # safe: the write was observed at old_gen, i.e. before the
            # takeover established the new fence.)
            if outcomes.get("apply", "").startswith("err") and not observed:
                self.assertEqual((wt / "src" / "app.py").read_text(), "BASE\n")


# ============================================================
# A07 — trusted budget evidence + single family ledger
# ============================================================

class TestA07BudgetEvidence(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _campaign(self, plan_id="pl-bd", grant_id="gr-bd", **kw):
        grant = _grant(self._db, plan_id=plan_id, grant_id=grant_id, **kw)
        camp = create_campaign(self._db, plan_id=grant.plan_id,
                               grant_id=grant.grant_id,
                               base_commit="a" * 64, base_tree_digest="b" * 64)
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        derive_admission(self._db, grant=grant,
                         chunk=_chunk(camp.campaign_id),
                         worker_id="wkr-1",
                         policy_profile_id=grant.policy_profile_id,
                         validator_profile_ids=list(grant.validator_profile_ids),
                         provider_profile_id=grant.provider_profile_id,
                         current_accepted_snapshot=_snap(), **_id_kwargs(grant))
        return grant, camp

    def test_escalation_budget_exhaustion_via_api(self):
        grant, camp = self._campaign(plan_id="pl-esc", grant_id="gr-esc",
                                     max_frontier_escalations=1)
        ledger_id = f"bl-{camp.campaign_id}"
        update_budget_after_chunk(self._db, ledger_id=ledger_id, delta_escalations=1)
        with self.assertRaises(SafetyError):
            update_budget_after_chunk(self._db, ledger_id=ledger_id, delta_escalations=1)

    def test_wall_time_exhaustion_via_api(self):
        grant, camp = self._campaign(plan_id="pl-wall", grant_id="gr-wall",
                                     max_wall_seconds=60)
        ledger_id = f"bl-{camp.campaign_id}"
        update_budget_after_chunk(self._db, ledger_id=ledger_id, delta_wall_seconds=60)
        with self.assertRaises(SafetyError):
            update_budget_after_chunk(self._db, ledger_id=ledger_id, delta_wall_seconds=1)

    def test_dimension_exhaustion_through_api_not_sql(self):
        grant, camp = self._campaign(plan_id="pl-dim", grant_id="gr-dim",
                                     max_chunks=2)
        ledger_id = f"bl-{camp.campaign_id}"
        update_budget_after_chunk(self._db, ledger_id=ledger_id, delta_chunks=2)
        with self.assertRaises(SafetyError):
            update_budget_after_chunk(self._db, ledger_id=ledger_id, delta_chunks=1)

    def test_secondary_family_ledger_refused(self):
        grant, camp = self._campaign(plan_id="pl-fam", grant_id="gr-fam")
        with self.assertRaises(SafetyError):
            derive_initial_ledger(self._db, campaign_id=camp.campaign_id,
                                  grant=grant, family_id="second-family")

    def test_new_worker_and_rechunk_reuse_same_ledger(self):
        grant, camp = self._campaign(plan_id="pl-reuse", grant_id="gr-reuse")
        ledger_id = f"bl-{camp.campaign_id}"
        update_budget_after_chunk(self._db, ledger_id=ledger_id, delta_model_calls=3)
        # Different worker + a rechunk (new revision, same package).
        derive_admission(self._db, grant=grant,
                         chunk=_chunk(camp.campaign_id, chunk_id="chk-2",
                                      idem="idem-2"),
                         worker_id="wkr-DIFFERENT",
                         policy_profile_id=grant.policy_profile_id,
                         validator_profile_ids=list(grant.validator_profile_ids),
                         provider_profile_id=grant.provider_profile_id,
                         current_accepted_snapshot=_snap(), **_id_kwargs(grant))
        same = self._db._conn.execute(
            "SELECT ledger_id FROM budget_ledgers WHERE campaign_id=?",
            (camp.campaign_id,)).fetchall()
        self.assertEqual(len(same), 1)
        self.assertEqual(same[0]["ledger_id"], ledger_id)
        totals = read_budget_totals(self._db, ledger_id=ledger_id)
        self.assertEqual(totals["cumulative_model_calls"], 3)

    def test_each_dimension_exhausted_via_trusted_api(self):
        grant, camp = self._campaign(plan_id="pl-alldim", grant_id="gr-alldim",
                                     max_wall_seconds=60, max_chunks=2)
        led = f"bl-{camp.campaign_id}"
        # Drive each counter to its bound THROUGH the trusted API, then
        # prove the next increment is refused.
        cases = [
            ({"delta_model_calls": 10}, {"delta_model_calls": 1}),
            ({"delta_tool_calls": 20}, {"delta_tool_calls": 1}),
            ({"delta_repairs": 2}, {"delta_repairs": 1}),
            ({"delta_rechunks": 1}, {"delta_rechunks": 1}),
            ({"delta_active_seconds": 3600}, {"delta_active_seconds": 1}),
            ({"delta_wall_seconds": 60}, {"delta_wall_seconds": 1}),
            ({"delta_cost_microusd": 1000}, {"delta_cost_microusd": 1}),
            ({"delta_chunks": 2}, {"delta_chunks": 1}),
            ({"delta_context_tokens": 8192}, {"delta_context_tokens": 1}),
        ]
        for to_bound, over in cases:
            with self.subTest(dim=list(to_bound)[0]):
                update_budget_after_chunk(self._db, ledger_id=led, **to_bound)
                with self.assertRaises(SafetyError):
                    update_budget_after_chunk(self._db, ledger_id=led, **over)

    def test_provider_drift_does_not_reset_ledger(self):
        grant, camp = self._campaign(plan_id="pl-prov", grant_id="gr-prov")
        ledger_id = f"bl-{camp.campaign_id}"
        update_budget_after_chunk(self._db, ledger_id=ledger_id, delta_model_calls=4)
        wrong = _id_kwargs(grant)
        wrong["current_provider_profile_id"] = "prv-OTHER"
        with self.assertRaises(SafetyError):
            derive_admission(self._db, grant=grant,
                             chunk=_chunk(camp.campaign_id, chunk_id="chk-2",
                                          idem="idem-2"),
                             worker_id="wkr-1",
                             policy_profile_id=grant.policy_profile_id,
                             validator_profile_ids=list(grant.validator_profile_ids),
                             provider_profile_id="prv-OTHER",
                             current_accepted_snapshot=_snap(),
                             **wrong)
        # Same single ledger, cumulative preserved.
        rows = self._db._conn.execute(
            "SELECT ledger_id FROM budget_ledgers WHERE campaign_id=?",
            (camp.campaign_id,)).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ledger_id"], ledger_id)
        self.assertEqual(read_budget_totals(self._db, ledger_id=ledger_id)["cumulative_model_calls"], 4)

    def test_grant_expiry_unchanged_across_transitions(self):
        expiry = int(time.time()) + 100_000
        grant, camp = self._campaign(plan_id="pl-exp", grant_id="gr-exp",
                                     grant_expires_at=expiry)
        ledger_id = f"bl-{camp.campaign_id}"
        update_budget_after_chunk(self._db, ledger_id=ledger_id, delta_chunks=1)
        bounds = json.loads(self._db._conn.execute(
            "SELECT bounds_json FROM budget_ledgers WHERE ledger_id=?",
            (ledger_id,)).fetchone()["bounds_json"])
        self.assertEqual(bounds["grant_expires_at"], expiry)


# ============================================================
# item 13 — continuation uses the trusted ledger
# ============================================================

class TestContinuationUsesLedger(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_incidental_chunk_rows_do_not_block_continuation(self):
        grant = _grant(self._db, plan_id="pl-cont", grant_id="gr-cont", max_chunks=3)
        camp = create_campaign(self._db, plan_id=grant.plan_id,
                               grant_id=grant.grant_id,
                               base_commit="a" * 64, base_tree_digest="b" * 64)
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        derive_admission(self._db, grant=grant, chunk=_chunk(camp.campaign_id),
                         worker_id="wkr-1",
                         policy_profile_id=grant.policy_profile_id,
                         validator_profile_ids=list(grant.validator_profile_ids),
                         provider_profile_id=grant.provider_profile_id,
                         current_accepted_snapshot=_snap(), **_id_kwargs(grant))
        # Insert incidental chunk rows that are NOT reflected in the ledger.
        for i in range(10):
            with self._db.transaction() as cur:
                cur.execute(
                    "INSERT INTO chunks (chunk_id, campaign_id, package_id, "
                    "idempotency_key, state, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (f"ghost-{i}", camp.campaign_id, "pkg-1", f"gidem-{i}",
                     "PROPOSED", 0, 0))
        # Continuation is allowed because the LEDGER shows 0 consumed chunks.
        check_campaign_continuation(self._db, campaign_id=camp.campaign_id,
                                    grant_id=grant.grant_id)
        # Consume the ledger chunk budget via the trusted API -> blocked.
        update_budget_after_chunk(self._db, ledger_id=f"bl-{camp.campaign_id}",
                                  delta_chunks=3)
        with self.assertRaises(SafetyError) as ctx:
            check_campaign_continuation(self._db, campaign_id=camp.campaign_id,
                                        grant_id=grant.grant_id)
        self.assertIn("budget", str(ctx.exception).lower())


# ============================================================
# item 15 — transactional protected-approval activation
# ============================================================

class TestActivationTransaction(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        os.environ.pop("TR_FAILPOINT_activate_grant_before_commit", None)
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_crash_before_commit_does_not_burn_approval(self):
        plan_id = "pl-act"
        packages = {"pkg-1": {"crit-1"}}
        register_plan(self._db, plan_id=plan_id, approved_artifact_id=plan_id,
                      work_package_criterion_ids=packages)
        digest = load_plan_digest(self._db, plan_id)
        grant = AutonomyGrant(
            schema_version="trio.grant.v1", grant_id="gr-act", state="draft",
            plan_id=plan_id, plan_revision=1, approved_plan_digest=digest,
            repository_paths=["src/app.py"], allowed_write_paths=["src/app.py"],
            protected_paths=[], allowed_operations=["noop"],
            runtime_digest="a" * 64, model_name="gemma", model_digest="b" * 64,
            policy_profile_id="pol-1", validator_profile_ids=["noop"],
            provider_profile_id="prv-1", egress_policy_id="eg-1",
            operator_id="op-1", operator_receipt_digest="c" * 64,
            budget=Budget(schema_version="trio.budget.v1", max_chunks=3),
        )
        register_protected_approval(
            self._db, approval_id="appr-act", operation="activate_grant",
            grant_digest_target=content_sha256(grant), operator_id="op-1",
            operator_receipt={"approval_id": "appr-act"})
        os.environ["TR_FAILPOINT_activate_grant_before_commit"] = "raise"
        try:
            with self.assertRaises(RuntimeError):
                activate_grant(self._db, grant=grant, operator_id="op-1",
                               approval_id="appr-act")
        finally:
            os.environ.pop("TR_FAILPOINT_activate_grant_before_commit")
        # Approval NOT burned, grant NOT inserted.
        ap = load_protected_approval(self._db, "appr-act")
        self.assertEqual(ap.consumed_at, 0)
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM grants").fetchone()["n"], 0)
        # Retry succeeds.
        activate_grant(self._db, grant=grant, operator_id="op-1",
                       approval_id="appr-act")
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM grants").fetchone()["n"], 1)
        self.assertGreater(load_protected_approval(self._db, "appr-act").consumed_at, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
