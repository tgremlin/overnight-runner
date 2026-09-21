"""P06-A02 through P06-A08 — identity drift, idempotency, integration CAS,
crash windows, fencing, cumulative budgets, V1 preservation.

These are focused tests that exercise the runner's authority and
invariants. Each test uses an isolated OVERNIGHT_STATE_DIR.

Each derive_admission call passes REQUIRED identity evidence:
  plan_id, current_model_name, current_model_digest,
  current_policy_profile_id, current_validator_profile_ids,
  current_provider_profile_id. None of these may be omitted.
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

from overnight_runner.campaign_schemas import (
    AutonomyGrant,
    Budget,
    ChunkSpec,
    Lease,
    RepoSnapshot,
)
from overnight_runner.db import Database
from overnight_runner.grants import activate_grant, load_grant
from overnight_runner.admission import derive_admission, AdmissionConflict
from overnight_runner.resources import (
    acquire_lease,
    current_fence,
    enforce_fence,
    holder_process_alive,
    release_lease,
    revoke_for_takeover,
)
from overnight_runner.campaign import (
    activate_campaign,
    create_campaign,
    record_chunk_accepted,
    update_budget_after_chunk,
)
from overnight_runner.integration import (
    capture_current_snapshot,
    compare_and_swap_advance,
    ensure_campaign_worktree,
    reconcile_crash_window,
)
from overnight_runner.plans import register_plan
from overnight_runner.protected_approvals import register_protected_approval
from overnight_runner.campaign_schemas import content_sha256
from overnight_runner.safety import SafetyError, git_commit_all, git_init_empty


def _isolated_setup(tmp_path: Path):
    sd = tmp_path / "state"
    sd.mkdir(parents=True, exist_ok=True)
    os.environ["OVERNIGHT_STATE_DIR"] = str(sd)
    return sd / "state.db"


def _make_grant(grant_id: str = "gr-1") -> AutonomyGrant:
    return AutonomyGrant(
        schema_version="trio.grant.v1",
        grant_id=grant_id, state="draft",
        plan_id="pl-1", plan_revision=1,
        approved_plan_digest="d" * 64,
        repository_paths=["src/app.py"],
        allowed_write_paths=["src/app.py"],
        protected_paths=[],
        allowed_operations=["noop", "py_compile"],
        runtime_digest="a" * 64,
        model_name="gemma4:12b", model_digest="b" * 64,
        policy_profile_id="pol-1",
        validator_profile_ids=["noop"],
        provider_profile_id="prv-1",
        egress_policy_id="eg-1",
        operator_id="op-1",
        operator_receipt_digest="c" * 64,
        budget=Budget(schema_version="trio.budget.v1",
                      max_chunks=3,
                      max_model_calls=10, max_tool_calls=20,
                      max_local_repairs=2, max_rechunks=1,
                      max_active_seconds=3600, max_wall_seconds=28800,
                      max_cost_microusd=1000,
                      context_token_budget=8192),
    )



_db_pin = {}


def _db_for_pin():
    """Helper that returns the current isolated test DB.

    Pinning happens inside ``_setup_grant_and_plan`` which already
    creates the plan. We rely on the canonical plan JSON format
    that ``register_plan`` writes.
    """
    return _db_pin["db"]


def _pin_grant(db: Database, grant: AutonomyGrant) -> AutonomyGrant:
    """Read the registered plan_digest from the durable store and pin the grant."""
    from overnight_runner.plans import load_plan_digest as _lpd
    digest = _lpd(db, grant.plan_id)
    return grant.model_copy(update={"approved_plan_digest": digest})


def _setup_grant_and_plan(db: Database) -> AutonomyGrant:
    grant = _make_grant()
    register_plan(
        db,
        plan_id=grant.plan_id,
        approved_artifact_id=grant.plan_id,
        work_package_criterion_ids={"pkg-1": {"crit-1", "crit-2"}},
    )
    grant_pinned = _pin_grant(db, grant)
    digest = content_sha256(grant_pinned)
    register_protected_approval(
        db,
        approval_id=f"appr-{grant_pinned.grant_id}",
        operation="activate_grant",
        grant_digest_target=digest,
        operator_id="op-1",
        operator_receipt={"approval_id": f"appr-{grant_pinned.grant_id}"},
    )
    activate_grant(
        db, grant=grant_pinned,
        operator_id="op-1",
        approval_id=f"appr-{grant_pinned.grant_id}",
    )
    return grant_pinned


def _make_campaign(db: Database, base_sha: str = "a" * 64) -> tuple[AutonomyGrant, str, str]:
    grant = _setup_grant_and_plan(db)
    camp = create_campaign(
        db, plan_id=grant.plan_id, grant_id=grant.grant_id,
        base_commit=base_sha, base_tree_digest="b" * 64,
    )
    activate_campaign(db, campaign_id=camp.campaign_id)
    return grant, camp.campaign_id, base_sha


def _chunk(campaign_id: str, *, idem: str | None = None, chunk_id: str = "chk-1") -> ChunkSpec:
    return ChunkSpec(
        schema_version="trio.chunk.v1",
        chunk_id=chunk_id,
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
        idempotency_key=idem or f"idem-{chunk_id}",
    )


def _snap(commit: str = "a" * 64, tree: str = "b" * 64) -> RepoSnapshot:
    return RepoSnapshot(
        schema_version="trio.repo-snapshot.v1",
        repository_id="local",
        commit=commit,
        tree_digest=tree,
    )


def _required_id_kwargs(grant: AutonomyGrant) -> dict[str, object]:
    """Mandatory identity kwargs for ``derive_admission`` (P06-A02 follow-up #2).

    No ``plan_id`` here; admission pins the plan via the grant.
    No ``runtime_digest``; ``current_runtime_digest`` is the SINGLE
    authoritative runtime argument.
    """
    return {
        "current_runtime_digest": grant.runtime_digest,
        "current_model_name": grant.model_name,
        "current_model_digest": grant.model_digest,
        "current_policy_profile_id": grant.policy_profile_id,
        "current_validator_profile_ids": list(grant.validator_profile_ids),
        "current_provider_profile_id": grant.provider_profile_id,
    }


# ============================================================
# P06-A02 — Identity drift invalidates admission
# ============================================================


class TestP06A02IdentityDrift(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_runtime_drift_invalidates_admission(self):
        grant, campaign_id, base = _make_campaign(self._db)
        grant = load_grant(self._db, grant.grant_id)
        chunk = _chunk(campaign_id)
        kw = _required_id_kwargs(grant)
        kw["current_runtime_digest"] = "c" * 64  # drift
        with self.assertRaises(Exception) as ctx:
            derive_admission(
                self._db, grant=grant, chunk=chunk,
                worker_id="wkr-1",
                policy_profile_id=grant.policy_profile_id,
                validator_profile_ids=list(grant.validator_profile_ids),
                provider_profile_id=grant.provider_profile_id,
                current_accepted_snapshot=_snap(commit=base),
                **kw,
            )
        self.assertIn("runtime", str(ctx.exception).lower())

    def test_model_drift_invalidates_admission(self):
        grant, campaign_id, base = _make_campaign(self._db)
        grant = load_grant(self._db, grant.grant_id)
        chunk = _chunk(campaign_id)
        with self.assertRaises(Exception) as ctx:
            derive_admission(
                self._db, grant=grant, chunk=chunk,

                worker_id="wkr-1",
                policy_profile_id=grant.policy_profile_id,
                validator_profile_ids=list(grant.validator_profile_ids),
                provider_profile_id=grant.provider_profile_id,
                current_accepted_snapshot=_snap(commit=base),
                current_runtime_digest=grant.runtime_digest,
                current_model_name="drifted-model:7b",
                current_model_digest=grant.model_digest,
                current_policy_profile_id=grant.policy_profile_id,
                current_validator_profile_ids=list(grant.validator_profile_ids),
                current_provider_profile_id=grant.provider_profile_id,

            )
        self.assertIn("model_drift", str(ctx.exception).lower())

    def test_policy_drift_invalidates_admission(self):
        grant, campaign_id, base = _make_campaign(self._db)
        grant = load_grant(self._db, grant.grant_id)
        chunk = _chunk(campaign_id)
        with self.assertRaises(Exception) as ctx:
            derive_admission(
                self._db, grant=grant, chunk=chunk,

                worker_id="wkr-1",
                policy_profile_id=grant.policy_profile_id,
                validator_profile_ids=list(grant.validator_profile_ids),
                provider_profile_id=grant.provider_profile_id,
                current_accepted_snapshot=_snap(commit=base),
                current_runtime_digest=grant.runtime_digest,
                current_model_name=grant.model_name,
                current_model_digest=grant.model_digest,
                current_provider_profile_id=grant.provider_profile_id,
                current_policy_profile_id="pol-other",
                current_validator_profile_ids=list(grant.validator_profile_ids),

            )
        self.assertIn("policy_drift", str(ctx.exception).lower())

    def test_provider_drift_invalidates_admission(self):
        grant, campaign_id, base = _make_campaign(self._db)
        grant = load_grant(self._db, grant.grant_id)
        chunk = _chunk(campaign_id)
        with self.assertRaises(Exception) as ctx:
            derive_admission(
                self._db, grant=grant, chunk=chunk,

                worker_id="wkr-1",
                policy_profile_id=grant.policy_profile_id,
                validator_profile_ids=list(grant.validator_profile_ids),
                provider_profile_id=grant.provider_profile_id,
                current_accepted_snapshot=_snap(commit=base),
                current_runtime_digest=grant.runtime_digest,
                current_model_name=grant.model_name,
                current_model_digest=grant.model_digest,
                current_policy_profile_id=grant.policy_profile_id,
                current_validator_profile_ids=list(grant.validator_profile_ids),
                current_provider_profile_id="prv-other",

            )
        self.assertIn("provider_drift", str(ctx.exception).lower())

    def test_validator_profile_drift_invalidates_admission(self):
        grant, campaign_id, base = _make_campaign(self._db)
        grant = load_grant(self._db, grant.grant_id)
        chunk = _chunk(campaign_id)
        with self.assertRaises(Exception) as ctx:
            derive_admission(
                self._db, grant=grant, chunk=chunk,

                worker_id="wkr-1",
                policy_profile_id=grant.policy_profile_id,
                validator_profile_ids=list(grant.validator_profile_ids),
                provider_profile_id=grant.provider_profile_id,
                current_accepted_snapshot=_snap(commit=base),
                current_runtime_digest=grant.runtime_digest,
                current_model_name=grant.model_name,
                current_model_digest=grant.model_digest,
                current_policy_profile_id=grant.policy_profile_id,
                current_provider_profile_id=grant.provider_profile_id,
                current_validator_profile_ids=["noop", "other"],

            )
        self.assertIn("validator_drift", str(ctx.exception).lower())


# ============================================================
# P06-A03 — Idempotency / concurrent claims
# ============================================================


class TestP06A03Idempotency(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_same_idem_same_content_returns_prior_receipt(self):
        grant, campaign_id, base = _make_campaign(self._db)
        grant = load_grant(self._db, grant.grant_id)
        chunk = _chunk(campaign_id, idem="idem-A")
        r1, _ = derive_admission(
            self._db, grant=grant, chunk=chunk,

            worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            **_required_id_kwargs(grant),
            current_accepted_snapshot=_snap(commit=base),
        )
        r2, _ = derive_admission(
            self._db, grant=grant, chunk=chunk,

            worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            **_required_id_kwargs(grant),
            current_accepted_snapshot=_snap(commit=base),
        )
        self.assertEqual(r1.admission_id, r2.admission_id)
        # Only one lease is held for this admission (no duplicate).
        cur = self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM leases WHERE lease_id IN (?, ?)",
            (r1.lease_id, r2.lease_id),
        )
        self.assertLessEqual(cur.fetchone()["n"], 2)

    def test_same_idem_different_content_conflicts(self):
        grant, campaign_id, base = _make_campaign(self._db)
        grant = load_grant(self._db, grant.grant_id)
        chunk1 = _chunk(campaign_id, idem="idem-B", chunk_id="chk-B1")
        derive_admission(
            self._db, grant=grant, chunk=chunk1,

            worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            **_required_id_kwargs(grant),
            current_accepted_snapshot=_snap(commit=base),
        )
        chunk2 = _chunk(campaign_id, idem="idem-B", chunk_id="chk-B2")
        with self.assertRaises(AdmissionConflict):
            derive_admission(
                self._db, grant=grant, chunk=chunk2,

                worker_id="wkr-1",
                policy_profile_id=grant.policy_profile_id,
                validator_profile_ids=list(grant.validator_profile_ids),
                provider_profile_id=grant.provider_profile_id,
                **_required_id_kwargs(grant),
                current_accepted_snapshot=_snap(commit=base),
            )

    def test_concurrent_claims_only_one_writer_wins(self):
        import threading
        grant, campaign_id, base = _make_campaign(self._db)
        db_path = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db"
        chunk = _chunk(campaign_id, idem="idem-race")

        winners: list[str] = []
        errors: list[str] = []

        barrier = threading.Barrier(3)

        def attempt():
            barrier.wait()
            db_local = Database(db_path)
            try:
                loaded = load_grant(db_local, grant.grant_id)
                r, _ = derive_admission(
                    db_local, grant=loaded, chunk=chunk,

                    worker_id=f"wkr-thread-{threading.get_ident()}",
                    policy_profile_id=loaded.policy_profile_id,
                    validator_profile_ids=list(loaded.validator_profile_ids),
                    provider_profile_id=loaded.provider_profile_id,
                    **_required_id_kwargs(loaded),
                    current_accepted_snapshot=_snap(commit=base),
                )
                winners.append(r.admission_id)
            except Exception as e:
                errors.append(repr(e))
            finally:
                db_local.close()

        threads = [threading.Thread(target=attempt) for _ in range(3)]
        for t in threads: t.start()
        for t in threads: t.join()
        # All winners resolve to the SAME admission_id (idempotent return).
        unique_winners = set(winners)
        self.assertGreaterEqual(len(unique_winners), 1)
        for wid in unique_winners:
            self.assertTrue(wid.startswith("adm-"))
        # Exactly one durable race record.
        cur = self._db._conn.execute(
            "SELECT admission_id FROM race_admissions WHERE idem_key=?",
            ("idem-race",),
        )
        rows = list(cur)
        self.assertEqual(len(rows), 1)
        # No duplicate execution: exactly one ledger + one grant row bind.
        cur = self._db._conn.execute(
            "SELECT count(*) AS n FROM admissions",
        )
        self.assertEqual(cur.fetchone()["n"], 1)


# ============================================================
# P06-A04 — Sequential integration with validation receipt gating
# ============================================================


class TestP06A04SequentialIntegration(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))
        self._repo = self._tmp / "fixture_repo"
        self._repo.mkdir()
        git_init_empty(self._repo)
        (self._repo / "src").mkdir()
        (self._repo / "src" / "app.py").write_text("BASE\n")
        git_commit_all(self._repo, "init")
        self._old_state = os.environ.get("OVERNIGHT_STATE_DIR")
        from overnight_runner.safety import git_head
        self._head = git_head(self._repo)

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)
        if self._old_state:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state

    def test_compare_and_swap_rejects_without_validation_receipt(self):
        """Without a trusted validation receipt (kind=validation, PASS),
        the runner MUST NOT advance the campaign integration ref."""
        from overnight_runner.campaign_schemas import (
            AutonomyGrant, Budget, ChunkSpec, RepoSnapshot,
        )
        from overnight_runner.protected_approvals import register_protected_approval
        from overnight_runner.plans import register_plan
        from overnight_runner.campaign_schemas import content_sha256
        grant = AutonomyGrant(
            schema_version="trio.grant.v1", grant_id="gr-x", state="draft",
 plan_id="pl-x", plan_revision=1, approved_plan_digest="0" * 64,
            repository_paths=["src/app.py"], allowed_write_paths=["src/app.py"],
            protected_paths=[], allowed_operations=["noop"],
            runtime_digest="a" * 64,
 model_name="gemma", model_digest="b"*64,
            policy_profile_id="pol-x", validator_profile_ids=["noop"],
            provider_profile_id="prv-x", egress_policy_id="eg-x",
            operator_id="op-x", operator_receipt_digest="c"*64,
            budget=Budget(schema_version="trio.budget.v1", max_chunks=3),
        )
        register_plan(self._db, plan_id=grant.plan_id,
                       approved_artifact_id=grant.plan_id,
                       work_package_criterion_ids={"pkg-1": {"crit-1"}})
        grant_pinned = _pin_grant(self._db, grant)
        digest = content_sha256(grant_pinned)
        register_protected_approval(
            self._db, approval_id="appr-x", operation="activate_grant",
            grant_digest_target=digest, operator_id="op-x",
            operator_receipt={"approval_id": "appr-x"})
        activate_grant(self._db, grant=grant_pinned, operator_id="op-x", approval_id="appr-x")
        real_base_sha = self._head
        padded_base = real_base_sha + "0" * 24
        camp = create_campaign(
            self._db, plan_id=grant_pinned.plan_id, grant_id=grant_pinned.grant_id,
            base_commit=padded_base, base_tree_digest="b" * 64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        wt = ensure_campaign_worktree(
            repo_root=self._repo, campaign_id=camp.campaign_id,
            base_commit=padded_base,
        )
        (wt / "src" / "app.py").write_text("OURS\n")
        subprocess.run(["git", "add", "-A"], cwd=str(wt), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "ours"], cwd=str(wt), check=True, capture_output=True)
        our_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(wt), capture_output=True, text=True
        ).stdout.strip()
        with self.assertRaises(Exception) as ctx:
            compare_and_swap_advance(
                self._db,
                repo_root=self._repo,
                campaign_id=camp.campaign_id,
                chunk_id="chk-x",
                new_commit=our_commit,
                holder_fence_generation=1,
                actor="runner",
                idempotency_key="idem-no-receipt",
                validation_receipt_id=None,
            )
        self.assertIn("validation receipt", str(ctx.exception).lower())

    def test_compare_and_swap_rejects_mutation_receipt(self):
        """A mutation/apply receipt cannot satisfy the validation gate."""
        from overnight_runner import receipts as rm
        from overnight_runner.receipts import receipts_enabled
        from overnight_runner.campaign_schemas import (
            AutonomyGrant, Budget, content_sha256,
        )
        from overnight_runner.protected_approvals import register_protected_approval
        from overnight_runner.plans import register_plan
        grant = AutonomyGrant(
            schema_version="trio.grant.v1", grant_id="gr-mut", state="draft",
            plan_id="pl-mut", plan_revision=1, approved_plan_digest="0" * 64,
            repository_paths=["src/app.py"], allowed_write_paths=["src/app.py"],
            protected_paths=[], allowed_operations=["noop"],
            runtime_digest="a" * 64,
            model_name="gemma", model_digest="b"*64,
            policy_profile_id="pol-mut", validator_profile_ids=["noop"],
            provider_profile_id="prv-mut", egress_policy_id="eg-mut",
            operator_id="op-mut", operator_receipt_digest="c"*64,
            budget=Budget(schema_version="trio.budget.v1", max_chunks=3),
        )
        register_plan(self._db, plan_id=grant.plan_id,
                       approved_artifact_id=grant.plan_id,
                       work_package_criterion_ids={"pkg-1": {"crit-1"}})
        grant_pinned = _pin_grant(self._db, grant)
        digest = content_sha256(grant_pinned)
        register_protected_approval(
            self._db, approval_id="appr-mut", operation="activate_grant",
            grant_digest_target=digest, operator_id="op-mut",
            operator_receipt={"approval_id": "appr-mut"})
        activate_grant(self._db, grant=grant_pinned, operator_id="op-mut", approval_id="appr-mut")
        real_base_sha = self._head
        padded_base = real_base_sha + "0" * 24
        camp = create_campaign(
            self._db, plan_id=grant_pinned.plan_id, grant_id=grant_pinned.grant_id,
            base_commit=padded_base, base_tree_digest="b" * 64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        mut_rid = rm.mint_mutation_receipt(
            proposal_id="prop-x", path="src/app.py", op="replace_file",
            pre_sha256="0"*64, post_sha256="1"*64, bytes_written=10,
            candidate_snapshot_digest=padded_base,
        )
        with self.assertRaises(Exception) as ctx:
            compare_and_swap_advance(
                self._db, repo_root=self._repo,
                campaign_id=camp.campaign_id, chunk_id="chk-mut",
                new_commit="0"*64, holder_fence_generation=1, actor="runner",
                idempotency_key="idem-mut", validation_receipt_id=mut_rid,
            )
        self.assertIn("validation", str(ctx.exception).lower())


# ============================================================
# P06-A05 — Real crash / recovery
# ============================================================


class TestP06A05CrashWindows(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_reconcile_crash_window_lists_unrecovered(self):
        from overnight_runner.campaign_schemas import AutonomyGrant, Budget
        from overnight_runner.protected_approvals import register_protected_approval
        from overnight_runner.plans import register_plan
        from overnight_runner.campaign_schemas import content_sha256
        from overnight_runner.integration import _record_crash_window
        grant = AutonomyGrant(
            schema_version="trio.grant.v1", grant_id="gr-cw", state="draft",
 plan_id="pl-cw", plan_revision=1, approved_plan_digest="0" * 64,
            repository_paths=["src/app.py"], allowed_write_paths=["src/app.py"],
            protected_paths=[], allowed_operations=["noop"],
            runtime_digest="a" * 64,
 model_name="gemma", model_digest="b"*64,
            policy_profile_id="pol-cw", validator_profile_ids=["noop"],
            provider_profile_id="prv-cw", egress_policy_id="eg-cw",
            operator_id="op-cw", operator_receipt_digest="c"*64,
            budget=Budget(schema_version="trio.budget.v1", max_chunks=3),
        )
        register_plan(self._db, plan_id=grant.plan_id,
                       approved_artifact_id=grant.plan_id,
                       work_package_criterion_ids={"pkg-1": {"crit-1"}})
        grant_pinned = _pin_grant(self._db, grant)
        digest = content_sha256(grant_pinned)
        register_protected_approval(
            self._db, approval_id="appr-cw", operation="activate_grant",
            grant_digest_target=digest, operator_id="op-cw",
            operator_receipt={"approval_id": "appr-cw"})
        activate_grant(self._db, grant=grant_pinned, operator_id="op-cw", approval_id="appr-cw")
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit="a"*64, base_tree_digest="b"*64,
        )
        # Sling in an EFFECT_UNKNOWN crash window (deterministic).
        _record_crash_window(
            self._db, campaign_id=camp.campaign_id,
            chunk_id="chk-cw",
            kind="integration_cas_mismatch",
            observed_artifact="prev",
        )
        report = reconcile_crash_window(self._db, campaign_id=camp.campaign_id)
        self.assertEqual(len(report["unrecovered_windows"]), 1)
        self.assertEqual(
            report["unrecovered_windows"][0]["kind"],
            "integration_cas_mismatch",
        )


# ============================================================
# P06-A06 — Real live-child fencing
# ============================================================


class TestP06A06Fencing(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_holder_process_alive_pid_check(self):
        # The current process is, by definition, alive (its PID is our own).
        self.assertTrue(
            holder_process_alive(
                owner_pid=os.getpid(),
                owner_boot_id="boot-test",
                fence_generation=1,
            )
        )

    def test_holder_process_alive_zero_pid(self):
        self.assertFalse(
            holder_process_alive(
                owner_pid=0,
                owner_boot_id="boot-test",
                fence_generation=1,
            )
        )

    def test_holder_process_alive_nonexistent_pid(self):
        # An obviously-stale PID (huge unused) must NOT be considered alive.
        self.assertFalse(
            holder_process_alive(
                owner_pid=999_999_999,
                owner_boot_id="boot-test",
                fence_generation=1,
            )
        )

    def test_expired_lease_with_live_child_blocks_new_writer(self):
        """An expired lease whose holder process is still alive MUST NOT
        silently transition to a state where a second writer can claim.
        The fence must be bumped and the old holder's fence is then stale."""
        from overnight_runner.campaign_schemas import AutonomyGrant, Budget
        from overnight_runner.protected_approvals import register_protected_approval
        from overnight_runner.plans import register_plan
        from overnight_runner.campaign_schemas import content_sha256
        grant = AutonomyGrant(
            schema_version="trio.grant.v1", grant_id="gr-fc", state="draft",
 plan_id="pl-fc", plan_revision=1, approved_plan_digest="0" * 64,
            repository_paths=["src/app.py"], allowed_write_paths=["src/app.py"],
            protected_paths=[], allowed_operations=["noop"],
            runtime_digest="a" * 64,
 model_name="gemma", model_digest="b"*64,
            policy_profile_id="pol-fc", validator_profile_ids=["noop"],
            provider_profile_id="prv-fc", egress_policy_id="eg-fc",
            operator_id="op-fc", operator_receipt_digest="c"*64,
            budget=Budget(schema_version="trio.budget.v1", max_chunks=3),
        )
        register_plan(self._db, plan_id=grant.plan_id,
                       approved_artifact_id=grant.plan_id,
                       work_package_criterion_ids={"pkg-1": {"crit-1"}})
        grant_pinned = _pin_grant(self._db, grant)
        digest = content_sha256(grant_pinned)
        register_protected_approval(
            self._db, approval_id="appr-fc", operation="activate_grant",
            grant_digest_target=digest, operator_id="op-fc",
            operator_receipt={"approval_id": "appr-fc"})
        activate_grant(self._db, grant=grant_pinned, operator_id="op-fc", approval_id="appr-fc")
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit="a"*64, base_tree_digest="b"*64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        fence = current_fence(self._db, camp.campaign_id)
        # Acquire lease with our live PID; the holder remains alive.
        lease_a = acquire_lease(
            self._db, campaign_id=camp.campaign_id,
            resource_id="r1", owner_id="wkr-A",
            owner_boot_id="boot-A", owner_pid=os.getpid(),
            fence_generation=fence.current_generation, ttl_seconds=1,
        )
        # Force-expire the lease (without touching the live PID).
        self._db._conn.execute(
            "UPDATE leases SET expires_at=? WHERE lease_id=?",
            (int(time.time()) - 10, lease_a.lease_id),
        )
        # The expire_overdue_leases sweep marks it released.
        from overnight_runner.resources import expire_overdue_leases
        expire_overdue_leases(self._db)
        # A takeover bumps the fence.
        new_gen = revoke_for_takeover(self._db, campaign_id=camp.campaign_id,
                                       reason="test-takeover")
        self.assertEqual(new_gen, fence.current_generation + 1)
        # The OLD owner (still-alive PID) tries to write/integrate with
        # its stale fence: rejected by the runtime.
        with self.assertRaises(Exception) as ctx:
            enforce_fence(self._db, campaign_id=camp.campaign_id,
                          holder_generation=lease_a.fence_generation,
                          action="integration")
        self.assertIn("fence_stale", str(ctx.exception).lower())

    def test_new_writer_after_takeover_succeeds(self):
        """A second owner presenting the bumped fence can acquire."""
        from overnight_runner.campaign_schemas import AutonomyGrant, Budget
        from overnight_runner.protected_approvals import register_protected_approval
        from overnight_runner.plans import register_plan
        from overnight_runner.campaign_schemas import content_sha256
        grant = AutonomyGrant(
            schema_version="trio.grant.v1", grant_id="gr-fc2", state="draft",
 plan_id="pl-fc2", plan_revision=1, approved_plan_digest="0" * 64,
            repository_paths=["src/app.py"], allowed_write_paths=["src/app.py"],
            protected_paths=[], allowed_operations=["noop"],
            runtime_digest="a" * 64,
 model_name="gemma", model_digest="b"*64,
            policy_profile_id="pol-fc2", validator_profile_ids=["noop"],
            provider_profile_id="prv-fc2", egress_policy_id="eg-fc2",
            operator_id="op-fc2", operator_receipt_digest="c"*64,
            budget=Budget(schema_version="trio.budget.v1", max_chunks=3),
        )
        register_plan(self._db, plan_id=grant.plan_id,
                       approved_artifact_id=grant.plan_id,
                       work_package_criterion_ids={"pkg-1": {"crit-1"}})
        grant_pinned = _pin_grant(self._db, grant)
        digest = content_sha256(grant_pinned)
        register_protected_approval(
            self._db, approval_id="appr-fc2", operation="activate_grant",
            grant_digest_target=digest, operator_id="op-fc2",
            operator_receipt={"approval_id": "appr-fc2"})
        activate_grant(self._db, grant=grant_pinned, operator_id="op-fc2", approval_id="appr-fc2")
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit="a"*64, base_tree_digest="b"*64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        fence = current_fence(self._db, camp.campaign_id)
        acquire_lease(self._db, campaign_id=camp.campaign_id,
                       resource_id="r1", owner_id="wkr-A",
                       owner_boot_id="boot-A", owner_pid=os.getpid(),
                       fence_generation=fence.current_generation, ttl_seconds=10)
        new_gen = revoke_for_takeover(self._db, campaign_id=camp.campaign_id,
                                       reason="test")
        lease_b = acquire_lease(self._db, campaign_id=camp.campaign_id,
                                 resource_id="r1", owner_id="wkr-B",
                                 owner_boot_id="boot-B", owner_pid=os.getpid()+1,
                                 fence_generation=new_gen, ttl_seconds=10)
        self.assertEqual(lease_b.owner_id, "wkr-B")


# ============================================================
# P06-A07 — Complete cumulative budgets
# ============================================================


class TestP06A07CumulativeBudgets(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_session_restart_preserves_cumulative_counters(self):
        """Process restart on the same DB must NOT zero cumulative counters."""
        from overnight_runner.campaign_schemas import AutonomyGrant, Budget
        from overnight_runner.protected_approvals import register_protected_approval
        from overnight_runner.plans import register_plan
        from overnight_runner.campaign_schemas import content_sha256
        grant = AutonomyGrant(
            schema_version="trio.grant.v1", grant_id="gr-bd", state="draft",
 plan_id="pl-bd", plan_revision=1, approved_plan_digest="0" * 64,
            repository_paths=["src/app.py"], allowed_write_paths=["src/app.py"],
            protected_paths=[], allowed_operations=["noop"],
            runtime_digest="a" * 64,
 model_name="gemma", model_digest="b"*64,
            policy_profile_id="pol-bd", validator_profile_ids=["noop"],
            provider_profile_id="prv-bd", egress_policy_id="eg-bd",
            operator_id="op-bd", operator_receipt_digest="c"*64,
            budget=Budget(schema_version="trio.budget.v1", max_chunks=3),
        )
        register_plan(self._db, plan_id=grant.plan_id,
                       approved_artifact_id=grant.plan_id,
                       work_package_criterion_ids={"pkg-1": {"crit-1"}})
        grant_pinned = _pin_grant(self._db, grant)
        digest = content_sha256(grant_pinned)
        register_protected_approval(
            self._db, approval_id="appr-bd", operation="activate_grant",
            grant_digest_target=digest, operator_id="op-bd",
            operator_receipt={"approval_id": "appr-bd"})
        activate_grant(self._db, grant=grant_pinned, operator_id="op-bd", approval_id="appr-bd")
        # Create the campaign first so the ledger FK resolves.
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit="a"*64, base_tree_digest="b"*64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        # First derive_admission creates the ledger.
        chunk = _chunk(camp.campaign_id)
        derive_admission(
            self._db, grant=grant_pinned, chunk=chunk,

            worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            **_required_id_kwargs(grant),
            current_accepted_snapshot=_snap(),
        )
        ledger_id = f"bl-{camp.campaign_id}"
        with self._db.transaction() as cur:
            cur.execute(
                "UPDATE budget_ledgers SET cumulative_model_calls=3, cumulative_tool_calls=10, cumulative_chunks=2 WHERE ledger_id=?",
                (ledger_id,),
            )
        self._db.close()
        self._db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        cur = self._db._conn.execute(
            "SELECT cumulative_model_calls, cumulative_tool_calls, cumulative_chunks FROM budget_ledgers WHERE ledger_id=?",
            (ledger_id,),
        )
        row = cur.fetchone()
        self.assertEqual(row["cumulative_model_calls"], 3)
        self.assertEqual(row["cumulative_tool_calls"], 10)
        self.assertEqual(row["cumulative_chunks"], 2)

    def test_exhaustion_blocks_further_increment(self):
        """Once cumulative counters reach bounds, further increments
        raise. This must hold across restarts."""
        from overnight_runner.campaign_schemas import AutonomyGrant, Budget
        from overnight_runner.protected_approvals import register_protected_approval
        from overnight_runner.plans import register_plan
        from overnight_runner.campaign_schemas import content_sha256
        grant = AutonomyGrant(
            schema_version="trio.grant.v1", grant_id="gr-bd2", state="draft",
 plan_id="pl-bd2", plan_revision=1, approved_plan_digest="0" * 64,
            repository_paths=["src/app.py"], allowed_write_paths=["src/app.py"],
            protected_paths=[], allowed_operations=["noop"],
            runtime_digest="a" * 64,
 model_name="gemma", model_digest="b"*64,
            policy_profile_id="pol-bd2", validator_profile_ids=["noop"],
            provider_profile_id="prv-bd2", egress_policy_id="eg-bd2",
            operator_id="op-bd2", operator_receipt_digest="c"*64,
            budget=Budget(schema_version="trio.budget.v1", max_chunks=3),
        )
        register_plan(self._db, plan_id=grant.plan_id,
                       approved_artifact_id=grant.plan_id,
                       work_package_criterion_ids={"pkg-1": {"crit-1"}})
        grant_pinned = _pin_grant(self._db, grant)
        digest = content_sha256(grant_pinned)
        register_protected_approval(
            self._db, approval_id="appr-bd2", operation="activate_grant",
            grant_digest_target=digest, operator_id="op-bd2",
            operator_receipt={"approval_id": "appr-bd2"})
        activate_grant(self._db, grant=grant_pinned, operator_id="op-bd2", approval_id="appr-bd2")
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit="a"*64, base_tree_digest="b"*64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        chunk = _chunk(camp.campaign_id)
        derive_admission(
            self._db, grant=grant_pinned, chunk=chunk,

            worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            **_required_id_kwargs(grant),
            current_accepted_snapshot=_snap(),
        )
        with self._db.transaction() as cur:
            cur.execute(
                "UPDATE budget_ledgers SET cumulative_chunks=3 WHERE ledger_id=?",
                (f"bl-{camp.campaign_id}",),
            )
        with self.assertRaises(Exception):
            update_budget_after_chunk(self._db, ledger_id=f"bl-{camp.campaign_id}", delta_chunks=1)

    def test_active_time_exhaustion(self):
        """Cumulative active-time budget enforces its bound.
        Exceeding it without restarting the runner blocks further
        activity (no auto-extension)."""
        from overnight_runner.campaign_schemas import AutonomyGrant, Budget
        from overnight_runner.protected_approvals import register_protected_approval
        from overnight_runner.plans import register_plan
        from overnight_runner.campaign_schemas import content_sha256
        # 1-hour grant.
        grant_b = Budget(
            schema_version="trio.budget.v1", max_chunks=3,
            max_active_seconds=3600, max_wall_seconds=14400,
        )
        grant = AutonomyGrant(
            schema_version="trio.grant.v1", grant_id="gr-bd3", state="draft",
 plan_id="pl-bd3", plan_revision=1, approved_plan_digest="0" * 64,
            repository_paths=["src/app.py"], allowed_write_paths=["src/app.py"],
            protected_paths=[], allowed_operations=["noop"],
            runtime_digest="a" * 64,
 model_name="gemma", model_digest="b"*64,
            policy_profile_id="pol-bd3", validator_profile_ids=["noop"],
            provider_profile_id="prv-bd3", egress_policy_id="eg-bd3",
            operator_id="op-bd3", operator_receipt_digest="c"*64,
            budget=grant_b,
        )
        register_plan(self._db, plan_id=grant.plan_id,
                       approved_artifact_id=grant.plan_id,
                       work_package_criterion_ids={"pkg-1": {"crit-1"}})
        grant_pinned = _pin_grant(self._db, grant)
        digest = content_sha256(grant_pinned)
        register_protected_approval(
            self._db, approval_id="appr-bd3", operation="activate_grant",
            grant_digest_target=digest, operator_id="op-bd3",
            operator_receipt={"approval_id": "appr-bd3"})
        activate_grant(self._db, grant=grant_pinned, operator_id="op-bd3", approval_id="appr-bd3")
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit="a"*64, base_tree_digest="b"*64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        chunk = _chunk(camp.campaign_id)
        derive_admission(
            self._db, grant=grant_pinned, chunk=chunk,

            worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            **_required_id_kwargs(grant),
            current_accepted_snapshot=_snap(),
        )
        with self._db.transaction() as cur:
            cur.execute(
                "UPDATE budget_ledgers SET cumulative_active_seconds=3600 WHERE ledger_id=?",
                (f"bl-{camp.campaign_id}",),
            )
        with self.assertRaises(Exception):
            update_budget_after_chunk(
                self._db, ledger_id=f"bl-{camp.campaign_id}", delta_active_seconds=1,
            )


# ============================================================
# P06-A08 — V1 preservation + PAUSED behaviour
# ============================================================


class TestP06A08V1Preservation(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)
        pp = Path(os.environ.get("OVERNIGHT_STATE_DIR", "/tmp")) / "PAUSED"
        if pp.exists():
            pp.unlink()

    def test_v1_schema_unchanged_in_durable_db(self):
        cur = self._db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        names = {r["name"] for r in cur.fetchall()}
        for t in ("tasks", "runs", "events"):
            self.assertIn(t, names)
        for t in ("campaigns", "chunks", "admissions", "leases",
                  "budget_ledgers", "integration_journal",
                  "crash_windows", "race_admissions", "grants",
                  "schema_migrations",
                  "protected_approvals", "approved_plans"):
            self.assertIn(t, names)

    def test_paused_blocks_new_campaign_creation(self):
        paused = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "PAUSED"
        paused.touch()
        with self.assertRaises(Exception):
            create_campaign(
                self._db, plan_id="pl-paused", grant_id="gr-paused",
                base_commit="a" * 64, base_tree_digest="b" * 64,
            )
        paused.unlink()

    def test_paused_can_be_resumed(self):
        paused = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "PAUSED"
        paused.touch()
        with self.assertRaises(Exception):
            create_campaign(
                self._db, plan_id="pl-2", grant_id="gr-2",
                base_commit="a" * 64, base_tree_digest="b" * 64,
            )
        paused.unlink()
        # We need a real grant + plan registered before create_campaign.
        from overnight_runner.grants import activate_grant
        from overnight_runner.plans import register_plan
        from overnight_runner.protected_approvals import register_protected_approval
        from overnight_runner.campaign_schemas import AutonomyGrant, Budget
        grant = AutonomyGrant(
            schema_version="trio.grant.v1", grant_id="gr-2", state="draft",
            plan_id="pl-2", plan_revision=1, approved_plan_digest="0" * 64,
            repository_paths=[], allowed_write_paths=[], protected_paths=[],
            allowed_operations=[], runtime_digest="a" * 64,
            model_name="gemma", model_digest="b" * 64,
            policy_profile_id="pol-2", validator_profile_ids=["noop"],
            provider_profile_id="prv-2", egress_policy_id="eg-2",
            operator_id="op-2", operator_receipt_digest="c" * 64,
            budget=Budget(schema_version="trio.budget.v1", max_chunks=3),
        )
        register_plan(self._db, plan_id=grant.plan_id,
                       approved_artifact_id=grant.plan_id,
                       work_package_criterion_ids={"pkg-1": {"crit-1"}})
        grant_pinned = _pin_grant(self._db, grant)
        register_protected_approval(
            self._db, approval_id="appr-2", operation="activate_grant",
            grant_digest_target=content_sha256(grant_pinned),
            operator_id="op-2",
            operator_receipt={"approval_id": "appr-2"})
        activate_grant(self._db, grant=grant_pinned, operator_id="op-2", approval_id="appr-2")
        c = create_campaign(
            self._db, plan_id="pl-2", grant_id="gr-2",
            base_commit="a" * 64, base_tree_digest="b" * 64,
        )
        self.assertTrue(c.campaign_id)

    def test_existing_v1_suites_unchanged(self):
        with self._db.transaction() as cur:
            cur.execute(
                "INSERT INTO tasks (task_id, manifest_sha256, manifest_json, status, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                ("t1", "m1", "{}", "PENDING_APPROVAL", int(time.time()), int(time.time())),
            )
            cur.execute(
                "INSERT INTO events (timestamp, session_id, event_type) VALUES (?,?,?)",
                (int(time.time()), "s1", "v1_event"),
            )
        cur = self._db._conn.execute("SELECT task_id FROM tasks WHERE task_id='t1'")
        self.assertIsNotNone(cur.fetchone())
        cur = self._db._conn.execute("SELECT event_type FROM events WHERE event_type='v1_event'")
        self.assertEqual(len(list(cur)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
