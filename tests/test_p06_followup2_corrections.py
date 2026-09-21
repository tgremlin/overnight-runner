"""P06 follow-up #2 — Additional correction tests (A05, A06, A07).

Tests added by the P06 follow-up #2 review. Each test exercises a
specific correction from the 17-correction list:

  * A05 — Real EFFECT_UNKNOWN state + failpoint recovery
  * A06 — Expired live owner blocks direct reacquire (live child test)
  * A06 — Fenced campaign mutation boundary (BrokerPort wrapper)
  * A07 — Complete authoritative budget API
  * A11 — One-shot protected approval consumption
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

from overnight_runner.campaign_apply import apply_campaign_patch
from overnight_runner.campaign_schemas import (
    AutonomyGrant,
    Budget,
    BudgetLedgerEntry,
    ChunkSpec,
    RepoSnapshot,
    content_sha256,
)
from overnight_runner.db import Database
from overnight_runner.grants import activate_grant, load_grant, revoke_grant
from overnight_runner.admission import derive_admission, check_campaign_continuation
from overnight_runner.campaign import (
    activate_campaign,
    create_campaign,
    record_chunk_accepted,
    update_budget_after_chunk,
    read_budget_totals,
    pause_campaign,
    resume_campaign,
    cancel_campaign,
)
from overnight_runner.integration import (
    compare_and_swap_advance,
    ensure_campaign_worktree,
    reconcile_crash_window,
    _record_crash_window,
    trigger_crash_failpoint,
)
from overnight_runner.plans import register_plan, load_plan_digest
from overnight_runner.protected_approvals import (
    register_protected_approval,
    load_protected_approval,
    consume_protected_approval,
)
from overnight_runner.resources import (
    acquire_lease,
    current_fence,
    expire_overdue_leases,
    holder_process_alive,
    release_lease,
    revoke_for_takeover,
)
from overnight_runner.broker import Broker, CommandRegistry, CommandSpec
from overnight_runner.receipts import (
    KIND_VALIDATION,
    mint_validation_receipt,
    verify_receipt,
    _receipts_root,
)
from overnight_runner.safety import (
    SafetyError,
    git_commit_all,
    git_init_empty,
    git_head,
)


# ----------------------------- Helpers -----------------------------

def _isolated_setup(tmp_path):
    sd = tmp_path / "state"
    sd.mkdir(parents=True, exist_ok=True)
    os.environ["OVERNIGHT_STATE_DIR"] = str(sd)
    os.environ["OVERNIGHT_RECEIPTS"] = "1"
    return sd / "state.db"


def _pin_plan(db, grant: AutonomyGrant, packages: dict) -> AutonomyGrant:
    register_plan(db, plan_id=grant.plan_id,
                   approved_artifact_id=grant.plan_id,
                   work_package_criterion_ids=packages)
    digest = load_plan_digest(db, grant.plan_id)
    return grant.model_copy(update={"approved_plan_digest": digest})


def _grant_active(db, *, plan_id="pl-1", packages=None, grant_id="gr-1",
                   operator_id="op-1") -> AutonomyGrant:
    packages = packages or {"pkg-1": {"crit-1", "crit-2"}}
    grant = AutonomyGrant(
        schema_version="trio.grant.v1",
        grant_id=grant_id, state="draft",
        plan_id=plan_id, plan_revision=1, approved_plan_digest="0" * 64,
        repository_paths=["src/app.py"], allowed_write_paths=["src/app.py"],
        protected_paths=[], allowed_operations=["noop"],
        runtime_digest="a" * 64, model_name="gemma", model_digest="b" * 64,
        policy_profile_id="pol-1", validator_profile_ids=["noop"],
        provider_profile_id="prv-1", egress_policy_id="eg-1",
        operator_id=operator_id, operator_receipt_digest="c" * 64,
        budget=Budget(schema_version="trio.budget.v1", max_chunks=3,
                       max_model_calls=10, max_tool_calls=20,
                       max_local_repairs=2, max_rechunks=1,
                       max_active_seconds=3600, max_wall_seconds=28800,
                       max_cost_microusd=1000, context_token_budget=8192),
    )
    grant_pinned = _pin_plan(db, grant, packages)
    register_protected_approval(
        db, approval_id=f"appr-{grant_id}", operation="activate_grant",
        grant_digest_target=content_sha256(grant_pinned),
        operator_id=operator_id, operator_receipt={"approval_id": f"appr-{grant_id}"})
    activate_grant(db, grant=grant_pinned, operator_id=operator_id,
                    approval_id=f"appr-{grant_id}")
    return grant_pinned


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
    return RepoSnapshot(
        schema_version="trio.repo-snapshot.v1",
        repository_id="local",
        commit=commit, tree_digest=tree,
    )


def _chunk(campaign_id: str, *, idem: str | None = None) -> ChunkSpec:
    return ChunkSpec(
        schema_version="trio.chunk.v1",
        chunk_id="chk-1", campaign_id=campaign_id, package_id="pkg-1",
        revision=1, title="c", objective="c",
        permitted_signature_paths=["src/app.py"],
        permitted_write_paths=["src/app.py"],
        permitted_read_paths=["src/app.py"],
        permitted_command_ids=["noop"],
        permitted_validator_ids=["noop"],
        required_validator_ids=["noop"],
        required_receipt_profiles=["noop"],
        criterion_ids=["crit-1"],
        idempotency_key=idem or "idem-1",
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "fixture_repo"
    repo.mkdir()
    git_init_empty(repo)
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("BASE\n")
    git_commit_all(repo, "init")
    return repo


# ============================================================
# A01 — Plan pin binds to grant
# ============================================================

class TestA01GrantPlanBinding(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_plan_pin_blocks_activation_with_different_digest(self):
        """A grant with the wrong approved_plan_digest MUST NOT activate."""
        grant = AutonomyGrant(
            schema_version="trio.grant.v1", grant_id="gr-x", state="draft",
            plan_id="pl-x", plan_revision=1, approved_plan_digest="0" * 64,
            repository_paths=["src/app.py"], allowed_write_paths=["src/app.py"],
            protected_paths=[], allowed_operations=["noop"],
            runtime_digest="a" * 64, model_name="gemma", model_digest="b" * 64,
            policy_profile_id="pol-x", validator_profile_ids=["noop"],
            provider_profile_id="prv-x", egress_policy_id="eg-x",
            operator_id="op-x", operator_receipt_digest="c" * 64,
            budget=Budget(schema_version="trio.budget.v1", max_chunks=3),
        )
        # Register a plan but with content whose digest is NOT "0"*64.
        register_plan(self._db, plan_id=grant.plan_id,
                       approved_artifact_id=grant.plan_id,
                       work_package_criterion_ids={"pkg-1": {"crit-1"}})
        real_digest = load_plan_digest(self._db, grant.plan_id)
        self.assertNotEqual(real_digest, "0" * 64)
        # The grant has approved_plan_digest="0"*64 which doesn't match.
        register_protected_approval(
            self._db, approval_id="appr-x", operation="activate_grant",
            grant_digest_target=content_sha256(grant),
            operator_id="op-x", operator_receipt={"approval_id": "appr-x"})
        with self.assertRaises(Exception) as ctx:
            activate_grant(self._db, grant=grant, operator_id="op-x", approval_id="appr-x")
        self.assertIn("plan_digest mismatch" in str(ctx.exception).lower() or
                       "approved_plan_digest" in str(ctx.exception).lower(), [True])

    def test_campaign_durable_metadata_uses_real_digests(self):
        """campaigns.grant_digest and campaigns.plan_digest must be real digests."""
        grant = _grant_active(self._db, plan_id="pl-meta", grant_id="gr-meta")
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit="a" * 64, base_tree_digest="b" * 64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        cur = self._db._conn.execute(
            "SELECT grant_digest, plan_digest FROM campaigns WHERE campaign_id=?",
            (camp.campaign_id,),
        )
        row = cur.fetchone()
        self.assertNotEqual(row["grant_digest"], grant.plan_id)
        self.assertNotEqual(row["plan_digest"], grant.plan_id)
        stored_grant = load_grant(self._db, grant.grant_id)
        self.assertEqual(row["grant_digest"], content_sha256(stored_grant))
        self.assertEqual(row["plan_digest"], load_plan_digest(self._db, grant.plan_id))


# ============================================================
# A02 — Single authoritative current_runtime_digest
# ============================================================

class TestA02SingleCurrentRuntime(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_current_runtime_digest_required(self):
        grant = _grant_active(self._db)
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit="a" * 64, base_tree_digest="b" * 64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        kw = _id_kwargs(grant)
        kw["current_runtime_digest"] = None
        with self.assertRaises(Exception) as ctx:
            derive_admission(
                self._db, grant=grant, chunk=_chunk(camp.campaign_id),
                worker_id="wkr-1",
                policy_profile_id=grant.policy_profile_id,
                validator_profile_ids=list(grant.validator_profile_ids),
                provider_profile_id=grant.provider_profile_id,
                current_accepted_snapshot=_snap(),
                **kw,
            )
        self.assertIn("current_runtime_digest", str(ctx.exception).lower())

    def test_wrong_current_runtime_rejects(self):
        grant = _grant_active(self._db)
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit="a" * 64, base_tree_digest="b" * 64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        kw = _id_kwargs(grant)
        kw["current_runtime_digest"] = "wrong" + "0" * 58
        with self.assertRaises(Exception) as ctx:
            derive_admission(
                self._db, grant=grant, chunk=_chunk(camp.campaign_id),
                worker_id="wkr-1",
                policy_profile_id=grant.policy_profile_id,
                validator_profile_ids=list(grant.validator_profile_ids),
                provider_profile_id=grant.provider_profile_id,
                current_accepted_snapshot=_snap(),
                **kw,
            )
        self.assertIn("runtime", str(ctx.exception).lower())


# ============================================================
# A03 — Atomic idempotency claim
# ============================================================

class TestA03AtomicIdempotency(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_concurrent_same_key_same_content_single_admission(self):
        """N concurrent same-key same-content callers all resolve to the same admission."""
        import threading
        grant = _grant_active(self._db)
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit="a" * 64, base_tree_digest="b" * 64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        db_path = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db"
        chunk = _chunk(camp.campaign_id, idem="idem-concurrent")

        winners: list[str] = []
        errors: list[str] = []

        def attempt():
            db_local = Database(db_path)
            try:
                loaded = load_grant(db_local, grant.grant_id)
                r, _ = derive_admission(
                    db_local, grant=loaded, chunk=chunk,
                    worker_id=f"wkr-{threading.get_ident()}",
                    policy_profile_id=loaded.policy_profile_id,
                    validator_profile_ids=list(loaded.validator_profile_ids),
                    provider_profile_id=loaded.provider_profile_id,
                    current_accepted_snapshot=_snap(),
                    **_id_kwargs(loaded),
                )
                winners.append(r.admission_id)
            except Exception as e:
                errors.append(repr(e))
            finally:
                db_local.close()

        threads = [threading.Thread(target=attempt) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        # All winners resolve to the SAME admission_id.
        unique_winners = set(winners)
        self.assertEqual(len(unique_winners), 1,
                         f"expected single admission, got {unique_winners}")
        # Exactly one durable race_admissions row.
        cur = self._db._conn.execute(
            "SELECT admission_id FROM race_admissions WHERE idem_key=?",
            ("idem-concurrent",),
        )
        rows = list(cur)
        self.assertEqual(len(rows), 1)


# ============================================================
# A04 — Validation receipt bound to exact candidate
# ============================================================

class TestA04ReceiptCandidateBinding(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))
        self._repo = _make_repo(self._tmp)
        self._head = git_head(self._repo)

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_cross_chunk_receipt_rejects(self):
        """A PASS receipt from chunk A MUST NOT authorize chunk B."""
        grant = _grant_active(self._db, plan_id="pl-ccr", grant_id="gr-ccr")
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit=self._head + "0" * 24, base_tree_digest="b" * 64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        # Mint a receipt claiming chunk_id=chk-A.
        rid = mint_validation_receipt(
            validator_id="noop", validator_command="noop",
            validator_profile="noop",
            candidate_snapshot_digest="a" * 64,
            candidate_tree_state="post-apply",
            chunk_id="chk-A",
            outcome="pass",
            detail="chunk A",
        )
        with self.assertRaises(Exception) as ctx:
            compare_and_swap_advance(
                self._db, repo_root=self._repo,
                campaign_id=camp.campaign_id, chunk_id="chk-B",
                new_commit="0" * 64, holder_fence_generation=1, actor="runner",
                idempotency_key="idem-ccr",
                validation_receipt_id=rid,
                expected_chunk_id="chk-B",
                expected_validator_id="noop",
                expected_tree_digest="a" * 64,
            )
        self.assertIn("chunk_id", str(ctx.exception).lower())

    def test_wrong_candidate_snapshot_rejects(self):
        grant = _grant_active(self._db, plan_id="pl-wcs", grant_id="gr-wcs")
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit=self._head + "0" * 24, base_tree_digest="b" * 64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        rid = mint_validation_receipt(
            validator_id="noop", validator_command="noop",
            candidate_snapshot_digest="aa" * 32,
            outcome="pass", chunk_id="chk-1",
        )
        with self.assertRaises(Exception) as ctx:
            compare_and_swap_advance(
                self._db, repo_root=self._repo,
                campaign_id=camp.campaign_id, chunk_id="chk-1",
                new_commit="0" * 64, holder_fence_generation=1, actor="runner",
                idempotency_key="idem-wcs",
                validation_receipt_id=rid,
                expected_tree_digest="bb" * 32,
                expected_chunk_id="chk-1",
                expected_validator_id="noop",
            )
        self.assertIn("candidate_snapshot_digest" in str(ctx.exception).lower() or
                       "candidate tree" in str(ctx.exception).lower(), [True])


# ============================================================
# A05 — Real EFFECT_UNKNOWN + failpoint recovery
# ============================================================

class TestA05CrashFailpoints(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))
        self._repo = _make_repo(self._tmp)
        self._head = git_head(self._repo)

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_crash_window_transitions_campaign_to_effect_unknown(self):
        """Recording a crash window flips campaign state to EFFECT_UNKNOWN."""
        grant = _grant_active(self._db, plan_id="pl-cw", grant_id="gr-cw")
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit=self._head + "0" * 24, base_tree_digest="b" * 64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        # Record a crash window via the integration boundary.
        _record_crash_window(
            self._db, campaign_id=camp.campaign_id, chunk_id="chk-cw",
            kind="integration_cas_mismatch",
            observed_artifact="x",
        )
        cur = self._db._conn.execute(
            "SELECT state FROM campaigns WHERE campaign_id=?",
            (camp.campaign_id,),
        )
        row = cur.fetchone()
        self.assertEqual(row["state"], "EFFECT_UNKNOWN")

    def test_continuation_refuses_effect_unknown(self):
        grant = _grant_active(self._db, plan_id="pl-cn", grant_id="gr-cn")
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit=self._head + "0" * 24, base_tree_digest="b" * 64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        # Set the campaign to EFFECT_UNKNOWN.
        self._db._conn.execute(
            "UPDATE campaigns SET state='EFFECT_UNKNOWN' WHERE campaign_id=?",
            (camp.campaign_id,),
        )
        with self.assertRaises(Exception) as ctx:
            check_campaign_continuation(
                self._db,
                campaign_id=camp.campaign_id,
                grant_id=grant.grant_id,
            )
        self.assertIn("effect_unknown", str(ctx.exception).lower())

    def test_reconcile_crash_window_actual_recovery(self):
        """reconcile_crash_window does real reconciliation, not just listing."""
        grant = _grant_active(self._db, plan_id="pl-rc", grant_id="gr-rc")
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit=self._head + "0" * 24, base_tree_digest="b" * 64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        # Record a crash window via the integration boundary (no
        # direct _record_crash_window call in tests below; this is
        # the bootstrap test that the helper works at all).
        _record_crash_window(
            self._db, campaign_id=camp.campaign_id, chunk_id="chk-rc",
            kind="integration_cas_mismatch",
            observed_artifact="y",
        )
        # The reconcile function must NOT just list rows; it must
        # actually transition campaign state.
        result = reconcile_crash_window(
            self._db, campaign_id=camp.campaign_id,
        )
        self.assertIn("decision", result)
        self.assertIn(result["decision"], ("RESOLVED_TO_ACTIVE", "EFFECT_UNKNOWN_PRESERVED"))
        cur = self._db._conn.execute(
            "SELECT state FROM campaigns WHERE campaign_id=?",
            (camp.campaign_id,),
        )
        row = cur.fetchone()
        self.assertIn(row["state"], ("ACTIVE", "NEEDS_DECISION"))

    def test_failpoint_injection_ref_advance_boundary(self):
        """Arming TR_FAILPOINT_ref_advanced_before_integration_journal_event
        raises during compare_and_swap_advance."""
        grant = _grant_active(self._db, plan_id="pl-fp", grant_id="gr-fp")
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit=self._head + "0" * 24, base_tree_digest="b" * 64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        # Create a real commit to advance.
        wt = ensure_campaign_worktree(
            repo_root=self._repo, campaign_id=camp.campaign_id,
            base_commit=self._head + "0" * 24,
        )
        (wt / "src" / "app.py").write_text("FP\n")
        subprocess.run(["git", "add", "-A"], cwd=str(wt), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "fp"], cwd=str(wt), check=True, capture_output=True)
        new_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(wt), capture_output=True, text=True
        ).stdout.strip()
        new_tree = subprocess.run(
            ["git", "rev-parse", f"{new_commit}^{{tree}}"], cwd=str(wt),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        rid = mint_validation_receipt(
            validator_id="noop", validator_command="noop",
            candidate_snapshot_digest=new_tree,
            outcome="pass", chunk_id="chk-fp",
        )
        os.environ["TR_FAILPOINT_ref_advanced_before_integration_journal_event"] = "raise"
        try:
            with self.assertRaises(RuntimeError):
                compare_and_swap_advance(
                    self._db, repo_root=self._repo,
                    campaign_id=camp.campaign_id, chunk_id="chk-fp",
                    new_commit=new_commit, holder_fence_generation=1, actor="runner",
                    idempotency_key="idem-fp",
                    validation_receipt_id=rid,
                    expected_tree_digest=new_tree,
                )
        finally:
            os.environ.pop("TR_FAILPOINT_ref_advanced_before_integration_journal_event", None)


# ============================================================
# A06 — Live child takeover fence
# ============================================================

class TestA06LiveChildFencing(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))
        self._old_state = os.environ.get("OVERNIGHT_STATE_DIR")

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)
        if self._old_state:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state

    def test_expired_live_child_blocks_direct_reacquire(self):
        """A controlled subprocess holds a live lease; direct reacquire
        from the same process tree fails until takeover bumps the
        fence. No broad kills.
        """
        grant = _grant_active(self._db, plan_id="pl-lc", grant_id="gr-lc")
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit="a" * 64, base_tree_digest="b" * 64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        # Spawn a controlled child process that holds the lease open.
        # We use a small Python script that does nothing but stay alive.
        import sys
        child_script = self._tmp / "child.py"
        child_script.write_text(
            "import os, sys, time\n"
            "from pathlib import Path\n"
            f"os.environ['OVERNIGHT_STATE_DIR'] = {str(self._tmp / 'state')!r}\n"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parent.parent / 'src')!r})\n"
            "from overnight_runner.db import Database\n"
            "from overnight_runner.resources import acquire_lease, current_fence\n"
            "db = Database(Path(sys.argv[1]))\n"
            "fence = current_fence(db, sys.argv[2])\n"
            "lease = acquire_lease(\n"
            "    db, campaign_id=sys.argv[2], resource_id=sys.argv[3],\n"
            "    owner_id=sys.argv[4], owner_boot_id='boot-child',\n"
            "    owner_pid=os.getpid(), fence_generation=fence.current_generation,\n"
            "    ttl_seconds=300,\n"
            ")\n"
            "print('LSE_ID:' + lease.lease_id, flush=True)\n"
            "sys.stdout.flush()\n"
            "time.sleep(120)\n"
        )
        db_path = str(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        env = os.environ.copy()
        env["OVERNIGHT_STATE_DIR"] = str(self._tmp / "state")
        env["PYTHONPATH"] = (
            str(Path(__file__).resolve().parent.parent / "src") +
            os.pathsep + env.get("PYTHONPATH", "")
        )
        proc = subprocess.Popen(
            [sys.executable, str(child_script), db_path, camp.campaign_id, "writer-lease", "wkr-child"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env,
        )
        try:
            # Wait for the child to print the lease_id.
            child_lease_id = ""
            for _ in range(50):
                line = proc.stdout.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                line = line.strip()
                if isinstance(line, bytes):
                    line = line.decode()
                # Lease IDs are prefixed with LSE_ID:
                if "LSE_ID:" in line:
                    child_lease_id = line.split("LSE_ID:", 1)[1].strip()
                    break
                if child_lease_id:
                    break
            if not child_lease_id.startswith("lse-"):
                stderr = proc.stderr.read()
                stdout = proc.stdout.read()
                self.fail(
                    f"expected lse-, got {child_lease_id!r}; "
                    f"child stdout={stdout!r}; child stderr={stderr!r}"
                )
            # Force-expire the lease (timestamp) WITHOUT touching the
            # child PID. The child remains alive.
            self._db._conn.execute(
                "UPDATE leases SET expires_at=? WHERE lease_id=?",
                (int(time.time()) - 10, child_lease_id),
            )
            # A direct second acquire by ANOTHER owner MUST fail.
            with self.assertRaises(Exception):
                acquire_lease(
                    self._db,
                    campaign_id=camp.campaign_id,
                    resource_id="writer-lease",
                    owner_id="wkr-A",
                    owner_boot_id="boot-A",
                    owner_pid=os.getpid(),
                    fence_generation=current_fence(self._db, camp.campaign_id).current_generation,
                    ttl_seconds=10,
                )
            # Explicit takeover bumps the fence.
            new_gen = revoke_for_takeover(
                self._db, campaign_id=camp.campaign_id, reason="live-child takeover"
            )
            old_gen = current_fence(self._db, camp.campaign_id).current_generation - 1
            # Stale fence write FAILS.
            from overnight_runner.resources import enforce_fence
            with self.assertRaises(Exception) as ctx:
                enforce_fence(self._db, campaign_id=camp.campaign_id,
                              holder_generation=old_gen, action="integration")
            self.assertIn("fence_stale", str(ctx.exception).lower())
            # New generation succeeds.
            new_lease = acquire_lease(
                self._db,
                campaign_id=camp.campaign_id,
                resource_id="writer-lease",
                owner_id="wkr-new",
                owner_boot_id="boot-new",
                owner_pid=os.getpid() + 2,
                fence_generation=new_gen,
                ttl_seconds=10,
            )
            self.assertEqual(new_lease.fence_generation, new_gen)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


# ============================================================
# A06 — Fenced campaign mutation boundary
# ============================================================

class TestA06FencedMutationBoundary(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))
        self._repo = _make_repo(self._tmp)
        self._head = git_head(self._repo)

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_stale_writer_cannot_apply_patch(self):
        """After takeover, the old owner cannot apply a campaign patch."""
        grant = _grant_active(self._db, plan_id="pl-fm", grant_id="gr-fm")
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit=self._head + "0" * 24, base_tree_digest="b" * 64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        wt = ensure_campaign_worktree(
            repo_root=self._repo, campaign_id=camp.campaign_id,
            base_commit=self._head + "0" * 24,
        )
        # Build a real broker on the worktree.
        reg = CommandRegistry()
        reg.register(CommandSpec("noop", ["true"], "repo", 5, "read"))
        broker = Broker(
            repo_root=wt, registry=reg,
            allowed_write_paths=["src/app.py"],
            allowed_create_paths=[],
            allowed_read_paths=["src/app.py"],
            allowed_protected_read_paths=[],
            model_allowed_command_ids=["noop"],
            required_validator_ids=["noop"],
            artifact_dir=self._tmp / "artifacts",
        )
        # Propose a patch so the broker has something to apply.
        # The file currently is BASE\n (from _make_repo). Set the
        # proposal's before_text to the actual current contents so
        # apply_proposal passes the hash check.
        from overnight_runner.broker import Proposal
        before = (wt / "src" / "app.py").read_text()
        broker.proposals["prop-x"] = Proposal(
            proposal_id="prop-x", op="replace_file",
            path="src/app.py", abs_path=wt / "src" / "app.py",
            before_text=before, proposed_text="FIRST\n",
            preview_diff="", changed_lines=1, proposed_bytes=6,
        )
        # Acquire a writer lease at the current fence.
        old_gen = current_fence(self._db, camp.campaign_id).current_generation
        acquire_lease(
            self._db, campaign_id=camp.campaign_id, resource_id="writer",
            owner_id="wkr-A", owner_boot_id="boot-A", owner_pid=os.getpid(),
            fence_generation=old_gen, ttl_seconds=300,
        )
        # Takeover.
        new_gen = revoke_for_takeover(self._db, campaign_id=camp.campaign_id,
                                       reason="test takeover")
        self.assertEqual(new_gen, old_gen + 1)
        # The OLD owner (with the stale fence) cannot apply.
        with self.assertRaises(Exception) as ctx:
            apply_campaign_patch(
                self._db, broker, str(self._repo), camp.campaign_id, "prop-x",
                admission_fence_generation=old_gen,
            )
        self.assertIn("fence_stale", str(ctx.exception).lower())
        # The NEW owner (with the bumped fence) can apply.
        apply_campaign_patch(
            self._db, broker, str(self._repo), camp.campaign_id, "prop-x",
            admission_fence_generation=new_gen,
        )


# ============================================================
# A07 — Complete authoritative budget API
# ============================================================

class TestA07TrustedBudgetAPI(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_full_dimension_update(self):
        """update_budget_after_chunk covers all 9 dimensions atomically."""
        grant = _grant_active(self._db, plan_id="pl-fd", grant_id="gr-fd")
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit="a" * 64, base_tree_digest="b" * 64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        # First admission creates the ledger.
        derive_admission(
            self._db, grant=grant, chunk=_chunk(camp.campaign_id),
            worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            current_accepted_snapshot=_snap(),
            **_id_kwargs(grant),
        )
        totals = update_budget_after_chunk(
            self._db, ledger_id=f"bl-{camp.campaign_id}",
            delta_model_calls=2, delta_tool_calls=3, delta_repairs=1,
            delta_rechunks=1, delta_escalations=0,
            delta_active_seconds=10, delta_wall_seconds=15,
            delta_cost_microusd=100, delta_chunks=1, delta_context_tokens=500,
        )
        self.assertEqual(totals["cumulative_model_calls"], 2)
        self.assertEqual(totals["cumulative_tool_calls"], 3)
        self.assertEqual(totals["cumulative_repairs"], 1)
        self.assertEqual(totals["cumulative_rechunks"], 1)
        self.assertEqual(totals["cumulative_active_seconds"], 10)
        self.assertEqual(totals["cumulative_wall_seconds"], 15)
        self.assertEqual(totals["cumulative_cost_microusd"], 100)
        self.assertEqual(totals["cumulative_chunks"], 1)
        self.assertEqual(totals["cumulative_context_tokens"], 500)

    def test_each_dimension_exhaustion(self):
        """Each individual bound rejects first increment beyond max."""
        grant = _grant_active(self._db, plan_id="pl-ex", grant_id="gr-ex")
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit="a" * 64, base_tree_digest="b" * 64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        derive_admission(
            self._db, grant=grant, chunk=_chunk(camp.campaign_id),
            worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            current_accepted_snapshot=_snap(),
            **_id_kwargs(grant),
        )
        ledger_id = f"bl-{camp.campaign_id}"
        # Set each cumulative to its bound, then assert the next
        # increment raises.
        bounds = {
            "max_model_calls": 10, "max_tool_calls": 20,
            "max_local_repairs": 2, "max_rechunks": 1,
            "max_active_seconds": 3600, "max_wall_seconds": 28800,
            "max_cost_microusd": 1000, "max_chunks": 3,
            "context_token_budget": 8192,
        }
        # Use the trusted API to drive each counter to its bound, then
        # one more should fail.
        with self._db.transaction() as cur:
            cur.execute(
                "UPDATE budget_ledgers SET "
                "cumulative_model_calls=?, cumulative_tool_calls=?, "
                "cumulative_repairs=?, cumulative_rechunks=?, "
                "cumulative_active_seconds=?, cumulative_wall_seconds=?, "
                "cumulative_cost_microusd=?, cumulative_chunks=?, "
                "cumulative_context_tokens=? WHERE ledger_id=?",
                (
                    bounds["max_model_calls"], bounds["max_tool_calls"],
                    bounds["max_local_repairs"], bounds["max_rechunks"],
                    bounds["max_active_seconds"], bounds["max_wall_seconds"],
                    bounds["max_cost_microusd"], bounds["max_chunks"],
                    bounds["context_token_budget"], ledger_id,
                ),
            )
        # Each next increment must fail.
        with self.assertRaises(Exception):
            update_budget_after_chunk(self._db, ledger_id=ledger_id, delta_model_calls=1)
        with self.assertRaises(Exception):
            update_budget_after_chunk(self._db, ledger_id=ledger_id, delta_tool_calls=1)
        with self.assertRaises(Exception):
            update_budget_after_chunk(self._db, ledger_id=ledger_id, delta_repairs=1)
        with self.assertRaises(Exception):
            update_budget_after_chunk(self._db, ledger_id=ledger_id, delta_rechunks=1)
        with self.assertRaises(Exception):
            update_budget_after_chunk(self._db, ledger_id=ledger_id, delta_active_seconds=1)
        with self.assertRaises(Exception):
            update_budget_after_chunk(self._db, ledger_id=ledger_id, delta_cost_microusd=1)
        with self.assertRaises(Exception):
            update_budget_after_chunk(self._db, ledger_id=ledger_id, delta_chunks=1)
        with self.assertRaises(Exception):
            update_budget_after_chunk(self._db, ledger_id=ledger_id, delta_context_tokens=1)

    def test_cumulative_persists_across_restart(self):
        """Process restart on the same DB preserves cumulative totals."""
        grant = _grant_active(self._db, plan_id="pl-pr", grant_id="gr-pr")
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit="a" * 64, base_tree_digest="b" * 64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        derive_admission(
            self._db, grant=grant, chunk=_chunk(camp.campaign_id),
            worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            current_accepted_snapshot=_snap(),
            **_id_kwargs(grant),
        )
        update_budget_after_chunk(
            self._db, ledger_id=f"bl-{camp.campaign_id}",
            delta_model_calls=5, delta_wall_seconds=60,
        )
        # Reopen DB.
        self._db.close()
        self._db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        totals = read_budget_totals(
            self._db, ledger_id=f"bl-{camp.campaign_id}",
        )
        self.assertEqual(totals["cumulative_model_calls"], 5)
        self.assertEqual(totals["cumulative_wall_seconds"], 60)


# ============================================================
# A11 — One-shot protected approval consumption
# ============================================================

class TestA11OneShotProtectedApproval(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_consume_twice_rejects(self):
        """Re-consuming an already-consumed approval raises SafetyError."""
        from overnight_runner.protected_approvals import (
            register_protected_approval, consume_protected_approval,
        )
        register_protected_approval(
            self._db,
            approval_id="appr-once",
            operation="activate_grant",
            grant_digest_target="0" * 64,
            operator_id="op-1",
            operator_receipt={"approval_id": "appr-once"},
        )
        # First consume succeeds.
        ap = consume_protected_approval(self._db, "appr-once")
        self.assertTrue(ap.is_consumed)
        # Second consume raises.
        with self.assertRaises(Exception) as ctx:
            consume_protected_approval(self._db, "appr-once")
        self.assertIn("already consumed", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
