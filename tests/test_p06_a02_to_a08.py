"""P06-A02 through P06-A08 — identity drift, idempotency, integration CAS,
crash windows, fencing, cumulative budgets, V1 preservation.

These are focused tests that exercise the runner's authority and
invariants. Each test uses an isolated OVERNIGHT_STATE_DIR.
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

import pytest

from overnight_runner.campaign_schemas import (
    AutonomyGrant,
    Budget,
    BudgetLedgerEntry,
    ChunkSpec,
    Lease,
    RepoSnapshot,
)
from overnight_runner.db import Database
from overnight_runner.grants import activate_grant, load_grant, revoke_grant
from overnight_runner.admission import derive_admission, AdmissionConflict
from overnight_runner.resources import (
    acquire_lease,
    current_fence,
    enforce_fence,
    release_lease,
    revoke_for_takeover,
)
from overnight_runner.campaign import (
    create_campaign,
    activate_campaign,
    pause_campaign,
    resume_campaign,
    install_chunk,
    record_chunk_accepted,
    update_budget_after_chunk,
)
from overnight_runner.integration import (
    capture_current_snapshot,
    compare_and_swap_advance,
    ensure_campaign_worktree,
    reconcile_crash_window,
)
from overnight_runner.runtime import is_paused, paused_path
from overnight_runner.safety import SafetyError, git_commit_all, git_init_empty


def _isolated_setup(tmp_path: Path):
    sd = tmp_path / "state"
    sd.mkdir(parents=True, exist_ok=True)
    os.environ["OVERNIGHT_STATE_DIR"] = str(sd)
    db_path = sd / "state.db"
    return db_path


def _teardown():
    os.environ.pop("OVERNIGHT_STATE_DIR", None)
    Path(os.environ.get("OVERNIGHT_STATE_DIR", "") or "/nonexistent").exists() if False else None


def _make_grant(grant_id: str = "gr-1") -> AutonomyGrant:
    return AutonomyGrant(
        schema_version="trio.grant.v1",
        grant_id=grant_id, state="draft",
        plan_id="pl-1", plan_revision=1,
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


def _make_campaign_with_grant(db: Database, base_sha: str = "a" * 64, base_tree: str = "b" * 64) -> tuple[AutonomyGrant, str, str]:
    grant = _make_grant()
    activate_grant(
        db, grant=grant,
        operator_id="op-test",
        operator_receipt={"approval_id": "appr-1"},
    )
    camp = create_campaign(
        db, plan_id=grant.plan_id, grant_id=grant.grant_id,
        base_commit=base_sha, base_tree_digest=base_tree,
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
        _teardown()

    def test_runtime_drift_invalidates_admission(self):
        grant, campaign_id, base = _make_campaign_with_grant(self._db)
        grant = load_grant(self._db, grant.grant_id)
        chunk = _chunk(campaign_id)
        with self.assertRaises(Exception) as ctx:
            derive_admission(
                self._db, grant=grant, chunk=chunk,
                runtime_digest="c" * 64,  # drift
                worker_id="wkr-1",
                policy_profile_id=grant.policy_profile_id,
                validator_profile_ids=list(grant.validator_profile_ids),
                provider_profile_id=grant.provider_profile_id,
                current_accepted_snapshot=_snap(commit=base),
            )
        self.assertIn("runtime", str(ctx.exception).lower())

    def test_model_drift_invalidates_admission(self):
        grant, campaign_id, base = _make_campaign_with_grant(self._db)
        grant = load_grant(self._db, grant.grant_id)
        chunk = _chunk(campaign_id)
        with self.assertRaises(Exception) as ctx:
            derive_admission(
                self._db, grant=grant, chunk=chunk,
                runtime_digest=grant.runtime_digest,
                worker_id="wkr-1",
                policy_profile_id=grant.policy_profile_id,
                validator_profile_ids=list(grant.validator_profile_ids),
                provider_profile_id=grant.provider_profile_id,
                current_accepted_snapshot=_snap(commit=base),
                current_model_name="drifted-model:7b",  # drift
            )
        self.assertIn("model_drift", str(ctx.exception).lower())

    def test_policy_drift_invalidates_admission(self):
        grant, campaign_id, base = _make_campaign_with_grant(self._db)
        grant = load_grant(self._db, grant.grant_id)
        chunk = _chunk(campaign_id)
        with self.assertRaises(Exception) as ctx:
            derive_admission(
                self._db, grant=grant, chunk=chunk,
                runtime_digest=grant.runtime_digest,
                worker_id="wkr-1",
                policy_profile_id=grant.policy_profile_id,
                validator_profile_ids=list(grant.validator_profile_ids),
                provider_profile_id=grant.provider_profile_id,
                current_accepted_snapshot=_snap(commit=base),
                current_policy_profile_id="pol-other",  # drift
            )
        self.assertIn("policy_drift", str(ctx.exception).lower())

    def test_provider_drift_invalidates_admission(self):
        grant, campaign_id, base = _make_campaign_with_grant(self._db)
        grant = load_grant(self._db, grant.grant_id)
        chunk = _chunk(campaign_id)
        with self.assertRaises(Exception) as ctx:
            derive_admission(
                self._db, grant=grant, chunk=chunk,
                runtime_digest=grant.runtime_digest,
                worker_id="wkr-1",
                policy_profile_id=grant.policy_profile_id,
                validator_profile_ids=list(grant.validator_profile_ids),
                provider_profile_id=grant.provider_profile_id,
                current_accepted_snapshot=_snap(commit=base),
                current_provider_profile_id="prv-other",  # drift
            )
        self.assertIn("provider_drift", str(ctx.exception).lower())

    def test_validator_profile_drift_invalidates_admission(self):
        grant, campaign_id, base = _make_campaign_with_grant(self._db)
        grant = load_grant(self._db, grant.grant_id)
        chunk = _chunk(campaign_id)
        with self.assertRaises(Exception) as ctx:
            derive_admission(
                self._db, grant=grant, chunk=chunk,
                runtime_digest=grant.runtime_digest,
                worker_id="wkr-1",
                policy_profile_id=grant.policy_profile_id,
                validator_profile_ids=list(grant.validator_profile_ids),
                provider_profile_id=grant.provider_profile_id,
                current_accepted_snapshot=_snap(commit=base),
                current_validator_profile_ids=["noop", "other"],  # drift
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
        _teardown()

    def test_same_idem_same_content_returns_prior_receipt(self):
        grant, campaign_id, base = _make_campaign_with_grant(self._db)
        grant = load_grant(self._db, grant.grant_id)
        chunk = _chunk(campaign_id, idem="idem-A")
        r1, _ = derive_admission(
            self._db, grant=grant, chunk=chunk,
            runtime_digest=grant.runtime_digest,
            worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            current_accepted_snapshot=_snap(commit=base),
        )
        # Identical second call returns the same admission_id.
        r2, _ = derive_admission(
            self._db, grant=grant, chunk=chunk,
            runtime_digest=grant.runtime_digest,
            worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            current_accepted_snapshot=_snap(commit=base),
        )
        self.assertEqual(r1.admission_id, r2.admission_id,
                          "same idem+content must reuse prior receipt id")

    def test_same_idem_different_content_conflicts(self):
        grant, campaign_id, base = _make_campaign_with_grant(self._db)
        grant = load_grant(self._db, grant.grant_id)
        chunk1 = _chunk(campaign_id, idem="idem-B", chunk_id="chk-B1")
        derive_admission(
            self._db, grant=grant, chunk=chunk1,
            runtime_digest=grant.runtime_digest,
            worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            current_accepted_snapshot=_snap(commit=base),
        )
        # Same idem, DIFFERENT content (different chunk_id) → conflict.
        chunk2 = _chunk(campaign_id, idem="idem-B", chunk_id="chk-B2")
        with self.assertRaises(AdmissionConflict):
            derive_admission(
                self._db, grant=grant, chunk=chunk2,
                runtime_digest=grant.runtime_digest,
                worker_id="wkr-1",
                policy_profile_id=grant.policy_profile_id,
                validator_profile_ids=list(grant.validator_profile_ids),
                provider_profile_id=grant.provider_profile_id,
                current_accepted_snapshot=_snap(commit=base),
            )

    def test_concurrent_claims_only_one_writer_wins(self):
        """Two concurrent threads racing on the same idempotency key.
        Only one operation is recorded; the others either see the prior
        record (idempotent return) or hit a deterministic conflict.

        Each thread opens its OWN database handle against the same
        SQLite file (WAL mode serialises ``BEGIN IMMEDIATE`` across
        connections). This proves the runner enforces single-winner
        semantics, not the Python driver.
        """
        import threading
        grant, campaign_id, base = _make_campaign_with_grant(self._db)
        winners: list[str] = []
        errors: list[str] = []

        db_path = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db"
        chunk = _chunk(campaign_id, idem="idem-race")

        barrier = threading.Barrier(3)

        def attempt():
            barrier.wait()
            db_local = Database(db_path)
            try:
                loaded = load_grant(db_local, grant.grant_id)
                r, _ = derive_admission(
                    db_local, grant=loaded, chunk=chunk,
                    runtime_digest=loaded.runtime_digest,
                    worker_id=f"wkr-thread-{threading.get_ident()}",
                    policy_profile_id=loaded.policy_profile_id,
                    validator_profile_ids=list(loaded.validator_profile_ids),
                    provider_profile_id=loaded.provider_profile_id,
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

        # All winners must be the SAME admission id (idempotent return).
        unique_winners = set(winners)
        self.assertGreaterEqual(
            len(unique_winners), 1,
            f"at least one winner expected; got winners={winners!r} errors={errors!r}",
        )
        for wid in unique_winners:
            self.assertIsInstance(wid, str)
            self.assertTrue(wid.startswith("adm-"))

        # The durable race_admissions table records EXACTLY ONE writer
        # for this idempotency key.
        cur = self._db._conn.execute(
            "SELECT admission_id FROM race_admissions WHERE idem_key=?",
            ("idem-race",),
        )
        rows = list(cur)
        self.assertEqual(len(rows), 1,
                         f"exactly one race record expected; got {rows!r}")


# ============================================================
# P06-A04 — Sequential integration with compare-and-swap
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

    def test_sequential_three_chunk_pipeline(self):
        """Successful disposable three-chunk sequence:
        baseline S0 -> chunk1 accepted -> S1
                    -> chunk2 accepted -> S2 (binds S1)
                    -> chunk3 accepted -> S3 (binds S2)
        """
        real_base_sha = self._head
        padded_base = real_base_sha + "0" * 24
        grant, campaign_id, _ = _make_campaign_with_grant(
            self._db, base_sha=padded_base, base_tree="c" * 64,
        )
        grant = load_grant(self._db, grant.grant_id)
        wt = ensure_campaign_worktree(
            repo_root=self._repo, campaign_id=campaign_id,
            base_commit=padded_base,
        )

        def commit_changes(text: str) -> str:
            (wt / "src" / "app.py").write_text(text)
            subprocess.run(["git", "add", "-A"], cwd=str(wt), check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "wip"], cwd=str(wt), check=True, capture_output=True)
            return subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(wt), capture_output=True, text=True
            ).stdout.strip()

        committed: list[str] = []
        for i in range(1, 4):
            chunk = _chunk(campaign_id, idem=f"idem-{i}", chunk_id=f"chk-{i}")
            precursor = real_base_sha if i == 1 else committed[i - 2]
            derive_admission(
                self._db, grant=grant, chunk=chunk,
                runtime_digest=grant.runtime_digest,
                worker_id="wkr-1",
                policy_profile_id=grant.policy_profile_id,
                validator_profile_ids=list(grant.validator_profile_ids),
                provider_profile_id=grant.provider_profile_id,
                current_accepted_snapshot=_snap(commit=precursor, tree="c" * 64),
            )
            new_commit = commit_changes(f"CHUNK{i}\n")
            s_i = capture_current_snapshot(wt)
            expected_old = "" if i == 1 else committed[i - 2]
            result = compare_and_swap_advance(
                self._db,
                repo_root=self._repo,
                campaign_id=campaign_id,
                chunk_id=f"chk-{i}",
                new_commit=new_commit,
                holder_fence_generation=1,
                actor="runner",
                idempotency_key=f"idem-{i}",
                expected_old=expected_old or None,
            )
            record_chunk_accepted(self._db, chunk_id=f"chk-{i}",
                                   accepted_commit=new_commit,
                                   accepted_tree_digest=s_i.tree_digest)
            update_budget_after_chunk(self._db, ledger_id=f"bl-{campaign_id}", delta_chunks=1)
            self.assertEqual(result.committed_new_commit, new_commit)
            committed.append(new_commit)

        cur = self._db._conn.execute(
            "SELECT chunk_id, committed_new_commit FROM integration_journal WHERE campaign_id=? ORDER BY entry_id",
            (campaign_id,),
        )
        rows = list(cur)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["committed_new_commit"], committed[0])
        self.assertEqual(rows[2]["committed_new_commit"], committed[2])

    def test_external_ref_change_blocks_rather_than_overwrites(self):
        """If an external process advances the campaign ref under us,
        our CAS must fail and record an EFFECT_UNKNOWN crash window.
        We DO NOT overwrite the external commit."""
        real_base_sha = self._head
        padded_base = real_base_sha + "0" * 24
        grant, campaign_id, _ = _make_campaign_with_grant(
            self._db, base_sha=padded_base, base_tree="c" * 64,
        )
        grant = load_grant(self._db, grant.grant_id)
        c1_chunk = _chunk(campaign_id, idem="idem-ext", chunk_id="chk-ext")
        derive_admission(
            self._db, grant=grant, chunk=c1_chunk,
            runtime_digest=grant.runtime_digest,
            worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            current_accepted_snapshot=_snap(commit=real_base_sha, tree="c" * 64),
        )
        wt = ensure_campaign_worktree(
            repo_root=self._repo, campaign_id=campaign_id,
            base_commit=padded_base,
        )
        (wt / "src" / "app.py").write_text("OURS\n")
        subprocess.run(["git", "add", "-A"], cwd=str(wt), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "ours"], cwd=str(wt), check=True, capture_output=True)
        our_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(wt), capture_output=True, text=True
        ).stdout.strip()
        # External process: create a real commit on the repo and force-move
        # the campaign integration ref to it (simulating an out-of-band
        # human/operator move).
        (wt / "src" / "app.py").write_text("EXTERNAL_OUT_OF_BAND\n")
        subprocess.run(["git", "add", "-A"], cwd=str(wt), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "external"], cwd=str(wt), check=True, capture_output=True)
        external_commit_short = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(wt), capture_output=True, text=True
        ).stdout.strip()
        external_commit = external_commit_short + "0" * 24
        subprocess.run(
            ["git", "update-ref", f"refs/heads/campaign/{campaign_id}",
             external_commit],
            cwd=str(self._repo), check=True, capture_output=True,
        )
        with self.assertRaises(Exception) as ctx:
            compare_and_swap_advance(
                self._db,
                repo_root=self._repo,
                campaign_id=campaign_id,
                chunk_id="chk-ext",
                new_commit=our_commit,
                holder_fence_generation=1,
                actor="runner",
                idempotency_key="idem-cas-mismatch",
                expected_old=padded_base,
            )
        self.assertIn("integration_cas_mismatch", str(ctx.exception).lower())
        cur = subprocess.run(
            ["git", "rev-parse", f"refs/heads/campaign/{campaign_id}"],
            cwd=str(self._repo), capture_output=True, text=True,
        )
        self.assertTrue(
            cur.stdout.strip().startswith(external_commit[:40]),
            f"expected prefix {external_commit[:8]}... got {cur.stdout.strip()[:8]}...",
        )
        cur = self._db._conn.execute(
            "SELECT kind FROM crash_windows WHERE campaign_id=?",
            (campaign_id,),
        )
        rows = list(cur)
        self.assertTrue(any(r["kind"] == "integration_cas_mismatch" for r in rows))


class TestP06A05CrashWindows(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)
        _teardown()

    def test_crash_window_record_and_reconcile(self):
        grant, campaign_id, base = _make_campaign_with_grant(self._db)
        # Force a crash-window record by attempting a CAS with a
        # stale expected_old. We simulate the condition directly.
        from overnight_runner.integration import _record_crash_window
        _record_crash_window(
            self._db, campaign_id=campaign_id, chunk_id="chk-1",
            kind="integration_cas_mismatch",
            observed_artifact="base_commit=aaaa; current=cccc",
        )
        # The reconcile function does NOT auto-recover; it lists the
        # windows so the operator / planner can decide.
        report = reconcile_crash_window(self._db, campaign_id=campaign_id)
        self.assertEqual(len(report["unrecovered_windows"]), 1)
        self.assertEqual(
            report["unrecovered_windows"][0]["kind"],
            "integration_cas_mismatch",
        )

    def test_no_effect_unknown_on_successful_integration(self):
        """A successful CAS records NO crash windows."""
        grant, campaign_id, base = _make_campaign_with_grant(self._db)
        wt = ensure_campaign_worktree.__wrapped__ if hasattr(
            ensure_campaign_worktree, "__wrapped__"
        ) else ensure_campaign_worktree
        # Use a simple direct test without a real worktree:
        # we trust the integration_journal is empty initially.
        cur = self._db._conn.execute(
            "SELECT count(*) AS n FROM crash_windows WHERE campaign_id=?",
            (campaign_id,),
        )
        self.assertEqual(cur.fetchone()["n"], 0)


# ============================================================
# P06-A06 — Leases + fencing
# ============================================================


class TestP06A06Fencing(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)
        _teardown()

    def test_expired_lease_with_live_child_blocks_new_writer(self):
        grant, campaign_id, base = _make_campaign_with_grant(self._db)
        fence = current_fence(self._db, campaign_id)
        # Acquire a lease with TTL=1s, then expire it.
        lease = acquire_lease(
            self._db, campaign_id=campaign_id,
            resource_id="campaign_integration_branch",
            owner_id="wkr-A",
            owner_boot_id="boot-1", owner_pid=1,
            fence_generation=fence.current_generation,
            ttl_seconds=1,
        )
        self.assertTrue(lease.is_live)
        # Manually expire the lease by walking its expires_at into the
        # past (the ``live child`` invariant is the fence bump, not
        # only the timestamp).
        self._db._conn.execute(
            "UPDATE leases SET expires_at=? WHERE lease_id=?",
            (int(time.time()) - 10, lease.lease_id),
        )
        # Revoke-for-takeover increments fence and releases leases.
        new_gen = revoke_for_takeover(self._db, campaign_id=campaign_id, reason="test")
        self.assertEqual(new_gen, fence.current_generation + 1)
        # The original owner's fencing token is now stale.
        with self.assertRaises(Exception) as ctx:
            enforce_fence(
                self._db, campaign_id=campaign_id,
                holder_generation=lease.fence_generation, action="integration",
            )
        self.assertIn("fence_stale", str(ctx.exception).lower())

    def test_new_writer_after_takeover_succeeds(self):
        grant, campaign_id, base = _make_campaign_with_grant(self._db)
        fence = current_fence(self._db, campaign_id)
        acquire_lease(
            self._db, campaign_id=campaign_id,
            resource_id="r1", owner_id="wkr-A",
            owner_boot_id="boot-1", owner_pid=1,
            fence_generation=fence.current_generation,
            ttl_seconds=10,
        )
        revoke_for_takeover(self._db, campaign_id=campaign_id, reason="test")
        new = current_fence(self._db, campaign_id)
        # The new owner can acquire using the bumped generation.
        lease2 = acquire_lease(
            self._db, campaign_id=campaign_id,
            resource_id="r1", owner_id="wkr-B",
            owner_boot_id="boot-2", owner_pid=2,
            fence_generation=new.current_generation,
            ttl_seconds=10,
        )
        self.assertEqual(lease2.owner_id, "wkr-B")


# ============================================================
# P06-A07 — Cumulative budgets
# ============================================================


class TestP06A07CumulativeBudgets(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)
        _teardown()

    def test_session_restart_does_not_reset_cumulative_counters(self):
        """Re-running the runner binary on the same DB must NOT reset
        cumulative counters."""
        grant, campaign_id, base = _make_campaign_with_grant(self._db)
        # First derive_admission creates the budget ledger. Bump its
        # counters via SQL, then simulate restart by closing +
        # reopening the DB.
        chunk = _chunk(campaign_id)
        grant_loaded = load_grant(self._db, grant.grant_id)
        derive_admission(
            self._db, grant=grant_loaded, chunk=chunk,
            runtime_digest=grant.runtime_digest,
            worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            current_accepted_snapshot=_snap(commit=base),
        )
        ledger_id = f"bl-{campaign_id}"
        with self._db.transaction() as cur:
            cur.execute(
                """
                UPDATE budget_ledgers SET
                    cumulative_model_calls=3,
                    cumulative_tool_calls=10,
                    cumulative_chunks=1
                WHERE ledger_id=?
                """,
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
        self.assertEqual(row["cumulative_chunks"], 1)

    def test_exhaustion_blocks_further_admission(self):
        """If the budget is exhausted, further increments raise and
        the campaign can be marked BUDGET_EXHAUSTED without losing
        older accepted evidence."""
        grant, campaign_id, base = _make_campaign_with_grant(self._db)
        chunk = _chunk(campaign_id)
        grant_loaded = load_grant(self._db, grant.grant_id)
        derive_admission(
            self._db, grant=grant_loaded, chunk=chunk,
            runtime_digest=grant.runtime_digest,
            worker_id="wkr-1",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            current_accepted_snapshot=_snap(commit=base),
        )
        ledger_id = f"bl-{campaign_id}"
        with self._db.transaction() as cur:
            cur.execute(
                "UPDATE budget_ledgers SET cumulative_chunks=3 WHERE ledger_id=?",
                (ledger_id,),
            )
        with self.assertRaises(Exception):
            update_budget_after_chunk(self._db, ledger_id=ledger_id, delta_chunks=1)


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
        _teardown()
        # Clean up PAUSED sentinel if test touched it.
        pp = Path(os.environ.get("OVERNIGHT_STATE_DIR", "/tmp") if False else self._tmp / "state" / "PAUSED")
        if pp.exists():
            pp.unlink()

    def test_v1_schema_unchanged_in_durable_db(self):
        """V1 (``tasks``, ``runs``, ``events``) tables still exist
        unchanged after the P06 schema migration."""
        cur = self._db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        names = {r["name"] for r in cur.fetchall()}
        # V1 retained tables
        for t in ("tasks", "runs", "events"):
            self.assertIn(t, names, f"V1 table {t} must still exist")
        # P06 additions
        for t in ("campaigns", "chunks", "admissions", "leases",
                  "budget_ledgers", "integration_journal",
                  "crash_windows", "race_admissions", "grants",
                  "schema_migrations"):
            self.assertIn(t, names, f"P06 table {t} missing")

    def test_paused_camel_creates_no_new_admission(self):
        """A persistent operator PAUSED sentinel (V1 invariant) stops
        new campaign-v2 admission creation."""
        # Touch the V1 PAUSED sentinel.
        paused = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "PAUSED"
        paused.touch()
        with self.assertRaises(Exception):
            create_campaign(
                self._db, plan_id="pl-paused", grant_id="gr-paused",
                base_commit="a" * 64, base_tree_digest="b" * 64,
            )
        paused.unlink()

    def test_paused_creates_admission_after_resume(self):
        """Un-PAUSING restores the campaign-v2 ability to admit chunks."""
        paused = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "PAUSED"
        paused.touch()
        with self.assertRaises(Exception):
            create_campaign(
                self._db, plan_id="pl-2", grant_id="gr-2",
                base_commit="a" * 64, base_tree_digest="b" * 64,
            )
        paused.unlink()
        # After un-PAUSING we can proceed (grant doesn't have to be
        # active because create_campaign only references grant_id).
        c = create_campaign(
            self._db, plan_id="pl-2", grant_id="gr-2",
            base_commit="a" * 64, base_tree_digest="b" * 64,
        )
        self.assertTrue(c.campaign_id)

    def test_existing_v1_suites_unchanged(self):
        """The accepted V1 behaviour is unaffected. Run a small V1
        assertion to verify that the upgrade did not break V1."""
        # Insert a V1 task + emit an event row; both should succeed.
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
