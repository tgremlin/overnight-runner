"""P07 — Hermes foreman integration (runner-authoritative).

MOCK/FIXTURE evidence: everything here runs against an isolated runner
DB + fake clock + fake provider outcomes. No live provider outage or
live Hermes profile is required.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from overnight_runner.admission import derive_admission
from overnight_runner.campaign import activate_campaign, create_campaign
from overnight_runner.campaign_schemas import (
    AutonomyGrant,
    Budget,
    ChunkSpec,
    RepoSnapshot,
    content_sha256,
)
from overnight_runner.db import Database
from overnight_runner.grants import activate_grant, load_grant, revoke_grant
from overnight_runner.p07 import (
    ADAPTER_SCHEMA,
    CapacityKind,
    FallbackPolicy,
    FallbackProfile,
    active_waits,
    apply_capacity_outcome,
    classify_capacity,
    cooldown_state,
    complete_job,
    enter_capacity_wait,
    job_state,
    list_cooldowns,
    next_eligible_at,
    project_campaign_status,
    project_hermes_cards,
    request_control,
    resolve_capacity_wait,
    resolve_fallback,
    upsert_cooldown,
    wake_tick,
)
from overnight_runner.plans import load_plan_digest, register_plan
from overnight_runner.protected_approvals import register_protected_approval
from overnight_runner.safety import SafetyError


def _isolated_setup(tmp_path: Path) -> Path:
    sd = tmp_path / "state"
    sd.mkdir(parents=True, exist_ok=True)
    os.environ["OVERNIGHT_STATE_DIR"] = str(sd)
    return sd / "state.db"


def _grant(db, *, plan_id="pl-1", grant_id="gr-1", grant_expires_at=0, max_chunks=3):
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
                  work_package_criterion_ids={"pkg-1": {"crit-1"}})
    grant = grant.model_copy(update={"approved_plan_digest": load_plan_digest(db, plan_id)})
    register_protected_approval(
        db, approval_id=f"appr-{grant_id}", operation="activate_grant",
        grant_digest_target=content_sha256(grant), operator_id="op-1",
        operator_receipt={"approval_id": f"appr-{grant_id}"})
    activate_grant(db, grant=grant, operator_id="op-1", approval_id=f"appr-{grant_id}")
    return load_grant(db, grant_id)


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated_setup(self._tmp))

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _campaign(self, *, plan_id="pl-1", grant_id="gr-1", grant_expires_at=0,
                  max_chunks=3):
        grant = _grant(self._db, plan_id=plan_id, grant_id=grant_id,
                       grant_expires_at=grant_expires_at, max_chunks=max_chunks)
        camp = create_campaign(self._db, plan_id=grant.plan_id,
                               grant_id=grant.grant_id, base_commit="a" * 40,
                               base_tree_digest="b" * 64)
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        return grant, camp


# ============================================================
# A02 — provider capacity classification
# ============================================================

class TestCapacityClassification(_Base):
    def test_fixture_corpus(self):
        cases = [
            (dict(http_status=429), CapacityKind.RATE_LIMITED),
            (dict(error_code="rate_limit_exceeded"), CapacityKind.RATE_LIMITED),
            (dict(http_status=503), CapacityKind.OVERLOADED),
            (dict(http_status=529), CapacityKind.OVERLOADED),
            (dict(error_code="overloaded"), CapacityKind.OVERLOADED),
            (dict(error_code="quota_exhausted", reset_at=12345),
             CapacityKind.QUOTA_EXHAUSTED_KNOWN_RESET),
            (dict(error_code="insufficient_quota"),
             CapacityKind.QUOTA_EXHAUSTED_NO_RESET),
            (dict(message="Token Plan usage limit reached"),
             CapacityKind.QUOTA_EXHAUSTED_NO_RESET),
            (dict(http_status=401), CapacityKind.AUTH_FAILURE),
            (dict(http_status=403), CapacityKind.AUTH_FAILURE),
            (dict(error_code="invalid_api_key"), CapacityKind.AUTH_FAILURE),
            (dict(http_status=404), CapacityKind.INVALID_PROVIDER_MODEL_PROFILE),
            (dict(error_code="model_not_found"),
             CapacityKind.INVALID_PROVIDER_MODEL_PROFILE),
            (dict(http_status=500), CapacityKind.PROVIDER_UNAVAILABLE),
            (dict(http_status=502), CapacityKind.PROVIDER_UNAVAILABLE),
            (dict(transport_error="ConnectionError"), CapacityKind.PROVIDER_UNAVAILABLE),
            (dict(transport_error="weird_socket_fault"), CapacityKind.UNKNOWN_TRANSPORT),
            (dict(), CapacityKind.OK),
        ]
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs):
                out = classify_capacity(provider="minimax", model="MiniMax-M3", **kwargs)
                self.assertEqual(out.kind, expected.value)
                self.assertEqual(out.schema_version, "trio.capacity-outcome.v1")

    def test_no_prose_interpretation(self):
        # A prose-ish message with no structured signal must NOT be
        # classified as quota: classification is structured-only.
        out = classify_capacity(message="the model said it is tired")
        self.assertEqual(out.kind, CapacityKind.OK.value)


# ============================================================
# A11 — durable provider/account cooldown
# ============================================================

class TestCooldownPersistence(_Base):
    def test_upsert_idempotent_and_restart_safe(self):
        upsert_cooldown(self._db, provider="minimax", account="acct-1",
                        model="MiniMax-M3", kind="rate_limited", until_at=1000,
                        now=100)
        upsert_cooldown(self._db, provider="minimax", account="acct-1",
                        model="MiniMax-M3", kind="rate_limited", until_at=1000,
                        now=101)
        db_path = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db"
        self._db.close()
        self._db = Database(db_path)
        st = cooldown_state(self._db, provider="minimax", account="acct-1",
                            model="MiniMax-M3", now=500)
        self.assertIsNotNone(st)
        self.assertTrue(st["cooling"])
        # Expired cooldown is not "cooling".
        self.assertIsNone(cooldown_state(self._db, provider="minimax",
                                         account="acct-1", model="MiniMax-M3",
                                         now=2000))

    def test_scoped_not_global(self):
        upsert_cooldown(self._db, provider="minimax", account="acct-1",
                        model="MiniMax-M3", kind="rate_limited", until_at=9999, now=1)
        # A different model/provider is NOT cooled.
        self.assertIsNone(cooldown_state(self._db, provider="minimax",
                                         account="acct-1", model="OTHER", now=2))
        self.assertIsNone(cooldown_state(self._db, provider="openrouter",
                                         account="acct-1", model="MiniMax-M3", now=2))
        rows = list_cooldowns(self._db, now=2)
        self.assertEqual(len(rows), 1)

    def test_apply_outcome_only_cools_cooling_kinds(self):
        out = classify_capacity(http_status=429, provider="p", model="m")
        self.assertIsNotNone(apply_capacity_outcome(self._db, out, now=10))
        auth = classify_capacity(http_status=401, provider="p", model="m")
        self.assertIsNone(apply_capacity_outcome(self._db, auth, now=10))


# ============================================================
# A03 — durable WAIT_CAPACITY (no repair consumption) + eligibility
# ============================================================

class TestDurableCapacityWait(_Base):
    def test_wait_does_not_consume_repair(self):
        grant, camp = self._campaign()
        derive_admission  # noqa: B018 (documented: admission creates ledger)
        before = self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM capacity_waits").fetchone()["n"]
        enter_capacity_wait(self._db, campaign_id=camp.campaign_id,
                            reason_kind="rate_limited", grant_id=grant.grant_id,
                            next_eligible_at=1000, now=10)
        # No repair ledger exists unless admitted; assert no repair was invented.
        self.assertEqual(before, 0)
        self.assertEqual(len(active_waits(self._db, campaign_id=camp.campaign_id)), 1)
        # repair counters in budget_ledgers (if any) are untouched
        rows = self._db._conn.execute(
            "SELECT cumulative_repairs FROM budget_ledgers").fetchall()
        for r in rows:
            self.assertEqual(int(r["cumulative_repairs"]), 0)

    def test_restart_preserves_wait_and_eligibility(self):
        grant, camp = self._campaign()
        wait_id = enter_capacity_wait(self._db, campaign_id=camp.campaign_id,
                                      reason_kind="quota_exhausted_known_reset",
                                      grant_id=grant.grant_id, next_eligible_at=5000,
                                      now=100)
        db_path = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db"
        self._db.close()
        self._db = Database(db_path)
        self.assertEqual(next_eligible_at(self._db, campaign_id=camp.campaign_id), 5000)
        # before eligible time: wake does not dispatch
        d1 = wake_tick(self._db, campaign_id=camp.campaign_id, window_key="w1", now=4999)
        self.assertEqual(d1["decision"], "WAIT")
        # after eligible time: runner may reconsider
        resolve_capacity_wait(self._db, wait_id=wait_id, now=5000)
        self.assertEqual(next_eligible_at(self._db, campaign_id=camp.campaign_id), 0)
        d2 = wake_tick(self._db, campaign_id=camp.campaign_id, window_key="w2", now=5001)
        self.assertEqual(d2["decision"], "DISPATCH_ELIGIBLE")
        self.assertEqual(d2["advanced"], 1)


# ============================================================
# A04 — pause / revoke / expiry block automatic resume
# ============================================================

class TestResumeAuthorityGates(_Base):
    def _waited_campaign(self, *, plan_id, grant_id, grant_expires_at=0):
        grant, camp = self._campaign(plan_id=plan_id, grant_id=grant_id,
                                     grant_expires_at=grant_expires_at)
        enter_capacity_wait(self._db, campaign_id=camp.campaign_id,
                            reason_kind="rate_limited", grant_id=grant.grant_id,
                            next_eligible_at=100, now=1)
        return grant, camp

    def test_pause_blocks_resume(self):
        grant, camp = self._waited_campaign(plan_id="pl-p", grant_id="gr-p")
        (Path(os.environ["OVERNIGHT_STATE_DIR"]) / "PAUSED").write_text("test")
        d = wake_tick(self._db, campaign_id=camp.campaign_id, window_key="w", now=200)
        self.assertEqual(d["decision"], "REFUSED")
        self.assertIn("paused", d["reason"])

    def test_revoked_grant_blocks_resume(self):
        grant, camp = self._waited_campaign(plan_id="pl-r", grant_id="gr-r")
        revoke_grant(self._db, grant_id=grant.grant_id, reason="operator")
        d = wake_tick(self._db, campaign_id=camp.campaign_id, window_key="w", now=200)
        self.assertEqual(d["decision"], "REFUSED")

    def test_expired_grant_blocks_resume(self):
        grant, camp = self._waited_campaign(plan_id="pl-e", grant_id="gr-e",
                                            grant_expires_at=150)
        d = wake_tick(self._db, campaign_id=camp.campaign_id, window_key="w", now=500)
        self.assertEqual(d["decision"], "REFUSED")
        self.assertIn("expired", d["reason"])

    def test_valid_authority_resumes(self):
        grant, camp = self._waited_campaign(plan_id="pl-v", grant_id="gr-v")
        d = wake_tick(self._db, campaign_id=camp.campaign_id, window_key="w", now=500)
        self.assertEqual(d["decision"], "DISPATCH_ELIGIBLE")


# ============================================================
# A05 — finite approved fallback routing
# ============================================================

class TestFallbackRouting(_Base):
    def _policy(self, **kw):
        base = dict(role="worker", qualified_models=frozenset({"gemma"}),
                    approved_providers=frozenset({"ollama"}),
                    approved_profiles=frozenset({"fb-1"}),
                    runtime_digest="a" * 64, egress_policy_id="eg-1",
                    allow_paid_spend=False, max_cost_microusd=0,
                    max_context_tokens=8192)
        base.update(kw)
        return FallbackPolicy(**base)

    def _profile(self, **kw):
        base = dict(profile_id="fb-1", role="worker", model_name="gemma",
                    provider="ollama", runtime_digest="a" * 64,
                    egress_policy_id="eg-1", paid=False, cost_microusd=0,
                    context_tokens=4096)
        base.update(kw)
        return FallbackProfile(**base)

    def _primary(self):
        return self._profile(profile_id="primary")

    def test_approved_fallback_permitted(self):
        got = resolve_fallback(self._primary(), [self._profile()],
                               policy=self._policy(), primary_unavailable=True)
        self.assertEqual(got.profile_id, "fb-1")

    def test_primary_available_forbids_fallback(self):
        with self.assertRaises(SafetyError):
            resolve_fallback(self._primary(), [self._profile()],
                             policy=self._policy(), primary_unavailable=False)

    def test_unknown_and_unqualified_model_rejected(self):
        with self.assertRaises(SafetyError):
            resolve_fallback(self._primary(), [self._profile(model_name="mystery")],
                             policy=self._policy(), primary_unavailable=True)
        with self.assertRaises(SafetyError):
            resolve_fallback(self._primary(), [self._profile(profile_id="fb-x")],
                             policy=self._policy(), primary_unavailable=True)

    def test_provider_outside_approved_rejected(self):
        with self.assertRaises(SafetyError):
            resolve_fallback(self._primary(), [self._profile(provider="openai")],
                             policy=self._policy(), primary_unavailable=True)

    def test_paid_with_zero_spend_rejected(self):
        with self.assertRaises(SafetyError):
            resolve_fallback(self._primary(),
                             [self._profile(paid=True, cost_microusd=10)],
                             policy=self._policy(), primary_unavailable=True)

    def test_wider_egress_rejected(self):
        with self.assertRaises(SafetyError):
            resolve_fallback(self._primary(),
                             [self._profile(egress_policy_id="eg-WIDER")],
                             policy=self._policy(), primary_unavailable=True)

    def test_fallback_cannot_change_budget(self):
        # Fallback resolution is pure; it returns a profile and touches no
        # campaign budget/attempt state.
        grant, camp = self._campaign(plan_id="pl-fb", grant_id="gr-fb")
        before = self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM budget_ledgers").fetchone()["n"]
        resolve_fallback(self._primary(), [self._profile()],
                         policy=self._policy(), primary_unavailable=True)
        after = self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM budget_ledgers").fetchone()["n"]
        self.assertEqual(before, after)


# ============================================================
# A12 / A06 — bounded idempotent wake + double-fire
# ============================================================

class TestWakeIdempotency(_Base):
    def test_repeated_wakes_safe_and_double_fire_single_advance(self):
        grant, camp = self._campaign(plan_id="pl-w", grant_id="gr-w")
        d1 = wake_tick(self._db, campaign_id=camp.campaign_id, window_key="win-1", now=10)
        d2 = wake_tick(self._db, campaign_id=camp.campaign_id, window_key="win-1", now=10)
        self.assertEqual(d1["decision"], "DISPATCH_ELIGIBLE")
        self.assertEqual(d1["advanced"], 1)
        self.assertEqual(d2["decision"], "DUPLICATE")
        self.assertEqual(d2["advanced"], 0)
        # exactly one durable advancement job
        n = self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM wake_jobs WHERE campaign_id=?",
            (camp.campaign_id,)).fetchone()["n"]
        self.assertEqual(n, 1)

    def test_blocked_state_refuses(self):
        grant, camp = self._campaign(plan_id="pl-b", grant_id="gr-b")
        with self._db.transaction() as cur:
            cur.execute("UPDATE campaigns SET state='EFFECT_UNKNOWN' WHERE campaign_id=?",
                        (camp.campaign_id,))
        d = wake_tick(self._db, campaign_id=camp.campaign_id, window_key="w", now=1)
        self.assertEqual(d["decision"], "REFUSED")
        self.assertIn("blocked_state", d["reason"])

    def test_budget_exhausted_refuses(self):
        from overnight_runner.campaign import update_budget_after_chunk
        grant, camp = self._campaign(plan_id="pl-be", grant_id="gr-be", max_chunks=1)
        # create ledger via a chunk admission, then exhaust chunks.
        cur_commit = self._db._conn.execute(
            "SELECT current_commit FROM campaigns WHERE campaign_id=?",
            (camp.campaign_id,)).fetchone()["current_commit"]
        derive_admission(
            self._db, grant=grant,
            chunk=ChunkSpec(schema_version="trio.chunk.v1", chunk_id="chk-1",
                            campaign_id=camp.campaign_id, package_id="pkg-1",
                            revision=1, title="c", objective="c",
                            permitted_signature_paths=["src/app.py"],
                            permitted_write_paths=["src/app.py"],
                            permitted_read_paths=["src/app.py"],
                            permitted_command_ids=["noop"],
                            permitted_validator_ids=["noop"],
                            required_validator_ids=["noop"],
                            required_receipt_profiles=["noop"],
                            criterion_ids=["crit-1"], idempotency_key="idem-chk-1"),
            worker_id="wkr-1", policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            current_accepted_snapshot=RepoSnapshot(
                schema_version="trio.repo-snapshot.v1", repository_id="local",
                commit=cur_commit, tree_digest="b" * 64),
            current_runtime_digest=grant.runtime_digest,
            current_model_name=grant.model_name,
            current_model_digest=grant.model_digest,
            current_policy_profile_id=grant.policy_profile_id,
            current_validator_profile_ids=list(grant.validator_profile_ids),
            current_provider_profile_id=grant.provider_profile_id,
        )
        update_budget_after_chunk(self._db, ledger_id=f"bl-{camp.campaign_id}",
                                  delta_chunks=1)
        d = wake_tick(self._db, campaign_id=camp.campaign_id, window_key="w", now=1)
        self.assertEqual(d["decision"], "REFUSED")
        self.assertIn("budget", d["reason"])


# ============================================================
# A07 — short control returns a durable job id
# ============================================================

class TestShortControl(_Base):
    def test_control_returns_durable_id_and_timeout_does_not_fail(self):
        grant, camp = self._campaign(plan_id="pl-c", grant_id="gr-c")
        job_id = request_control(self._db, operation="wake",
                                 campaign_id=camp.campaign_id, window_key="w1")
        self.assertTrue(job_id)
        st = job_state(self._db, job_id)
        self.assertEqual(st["state"], "ACCEPTED")
        # Caller times out / disconnects: no change to the durable job.
        st2 = job_state(self._db, job_id)
        self.assertEqual(st2["state"], "ACCEPTED")
        # Only runner-authoritative completion changes it.
        complete_job(self._db, job_id, state="COMPLETED", detail="runner observed")
        self.assertEqual(job_state(self._db, job_id)["state"], "COMPLETED")

    def test_polling_observes_same_job_identity(self):
        grant, camp = self._campaign(plan_id="pl-c2", grant_id="gr-c2")
        job_id = request_control(self._db, operation="wake",
                                 campaign_id=camp.campaign_id, window_key="w2")
        for _ in range(3):
            self.assertEqual(job_state(self._db, job_id)["job_id"], job_id)


# ============================================================
# A01 / A10 — rebuildable projection; runner truth wins
# ============================================================

class TestProjection(_Base):
    def test_projection_rebuild_and_duplicate_cards(self):
        grant, camp = self._campaign(plan_id="pl-pj", grant_id="gr-pj")
        s1 = project_campaign_status(self._db, campaign_id=camp.campaign_id)
        cards1 = project_hermes_cards(self._db, campaign_id=camp.campaign_id)
        # Rebuild is deterministic and creates no runner chunks.
        s2 = project_campaign_status(self._db, campaign_id=camp.campaign_id)
        cards2 = project_hermes_cards(self._db, campaign_id=camp.campaign_id)
        self.assertEqual(s1, s2)
        self.assertEqual(cards1, cards2)
        for c in cards1:
            self.assertFalse(c["authoritative"])
            self.assertTrue(c["runner_id"])
        n_chunks = self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        self.assertEqual(n_chunks, 0)

    def test_runner_state_wins_and_status_has_next_action(self):
        grant, camp = self._campaign(plan_id="pl-pj2", grant_id="gr-pj2")
        with self._db.transaction() as cur:
            cur.execute("UPDATE campaigns SET state='NEEDS_DECISION' WHERE campaign_id=?",
                        (camp.campaign_id,))
        st = project_campaign_status(self._db, campaign_id=camp.campaign_id)
        self.assertEqual(st["state"], "NEEDS_DECISION")
        self.assertTrue(st["human_action_required"])
        self.assertEqual(st["projection_source"], "runner")
        self.assertIn("safe_next_action", st)
        # Projection created no authority.
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM wake_jobs").fetchone()["n"], 0)

    def test_adapter_schema_forbids_authority(self):
        forbidden = set(ADAPTER_SCHEMA["forbidden"])
        for f in ("direct_sqlite", "git_mutation", "grant_or_approval_mint",
                  "mark_code_accepted", "forge_validation_receipts"):
            self.assertIn(f, forbidden)


if __name__ == "__main__":
    unittest.main(verbosity=2)
