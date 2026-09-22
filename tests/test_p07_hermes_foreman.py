"""P07 — Hermes foreman integration (runner-authoritative).

MOCK/FIXTURE evidence: isolated runner DB + fake clock + fixture provider
outcomes. No live provider outage or live Hermes profile is required.
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
    ALLOWED_CONTROL_OPERATIONS,
    CapacityKind,
    ExecutionStartConflict,
    FallbackPolicy,
    FallbackProfile,
    _resolve_with_policy,
    active_waits,
    apply_capacity_outcome,
    capacity_outcome_history,
    classify_capacity,
    cooldown_state,
    complete_job,
    consume_control_requests,
    enter_capacity_wait,
    handle_provider_result,
    hermes_tick,
    job_state,
    list_cooldowns,
    load_fallback_policy,
    next_eligible_at,
    poll_wake_claim_execution,
    project_campaign_status,
    project_hermes_cards,
    reconcile_wake_claims,
    register_approved_fallback,
    request_control,
    resolve_capacity_wait,
    resolve_fallback,
    start_wake_claim_execution,
    stop_wake_claim_execution,
    upsert_cooldown,
    wake_claim,
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


def _grant(db, *, plan_id="pl-1", grant_id="gr-1", grant_expires_at=0,
           max_chunks=3, max_cost_microusd=1000):
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
                      max_wall_seconds=28800, max_cost_microusd=max_cost_microusd,
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
                  max_chunks=3, max_cost_microusd=1000):
        grant = _grant(self._db, plan_id=plan_id, grant_id=grant_id,
                       grant_expires_at=grant_expires_at, max_chunks=max_chunks,
                       max_cost_microusd=max_cost_microusd)
        camp = create_campaign(self._db, plan_id=grant.plan_id,
                               grant_id=grant.grant_id, base_commit="a" * 40,
                               base_tree_digest="b" * 64)
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        return grant, camp

    def _reopen(self):
        db_path = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db"
        self._db.close()
        self._db = Database(db_path)


# ============================================================
# A02 — classification (structured inputs only)
# ============================================================

class TestCapacityClassification(_Base):
    def test_fixture_corpus(self):
        cases = [
            (dict(http_status=429), CapacityKind.RATE_LIMITED),
            (dict(provider="minimax", provider_error_type="rate_limit_error"),
             CapacityKind.RATE_LIMITED),
            (dict(http_status=503), CapacityKind.OVERLOADED),
            (dict(http_status=529), CapacityKind.OVERLOADED),
            (dict(provider="anthropic", provider_error_type="overloaded_error"),
             CapacityKind.OVERLOADED),
            (dict(error_code="quota_exhausted", reset_at=12345),
             CapacityKind.QUOTA_EXHAUSTED_KNOWN_RESET),
            (dict(error_code="insufficient_quota"),
             CapacityKind.QUOTA_EXHAUSTED_NO_RESET),
            (dict(http_status=401), CapacityKind.AUTH_FAILURE),
            (dict(http_status=403), CapacityKind.AUTH_FAILURE),
            (dict(provider="openai", provider_error_type="invalid_api_key"),
             CapacityKind.AUTH_FAILURE),
            (dict(http_status=404), CapacityKind.INVALID_PROVIDER_MODEL_PROFILE),
            (dict(provider="anthropic", provider_error_type="not_found_error"),
             CapacityKind.INVALID_PROVIDER_MODEL_PROFILE),
            (dict(http_status=500), CapacityKind.PROVIDER_UNAVAILABLE),
            (dict(http_status=502), CapacityKind.PROVIDER_UNAVAILABLE),
            (dict(transport_error="ConnectionError"), CapacityKind.PROVIDER_UNAVAILABLE),
            (dict(transport_error="weird_socket_fault"), CapacityKind.UNKNOWN_TRANSPORT),
            (dict(), CapacityKind.OK),
        ]
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs):
                out = classify_capacity(provider=kwargs.pop("provider", "minimax"),
                                        model="MiniMax-M3", **kwargs)
                self.assertEqual(out.kind, expected.value)

    def test_free_form_message_is_not_authority(self):
        # The exact fixture the review flagged must NOT become quota on its
        # own: prose is evidence only.
        out = classify_capacity(message="Token Plan usage limit reached")
        self.assertEqual(out.kind, CapacityKind.OK.value)
        self.assertEqual(out.evidence, "Token Plan usage limit reached")

    def test_provider_adapter_requires_documented_code(self):
        # Unknown provider error type -> no authority.
        out = classify_capacity(provider="minimax", provider_error_type="totally_made_up")
        self.assertEqual(out.kind, CapacityKind.OK.value)


# ============================================================
# A02 — every outcome persisted; cooldown only when appropriate
# ============================================================

class TestCapacityOutcomePersistence(_Base):
    def test_every_kind_persists_a_durable_row(self):
        kinds = [
            dict(http_status=429),
            dict(http_status=503),
            dict(error_code="quota_exhausted", reset_at=9999),
            dict(error_code="insufficient_quota"),
            dict(http_status=401),
            dict(http_status=404),
            dict(http_status=500),
            dict(transport_error="weird_socket_fault"),
        ]
        for kwargs in kinds:
            out = classify_capacity(provider="minimax", model="MiniMax-M3", **kwargs)
            res = apply_capacity_outcome(self._db, out, account="acct-1", now=10)
            self.assertTrue(res["outcome_id"])
        rows = capacity_outcome_history(self._db)
        self.assertEqual(len(rows), len(kinds))
        self.assertEqual({r["kind"] for r in rows},
                         {classify_capacity(provider="minimax", **k).kind for k in kinds})

    def test_auth_stays_distinct_and_cooldown_only_when_appropriate(self):
        auth = classify_capacity(provider="minimax", http_status=401, model="m")
        res = apply_capacity_outcome(self._db, auth, now=10)
        self.assertEqual(auth.kind, CapacityKind.AUTH_FAILURE.value)
        self.assertIsNone(res["cooldown"])  # auth is not a cooling kind
        self.assertEqual(capacity_outcome_history(self._db,
                                                  kind="auth_failure")[0]["kind"],
                         "auth_failure")
        # rate limit DOES create a cooldown
        rl = classify_capacity(provider="minimax", http_status=429, model="m")
        res2 = apply_capacity_outcome(self._db, rl, now=10)
        self.assertIsNotNone(res2["cooldown"])

    def test_unknown_transport_distinct_and_restart_preserves(self):
        ut = classify_capacity(provider="minimax", transport_error="weird", model="m")
        apply_capacity_outcome(self._db, ut, now=10)
        self._reopen()
        rows = capacity_outcome_history(self._db)
        self.assertEqual(rows[0]["kind"], "unknown_transport")
        self.assertEqual(rows[0]["transport_class"], "weird")

    def test_restart_preserves_and_kind_query(self):
        for k in (dict(http_status=429), dict(error_code="insufficient_quota")):
            apply_capacity_outcome(
                self._db, classify_capacity(provider="p", model="m", **k), now=1)
        self._reopen()
        self.assertEqual(len(capacity_outcome_history(self._db)), 2)
        self.assertEqual(len(capacity_outcome_history(self._db, kind="rate_limited")), 1)


# ============================================================
# A03 — wake consults cooldown authority
# ============================================================

class TestWakeCooldownAuthority(_Base):
    def _cooldown(self, kind, until, model="gemma", provider="prv-1"):
        upsert_cooldown(self._db, provider=provider, model=model, kind=kind,
                        until_at=until, now=1)

    def test_known_reset_before_then_after(self):
        grant, camp = self._campaign(plan_id="pl-kr", grant_id="gr-kr")
        self._cooldown("rate_limited", 1000)
        d1 = wake_tick(self._db, campaign_id=camp.campaign_id, trigger_id="a", now=999)
        self.assertEqual(d1["decision"], "WAIT")
        self.assertIn("cooldown_until", d1["reason"])
        self.assertEqual(d1["advanced"], 0)
        d2 = wake_tick(self._db, campaign_id=camp.campaign_id, trigger_id="b", now=1001)
        self.assertEqual(d2["decision"], "DISPATCH_ELIGIBLE")
        self.assertEqual(d2["advanced"], 1)
        self.assertTrue(d2["claim_id"])

    def test_no_reset_does_not_auto_resume(self):
        grant, camp = self._campaign(plan_id="pl-nr", grant_id="gr-nr")
        self._cooldown("quota_exhausted_no_reset", 0)
        d = wake_tick(self._db, campaign_id=camp.campaign_id, now=10_000_000)
        self.assertEqual(d["decision"], "REFUSED")
        self.assertIn("no_reset", d["reason"])
        self.assertEqual(d["advanced"], 0)

    def test_auth_does_not_auto_resume(self):
        grant, camp = self._campaign(plan_id="pl-au", grant_id="gr-au")
        enter_capacity_wait(self._db, campaign_id=camp.campaign_id,
                            reason_kind="auth_failure", grant_id=grant.grant_id, now=1)
        d = wake_tick(self._db, campaign_id=camp.campaign_id, now=10_000_000)
        self.assertEqual(d["decision"], "REFUSED")
        self.assertIn("authority_required", d["reason"])

    def test_cooldown_blocks_even_with_zero_next_eligible(self):
        grant, camp = self._campaign(plan_id="pl-z", grant_id="gr-z")
        enter_capacity_wait(self._db, campaign_id=camp.campaign_id,
                            reason_kind="rate_limited", grant_id=grant.grant_id,
                            next_eligible_at=0, now=1)
        self._cooldown("rate_limited", 5_000)
        d = wake_tick(self._db, campaign_id=camp.campaign_id, now=100)
        self.assertEqual(d["decision"], "WAIT")
        self.assertIn("cooldown_until", d["reason"])

    def test_wait_resumes_without_manual_resolve_and_repair_unchanged(self):
        grant, camp = self._campaign(plan_id="pl-rs", grant_id="gr-rs")
        enter_capacity_wait(self._db, campaign_id=camp.campaign_id,
                            reason_kind="quota_exhausted_known_reset",
                            grant_id=grant.grant_id, next_eligible_at=500, now=1)
        # No manual resolve_capacity_wait() call.
        d = wake_tick(self._db, campaign_id=camp.campaign_id, now=501)
        self.assertEqual(d["decision"], "DISPATCH_ELIGIBLE")
        self.assertEqual(d["advanced"], 1)
        # The stale wait is resolved atomically with the claim.
        self.assertEqual(active_waits(self._db, campaign_id=camp.campaign_id), [])
        # No repair consumed.
        for r in self._db._conn.execute("SELECT cumulative_repairs FROM budget_ledgers").fetchall():
            self.assertEqual(int(r["cumulative_repairs"]), 0)

    def test_restart_preserves_cooldown_behavior(self):
        grant, camp = self._campaign(plan_id="pl-rst", grant_id="gr-rst")
        self._cooldown("rate_limited", 5000)
        self._reopen()
        d = wake_tick(self._db, campaign_id=camp.campaign_id, now=100)
        self.assertEqual(d["decision"], "WAIT")


# ============================================================
# A06 — wake claims real runner work; runner-owned idempotency
# ============================================================

class TestWakeRealClaim(_Base):
    def test_advanced_only_with_real_claim(self):
        grant, camp = self._campaign(plan_id="pl-cl", grant_id="gr-cl")
        d = wake_tick(self._db, campaign_id=camp.campaign_id, trigger_id="x", now=10)
        self.assertEqual(d["advanced"], 1)
        row = self._db._conn.execute(
            "SELECT * FROM wake_claims WHERE claim_id=?", (d["claim_id"],)).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["obligation_id"], d["obligation_id"])

    def test_competing_different_triggers_converge_on_one_claim(self):
        grant, camp = self._campaign(plan_id="pl-cmp", grant_id="gr-cmp")
        a = wake_tick(self._db, campaign_id=camp.campaign_id, trigger_id="trigger-A", now=10)
        b = wake_tick(self._db, campaign_id=camp.campaign_id, trigger_id="trigger-B", now=11)
        self.assertEqual(a["advanced"], 1)
        self.assertEqual(b["advanced"], 0)
        self.assertEqual(b["decision"], "DUPLICATE")
        n = self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM wake_claims WHERE campaign_id=?",
            (camp.campaign_id,)).fetchone()["n"]
        self.assertEqual(n, 1)

    def test_hermes_tick_discovers_due_obligations(self):
        grant, camp = self._campaign(plan_id="pl-tic", grant_id="gr-tic")
        out = hermes_tick(self._db, trigger_id="tick-1", now=10)
        self.assertEqual(out["claims"], 1)
        # Duplicate tick converges.
        out2 = hermes_tick(self._db, trigger_id="tick-2", now=11)
        self.assertEqual(out2["claims"], 0)


# ============================================================
# A05 — fallback authority provenance
# ============================================================

class TestFallbackProvenance(_Base):
    def _primary(self):
        return FallbackProfile(profile_id="primary", role="worker", model_name="gemma",
                               provider="prv-1", runtime_digest="a" * 64,
                               egress_policy_id="eg-1")

    def _candidate(self, **kw):
        base = dict(profile_id="fb-1", role="worker", model_name="gemma",
                    provider="prv-1", runtime_digest="a" * 64,
                    egress_policy_id="eg-1", paid=False, cost_microusd=0,
                    context_tokens=4096)
        base.update(kw)
        return FallbackProfile(**base)

    def test_caller_fabricated_policy_is_untrusted(self):
        fabricated = FallbackPolicy(
            role="worker", qualified_models=frozenset({"gemma"}),
            approved_providers=frozenset({"prv-1"}),
            approved_profiles=frozenset({"fb-1"}), runtime_digest="a" * 64,
            egress_policy_id="eg-1", allow_paid_spend=True,
            max_cost_microusd=10 ** 9, max_context_tokens=10 ** 9,
            provenance="caller")
        with self.assertRaises(SafetyError) as ctx:
            _resolve_with_policy(self._primary(), [self._candidate(paid=True)],
                                 policy=fabricated, primary_unavailable=True)
        self.assertIn("untrusted", str(ctx.exception).lower())

    def test_grant_without_paid_spend_rejects_paid_fallback(self):
        grant, camp = self._campaign(plan_id="pl-np", grant_id="gr-np",
                                     max_cost_microusd=0)
        register_approved_fallback(self._db, role="worker", profile_id="fb-1",
                                   model_name="gemma", provider="prv-1",
                                   runtime_digest="a" * 64, egress_policy_id="eg-1",
                                   paid=True, cost_microusd=10, context_tokens=4096)
        with self.assertRaises(SafetyError) as ctx:
            resolve_fallback(self._db, grant_id=grant.grant_id, role="worker",
                             primary=self._primary(),
                             candidates=[self._candidate(paid=True, cost_microusd=10)],
                             primary_unavailable=True)
        self.assertIn("paid", str(ctx.exception).lower())

    def test_unregistered_provider_is_rejected(self):
        grant, camp = self._campaign(plan_id="pl-ur", grant_id="gr-ur")
        register_approved_fallback(self._db, role="worker", profile_id="fb-1",
                                   model_name="gemma", provider="prv-1",
                                   runtime_digest="a" * 64, egress_policy_id="eg-1")
        with self.assertRaises(SafetyError):
            resolve_fallback(self._db, grant_id=grant.grant_id, role="worker",
                             primary=self._primary(),
                             candidates=[self._candidate(provider="OTHER")],
                             primary_unavailable=True)

    def test_qualified_approved_fallback_passes_and_budget_unchanged(self):
        grant, camp = self._campaign(plan_id="pl-ok", grant_id="gr-ok")
        register_approved_fallback(self._db, role="worker", profile_id="fb-1",
                                   model_name="gemma", provider="prv-1",
                                   runtime_digest="a" * 64, egress_policy_id="eg-1",
                                   paid=False, cost_microusd=0, context_tokens=4096)
        before = self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM budget_ledgers").fetchone()["n"]
        got = resolve_fallback(self._db, grant_id=grant.grant_id, role="worker",
                               primary=self._primary(), candidates=[self._candidate()],
                               primary_unavailable=True)
        self.assertEqual(got.profile_id, "fb-1")
        after = self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM budget_ledgers").fetchone()["n"]
        self.assertEqual(before, after)

    def test_load_policy_is_runner_provenance(self):
        grant, camp = self._campaign(plan_id="pl-pv", grant_id="gr-pv")
        pol = load_fallback_policy(self._db, grant_id=grant.grant_id, role="worker")
        self.assertEqual(pol.provenance, "runner")
        self.assertEqual(pol.runtime_digest, "a" * 64)
        self.assertEqual(pol.egress_policy_id, "eg-1")


# ============================================================
# A07 — short control + public job polling + allowlist
# ============================================================

class TestShortControlAndPolling(_Base):
    def test_public_job_polling_continuity(self):
        grant, camp = self._campaign(plan_id="pl-jb", grant_id="gr-jb")
        job_id = request_control(self._db, operation="wake",
                                 campaign_id=camp.campaign_id, window_key="w1")
        # Public polling (hermes-job) reads the SAME id/state.
        st = job_state(self._db, job_id)
        self.assertEqual(st["job_id"], job_id)
        self.assertEqual(st["state"], "ACCEPTED")
        # Caller timeout / disconnect: no change.
        self.assertEqual(job_state(self._db, job_id)["state"], "ACCEPTED")
        # Runner changes terminal state; public polling observes it.
        complete_job(self._db, job_id, state="COMPLETED", detail="runner")
        self.assertEqual(job_state(self._db, job_id)["state"], "COMPLETED")

    def test_control_operation_allowlist(self):
        grant, camp = self._campaign(plan_id="pl-al", grant_id="gr-al")
        for bad in ("approve", "shell", "mark-complete", "rm -rf", ""):
            with self.assertRaises(SafetyError):
                request_control(self._db, operation=bad, campaign_id=camp.campaign_id)
        for good in sorted(ALLOWED_CONTROL_OPERATIONS):
            jid = request_control(self._db, operation=good, campaign_id=camp.campaign_id)
            self.assertTrue(jid)


# ============================================================
# A01 / A10 — projection truth
# ============================================================

class TestProjection(_Base):
    def test_projection_reports_real_routing_and_job(self):
        grant, camp = self._campaign(plan_id="pl-pr", grant_id="gr-pr")
        st = project_campaign_status(self._db, campaign_id=camp.campaign_id)
        self.assertEqual(st["provider_profile"], "prv-1")  # NOT plan_id
        self.assertNotEqual(st["provider_profile"], camp.plan_id)
        self.assertEqual(st["model"], "gemma")
        self.assertEqual(st["runtime_digest"], "a" * 64)
        # After a wake, the authoritative runner job id is exposed.
        wake_tick(self._db, campaign_id=camp.campaign_id, trigger_id="t", now=5)
        st2 = project_campaign_status(self._db, campaign_id=camp.campaign_id)
        self.assertIsNotNone(st2["authoritative_runner_job_id"])

    def test_capacity_aware_safe_next_action(self):
        grant, camp = self._campaign(plan_id="pl-ca", grant_id="gr-ca")
        upsert_cooldown(self._db, provider="prv-1", model="gemma",
                        kind="quota_exhausted_no_reset", until_at=0, now=1)
        st = project_campaign_status(self._db, campaign_id=camp.campaign_id, now=2)
        self.assertIn("parked", st["safe_next_action"])
        self.assertNotIn("wake or admit", st["safe_next_action"])

    def test_projection_rebuild_and_duplicate_cards(self):
        grant, camp = self._campaign(plan_id="pl-pj", grant_id="gr-pj")
        s1 = project_campaign_status(self._db, campaign_id=camp.campaign_id)
        s2 = project_campaign_status(self._db, campaign_id=camp.campaign_id)
        self.assertEqual(s1, s2)
        c1 = project_hermes_cards(self._db, campaign_id=camp.campaign_id)
        c2 = project_hermes_cards(self._db, campaign_id=camp.campaign_id)
        self.assertEqual(c1, c2)
        for c in c1:
            self.assertFalse(c["authoritative"])
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM chunks").fetchone()["n"], 0)

    def test_adapter_schema_forbids_authority(self):
        forbidden = set(ADAPTER_SCHEMA["forbidden"])
        for f in ("direct_sqlite", "git_mutation", "grant_or_approval_mint",
                  "mark_code_accepted", "second_scheduler", "second_queue"):
            self.assertIn(f, forbidden)
        self.assertEqual(set(ADAPTER_SCHEMA["surfaces"]["control"]["allowed_operations"]),
                         set(ALLOWED_CONTROL_OPERATIONS))


# ============================================================
# A06 — real runner-owned EXECUTION (duplicate-safe)
# ============================================================

class TestWakeExecution(_Base):
    def _alive(self, pid: int) -> bool:
        # Zombie-aware: a defunct (unreaped) child is DEAD.
        if pid <= 0:
            return False
        try:
            with open(f"/proc/{pid}/stat", "r") as f:
                line = f.read()
        except OSError:
            return False
        rest = line.rsplit(")", 1)[-1].split()
        return bool(rest) and rest[0] not in ("Z", "X")

    def _budget_snapshot(self):
        rows = self._db._conn.execute(
            "SELECT campaign_id, cumulative_chunks, cumulative_model_calls, "
            "cumulative_tool_calls FROM budget_ledgers").fetchall()
        return sorted(tuple(r) for r in rows)

    def test_two_triggers_one_execution_concurrent(self):
        grant, camp = self._campaign(plan_id="pl-ex", grant_id="gr-ex")
        dbfile = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db"
        results: list[dict] = []
        barrier = threading.Barrier(2)

        def _trigger(tid):
            db2 = Database(dbfile)
            try:
                barrier.wait(timeout=10)
                results.append(wake_tick(db2, campaign_id=camp.campaign_id,
                                         trigger_id=tid, now=10))
            finally:
                db2.close()

        ts = [threading.Thread(target=_trigger, args=(f"trig-{i}",)) for i in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=20)
        # Exactly one claim (one trigger advances; the other converges).
        advanced = [r for r in results if r.get("advanced") == 1]
        self.assertEqual(len(advanced), 1)
        n_claims = self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM wake_claims WHERE campaign_id=?",
            (camp.campaign_id,)).fetchone()["n"]
        self.assertEqual(n_claims, 1)
        claim_id = advanced[0]["claim_id"]
        before = self._budget_snapshot()
        # Runner starts the OWNED bounded worker exactly once.
        run = start_wake_claim_execution(
            self._db, claim_id=claim_id,
            argv=[sys.executable, "-c", "print('ok')"], cwd=str(self._tmp), now=11)
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM wake_claims WHERE run_id=?",
            (run["run_id"],)).fetchone()["n"], 1)
        # Complete + observe the single terminal result.
        deadline = time.time() + 10
        last = None
        while time.time() < deadline:
            last = poll_wake_claim_execution(self._db, run_id=run["run_id"], now=12)
            if last["state"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.05)
        self.assertEqual(last["state"], "COMPLETED")
        self.assertEqual(last["exit_code"], 0)
        # No duplicate budget consumption / attempt.
        att = self._db._conn.execute(
            "SELECT attempts FROM wake_claims WHERE claim_id=?",
            (claim_id,)).fetchone()["attempts"]
        self.assertEqual(int(att), 1)
        self.assertEqual(self._budget_snapshot(), before)
        # Both callers observe the same authoritative lineage.
        job = job_state(self._db, f"wake-{advanced[0]['obligation_id']}")
        self.assertIn(job["state"], ("COMPLETED",))

    def test_crash_before_execution_reconciles_to_one_execution(self):
        grant, camp = self._campaign(plan_id="pl-cr", grant_id="gr-cr")
        d = wake_tick(self._db, campaign_id=camp.campaign_id, trigger_id="a", now=10)
        self.assertEqual(d["advanced"], 1)
        self.assertEqual(wake_claim(self._db, d["claim_id"])["state"], "CLAIMED")
        # Restart before execution started.
        self._reopen()
        recon = reconcile_wake_claims(self._db, now=11)
        self.assertEqual(len(recon), 1)
        self.assertEqual(wake_claim(self._db, d["claim_id"])["state"], "RETRY_ELIGIBLE")
        # Exactly one eventual execution.
        run = start_wake_claim_execution(
            self._db, claim_id=d["claim_id"],
            argv=[sys.executable, "-c", "pass"], cwd=str(self._tmp), now=12)
        deadline = time.time() + 10
        last = None
        while time.time() < deadline:
            last = poll_wake_claim_execution(self._db, run_id=run["run_id"], now=13)
            if last["state"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.05)
        self.assertEqual(last["state"], "COMPLETED")
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM wake_claims WHERE campaign_id=?",
            (camp.campaign_id,)).fetchone()["n"], 1)


# ============================================================
# A07 — real process supervision + durable job continuity
# ============================================================

class TestProcessSupervision(_Base):
    def _alive(self, pid: int) -> bool:
        # Zombie-aware: a defunct (unreaped) child is DEAD.
        if pid <= 0:
            return False
        try:
            with open(f"/proc/{pid}/stat", "r") as f:
                line = f.read()
        except OSError:
            return False
        rest = line.rsplit(")", 1)[-1].split()
        return bool(rest) and rest[0] not in ("Z", "X")

    def test_short_control_owns_long_worker_and_survives_caller(self):
        grant, camp = self._campaign(plan_id="pl-sup", grant_id="gr-sup")
        # 1) short control request -> durable control job id
        job_id = request_control(self._db, operation="wake",
                                 campaign_id=camp.campaign_id, window_key="w")
        self.assertEqual(job_state(self._db, job_id)["state"], "ACCEPTED")
        # 2) runner consumes it and starts the OWNED long-running fixture
        consumed = consume_control_requests(
            self._db,
            worker_argv=[sys.executable, "-c", "import time; time.sleep(1.5)"],
            cwd=str(self._tmp), now=10)
        self.assertEqual(len(consumed), 1)
        run_id = consumed[0]["run_id"]
        self.assertTrue(run_id)
        claim = wake_claim(self._db, consumed[0]["claim_id"])
        child_pid = int(claim["pid"])
        # 3) the durable job is RUNNING (NOT accepted-as-complete)
        self.assertEqual(job_state(self._db, job_id)["state"], "RUNNING")
        self.assertTrue(self._alive(child_pid))
        # 4) caller times out / disconnects: nothing happens to the worker
        time.sleep(0.2)
        self.assertTrue(self._alive(child_pid))
        self.assertEqual(poll_wake_claim_execution(self._db, run_id=run_id, now=11)["state"],
                         "RUNNING")
        self.assertEqual(job_state(self._db, job_id)["state"], "RUNNING")
        # 5) polling observes the SAME identity/state
        self.assertEqual(job_state(self._db, job_id)["job_id"], job_id)
        # 6) runner-owned normal completion
        deadline = time.time() + 10
        last = None
        while time.time() < deadline:
            last = poll_wake_claim_execution(self._db, run_id=run_id, now=12)
            if last["state"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.05)
        self.assertEqual(last["state"], "COMPLETED")
        self.assertEqual(job_state(self._db, job_id)["state"], "COMPLETED")

    def test_only_runner_supervision_terminates_owned_worker(self):
        grant, camp = self._campaign(plan_id="pl-kill", grant_id="gr-kill")
        d = wake_tick(self._db, campaign_id=camp.campaign_id, trigger_id="k", now=1)
        run = start_wake_claim_execution(
            self._db, claim_id=d["claim_id"],
            argv=[sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=str(self._tmp), now=2)
        claim = wake_claim(self._db, d["claim_id"])
        child_pid = int(claim["pid"])
        self.assertTrue(self._alive(child_pid))
        # An unrelated bystander process (our own shell child) is untouched.
        bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            res = stop_wake_claim_execution(self._db, run_id=run["run_id"], now=3)
            self.assertIn(res["state"], ("COMPLETED", "FAILED"))
            time.sleep(0.2)
            self.assertFalse(self._alive(child_pid))
            self.assertTrue(self._alive(bystander.pid))  # no broad kill
        finally:
            bystander.terminate()
            bystander.wait(timeout=5)


# ============================================================
# A03/A06 — one trusted provider-result -> capacity-state path
# ============================================================

class TestProviderResultToState(_Base):
    def test_auth_result_establishes_blocking_state_atomically(self):
        grant, camp = self._campaign(plan_id="pl-auth", grant_id="gr-auth")
        out = classify_capacity(provider="prv-1", http_status=401, model="gemma")
        res = handle_provider_result(self._db, out, campaign_id=camp.campaign_id,
                                     grant_id=grant.grant_id, now=10)
        self.assertEqual(res["wait_reason"], "auth_failure")
        # Persisted outcome AND a durable blocking wait exist together.
        self.assertEqual(len(capacity_outcome_history(self._db, kind="auth_failure")), 1)
        waits = active_waits(self._db, campaign_id=camp.campaign_id)
        self.assertEqual([w["reason_kind"] for w in waits], ["auth_failure"])
        # Wake cannot dispatch.
        d = wake_tick(self._db, campaign_id=camp.campaign_id, now=20)
        self.assertEqual(d["decision"], "REFUSED")
        self.assertIn("authority_required", d["reason"])

    def test_rate_limit_creates_wait_and_cooldown(self):
        grant, camp = self._campaign(plan_id="pl-rl2", grant_id="gr-rl2")
        out = classify_capacity(provider="prv-1", http_status=429, model="gemma",
                                retry_after_seconds=60)
        res = handle_provider_result(self._db, out, campaign_id=camp.campaign_id, now=10)
        self.assertEqual(res["wait_reason"], "rate_limited")
        self.assertEqual(res["cooldown_until"], 70)
        self.assertIsNotNone(cooldown_state(self._db, provider="prv-1",
                                            model="gemma", now=20))
        d = wake_tick(self._db, campaign_id=camp.campaign_id, now=20)
        self.assertEqual(d["decision"], "WAIT")

    def test_no_reset_quota_parks_indefinitely(self):
        grant, camp = self._campaign(plan_id="pl-nq", grant_id="gr-nq")
        out = classify_capacity(provider="prv-1", error_code="insufficient_quota",
                                model="gemma")
        handle_provider_result(self._db, out, campaign_id=camp.campaign_id, now=10)
        cool = cooldown_state(self._db, provider="prv-1", model="gemma", now=10_000_000)
        self.assertIsNotNone(cool)  # until_at == 0 => indefinite
        d = wake_tick(self._db, campaign_id=camp.campaign_id, now=10_000_001)
        self.assertEqual(d["decision"], "REFUSED")
        self.assertIn("no_reset", d["reason"])

    def test_unknown_transport_fails_closed(self):
        grant, camp = self._campaign(plan_id="pl-ut", grant_id="gr-ut")
        out = classify_capacity(provider="prv-1", transport_error="weird", model="gemma")
        res = handle_provider_result(self._db, out, campaign_id=camp.campaign_id, now=10)
        self.assertEqual(res["wait_reason"], "unknown_transport")
        d = wake_tick(self._db, campaign_id=camp.campaign_id, now=11)
        self.assertEqual(d["decision"], "REFUSED")

    def test_idempotent_no_duplicate_wait(self):
        grant, camp = self._campaign(plan_id="pl-idem", grant_id="gr-idem")
        for _ in range(3):
            out = classify_capacity(provider="prv-1", http_status=429, model="gemma")
            handle_provider_result(self._db, out, campaign_id=camp.campaign_id, now=10)
        self.assertEqual(len(active_waits(self._db, campaign_id=camp.campaign_id)), 1)


# ============================================================
# A03 — exact provider/account/model cooldown identity
# ============================================================

class TestCooldownModelIdentity(_Base):
    def test_wait_model_identity_matches_cooldown(self):
        grant, camp = self._campaign(plan_id="pl-mi", grant_id="gr-mi")
        enter_capacity_wait(self._db, campaign_id=camp.campaign_id,
                            reason_kind="rate_limited", grant_id=grant.grant_id,
                            provider="minimax", account="A",
                            profile="minimax-oauth-profile", model="MiniMax-M3",
                            next_eligible_at=1000, now=1)
        upsert_cooldown(self._db, provider="minimax", account="A", model="MiniMax-M3",
                        kind="rate_limited", until_at=5000, now=1)
        d = wake_tick(self._db, campaign_id=camp.campaign_id, now=100)
        self.assertEqual(d["decision"], "WAIT")
        self.assertIn("cooldown_until", d["reason"])

    def test_other_model_not_blocked(self):
        grant, camp = self._campaign(plan_id="pl-om", grant_id="gr-om")
        # Wait is for model=gemma; the cooldown is for a DIFFERENT model.
        enter_capacity_wait(self._db, campaign_id=camp.campaign_id,
                            reason_kind="rate_limited", grant_id=grant.grant_id,
                            provider="minimax", account="A", model="gemma",
                            next_eligible_at=0, now=1)
        upsert_cooldown(self._db, provider="minimax", account="A", model="OTHER-MODEL",
                        kind="rate_limited", until_at=5000, now=1)
        # OTHER-MODEL cooldown must NOT block gemma on the same provider/account.
        d = wake_tick(self._db, campaign_id=camp.campaign_id, now=100)
        self.assertEqual(d["decision"], "DISPATCH_ELIGIBLE")


# ============================================================
# A05/A09 — paid fallback authorization contract
# ============================================================

class TestPaidFallbackContract(_Base):
    def _primary(self):
        return FallbackProfile(profile_id="primary", role="worker", model_name="gemma",
                               provider="prv-1", runtime_digest="a" * 64,
                               egress_policy_id="eg-1")

    def _candidate(self, paid=True, cost=10):
        return FallbackProfile(profile_id="fb-1", role="worker", model_name="gemma",
                               provider="prv-1", runtime_digest="a" * 64,
                               egress_policy_id="eg-1", paid=paid, cost_microusd=cost,
                               context_tokens=4096)

    def test_grant_spend_allowance_is_the_paid_authorization(self):
        # max_cost_microusd > 0 is the EXPLICIT grant-level paid-spend
        # authorization; == 0 denies paid fallback.
        grant_ok, _ = self._campaign(plan_id="pl-paid", grant_id="gr-paid",
                                     max_cost_microusd=5000)
        register_approved_fallback(self._db, role="worker", profile_id="fb-1",
                                   model_name="gemma", provider="prv-1",
                                   runtime_digest="a" * 64, egress_policy_id="eg-1",
                                   paid=True, cost_microusd=10, context_tokens=4096)
        got = resolve_fallback(self._db, grant_id=grant_ok.grant_id, role="worker",
                               primary=self._primary(), candidates=[self._candidate()],
                               primary_unavailable=True)
        self.assertEqual(got.profile_id, "fb-1")

    def test_registry_has_no_hermes_or_worker_surface(self):
        # The approved_fallbacks registry is a trusted operator surface;
        # the adapter schema exposes no registration capability.
        self.assertNotIn("register_fallback", json.dumps(ADAPTER_SCHEMA))
        for surface in ADAPTER_SCHEMA["surfaces"].values():
            self.assertNotIn("fallback", json.dumps(surface).lower())


# ============================================================
# A06/A07 — execution-start atomicity + restart supervision
# ============================================================

def _marker_worker(marker_dir: Path) -> list[str]:
    """Fixture worker that writes a UNIQUE per-process marker file."""
    return [sys.executable, "-c",
            "import os,sys;open(os.path.join(sys.argv[1],str(os.getpid())),'w').write('x')",
            str(marker_dir)]


class TestExecutionAuthority(_Base):
    def _alive(self, pid: int) -> bool:
        # Zombie-aware: a defunct (unreaped) child is DEAD.
        if pid <= 0:
            return False
        try:
            with open(f"/proc/{pid}/stat", "r") as f:
                line = f.read()
        except OSError:
            return False
        rest = line.rsplit(")", 1)[-1].split()
        return bool(rest) and rest[0] not in ("Z", "X")

    def _wait_terminal(self, run_id, timeout=10.0):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            last = poll_wake_claim_execution(self._db, run_id=run_id, now=int(time.time()))
            if last["state"] in ("COMPLETED", "FAILED"):
                return last
            time.sleep(0.03)
        return last

    def test_concurrent_start_one_winner(self):
        import overnight_runner.p07 as p07mod
        grant, camp = self._campaign(plan_id="pl-cstart", grant_id="gr-cstart")
        d = wake_tick(self._db, campaign_id=camp.campaign_id, trigger_id="x", now=1)
        claim_id = d["claim_id"]
        marker = self._tmp / "markers"
        marker.mkdir()
        dbfile = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db"
        results: list[dict] = []
        errors: list[str] = []
        barrier = threading.Barrier(2)

        def _start():
            db2 = Database(dbfile)
            try:
                barrier.wait(timeout=10)
                results.append(start_wake_claim_execution(
                    db2, claim_id=claim_id, argv=_marker_worker(marker),
                    cwd=str(self._tmp), now=2, result_dir=str(self._tmp / "res")))
            except Exception as e:  # noqa: BLE001
                errors.append(type(e).__name__)
            finally:
                db2.close()

        ts = [threading.Thread(target=_start) for _ in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=20)
        # Exactly one winner, one conflict.
        self.assertEqual(len(results), 1)
        self.assertEqual(errors, ["ExecutionStartConflict"])
        run_id = results[0]["run_id"]
        self.assertEqual(wake_claim(self._db, claim_id)["run_id"], run_id)
        self.assertEqual(int(wake_claim(self._db, claim_id)["attempts"]), 1)
        last = self._wait_terminal(run_id)
        self.assertEqual(last["state"], "COMPLETED")
        # Exactly one OS worker process wrote a marker (no second spawn).
        self.assertEqual(len(list(marker.iterdir())), 1)
        p07mod._CHILDREN.clear()

    def test_concurrent_consume_one_winner(self):
        grant, camp = self._campaign(plan_id="pl-cc", grant_id="gr-cc")
        job_id = request_control(self._db, operation="wake",
                                 campaign_id=camp.campaign_id, window_key="w")
        marker = self._tmp / "m2"
        marker.mkdir()
        dbfile = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db"
        outs: list[list] = []
        barrier = threading.Barrier(2)

        def _consume():
            db2 = Database(dbfile)
            try:
                barrier.wait(timeout=10)
                outs.append(consume_control_requests(
                    db2, worker_argv=_marker_worker(marker), cwd=str(self._tmp), now=3))
            finally:
                db2.close()

        ts = [threading.Thread(target=_consume) for _ in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=20)
        consumed = [o for o in outs if any(x.get("consumed") for x in o)]
        self.assertEqual(len(consumed), 1)
        # exactly one claim / one run / one marker
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM wake_claims").fetchone()["n"], 1)
        claim = self._db._conn.execute("SELECT * FROM wake_claims").fetchone()
        self.assertTrue(claim["run_id"])
        self.assertEqual(int(claim["attempts"]), 1)
        last = self._wait_terminal(claim["run_id"])
        self.assertEqual(last["state"], "COMPLETED")
        self.assertEqual(len(list(marker.iterdir())), 1)

    def test_starting_crash_before_spawn_recovery(self):
        grant, camp = self._campaign(plan_id="pl-st", grant_id="gr-st")
        d = wake_tick(self._db, campaign_id=camp.campaign_id, trigger_id="x", now=1)
        claim_id = d["claim_id"]
        # Simulate a crash at STARTING with NO process spawned.
        with self._db.transaction() as cur:
            cur.execute("UPDATE wake_claims SET state='STARTING', run_id=?, "
                        "exec_token='tok', pid=0, result_path=? WHERE claim_id=?",
                        (f"run-{claim_id}", str(self._tmp / "none.json"), claim_id))
        self._reopen()
        recon = reconcile_wake_claims(self._db, now=2)
        self.assertEqual(recon[0]["to"], "RETRY_ELIGIBLE")
        # Exactly one eventual execution.
        run = start_wake_claim_execution(
            self._db, claim_id=claim_id, argv=[sys.executable, "-c", "pass"],
            cwd=str(self._tmp), now=3)
        self.assertEqual(self._wait_terminal(run["run_id"])["state"], "COMPLETED")

    def test_spawn_failure_no_starting_deadlock(self):
        grant, camp = self._campaign(plan_id="pl-sf", grant_id="gr-sf")
        d = wake_tick(self._db, campaign_id=camp.campaign_id, trigger_id="x", now=1)
        # Invalid cwd forces the OUTER spawn to fail (bounded policy).
        with self.assertRaises(SafetyError):
            start_wake_claim_execution(
                self._db, claim_id=d["claim_id"],
                argv=[sys.executable, "-c", "pass"],
                cwd="/nonexistent-dir-xyz-123", now=2)
        # Never a permanent STARTING deadlock.
        self.assertEqual(wake_claim(self._db, d["claim_id"])["state"], "RETRY_ELIGIBLE")

    def test_restart_preserves_running_and_recovers_durable_result(self):
        import overnight_runner.p07 as p07mod
        grant, camp = self._campaign(plan_id="pl-rr", grant_id="gr-rr")
        d = wake_tick(self._db, campaign_id=camp.campaign_id, trigger_id="x", now=1)
        run = start_wake_claim_execution(
            self._db, claim_id=d["claim_id"],
            argv=[sys.executable, "-c", "import time;time.sleep(0.6)"],
            cwd=str(self._tmp), now=2, result_dir=str(self._tmp / "res2"))
        claim = wake_claim(self._db, d["claim_id"])
        self.assertGreater(int(claim["pid"]), 0)
        self.assertTrue(claim["boot_id"] and claim["start_time"])
        # Simulate a parent restart: durable DB identity survives; handles lost.
        self._reopen()
        p07mod._CHILDREN.clear()
        recon = reconcile_wake_claims(self._db, now=3)
        self.assertEqual(recon[0]["to"], "RUNNING")  # exact owned process alive
        # Post-restart polling uses identity + durable result.
        last = self._wait_terminal(run["run_id"])
        self.assertEqual(last["state"], "COMPLETED")
        # Lineage preserved on the public job surface.
        job = job_state(self._db, f"wake-{d['obligation_id']}")
        self.assertEqual(job["state"], "COMPLETED")

    def test_post_restart_owned_termination_and_bystander(self):
        import overnight_runner.p07 as p07mod
        grant, camp = self._campaign(plan_id="pl-rt", grant_id="gr-rt")
        d = wake_tick(self._db, campaign_id=camp.campaign_id, trigger_id="x", now=1)
        run = start_wake_claim_execution(
            self._db, claim_id=d["claim_id"],
            argv=[sys.executable, "-c", "import time;time.sleep(30)"],
            cwd=str(self._tmp), now=2)
        child_pid = int(wake_claim(self._db, d["claim_id"])["pid"])
        bystander = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
        try:
            self._reopen()
            p07mod._CHILDREN.clear()  # restart: no in-process handle
            res = stop_wake_claim_execution(self._db, run_id=run["run_id"], now=3)
            self.assertEqual(res["state"], "FAILED")
            time.sleep(0.3)
            self.assertFalse(self._alive(child_pid))
            self.assertTrue(self._alive(bystander.pid))  # never broad-kill
        finally:
            bystander.terminate()
            bystander.wait(timeout=5)

    def test_pid_identity_mismatch_never_signalled(self):
        import overnight_runner.p07 as p07mod
        grant, camp = self._campaign(plan_id="pl-pid", grant_id="gr-pid")
        d = wake_tick(self._db, campaign_id=camp.campaign_id, trigger_id="x", now=1)
        run = start_wake_claim_execution(
            self._db, claim_id=d["claim_id"],
            argv=[sys.executable, "-c", "import time;time.sleep(30)"],
            cwd=str(self._tmp), now=2)
        claim = wake_claim(self._db, d["claim_id"])
        child_pid, pgid = int(claim["pid"]), int(claim["pgid"])
        # Tamper the recorded start-time (simulated PID reuse).
        with self._db.transaction() as cur:
            cur.execute("UPDATE wake_claims SET start_time='0' WHERE claim_id=?",
                        (d["claim_id"],))
        p07mod._CHILDREN.clear()
        try:
            stop_wake_claim_execution(self._db, run_id=run["run_id"], now=3)
            # Identity mismatch => the process must NOT be signalled.
            self.assertTrue(self._alive(child_pid))
        finally:
            try:
                os.killpg(pgid, 9)
            except OSError:
                pass

    def test_double_trigger_end_to_end_rehearsal(self):
        grant, camp = self._campaign(plan_id="pl-e2e", grant_id="gr-e2e")
        # Two INDEPENDENT short-control invocations (different trigger ids).
        j1 = request_control(self._db, operation="wake",
                             campaign_id=camp.campaign_id, window_key="trig-A")
        j2 = request_control(self._db, operation="wake",
                             campaign_id=camp.campaign_id, window_key="trig-B")
        marker = self._tmp / "m3"
        marker.mkdir()
        for jid in (j1, j2):
            # Each consumed by an independent runner consumer.
            tmp_claim = self._db._conn.execute(
                "SELECT COUNT(*) AS n FROM wake_claims").fetchone()["n"]
            consume_control_requests(self._db, worker_argv=_marker_worker(marker),
                                     cwd=str(self._tmp), now=5)
            del tmp_claim
        # One control/wake winner, one claim, one run, one owned child.
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM wake_claims").fetchone()["n"], 1)
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM chunks").fetchone()["n"], 0)
        claim = self._db._conn.execute("SELECT * FROM wake_claims").fetchone()
        self.assertEqual(int(claim["attempts"]), 1)
        last = self._wait_terminal(claim["run_id"])
        self.assertEqual(last["state"], "COMPLETED")
        self.assertEqual(len(list(marker.iterdir())), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
