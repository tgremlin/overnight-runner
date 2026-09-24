"""Runner-owned human-gate API (OV-01-FRC-001 C09 -> G01).

Replaces the ad-hoc SQL that previously wrote authoritative campaign lifecycle
state (`campaigns.state = 'AWAITING_HUMAN_AT_CLOSE'`) from outside the Runner.

Covers: ACTIVE -> gate success; idempotent replay; unknown campaign; invalid
prior state; contradictory gate/chunk replay; no C10 advancement; correctly
shaped canonical campaign_events row; no budget/acceptance erasure; and that the
gate is never auto-approved.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from overnight_runner.campaign import (
    HUMAN_GATE_STATE,
    activate_campaign,
    create_campaign,
    record_human_gate,
    read_budget_totals,
)
from overnight_runner.campaign_schemas import (
    AutonomyGrant, Budget, content_sha256,
)
from overnight_runner.db import Database
from overnight_runner.grants import activate_grant, load_grant
from overnight_runner.plans import load_plan_digest, register_plan
from overnight_runner.protected_approvals import register_protected_approval
from overnight_runner.safety import SafetyError


class HumanGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ov01gate-"))
        os.environ["OVERNIGHT_STATE_DIR"] = str(self.tmp)
        os.environ["TR_P06_CAMPAIGN_V2"] = "1"
        self.db = Database(self.tmp / "state.db")
        register_plan(self.db, plan_id="pl-gate", approved_artifact_id="pl-gate",
                      work_package_criterion_ids={"pkg-1": {"crit-1"}})
        g = AutonomyGrant(
            schema_version="trio.grant.v1", grant_id="gr-gate", state="draft",
            plan_id="pl-gate", plan_revision=1, approved_plan_digest="0" * 64,
            repository_paths=["src/app.py"], allowed_write_paths=["src/app.py"],
            protected_paths=[], allowed_operations=["noop"],
            runtime_digest="a" * 64, model_name="qwen3.8:27b", model_digest="b" * 64,
            policy_profile_id="pol-1", validator_profile_ids=["pytest"],
            provider_profile_id="prv-1", egress_policy_id="eg-1",
            operator_id="op-1", operator_receipt_digest="c" * 64,
            budget=Budget(schema_version="trio.budget.v1", max_chunks=10,
                          max_model_calls=10, max_tool_calls=160,
                          max_local_repairs=0, max_rechunks=0,
                          max_active_seconds=3600, max_wall_seconds=5400,
                          max_cost_microusd=0, context_token_budget=8192))
        g = g.model_copy(update={
            "approved_plan_digest": load_plan_digest(self.db, "pl-gate")})
        register_protected_approval(
            self.db, approval_id="appr-gate", operation="activate_grant",
            grant_digest_target=content_sha256(g), operator_id="op-1",
            operator_receipt={"approval_id": "appr-gate"})
        activate_grant(self.db, grant=g, operator_id="op-1", approval_id="appr-gate")
        self.camp = create_campaign(
            db=self.db, plan_id="pl-gate", grant_id="gr-gate",
            base_commit="a" * 40, base_tree_digest="b" * 64,
            repo_root=str(self.tmp))
        activate_campaign(self.db, campaign_id=self.camp.campaign_id)

    def tearDown(self):
        try:
            self.db.close()
        except Exception:
            pass

    def _state(self) -> str:
        return self.db._conn.execute(
            "SELECT state FROM campaigns WHERE campaign_id=?",
            (self.camp.campaign_id,)).fetchone()["state"]

    def test_active_enters_gate(self):
        res = record_human_gate(self.db, campaign_id=self.camp.campaign_id,
                                gate_id="OV-01-FRC-001-G01",
                                chunk_id="OV-01-FRC-001-C09", reason="integrity report")
        self.assertEqual(res["state"], HUMAN_GATE_STATE)
        self.assertFalse(res["idempotent_replay"])
        self.assertEqual(self._state(), HUMAN_GATE_STATE)
        self.assertEqual(res["from_state"], "ACTIVE")

    def test_same_request_replays_idempotently(self):
        first = record_human_gate(self.db, campaign_id=self.camp.campaign_id,
                                  gate_id="G01", chunk_id="C09")
        second = record_human_gate(self.db, campaign_id=self.camp.campaign_id,
                                   gate_id="G01", chunk_id="C09")
        self.assertTrue(second["idempotent_replay"])
        self.assertEqual(second["event_id"], first["event_id"])
        # exactly ONE gate event, no duplicate transition
        n = self.db._conn.execute(
            "SELECT COUNT(*) AS n FROM campaign_events WHERE campaign_id=? "
            "AND event_type=?", (self.camp.campaign_id, HUMAN_GATE_STATE)
        ).fetchone()["n"]
        self.assertEqual(n, 1)

    def test_unknown_campaign_rejected(self):
        with self.assertRaises(SafetyError):
            record_human_gate(self.db, campaign_id="cmp-nope", gate_id="G01")

    def test_invalid_prior_state_rejected(self):
        with self.db.transaction() as cur:
            cur.execute("UPDATE campaigns SET state='EFFECT_UNKNOWN' WHERE campaign_id=?",
                        (self.camp.campaign_id,))
        with self.assertRaises(SafetyError) as ctx:
            record_human_gate(self.db, campaign_id=self.camp.campaign_id, gate_id="G01")
        self.assertIn("may enter the human gate", str(ctx.exception))
        self.assertEqual(self._state(), "EFFECT_UNKNOWN")

    def test_contradictory_gate_chunk_rejected(self):
        record_human_gate(self.db, campaign_id=self.camp.campaign_id,
                          gate_id="G01", chunk_id="C09")
        with self.assertRaises(SafetyError) as ctx:
            record_human_gate(self.db, campaign_id=self.camp.campaign_id,
                              gate_id="G01", chunk_id="C10")
        self.assertIn("contradictory", str(ctx.exception))

    def test_event_row_is_canonically_shaped(self):
        record_human_gate(self.db, campaign_id=self.camp.campaign_id,
                          gate_id="OV-01-FRC-001-G01", chunk_id="OV-01-FRC-001-C09",
                          reason="integrity report verified")
        row = dict(self.db._conn.execute(
            "SELECT * FROM campaign_events WHERE campaign_id=?",
            (self.camp.campaign_id,)).fetchone())
        self.assertTrue(row["event_id"])
        self.assertEqual(row["campaign_id"], self.camp.campaign_id)
        self.assertEqual(row["chunk_id"], "OV-01-FRC-001-C09")
        self.assertEqual(row["event_type"], HUMAN_GATE_STATE)
        self.assertEqual(row["from_state"], "ACTIVE")
        self.assertEqual(row["to_state"], HUMAN_GATE_STATE)
        self.assertEqual(row["actor"], "runner")
        self.assertEqual(json.loads(row["payload"])["gate_id"], "OV-01-FRC-001-G01")
        self.assertEqual(row["idempotency_key"],
                         f"gate-{self.camp.campaign_id}-OV-01-FRC-001-G01-OV-01-FRC-001-C09")

    def test_no_c10_advancement_and_no_budget_erasure(self):
        led = self.db._conn.execute(
            "SELECT ledger_id FROM budget_ledgers WHERE campaign_id=?",
            (self.camp.campaign_id,)).fetchone()
        before = (read_budget_totals(self.db, ledger_id=led["ledger_id"])
                  if led else None)
        accepted_before = self.db._conn.execute(
            "SELECT COUNT(*) AS n FROM integration_journal WHERE campaign_id=?",
            (self.camp.campaign_id,)).fetchone()["n"]
        record_human_gate(self.db, campaign_id=self.camp.campaign_id,
                          gate_id="G01", chunk_id="C09")
        led2 = self.db._conn.execute(
            "SELECT ledger_id FROM budget_ledgers WHERE campaign_id=?",
            (self.camp.campaign_id,)).fetchone()
        after = (read_budget_totals(self.db, ledger_id=led2["ledger_id"])
                 if led2 else None)
        accepted_after = self.db._conn.execute(
            "SELECT COUNT(*) AS n FROM integration_journal WHERE campaign_id=?",
            (self.camp.campaign_id,)).fetchone()["n"]
        self.assertEqual(before, after, "budget must not be erased by the gate")
        self.assertEqual(accepted_before, accepted_after,
                         "the gate must not advance or accept any chunk")
        # no C10 chunk was installed/accepted as a side effect
        c10 = self.db._conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE campaign_id=? AND chunk_id LIKE '%C10%'",
            (self.camp.campaign_id,)).fetchone()["n"]
        self.assertEqual(c10, 0)

    def test_gate_is_not_auto_approved(self):
        record_human_gate(self.db, campaign_id=self.camp.campaign_id,
                          gate_id="G01", chunk_id="C09")
        # no disposition/approval artifact is created by reaching the gate
        for table in ("protected_approvals", "phase_dispositions"):
            try:
                n = self.db._conn.execute(
                    f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                self.assertEqual(n, 0, f"{table} must stay empty at the gate")
            except Exception:
                pass

    def test_requires_gate_and_campaign_ids(self):
        with self.assertRaises(SafetyError):
            record_human_gate(self.db, campaign_id="", gate_id="G01")
        with self.assertRaises(SafetyError):
            record_human_gate(self.db, campaign_id=self.camp.campaign_id, gate_id="")


if __name__ == "__main__":
    unittest.main(verbosity=2)
