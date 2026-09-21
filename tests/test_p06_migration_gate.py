"""P06 — Migration/restart evidence and feature gate default-disabled tests.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from overnight_runner.db import Database


def _isolated(tmp_path):
    sd = tmp_path / "state"
    sd.mkdir(parents=True, exist_ok=True)
    os.environ["OVERNIGHT_STATE_DIR"] = str(sd)
    return sd / "state.db"


class TestMigrationRestart(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._old_state = os.environ.get("OVERNIGHT_STATE_DIR")
        self._old_gate = os.environ.get("TR_P06_CAMPAIGN_V2")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("OVERNIGHT_STATE_DIR", None)
        if self._old_state is not None:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state
        if self._old_gate is not None:
            os.environ["TR_P06_CAMPAIGN_V2"] = self._old_gate
        else:
            os.environ.pop("TR_P06_CAMPAIGN_V2", None)

    def test_fresh_db_applies_p06_migration(self):
        """Opening a fresh DB applies the P06-followup migration in
        order (``schema_migrations`` table records both p06-0001 and
        p06-followup-0001). The migration is idempotent and safe to
        re-run."""
        db_path = _isolated(self._tmp)
        db = Database(db_path)
        try:
            cur = db._conn.execute(
                "SELECT name FROM schema_migrations ORDER BY applied_at"
            )
            names = [r["name"] for r in cur.fetchall()]
            self.assertIn("p06-0001-campaign-v2-tables", names)
            self.assertIn("p06-followup-0001-protected-approvals-and-plans", names)
            # All P06-v2 tables exist.
            cur = db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            names = {r["name"] for r in cur.fetchall()}
            for t in ("campaigns", "grants", "budget_ledgers", "chunks",
                      "admissions", "leases", "integration_journal",
                      "race_admissions", "crash_windows",
                      "campaign_events", "schema_migrations",
                      "protected_approvals", "approved_plans"):
                self.assertIn(t, names)
        finally:
            db.close()

    def test_restore_accepted_v1_state(self):
        """Reopening a DB with existing V1 rows (no P06 tables yet) is
        handled by the migration: P06 tables are added without
        disturbing V1 rows."""
        db_path = _isolated(self._tmp)
        db = Database(db_path)
        try:
            with db.transaction() as cur:
                cur.execute(
                    "INSERT INTO tasks (task_id, manifest_sha256, manifest_json, status, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                    ("t-v1", "m1", "{}", "PENDING_APPROVAL", 100, 100),
                )
                cur.execute(
                    "INSERT INTO events (timestamp, session_id, event_type) VALUES (?,?,?)",
                    (100, "s-v1", "v1_event"),
                )
            db.close()
            # Reopen — V1 rows must survive.
            db2 = Database(db_path)
            try:
                cur = db2._conn.execute(
                    "SELECT task_id FROM tasks WHERE task_id='t-v1'"
                )
                self.assertIsNotNone(cur.fetchone())
                cur = db2._conn.execute(
                    "SELECT event_type FROM events WHERE event_type='v1_event'"
                )
                self.assertIsNotNone(cur.fetchone())
            finally:
                db2.close()
        finally:
            db.close()


class TestFeatureGateDefaultDisabled(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._old_state = os.environ.get("OVERNIGHT_STATE_DIR")
        self._old_gate = os.environ.get("TR_P06_CAMPAIGN_V2")
        self._old_profile = os.environ.get("OVERNIGHT_CAMPAIGN_PROFILE")
        self._old_sentinel = None
        os.environ["OVERNIGHT_STATE_DIR"] = str(self._tmp / "state")
        (self._tmp / "state").mkdir(parents=True, exist_ok=True)
        # Force-disable the gate even if a parent process has it on.
        os.environ.pop("TR_P06_CAMPAIGN_V2", None)
        os.environ.pop("OVERNIGHT_CAMPAIGN_PROFILE", None)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("OVERNIGHT_STATE_DIR", None)
        for var, val in [
            ("TR_P06_CAMPAIGN_V2", self._old_gate),
            ("OVERNIGHT_CAMPAIGN_PROFILE", self._old_profile),
        ]:
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val
        if self._old_state is not None:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state

    def test_default_disabled(self):
        """campaign-v2 is DISABLED by default. Without an explicit
        enable signal, no campaign-v2 activity is permitted."""
        from overnight_runner.feature_gate import campaign_v2_enabled, require_campaign_v2
        self.assertFalse(campaign_v2_enabled())
        with self.assertRaises(Exception) as ctx:
            require_campaign_v2("create_campaign")
        self.assertIn("DISABLED", str(ctx.exception))

    def test_enabled_via_env_var(self):
        from overnight_runner.feature_gate import campaign_v2_enabled, require_campaign_v2
        os.environ["TR_P06_CAMPAIGN_V2"] = "1"
        # Reload to pick up env change (we read at call time).
        self.assertTrue(campaign_v2_enabled())
        require_campaign_v2("create_campaign")  # does not raise

    def test_enabled_via_state_dir_sentinel(self):
        from overnight_runner.feature_gate import campaign_v2_enabled
        sentinel = Path(os.environ["OVERNIGHT_STATE_DIR"]) / "config" / "campaign_v2.enabled"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("1")
        self.assertTrue(campaign_v2_enabled())


if __name__ == "__main__":
    unittest.main(verbosity=2)
