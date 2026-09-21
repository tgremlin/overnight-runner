"""P06-A01 — Delegated admission tests.

All current identity evidence is REQUIRED. Plan authority is
REQUIRED. The chunk's package_id and criterion ids MUST match a
registered plan. Protected operator approvals are the only
authority surface.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from overnight_runner.campaign_schemas import (
    AdmissionReceipt,
    AutonomyGrant,
    Budget,
    ChunkSpec,
    RepoSnapshot,
    content_sha256,
)
from overnight_runner.db import Database
from overnight_runner.grants import activate_grant, load_grant, revoke_grant
from overnight_runner.admission import derive_admission, AdmissionConflict
from overnight_runner.plans import register_plan
from overnight_runner.protected_approvals import register_protected_approval
from overnight_runner.safety import git_commit_all, git_init_empty
from overnight_runner.campaign import create_campaign, activate_campaign


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "fixture_repo"
    repo.mkdir()
    git_init_empty(repo)
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("ORIGINAL\n")
    git_commit_all(repo, "init")
    return repo


def _valid_grant(grant_id: str = "gr-test-1") -> AutonomyGrant:
    return AutonomyGrant(
        schema_version="trio.grant.v1",
        grant_id=grant_id,
        state="draft",
        plan_id="pl-test-1",
        plan_revision=1,
        approved_plan_digest="d" * 64,
        repository_paths=["src/app.py"],
        allowed_write_paths=["src/app.py"],
        protected_paths=[],
        allowed_operations=["noop", "py_compile"],
        runtime_digest="a" * 64,
        model_name="gemma4:12b",
        model_digest="b" * 64,
        policy_profile_id="pol-1",
        validator_profile_ids=["noop"],
        provider_profile_id="prv-1",
        egress_policy_id="eg-1",
        operator_id="op-1",
        operator_receipt_digest="c" * 64,
        budget=Budget(
            schema_version="trio.budget.v1",
            max_model_calls=10, max_tool_calls=20,
            max_local_repairs=2, max_rechunks=1,
            max_active_seconds=3600, max_wall_seconds=28800,
            max_cost_microusd=1000,
            grant_expires_at=0,
            max_chunks=3, max_families=1,
            context_token_budget=8192,
        ),
    )


def _grant_digest(g: AutonomyGrant) -> str:
    return content_sha256(g)


def _plan_digest_for(plan_id: str, packages: dict) -> str:
    """Compute the plan_digest the way ``register_plan`` does."""
    import hashlib
    canonical = {
        "plan_id": plan_id,
        "work_packages": [
            {"package_id": pkg, "criterion_ids": sorted(sorted(crits))}
            for pkg, crits in sorted(packages.items())
        ],
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _required_kwargs(grant: AutonomyGrant) -> dict:
    """All REQUIRED identity kwargs (P06 follow-up #2 A02).

    No ``plan_id`` here; admission pins the plan via the grant.
    No ``runtime_digest``; current_runtime_digest is the SINGLE
    authoritative runtime argument.
    """
    return dict(
        current_runtime_digest=grant.runtime_digest,
        current_model_name=grant.model_name,
        current_model_digest=grant.model_digest,
        current_policy_profile_id=grant.policy_profile_id,
        current_validator_profile_ids=list(grant.validator_profile_ids),
        current_provider_profile_id=grant.provider_profile_id,
    )


def _activate_via_protected_approval(
    db: Database, *, grant: AutonomyGrant, operator_id: str = "op-test"
) -> str:
    digest = _grant_digest(grant)
    approval_id = f"appr-gr-{grant.grant_id}-{int(time.time()*1000)}"
    register_protected_approval(
        db,
        approval_id=approval_id,
        operation="activate_grant",
        grant_digest_target=digest,
        operator_id=operator_id,
        operator_receipt={"approval_id": approval_id, "auth_token": "OPS/12345"},
    )
    activate_grant(
        db, grant=grant,
        operator_id=operator_id,
        approval_id=approval_id,
    )
    return approval_id


def _register_plan_with_grant_pinning(
    db: Database, *, grant: AutonomyGrant, packages: dict[str, set[str]]
) -> str:
    """Register a plan AND pin the grant's ``approved_plan_digest`` to it.

    Returns the plan_digest so callers can verify pin in tests.
    """
    plan_digest = _plan_digest_for(grant.plan_id, packages)
    grant_pinned = grant.model_copy(update={"approved_plan_digest": plan_digest})
    register_plan(
        db,
        plan_id=grant_pinned.plan_id,
        approved_artifact_id=grant_pinned.plan_id,
        work_package_criterion_ids=packages,
    )
    return plan_digest, grant_pinned


def _setup_with_plan_and_active_grant(
    db: Database, *, base_sha_padded: str, grant: AutonomyGrant
) -> tuple[str, AutonomyGrant]:
    plan_digest, grant_pinned = _register_plan_with_grant_pinning(
        db, grant=grant, packages={"pkg-1": {"crit-1", "crit-2"}},
    )
    _activate_via_protected_approval(db, grant=grant_pinned)
    camp = create_campaign(
        db, plan_id=grant_pinned.plan_id, grant_id=grant_pinned.grant_id,
        base_commit=base_sha_padded, base_tree_digest="b" * 64,
    )
    activate_campaign(db, campaign_id=camp.campaign_id)
    return camp.campaign_id, grant_pinned


def _make_plan_and_active_grant(db: Database, grant: AutonomyGrant) -> tuple[str, AutonomyGrant]:
    """Register the plan, pin the grant, activate, and create the campaign.

    Returns ``(campaign_id, pinned_grant)``. The ``pinned_grant`` has
    ``approved_plan_digest`` set to the registered plan's digest and
    is what callers should pass to ``derive_admission`` so the
    admission-time plan binding check matches the stored grant.
    """
    plan_digest, grant_pinned = _register_plan_with_grant_pinning(
        db, grant=grant, packages={"pkg-1": {"crit-1", "crit-2"}},
    )
    _activate_via_protected_approval(db, grant=grant_pinned)
    camp = create_campaign(
        db, plan_id=grant_pinned.plan_id, grant_id=grant_pinned.grant_id,
        base_commit="a" * 64, base_tree_digest="b" * 64,
    )
    activate_campaign(db, campaign_id=camp.campaign_id)
    return camp.campaign_id, grant_pinned


def _valid_chunk(campaign_id: str) -> ChunkSpec:
    return ChunkSpec(
        schema_version="trio.chunk.v1",
        chunk_id="chk-1",
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
        idempotency_key="idem-test-1",
    )


def _snap(commit: str = "a" * 64, tree: str = "b" * 64) -> RepoSnapshot:
    return RepoSnapshot(
        schema_version="trio.repo-snapshot.v1",
        repository_id="local",
        commit=commit, tree_digest=tree,
    )


def _insert_draft_grant_directly(db: Database, grant: AutonomyGrant) -> None:
    """Insert the grant row directly with state='draft' (skipping
    the protected-approval activation path), for tests that need a
    draft grant in the durable store but without going through
    ``activate_grant``."""
    import json as _json
    from overnight_runner.grants import grant_payload
    with db.transaction() as cur:
        cur.execute(
            """
            INSERT INTO grants (
                grant_id, grant_digest, state, plan_id, plan_revision,
                operator_id, operator_receipt_digest,
                activated_at, revoked_at, revoked_reason,
                payload_json
            ) VALUES (?,?,?,?,?,?,?,0,0,'',?)
            """,
            (
                grant.grant_id, "draft-digest", grant.state,
                grant.plan_id, grant.plan_revision,
                grant.operator_id, grant.operator_receipt_digest,
                _json.dumps(grant_payload(grant)),
            ),
        )


def _head_sha_padded(repo: Path) -> str:
    short = __import__("subprocess").run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo),
        capture_output=True, text=True,
    ).stdout.strip()
    return short + "0" * 24


class TestP06A01DelegatedAdmission(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._old_state = os.environ.get("OVERNIGHT_STATE_DIR")
        self._old_flag = os.environ.get("OVERNIGHT_RECEIPTS")
        os.environ["OVERNIGHT_STATE_DIR"] = str(self._tmp / "state")
        os.environ["OVERNIGHT_RECEIPTS"] = "1"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("OVERNIGHT_STATE_DIR", None)
        os.environ.pop("OVERNIGHT_RECEIPTS", None)
        if self._old_state is not None:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state
        if self._old_flag is not None:
            os.environ["OVERNIGHT_RECEIPTS"] = self._old_flag

    # ---------- Valid protected approval + plan admits ----------

    def test_valid_protected_approval_admits_later_chunk(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant()
            plan_digest, grant_pinned = _register_plan_with_grant_pinning(
                db, grant=grant, packages={"pkg-1": {"crit-1", "crit-2"}},
            )
            _activate_via_protected_approval(db, grant=grant_pinned)
            base = _head_sha_padded(repo)
            camp = create_campaign(
                db, plan_id=grant_pinned.plan_id, grant_id=grant_pinned.grant_id,
                base_commit=base, base_tree_digest="b" * 64,
            )
            activate_campaign(db, campaign_id=camp.campaign_id)
            snap = _snap(commit=base)
            chunk = _valid_chunk(camp.campaign_id)
            receipt, _ = derive_admission(
                db, grant=grant_pinned, chunk=chunk,
                worker_id="wkr-1",
                policy_profile_id=grant_pinned.policy_profile_id,
                validator_profile_ids=list(grant_pinned.validator_profile_ids),
                provider_profile_id=grant_pinned.provider_profile_id,
                current_accepted_snapshot=snap,
                **_required_kwargs(grant_pinned),
            )
            self.assertIsInstance(receipt, AdmissionReceipt)
        finally:
            db.close()

    # ---------- Draft grant rejects ----------

    def test_draft_grant_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant()
            plan_digest, grant_pinned = _register_plan_with_grant_pinning(
                db, grant=grant, packages={"pkg-1": {"crit-1", "crit-2"}},
            )
            base = _head_sha_padded(repo)
            # Insert a draft grant (no protected approval activation).
            _insert_draft_grant_directly(db, grant_pinned)
            # P06 follow-up #3 (item 1/14): a non-active grant may not
            # create a campaign — campaign creation now carries the
            # grant-state authority check.
            with self.assertRaises(Exception) as ctx:
                create_campaign(
                    db, plan_id=grant_pinned.plan_id, grant_id=grant_pinned.grant_id,
                    base_commit=base, base_tree_digest="b" * 64,
                )
            self.assertIn("grant state", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Arbitrary caller-supplied approval rejects ----------

    def test_arbitrary_caller_supplied_approval_dict_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant()
            plan_digest, grant_pinned = _register_plan_with_grant_pinning(
                db, grant=grant, packages={"pkg-1": {"crit-1", "crit-2"}},
            )
            with self.assertRaises(Exception) as ctx:
                activate_grant(
                    db, grant=grant_pinned,
                    operator_id="op-attacker",
                    approval_id="appr-forged",
                )
            # Without the plan digest pin, this would be a "plan not
            # registered" error. With the plan registered, the first
            # check that fires is the protected-approval lookup, so
            # the error mentions "approval".
            self.assertTrue(
                "approval" in str(ctx.exception).lower() or
                "registered" in str(ctx.exception).lower()
            )
        finally:
            db.close()

    # ---------- Approval for wrong grant digest rejects ----------

    def test_forged_approval_for_wrong_grant_digest_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant()
            # Register the plan first so the approval digest mismatch
            # is what fires (not the plan-not-registered guard).
            plan_digest, grant_pinned = _register_plan_with_grant_pinning(
                db, grant=grant, packages={"pkg-1": {"crit-1", "crit-2"}},
            )
            register_protected_approval(
                db,
                approval_id="appr-wrong-digest",
                operation="activate_grant",
                grant_digest_target="0" * 64,
                operator_id="op-1",
                operator_receipt={"approval_id": "appr-wrong-digest"},
            )
            with self.assertRaises(Exception) as ctx:
                activate_grant(
                    db, grant=grant_pinned,
                    operator_id="op-1",
                    approval_id="appr-wrong-digest",
                )
            self.assertIn("digest", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Approval operator mismatch rejects ----------

    def test_approval_operator_mismatch_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant()
            plan_digest, grant_pinned = _register_plan_with_grant_pinning(
                db, grant=grant, packages={"pkg-1": {"crit-1", "crit-2"}},
            )
            digest = _grant_digest(grant_pinned)
            register_protected_approval(
                db,
                approval_id="appr-A",
                operation="activate_grant",
                grant_digest_target=digest,
                operator_id="op-real",
                operator_receipt={"approval_id": "appr-A"},
            )
            with self.assertRaises(Exception) as ctx:
                activate_grant(
                    db, grant=grant_pinned,
                    operator_id="op-attacker",
                    approval_id="appr-A",
                )
            self.assertIn("operator", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Revoked grant rejects ----------

    def test_revoked_grant_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant()
            plan_digest, grant_pinned = _register_plan_with_grant_pinning(
                db, grant=grant, packages={"pkg-1": {"crit-1", "crit-2"}},
            )
            _activate_via_protected_approval(db, grant=grant_pinned)
            revoke_grant(db, grant_id=grant_pinned.grant_id, reason="operator revoked")
            stored = load_grant(db, grant_pinned.grant_id)
            self.assertEqual(stored.state, "revoked")
            base = _head_sha_padded(repo)
            # P06 follow-up #3 (item 1/14): a revoked grant may not
            # create a campaign; the rejection fires at creation.
            with self.assertRaises(Exception) as ctx:
                create_campaign(
                    db, plan_id=grant.plan_id, grant_id=grant.grant_id,
                    base_commit=base, base_tree_digest="b" * 64,
                )
            self.assertIn("grant state", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Expired grant rejects ----------

    def test_expired_grant_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            expired_budget = Budget(
                schema_version="trio.budget.v1",
                max_model_calls=10, max_tool_calls=20,
                max_local_repairs=2, max_rechunks=1,
                max_active_seconds=3600, max_wall_seconds=28800,
                max_cost_microusd=1000,
                grant_expires_at=int(time.time()) - 10,
                max_chunks=3, max_families=1,
                context_token_budget=8192,
            )
            grant = _valid_grant().model_copy(update={"budget": expired_budget})
            plan_digest, grant_pinned = _register_plan_with_grant_pinning(
                db, grant=grant, packages={"pkg-1": {"crit-1", "crit-2"}},
            )
            _activate_via_protected_approval(db, grant=grant_pinned)
            base = _head_sha_padded(repo)
            camp = create_campaign(
                db, plan_id=grant_pinned.plan_id, grant_id=grant_pinned.grant_id,
                base_commit=base, base_tree_digest="b" * 64,
            )
            activate_campaign(db, campaign_id=camp.campaign_id)
            stored = load_grant(db, grant_pinned.grant_id)
            with self.assertRaises(Exception) as ctx:
                derive_admission(
                    db, grant=stored, chunk=_valid_chunk(camp.campaign_id),

                    worker_id="wkr-1",
                    policy_profile_id=stored.policy_profile_id,
                    validator_profile_ids=list(stored.validator_profile_ids),
                    provider_profile_id=stored.provider_profile_id,
                    current_accepted_snapshot=_snap(commit=base),
                    **_required_kwargs(stored),
                )
            self.assertIn("grant", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Path expansion rejects ----------

    def test_path_expansion_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant()
            base = _head_sha_padded(repo)
            cid, grant = _make_plan_and_active_grant(db, grant)
            chunk = _valid_chunk(cid).model_copy(update={
                "permitted_write_paths": ["src/app.py", "outside.py"],
            })
            with self.assertRaises(Exception) as ctx:
                derive_admission(
                    db, grant=grant, chunk=chunk,

                    worker_id="wkr-1",
                    policy_profile_id=grant.policy_profile_id,
                    validator_profile_ids=list(grant.validator_profile_ids),
                    provider_profile_id=grant.provider_profile_id,
                    current_accepted_snapshot=_snap(commit=base),
                    **_required_kwargs(grant),
                )
            self.assertIn("write_paths", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Validator expansion rejects ----------

    def test_validator_expansion_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant()
            base = _head_sha_padded(repo)
            cid, grant = _make_plan_and_active_grant(db, grant)
            chunk = _valid_chunk(cid).model_copy(update={
                "required_validator_ids": ["noop", "shell_echo"],
            })
            with self.assertRaises(Exception) as ctx:
                derive_admission(
                    db, grant=grant, chunk=chunk,

                    worker_id="wkr-1",
                    policy_profile_id=grant.policy_profile_id,
                    validator_profile_ids=list(grant.validator_profile_ids),
                    provider_profile_id=grant.provider_profile_id,
                    current_accepted_snapshot=_snap(commit=base),
                    **_required_kwargs(grant),
                )
            self.assertIn("validator", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Runtime drift rejects ----------

    def test_runtime_drift_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant()
            cid, grant = _make_plan_and_active_grant(db, grant)
            kw = _required_kwargs(grant)
            kw["current_runtime_digest"] = "wrong" + "0" * 58
            kw["current_accepted_snapshot"] = _snap(commit="a" * 64)
            with self.assertRaises(Exception) as ctx:
                derive_admission(
                    db, grant=grant, chunk=_valid_chunk(cid),
                    worker_id="wkr-1",
                    policy_profile_id=grant.policy_profile_id,
                    validator_profile_ids=list(grant.validator_profile_ids),
                    provider_profile_id=grant.provider_profile_id,
                    **kw,
                )
            self.assertIn("runtime", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Baseline mismatch rejects ----------

    def test_baseline_mismatch_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant()
            base = _head_sha_padded(repo)
            cid, grant = _make_plan_and_active_grant(db, grant)
            with self.assertRaises(Exception) as ctx:
                derive_admission(
                    db, grant=grant, chunk=_valid_chunk(cid),

                    worker_id="wkr-1",
                    policy_profile_id=grant.policy_profile_id,
                    validator_profile_ids=list(grant.validator_profile_ids),
                    provider_profile_id=grant.provider_profile_id,
                    current_accepted_snapshot=_snap(commit="c" * 64,
                                                       tree="e" * 64),
                    **_required_kwargs(grant),
                )
            self.assertIn("baseline_mismatch", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Unknown package_id rejects (plan-required) ----------

    def test_unknown_package_id_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant()
            base = _head_sha_padded(repo)
            cid, grant = _make_plan_and_active_grant(db, grant)
            chunk = _valid_chunk(cid).model_copy(update={
                "package_id": "pkg-FORGED",
            })
            with self.assertRaises(Exception) as ctx:
                derive_admission(
                    db, grant=grant, chunk=chunk,

                    worker_id="wkr-1",
                    policy_profile_id=grant.policy_profile_id,
                    validator_profile_ids=list(grant.validator_profile_ids),
                    provider_profile_id=grant.provider_profile_id,
                    current_accepted_snapshot=_snap(commit=base),
                    **_required_kwargs(grant),
                )
            self.assertIn("unknown", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Foreign/unapproved criterion rejects ----------

    def test_foreign_criterion_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant()
            base = _head_sha_padded(repo)
            cid, grant = _make_plan_and_active_grant(db, grant)
            chunk = _valid_chunk(cid).model_copy(update={
                "criterion_ids": ["crit-1", "crit-FOREIGN"],
            })
            with self.assertRaises(Exception) as ctx:
                derive_admission(
                    db, grant=grant, chunk=chunk,

                    worker_id="wkr-1",
                    policy_profile_id=grant.policy_profile_id,
                    validator_profile_ids=list(grant.validator_profile_ids),
                    provider_profile_id=grant.provider_profile_id,
                    current_accepted_snapshot=_snap(commit=base),
                    **_required_kwargs(grant),
                )
            self.assertIn("unknown package criterion", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Missing runtime_digest rejects (REQUIRED identity evidence) ----------

    def test_missing_runtime_digest_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant()
            cid, grant = _make_plan_and_active_grant(db, grant)
            kw = _required_kwargs(grant)
            kw["current_runtime_digest"] = None
            kw["current_accepted_snapshot"] = _snap(commit="a" * 64)
            with self.assertRaises(Exception) as ctx:
                derive_admission(
                    db, grant=grant, chunk=_valid_chunk(cid),
                    worker_id="wkr-1",
                    policy_profile_id=grant.policy_profile_id,
                    validator_profile_ids=list(grant.validator_profile_ids),
                    provider_profile_id=grant.provider_profile_id,
                    **kw,
                )
            self.assertIn("current_runtime_digest", str(ctx.exception).lower())
        finally:
            db.close()

    # ---------- Missing current_model_name rejects (REQUIRED identity evidence) ----------

    def test_missing_model_name_rejects(self):
        repo = _make_repo(self._tmp)
        db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")
        try:
            grant = _valid_grant()
            cid, grant = _make_plan_and_active_grant(db, grant)
            kw = _required_kwargs(grant)
            kw["current_model_name"] = None
            kw["current_accepted_snapshot"] = _snap(commit="a" * 64)
            with self.assertRaises(Exception) as ctx:
                derive_admission(
                    db, grant=grant, chunk=_valid_chunk(cid),
                    worker_id="wkr-1",
                    policy_profile_id=grant.policy_profile_id,
                    validator_profile_ids=list(grant.validator_profile_ids),
                    provider_profile_id=grant.provider_profile_id,
                    **kw,
                )
            self.assertIn("model_name", str(ctx.exception).lower())
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
