"""OV-01-FRC-001 fence test matrix.

P06 follow-up #4 + directive §10. Locks the production invariant:

  * an admitted worker presents ONE lease_id with its admission_fence;
  * a heartbeat extends expiry WITHOUT mutating the fence;
  * a takeover bumps the fence permanently;
  * same generation + right lease + same process => apply PASS;
  * the same generation + WRONG lease => REJECT;
  * the same generation + right lease + WRONG process => REJECT;
  * after a real takeover at N+1, the original holder's apply using N => REJECT;
  * the new owner's apply at N+1 => PASS;
  * admission -> apply (no heartbeat, no takeover) leaves the fence at 1.

These tests are PURE to the runner: no PydanticAI, no model, no Hermes.
They run against a disposable in-memory DB and a tmp worktree.

Reference: directive §6 ("ONE authoritative interpretation"), §7 (no unsafe
fixes), §10 (this matrix).
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

# Campaign-v2 must be enabled for ``apply_campaign_patch`` / takeover / CAS.
os.environ.setdefault("TR_P06_CAMPAIGN_V2", "1")

from overnight_runner.admission import derive_admission
from overnight_runner.admission_lease import heartbeat_admission_lease
from overnight_runner.broker import Broker
from overnight_runner.campaign import (
    activate_campaign,
    create_campaign,
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
from overnight_runner.grants import activate_grant
from overnight_runner.plans import register_plan, load_plan_digest
from overnight_runner.protected_approvals import register_protected_approval
from overnight_runner.resources import (
    current_fence,
    revoke_for_takeover,
)
from overnight_runner.safety import SafetyError
from overnight_runner.schemas import CreateFileArgs, ToolCall


# -- Fixture scaffolding ----------------------------------------------------

LOCK_TEMPLATE = {
    "schema_version": "trio.grant.v1",
    "plan_revision": 1,
    "runtime_identity": {
        "runtime_digest": "ab" * 32,
        "model_name": "stub-model",
        "model_digest": "cd" * 32,
        "provider_profile_id": "stub-provider",
    },
    "identity": {
        "policy_profile_id": "policy-stub",
        "validator_profile_ids": ["py_compile"],
    },
    "egress_policy_id": "egress-stub",
    "repository_paths": ["scripts", "tests"],
    "allowed_write_paths": ["scripts/test_fence_authority.py"],
    "protected_paths": ["secrets/"],
    "allowed_operations": ["noop"],
    "budget": {
        "schema_version": "trio.budget.v1",
        "max_chunks": 4,
        "max_model_calls": 8,
        "max_tool_calls": 16,
        "max_local_repairs": 4,
        "max_rechunks": 1,
        "max_active_seconds": 2400,
        "max_wall_seconds": 7200,
        "max_cost_microusd": 100000,
        "context_token_budget": 16384,
    },
}


def _build_plan_and_grant(db, *, plan_id: str, grant_id: str,
                           extra_package_ids: list[str] | None = None):
    tpl = LOCK_TEMPLATE
    b = tpl["budget"]
    tident = tpl["runtime_identity"]
    criteria_map = {
        "OV-01-FRC-001-C01": {"OV-01-FRC-001-C01.inventories-workspace-fingerprint"},
    }
    for extra in extra_package_ids or []:
        criteria_map[extra] = {"OV-01-FRC-001-C01.inventories-workspace-fingerprint"}
    register_plan(
        db,
        plan_id=plan_id,
        approved_artifact_id="fence-authority-test",
        work_package_criterion_ids=criteria_map,
    )
    plan_digest = load_plan_digest(db, plan_id)
    grant = AutonomyGrant(
        schema_version=tpl["schema_version"],
        grant_id=grant_id,
        state="draft",
        plan_id=plan_id,
        plan_revision=tpl["plan_revision"],
        approved_plan_digest="0" * 64,
        repository_paths=list(tpl["repository_paths"]),
        allowed_write_paths=list(tpl["allowed_write_paths"]),
        protected_paths=list(tpl["protected_paths"]),
        allowed_operations=list(tpl["allowed_operations"]),
        runtime_digest=tident["runtime_digest"],
        model_name=tident["model_name"],
        model_digest=tident["model_digest"],
        policy_profile_id=tpl["identity"]["policy_profile_id"],
        validator_profile_ids=list(tpl["identity"]["validator_profile_ids"]),
        provider_profile_id=tident["provider_profile_id"],
        egress_policy_id=tpl["egress_policy_id"],
        operator_id="op-fence-test",
        operator_receipt_digest="0" * 64,
        budget=Budget(
            schema_version=b["schema_version"],
            max_chunks=b["max_chunks"],
            max_model_calls=b["max_model_calls"],
            max_tool_calls=b["max_tool_calls"],
            max_local_repairs=b["max_local_repairs"],
            max_rechunks=b["max_rechunks"],
            max_active_seconds=b["max_active_seconds"],
            max_wall_seconds=b["max_wall_seconds"],
            max_cost_microusd=b["max_cost_microusd"],
            context_token_budget=b["context_token_budget"],
        ),
    )
    grant = grant.model_copy(
        update={"approved_plan_digest": plan_digest, "operator_receipt_digest": "0" * 64}
    )
    register_protected_approval(
        db,
        approval_id="ap-fence-test",
        operation="activate_grant",
        grant_digest_target=content_sha256(grant),
        operator_id="op-fence-test",
        operator_receipt={"x": 1},
    )
    activate_grant(db, grant=grant, operator_id="op-fence-test", approval_id="ap-fence-test")
    return grant, plan_digest


def _build_worktree_and_campaign(db, *, grant, plan_id, grant_id, worktree):
    camp = create_campaign(
        db,
        plan_id=plan_id,
        grant_id=grant_id,
        base_commit="0" * 40,
        base_tree_digest="0" * 64,
        repo_path_text="local",
        repo_root=str(worktree),
        worktree_path=str(worktree),
    )
    activate_campaign(db, campaign_id=camp.campaign_id)
    return camp


def _build_broker(worktree, *, tpl=LOCK_TEMPLATE):
    return Broker(
        repo_root=Path(str(worktree)),
        allowed_write_paths=list(tpl["allowed_write_paths"]),
        allowed_create_paths=list(tpl["allowed_write_paths"]),
        allowed_read_paths=list(tpl["repository_paths"]),
        allowed_protected_read_paths=list(tpl["protected_paths"]),
        model_allowed_command_ids=list(tpl["allowed_operations"]),
        required_validator_ids=list(tpl["identity"]["validator_profile_ids"]),
        max_tool_result_bytes=10000000,
    )


def _propose_create(broker):
    tool_call = ToolCall(
        call_id="prop-fence-authority",
        args=CreateFileArgs(
            path="scripts/test_fence_authority.py", new_content="# fence-test\n"
        ),
    )
    return broker.handle(tool_call)


def _admit(db, *, grant, campaign_id, tpl=LOCK_TEMPLATE, worker_id="wkr-fence",
            chunk_id="OV-01-FRC-001-C01"):
    snap = RepoSnapshot(
        schema_version="trio.repo-snapshot.v1",
        repository_id="local",
        commit="0" * 40,
        tree_digest="0" * 64,
    )
    chunk = ChunkSpec(
        schema_version="trio.chunk.v1",
        chunk_id=chunk_id,
        campaign_id=campaign_id,
        package_id=chunk_id,
        parent_chunk_id=None,
        revision=1,
        title="fence",
        objective="fence test",
        permitted_read_paths=[],
        permitted_write_paths=["scripts/test_fence_authority.py"],
        permitted_command_ids=["noop"],
        permitted_validator_ids=["py_compile"],
        required_validator_ids=["py_compile"],
        required_receipt_profiles=[],
        criterion_ids=["OV-01-FRC-001-C01.inventories-workspace-fingerprint"],
        idempotency_key=f"k-{int(time.time() * 1e6)}-{chunk_id}",
    )
    return derive_admission(
        db,
        grant=grant,
        chunk=chunk,
        current_runtime_digest=tpl["runtime_identity"]["runtime_digest"],
        worker_id=worker_id,
        policy_profile_id=tpl["identity"]["policy_profile_id"],
        validator_profile_ids=tpl["identity"]["validator_profile_ids"],
        provider_profile_id=tpl["runtime_identity"]["provider_profile_id"],
        current_model_name=tpl["runtime_identity"]["model_name"],
        current_model_digest=tpl["runtime_identity"]["model_digest"],
        current_policy_profile_id=tpl["identity"]["policy_profile_id"],
        current_validator_profile_ids=tpl["identity"]["validator_profile_ids"],
        current_provider_profile_id=tpl["runtime_identity"]["provider_profile_id"],
        current_accepted_snapshot=snap,
    )


def _lease_row(db, lease_id):
    cur = db._conn.execute(
        "SELECT fence_generation, released_at, owner_id, owner_pid, "
        "owner_boot_id, owner_start_time, expires_at "
        "FROM leases WHERE lease_id=?",
        (lease_id,),
    )
    return cur.fetchone()


def _identity(pid):
    from overnight_runner.resources import process_identity
    ident = process_identity(pid)
    return os.getpid(), ident["boot_id"], ident["start_time"]


# -- The matrix ------------------------------------------------------------


def test_same_writer_apply_pass_and_fence_stable(tmp_path):
    """§10 Same writer: admission → propose → apply PASS, fence remains valid.

    No heartbeat. No takeover. Fence stays at 1.
    """
    db = Database(tmp_path / "state.db")
    grant, _ = _build_plan_and_grant(db, plan_id="OV-01-FRC-001", grant_id="OV-01-FRC-001")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    camp = _build_worktree_and_campaign(
        db, grant=grant, plan_id="OV-01-FRC-001", grant_id="OV-01-FRC-001",
        worktree=worktree,
    )
    broker = _build_broker(worktree)

    receipt, _ = _admit(db, grant=grant, campaign_id=camp.campaign_id)
    arow = db._conn.execute(
        "SELECT fence_generation, lease_id FROM admissions WHERE admission_id=?",
        (receipt.admission_id,),
    ).fetchone()
    admission_fence = int(arow["fence_generation"])
    lease_id = arow["lease_id"]

    propose = _propose_create(broker)
    proposal_id = propose["proposal_id"]

    pid_now, boot_id_now, start_time_now = _identity(os.getpid())
    result = apply_campaign_patch(
        db, broker, str(worktree), camp.campaign_id, proposal_id,
        admission_fence_generation=admission_fence,
        lease_id=lease_id, owner_id="wkr-fence",
        owner_pid=pid_now, owner_start_time=start_time_now,
    )
    assert result.get("applied") is True
    assert current_fence(db, camp.campaign_id).current_generation == 1


def test_heartbeat_preserves_fence_and_still_passes_apply(tmp_path):
    """§10 Heartbeat: admission fence N → heartbeat → apply with N PASS.

    Heartbeat MUST NOT bump the fence; same writer keeps authority.
    """
    db = Database(tmp_path / "state.db")
    grant, _ = _build_plan_and_grant(db, plan_id="OV-01-FRC-001", grant_id="OV-01-FRC-001")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    camp = _build_worktree_and_campaign(
        db, grant=grant, plan_id="OV-01-FRC-001", grant_id="OV-01-FRC-001",
        worktree=worktree,
    )
    broker = _build_broker(worktree)

    receipt, _ = _admit(db, grant=grant, campaign_id=camp.campaign_id)
    arow = db._conn.execute(
        "SELECT fence_generation, lease_id FROM admissions WHERE admission_id=?",
        (receipt.admission_id,),
    ).fetchone()
    admission_fence = int(arow["fence_generation"])
    lease_id = arow["lease_id"]
    lease_row = _lease_row(db, lease_id)

    pid_now, boot_id_now, start_time_now = _identity(os.getpid())
    hb = heartbeat_admission_lease(
        db,
        admission_id=receipt.admission_id,
        owner_id=lease_row["owner_id"],
        owner_pid=pid_now,
        owner_start_time=start_time_now,
        owner_boot_id=boot_id_now,
        extend_seconds=300,
        reason="fence-test-heartbeat",
    )
    assert hb["fence_generation"] == admission_fence
    assert current_fence(db, camp.campaign_id).current_generation == admission_fence

    propose = _propose_create(broker)
    proposal_id = propose["proposal_id"]

    result = apply_campaign_patch(
        db, broker, str(worktree), camp.campaign_id, proposal_id,
        admission_fence_generation=admission_fence,
        lease_id=lease_id, owner_id=lease_row["owner_id"],
        owner_pid=pid_now, owner_start_time=start_time_now,
    )
    assert result.get("applied") is True


def test_real_takeover_bumps_fence_and_old_holder_apply_fails(tmp_path):
    """§10 Real takeover: N+1 fence after takeover; old apply with N MUST FAIL."""
    db = Database(tmp_path / "state.db")
    grant, _ = _build_plan_and_grant(db, plan_id="OV-01-FRC-001", grant_id="OV-01-FRC-001")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    camp = _build_worktree_and_campaign(
        db, grant=grant, plan_id="OV-01-FRC-001", grant_id="OV-01-FRC-001",
        worktree=worktree,
    )
    broker = _build_broker(worktree)

    receipt, _ = _admit(db, grant=grant, campaign_id=camp.campaign_id, worker_id="wkr-A")
    arow = db._conn.execute(
        "SELECT fence_generation, lease_id FROM admissions WHERE admission_id=?",
        (receipt.admission_id,),
    ).fetchone()
    admission_fence_N = int(arow["fence_generation"])
    lease_id_A = arow["lease_id"]
    propose = _propose_create(broker)
    proposal_id_A = propose["proposal_id"]
    pid_A, _, start_A = _identity(os.getpid())

    # Real takeover (only legitimate source of +1).
    new_gen = revoke_for_takeover(db, campaign_id=camp.campaign_id, reason="fence-test-takeover")
    assert new_gen == admission_fence_N + 1
    assert current_fence(db, camp.campaign_id).current_generation == new_gen

    # Old holder applying with N must FAIL fence_stale.
    with pytest.raises(SafetyError) as ei:
        apply_campaign_patch(
            db, broker, str(worktree), camp.campaign_id, proposal_id_A,
            admission_fence_generation=admission_fence_N,
            lease_id=lease_id_A, owner_id="wkr-A",
            owner_pid=pid_A, owner_start_time=start_A,
        )
    assert "fence_stale" in str(ei.value).lower()


def test_new_owner_after_takeover_with_correct_fence_passes(tmp_path):
    """§10 New owner legitimately owns N+1 → apply PASS.

    Strategy: take over the campaign, then admit a fresh B chunk whose
    plan criterion is registered. Same approach as the rest of the matrix
    but driven off a fresh chunk_id so plan_criteria_for_package accepts it.
    """
    db = Database(tmp_path / "state.db")
    grant, _ = _build_plan_and_grant(
        db, plan_id="OV-01-FRC-001", grant_id="OV-01-FRC-001",
        extra_package_ids=["OV-01-FRC-001-C01B"],
    )
    worktree = tmp_path / "wt"
    worktree.mkdir()
    camp = _build_worktree_and_campaign(
        db, grant=grant, plan_id="OV-01-FRC-001", grant_id="OV-01-FRC-001",
        worktree=worktree,
    )
    broker = _build_broker(worktree)

    # First admission of C01 (will be revoked).
    receipt_A, _ = _admit(db, grant=grant, campaign_id=camp.campaign_id, worker_id="wkr-A")
    arow = db._conn.execute(
        "SELECT fence_generation, lease_id FROM admissions WHERE admission_id=?",
        (receipt_A.admission_id,),
    ).fetchone()
    assert int(arow["fence_generation"]) == 1
    new_gen = revoke_for_takeover(
        db, campaign_id=camp.campaign_id, reason="fence-test-takeover-2"
    )
    new_fence = new_gen

    # Owner B re-enters via a fresh admission at the new fence, on a
    # different chunk so the strict plan lookup does not refuse.
    receipt_B, _ = _admit(
        db, grant=grant, campaign_id=camp.campaign_id, worker_id="wkr-B",
        chunk_id="OV-01-FRC-001-C01B",
    )
    brow = db._conn.execute(
        "SELECT fence_generation, lease_id FROM admissions WHERE admission_id=?",
        (receipt_B.admission_id,),
    ).fetchone()
    assert int(brow["fence_generation"]) == new_fence
    new_lease = brow["lease_id"]
    propose = _propose_create(broker)
    proposal_id = propose["proposal_id"]
    pid_B, _, start_B = _identity(os.getpid())

    result = apply_campaign_patch(
        db, broker, str(worktree), camp.campaign_id, proposal_id,
        admission_fence_generation=new_fence,
        lease_id=new_lease, owner_id="wkr-B",
        owner_pid=pid_B, owner_start_time=start_B,
    )
    assert result.get("applied") is True


def test_correct_generation_with_wrong_lease_is_rejected(tmp_path):
    """§10 Wrong lease: correct fence but non-owning lease MUST FAIL."""
    db = Database(tmp_path / "state.db")
    grant, _ = _build_plan_and_grant(db, plan_id="OV-01-FRC-001", grant_id="OV-01-FRC-001")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    camp = _build_worktree_and_campaign(
        db, grant=grant, plan_id="OV-01-FRC-001", grant_id="OV-01-FRC-001",
        worktree=worktree,
    )
    broker = _build_broker(worktree)

    receipt, _ = _admit(db, grant=grant, campaign_id=camp.campaign_id)
    arow = db._conn.execute(
        "SELECT fence_generation, lease_id FROM admissions WHERE admission_id=?",
        (receipt.admission_id,),
    ).fetchone()
    admission_fence = int(arow["fence_generation"])
    propose = _propose_create(broker)
    proposal_id = propose["proposal_id"]
    pid, _, start = _identity(os.getpid())

    # A different (non-existent) lease id with the SAME fence must FAIL.
    with pytest.raises(SafetyError):
        apply_campaign_patch(
            db, broker, str(worktree), camp.campaign_id, proposal_id,
            admission_fence_generation=admission_fence,
            lease_id="lse-doesnotexist", owner_id="wkr-fence",
            owner_pid=pid, owner_start_time=start,
        )


def test_correct_lease_with_wrong_pid_is_rejected(tmp_path):
    """§10 Wrong process: right lease/fence but wrong pid MUST FAIL.

    Lease enforcement binds PID + boot id + start-time together.
    """
    db = Database(tmp_path / "state.db")
    grant, _ = _build_plan_and_grant(db, plan_id="OV-01-FRC-001", grant_id="OV-01-FRC-001")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    camp = _build_worktree_and_campaign(
        db, grant=grant, plan_id="OV-01-FRC-001", grant_id="OV-01-FRC-001",
        worktree=worktree,
    )
    broker = _build_broker(worktree)

    receipt, _ = _admit(db, grant=grant, campaign_id=camp.campaign_id)
    arow = db._conn.execute(
        "SELECT fence_generation, lease_id FROM admissions WHERE admission_id=?",
        (receipt.admission_id,),
    ).fetchone()
    admission_fence = int(arow["fence_generation"])
    lease_id = arow["lease_id"]
    propose = _propose_create(broker)
    proposal_id = propose["proposal_id"]
    _, _, start = _identity(os.getpid())

    with pytest.raises(SafetyError) as ei:
        apply_campaign_patch(
            db, broker, str(worktree), camp.campaign_id, proposal_id,
            admission_fence_generation=admission_fence,
            lease_id=lease_id, owner_id="wkr-fence",
            owner_pid=os.getpid() + 1_000_000,  # wrong pid
            owner_start_time=start,
        )
    assert "lease_owner_identity_mismatch" in str(ei.value).lower()


def test_capacity_resume_uses_canonic_mint_path(tmp_path):
    """§10 Restart/resume reuses existing canonical capacity-resume path
    (refresh_admission_lease_for_resume). Confirms the P08 resume mints
    at the SAME fence the campaign has (no bumper)."""
    db = Database(tmp_path / "state.db")
    grant, _ = _build_plan_and_grant(db, plan_id="OV-01-FRC-001", grant_id="OV-01-FRC-001")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    camp = _build_worktree_and_campaign(
        db, grant=grant, plan_id="OV-01-FRC-001", grant_id="OV-01-FRC-001",
        worktree=worktree,
    )

    receipt, _ = _admit(db, grant=grant, campaign_id=camp.campaign_id)
    arow = db._conn.execute(
        "SELECT fence_generation, lease_id FROM admissions WHERE admission_id=?",
        (receipt.admission_id,),
    ).fetchone()
    prior_fence = int(arow["fence_generation"])
    prior_lease = arow["lease_id"]

    # Make the original lease released so resume is permitted (live leases
    # are never replaced per the documented semantics).
    db._conn.execute(
        "UPDATE leases SET released_at=? WHERE lease_id=?",
        (int(time.time()), prior_lease),
    )
    assert current_fence(db, camp.campaign_id).current_generation == 1

    # The refresh path requires a P07 wake claim. Synthesize one for the
    # campaign so the resume API does not refuse for missing lineage.
    db._conn.execute(
        "INSERT INTO wake_claims "
        "(claim_id, campaign_id, state, claimed_at, obligation_id, generation) "
        "VALUES (?, ?, 'CLAIMED', ?, ?, ?)",
        (f"wc-fence-{int(time.time()*1e6)}", camp.campaign_id,
         int(time.time()),
         f"{camp.campaign_id}:commit:1",
         1),
    )
    wcrow = db._conn.execute(
        "SELECT claim_id FROM wake_claims WHERE campaign_id=? ORDER BY claimed_at DESC LIMIT 1",
        (camp.campaign_id,),
    ).fetchone()
    wake_claim_id = wcrow["claim_id"]

    from overnight_runner.admission_lease import refresh_admission_lease_for_resume
    pid_now, _, start_now = _identity(os.getpid())
    rb = refresh_admission_lease_for_resume(
        db,
        admission_id=receipt.admission_id,
        wake_claim_id=wake_claim_id,
        reason="fence-test-resume",
        owner_id="wkr-fence",
        owner_pid=pid_now,
        ttl_seconds=300,
        owner_start_time=start_now,
    )
    assert int(rb["fence_generation"]) == prior_fence
    assert current_fence(db, camp.campaign_id).current_generation == prior_fence
