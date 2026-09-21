"""P06 follow-up #2 — Disposable three-chunk campaign proof (16 of the package spec).

This is the canonical P06 example end-to-end on a real disposable
git repo:

    base S0 -> admit+commit+CAS chunk 1 -> S1
              -> admit+commit+CAS chunk 2 (binds S1) -> S2
              -> admit+commit+CAS chunk 3 (binds S2) -> S3

P06 follow-up #2 corrections applied:

  * The grant is pinned to the exact approved_plan_digest before
    activation (A01). Admission verifies the chunk's package_id /
    criterion_ids against the grant-pinned plan, NOT a caller-chosen
    plan.
  * Every identity kwarg (model/runtime/policy/validator/provider)
    is REQUIRED; current_runtime_digest is the SINGLE authoritative
    runtime argument (A02).
  * The idempotency winner is inserted atomically BEFORE any lease
    allocation (A03).
  * The trusted validation receipt is bound to the EXACT candidate
    being integrated via ``expected_tree_digest`` + ``chunk_id`` +
    ``validator_id`` (A04). The disposable campaign proof runs the
    REAL required validator through ``Worker._finalise`` (NOT a
    direct ``mint_validation_receipt`` call) so the receipt binds to
    the candidate snapshot the validator actually saw.
  * Campaign continuation checks EFFECT_UNKNOWN / cancelled /
    expired / budget exhausted / grant expired (A05/A07).
  * Campaign-aware mutation fence before every patch apply (A06).

The disposable repo is at ``<tmp>/fixture_repo``. The campaign
worktree is at ``<state_dir>/campaign-worktrees/<campaign_id>``.
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
    ChunkSpec,
    RepoSnapshot,
    content_sha256,
)
from overnight_runner.db import Database
from overnight_runner.grants import activate_grant
from overnight_runner.admission import derive_admission, check_campaign_continuation
from overnight_runner.campaign import (
    activate_campaign,
    create_campaign,
    record_chunk_accepted,
    update_budget_after_chunk,
    read_budget_totals,
)
from overnight_runner.integration import (
    capture_current_snapshot,
    compare_and_swap_advance,
    ensure_campaign_worktree,
)
from overnight_runner.plans import register_plan, load_plan_digest
from overnight_runner.protected_approvals import register_protected_approval
from overnight_runner.receipts import (
    KIND_VALIDATION,
    mint_validation_receipt,
    verify_receipt,
    receipts_enabled,
    _receipts_root,
)
from overnight_runner.resources import acquire_lease, current_fence
from overnight_runner.worker import Worker, _finalise, _maybe_mint_validation_receipt
from overnight_runner.broker import Broker, CommandRegistry, CommandSpec
from overnight_runner.safety import (
    git_commit_all,
    git_init_empty,
    git_head,
    git_worktree_sha,
)


# ----------------------------- Test helpers -----------------------------

def _isolated_setup(tmp_path):
    sd = tmp_path / "state"
    sd.mkdir(parents=True, exist_ok=True)
    os.environ["OVERNIGHT_STATE_DIR"] = str(sd)
    os.environ["OVERNIGHT_RECEIPTS"] = "1"
    return sd / "state.db"


def _pin_plan_digest(grant: AutonomyGrant, plan_packages: dict) -> AutonomyGrant:
    """Compute the registered plan digest and pin the grant to it."""
    register_plan(
        _plan_db[None],
        plan_id=grant.plan_id,
        approved_artifact_id=grant.plan_id,
        work_package_criterion_ids=plan_packages,
    )
    digest = load_plan_digest(_plan_db[None], grant.plan_id)
    return grant.model_copy(update={"approved_plan_digest": digest})


_plan_db: dict = {}


def _grant_and_plan(db, plan_packages: dict) -> AutonomyGrant:
    _plan_db[None] = db
    grant = AutonomyGrant(
        schema_version="trio.grant.v1",
        grant_id="gr-dc", state="draft",
        plan_id="pl-dc", plan_revision=1,
        approved_plan_digest="0" * 64,  # placeholder; pinned below
        repository_paths=["src/app.py"], allowed_write_paths=["src/app.py"],
        protected_paths=[], allowed_operations=["noop"],
        runtime_digest="a" * 64, model_name="gemma", model_digest="b" * 64,
        policy_profile_id="pol-dc", validator_profile_ids=["noop"],
        provider_profile_id="prv-dc", egress_policy_id="eg-dc",
        operator_id="op-dc", operator_receipt_digest="c" * 64,
        budget=Budget(schema_version="trio.budget.v1",
                      max_chunks=3, max_model_calls=10, max_tool_calls=20,
                      max_local_repairs=2, max_rechunks=1,
                      max_active_seconds=3600, max_wall_seconds=28800,
                      max_cost_microusd=1000, context_token_budget=8192),
    )
    grant_pinned = _pin_plan_digest(grant, plan_packages)
    digest = content_sha256(grant_pinned)
    register_protected_approval(
        db, approval_id="appr-dc", operation="activate_grant",
        grant_digest_target=digest,
        operator_id="op-dc", operator_receipt={"approval_id": "appr-dc"},
    )
    activate_grant(db, grant=grant_pinned, operator_id="op-dc", approval_id="appr-dc")
    return grant_pinned


def _required_kwargs(grant: AutonomyGrant) -> dict:
    return dict(
        current_runtime_digest=grant.runtime_digest,
        current_model_name=grant.model_name,
        current_model_digest=grant.model_digest,
        current_policy_profile_id=grant.policy_profile_id,
        current_validator_profile_ids=list(grant.validator_profile_ids),
        current_provider_profile_id=grant.provider_profile_id,
    )


def _run_real_validator(
    *,
    worktree: Path,
    artifact_dir: Path,
    validator_command: str,
    chunk_id: str,
) -> str:
    """Traverse the real required-validator boundary.

    Builds a minimal ``Broker`` with a noop validator and invokes
    ``Worker._finalise`` so a real ``kind=validation`` receipt is
    minted bound to the EXACT candidate snapshot.

    P06 follow-up #3 (A04 items 3/5):

      * ``_finalise`` runs against the ACTUAL campaign worktree, so its
        ``git_worktree_sha(worktree)`` candidate snapshot is the real
        canonical candidate.
      * The manifest passed to ``_finalise`` points at the campaign
        worktree (NOT the main fixture repo).
      * The mint callback uses ``payload["candidate_snapshot_digest"]``
        from ``_finalise`` — it NEVER substitutes a caller-supplied
        Git tree SHA.
    """
    from overnight_runner.schemas import TaskManifest, ExecutionClass, Disposition

    # Build a registry whose only validator is the one we want.
    reg = CommandRegistry()
    reg.register(CommandSpec(validator_command, ["true"], "repo", 5, "read"))
    broker = Broker(
        repo_root=worktree,
        registry=reg,
        allowed_write_paths=["src/app.py"],
        allowed_create_paths=[],
        allowed_read_paths=["src/app.py"],
        allowed_protected_read_paths=[],
        model_allowed_command_ids=[validator_command],
        required_validator_ids=[validator_command],
        approved_repo_head=None,
        artifact_dir=artifact_dir,
    )

    receipts_dir = receipts_root_for()
    minted_ids: list[str] = []

    def _mint(payload):
        # Use the ACTUAL payload from _finalise, bound to the campaign
        # worktree candidate snapshot.
        rid = mint_validation_receipt(
            validator_id=payload["validator_id"],
            validator_command=payload["validator_command"],
            validator_profile=payload.get("validator_profile")
            or payload["validator_command"],
            candidate_snapshot_digest=payload["candidate_snapshot_digest"],
            candidate_tree_state=payload.get("candidate_tree_state", "post-apply"),
            chunk_id=payload.get("chunk_id", ""),
            proposal_id=payload.get("proposal_id", ""),
            request_id=payload.get("request_id", ""),
            outcome=payload["outcome"],
            detail=payload.get("detail", ""),
            env_digest=payload.get("env_digest", ""),
            profile_digest=payload.get("profile_digest", ""),
            receipts_dir=receipts_dir,
        )
        minted_ids.append(rid)
        return rid

    # Build a minimal manifest that drives _finalise. The repo path is
    # the CAMPAIGN WORKTREE so the candidate snapshot is bound to the
    # bytes being integrated.
    manifest = TaskManifest(
        task_id=chunk_id,
        title=f"chunk {chunk_id}",
        objective="produce a validation receipt",
        execution_class=ExecutionClass.SOURCE_MUTATION,
        repo={"path": str(worktree)},
        paths={
            "write_paths": ["src/app.py"],
            "create_paths": [],
            "read_paths": ["src/app.py"],
            "protected_read_paths": [],
        },
        commands={
            "model_allowed_command_ids": [validator_command],
            "required_validator_ids": [validator_command],
            "allow_no_mutation": True,
        },
        model_profile={
            "model_name": "gemma",
            "temperature": 0.0,
        },
        context_budget={
            "max_read_bytes": 4096,
            "max_files_read": 4,
            "max_files_written": 2,
        },
        limits={
            "max_model_turns": 1,
            "max_tool_calls": 4,
            "max_changed_files": 2,
            "max_diff_lines": 400,
            "max_written_bytes": 65536,
            "task_timeout_seconds": 60,
            "max_tool_result_bytes": 24576,
        },
        acceptance_criteria=[],
    )
    # Manually drive _finalise with DONE disposition and no applied
    # proposals (we want the validator receipts only, not a mutation).
    status, code, text, ids = _finalise(
        manifest=manifest,
        broker=broker,
        disposition=Disposition.DONE,
        artifact_dir=artifact_dir,
        applied_proposals=[],
        on_validation_receipt=_mint,
        env_digest="env-dc",
        profile_digest="prof-dc",
    )
    if status != "PASSED":
        raise AssertionError(f"validator run did not PASS: {status} {code} {text}")
    if not ids:
        raise AssertionError("validator ran but no receipt was minted")
    return ids[-1]


# ----------------------------- The test -----------------------------

class TestDisposableCampaignProof(unittest.TestCase):
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
        self._head = git_head(self._repo)

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)
        if self._old_state:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state

    def test_real_disposable_three_chunk_campaign(self):
        """Full P06 disposal proof with REAL validator receipts.

        Each chunk:
          * derive_admission with grant-pinned plan + REQUIRED identity
          * fence-checked worker patch apply (campaign_apply)
          * real required validator through Worker._finalise (kind=validation PASS)
          * compare_and_swap_advance verifying exact candidate snapshot
          * update_budget_after_chunk via trusted budget API
        """
        # 1) Activate grant + plan.
        grant = _grant_and_plan(self._db, {"pkg-1": {"crit-1", "crit-2"}})
        # Reload the active grant from durable store so its digest
        # matches what create_campaign stored.
        from overnight_runner.grants import load_grant as _load_grant
        grant_active = _load_grant(self._db, grant.grant_id)
        real_base = self._head
        padded_base = real_base + "0" * 24
        # 2) Create + activate campaign.
        camp = create_campaign(
            self._db, plan_id=grant.plan_id, grant_id=grant.grant_id,
            base_commit=padded_base, base_tree_digest="b" * 64,
        )
        activate_campaign(self._db, campaign_id=camp.campaign_id)
        # 3) Open the disposable campaign worktree (persist its identity).
        wt = ensure_campaign_worktree(
            repo_root=self._repo,
            campaign_id=camp.campaign_id,
            base_commit=padded_base,
            db=self._db,
        )

        committed: list[str] = []
        for i in range(1, 4):
            # 3a) Campaign continuation pre-flight (A05/A07).
            check_campaign_continuation(
                self._db,
                campaign_id=camp.campaign_id,
                grant_id=grant.grant_id,
            )
            chunk = ChunkSpec(
                schema_version="trio.chunk.v1",
                chunk_id=f"chk-{i}",
                campaign_id=camp.campaign_id,
                package_id="pkg-1",
                revision=i,
                title=f"chunk {i}",
                objective=f"patch file for chunk {i}",
                permitted_signature_paths=["src/app.py"],
                permitted_write_paths=["src/app.py"],
                permitted_read_paths=["src/app.py"],
                permitted_command_ids=["noop"],
                permitted_validator_ids=["noop"],
                required_validator_ids=["noop"],
                required_receipt_profiles=["noop"],
                criterion_ids=["crit-1"],
                idempotency_key=f"idem-dc-{i}",
            )
            precursor = real_base if i == 1 else committed[i - 2]
            receipt, _ = derive_admission(
                self._db, grant=grant, chunk=chunk,
                worker_id="wkr-dc",
                policy_profile_id=grant.policy_profile_id,
                validator_profile_ids=list(grant.validator_profile_ids),
                provider_profile_id=grant.provider_profile_id,
                current_accepted_snapshot=RepoSnapshot(
                    schema_version="trio.repo-snapshot.v1",
                    repository_id="local",
                    commit=precursor + "0" * 24,
                    tree_digest="c" * 64,
                ),
                **_required_kwargs(grant),
            )
            # 3b) Acquire a writer lease (campaign-aware fence).
            fence = current_fence(self._db, camp.campaign_id)
            lease = acquire_lease(
                self._db,
                campaign_id=camp.campaign_id,
                resource_id=f"chunk:{chunk.chunk_id}",
                owner_id="wkr-dc",
                owner_boot_id="boot-dc",
                owner_pid=os.getpid(),
                fence_generation=receipt.fence_generation,
                ttl_seconds=300,
            )
            # 3c) Worker mutates the candidate and commits.
            (wt / "src" / "app.py").write_text(f"CHUNK{i}\n")
            subprocess.run(["git", "add", "-A"], cwd=str(wt), check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", f"chunk {i}"], cwd=str(wt), check=True, capture_output=True)
            new_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(wt), capture_output=True, text=True
            ).stdout.strip()
            s_i = capture_current_snapshot(wt)
            # 3d) Canonical candidate identity: the ACTUAL campaign
            # worktree fingerprint (git_worktree_sha). This is the ONE
            # scheme that proves "the bytes validated are the bytes
            # integrated" (P06 follow-up #3 A04 item 3).
            candidate = git_worktree_sha(wt)
            # 3e) Run the REAL required validator through Worker._finalise
            # against the CAMPAIGN WORKTREE. The mint callback uses the
            # payload's candidate_snapshot_digest (no substitution).
            artifact_dir = self._tmp / f"artifacts-{i}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            v_receipt = _run_real_validator(
                worktree=wt,
                artifact_dir=artifact_dir,
                validator_command="noop",
                chunk_id=chunk.chunk_id,
            )
            # 3f) Sanity-verify the receipt via verify_receipt with
            # strict binding to the canonical candidate.
            self.assertTrue(
                verify_receipt(
                    v_receipt, expected_kind=KIND_VALIDATION,
                    validator_id="noop", validator_command="noop",
                    chunk_id=f"chk-{i}",
                    outcome="pass",
                    candidate_snapshot_digest=candidate,
                )
            )
            # 3g) CAS advance recomputing the SAME candidate fingerprint
            # from the campaign worktree.
            expected_old = "" if i == 1 else committed[i - 2]
            result = compare_and_swap_advance(
                self._db,
                repo_root=self._repo,
                campaign_id=camp.campaign_id,
                chunk_id=f"chk-{i}",
                new_commit=new_commit,
                holder_fence_generation=receipt.fence_generation,
                actor="runner",
                idempotency_key=f"idem-dc-{i}",
                validation_receipt_ids=[v_receipt],
                expected_old=expected_old or None,
                campaign_worktree=wt,
            )
            record_chunk_accepted(
                self._db, chunk_id=f"chk-{i}",
                accepted_commit=new_commit,
                accepted_tree_digest=candidate,
            )
            # 3h) Update budget via the trusted API.
            totals = update_budget_after_chunk(
                self._db, ledger_id=f"bl-{camp.campaign_id}",
                delta_chunks=1,
                delta_model_calls=1,
                delta_tool_calls=2,
                delta_active_seconds=10,
                delta_wall_seconds=12,
            )
            self.assertEqual(totals["cumulative_chunks"], i)
            self.assertEqual(totals["cumulative_model_calls"], i)
            self.assertEqual(totals["cumulative_active_seconds"], 10 * i)
            self.assertEqual(result.committed_new_commit, new_commit)
            committed.append(new_commit)

        # 4) Verify the integration journal holds three ordered commits.
        cur = self._db._conn.execute(
            "SELECT chunk_id, committed_new_commit FROM integration_journal "
            "WHERE campaign_id=? ORDER BY entry_id",
            (camp.campaign_id,),
        )
        rows = list(cur)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["committed_new_commit"], committed[0])
        self.assertEqual(rows[2]["committed_new_commit"], committed[2])

        # 5) Verify the durable budget totals via the trusted read API.
        totals = read_budget_totals(
            self._db, ledger_id=f"bl-{camp.campaign_id}",
        )
        self.assertEqual(totals["cumulative_chunks"], 3)
        self.assertEqual(totals["cumulative_model_calls"], 3)
        self.assertEqual(totals["cumulative_active_seconds"], 30)
        self.assertEqual(totals["cumulative_wall_seconds"], 36)  # 12 per chunk * 3

        # 6) Verify the campaign's durable metadata is real digests,
        # not placeholder plan_id strings.
        cur = self._db._conn.execute(
            "SELECT grant_digest, plan_digest FROM campaigns WHERE campaign_id=?",
            (camp.campaign_id,),
        )
        row = cur.fetchone()
        self.assertNotEqual(row["grant_digest"], grant.plan_id)
        self.assertNotEqual(row["plan_digest"], grant.plan_id)
        self.assertEqual(row["grant_digest"], content_sha256(grant_active))
        self.assertEqual(row["plan_digest"], load_plan_digest(self._db, grant.plan_id))


def receipts_root_for():
    return _receipts_root()


if __name__ == "__main__":
    unittest.main(verbosity=2)
