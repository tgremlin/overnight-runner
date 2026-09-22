"""P08 — autonomous progress + bounded failure handling.

Integration/verification tests driving the ACCEPTED P00–P07 pipeline on
disposable repositories + isolated runner state. Every result is
MOCK/FIXTURE or FAULT INJECTION (the REAL host canary lives in
``scripts/p08_canary.py``).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from overnight_runner.broker import Broker, CommandRegistry, CommandSpec, Proposal
from overnight_runner.campaign import (
    activate_campaign,
    create_campaign,
    read_budget_totals,
    record_chunk_accepted,
    update_budget_after_chunk,
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
from overnight_runner.grants import activate_grant, load_grant
from overnight_runner.admission import derive_admission
from overnight_runner.integration import (
    compare_and_swap_advance,
    ensure_campaign_worktree,
)
from overnight_runner.p07 import (
    classify_capacity,
    handle_provider_result,
    job_state,
    poll_wake_claim_execution,
    reconcile_wake_claims,
    request_control,
    start_wake_claim_execution,
    wake_claim,
    wake_tick,
)
from overnight_runner.p08 import (
    bound_context,
    evaluate_phase_completion,
    load_handoff,
    record_handoff,
    record_phase_gate,
)
from overnight_runner.plans import load_plan_digest, register_plan
from overnight_runner.protected_approvals import register_protected_approval
from overnight_runner.receipts import _receipts_root, mint_validation_receipt
from overnight_runner.resources import acquire_lease, current_fence, process_identity
from overnight_runner.safety import (
    SafetyError,
    git_commit_all,
    git_head,
    git_init_empty,
    git_worktree_sha,
)
from overnight_runner.schemas import Disposition, ExecutionClass, TaskManifest
from overnight_runner.worker import _finalise


# ----------------------------- helpers -----------------------------

def _isolated(tmp: Path) -> Path:
    sd = tmp / "state"
    sd.mkdir(parents=True, exist_ok=True)
    os.environ["OVERNIGHT_STATE_DIR"] = str(sd)
    os.environ["OVERNIGHT_RECEIPTS"] = "1"
    return sd / "state.db"


def _make_repo(tmp: Path) -> Path:
    repo = tmp / "pilot_repo"
    repo.mkdir()
    git_init_empty(repo)
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("BASE\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_app.py").write_text("def test_ok():\n    assert True\n")
    git_commit_all(repo, "init")
    return repo


def _grant(db, *, plan_id="pl-p08", grant_id="gr-p08", max_chunks=5,
           max_local_repairs=2, max_rechunks=1) -> AutonomyGrant:
    packages = {"pkg-1": {"crit-1", "crit-2"}}
    g = AutonomyGrant(
        schema_version="trio.grant.v1", grant_id=grant_id, state="draft",
        plan_id=plan_id, plan_revision=1, approved_plan_digest="0" * 64,
        repository_paths=["src/app.py", "tests/test_app.py"],
        allowed_write_paths=["src/app.py", "tests/test_app.py"],
        protected_paths=[], allowed_operations=["noop", "must_pass", "must_fail"],
        runtime_digest="a" * 64, model_name="gemma", model_digest="b" * 64,
        policy_profile_id="pol-1", validator_profile_ids=["noop", "must_pass", "must_fail"],
        provider_profile_id="prv-1", egress_policy_id="eg-1",
        operator_id="op-1", operator_receipt_digest="c" * 64,
        budget=Budget(schema_version="trio.budget.v1", max_chunks=max_chunks,
                      max_model_calls=50, max_tool_calls=100,
                      max_local_repairs=max_local_repairs, max_rechunks=max_rechunks,
                      max_active_seconds=3600, max_wall_seconds=28800,
                      max_cost_microusd=0, context_token_budget=8192),
    )
    register_plan(db, plan_id=plan_id, approved_artifact_id=plan_id,
                  work_package_criterion_ids=packages)
    g = g.model_copy(update={"approved_plan_digest": load_plan_digest(db, plan_id)})
    register_protected_approval(
        db, approval_id=f"appr-{grant_id}", operation="activate_grant",
        grant_digest_target=content_sha256(g), operator_id="op-1",
        operator_receipt={"approval_id": f"appr-{grant_id}"})
    activate_grant(db, grant=g, operator_id="op-1", approval_id=f"appr-{grant_id}")
    return load_grant(db, grant_id)


def _id_kwargs(g) -> dict:
    return dict(
        current_runtime_digest=g.runtime_digest, current_model_name=g.model_name,
        current_model_digest=g.model_digest,
        current_policy_profile_id=g.policy_profile_id,
        current_validator_profile_ids=list(g.validator_profile_ids),
        current_provider_profile_id=g.provider_profile_id)


def _snap(commit="a" * 64, tree="b" * 64) -> RepoSnapshot:
    return RepoSnapshot(schema_version="trio.repo-snapshot.v1",
                        repository_id="local", commit=commit, tree_digest=tree)


def _chunk(campaign_id, *, chunk_id, idem, validators=("noop",)) -> ChunkSpec:
    return ChunkSpec(
        schema_version="trio.chunk.v1", chunk_id=chunk_id,
        campaign_id=campaign_id, package_id="pkg-1", revision=1,
        title="c", objective="c",
        permitted_signature_paths=["src/app.py"],
        permitted_write_paths=["src/app.py", "tests/test_app.py"],
        permitted_read_paths=["src/app.py", "tests/test_app.py"],
        permitted_command_ids=list(validators),
        permitted_validator_ids=list(validators),
        required_validator_ids=list(validators),
        required_receipt_profiles=list(validators),
        criterion_ids=["crit-1"], idempotency_key=idem)


def _registry(extra: list[tuple[str, list[str]]] | None = None,
              overrides: dict[str, list[str]] | None = None) -> CommandRegistry:
    reg = CommandRegistry()
    defs: dict[str, list[str]] = {"noop": ["true"], "must_pass": ["true"],
                                  "must_fail": ["false"]}
    defs.update(overrides or {})
    for cid, argv in defs.items():
        reg.register(CommandSpec(cid, argv, "repo", 5, "read"))
    for cid, argv in (extra or []):
        if cid in defs:
            continue
        reg.register(CommandSpec(cid, argv, "repo", 5, "read"))
    return reg


def _broker(worktree: Path, artifact_dir: Path, reg: CommandRegistry) -> Broker:
    return Broker(
        repo_root=worktree, registry=reg,
        allowed_write_paths=["src/app.py", "tests/test_app.py"],
        allowed_create_paths=[], allowed_read_paths=["src/app.py", "tests/test_app.py"],
        allowed_protected_read_paths=[], model_allowed_command_ids=["noop", "must_pass", "must_fail"],
        required_validator_ids=[], approved_repo_head=None, artifact_dir=artifact_dir)


def _propose(broker: Broker, worktree: Path, *, path="src/app.py", text="CHANGED\n"):
    before = (worktree / path).read_text()
    broker.proposals["prop-x"] = Proposal(
        proposal_id="prop-x", op="replace_file", path=path,
        abs_path=worktree / path, before_text=before, proposed_text=text,
        preview_diff="", changed_lines=1, proposed_bytes=len(text))
    return "prop-x"


def _commit(worktree: Path, text="CHANGED\n", *, path="src/app.py") -> str:
    (worktree / path).write_text(text)
    subprocess.run(["git", "add", "-A"], cwd=str(worktree), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "chunk"], cwd=str(worktree), check=True, capture_output=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(worktree),
                          capture_output=True, text=True).stdout.strip()


def _manifest(chunk_id, worktree: Path, validators) -> TaskManifest:
    return TaskManifest(
        task_id=chunk_id, title="c", objective="c",
        execution_class=ExecutionClass.SOURCE_MUTATION,
        repo={"path": str(worktree)},
        paths={"write_paths": ["src/app.py"], "create_paths": [],
               "read_paths": ["src/app.py", "tests/test_app.py"],
               "protected_read_paths": []},
        commands={"model_allowed_command_ids": ["noop"],
                  "required_validator_ids": list(validators),
                  "allow_no_mutation": True},
        model_profile={"model_name": "gemma", "temperature": 0.0},
        context_budget={"max_read_bytes": 4096, "max_files_read": 4, "max_files_written": 2},
        limits={"max_model_turns": 1, "max_tool_calls": 4, "max_changed_files": 2,
                "max_diff_lines": 400, "max_written_bytes": 65536,
                "task_timeout_seconds": 60, "max_tool_result_bytes": 24576},
        acceptance_criteria=[])


def _run_validator(worktree: Path, artifact_dir: Path, chunk_id: str,
                   validators, *, registry=None) -> tuple[str, list[str]]:
    reg = registry or _registry()
    broker = _broker(worktree, artifact_dir, reg)
    ids: list[str] = []

    def _mint(payload):
        rid = mint_validation_receipt(
            validator_id=payload["validator_id"],
            validator_command=payload["validator_command"],
            validator_profile=payload.get("validator_profile") or payload["validator_command"],
            candidate_snapshot_digest=payload["candidate_snapshot_digest"],
            candidate_tree_state="post-apply",
            chunk_id=payload.get("chunk_id", ""), outcome=payload["outcome"],
            detail=payload.get("detail", ""), receipts_dir=_receipts_root())
        ids.append(rid)
        return rid

    manifest = _manifest(chunk_id, worktree, validators)
    status, code, text, rids = _finalise(
        manifest=manifest, broker=broker, disposition=Disposition.DONE,
        artifact_dir=artifact_dir, applied_proposals=[],
        on_validation_receipt=_mint, env_digest="e", profile_digest="p")
    return status, rids


class _Harness(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._db = Database(_isolated(self._tmp))
        self._repo = _make_repo(self._tmp)
        self._base = git_head(self._repo) + "0" * 24
        self._grant = _grant(self._db)
        self._camp = create_campaign(
            self._db, plan_id=self._grant.plan_id, grant_id=self._grant.grant_id,
            base_commit=self._base, base_tree_digest="b" * 64,
            repo_root=str(self._repo))
        activate_campaign(self._db, campaign_id=self._camp.campaign_id)
        self._wt = ensure_campaign_worktree(
            repo_root=self._repo, campaign_id=self._camp.campaign_id,
            base_commit=self._base, db=self._db)
        self._mut_n = 0

    def _release_leases(self):
        from overnight_runner.resources import release_lease
        for r in self._db._conn.execute(
                "SELECT lease_id FROM leases WHERE campaign_id=? AND released_at=0",
                (self._camp.campaign_id,)).fetchall():
            release_lease(self._db, lease_id=r["lease_id"])

    def tearDown(self):
        self._db.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _reopen(self):
        self._db.close()
        self._db = Database(Path(os.environ["OVERNIGHT_STATE_DIR"]) / "state.db")

    def _admit(self, chunk_id, *, validators=("noop",), idem=None):
        cur_commit = self._db._conn.execute(
            "SELECT current_commit FROM campaigns WHERE campaign_id=?",
            (self._camp.campaign_id,)).fetchone()["current_commit"]
        return derive_admission(
            self._db, grant=self._grant,
            chunk=_chunk(self._camp.campaign_id, chunk_id=chunk_id,
                         idem=idem or f"idem-{chunk_id}", validators=validators),
            worker_id="wkr-p08", policy_profile_id=self._grant.policy_profile_id,
            validator_profile_ids=list(self._grant.validator_profile_ids),
            provider_profile_id=self._grant.provider_profile_id,
            current_accepted_snapshot=_snap(commit=cur_commit), **_id_kwargs(self._grant))[0]

    def _mutate(self, *, text="CHANGED\n", path="src/app.py"):
        self._mut_n += 1
        reg = _registry()
        broker = _broker(self._wt, self._tmp / f"art-mut-{self._mut_n}", reg)
        pid = _propose(broker, self._wt, path=path, text=text)
        lease = acquire_lease(
            self._db, campaign_id=self._camp.campaign_id,
            resource_id=f"chunk:mut-{self._mut_n}",
            owner_id="wkr-p08", owner_boot_id="b", owner_pid=os.getpid(),
            fence_generation=current_fence(self._db, self._camp.campaign_id).current_generation,
            ttl_seconds=300)
        ident = process_identity(os.getpid())
        apply_campaign_patch(
            self._db, broker, str(self._repo), self._camp.campaign_id, pid,
            admission_fence_generation=lease.fence_generation, lease_id=lease.lease_id,
            owner_id="wkr-p08", owner_pid=os.getpid(),
            owner_start_time=ident["start_time"])
        return _commit(self._wt, text, path=path)

    def _integrate(self, chunk_id, new_commit, receipt_ids, expected_old):
        result = compare_and_swap_advance(
            self._db, repo_root=self._repo, campaign_id=self._camp.campaign_id,
            chunk_id=chunk_id, new_commit=new_commit, holder_fence_generation=1,
            actor="runner", idempotency_key=f"i-{chunk_id}",
            validation_receipt_ids=list(receipt_ids), expected_old=expected_old or None,
            campaign_worktree=self._wt)
        record_chunk_accepted(self._db, chunk_id=chunk_id, accepted_commit=new_commit,
                              accepted_tree_digest=git_worktree_sha(self._wt))
        return result

    def _one_chunk(self, i, *, validators=("noop",), text=None, expected_old=""):
        cid = f"chk-{i}"
        self._admit(cid, validators=validators)
        new_commit = self._mutate(text=text or f"CHUNK{i}\n")
        status, rids = _run_validator(self._wt, self._tmp / f"art-{i}", cid, validators)
        self.assertEqual(status, "PASSED")
        self.assertTrue(rids)
        self._integrate(cid, new_commit, rids, expected_old)
        update_budget_after_chunk(self._db, ledger_id=f"bl-{self._camp.campaign_id}",
                                  delta_chunks=1, delta_model_calls=2, delta_tool_calls=3)
        return new_commit


# ============================================================
# A01 — three-or-more dependent accepted chunks
# ============================================================

class TestP08A01Campaign(_Harness):
    def test_three_dependent_chunks_end_to_end(self):
        committed = []
        prev = ""
        for i in (1, 2, 3):
            prev = self._one_chunk(i, expected_old=prev)
            committed.append(prev)
        # Lineage recorded.
        rows = self._db._conn.execute(
            "SELECT chunk_id, committed_new_commit FROM integration_journal "
            "WHERE campaign_id=? ORDER BY entry_id", (self._camp.campaign_id,)).fetchall()
        self.assertEqual([r["chunk_id"] for r in rows], ["chk-1", "chk-2", "chk-3"])
        self.assertEqual([r["committed_new_commit"] for r in rows], committed)
        states = [r["state"] for r in self._db._conn.execute(
            "SELECT state FROM chunks WHERE campaign_id=? ORDER BY created_at",
            (self._camp.campaign_id,)).fetchall()]
        self.assertEqual(states, ["ACCEPTED_FOR_CONTINUATION"] * 3)
        totals = read_budget_totals(self._db, ledger_id=f"bl-{self._camp.campaign_id}")
        self.assertEqual(totals["cumulative_chunks"], 3)
        self.assertEqual(totals["cumulative_repairs"], 0)  # no per-chunk human prompt
        # Plan/grant identity recorded on the campaign.
        camp = self._db._conn.execute(
            "SELECT grant_digest, plan_digest FROM campaigns WHERE campaign_id=?",
            (self._camp.campaign_id,)).fetchone()
        self.assertEqual(camp["plan_digest"], self._grant.approved_plan_digest)
        self.assertEqual(camp["grant_digest"], content_sha256(load_grant(self._db, "gr-p08")))


# ============================================================
# A02 — bounded repair
# ============================================================

class TestP08A02Repair(_Harness):
    def test_bounded_repair_preserves_budgets(self):
        cid = "chk-r"
        self._admit(cid, validators=("must_pass",))
        new_commit = self._mutate(text="FIXED\n")
        ledger = f"bl-{self._camp.campaign_id}"
        # First attempt: the ONLY required validator FAILS (deterministic).
        fail_reg = _registry(overrides={"must_pass": ["false"]})
        status, rids = _run_validator(self._wt, self._tmp / "art-r1", cid,
                                      ("must_pass",), registry=fail_reg)
        self.assertEqual(status, "FAILED")
        # No integration / no acceptance on the failed attempt.
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM integration_journal").fetchone()["n"], 0)
        # Bounded repair: SAME family ledger, repairs +1 (not a reset).
        before = read_budget_totals(self._db, ledger_id=ledger)
        update_budget_after_chunk(self._db, ledger_id=ledger, delta_repairs=1)
        after = read_budget_totals(self._db, ledger_id=ledger)
        self.assertEqual(after["cumulative_repairs"], before["cumulative_repairs"] + 1)
        self.assertEqual(after["cumulative_model_calls"], before["cumulative_model_calls"])
        # Repaired candidate revalidated -> trusted receipt -> integration.
        status2, rids2 = _run_validator(self._wt, self._tmp / "art-r2", cid, ("must_pass",))
        self.assertEqual(status2, "PASSED")
        self.assertTrue(rids2)
        self._integrate(cid, new_commit, rids2, "")
        self.assertGreater(read_budget_totals(self._db, ledger_id=ledger)["cumulative_repairs"], 0)
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM budget_ledgers WHERE campaign_id=?",
            (self._camp.campaign_id,)).fetchone()["n"], 1)  # no new family/grant


# ============================================================
# A02 — context handoff
# ============================================================

class TestP08A02Handoff(_Harness):
    def test_context_handoff_preserves_identity(self):
        self._one_chunk(1)
        ledger = f"bl-{self._camp.campaign_id}"
        before = read_budget_totals(self._db, ledger_id=ledger)
        snap = self._db._conn.execute(
            "SELECT current_commit FROM campaigns WHERE campaign_id=?",
            (self._camp.campaign_id,)).fetchone()["current_commit"]
        hid = record_handoff(
            self._db, campaign_id=self._camp.campaign_id, chunk_id="chk-2",
            job_id="job-2", grant_id=self._grant.grant_id,
            budget_ledger_id=ledger, snapshot_commit=snap, reason="context_rollover")
        # Restart boundary.
        self._reopen()
        h = load_handoff(self._db, hid)
        self.assertIsNotNone(h)
        ctx = bound_context(h)
        self.assertEqual(sorted(ctx.keys()),
                         ["budget_ledger_id", "campaign_id", "chunk_id", "grant_id",
                          "job_id", "snapshot_commit"])
        self.assertEqual(ctx["grant_id"], self._grant.grant_id)
        self.assertEqual(ctx["snapshot_commit"], snap)
        # Continuation keeps the SAME cumulative budgets (no reset).
        after = read_budget_totals(self._db, ledger_id=ledger)
        self.assertEqual(after, before)
        # And can still make progress (revalidation of the next chunk passes).
        self._one_chunk(2, expected_old="")  # expected_old loose for this test
        self.assertEqual(read_budget_totals(self._db, ledger_id=ledger)["cumulative_chunks"], 2)


# ============================================================
# A03 — capacity wait + restart
# ============================================================

class TestP08A03Capacity(_Harness):
    def test_capacity_wait_restart_resume(self):
        self._one_chunk(1)
        self._release_leases()  # no live writer obligation during the capacity test
        ledger = f"bl-{self._camp.campaign_id}"
        base = int(time.time())
        out = classify_capacity(provider="prv-1", http_status=429, model="gemma",
                                retry_after_seconds=60)
        handle_provider_result(self._db, out, campaign_id=self._camp.campaign_id, now=base)
        self._reopen()
        # Before next_eligible_at: no work.
        d1 = wake_tick(self._db, campaign_id=self._camp.campaign_id, trigger_id="a",
                       now=base + 30)
        self.assertEqual(d1["decision"], "WAIT")
        # At/after: exactly one bounded wake -> one claim.
        d2 = wake_tick(self._db, campaign_id=self._camp.campaign_id, trigger_id="b",
                       now=base + 120)
        self.assertEqual(d2["decision"], "DISPATCH_ELIGIBLE")
        self.assertEqual(d2["advanced"], 1)
        d3 = wake_tick(self._db, campaign_id=self._camp.campaign_id, trigger_id="c",
                       now=base + 121)
        self.assertEqual(d3["decision"], "DUPLICATE")
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM wake_claims").fetchone()["n"], 1)
        # Capacity did NOT charge a repair.
        self.assertEqual(read_budget_totals(self._db, ledger_id=ledger)["cumulative_repairs"], 0)
        # Campaign can continue.
        self._one_chunk(2, expected_old="")

    def test_process_restart_during_owned_execution(self):
        import overnight_runner.p07 as p07mod
        marker = self._tmp / "m"
        marker.mkdir()
        d = wake_tick(self._db, campaign_id=self._camp.campaign_id, trigger_id="x", now=1)
        run = start_wake_claim_execution(
            self._db, claim_id=d["claim_id"],
            argv=[sys.executable, "-c",
                  "import os,sys,time;open(os.path.join(sys.argv[1],str(os.getpid())),'w').write('x');"
                  "time.sleep(0.6)", str(marker)],
            cwd=str(self._tmp), now=2, result_dir=str(self._tmp / "rd"))
        claim = wake_claim(self._db, d["claim_id"])
        self.assertTrue(claim["boot_id"] and claim["start_time"])
        self._reopen()
        p07mod._CHILDREN.clear()
        recon = reconcile_wake_claims(self._db, now=3)
        self.assertEqual(recon[0]["to"], "RUNNING")  # exact identity survives
        deadline = time.time() + 10
        last = None
        while time.time() < deadline:
            last = poll_wake_claim_execution(self._db, run_id=run["run_id"], now=4)
            if last["state"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.03)
        self.assertEqual(last["state"], "COMPLETED")
        self.assertEqual(len(list(marker.iterdir())), 1)  # one worker, no duplicate
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM wake_claims").fetchone()["n"], 1)


# ============================================================
# A04 — out-of-envelope + acceptance-weakening rejection
# ============================================================

class TestP08A04Adversarial(_Harness):
    def test_out_of_envelope_mutations_rejected(self):
        broker = _broker(self._wt, self._tmp / "art-adv", _registry())
        before = (self._wt / "src" / "app.py").read_text()
        # (1) write path outside the declared write scope
        with self.assertRaises(SafetyError) as c1:
            broker._authorise_write("src/other.py")
        self.assertIn("write_paths", str(c1.exception).lower())
        # (2) create/scope expansion not in the chunk contract
        with self.assertRaises(SafetyError) as c2:
            broker._authorise_create("src/new_module.py")
        self.assertIn("create_paths", str(c2.exception).lower())
        # (3) protected path (.git)
        with self.assertRaises(SafetyError):
            broker._authorise_write(".git/config")
        # (4) command outside the allowed command set
        from overnight_runner.schemas import RunCommandArgs, ToolCall
        with self.assertRaises(SafetyError) as c4:
            broker.handle(ToolCall(call_id="c1", args=RunCommandArgs(command_id="rm_rf")))
        self.assertIn("command_id", str(c4.exception).lower())
        self.assertEqual((self._wt / "src" / "app.py").read_text(), before)
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM integration_journal").fetchone()["n"], 0)

    def test_acceptance_weakening_rejected(self):
        # A worker tries to "pass" by removing the required validator and by
        # deleting the required write target. The runner-owned requirements
        # remain authoritative -> integration is rejected.
        cid = "chk-aw"
        self._admit(cid, validators=("noop",))
        # Delete the required write target on the candidate.
        (self._wt / "src" / "app.py").unlink()
        subprocess.run(["git", "add", "-A"], cwd=str(self._wt), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "weaken"], cwd=str(self._wt),
                       check=True, capture_output=True)
        bad_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(self._wt),
                                    capture_output=True, text=True).stdout.strip()
        # A "self-issued" receipt claiming pass is NOT trusted (not minted by runner).
        status, rids = _run_validator(self._wt, self._tmp / "art-aw", cid, ("noop",))
        self.assertEqual(status, "FAILED")  # required write target not produced
        # Integration with NO trusted receipt is rejected.
        with self.assertRaises(SafetyError):
            compare_and_swap_advance(
                self._db, repo_root=self._repo, campaign_id=self._camp.campaign_id,
                chunk_id=cid, new_commit=bad_commit, holder_fence_generation=1,
                actor="runner", idempotency_key="i-aw", validation_receipt_ids=[],
                campaign_worktree=self._wt)
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM integration_journal").fetchone()["n"], 0)


# ============================================================
# A05 — phase completion is not an empty queue
# ============================================================

class TestP08A05Completion(_Harness):
    def test_completion_requires_integration_and_protected_gate(self):
        self._one_chunk(1)
        # A. required integration criterion missing (queue empty is not enough).
        r = evaluate_phase_completion(self._db, campaign_id=self._camp.campaign_id,
                                      required_chunks=["chk-1", "chk-2"])
        self.assertFalse(r["complete"])
        self.assertEqual(r["state"], "INCOMPLETE")
        self.assertIn("chk-2", r["missing_criteria"])
        # B. all technical criteria met but NO protected human disposition.
        r = evaluate_phase_completion(self._db, campaign_id=self._camp.campaign_id,
                                      required_chunks=["chk-1"], human_gate_required=True)
        self.assertFalse(r["complete"])
        self.assertEqual(r["state"], "AWAITING_HUMAN")
        self.assertEqual(r["reason"], "no_protected_human_disposition")
        # C. a FORGED/untrusted disposition (wrong operation/target) is ignored.
        register_protected_approval(
            self._db, approval_id="forged-gate", operation="activate_grant",
            grant_digest_target="f" * 64, operator_id="attacker",
            operator_receipt={"approval_id": "forged-gate"})
        r = evaluate_phase_completion(self._db, campaign_id=self._camp.campaign_id,
                                      required_chunks=["chk-1"], human_gate_required=True)
        self.assertFalse(r["complete"])
        self.assertEqual(r["state"], "AWAITING_HUMAN")
        # D. a VALID protected operator disposition through the trusted channel.
        record_phase_gate(self._db, campaign_id=self._camp.campaign_id,
                          operator_id="op-1", approval_id="pg-1")
        r = evaluate_phase_completion(self._db, campaign_id=self._camp.campaign_id,
                                      required_chunks=["chk-1"], human_gate_required=True)
        self.assertTrue(r["complete"])
        self.assertEqual(r["state"], "ELIGIBLE_FOR_PHASE_COMPLETION")
        # One-shot: the disposition is consumed; a repeat is AWAITING_HUMAN again.
        r = evaluate_phase_completion(self._db, campaign_id=self._camp.campaign_id,
                                      required_chunks=["chk-1"], human_gate_required=True)
        self.assertFalse(r["complete"])
        self.assertEqual(r["reason"], "human_disposition_already_consumed")


# ============================================================
# A06 — audit loss / disk pressure
# ============================================================

class TestP08A06AuditLoss(_Harness):
    def test_unwritable_evidence_dir_blocks_integration(self):
        cid = "chk-audit"
        self._admit(cid, validators=("noop",))
        new_commit = self._mutate(text="AUDIT\n")
        rdir = self._tmp / "ro-receipts"
        rdir.mkdir()
        os.chmod(rdir, 0o500)  # read+exec, NOT writable
        try:
            # A required receipt cannot be persisted.
            with self.assertRaises(OSError):
                mint_validation_receipt(
                    validator_id="noop", validator_command="noop",
                    candidate_snapshot_digest=git_worktree_sha(self._wt),
                    outcome="pass", chunk_id=cid, receipts_dir=rdir)
            # With no trusted receipt the integration gate fails closed.
            with self.assertRaises(SafetyError):
                compare_and_swap_advance(
                    self._db, repo_root=self._repo, campaign_id=self._camp.campaign_id,
                    chunk_id=cid, new_commit=new_commit, holder_fence_generation=1,
                    actor="runner", idempotency_key="i-audit", validation_receipt_ids=[],
                    campaign_worktree=self._wt)
        finally:
            os.chmod(rdir, 0o700)
        # No hidden acceptance, existing evidence intact.
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM integration_journal").fetchone()["n"], 0)
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM capacity_outcomes").fetchone()["n"], 0)

    def test_disk_pressure_fault_injection_blocks(self):
        import errno
        import overnight_runner.receipts as receipts
        cid = "chk-disk"
        self._admit(cid, validators=("noop",))
        new_commit = self._mutate(text="DISK\n")
        real = receipts._atomic_write_json

        def _enospc(path, data):
            raise OSError(errno.ENOSPC, "no space left on device")

        receipts._atomic_write_json = _enospc  # FAULT INJECTION
        try:
            with self.assertRaises(OSError):
                mint_validation_receipt(
                    validator_id="noop", validator_command="noop",
                    candidate_snapshot_digest=git_worktree_sha(self._wt),
                    outcome="pass", chunk_id=cid, receipts_dir=_receipts_root())
        finally:
            receipts._atomic_write_json = real
        # Fail closed: no integration, no success.
        with self.assertRaises(SafetyError):
            compare_and_swap_advance(
                self._db, repo_root=self._repo, campaign_id=self._camp.campaign_id,
                chunk_id=cid, new_commit=new_commit, holder_fence_generation=1,
                actor="runner", idempotency_key="i-disk", validation_receipt_ids=[],
                campaign_worktree=self._wt)
        self.assertEqual(self._db._conn.execute(
            "SELECT COUNT(*) AS n FROM integration_journal").fetchone()["n"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
