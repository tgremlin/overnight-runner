#!/usr/bin/env python3
"""P08-A07 — REAL local model canary (model + toolchain + permitted scope).

Run with the P05 worker venv (pydantic-ai installed):

    /tmp/p08-venv/bin/python scripts/p08_real_canary.py

One linked lineage:

    actual installed qualified Qwen (qwen3.8:27b)
      -> P05 Pydantic-AI bounded worker (build_worker_agent/run_worker_agent)
      -> disposable source task
      -> broker-only proposals/applies (NO direct filesystem writes)
      -> real local Python/git toolchain
      -> real validators (py_compile + pytest)
      -> trusted runner validation receipts
      -> runner integration into a disposable private campaign branch
      -> actual accepted snapshot + budget/cost record (0 cost)

If the qualified local model/runtime is unavailable, prints a BLOCKED
record (never a synthesized PASS).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RUNNER_SRC = "/home/tgremlin/work/overnight-runner-ae/src"
FORGE_WORKERS = "/home/tgremlin/work/trio-game-forge-p5/python/trio-workers"
for p in (RUNNER_SRC, FORGE_WORKERS):
    if p not in sys.path:
        sys.path.insert(0, p)


def _blocked(reason: str, detail: str = "") -> int:
    """Emit a BLOCKED record AND return a NON-ZERO exit code.

    A blocked/failed canary MUST NOT look like shell success; consumers must
    inspect both the process status and the ``gate`` field.
    """
    print(json.dumps({"schema_version": "trio.p08-real-canary.v1",
                      "gate": "BLOCKED", "reason": reason, "detail": detail}, indent=2))
    return 3


def _git_identity(root: str) -> dict:
    """Real provenance for a source root: sha, branch, clean/dirty state."""
    try:
        sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        branch = subprocess.run(["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
        # Tracked-file drift only; untracked build/local artifacts do not mark
        # the source tree dirty.
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return {"root": root, "sha": "", "branch": "", "clean": False, "error": str(e)}
    return {"root": root, "sha": sha, "branch": branch,
            "clean": dirty == "", "dirty_entries": dirty.splitlines()[:20]}


def _provenance() -> dict:
    """Exact source revisions actually exercised, with verification.

    Runner source = the tree CONTAINING this script (the P08 candidate).
    Forge worker source = an explicitly supplied P08_FORGE_ROOT (pinned).
    """
    runner_root = str(Path(__file__).resolve().parent.parent)
    forge_root = os.environ.get("P08_FORGE_ROOT", "/home/tgremlin/work/trio-game-forge-p5")
    return {
        "runner": {**_git_identity(runner_root),
                   "package_path": str(Path(runner_root) / "src" / "overnight_runner")},
        "forge_worker": {**_git_identity(forge_root),
                         "package_path": str(Path(forge_root) / "python" / "trio-workers" / "trio_workers")},
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }


def main() -> int:
    try:
        import pydantic_ai  # noqa: F401
        from trio_workers.qualification import qualify_ollama
        from trio_workers.pydanticai_worker import WorkerDeps, build_worker_agent, run_worker_agent
        from trio_workers.runner_adapter import RunnerBrokerClient
        from trio_workers.worker_contracts import WorkerRequest
        from trio_workers.policy import WorkerPolicy
    except Exception as e:  # noqa: BLE001
        return _blocked("worker_runtime_unavailable", f"{type(e).__name__}: {e}")

    ev: dict = {"schema_version": "trio.p08-real-canary.v1"}

    # 0) SOURCE PROVENANCE — prove which revisions were actually exercised.
    prov = _provenance()
    ev["source_provenance"] = prov
    if not prov["runner"]["sha"] or not prov["forge_worker"]["sha"]:
        return _blocked("source_provenance_unverifiable",
                        f"runner={prov['runner']} forge={prov['forge_worker']}")
    if not prov["runner"]["clean"]:
        ev["gate"] = "BLOCKED"
        return _blocked("runner_source_dirty", str(prov["runner"]["dirty_entries"]))
    if not prov["forge_worker"]["clean"]:
        return _blocked("forge_worker_source_dirty",
                        str(prov["forge_worker"]["dirty_entries"]))

    # 1) REAL qualification of the accepted local model.
    profile = qualify_ollama(model_name="qwen3.8:27b")
    ev["qualified_profile"] = {
        "gate": profile.qualification_gate,
        "model_name": profile.model_name,
        "model_digest": profile.model_digest,
        "endpoint_url": profile.endpoint_url,
        "tokenizer_id": profile.tokenizer_id,
        "tokenizer_digest": profile.tokenizer_digest,
        "chat_template_digest": profile.chat_template_digest,
        "advertised_context_tokens": profile.advertised_context_tokens,
        "qualified_max_input_tokens": profile.qualified_max_input_tokens,
        "runtime": profile.runtime_version,
        "python": profile.python_version,
    }
    if profile.qualification_gate != "PASS":
        return _blocked("qualified_local_model_unavailable",
                        f"gate={profile.qualification_gate} notes={profile.notes[-2:]}")
    ev["classification"] = "REAL HOST"

    # runner imports (after qualification)
    from overnight_runner.admission import derive_admission
    from overnight_runner.campaign import (activate_campaign, create_campaign,
                                           record_chunk_accepted, update_budget_after_chunk)
    from overnight_runner.campaign_apply import apply_campaign_patch
    from overnight_runner.campaign_schemas import (AutonomyGrant, Budget, ChunkSpec,
                                                   RepoSnapshot, content_sha256)
    from overnight_runner.db import Database
    from overnight_runner.grants import activate_grant, load_grant
    from overnight_runner.integration import compare_and_swap_advance, ensure_campaign_worktree
    from overnight_runner.plans import load_plan_digest, register_plan
    from overnight_runner.protected_approvals import register_protected_approval
    from overnight_runner.receipts import mint_validation_receipt
    from overnight_runner.resources import acquire_lease, current_fence, process_identity
    from overnight_runner.safety import (git_commit_all, git_head, git_init_empty,
                                         git_worktree_sha, sha256_file)
    from overnight_runner.schemas import Disposition, ExecutionClass, TaskManifest
    from overnight_runner.worker import _finalise

    tmp = Path(tempfile.mkdtemp(prefix="p08-real-"))
    try:
        state = tmp / "state"
        state.mkdir(parents=True, exist_ok=True)
        os.environ["OVERNIGHT_STATE_DIR"] = str(state)
        os.environ["OVERNIGHT_RECEIPTS"] = "1"
        os.environ["TR_P06_CAMPAIGN_V2"] = "1"
        db = Database(state / "state.db")

        # 2) Disposable pilot repo.
        repo = tmp / "pilot"
        repo.mkdir()
        git_init_empty(repo)
        (repo / "src").mkdir()
        (repo / "tests").mkdir()
        (repo / "src" / "app.py").write_text("def value():\n    return 1\n")
        # PROTECTED immutable acceptance test: expresses the DESIRED behavior
        # (double() == 2), so it FAILS at baseline before implementation. It is
        # outside the worker's write authority (see policy below).
        protected_test = (
            "from src.app import value, double\n\n\n"
            "def test_value():\n    assert value() == 1\n\n\n"
            "def test_double():\n    assert double() == 2\n")
        (repo / "tests" / "test_app.py").write_text(protected_test)
        (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n.pytest_cache/\n")
        (repo / "tests" / "__init__.py").write_text("")
        git_commit_all(repo, "init")
        ev["pilot_repo"] = str(repo)
        ev["pilot_base_commit"] = git_head(repo)
        protected_test_sha = sha256_file(repo / "tests" / "test_app.py")
        ev["protected_test_sha256"] = protected_test_sha
        ev["protected_test_path"] = "tests/test_app.py"

        # 3) Plan + grant bound to the REAL qualified model identity.
        register_plan(db, plan_id="pl-canary", approved_artifact_id="pl-canary",
                      work_package_criterion_ids={"pkg-1": {"crit-1"}})
        # Canonical runtime_digest derivation FROM the qualification record
        # (no opaque synthetic digest). The exact derivation is persisted.
        canonical_runtime = json.dumps({
            "runtime": profile.runtime_version, "python": profile.python_version,
            "model_name": profile.model_name, "model_digest": profile.model_digest,
            "tokenizer_id": profile.tokenizer_id, "tokenizer_digest": profile.tokenizer_digest,
            "chat_template_digest": profile.chat_template_digest,
        }, sort_keys=True, separators=(",", ":"))
        runtime_digest = hashlib.sha256(canonical_runtime.encode()).hexdigest()
        ev["runtime_digest"] = runtime_digest
        ev["runtime_digest_derivation"] = (
            "sha256(canonical_json({runtime,python,model_name,model_digest,"
            "tokenizer_id,tokenizer_digest,chat_template_digest}))")
        grant = AutonomyGrant(
            schema_version="trio.grant.v1", grant_id="gr-canary", state="draft",
            plan_id="pl-canary", plan_revision=1,
            approved_plan_digest=load_plan_digest(db, "pl-canary"),
            repository_paths=["src/app.py", "tests/test_app.py"],
            allowed_write_paths=["src/app.py"],
            protected_paths=["tests/test_app.py"], allowed_operations=["noop"],
            runtime_digest=runtime_digest,
            model_name=profile.model_name or "qwen3.8:27b",
            model_digest=profile.model_digest or "0" * 64,
            policy_profile_id="pol-1", validator_profile_ids=["python_compile", "pytest"],
            provider_profile_id="local-ollama", egress_policy_id="eg-local",
            operator_id="op-canary",
            operator_receipt_digest=hashlib.sha256(json.dumps(
                {"approval_id": "appr-canary", "operator_id": "op-canary"},
                sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            budget=Budget(schema_version="trio.budget.v1", max_chunks=3, max_model_calls=10,
                          max_tool_calls=30, max_local_repairs=1, max_rechunks=1,
                          max_active_seconds=1800, max_wall_seconds=3600,
                          max_cost_microusd=0, context_token_budget=8192))
        register_protected_approval(
            db, approval_id="appr-canary", operation="activate_grant",
            grant_digest_target=content_sha256(grant), operator_id="op-canary",
            operator_receipt={"approval_id": "appr-canary"})
        activate_grant(db, grant=grant, operator_id="op-canary", approval_id="appr-canary")
        grant = load_grant(db, "gr-canary")

        base = git_head(repo)
        camp = create_campaign(db, plan_id="pl-canary", grant_id="gr-canary",
                               base_commit=base, base_tree_digest="b" * 64,
                               repo_root=str(repo))
        activate_campaign(db, campaign_id=camp.campaign_id)
        wt = ensure_campaign_worktree(repo_root=repo, campaign_id=camp.campaign_id,
                                      base_commit=base, db=db)
        ev["campaign_id"] = camp.campaign_id
        ev["grant_id"] = grant.grant_id
        ev["grant_digest"] = content_sha256(grant)

        chunk = ChunkSpec(
            schema_version="trio.chunk.v1", chunk_id="chk-canary",
            campaign_id=camp.campaign_id, package_id="pkg-1", revision=1,
            title="canary", objective="add double()",
            permitted_signature_paths=["src/app.py"],
            permitted_write_paths=["src/app.py"],
            permitted_read_paths=["src/app.py", "tests/test_app.py"],
            permitted_command_ids=["noop"],
            permitted_validator_ids=["python_compile", "pytest"],
            required_validator_ids=["python_compile", "pytest"],
            required_receipt_profiles=["python_compile", "pytest"],
            criterion_ids=["crit-1"], idempotency_key="idem-canary")
        receipt, _ = derive_admission(
            db, grant=grant, chunk=chunk, worker_id="wkr-canary",
            policy_profile_id=grant.policy_profile_id,
            validator_profile_ids=list(grant.validator_profile_ids),
            provider_profile_id=grant.provider_profile_id,
            current_accepted_snapshot=RepoSnapshot(
                schema_version="trio.repo-snapshot.v1", repository_id="local",
                commit=base, tree_digest="b" * 64),
            current_runtime_digest=grant.runtime_digest,
            current_model_name=grant.model_name, current_model_digest=grant.model_digest,
            current_policy_profile_id=grant.policy_profile_id,
            current_validator_profile_ids=list(grant.validator_profile_ids),
            current_provider_profile_id=grant.provider_profile_id)
        ev["admission_id"] = receipt.admission_id

        # 4) Fenced broker client: the worker proposes/applies ONLY through the
        #    accepted runner broker AND the P06 campaign fence.
        lease = acquire_lease(db, campaign_id=camp.campaign_id, resource_id="chunk:canary",
                              owner_id="wkr-canary", owner_boot_id="b", owner_pid=os.getpid(),
                              fence_generation=current_fence(db, camp.campaign_id).current_generation,
                              ttl_seconds=600)
        ident = process_identity(os.getpid())
        applied_ids: list[str] = []
        mutation_receipt_ids: list[str] = []

        class FencedBroker(RunnerBrokerClient):
            def apply(self, proposal_id: str) -> dict:
                res = apply_campaign_patch(
                    db, self._inner, str(repo), camp.campaign_id, proposal_id,
                    admission_fence_generation=lease.fence_generation,
                    lease_id=lease.lease_id, owner_id="wkr-canary",
                    owner_pid=os.getpid(), owner_start_time=ident["start_time"])
                applied_ids.append(proposal_id)
                if res.get("receipt_id"):
                    mutation_receipt_ids.append(res["receipt_id"])
                return res

        broker = FencedBroker(wt, runner_src=RUNNER_SRC,
                              allowed_write_paths={"src/app.py"},
                              allowed_create_paths=set(),
                              allowed_read_paths={"src/app.py", "tests/test_app.py"},
                              receipt_enabled=True)

        # 5) REAL model task (single-file, deterministic, harmless).
        original = (wt / "src" / "app.py").read_text()
        expected_sha = sha256_file(wt / "src" / "app.py")
        from trio_workers.policy import WorkerPolicy
        policy = WorkerPolicy(allowed_write_paths={"src/app.py"},
                              protected_test_paths={"tests/test_app.py"},
                              allowed_command_ids=set())
        req = WorkerRequest(request_id="req-canary", chunk_id="chk-canary",
                            job_id="job-canary", role="implementation",
                            attempt=1, cumulative_budget_ledger_id=f"bl-{camp.campaign_id}",
                            snapshot_digest=git_worktree_sha(wt), contract_tokens=0,
                            context_tokens=0, max_attempts=1)
        new_text = (original.rstrip("\n") + "\n\n\ndef double():\n    return value() * 2\n")
        ev["worker_policy"] = {"allowed_write_paths": sorted(policy.allowed_write_paths),
                               "protected_test_paths": sorted(policy.protected_test_paths)}
        prompt = (
            "You are a bounded worker. TASK INTENT: implement `double()` in src/app.py "
            "so the PROTECTED test suite passes (it requires double() == 2). "
            "You may modify src/app.py ONLY; you MUST NOT modify any test file.\n"
            "You MUST do exactly this sequence:\n"
            "1. call propose_patch with path='src/app.py', "
            f"expected_sha256='{expected_sha}', and new_text set to EXACTLY the following "
            "file content (verbatim):\n---BEGIN---\n"
            f"{new_text}---END---\n"
            "2. call apply_patch with the proposal_id returned by step 1.\n"
            "Then stop. Do not modify any other file. Do not explain."
        )
        agent = build_worker_agent(profile, role="implementation")
        deps = WorkerDeps(broker=broker, policy=policy, request=req)
        report = asyncio.run(run_worker_agent(agent, deps, prompt))
        ev["worker_completion_report"] = report.model_dump(mode="json")
        ev["broker_proposal_ids"] = list(broker._proposals.keys())
        ev["fenced_applied_proposal_ids"] = applied_ids
        ev["mutation_receipt_ids"] = mutation_receipt_ids
        ev["mutation_receipt_kinds"] = ["mutation/apply"] if mutation_receipt_ids else []

        if not applied_ids:
            return _blocked("model_did_not_apply_candidate",
                            "the local model did not complete a brokered apply")
        # Broker-only proof: the change is exactly what the broker applied.
        new_content = (wt / "src" / "app.py").read_text()
        if "def double()" not in new_content:
            return _blocked("candidate_missing_expected_change", new_content[:200])
        # Broker-only + scope integrity: the source changed, the PROTECTED test
        # did NOT (the worker had no write authority over it).
        if sha256_file(wt / "src" / "app.py") == sha256_file(repo / "src" / "app.py"):
            return _blocked("candidate_unchanged", "broker apply did not change the source")
        if sha256_file(wt / "tests" / "test_app.py") != protected_test_sha:
            return _blocked("protected_test_mutated",
                            "the protected acceptance test changed — refusing")
        # Semantic check on the ACTUAL candidate source (not name presence).
        sem = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0,'.'); from src.app import value, double; "
             "assert value()==1 and double()==2, (value(), double()); print('semantics OK')"],
            cwd=str(wt), capture_output=True, text=True)
        ev["semantic_check"] = {"ok": sem.returncode == 0, "stdout": sem.stdout.strip(),
                                "stderr": sem.stderr.strip()[-400:]}
        if sem.returncode != 0:
            return _blocked("candidate_semantics_failed", sem.stderr.strip()[-400:])

        # 6) Commit the candidate, run REAL validators, integrate.
        subprocess.run(["git", "add", "-A"], cwd=str(wt), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "canary"], cwd=str(wt),
                       check=True, capture_output=True)
        new_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(wt),
                                    capture_output=True, text=True).stdout.strip()
        candidate_snapshot = git_worktree_sha(wt)
        ev["candidate_commit"] = new_commit
        ev["candidate_snapshot_digest"] = candidate_snapshot
        ev["candidate_tree_digest"] = subprocess.run(
            ["git", "rev-parse", f"{new_commit}^{{tree}}"], cwd=str(wt),
            capture_output=True, text=True).stdout.strip()

        from overnight_runner.broker import Broker, CommandRegistry, CommandSpec
        from trio_workers.broker import BrokerError  # noqa: F401
        ids: list[str] = []

        def _mint(payload):
            rid = mint_validation_receipt(
                validator_id=payload["validator_id"],
                validator_command=payload["validator_command"],
                validator_profile=payload.get("validator_profile") or payload["validator_command"],
                candidate_snapshot_digest=payload["candidate_snapshot_digest"],
                chunk_id=payload.get("chunk_id", ""), outcome=payload["outcome"],
                detail=payload.get("detail", ""), receipts_dir=state / "receipts")
            ids.append(rid)
            return rid

        reg = CommandRegistry()
        reg.register(CommandSpec("pytest", [sys.executable, "-m", "pytest", "-q",
                                            "tests/test_app.py"], "repo", 180, "read"))
        vbroker = Broker(repo_root=wt, registry=reg,
                         allowed_write_paths=["src/app.py"], allowed_create_paths=[],
                         allowed_read_paths=["src/app.py", "tests/test_app.py"],
                         allowed_protected_read_paths=[], model_allowed_command_ids=[],
                         required_validator_ids=[], approved_repo_head=None,
                         artifact_dir=tmp / "art")
        manifest = TaskManifest(
            task_id="chk-canary", title="canary", objective="add double()",
            execution_class=ExecutionClass.SOURCE_MUTATION, repo={"path": str(wt)},
            paths={"write_paths": ["src/app.py"], "create_paths": [],
                   "read_paths": ["src/app.py", "tests/test_app.py"],
                   "protected_read_paths": []},
            commands={"model_allowed_command_ids": [], "required_validator_ids":
                      ["python_compile", "pytest"], "allow_no_mutation": True},
            model_profile={"model_name": profile.model_name or "qwen3.8:27b", "temperature": 0.0},
            context_budget={"max_read_bytes": 8192, "max_files_read": 4, "max_files_written": 2},
            limits={"max_model_turns": 1, "max_tool_calls": 4, "max_changed_files": 2,
                    "max_diff_lines": 400, "max_written_bytes": 65536,
                    "task_timeout_seconds": 180, "max_tool_result_bytes": 24576},
            acceptance_criteria=[])
        status, code, text, rids = _finalise(
            manifest=manifest, broker=vbroker, disposition=Disposition.DONE,
            artifact_dir=tmp / "art", applied_proposals=[],
            on_validation_receipt=_mint, env_digest="env", profile_digest="prof")
        ev["validator_status"] = status
        if status != "PASSED":
            return _blocked("real_validation_failed", f"{code}: {text}")
        ev["validation_receipt_ids"] = list(rids)

        result = compare_and_swap_advance(
            db, repo_root=repo, campaign_id=camp.campaign_id, chunk_id="chk-canary",
            new_commit=new_commit, holder_fence_generation=1, actor="runner",
            idempotency_key="i-canary", validation_receipt_ids=list(rids),
            campaign_worktree=wt)
        record_chunk_accepted(db, chunk_id="chk-canary", accepted_commit=new_commit,
                              accepted_tree_digest=candidate_snapshot)
        update_budget_after_chunk(db, ledger_id=f"bl-{camp.campaign_id}",
                                  delta_chunks=1, delta_model_calls=1)
        ev["integration_commit"] = result.committed_new_commit
        ev["accepted_snapshot_digest"] = candidate_snapshot
        ev["campaign_branch"] = f"refs/heads/campaign/{camp.campaign_id}"
        ev["budget_after"] = dict(db._conn.execute(
            "SELECT cumulative_chunks, cumulative_cost_microusd FROM budget_ledgers "
            "WHERE campaign_id=?", (camp.campaign_id,)).fetchone())
        ev["toolchain"] = {"python": sys.version.split()[0],
                           "git": subprocess.run(["git", "--version"], capture_output=True,
                                                 text=True).stdout.strip()}
        ev["cost_microusd"] = 0
        ev["gate"] = "PASS"
        db.close()
        print(json.dumps(ev, indent=2, default=str))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
