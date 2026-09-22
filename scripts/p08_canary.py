#!/usr/bin/env python3
"""P08-A07 — REAL host canary (read-only + harmless disposable mutation).

Run with the host Python:

    python3 scripts/p08_canary.py

Performs, against the ACTUAL installed local model/toolchain:

  A. REAL READ-ONLY canary — real local model identity + toolchain
     versions + a read-only repository/broker analysis operation.
  B. REAL HARMLESS MUTATION canary — a disposable local pilot repo,
     a synthetic finite grant, a bounded source/test change applied
     through the broker, REAL validators (py_compile + pytest), trusted
     validation receipts, and integration into a DISPOSABLE private
     campaign branch only.

Writes JSON evidence to stdout. No paid provider, no live Hermes, no
Foundation/game/vendor mutation.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from overnight_runner.admission import derive_admission  # noqa: E402
from overnight_runner.broker import Broker, CommandRegistry, CommandSpec, Proposal  # noqa: E402
from overnight_runner.campaign import (  # noqa: E402
    activate_campaign, create_campaign, record_chunk_accepted, update_budget_after_chunk,
)
from overnight_runner.campaign_apply import apply_campaign_patch  # noqa: E402
from overnight_runner.campaign_schemas import (  # noqa: E402
    AutonomyGrant, Budget, ChunkSpec, RepoSnapshot, content_sha256,
)
from overnight_runner.db import Database  # noqa: E402
from overnight_runner.grants import activate_grant, load_grant  # noqa: E402
from overnight_runner.integration import (  # noqa: E402
    compare_and_swap_advance, ensure_campaign_worktree,
)
from overnight_runner.plans import load_plan_digest, register_plan  # noqa: E402
from overnight_runner.protected_approvals import register_protected_approval  # noqa: E402
from overnight_runner.receipts import mint_validation_receipt  # noqa: E402
from overnight_runner.resources import acquire_lease, current_fence, process_identity  # noqa: E402
from overnight_runner.safety import git_commit_all, git_head, git_init_empty  # noqa: E402
from overnight_runner.schemas import Disposition, ExecutionClass, TaskManifest  # noqa: E402
from overnight_runner.worker import _finalise  # noqa: E402

OLLAMA = "http://127.0.0.1:11434"


def _run(argv, **kw):
    return subprocess.run(argv, capture_output=True, text=True, **kw)


def read_only_canary(tmp: Path) -> dict:
    ev: dict = {"classification": "REAL HOST", "kind": "read_only"}
    # Local model identity (actual Ollama).
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=5) as r:
            tags = json.loads(r.read().decode())
        models = [{"name": m.get("name"), "digest": m.get("digest"),
                   "params": m.get("details", {}).get("parameter_size"),
                   "quant": m.get("details", {}).get("quantization_level")}
                  for m in tags.get("models", [])]
        ev["models"] = models
        qwen = next((m for m in models if "qwen" in (m["name"] or "")), None)
        ev["qualified_local_model"] = qwen
        ev["ollama_reachable"] = True
    except Exception as e:  # noqa: BLE001
        ev["ollama_reachable"] = False
        ev["ollama_error"] = f"{type(e).__name__}: {e}"
    # Toolchain identity.
    ev["toolchain"] = {
        "python": sys.version.split()[0],
        "git": _run(["git", "--version"]).stdout.strip(),
        "gcc": (_run(["gcc", "--version"]).stdout.splitlines() or [""])[0],
    }
    # Read-only repository/broker analysis operation on a disposable repo.
    ro = tmp / "ro_repo"
    ro.mkdir()
    git_init_empty(ro)
    (ro / "src").mkdir()
    (ro / "src" / "app.py").write_text("BASE\n")
    git_commit_all(ro, "init")
    reg = CommandRegistry()
    reg.register(CommandSpec("noop", ["true"], "repo", 5, "read"))
    broker = Broker(repo_root=ro, registry=reg, allowed_write_paths=[],
                    allowed_create_paths=[], allowed_read_paths=["src/app.py"],
                    allowed_protected_read_paths=[], model_allowed_command_ids=["noop"],
                    required_validator_ids=[], approved_repo_head=None,
                    artifact_dir=tmp / "ro_art")
    read = broker.read_exact("src/app.py")
    ev["read_only_operation"] = {"git_head": git_head(ro), "read_ok": bool(read)}
    return ev


def mutation_canary(tmp: Path) -> dict:
    ev: dict = {"classification": "REAL LOCAL DISPOSABLE", "kind": "harmless_mutation"}
    state = tmp / "state"
    state.mkdir(parents=True, exist_ok=True)
    os.environ["OVERNIGHT_STATE_DIR"] = str(state)
    os.environ["OVERNIGHT_RECEIPTS"] = "1"
    # Enable campaign-v2 for this ISOLATED disposable canary only.
    os.environ["TR_P06_CAMPAIGN_V2"] = "1"
    db = Database(state / "state.db")
    repo = tmp / "pilot"
    repo.mkdir()
    git_init_empty(repo)
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "src" / "app.py").write_text("def value():\n    return 1\n")
    (repo / "tests" / "test_app.py").write_text(
        "from src.app import value\n\n\ndef test_value():\n    assert value() == 1\n")
    # Keep the candidate worktree clean: real validators (py_compile/pytest)
    # write caches that must not register as mutation drift.
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n.pytest_cache/\n")
    (repo / "tests" / "__init__.py").write_text("")
    git_commit_all(repo, "init")
    ev["pilot_repo"] = str(repo)
    ev["pilot_base_commit"] = git_head(repo)

    # Synthetic finite grant + plan.
    register_plan(db, plan_id="pl-canary", approved_artifact_id="pl-canary",
                  work_package_criterion_ids={"pkg-1": {"crit-1"}})
    g = AutonomyGrant(
        schema_version="trio.grant.v1", grant_id="gr-canary", state="draft",
        plan_id="pl-canary", plan_revision=1,
        approved_plan_digest=load_plan_digest(db, "pl-canary"),
        repository_paths=["src/app.py", "tests/test_app.py"],
        allowed_write_paths=["src/app.py", "tests/test_app.py"],
        protected_paths=[], allowed_operations=["noop"],
        runtime_digest="a" * 64, model_name="qwen-local", model_digest="b" * 64,
        policy_profile_id="pol-1", validator_profile_ids=["python_compile", "pytest"],
        provider_profile_id="local-ollama", egress_policy_id="eg-local",
        operator_id="op-canary", operator_receipt_digest="c" * 64,
        budget=Budget(schema_version="trio.budget.v1", max_chunks=3, max_model_calls=10,
                      max_tool_calls=20, max_local_repairs=1, max_rechunks=1,
                      max_active_seconds=600, max_wall_seconds=3600,
                      max_cost_microusd=0, context_token_budget=8192),
    )
    register_protected_approval(
        db, approval_id="appr-canary", operation="activate_grant",
        grant_digest_target=content_sha256(g), operator_id="op-canary",
        operator_receipt={"approval_id": "appr-canary"})
    activate_grant(db, grant=g, operator_id="op-canary", approval_id="appr-canary")
    grant = load_grant(db, "gr-canary")
    ev["grant_id"] = grant.grant_id
    ev["grant_digest"] = content_sha256(grant)

    base = git_head(repo)
    camp = create_campaign(db, plan_id="pl-canary", grant_id="gr-canary",
                           base_commit=base, base_tree_digest="b" * 64,
                           repo_root=str(repo))
    activate_campaign(db, campaign_id=camp.campaign_id)
    ev["campaign_id"] = camp.campaign_id
    wt = ensure_campaign_worktree(repo_root=repo, campaign_id=camp.campaign_id,
                                  base_commit=base, db=db)

    # Admission (durable, runner-owned).
    chunk = ChunkSpec(
        schema_version="trio.chunk.v1", chunk_id="chk-canary",
        campaign_id=camp.campaign_id, package_id="pkg-1", revision=1,
        title="canary", objective="bounded change",
        permitted_signature_paths=["src/app.py"],
        permitted_write_paths=["src/app.py", "tests/test_app.py"],
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
        current_runtime_digest=grant.runtime_digest, current_model_name=grant.model_name,
        current_model_digest=grant.model_digest,
        current_policy_profile_id=grant.policy_profile_id,
        current_validator_profile_ids=list(grant.validator_profile_ids),
        current_provider_profile_id=grant.provider_profile_id)
    ev["admission_id"] = receipt.admission_id

    # Brokered real mutation.
    reg = CommandRegistry()
    reg.register(CommandSpec("noop", ["true"], "repo", 5, "read"))
    from overnight_runner.receipts import mint_mutation_receipt

    def _mint(payload: dict) -> str:
        try:
            return mint_mutation_receipt(
                proposal_id=payload.get("proposal_id", ""), path=payload.get("path", ""),
                op=payload.get("op", ""), pre_sha256=payload.get("pre_sha256", ""),
                post_sha256=payload.get("post_sha256", ""),
                bytes_written=int(payload.get("bytes_written", 0)),
                candidate_snapshot_digest=payload.get("candidate_snapshot_digest", ""),
                receipts_dir=state / "receipts")
        except Exception:
            return ""

    broker = Broker(repo_root=wt, registry=reg,
                    allowed_write_paths=["src/app.py", "tests/test_app.py"],
                    allowed_create_paths=[],
                    allowed_read_paths=["src/app.py", "tests/test_app.py"],
                    on_apply_receipt=_mint,
                    allowed_protected_read_paths=[],
                    model_allowed_command_ids=["noop"], required_validator_ids=[],
                    approved_repo_head=None, artifact_dir=tmp / "art")
    broker.proposals["p-canary"] = Proposal(
        proposal_id="p-canary", op="replace_file", path="src/app.py",
        abs_path=wt / "src" / "app.py", before_text=(wt / "src" / "app.py").read_text(),
        proposed_text="def value():\n    return 2\n", preview_diff="",
        changed_lines=1, proposed_bytes=24)
    lease = acquire_lease(db, campaign_id=camp.campaign_id, resource_id="chunk:canary",
                          owner_id="wkr-canary", owner_boot_id="b", owner_pid=os.getpid(),
                          fence_generation=current_fence(db, camp.campaign_id).current_generation,
                          ttl_seconds=300)
    ident = process_identity(os.getpid())
    applied = apply_campaign_patch(
        db, broker, str(repo), camp.campaign_id, "p-canary",
        admission_fence_generation=lease.fence_generation, lease_id=lease.lease_id,
        owner_id="wkr-canary", owner_pid=os.getpid(), owner_start_time=ident["start_time"])
    ev["mutation_receipt_id"] = applied.get("receipt_id", "")
    # Update the test to match the real change, then commit the candidate.
    (wt / "tests" / "test_app.py").write_text(
        "from src.app import value\n\n\ndef test_value():\n    assert value() == 2\n")
    _run(["git", "add", "-A"], cwd=str(wt))
    _run(["git", "commit", "-m", "canary"], cwd=str(wt))
    new_commit = _run(["git", "rev-parse", "HEAD"], cwd=str(wt)).stdout.strip()
    ev["candidate_commit"] = new_commit

    # REAL validators through the accepted validator boundary.
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

    reg2 = CommandRegistry()
    reg2.register(CommandSpec("pytest", [sys.executable, "-m", "pytest", "-q",
                                         "tests/test_app.py"], "repo", 120, "read"))
    broker2 = Broker(repo_root=wt, registry=reg2,
                     allowed_write_paths=["src/app.py", "tests/test_app.py"],
                     allowed_create_paths=[],
                     allowed_read_paths=["src/app.py", "tests/test_app.py"],
                     allowed_protected_read_paths=[], model_allowed_command_ids=[],
                     required_validator_ids=[], approved_repo_head=None,
                     artifact_dir=tmp / "art2")
    manifest = TaskManifest(
        task_id="chk-canary", title="canary", objective="bounded change",
        execution_class=ExecutionClass.SOURCE_MUTATION, repo={"path": str(wt)},
        paths={"write_paths": ["src/app.py"], "create_paths": [],
               "read_paths": ["src/app.py", "tests/test_app.py"],
               "protected_read_paths": []},
        commands={"model_allowed_command_ids": [], "required_validator_ids":
                  ["python_compile", "pytest"], "allow_no_mutation": True},
        model_profile={"model_name": "qwen-local", "temperature": 0.0},
        context_budget={"max_read_bytes": 4096, "max_files_read": 4, "max_files_written": 2},
        limits={"max_model_turns": 1, "max_tool_calls": 4, "max_changed_files": 2,
                "max_diff_lines": 400, "max_written_bytes": 65536,
                "task_timeout_seconds": 120, "max_tool_result_bytes": 24576},
        acceptance_criteria=[])
    status, code, text, rids = _finalise(
        manifest=manifest, broker=broker2, disposition=Disposition.DONE,
        artifact_dir=tmp / "art2", applied_proposals=[],
        on_validation_receipt=_mint, env_digest="env", profile_digest="prof")
    ev["validator_status"] = status
    ev["validator_receipt_ids"] = rids
    if status != "PASSED":
        ev["failed_detail"] = text
        db.close()
        return ev

    result = compare_and_swap_advance(
        db, repo_root=repo, campaign_id=camp.campaign_id, chunk_id="chk-canary",
        new_commit=new_commit, holder_fence_generation=1, actor="runner",
        idempotency_key="i-canary", validation_receipt_ids=list(rids),
        campaign_worktree=wt)
    record_chunk_accepted(db, chunk_id="chk-canary", accepted_commit=new_commit,
                          accepted_tree_digest="c" * 64)
    update_budget_after_chunk(db, ledger_id=f"bl-{camp.campaign_id}",
                              delta_chunks=1, delta_model_calls=1)
    ev["integration_commit"] = result.committed_new_commit
    ev["campaign_branch"] = f"refs/heads/campaign/{camp.campaign_id}"
    ev["budget_after"] = dict(db._conn.execute(
        "SELECT cumulative_chunks, cumulative_cost_microusd FROM budget_ledgers "
        "WHERE campaign_id=?", (camp.campaign_id,)).fetchone())
    db.close()
    return ev


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="p08-canary-"))
    try:
        out = {
            "schema_version": "trio.p08-canary.v1",
            "read_only": read_only_canary(tmp),
            "mutation": mutation_canary(tmp),
        }
        print(json.dumps(out, indent=2))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
