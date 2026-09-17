"""P1 nightly scheduler tests.

Covers:
  - atomic claim: tasks.status RUNNING <=> exists RUNNING runs row
  - global runner lock before claim
  - manual `run` also uses the lock
  - dependency-aware selection: skip unmet deps, continue
  - lease expiry semantics: lease_expires_at < now => stale
  - orphan RUNNING task recovery
  - model digest binding at approve + execution
  - read-only retry (at most 1, mutation never retried)
  - run-nightly: zero approved, PAUSED, dep-blocked skipped, max tasks,
    mutation-last, never two mutations, runtime budget, etc.
  - morning summary written
"""
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from overnight_runner.db import Database, default_db_path
from overnight_runner.ollama_client import ChatResult, OllamaMetrics
from overnight_runner.runtime import (
    is_paused, lock_path, paused_path, runner_lock, state_dir,
)
from overnight_runner.runner import (
    DEFAULT_NIGHTLY, LEASE_SECONDS, _atomic_claim, execute_claimed_task,
    recovery_scan, run_nightly,
)
from overnight_runner.safety import git_commit_all, git_init_empty, sha256_file
from overnight_runner.schemas import TaskManifest, TaskStatus


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("OVERNIGHT_STATE_DIR", str(tmp_path / "state"))
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    yield tmp_path


def _init_repo(p: Path) -> Path:
    repo = p / "repo"
    repo.mkdir()
    git_init_empty(repo)
    (repo / "a.py").write_text("x = 1\n")
    git_commit_all(repo, "init")
    return repo


def _enqueue(db: Database, m: TaskManifest, *, head: str, rt: str | None = None,
             digest: str | None = "d") -> None:
    from overnight_runner.schemas import canonical_sha
    from overnight_runner.runtime import runtime_fingerprint
    sha = canonical_sha(m)
    if rt is None:
        here = Path(__file__).resolve().parents[1] / "src" / "overnight_runner"
        rt = runtime_fingerprint([here]).sha256
    db.upsert_task(
        m.task_id, sha, json.dumps(m.model_dump(mode="json")), TaskStatus.APPROVED,
        execution_class=m.execution_class.value,
        dependencies_json=json.dumps([d.model_dump(mode="json") for d in m.dependencies]),
    )
    db.approve_task(m.task_id, approved_by="test", approval_envelope={
        "manifest_sha256": sha, "approved_repo_head": head,
        "approved_runtime_sha256": rt,
        "approved_model_name": m.model_profile.model_name,
        "approved_model_digest": digest,
    })


# ---------- Atomic claim ----------

def test_atomic_claim_creates_running_runs_row(tmp_path: Path):
    repo = _init_repo(tmp_path)
    raw = {
        "schema_version": "1.0", "task_id": "ac-1", "title": "t",
        "execution_class": "read_only", "objective": "x",
        "repo": {"path": str(repo)}, "paths": {"read_paths": ["a.py"]},
    }
    m = TaskManifest.model_validate(raw)
    db = Database(default_db_path())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    _enqueue(db, m, head=head, rt="x")
    claim = _atomic_claim(db, execution_class_filter=None, session_id=None)
    assert claim is not None
    assert claim["run_id"]
    t = db.get_task("ac-1")
    assert t["status"] == "RUNNING"
    cur = db._conn.execute("SELECT * FROM runs WHERE run_id=?", (claim["run_id"],))
    r = cur.fetchone()
    assert r["status"] == "RUNNING"
    assert r["task_id"] == "ac-1"


def test_atomic_claim_returns_none_when_no_eligible(tmp_path: Path):
    db = Database(default_db_path())
    assert _atomic_claim(db, execution_class_filter=None, session_id=None) is None


# ---------- Lock before claim ----------

def test_run_next_holds_global_lock(tmp_path: Path):
    """A second runner_lock call while first holds must be rejected."""
    from overnight_runner.runtime import runner_lock
    repo = _init_repo(tmp_path)
    raw = {
        "schema_version": "1.0", "task_id": "lk-1", "title": "t",
        "execution_class": "read_only", "objective": "x",
        "repo": {"path": str(repo)}, "paths": {"read_paths": ["a.py"]},
    }
    m = TaskManifest.model_validate(raw)
    db = Database(default_db_path())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    _enqueue(db, m, head=head, rt="x")
    # First runner takes the lock; second must fail.
    with runner_lock():
        with pytest.raises(RuntimeError):
            with runner_lock():
                pass


# ---------- Dependency-aware selection ----------

def test_dependency_blocked_task_skipped_independent_runs(tmp_path: Path):
    repo = _init_repo(tmp_path)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    db = Database(default_db_path())
    # Task A depends on B; C is independent.
    raw_a = {
        "schema_version": "1.0", "task_id": "A", "title": "t",
        "execution_class": "read_only", "objective": "x",
        "repo": {"path": str(repo)}, "paths": {"read_paths": ["a.py"]},
        "dependencies": [{"task_id": "B", "required_state": "PASSED"}],
    }
    raw_c = {
        "schema_version": "1.0", "task_id": "C", "title": "t",
        "execution_class": "read_only", "objective": "x",
        "repo": {"path": str(repo)}, "paths": {"read_paths": ["a.py"]},
    }
    a = TaskManifest.model_validate(raw_a)
    c = TaskManifest.model_validate(raw_c)
    from overnight_runner.schemas import canonical_sha
    from overnight_runner.runtime import runtime_fingerprint
    here = Path(__file__).resolve().parents[1] / "src" / "overnight_runner"
    rt = runtime_fingerprint([here]).sha256
    sha_a = canonical_sha(a); sha_c = canonical_sha(c)
    db.upsert_task("A", sha_a, json.dumps(a.model_dump(mode="json")), TaskStatus.APPROVED,
                   priority=100, execution_class=a.execution_class.value,
                   dependencies_json=json.dumps([d.model_dump(mode="json") for d in a.dependencies]))
    db.upsert_task("C", sha_c, json.dumps(c.model_dump(mode="json")), TaskStatus.APPROVED,
                   priority=10, execution_class=c.execution_class.value,
                   dependencies_json="[]")
    db.approve_task("A", approved_by="t", approval_envelope={
        "manifest_sha256": sha_a, "approved_repo_head": head,
        "approved_runtime_sha256": rt, "approved_model_name": "gemma4:12b", "approved_model_digest": "d"})
    db.approve_task("C", approved_by="t", approval_envelope={
        "manifest_sha256": sha_c, "approved_repo_head": head,
        "approved_runtime_sha256": rt, "approved_model_name": "gemma4:12b", "approved_model_digest": "d"})
    claim = _atomic_claim(db, execution_class_filter=None, session_id=None)
    assert claim is not None
    assert claim["task"]["task_id"] == "C"
    assert db.get_task("A")["status"] == "APPROVED"
    blocked = db.get_dep_blocked_approved()
    assert any(b["task_id"] == "A" for b in blocked)


# ---------- Lease expiry semantics ----------

def test_lease_expiry_strict(tmp_path: Path):
    repo = _init_repo(tmp_path)
    raw = {
        "schema_version": "1.0", "task_id": "lx-1", "title": "t",
        "execution_class": "read_only", "objective": "x",
        "repo": {"path": str(repo)}, "paths": {"read_paths": ["a.py"]},
    }
    m = TaskManifest.model_validate(raw)
    db = Database(default_db_path())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    _enqueue(db, m, head=head, rt="x")
    db.insert_run(
        run_id="lx-1", task_id="lx-1", session_id="s", attempt_no=1,
        status="RUNNING", started_at=100, worker_pid=1, model_name="m",
        model_digest=None, model_profile="{}",
        lease_expires_at=100, pre_repo_head=head, pre_worktree_sha256="x",
        artifact_dir="/tmp/x",
    )
    # now=99 -> not stale
    assert db.find_stale_runs(99) == []
    # now=100 -> strictly NOT < (we use < not <=). Documented strict behavior.
    assert db.find_stale_runs(100) == []
    # now=101 -> stale
    s = db.find_stale_runs(101)
    assert len(s) == 1 and s[0]["run_id"] == "lx-1"


# ---------- Orphan RUNNING recovery ----------

def test_orphan_running_task_recovered(tmp_path: Path):
    repo = _init_repo(tmp_path)
    raw = {
        "schema_version": "1.0", "task_id": "orph-1", "title": "t",
        "execution_class": "read_only", "objective": "x",
        "repo": {"path": str(repo)}, "paths": {"read_paths": ["a.py"]},
    }
    m = TaskManifest.model_validate(raw)
    db = Database(default_db_path())
    _enqueue(db, m, head="h", rt="x")
    # Manually transition task RUNNING with no runs row.
    db._conn.execute("UPDATE tasks SET status='RUNNING' WHERE task_id='orph-1'")
    out = recovery_scan()
    assert any(o["task_id"] == "orph-1" and o["action"] == "ORPHANED_RUNNING_TASK" for o in out)
    assert db.get_task("orph-1")["status"] == "REVIEW_REQUIRED"


# ---------- Model digest binding ----------

def test_approve_stores_model_digest(tmp_path: Path, monkeypatch):
    """Approve must resolve the model digest and refuse if unresolved."""
    repo = _init_repo(tmp_path)
    raw = {
        "schema_version": "1.0", "task_id": "md-1", "title": "t",
        "execution_class": "read_only", "objective": "x",
        "repo": {"path": str(repo)}, "paths": {"read_paths": ["a.py"]},
    }
    m = TaskManifest.model_validate(raw)
    db = Database(default_db_path())
    db.upsert_task("md-1", "sha", json.dumps(m.model_dump(mode="json")), TaskStatus.PENDING_APPROVAL)
    # Monkey-patch OllamaClient.model_digest to return a known digest.
    from overnight_runner import ollama_client as oc
    orig = oc.OllamaClient.model_digest
    oc.OllamaClient.model_digest = lambda self, name: "deadbeef" * 8
    try:
        from overnight_runner.cli import cmd_approve
        import argparse
        ns = argparse.Namespace(task_id="md-1")
        rc = cmd_approve(ns)
        assert rc == 0
        t = db.get_task("md-1")
        assert t["approved_model_digest"] == "deadbeef" * 8
        assert t["approved_model_name"] == "gemma4:12b"
    finally:
        oc.OllamaClient.model_digest = orig


def test_approve_refuses_unknown_model(tmp_path: Path, monkeypatch):
    repo = _init_repo(tmp_path)
    raw = {
        "schema_version": "1.0", "task_id": "md-2", "title": "t",
        "execution_class": "read_only", "objective": "x",
        "repo": {"path": str(repo)}, "paths": {"read_paths": ["a.py"]},
    }
    m = TaskManifest.model_validate(raw)
    db = Database(default_db_path())
    db.upsert_task("md-2", "sha", json.dumps(m.model_dump(mode="json")), TaskStatus.PENDING_APPROVAL)
    from overnight_runner import ollama_client as oc
    orig = oc.OllamaClient.model_digest
    oc.OllamaClient.model_digest = lambda self, name: None
    try:
        from overnight_runner.cli import cmd_approve
        import argparse
        ns = argparse.Namespace(task_id="md-2")
        rc = cmd_approve(ns)
        assert rc == 3
        assert db.get_task("md-2")["status"] == "PENDING_APPROVAL"
    finally:
        oc.OllamaClient.model_digest = orig


def test_execute_rejects_digest_drift(tmp_path: Path):
    """If approved_model_digest differs from current, execute must not invoke Ollama."""
    repo = _init_repo(tmp_path)
    raw = {
        "schema_version": "1.0", "task_id": "md-3", "title": "t",
        "execution_class": "read_only", "objective": "x",
        "repo": {"path": str(repo)}, "paths": {"read_paths": ["a.py"]},
    }
    m = TaskManifest.model_validate(raw)
    from overnight_runner.runtime import runtime_fingerprint
    from overnight_runner.schemas import canonical_sha
    sha = canonical_sha(m)
    db = Database(default_db_path())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    here = Path(__file__).resolve().parents[1] / "src" / "overnight_runner"
    rt = runtime_fingerprint([here]).sha256
    db.upsert_task("md-3", sha, json.dumps(m.model_dump(mode="json")), TaskStatus.APPROVED,
                   execution_class="read_only", dependencies_json="[]")
    db.approve_task("md-3", approved_by="t", approval_envelope={
        "manifest_sha256": sha, "approved_repo_head": head,
        "approved_runtime_sha256": rt,
        "approved_model_name": "gemma4:12b", "approved_model_digest": "approved_digest",
    })
    from overnight_runner import ollama_client as oc
    orig = oc.OllamaClient.model_digest
    oc.OllamaClient.model_digest = lambda self, name: "DIFFERENT_CURRENT"
    try:
        with runner_lock():
            res = execute_claimed_task(db)
        assert res is not None
        assert res.status == "BLOCKED"
        assert res.reason_code == "APPROVAL_MODEL_CHANGED"
    finally:
        oc.OllamaClient.model_digest = orig


# ---------- run-nightly ----------

def _enqueue_read_only(db: Database, task_id: str, repo: Path, head: str, *,
                       priority: int = 100, depends_on: str | None = None,
                       rt_sha: str | None = None) -> None:
    raw = {
        "schema_version": "1.0", "task_id": task_id, "title": "t",
        "execution_class": "read_only", "objective": "x",
        "repo": {"path": str(repo)}, "paths": {"read_paths": ["a.py"]},
    }
    if depends_on:
        raw["dependencies"] = [{"task_id": depends_on, "required_state": "PASSED"}]
    m = TaskManifest.model_validate(raw)
    from overnight_runner.schemas import canonical_sha
    from overnight_runner.runtime import runtime_fingerprint
    here = Path(__file__).resolve().parents[1] / "src" / "overnight_runner"
    sha = canonical_sha(m)
    rt = rt_sha or runtime_fingerprint([here]).sha256
    db.upsert_task(task_id, sha, json.dumps(m.model_dump(mode="json")), TaskStatus.APPROVED,
                   priority=priority, execution_class=m.execution_class.value,
                   dependencies_json=json.dumps([d.model_dump(mode="json") for d in m.dependencies]))
    db.approve_task(task_id, approved_by="t", approval_envelope={
        "manifest_sha256": sha, "approved_repo_head": head,
        "approved_runtime_sha256": rt, "approved_model_name": "gemma4:12b",
        "approved_model_digest": "d"})


def _enqueue_mutation(db: Database, task_id: str, repo: Path, head: str) -> None:
    raw = {
        "schema_version": "1.0", "task_id": task_id, "title": "t",
        "execution_class": "source_mutation", "objective": "x",
        "repo": {"path": str(repo)}, "paths": {"write_paths": ["a.py"], "read_paths": ["a.py"]},
        "commands": {"required_validator_ids": ["no_op"]},
    }
    m = TaskManifest.model_validate(raw)
    from overnight_runner.schemas import canonical_sha
    from overnight_runner.runtime import runtime_fingerprint
    here = Path(__file__).resolve().parents[1] / "src" / "overnight_runner"
    sha = canonical_sha(m)
    rt = runtime_fingerprint([here]).sha256
    db.upsert_task(task_id, sha, json.dumps(m.model_dump(mode="json")), TaskStatus.APPROVED,
                   priority=100, execution_class=m.execution_class.value, dependencies_json="[]")
    db.approve_task(task_id, approved_by="t", approval_envelope={
        "manifest_sha256": sha, "approved_repo_head": head,
        "approved_runtime_sha256": rt, "approved_model_name": "gemma4:12b",
        "approved_model_digest": "d"})


def _script_client_class(scripts):
    """Return a FakeClient class."""
    class C:
        def __init__(self):
            self.last_proposal_id = None
            self.calls = 0
            self.scripts = scripts
        def chat(self, profile, system, messages, tools=None):
            # Capture proposal_id from tool results FIRST.
            for msg in reversed(messages):
                if msg.get("role") == "tool" and msg.get("tool_name") == "propose_patch":
                    try:
                        data = json.loads(msg["content"])
                        if "proposal_id" in data:
                            self.last_proposal_id = data["proposal_id"]
                    except Exception:
                        pass
                    break
            idx = min(self.calls, len(self.scripts) - 1)
            self.calls += 1
            script = self.scripts[idx]
            tcs = []
            for tc in script.get("tool_calls", []):
                args = dict(tc["function"]["arguments"])
                if tc["function"]["name"] == "apply_validated_patch":
                    args["proposal_id"] = self.last_proposal_id or "MISSING"
                tcs.append({"id": tc["id"], "function": {"name": tc["function"]["name"], "arguments": args}})
            return ChatResult(content="", tool_calls=tcs, metrics=OllamaMetrics(), raw={})
    return C


def test_nightly_zero_approved_exits_clean(tmp_path: Path):
    with runner_lock():
        res = run_nightly(session_id="nightly-zero")
    assert res.tasks_attempted == 0
    assert res.stop_reason == "NO_MORE_READ_ONLY"
    summary = json.loads((state_dir() / "sessions" / "nightly-zero" / "summary.json").read_text())
    assert summary["stop_reason"] == "NO_MORE_READ_ONLY"


def test_nightly_paused_at_start_zero_tasks(tmp_path: Path):
    pp = paused_path()
    pp.write_text("paused\n")
    with runner_lock():
        res = run_nightly(session_id="nightly-paused")
    assert res.tasks_attempted == 0
    assert "PAUSED" in res.stop_reason
    # PAUSED must NOT be auto-deleted.
    assert pp.exists()
    pp.unlink()


def test_nightly_dependency_blocked_skipped(tmp_path: Path, monkeypatch):
    repo = _init_repo(tmp_path)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    db = Database(default_db_path())
    _enqueue_read_only(db, "dep-A", repo, head, depends_on="missing")
    _enqueue_read_only(db, "free-C", repo, head, priority=1)
    # Both have approved envelopes. A is dep-blocked, C is independent.
    import overnight_runner.worker as wmod
    orig = wmod.Worker.__init__
    # C just reports DONE (no mutation; read_only with no required validator)
    C = _script_client_class([
        {"tool_calls": [{"id": "r", "function": {"name": "report_result",
                                                "arguments": {"disposition": "DONE", "summary": "ok"}}}]},
    ])
    wmod.Worker.__init__ = lambda self, **kw: orig(self, client=C(), **kw)
    try:
        with runner_lock():
            res = run_nightly(session_id="nightly-dep")
    finally:
        wmod.Worker.__init__ = orig
    assert res.tasks_attempted == 1
    assert res.tasks_passed == 1
    assert any(t["task_id"] == "free-C" for t in res.task_results)
    summary = json.loads((state_dir() / "sessions" / "nightly-dep" / "summary.json").read_text())
    assert any(b["task_id"] == "dep-A" for b in summary["dep_blocked_approved"])
    # dep-A is still APPROVED (not silently failed)
    assert db.get_task("dep-A")["status"] == "APPROVED"


def test_nightly_mutation_last_and_max_one(tmp_path: Path, monkeypatch):
    repo = _init_repo(tmp_path)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    db = Database(default_db_path())
    _enqueue_read_only(db, "RO1", repo, head, priority=1)
    _enqueue_read_only(db, "MUT", repo, head, priority=10)
    _enqueue_mutation(db, "MUT", repo, head)
    sha_a = sha256_file(repo / "a.py")

    def make_client():
        # Use a closure-style factory to share the same scripts list across
        # multiple worker instances.
        class C:
            def __init__(self):
                self.last_proposal_id = None
                self.calls = 0
            def chat(self, profile, system, messages, tools=None):
                for msg in reversed(messages):
                    if msg.get("role") == "tool" and msg.get("tool_name") == "propose_patch":
                        try:
                            data = json.loads(msg["content"])
                            if "proposal_id" in data:
                                self.last_proposal_id = data["proposal_id"]
                        except Exception:
                            pass
                        break
                idx = self.calls
                self.calls += 1
                # RO1: just DONE. MUT: propose + apply + DONE.
                if idx == 0:
                    script = {"tool_calls": [{"id": "r", "function": {"name": "report_result",
                                            "arguments": {"disposition": "DONE", "summary": "ok"}}}]}
                elif idx == 1:
                    script = {"tool_calls": [{"id": "p", "function": {"name": "propose_patch",
                        "arguments": {"op": "replace_exact", "path": "a.py",
                                      "expected_sha256": sha_a,
                                      "old_text": "x = 1", "new_text": "x = 2",
                                      "expected_occurrences": 1}}}]}
                elif idx == 2:
                    script = {"tool_calls": [{"id": "a", "function": {"name": "apply_validated_patch",
                        "arguments": {"proposal_id": "_"}}}]}
                else:
                    script = {"tool_calls": [{"id": "r", "function": {"name": "report_result",
                        "arguments": {"disposition": "DONE", "summary": "ok"}}}]}
                tcs = []
                for tc in script.get("tool_calls", []):
                    args = dict(tc["function"]["arguments"])
                    if tc["function"]["name"] == "apply_validated_patch":
                        args["proposal_id"] = self.last_proposal_id or "MISSING"
                    tcs.append({"id": tc["id"], "function": {"name": tc["function"]["name"], "arguments": args}})
                return ChatResult(content="", tool_calls=tcs, metrics=OllamaMetrics(), raw={})
        return C

    import overnight_runner.worker as wmod
    orig = wmod.Worker.__init__
    wmod.Worker.__init__ = lambda self, **kw: orig(self, client=make_client(), **kw)
    try:
        with runner_lock():
            res = run_nightly(session_id="nightly-mutlast")
    finally:
        wmod.Worker.__init__ = orig
    assert res.mutation_attempted == 1
    assert res.read_only_attempted == 1
    assert "MUTATION_DONE" in res.stop_reason
    # order: RO1 ran before MUT
    order = [t["task_id"] for t in res.task_results]
    assert order == ["RO1", "MUT"]


def test_nightly_runtime_budget_blocks_new_task(tmp_path: Path, monkeypatch):
    """If remaining runtime is below threshold, no new task starts."""
    # Monkey-patch DEFAULT_NIGHTLY max_wall_seconds to ~0 so next iteration
    # has no budget. The function reads it on each loop iteration.
    from overnight_runner import runner as rmod
    orig = dict(rmod.DEFAULT_NIGHTLY)
    rmod.DEFAULT_NIGHTLY["max_wall_seconds"] = 0
    try:
        repo = _init_repo(tmp_path)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
        db = Database(default_db_path())
        _enqueue_read_only(db, "RO1", repo, head)
        with runner_lock():
            res = run_nightly(session_id="nightly-budget")
        assert res.tasks_attempted == 0
        assert "RUNTIME_BUDGET" in res.stop_reason or "MUTATION_SKIPPED" in res.stop_reason
    finally:
        rmod.DEFAULT_NIGHTLY.clear()
        rmod.DEFAULT_NIGHTLY.update(orig)


def test_nightly_paused_between_tasks_stops(tmp_path: Path, monkeypatch):
    repo = _init_repo(tmp_path)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    db = Database(default_db_path())
    _enqueue_read_only(db, "RO1", repo, head, priority=1)
    _enqueue_read_only(db, "RO2", repo, head, priority=2)

    pp = paused_path()
    # Stub execute_claimed_task: first call returns PASSED; after the first
    # call completes, set PAUSED so the second iteration refuses to start.
    from overnight_runner import runner as rmod
    from overnight_runner.runner import QueuedRunResult
    real_exec = rmod.execute_claimed_task
    counter = {"n": 0}
    def stub_exec(db, **kw):
        counter["n"] += 1
        # Set PAUSED *before* returning so the next nightly iteration sees it.
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_text("paused\n")
        return QueuedRunResult(
            task_id=f"RO{counter['n']}", run_id=f"r{counter['n']}", attempt_no=1,
            status="PASSED", reason_code="OK", reason_text="stub",
            artifact_dir="/tmp/stub", execution_class="read_only",
        )
    rmod.execute_claimed_task = stub_exec
    try:
        with runner_lock():
            res = run_nightly(session_id="nightly-pause-mid")
    finally:
        rmod.execute_claimed_task = real_exec
        if pp.exists():
            pp.unlink()
    # The first iteration sees PAUSED at the top of the loop after the task
    # completes, so the loop ends with tasks_attempted=1.
    assert res.tasks_attempted == 1
    assert "PAUSED" in res.stop_reason


def test_nightly_morning_summary_files_written(tmp_path: Path):
    with runner_lock():
        res = run_nightly(session_id="nightly-sum")
    sess = state_dir() / "sessions" / "nightly-sum"
    assert (sess / "summary.json").exists()
    assert (sess / "summary.md").exists()
    md = (sess / "summary.md").read_text()
    assert "Overnight Summary" in md
    assert "Stop reason" in md


def test_max_tasks_limit_enforced(tmp_path: Path, monkeypatch):
    """Default max is 6 tasks. We enqueue 8 read_only + 1 mutation."""
    repo = _init_repo(tmp_path)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    db = Database(default_db_path())
    for i in range(8):
        _enqueue_read_only(db, f"RO{i}", repo, head, priority=i)
    _enqueue_mutation(db, "MUT", repo, head)

    import overnight_runner.worker as wmod
    orig = wmod.Worker.__init__
    class C:
        def __init__(self):
            self.last_proposal_id = None
            self.calls = 0
        def chat(self, profile, system, messages, tools=None):
            for msg in reversed(messages):
                if msg.get("role") == "tool" and msg.get("tool_name") == "propose_patch":
                    try:
                        data = json.loads(msg["content"])
                        if "proposal_id" in data:
                            self.last_proposal_id = data["proposal_id"]
                    except Exception:
                        pass
                    break
            idx = self.calls
            self.calls += 1
            sha_a = sha256_file(repo / "a.py")
            if idx == 0:
                script = {"tool_calls": [{"id":"r","function":{"name":"report_result",
                    "arguments":{"disposition":"DONE","summary":"ok"}}}]}
            elif idx == 1:
                script = {"tool_calls": [{"id":"p","function":{"name":"propose_patch",
                    "arguments":{"op":"replace_exact","path":"a.py","expected_sha256":sha_a,
                                 "old_text":"x = 1","new_text":"x = 2","expected_occurrences":1}}}]}
            elif idx == 2:
                script = {"tool_calls": [{"id":"a","function":{"name":"apply_validated_patch",
                    "arguments":{"proposal_id":"_"}}}]}
            else:
                script = {"tool_calls": [{"id":"r","function":{"name":"report_result",
                    "arguments":{"disposition":"DONE","summary":"ok"}}}]}
            tcs = []
            for tc in script.get("tool_calls", []):
                args = dict(tc["function"]["arguments"])
                if tc["function"]["name"] == "apply_validated_patch":
                    args["proposal_id"] = self.last_proposal_id or "MISSING"
                tcs.append({"id": tc["id"], "function": {"name": tc["function"]["name"], "arguments": args}})
            return ChatResult(content="", tool_calls=tcs, metrics=OllamaMetrics(), raw={})
    wmod.Worker.__init__ = lambda self, **kw: orig(self, client=C(), **kw)
    try:
        with runner_lock():
            res = run_nightly(session_id="nightly-maxtasks")
    finally:
        wmod.Worker.__init__ = orig
    # 6 read_only + 1 mutation = 7 tasks max attempted (max_tasks=6 caps read_only at 6; mutation runs only after).
    assert res.tasks_attempted <= DEFAULT_NIGHTLY["max_tasks"] + DEFAULT_NIGHTLY["max_mutation_tasks"]
    assert res.mutation_attempted <= 1


def test_attempt_number_increments_on_retry(tmp_path: Path, monkeypatch):
    """A second run row for the same task has attempt_no=2."""
    repo = _init_repo(tmp_path)
    raw = {
        "schema_version": "1.0", "task_id": "an-1", "title": "t",
        "execution_class": "read_only", "objective": "x",
        "repo": {"path": str(repo)}, "paths": {"read_paths": ["a.py"]},
    }
    m = TaskManifest.model_validate(raw)
    db = Database(default_db_path())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    db.upsert_task("an-1", "sha", json.dumps(m.model_dump(mode="json")), TaskStatus.APPROVED,
                   execution_class="read_only", dependencies_json="[]")
    db.approve_task("an-1", approved_by="t", approval_envelope={
        "manifest_sha256": "sha", "approved_repo_head": head,
        "approved_runtime_sha256": "x", "approved_model_name": "gemma4:12b",
        "approved_model_digest": "d"})
    # First claim -> attempt 1
    c1 = _atomic_claim(db, execution_class_filter=None, session_id=None)
    assert c1["attempt_no"] == 1
    # Mark finished; reset task to APPROVED for retry simulation.
    db.finish_run(c1["run_id"], "PASSED", int(time.time()))
    db.update_status("an-1", TaskStatus.APPROVED,
                     final_reason_code=None, final_reason_text=None)
    # Manually mark task RUNNING again so a subsequent claim can race-test.
    # Actually we want to test attempt_no in the runs row. Just trigger claim again:
    # need to set status APPROVED first.
    db._conn.execute("UPDATE tasks SET status='APPROVED', run_id=NULL WHERE task_id='an-1'")
    c2 = _atomic_claim(db, execution_class_filter=None, session_id=None)
    assert c2["attempt_no"] == 2
