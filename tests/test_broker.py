"""Broker + proposal + atomic write tests.

Tests run against temporary git repositories, NOT Hoard & Havoc.
"""
import os
import subprocess
from pathlib import Path

import pytest

from overnight_runner.broker import (
    Broker,
    SafetyError,
    default_registry,
)
from overnight_runner.safety import (
    git_commit_all,
    git_init_empty,
    sha256_file,
)
from overnight_runner.schemas import (
    CreateFileArgs,
    ReplaceExactArgs,
    ReplaceFileArgs,
    RunCommandArgs,
    ToolCall,
)


def _init_repo(tmp_path: Path) -> Path:
    git_init_empty(tmp_path)
    return tmp_path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = _init_repo(tmp_path)
    (repo / "hello.py").write_text("def hello():\n    return 'old'\n")
    (repo / "test_hello.py").write_text("from hello import hello\nassert hello() == 'old'\n")
    git_commit_all(repo, "init")
    return repo


def _broker(repo: Path, **kw) -> Broker:
    return Broker(
        repo_root=repo,
        allowed_write_paths=["hello.py"],
        allowed_create_paths=["new.py"],
        allowed_read_paths=["hello.py", "test_hello.py"],
        max_tool_result_bytes=64_000,
        **kw,
    )


def test_read_exact(repo):
    b = _broker(repo)
    res = b.read_exact("hello.py")
    assert res["path"] == "hello.py"
    assert res["sha256"] == sha256_file(repo / "hello.py")
    assert "def hello()" in res["content"]


def test_read_unauthorised_path(repo):
    b = _broker(repo)
    with pytest.raises(SafetyError):
        b.read_exact("not_allowed.py")


def test_propose_replace_exact_applies(repo):
    b = _broker(repo)
    sha = sha256_file(repo / "hello.py")
    call = ToolCall(call_id="c1", args=ReplaceExactArgs(
        path="hello.py",
        expected_sha256=sha,
        old_text="return 'old'",
        new_text="return 'new'",
        expected_occurrences=1,
    ))
    out = b.handle(call)
    pid = out["proposal_id"]
    assert out["changed_lines"] >= 2
    assert "preview_diff" in out
    # Not applied yet.
    assert (repo / "hello.py").read_text().endswith("return 'old'\n")
    # Apply.
    b.apply_proposal(pid)
    assert (repo / "hello.py").read_text().endswith("return 'new'\n")


def test_stale_sha_denied(repo):
    b = _broker(repo)
    call = ToolCall(call_id="c1", args=ReplaceExactArgs(
        path="hello.py",
        expected_sha256="0" * 64,
        old_text="return 'old'",
        new_text="return 'new'",
    ))
    with pytest.raises(SafetyError):
        b.handle(call)


def test_duplicate_old_text_denied(repo):
    # Make hello.py contain duplicate.
    (repo / "hello.py").write_text("'old'\n'old'\n")
    b = _broker(repo)
    sha = sha256_file(repo / "hello.py")
    call = ToolCall(call_id="c1", args=ReplaceExactArgs(
        path="hello.py",
        expected_sha256=sha,
        old_text="'old'",
        new_text="'new'",
        expected_occurrences=1,
    ))
    with pytest.raises(SafetyError):
        b.handle(call)


def test_propose_create_file(repo):
    b = _broker(repo)
    call = ToolCall(call_id="c1", args=CreateFileArgs(
        path="new.py",
        new_content="X = 1\n",
    ))
    out = b.handle(call)
    b.apply_proposal(out["proposal_id"])
    assert (repo / "new.py").read_text() == "X = 1\n"


def test_create_file_on_existing_denied(repo):
    b = _broker(repo)
    call = ToolCall(call_id="c1", args=CreateFileArgs(
        path="hello.py",
        new_content="X = 1\n",
    ))
    with pytest.raises(SafetyError):
        b.handle(call)


def test_run_command_id_runs_safe_fixture(repo):
    reg = default_registry()
    # Register a simple echo-style command that returns success.
    from overnight_runner.broker import CommandSpec
    reg.register(CommandSpec("true_command", ["true"], "repo", 10, "none"))
    b = Broker(repo_root=repo, registry=reg,
               allowed_write_paths=["hello.py"],
               model_allowed_command_ids=["true_command"],
               max_tool_result_bytes=10_000)
    call = ToolCall(call_id="c1", args=RunCommandArgs(command_id="true_command"))
    out = b.handle(call)
    assert out["exit_code"] == 0


def test_unknown_command_denied(repo):
    b = _broker(repo)
    call = ToolCall(call_id="c1", args=RunCommandArgs(command_id="definitely_not_registered"))
    with pytest.raises(SafetyError):
        b.handle(call)


def test_dirty_worktree_detected(repo):
    # Simulate a dirty tracked file (modify-then-not-commit).
    (repo / "hello.py").write_text("MODIFIED\n")
    from overnight_runner.safety import git_is_clean
    assert not git_is_clean(repo)


def test_dirty_worktree_blocks_execution(repo):
    """source_mutation + dirty tree -> run() must terminate before Ollama is invoked."""
    (repo / "hello.py").write_text("MODIFIED\n")
    from overnight_runner.schemas import TaskManifest
    from overnight_runner.worker import Worker
    raw = {
        "schema_version": "1.0",
        "task_id": "x",
        "title": "x",
        "execution_class": "source_mutation",
        "objective": "x",
        "repo": {"path": str(repo), "require_clean_tree": True},
        "paths": {"write_paths": ["hello.py"]},
    }
    m = TaskManifest.model_validate(raw)
    called = {"chat": 0}
    class NoChat:
        def chat(self, profile, system, messages, tools=None):
            called["chat"] += 1
            raise AssertionError("Ollama must not be called when binding fails")
    w = Worker(client=NoChat())
    res = w.run(m)
    assert res.status == "BLOCKED"
    assert res.reason_code == "APPROVAL_DIRTY_WORKTREE"
    assert called["chat"] == 0


def test_git_escape_denied(repo):
    b = _broker(repo)
    call = ToolCall(call_id="c1", args=CreateFileArgs(
        path=".git/evil",
        new_content="x",
    ))
    with pytest.raises(SafetyError):
        b.handle(call)


def test_path_traversal_denied(repo):
    b = _broker(repo)
    call = ToolCall(call_id="c1", args=CreateFileArgs(
        path="../escape.py",
        new_content="x",
    ))
    with pytest.raises(SafetyError):
        b.handle(call)


def test_apply_proposal_idempotent_once(repo):
    b = _broker(repo)
    sha = sha256_file(repo / "hello.py")
    call = ToolCall(call_id="c1", args=ReplaceExactArgs(
        path="hello.py", expected_sha256=sha,
        old_text="return 'old'", new_text="return 'v2'",
    ))
    pid = b.handle(call)["proposal_id"]
    b.apply_proposal(pid)
    # Second apply fails: proposal is single-use.
    with pytest.raises(SafetyError):
        b.apply_proposal(pid)


def test_unexpected_changed_path_caught(repo):
    """If a proposal was created against hello.py, but a different path was
    somehow introduced, apply must still pin to the proposal path."""
    b = _broker(repo)
    sha = sha256_file(repo / "hello.py")
    call = ToolCall(call_id="c1", args=ReplaceExactArgs(
        path="hello.py", expected_sha256=sha,
        old_text="return 'old'", new_text="return 'v2'",
    ))
    pid = b.handle(call)["proposal_id"]
    # Mutate file externally after proposal -> apply should fail.
    (repo / "hello.py").write_text("CHANGED\n")
    with pytest.raises(SafetyError):
        b.apply_proposal(pid)


def test_no_git_commit_made_by_broker(repo):
    before = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    b = _broker(repo)
    sha = sha256_file(repo / "hello.py")
    out = b.handle(ToolCall(call_id="c1", args=ReplaceExactArgs(
        path="hello.py", expected_sha256=sha,
        old_text="return 'old'", new_text="return 'v2'",
    )))
    b.apply_proposal(out["proposal_id"])
    after = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert before == after
