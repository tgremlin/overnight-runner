"""Integration test: full worker run with a scripted (mock) Ollama client.

Verifies the end-to-end Phase 1 sequence:
  load -> preflight -> Ollama -> tool calls -> proposal -> apply -> DONE -> validators
"""
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from overnight_runner.ollama_client import ChatResult, OllamaMetrics
from overnight_runner.safety import git_commit_all, git_init_empty, sha256_file
from overnight_runner.schemas import TaskManifest
from overnight_runner.worker import Worker


class FakeClient:
    """Replays scripted chat responses and tracks the proposal_id returned by propose_patch."""

    def __init__(self, scripts: list[dict[str, Any]]):
        self.scripts = list(scripts)
        self.calls = 0
        self.last_proposal_id: str | None = None
        self.metrics = OllamaMetrics(prompt_eval_count=12, eval_count=8, total_duration_ns=1_000_000)

    def chat(self, profile, system, messages, tools=None):
        idx = min(self.calls, len(self.scripts) - 1)
        self.calls += 1
        # Track most recent propose_patch proposal_id from tool results.
        # We only consider tool messages whose preceding tool_name was
        # propose_patch. Since our tool message format includes tool_name, we
        # can filter.
        for msg in reversed(messages):
            if msg.get("role") == "tool" and msg.get("tool_name") == "propose_patch":
                try:
                    data = json.loads(msg["content"])
                except Exception:
                    continue
                if isinstance(data, dict) and "proposal_id" in data:
                    self.last_proposal_id = data["proposal_id"]
                    break
        # Patch any apply tool call to use the captured proposal_id.
        script = self.scripts[idx]
        tcs: list[dict[str, Any]] = []
        for tc in script.get("tool_calls", []):
            args = dict(tc["function"]["arguments"])
            if tc["function"]["name"] == "apply_validated_patch":
                args["proposal_id"] = self.last_proposal_id or "MISSING"
            tcs.append({
                "id": tc["id"],
                "function": {"name": tc["function"]["name"], "arguments": args},
            })
        return ChatResult(
            content=script.get("content", ""),
            tool_calls=tcs,
            metrics=self.metrics,
            raw={"ok": True},
        )


def _init_repo_with_hello(tmp_path: Path) -> Path:
    repo = tmp_path
    git_init_empty(repo)
    (repo / "hello.py").write_text('def hello():\n    return "old"\n')
    (repo / "test_hello.py").write_text('from hello import hello\nassert hello() == "old"\n')
    git_commit_all(repo, "init")
    return repo


def _manifest(repo: Path) -> TaskManifest:
    raw = {
        "schema_version": "1.0",
        "task_id": "int-001",
        "title": "flip hello",
        "execution_class": "source_mutation",
        "objective": "Change hello() to return 'v2'; update test.",
        "repo": {"path": str(repo), "require_clean_tree": True},
        "paths": {
            "read_paths": ["hello.py", "test_hello.py"],
            "write_paths": ["hello.py", "test_hello.py"],
        },
        "commands": {"required_validator_ids": ["python_compile"]},
        "limits": {"max_model_turns": 10, "max_tool_calls": 20},
    }
    return TaskManifest.model_validate(raw)


def test_worker_happy_path_end_to_end(tmp_path: Path):
    repo = _init_repo_with_hello(tmp_path)
    m = _manifest(repo)

    sha_hello = sha256_file(repo / "hello.py")
    sha_test = sha256_file(repo / "test_hello.py")

    fake = FakeClient(scripts=[
        {"tool_calls": [{"id": "t1", "function": {"name": "read_exact", "arguments": {"path": "hello.py"}}}]},
        {"tool_calls": [{"id": "t2", "function": {"name": "read_exact", "arguments": {"path": "test_hello.py"}}}]},
        {"tool_calls": [{
            "id": "t3", "function": {"name": "propose_patch", "arguments": {
                "op": "replace_exact",
                "path": "hello.py",
                "expected_sha256": sha_hello,
                "old_text": 'return "old"',
                "new_text": 'return "v2"',
                "expected_occurrences": 1,
            }}
        }]},
        {"tool_calls": [{
            "id": "t4", "function": {"name": "apply_validated_patch", "arguments": {"proposal_id": "_"}}
        }]},
        {"tool_calls": [{
            "id": "t5", "function": {"name": "propose_patch", "arguments": {
                "op": "replace_exact",
                "path": "test_hello.py",
                "expected_sha256": sha_test,
                "old_text": 'assert hello() == "old"',
                "new_text": 'assert hello() == "v2"',
                "expected_occurrences": 1,
            }}
        }]},
        {"tool_calls": [{
            "id": "t6", "function": {"name": "apply_validated_patch", "arguments": {"proposal_id": "_"}}
        }]},
        {"tool_calls": [{
            "id": "t7", "function": {"name": "report_result", "arguments": {"disposition": "DONE", "summary": "done"}}
        }]},
    ])

    w = Worker(client=fake)
    res = w.run(m)
    assert res.status == "PASSED", (res.status, res.reason_code, res.reason_text)
    assert (repo / "hello.py").read_text().endswith('return "v2"\n')
    assert 'assert hello() == "v2"' in (repo / "test_hello.py").read_text()
    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    # Only the init commit
    assert log.count("\n") == 0
    # Artifacts persisted
    assert Path(res.artifacts_dir, "manifest.json").exists()
    assert Path(res.artifacts_dir, "approval.json").exists()
    assert Path(res.artifacts_dir, "transcript.jsonl").exists()
    assert Path(res.artifacts_dir, "proposal.json").exists()
    assert Path(res.artifacts_dir, "result.json").exists()


def test_worker_blocked_when_validator_fails(tmp_path: Path):
    repo = _init_repo_with_hello(tmp_path)
    m = _manifest(repo)
    # Make python_compile fail by requiring a non-existent file pattern.
    # We instead mutate the validator's argv via the broker registry.
    from overnight_runner.broker import CommandRegistry, CommandSpec
    reg = CommandRegistry()
    reg.register(CommandSpec("python_compile", ["python3", "-c", "import sys; sys.exit(1)"], "argv_path", 30, "read"))
    fake = FakeClient(scripts=[
        {"tool_calls": [{"id": "t1", "function": {"name": "report_result", "arguments": {"disposition": "DONE", "summary": "done"}}}]},
    ])
    w = Worker(client=fake, registry=reg)
    res = w.run(m)
    assert res.status == "FAILED"
    assert "VALIDATORS_FAILED" in res.reason_code


def test_worker_review_required_when_model_says_rr(tmp_path: Path):
    repo = _init_repo_with_hello(tmp_path)
    m = _manifest(repo)
    fake = FakeClient(scripts=[
        {"tool_calls": [{"id": "t1", "function": {"name": "report_result", "arguments": {"disposition": "REVIEW_REQUIRED", "summary": "ambiguous"}}}]},
    ])
    w = Worker(client=fake)
    res = w.run(m)
    assert res.status == "REVIEW_REQUIRED"
    assert res.reason_code == "MODEL_REVIEW_REQUIRED"


def test_worker_unreal_editor_blocked(tmp_path: Path):
    repo = _init_repo_with_hello(tmp_path)
    raw = {
        "schema_version": "1.0",
        "task_id": "ue-1",
        "title": "ue",
        "execution_class": "unreal_editor",
        "objective": "x",
        "repo": {"path": str(repo)},
        "unreal": {"editor_access": "editor_automation", "automation_test_names": ["foo"]},
    }
    m = TaskManifest.model_validate(raw)
    fake = FakeClient(scripts=[])
    w = Worker(client=fake)
    res = w.run(m)
    assert res.status == "BLOCKED"
    assert res.reason_code == "UNREAL_EDITOR_NOT_IMPLEMENTED"
