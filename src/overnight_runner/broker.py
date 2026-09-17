"""Broker: the only surface a model is allowed to invoke.

Tools:
  - read_exact(path, [start_line], [end_line])
  - propose_patch(replace_exact|replace_file|create_file, ...)
  - run_command_id(command_id)
  - report_result(disposition, summary, evidence)

The broker validates EVERYTHING before doing anything.
"""
from __future__ import annotations

import difflib
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .safety import (
    DEFAULT_PROTECTED_RELATIVE,
    SafetyError,
    ensure_writable,
    is_protected,
    resolve_repo_path,
    sha256_bytes,
    sha256_file,
)
from .schemas import (
    CreateFileArgs,
    Disposition,
    ReplaceExactArgs,
    ReplaceFileArgs,
    RunCommandArgs,
    ToolCall,
)


# ----------------------------- Command registry -----------------------------

@dataclass(frozen=True)
class CommandSpec:
    command_id: str
    argv: list[str]
    cwd_kind: str  # "repo" | "abs"
    timeout_seconds: int
    side_effects: str  # "none" | "read" | "mutate"


class CommandRegistry:
    def __init__(self) -> None:
        self._cmds: dict[str, CommandSpec] = {}

    def register(self, spec: CommandSpec) -> None:
        if spec.command_id in self._cmds:
            raise ValueError(f"duplicate command_id: {spec.command_id}")
        self._cmds[spec.command_id] = spec

    def get(self, command_id: str) -> CommandSpec:
        if command_id not in self._cmds:
            raise SafetyError(f"unknown command_id: {command_id!r}")
        return self._cmds[command_id]

    def ids(self) -> list[str]:
        return sorted(self._cmds)


def default_registry() -> CommandRegistry:
    r = CommandRegistry()
    # SAFE fixtures only. No arbitrary shell.
    # Validators and helpers. cwd_kind 'repo' means run with cwd=repo_root.
    # Built-in deterministic validators (handled by the worker, not the registry):
    #   python_compile, no_op
    # The registry only contains commands that are spawned via subprocess.
    r.register(CommandSpec("pytest_runner_tests", ["python3", "-m", "pytest", "-q", "-x"], "repo", 300, "read"))
    r.register(CommandSpec("git_diff_check", ["git", "--no-pager", "diff", "--no-color"], "repo", 30, "read"))
    r.register(CommandSpec("git_status", ["git", "status", "--porcelain"], "repo", 10, "read"))
    return r


# ----------------------------- Proposal store -----------------------------

@dataclass
class Proposal:
    proposal_id: str
    op: str
    path: str
    abs_path: Path
    before_text: str | None
    proposed_text: str
    preview_diff: str
    created_at: float = field(default_factory=time.time)


# ----------------------------- Broker -----------------------------

class Broker:
    def __init__(
        self,
        repo_root: Path,
        registry: CommandRegistry | None = None,
        protected_relative: set[str] | None = None,
        allowed_write_paths: list[str] | None = None,
        allowed_create_paths: list[str] | None = None,
        allowed_read_paths: list[str] | None = None,
        max_tool_result_bytes: int = 24576,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.registry = registry or default_registry()
        self.protected = set(protected_relative) if protected_relative else set(DEFAULT_PROTECTED_RELATIVE)
        self.allowed_write_paths = set(allowed_write_paths or [])
        self.allowed_create_paths = set(allowed_create_paths or [])
        self.allowed_read_paths = set(allowed_read_paths or [])
        self.max_tool_result_bytes = max_tool_result_bytes
        self.proposals: dict[str, Proposal] = {}

    # ---------- Path authorisation ----------

    def _authorise_write(self, raw_path: str) -> Path:
        rp = resolve_repo_path(self.repo_root, raw_path)
        ensure_writable(self.repo_root, rp.relative_to_repo, self.protected)
        # Must be in allowed_write_paths (or allowed_create_paths for creates)
        if raw_path not in self.allowed_write_paths and raw_path not in self.allowed_create_paths:
            raise SafetyError(f"path not in declared write/create_paths: {raw_path!r}")
        return rp.absolute

    def _authorise_create(self, raw_path: str) -> Path:
        rp = resolve_repo_path(self.repo_root, raw_path)
        # Cannot create over existing files unless also declared writable (handled separately).
        if rp.absolute.exists():
            raise SafetyError(f"create_file target already exists: {raw_path!r}")
        # .git block
        if ".git" in rp.relative_to_repo.parts:
            raise SafetyError(f".git path not writable: {raw_path!r}")
        if raw_path not in self.allowed_create_paths:
            raise SafetyError(f"path not in declared create_paths: {raw_path!r}")
        return rp.absolute

    def _authorise_read(self, raw_path: str) -> Path:
        rp = resolve_repo_path(self.repo_root, raw_path)
        rel = rp.relative_to_repo.as_posix()
        if is_protected(self.repo_root, rp.relative_to_repo, self.protected):
            # protected reads must be explicitly declared
            if raw_path not in self.allowed_read_paths:
                # allow if in read_paths or protected_read_paths
                if raw_path not in self.allowed_read_paths and rel not in self.allowed_read_paths:
                    raise SafetyError(f"protected read not authorised: {raw_path!r}")
        else:
            if self.allowed_read_paths and raw_path not in self.allowed_read_paths and rel not in self.allowed_read_paths:
                raise SafetyError(f"path not in declared read_paths: {raw_path!r}")
        return rp.absolute

    # ---------- Tool implementations ----------

    def handle(self, call: ToolCall) -> dict[str, Any]:
        a = call.args
        if isinstance(a, ReplaceExactArgs) or isinstance(a, ReplaceFileArgs) or isinstance(a, CreateFileArgs):
            return self._propose(call.call_id, a)
        if isinstance(a, RunCommandArgs):
            return self._run_command(a)
        raise SafetyError(f"unsupported tool: {type(a).__name__}")

    def read_exact(self, raw_path: str, start_line: int | None = None, end_line: int | None = None) -> dict[str, Any]:
        abs_path = self._authorise_read(raw_path)
        if not abs_path.is_file():
            raise SafetyError(f"not a regular file: {raw_path!r}")
        size = abs_path.stat().st_size
        data = abs_path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise SafetyError(f"non-UTF8 file: {raw_path!r}") from e
        if self.max_tool_result_bytes and len(data) > self.max_tool_result_bytes:
            raise SafetyError(
                f"file exceeds max_tool_result_bytes ({len(data)}>{self.max_tool_result_bytes})"
            )
        if start_line is not None or end_line is not None:
            lines = text.splitlines(keepends=True)
            s = start_line or 0
            e = end_line if end_line is not None else len(lines)
            text = "".join(lines[s:e])
        return {
            "path": raw_path,
            "sha256": sha256_bytes(data),
            "size_bytes": size,
            "content": text,
        }

    def _propose(self, call_id: str, args: ReplaceExactArgs | ReplaceFileArgs | CreateFileArgs) -> dict[str, Any]:
        if isinstance(args, ReplaceExactArgs):
            abs_path = self._authorise_write(args.path)
            if not abs_path.is_file():
                raise SafetyError(f"replace_exact target missing: {args.path!r}")
            current = abs_path.read_bytes()
            current_sha = sha256_bytes(current)
            if current_sha != args.expected_sha256:
                raise SafetyError(
                    f"stale file: expected_sha256={args.expected_sha256[:8]} actual={current_sha[:8]}"
                )
            text = current.decode("utf-8")
            count = text.count(args.old_text)
            if count != args.expected_occurrences:
                raise SafetyError(
                    f"old_text occurrences {count} != expected {args.expected_occurrences}"
                )
            proposed = text.replace(args.old_text, args.new_text, 1)  # first occurrence; count guaranteed
            op = "replace_exact"
            before = text
        elif isinstance(args, ReplaceFileArgs):
            abs_path = self._authorise_write(args.path)
            if not abs_path.is_file():
                raise SafetyError(f"replace_file target missing: {args.path!r}")
            current = abs_path.read_bytes()
            current_sha = sha256_bytes(current)
            if current_sha != args.expected_sha256:
                raise SafetyError(
                    f"stale file: expected_sha256={args.expected_sha256[:8]} actual={current_sha[:8]}"
                )
            proposed = args.new_content
            before = current.decode("utf-8", errors="replace")
            op = "replace_file"
        else:  # CreateFileArgs
            abs_path = self._authorise_create(args.path)
            proposed = args.new_content
            before = ""
            op = "create_file"

        diff = _make_diff(before, proposed, args.path)
        proposal_id = f"prop-{call_id}-{int(time.time() * 1000)}"
        self.proposals[proposal_id] = Proposal(
            proposal_id=proposal_id,
            op=op,
            path=args.path,
            abs_path=abs_path,
            before_text=before,
            proposed_text=proposed,
            preview_diff=diff,
        )
        return {
            "proposal_id": proposal_id,
            "preview_diff": diff,
            "changed_lines": _changed_line_count(diff),
        }

    def apply_proposal(self, proposal_id: str) -> dict[str, Any]:
        """Apply a proposal atomically (per file).

        Re-validates: file existence / hash / path / repo state.
        """
        prop = self.proposals.get(proposal_id)
        if prop is None:
            raise SafetyError(f"unknown proposal_id: {proposal_id}")
        rp = resolve_repo_path(self.repo_root, prop.path)
        ensure_writable(self.repo_root, rp.relative_to_repo, self.protected)
        if prop.op == "create_file":
            if rp.absolute.exists():
                raise SafetyError(f"create_file target already exists: {prop.path!r}")
            _atomic_write(rp.absolute, prop.proposed_text.encode("utf-8"), mode=None)
        else:
            if not rp.absolute.is_file():
                raise SafetyError(f"target missing: {prop.path!r}")
            current_sha = sha256_file(rp.absolute)
            # Recompute expected by comparing before_text current hash if recorded.
            if prop.before_text is None:
                raise SafetyError("proposal missing before_text")
            if current_sha != sha256_bytes(prop.before_text.encode("utf-8")):
                raise SafetyError(
                    f"file changed since proposal: current_sha={current_sha[:8]}"
                )
            current_mode = rp.absolute.stat().st_mode & 0o777
            _atomic_write(rp.absolute, prop.proposed_text.encode("utf-8"), mode=current_mode)
        # Single-use proposal
        del self.proposals[proposal_id]
        return {
            "proposal_id": proposal_id,
            "path": prop.path,
            "op": prop.op,
            "applied": True,
        }

    def _run_command(self, args: RunCommandArgs) -> dict[str, Any]:
        spec = self.registry.get(args.command_id)
        cwd = self.repo_root
        argv = list(spec.argv)
        # Allow argv to carry a {path} placeholder? For MVP we keep argv literal.
        proc = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=spec.timeout_seconds,
            shell=False,
            start_new_session=True,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        truncated = False
        if len(out) > self.max_tool_result_bytes:
            out = out[: self.max_tool_result_bytes]
            truncated = True
        return {
            "command_id": spec.command_id,
            "exit_code": proc.returncode,
            "stdout_tail": out,
            "truncated": truncated,
            "timed_out": False,
        }


# ----------------------------- Helpers -----------------------------

def _make_diff(before: str, after: str, path: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff)


def _changed_line_count(diff: str) -> int:
    n = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            n += 1
        elif line.startswith("-") and not line.startswith("---"):
            n += 1
    return n


def _atomic_write(target: Path, data: bytes, mode: int | None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".overnight-", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
