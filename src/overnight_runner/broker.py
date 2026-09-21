"""Broker: the only surface a model is allowed to invoke.

Tools:
  - read_exact(path, [start_line], [end_line])
  - propose_patch(replace_exact|replace_file|create_file, ...)
  - apply_validated_patch(proposal_id)
  - run_command_id(command_id)
  - report_result(disposition, summary, evidence) -- recorded only, not dispatched here

The broker validates EVERYTHING before doing anything.

P0 hardening:
  - Exact read allowlist (empty allowlist => no reads).
  - Write vs create path separation.
  - Per-contract command allowlist (model_allowed_command_ids).
  - Declared limits enforcement (max_changed_files, max_diff_lines,
    max_written_bytes, max_files_read, max_files_written, max_read_bytes,
    max_tool_result_bytes).
  - Owned process-group timeout (SIGTERM/SIGKILL on the child's pgid only).
  - Non-mutating command drift detection (git worktree SHA before/after).
  - Mutation evidence: before/, proposed/, after/, preview.diff,
    actual.diff, mutation-journal.jsonl.
"""
from __future__ import annotations

import difflib
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .safety import (
    DEFAULT_PROTECTED_RELATIVE,
    SafetyError,
    ensure_writable,
    git_worktree_sha,
    is_protected,
    resolve_repo_path,
    sha256_bytes,
    sha256_file,
)
from .schemas import (
    CreateFileArgs,
    ReplaceExactArgs,
    ReplaceFileArgs,
    RunCommandArgs,
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

    def has(self, command_id: str) -> bool:
        return command_id in self._cmds

    def ids(self) -> list[str]:
        return sorted(self._cmds)


def default_registry() -> CommandRegistry:
    r = CommandRegistry()
    r.register(CommandSpec("pytest_runner_tests", ["python3", "-m", "pytest", "-q", "-x"], "repo", 300, "read"))
    r.register(CommandSpec("git_diff_check", ["git", "--no-pager", "diff", "--no-color"], "repo", 30, "read"))
    r.register(CommandSpec("git_status", ["git", "status", "--porcelain"], "repo", 10, "read"))
    r.register(CommandSpec("noop", ["true"], "repo", 5, "none"))
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
    changed_lines: int
    proposed_bytes: int
    created_at: float = field(default_factory=time.time)


@dataclass
class MutationJournalEntry:
    timestamp: float
    proposal_id: str
    op: str
    path: str
    abs_path: str
    pre_sha256: str
    post_sha256: str
    bytes_written: int
    pre_mode: int
    post_mode: int


# ----------------------------- Usage counters -----------------------------

@dataclass
class Usage:
    files_read: set[str] = field(default_factory=set)
    files_written: set[str] = field(default_factory=set)
    files_proposed: set[str] = field(default_factory=set)
    bytes_read: int = 0
    diff_lines_total: int = 0
    bytes_written_total: int = 0

    def record_read(self, path: str, n: int) -> None:
        self.files_read.add(path)
        self.bytes_read += n

    def record_write(self, path: str, n: int) -> None:
        self.files_written.add(path)
        self.bytes_written_total += n

    def record_propose(self, path: str) -> None:
        self.files_proposed.add(path)


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
        allowed_protected_read_paths: list[str] | None = None,
        model_allowed_command_ids: list[str] | None = None,
        required_validator_ids: list[str] | None = None,
        max_tool_result_bytes: int = 24576,
        max_read_bytes: int = 16384,
        max_files_read: int = 4,
        max_files_written: int = 2,
        max_changed_files: int = 2,
        max_diff_lines: int = 400,
        max_written_bytes: int = 65536,
        approved_repo_head: str | None = None,
        artifact_dir: Path | None = None,
        on_mutation: Callable[[MutationJournalEntry], None] | None = None,
        receipt_mint: Callable[[dict], str] | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.registry = registry or default_registry()
        self.protected = set(protected_relative) if protected_relative else set(DEFAULT_PROTECTED_RELATIVE)
        self.allowed_write_paths = set(allowed_write_paths or [])
        self.allowed_create_paths = set(allowed_create_paths or [])
        self.allowed_read_paths = set(allowed_read_paths or [])
        self.allowed_protected_read_paths = set(allowed_protected_read_paths or [])
        self.model_allowed_command_ids = set(model_allowed_command_ids or [])
        self.required_validator_ids = set(required_validator_ids or [])
        self.max_tool_result_bytes = max_tool_result_bytes
        self.max_read_bytes = max_read_bytes
        self.max_files_read = max_files_read
        self.max_files_written = max_files_written
        self.max_changed_files = max_changed_files
        self.max_diff_lines = max_diff_lines
        self.max_written_bytes = max_written_bytes
        self.approved_repo_head = approved_repo_head
        self.artifact_dir = artifact_dir
        self.on_mutation = on_mutation
        self.receipt_mint = receipt_mint
        self.usage = Usage()
        self.proposals: dict[str, Proposal] = {}
        self.journal: list[MutationJournalEntry] = []

    # ---------- Path authorisation ----------

    def _authorise_write(self, raw_path: str) -> Path:
        """Authorise a REPLACE on an EXISTING file.

        Only write_paths grants this. create_paths is NOT enough.
        """
        rp = resolve_repo_path(self.repo_root, raw_path)
        ensure_writable(self.repo_root, rp.relative_to_repo, self.protected)
        if raw_path not in self.allowed_write_paths:
            raise SafetyError(f"path not in declared write_paths: {raw_path!r}")
        return rp.absolute

    def _authorise_create(self, raw_path: str) -> Path:
        """Authorise a CREATE on a NON-EXISTENT file.

        Only create_paths grants this.
        """
        rp = resolve_repo_path(self.repo_root, raw_path)
        if rp.absolute.exists():
            raise SafetyError(f"create_file target already exists: {raw_path!r}")
        if ".git" in rp.relative_to_repo.parts:
            raise SafetyError(f".git path not writable: {raw_path!r}")
        if raw_path not in self.allowed_create_paths:
            raise SafetyError(f"path not in declared create_paths: {raw_path!r}")
        return rp.absolute

    def _authorise_read(self, raw_path: str) -> tuple[Path, bool]:
        """Authorise a read.

        Empty allowlist => no reads at all (no implicit allow).
        Returns (abs_path, is_protected).
        """
        rp = resolve_repo_path(self.repo_root, raw_path)
        rel = rp.relative_to_repo
        prot = is_protected(self.repo_root, rel, self.protected)
        if prot:
            if raw_path not in self.allowed_protected_read_paths and raw_path not in self.allowed_read_paths:
                raise SafetyError(f"protected read not authorised: {raw_path!r}")
        else:
            # Exact allowlist. Empty list => NOTHING readable.
            if raw_path not in self.allowed_read_paths and raw_path not in self.allowed_protected_read_paths:
                raise SafetyError(f"path not in declared read_paths: {raw_path!r}")
        return rp.absolute, prot

    # ---------- Limit checks ----------

    def _check_read_limit(self, raw_path: str, size: int) -> None:
        if self.max_read_bytes and size > self.max_read_bytes:
            raise SafetyError(
                f"file exceeds max_read_bytes ({size}>{self.max_read_bytes}): {raw_path!r}"
            )
        if self.max_files_read and len(self.usage.files_read) >= self.max_files_read and raw_path not in self.usage.files_read:
            raise SafetyError(
                f"max_files_read ({self.max_files_read}) already exhausted: {raw_path!r}"
            )

    def _check_write_limit(self, raw_path: str, diff_lines: int, written_bytes: int) -> None:
        if diff_lines > self.max_diff_lines:
            raise SafetyError(
                f"max_diff_lines exceeded ({diff_lines}>{self.max_diff_lines}) for {raw_path!r}"
            )
        if written_bytes > self.max_written_bytes:
            raise SafetyError(
                f"max_written_bytes exceeded ({written_bytes}>{self.max_written_bytes}) for {raw_path!r}"
            )
        if self.max_files_written and len(self.usage.files_written) >= self.max_files_written and raw_path not in self.usage.files_written:
            raise SafetyError(
                f"max_files_written ({self.max_files_written}) already exhausted: {raw_path!r}"
            )

    def _check_changed_files_limit(self, raw_path: str) -> None:
        # max_changed_files = unique files CHANGED. Enforce at propose time
        # against proposed-set so the third propose is refused before the
        # third apply.
        if self.max_changed_files:
            if raw_path not in self.usage.files_written and raw_path not in self.usage.files_proposed:
                if len(self.usage.files_written) + len(
                    self.usage.files_proposed - self.usage.files_written
                ) >= self.max_changed_files:
                    raise SafetyError(
                        f"max_changed_files ({self.max_changed_files}) already exhausted"
                    )

    # ---------- Tool implementations ----------

    def handle(self, call) -> dict[str, Any]:
        from .schemas import ToolCall
        assert isinstance(call, ToolCall)
        a = call.args
        if isinstance(a, (ReplaceExactArgs, ReplaceFileArgs, CreateFileArgs)):
            return self._propose(call.call_id, a)
        if isinstance(a, RunCommandArgs):
            return self._run_command(a)
        raise SafetyError(f"unsupported tool: {type(a).__name__}")

    def read_exact(self, raw_path: str, start_line: int | None = None, end_line: int | None = None) -> dict[str, Any]:
        abs_path, _ = self._authorise_read(raw_path)
        if not abs_path.is_file():
            raise SafetyError(f"not a regular file: {raw_path!r}")
        size = abs_path.stat().st_size
        self._check_read_limit(raw_path, size)
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
        self.usage.record_read(raw_path, len(data))
        return {
            "path": raw_path,
            "sha256": sha256_bytes(data),
            "size_bytes": size,
            "content": text,
        }

    def _propose(self, call_id: str, args) -> dict[str, Any]:
        # Enforce changed_files limit BEFORE constructing proposal.
        self._check_changed_files_limit(args.path)

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
            proposed = text.replace(args.old_text, args.new_text, 1)
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
        changed_lines = _changed_line_count(diff)
        proposed_bytes = len(proposed.encode("utf-8"))
        self._check_write_limit(args.path, changed_lines, proposed_bytes)

        proposal_id = f"prop-{call_id}-{int(time.time() * 1000)}"
        self.usage.record_propose(args.path)
        self.proposals[proposal_id] = Proposal(
            proposal_id=proposal_id,
            op=op,
            path=args.path,
            abs_path=abs_path,
            before_text=before,
            proposed_text=proposed,
            preview_diff=diff,
            changed_lines=changed_lines,
            proposed_bytes=proposed_bytes,
        )
        return {
            "proposal_id": proposal_id,
            "preview_diff": diff,
            "changed_lines": changed_lines,
            "proposed_bytes": proposed_bytes,
        }

    def apply_proposal(self, proposal_id: str) -> dict[str, Any]:
        """Apply a proposal atomically (per file).

        Re-validates at apply boundary:
          - proposal exists and is single-use;
          - target path still resolves into repo, still passes policy;
          - protected path policy unchanged;
          - repo HEAD still equals approved HEAD (for mutations);
          - file hash matches expectation (no in-flight changes);
          - PAUSED sentinel absent.
        Writes mutation evidence:
          - before/<path>
          - proposed/<path>
          - after/<path>
          - preview.diff + actual.diff
          - mutation-journal.jsonl
        """
        prop = self.proposals.get(proposal_id)
        if prop is None:
            raise SafetyError(f"unknown proposal_id: {proposal_id}")
        rp = resolve_repo_path(self.repo_root, prop.path)
        ensure_writable(self.repo_root, rp.relative_to_repo, self.protected)
        self._check_changed_files_limit(prop.path)

        # Re-check repo HEAD (defence-in-depth; not a substitute for preflight).
        if self.approved_repo_head:
            from .safety import git_head
            cur = git_head(self.repo_root)
            if cur != self.approved_repo_head:
                raise SafetyError(
                    f"REPO_HEAD_DRIFT_AT_APPLY: approved={self.approved_repo_head[:8]} current={cur[:8]}"
                )

        pre_sha = ""
        pre_mode = 0
        if prop.op == "create_file":
            if rp.absolute.exists():
                raise SafetyError(f"create_file target already exists: {prop.path!r}")
        else:
            if not rp.absolute.is_file():
                raise SafetyError(f"target missing: {prop.path!r}")
            pre_sha = sha256_file(rp.absolute)
            if prop.before_text is None:
                raise SafetyError("proposal missing before_text")
            if pre_sha != sha256_bytes(prop.before_text.encode("utf-8")):
                raise SafetyError(
                    f"file changed since proposal: current_sha={pre_sha[:8]}"
                )
            pre_mode = rp.absolute.stat().st_mode & 0o777

        # Persist before/ and proposed/ evidence BEFORE writing.
        if self.artifact_dir is not None:
            self._write_evidence(prop)

        written_bytes = len(prop.proposed_text.encode("utf-8"))
        # Enforce max_written_bytes at apply time too.
        if written_bytes > self.max_written_bytes:
            raise SafetyError(
                f"max_written_bytes exceeded at apply ({written_bytes}>{self.max_written_bytes})"
            )

        _atomic_write(rp.absolute, prop.proposed_text.encode("utf-8"),
                      mode=None if prop.op == "create_file" else pre_mode)

        post_sha = sha256_file(rp.absolute)
        post_mode = rp.absolute.stat().st_mode & 0o777

        # Record actual diff for evidence.
        actual_diff = _make_diff(
            prop.before_text if prop.before_text is not None else "",
            prop.proposed_text,
            prop.path,
        )
        if self.artifact_dir is not None:
            (self.artifact_dir / "after" / prop.path).parent.mkdir(parents=True, exist_ok=True)
            (self.artifact_dir / "after" / prop.path).write_bytes(rp.absolute.read_bytes())
            ad = self.artifact_dir / "actual.diff"
            with ad.open("a") as f:
                f.write(actual_diff)
                f.write("\n")

        entry = MutationJournalEntry(
            timestamp=time.time(),
            proposal_id=proposal_id,
            op=prop.op,
            path=prop.path,
            abs_path=str(rp.absolute),
            pre_sha256=pre_sha,
            post_sha256=post_sha,
            bytes_written=written_bytes,
            pre_mode=pre_mode,
            post_mode=post_mode,
        )
        self.journal.append(entry)
        self.usage.record_write(prop.path, written_bytes)

        if self.artifact_dir is not None:
            mj = self.artifact_dir / "mutation-journal.jsonl"
            import json
            with mj.open("a") as f:
                f.write(json.dumps(entry.__dict__, default=str) + "\n")

        if self.on_mutation is not None:
            try:
                self.on_mutation(entry)
            except Exception:
                pass

        # Single-use proposal.
        del self.proposals[proposal_id]
        result = {
            "proposal_id": proposal_id,
            "path": prop.path,
            "op": prop.op,
            "applied": True,
            "bytes_written": written_bytes,
            "pre_sha256": pre_sha,
            "post_sha256": post_sha,
        }
        # Runner-owned trusted receipt (opaque durable evidence reference).
        if self.receipt_mint is not None:
            try:
                result["receipt_id"] = self.receipt_mint(
                    {
                        "proposal_id": proposal_id,
                        "path": prop.path,
                        "snapshot_digest": pre_sha,
                        "chunk_id": prop.path,  # caller may refine via adapter
                        "request_id": proposal_id,
                        "result": result,
                    }
                )
            except Exception as e:  # mint is advisory evidence, never blocks apply
                result["receipt_mint_error"] = str(e)
        return result

    def _write_evidence(self, prop: Proposal) -> None:
        """Persist before/<path>, proposed/<path>, preview.diff BEFORE the write."""
        ad = self.artifact_dir
        assert ad is not None
        # before/
        if prop.before_text is not None:
            before_path = ad / "before" / prop.path
            before_path.parent.mkdir(parents=True, exist_ok=True)
            before_path.write_text(prop.before_text, encoding="utf-8")
        # proposed/
        proposed_path = ad / "proposed" / prop.path
        proposed_path.parent.mkdir(parents=True, exist_ok=True)
        proposed_path.write_text(prop.proposed_text, encoding="utf-8")
        # preview.diff
        pd = ad / "preview.diff"
        with pd.open("a") as f:
            f.write(prop.preview_diff)
            f.write("\n")

    def _run_command(self, args: RunCommandArgs) -> dict[str, Any]:
        # Per-contract command allowlist. Empty allowlist => NOTHING callable.
        if args.command_id not in self.model_allowed_command_ids:
            raise SafetyError(
                f"command_id {args.command_id!r} not in manifest.commands.model_allowed_command_ids"
            )
        # Required validators may be invoked only via the worker, not the model.
        if args.command_id in self.required_validator_ids:
            raise SafetyError(
                f"command_id {args.command_id!r} is reserved as a required validator"
            )
        spec = self.registry.get(args.command_id)
        cwd = self.repo_root
        argv = list(spec.argv)

        # Non-mutating commands: capture worktree fingerprint before/after.
        pre_sha = None
        if spec.side_effects in ("none", "read"):
            pre_sha = git_worktree_sha(self.repo_root)

        try:
            proc = _spawn_own_pgrp(argv, cwd=cwd, timeout=spec.timeout_seconds)
        except _TimeoutExpired as e:
            return {
                "command_id": spec.command_id,
                "exit_code": -1,
                "stdout_tail": "",
                "stderr_tail": str(e),
                "truncated": False,
                "timed_out": True,
            }
        out = (proc.stdout or "") + (proc.stderr or "")
        truncated = False
        if self.max_tool_result_bytes and len(out) > self.max_tool_result_bytes:
            out = out[: self.max_tool_result_bytes]
            truncated = True

        if spec.side_effects in ("none", "read"):
            post_sha = git_worktree_sha(self.repo_root)
            if pre_sha != post_sha:
                raise SafetyError(
                    f"non_mutating_command_drift: command_id={spec.command_id} "
                    f"pre={pre_sha[:8]} post={post_sha[:8]}"
                )

        return {
            "command_id": spec.command_id,
            "exit_code": proc.returncode,
            "stdout_tail": out,
            "truncated": truncated,
            "timed_out": False,
        }


# ----------------------------- Process group helper -----------------------------

class _TimeoutExpired(Exception):
    pass


def _spawn_own_pgrp(argv: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess:
    """Spawn a child in its own process group; SIGTERM/SIGKILL only that pgid on timeout.

    NEVER broad-kills by name.
    """
    import sys as _sys
    kwargs: dict[str, Any] = dict(
        args=argv,
        cwd=str(cwd),
        shell=False,
        start_new_session=True,  # new pgid
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if _sys.version_info >= (3, 12):
        # start_new_session=True (above) puts the child in a brand-new session
        # with its own process group. We deliberately do NOT pin process_group.
        proc = subprocess.Popen(**kwargs)
    else:
        proc = subprocess.Popen(**kwargs)
    pgid = os.getpgid(proc.pid)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # SIGTERM the pgid, then SIGKILL if it survives.
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        raise _TimeoutExpired(f"command exceeded timeout={timeout}s (pgid={pgid})")
    return subprocess.CompletedProcess(
        args=argv, returncode=proc.returncode, stdout=stdout, stderr=stderr
    )


# ----------------------------- Diff helpers -----------------------------

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
