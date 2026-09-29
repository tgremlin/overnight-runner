"""OV6-3 — sandboxed `host-tests` in the Runner path.

The Runner invokes pytest itself, inside a bubblewrap confinement, on a
READ-ONLY copy of the workspace, with the pinned trusted toolchain root bound
read-only. Nothing the tests write can reach the candidate tree (writes land in a
Runner-created scratch copy), the network is unshared, and the process group is
killed on timeout.

How the Runner calls it
-----------------------
`run_host_tests(workspace_dir=..., toolchain_root=..., toolchain_python=...,
timeout_s=...)` returns `HostTestsResult`. `validator_evidence.run_trusted_validators`
uses it when the validator id is `host-tests`, so the completion gate sees
Runner-produced evidence for the sandboxed run as well.

The bwrap argv mirrors the TypeScript runner's rules
(`packages/execution/src/bwrap.ts`): read-only host roots, a tmpfs for /tmp and
HOME, the workspace bound READ-ONLY, exactly one extra read-write bind (the
scratch copy), `--clearenv` with an explicit allowlist, and `--die-with-parent`.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_TOOLCHAIN_ROOT",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "HostTestsResult",
    "build_host_tests_bwrap",
    "cap_output",
    "run_host_tests",
]

#: §OV7-1: captured output is capped; the digest is taken over the CAPPED bytes.
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024

#: §OV7-2: roots that must never be exposed, mirroring the TS runner.
RESERVED_ROOTS = ("/", "/home", "/root", "/mnt", "/run/user")
CHANGE_UNSET = object()


def _is_under(parent: str, child: str) -> bool:
    parent = parent.rstrip("/") or "/"
    return child == parent or child.startswith(parent.rstrip("/") + "/")


def _workspace_refusal(workspace_dir: str, forbidden: tuple[str, ...], home: str) -> str | None:
    """The TS `buildBwrapArgs` workspace rules, in the same order."""
    if not isinstance(workspace_dir, str) or not os.path.isabs(workspace_dir):
        return f"refusing to build a bwrap command with a relative workspaceDir: {workspace_dir!r}"
    ws = os.path.realpath(workspace_dir)
    if ws == "/":
        return "refusing to build a bwrap command that binds / read-write"
    if ws == os.path.realpath(home):
        return "refusing to build a bwrap command that binds the real HOME read-write"
    for f in forbidden:
        # the workspace must not be an ANCESTOR of, or equal to, a forbidden path
        if _is_under(ws, os.path.realpath(f)):
            return f"refusing to build a bwrap command whose workspace is an ancestor of, or equal to, a forbidden path ({f})"
    return None


def _read_only_root_issue(root: str, forbidden: tuple[str, ...], home: str) -> str | None:
    """The TS `readOnlyRootIssue` rules, in the same order."""
    rr = os.path.realpath(root)
    if rr in RESERVED_ROOTS:
        return f"refusing the reserved root {root}"
    real_home = os.path.realpath(home)
    if _is_under(rr, real_home):
        return f"refusing a read-only root that is an ancestor of the real HOME ({root})"
    for f in forbidden:
        if _is_under(rr, f):
            return f"refusing a read-only root that is an ancestor of a forbidden path ({root})"
    if _is_under(real_home, rr):
        return f"refusing a read-only root under the real HOME ({root})"
    for f in forbidden:
        if _is_under(f, rr):
            return f"refusing a read-only root under a forbidden path ({root})"
    return None


def cap_output(data: bytes, cap: int = DEFAULT_MAX_OUTPUT_BYTES) -> tuple[bytes, bool]:
    """§OV7-1 — truncate to `cap` bytes and say whether it was truncated."""
    if len(data) <= cap:
        return data, False
    return data[:cap], True

#: The pinned, read-only toolchain root provisioned in OV3-4.
DEFAULT_TOOLCHAIN_ROOT = "/mnt/ue/Projects/trio-toolchain/venv"
READ_ONLY_ROOTS = ("/usr", "/lib", "/lib64", "/bin", "/sbin")


@dataclass(frozen=True)
class HostTestsResult:
    ok: bool
    outcome: str  # passed | failed | timeout | error
    exit_code: int | None
    stdout: str
    stderr: str
    seconds: float
    scratch_dir: str
    argv: tuple[str, ...]
    #: §OV7-1: whether the captured output hit the cap, and the capped digests.
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_digest: str = ""
    stderr_digest: str = ""


def build_host_tests_bwrap(
    *,
    workspace_dir: str,
    scratch_dir: str,
    toolchain_python: str,
    roots=None,
    forbidden_paths=(),
    home=None,
) -> list[str]:
    """The exact argv the Runner runs for host-tests. Pure; safe to assert on.

    §OV7-2: the path rules mirror the TypeScript runner's builder - a relative
    workspace, `/`, the real HOME, a workspace that is an ancestor of (or equal
    to) a forbidden path, and any read-only root that is reserved, under the real
    HOME, or an ancestor of / under a forbidden path are all REFUSED.
    """
    home = home or os.path.expanduser("~")
    forbidden = tuple(forbidden_paths)
    issue = _workspace_refusal(workspace_dir, forbidden, home)
    if issue is not None:
        raise ValueError(issue)
    if not isinstance(scratch_dir, str) or not os.path.isabs(scratch_dir):
        raise ValueError(f"refusing to build a bwrap command with a relative scratch dir: {scratch_dir!r}")
    if os.path.realpath(scratch_dir) == os.path.realpath(workspace_dir):
        raise ValueError("refusing to build a bwrap command whose scratch dir IS the workspace")
    for root in (roots or READ_ONLY_ROOTS):
        root_issue = _read_only_root_issue(root, forbidden, home)
        if root_issue is not None:
            raise ValueError(root_issue)
    argv = ["bwrap", "--die-with-parent", "--new-session", "--unshare-user", "--unshare-pid", "--unshare-net", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup"]
    for root in (roots or READ_ONLY_ROOTS):
        argv += ["--ro-bind", root, root]
    argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/home/sandbox"]
    # the workspace is READ-ONLY; the only read-write bind is the scratch copy
    argv += ["--ro-bind", workspace_dir, workspace_dir, "--bind", scratch_dir, scratch_dir]
    argv += ["--clearenv", "--setenv", "PATH", f"{os.path.dirname(toolchain_python)}:/usr/bin:/bin",
             "--setenv", "LANG", "C.UTF-8", "--setenv", "HOME", "/home/sandbox",
             "--setenv", "PYTHONDONTWRITEBYTECODE", "1"]
    argv += ["--chdir", scratch_dir, "--", toolchain_python, "-B", "-m", "pytest", "-q"]
    return argv


def run_host_tests(
    *,
    workspace_dir: str,
    toolchain_root: str = DEFAULT_TOOLCHAIN_ROOT,
    timeout_s: int = 120,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> HostTestsResult:
    """Run pytest on a read-only copy of the workspace, inside bwrap."""
    # §EXEC3 fix 2 — the scratch is Runner-owned AND lives under the Runner's own root,
    # not in the shared system temp dir, and it is removed on EVERY path out of here.
    from .validator_evidence import host_tests_scratch_root

    scratch = tempfile.mkdtemp(prefix="host-tests-", dir=str(host_tests_scratch_root()))
    try:
        return _run_host_tests_in(scratch, workspace_dir=workspace_dir, toolchain_root=toolchain_root,
                                 timeout_s=timeout_s, max_output_bytes=max_output_bytes)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _run_host_tests_in(
    scratch: str,
    *,
    workspace_dir: str,
    toolchain_root: str,
    timeout_s: int,
    max_output_bytes: int,
) -> HostTestsResult:
    shutil.copytree(workspace_dir, scratch, dirs_exist_ok=True, symlinks=True)
    python = os.path.join(toolchain_root, "bin", "python")
    roots = list(READ_ONLY_ROOTS) + ([toolchain_root] if os.path.isdir(toolchain_root) else [])
    argv = build_host_tests_bwrap(workspace_dir=workspace_dir, scratch_dir=scratch, toolchain_python=python, roots=roots)
    started = time.time()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout_s)
        outcome = "passed" if proc.returncode == 0 else "failed"
        code: int | None = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:  # already gone
            pass
        out, err = proc.communicate()
        outcome, code = "timeout", None
    finally:
        # belt and braces: no orphan may survive the kill
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    # §OV7-1: cap both streams and digest the CAPPED bytes (never the unbounded
    # ones), so a flooding test cannot grow the evidence without bound.
    raw_out = out or b""
    raw_err = err or b""
    capped_out, out_truncated = cap_output(raw_out, max_output_bytes)
    capped_err, err_truncated = cap_output(raw_err, max_output_bytes)
    return HostTestsResult(
        ok=outcome == "passed", outcome=outcome, exit_code=code,
        stdout=capped_out.decode(errors="replace"), stderr=capped_err.decode(errors="replace"),
        seconds=round(time.time() - started, 2), scratch_dir=scratch, argv=tuple(argv),
        stdout_truncated=out_truncated, stderr_truncated=err_truncated,
        stdout_bytes=len(capped_out), stderr_bytes=len(capped_err),
        stdout_digest="sha256:" + hashlib.sha256(capped_out).hexdigest(),
        stderr_digest="sha256:" + hashlib.sha256(capped_err).hexdigest(),
    )
