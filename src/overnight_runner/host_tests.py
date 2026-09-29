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

import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any

__all__ = ["DEFAULT_TOOLCHAIN_ROOT", "HostTestsResult", "build_host_tests_bwrap", "run_host_tests"]

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


def build_host_tests_bwrap(*, workspace_dir: str, scratch_dir: str, toolchain_python: str, roots=None) -> list[str]:
    """The exact argv the Runner runs for host-tests. Pure; safe to assert on."""
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
) -> HostTestsResult:
    """Run pytest on a read-only copy of the workspace, inside bwrap."""
    scratch = tempfile.mkdtemp(prefix="trio-host-tests-")  # Runner-owned, outside the workspace
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
    shutil.rmtree(scratch, ignore_errors=True)
    return HostTestsResult(
        ok=outcome == "passed", outcome=outcome, exit_code=code,
        stdout=(out or b"").decode(errors="replace"), stderr=(err or b"").decode(errors="replace"),
        seconds=round(time.time() - started, 2), scratch_dir=scratch, argv=tuple(argv),
    )
