"""§T2 — Runner-side validators for the T2 (native-compile / NullRHI) chunk track.

These validators are NOT like `host-tests`: a native build and an editor automation run
are HEAVY, may not run while an editor owns the GPU, and — critically — their validator
COMMANDS must come from a TRUSTED source, never from the candidate workspace a model
just wrote (a candidate could otherwise rewrite the validator that judges it).

Controls, all enforced here:

* **trusted validator copy.** The validator scripts are read from the target repository
  at the MERGE-BASE with `git show <merge_base>:<path>` — never from the candidate
  workspace. A path that is absent at the merge-base is `T2_SOURCE_MISSING`.
* **editor guard.** While an `UnrealEditor` process runs, the run refuses
  (`T2_EDITOR_RUNNING`) — the same rule the heavy lock protects.
* **heavy lock.** A non-blocking `flock` on the shared heavy lock; contention is
  `T2_LOCK_HELD`.
* **sandbox flags.** The command runs inside bubblewrap on a READ-ONLY copy of the
  workspace with exactly one read-write bind (a Runner-owned scratch), the pinned
  toolchain read-only, and the network unshared.
* **digests + typed codes only.** The result carries `stdout_digest`/`stderr_digest`
  and ONE typed code — never raw validator output.

The trusted validator scripts also join the protected-path set.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    "T2_REFUSALS",
    "T2Result",
    "T2_VALIDATOR_IDS",
    "TRUSTED_VALIDATOR_PREFIXES",
    "T2_PROTECTED_PATHS",
    "copy_trusted_validators",
    "editor_running",
    "heavy_lock",
    "run_t2_validator",
]

#: Typed refusals. The caller never sees a raw validator string, only one of these.
T2_REFUSALS = (
    "T2_SOURCE_MISSING",
    "T2_DIGEST_MISMATCH",
    "T2_EDITOR_RUNNING",
    "T2_LOCK_HELD",
    "T2_WORKSPACE_UNSAFE",
    "T2_SANDBOX_UNAVAILABLE",
    "T2_TIMEOUT",
    "T2_NATIVE_COMPILE_FAILED",
    "T2_NATIVE_COMPILE_NO_SUCCESS_MARKER",
    "T2_NATIVE_COMPILE_ERRORS",
    "T2_AUTOMATION_FAILED",
    "T2_AUTOMATION_NO_MARKER",
    "T2_INTERNAL_ERROR",
)

#: The validator ids this module serves.
T2_VALIDATOR_IDS = ("native-compile", "native_compile", "host-nullrhi", "nullrhi", "nullrhi-automation")

#: Paths under the target repo whose bytes are the TRUSTED validator. Copied from the
#: merge-base; a candidate copy is never read.
TRUSTED_VALIDATOR_PREFIXES = (
    "scripts/trio_validators.py",
    "scripts/trio_native_",
    "scripts/trio_log_markers.py",
)

#: Protected paths extended by the T2 validator track.
T2_PROTECTED_PATHS = (
    "scripts/trio_validators.py",
    "scripts/trio_native_compile.py",
    "scripts/trio_native_harness.py",
    "scripts/trio_log_markers.py",
)

DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024
DEFAULT_TOOLCHAIN_ROOT = "/mnt/ue/Projects/trio-toolchain/venv"
READ_ONLY_ROOTS = ("/usr", "/lib", "/lib64", "/bin", "/sbin")


@dataclass(frozen=True)
class T2Result:
    validator_id: str
    outcome: str  # passed | failed | refused
    code: str  # a T2_REFUSALS code, or "" when passed
    exit_code: int | None
    stdout_digest: str
    stderr_digest: str
    seconds: float
    trusted_digest: str  # sha256 over the trusted validator set actually used

    def as_evidence(self) -> dict[str, Any]:
        return {
            "validator_id": self.validator_id,
            "outcome": self.outcome,
            "code": self.code,
            "exit_code": self.exit_code,
            "stdout_digest": self.stdout_digest,
            "stderr_digest": self.stderr_digest,
            "trusted_digest": self.trusted_digest,
        }


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _cap(data: bytes, cap: int = DEFAULT_MAX_OUTPUT_BYTES) -> bytes:
    return data[:cap]


def _refused(validator_id: str, code: str) -> T2Result:
    return T2Result(validator_id=validator_id, outcome="refused", code=code, exit_code=None,
                    stdout_digest="", stderr_digest="", seconds=0.0, trusted_digest="")


def copy_trusted_validators(
    *,
    target_repo: str,
    merge_base: str,
    dest_dir: str,
    run: Callable[..., subprocess.CompletedProcess] | None = None,
) -> tuple[bool, str, dict[str, str]]:
    """Copy the trusted validator scripts from the target repo at `merge_base`.

    Returns `(ok, code_or_empty, {name: sha256})`. The candidate workspace is NEVER
    consulted: the bytes come from `git -C <target_repo> show <merge_base>:<path>`.
    """
    run = run or subprocess.run
    listing = run(["git", "-C", target_repo, "ls-tree", "-r", "--name-only", merge_base],
                  capture_output=True, text=True, check=False)
    if listing.returncode != 0:
        return False, "T2_SOURCE_MISSING", {}
    paths = [
        p for p in (listing.stdout or "").splitlines()
        if any(p == pref or p.startswith(pref) for pref in TRUSTED_VALIDATOR_PREFIXES)
    ]
    if not paths:
        return False, "T2_SOURCE_MISSING", {}
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for path in paths:
        show = run(["git", "-C", target_repo, "show", f"{merge_base}:{path}"],
                   capture_output=True, check=False)
        if show.returncode != 0:
            return False, "T2_SOURCE_MISSING", {}
        raw = show.stdout if isinstance(show.stdout, (bytes, bytearray)) else str(show.stdout).encode()
        name = os.path.basename(path)
        target = dest / name
        target.write_bytes(raw)
        os.chmod(target, 0o644)  # read-only to the sandbox user
        digests[path] = _digest(raw)
    return True, "", digests


def editor_running(pgrep: Callable[..., subprocess.CompletedProcess] | None = None) -> bool:
    """True while an UnrealEditor process is running on the host."""
    pgrep = pgrep or subprocess.run
    proc = pgrep(["pgrep", "-x", "UnrealEditor"], capture_output=True, text=True, check=False)
    return proc.returncode == 0 and bool((proc.stdout or "").strip())


@contextmanager
def heavy_lock(lock_path: str):
    """A non-blocking exclusive flock. Raises RuntimeError('T2_LOCK_HELD') on contention."""
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as exc:  # pragma: no cover - exercised with a real second fd
            raise RuntimeError("T2_LOCK_HELD") from exc
        os.write(fd, f"{os.getpid()}\n".encode())
        yield
    finally:
        if locked:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


def _build_bwrap(*, workspace_dir: str, scratch_dir: str, trusted_dir: str,
                 toolchain_python: str, toolchain_root: str) -> list[str]:
    argv = ["bwrap", "--die-with-parent", "--new-session", "--unshare-user", "--unshare-pid",
            "--unshare-net", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup"]
    roots = list(READ_ONLY_ROOTS)
    if os.path.isdir(toolchain_root):
        roots.append(toolchain_root)
    for root in roots:
        argv += ["--ro-bind", root, root]
    argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/home/sandbox"]
    # the workspace is READ-ONLY; the ONLY read-write bind is the Runner-owned scratch
    argv += ["--ro-bind", workspace_dir, workspace_dir]
    argv += ["--ro-bind", trusted_dir, trusted_dir]
    argv += ["--bind", scratch_dir, scratch_dir]
    argv += ["--clearenv", "--setenv", "PATH", f"{os.path.dirname(toolchain_python)}:/usr/bin:/bin",
             "--setenv", "LANG", "C.UTF-8", "--setenv", "HOME", "/home/sandbox",
             "--setenv", "PYTHONDONTWRITEBYTECODE", "1"]
    return argv


def run_t2_validator(
    *,
    validator_id: str,
    repo_root: str,
    merge_base: str,
    target_repo: str,
    lock_path: str,
    toolchain_root: str = DEFAULT_TOOLCHAIN_ROOT,
    timeout_s: int = 3600,
    scratch_root: str | None = None,
    expected: Mapping[str, str] | None = None,
    runner: Callable[..., subprocess.Popen] | None = None,
    run_git: Callable[..., subprocess.CompletedProcess] | None = None,
    is_editor_running: Callable[[], bool] | None = None,
) -> T2Result:
    """Run one T2 validator under every control, returning DIGESTS + a TYPED code only."""
    if validator_id not in T2_VALIDATOR_IDS:
        return _refused(validator_id, "T2_INTERNAL_ERROR")
    if not os.path.isabs(repo_root) or not os.path.isdir(repo_root):
        return _refused(validator_id, "T2_WORKSPACE_UNSAFE")
    if shutil.which("bwrap") is None and runner is None:
        return _refused(validator_id, "T2_SANDBOX_UNAVAILABLE")

    # 1. editor guard
    if (is_editor_running or editor_running)():
        return _refused(validator_id, "T2_EDITOR_RUNNING")

    # 2. heavy lock
    try:
        with heavy_lock(lock_path):
            return _run_locked(
                validator_id=validator_id, repo_root=repo_root, merge_base=merge_base,
                target_repo=target_repo, toolchain_root=toolchain_root, timeout_s=timeout_s,
                scratch_root=scratch_root, expected=expected, runner=runner, run_git=run_git,
            )
    except RuntimeError as exc:
        if str(exc) == "T2_LOCK_HELD":
            return _refused(validator_id, "T2_LOCK_HELD")
        raise


def _run_locked(*, validator_id, repo_root, merge_base, target_repo, toolchain_root,
                timeout_s, scratch_root, expected, runner, run_git) -> T2Result:
    parent = scratch_root or tempfile.gettempdir()
    scratch = tempfile.mkdtemp(prefix="t2-scratch-", dir=parent)
    try:
        trusted_dir = os.path.join(scratch, "trusted")
        ok, code, digests = copy_trusted_validators(
            target_repo=target_repo, merge_base=merge_base, dest_dir=trusted_dir, run=run_git,
        )
        if not ok:
            return _refused(validator_id, code)
        trusted_digest = _digest("".join(f"{k}:{digests[k]}\n" for k in sorted(digests)).encode())
        if expected is not None and any(expected.get(k) != v for k, v in digests.items()):
            return T2Result(validator_id=validator_id, outcome="refused", code="T2_DIGEST_MISMATCH",
                            exit_code=None, stdout_digest="", stderr_digest="", seconds=0.0,
                            trusted_digest=trusted_digest)

        work = os.path.join(scratch, "ws")
        shutil.copytree(repo_root, work, dirs_exist_ok=True, symlinks=True)
        toolchain_python = os.path.join(toolchain_root, "bin", "python")
        argv = _build_bwrap(workspace_dir=repo_root, scratch_dir=work, trusted_dir=trusted_dir,
                            toolchain_python=toolchain_python, toolchain_root=toolchain_root)
        argv += ["--chdir", work, "--", toolchain_python, "-B", os.path.join(trusted_dir, _script_for(validator_id))]

        started = time.time()
        spawn = runner or subprocess.Popen
        proc = spawn(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            out, err = proc.communicate(timeout=timeout_s)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except (ProcessLookupError, PermissionError):
                pass
            out, err = proc.communicate()
            return T2Result(validator_id=validator_id, outcome="refused", code="T2_TIMEOUT",
                            exit_code=None, stdout_digest=_digest(_cap(out or b"")),
                            stderr_digest=_digest(_cap(err or b"")), seconds=round(time.time() - started, 2),
                            trusted_digest=trusted_digest)
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except (ProcessLookupError, PermissionError):
                pass

        raw_out = out or b""
        raw_err = err or b""
        code = _classify(validator_id, exit_code, raw_out, raw_err)
        outcome = "passed" if code == "" else "failed"
        return T2Result(validator_id=validator_id, outcome=outcome, code=code, exit_code=exit_code,
                        stdout_digest=_digest(_cap(raw_out)), stderr_digest=_digest(_cap(raw_err)),
                        seconds=round(time.time() - started, 2), trusted_digest=trusted_digest)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _script_for(validator_id: str) -> str:
    return "trio_native_compile.py" if validator_id.startswith("native") else "trio_native_harness.py"


def _classify(validator_id: str, exit_code: int | None, out: bytes, err: bytes) -> str:
    """Typed code from the exit code and the structured markers only (never prose)."""
    text = (out + b"\n" + err).decode(errors="replace")
    if exit_code != 0:
        if validator_id.startswith("native"):
            if "error:" in text:
                return "T2_NATIVE_COMPILE_ERRORS"
            return "T2_NATIVE_COMPILE_FAILED"
        return "T2_AUTOMATION_FAILED"
    if validator_id.startswith("native"):
        return "" if "Result: Succeeded" in text else "T2_NATIVE_COMPILE_NO_SUCCESS_MARKER"
    return "" if "TRIO_AUTOMATION_PASS" in text else "T2_AUTOMATION_NO_MARKER"
