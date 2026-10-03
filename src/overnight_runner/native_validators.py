"""M5 candidate — trusted validators for the T2 execution profiles (`native-compile`, `native-automation-nullrhi`).

NOT activated. The Runner judges a native (Unreal) candidate by running a TRUSTED judge, never code the candidate wrote:

  * the judge scripts (`scripts/trio_native_*.py`, `scripts/trio_validators.py`, `scripts/trio_log_markers.py`) and the judge
    configuration (`pytest.ini`, `pipeline.config.json`, `pipeline.operator.pub`) are resolved from the TARGET repository at the
    BASE commit (the merge-base the candidate was cut from) with `git archive` into a Runner-owned directory;
  * if the candidate's workspace differs from the base on ANY judge path (edited, added or deleted) the validator refuses with a
    typed code BEFORE anything runs — a candidate cannot edit its own judges;
  * the build/automation runs in a Runner-owned COPY of the candidate in which every judge path is the trusted base copy, so the
    script that executes is byte-identical to the base blob (re-verified by sha256 immediately before exec);
  * it runs under the heavy lock (`flock`, taken by the trusted script), refuses while an `UnrealEditor` executable is running, and
    uses the trusted script's sandbox (`--sandbox`: read-only root first, then /proc /dev /tmp /dev/shm, `--unshare-pid`,
    `--unshare-net`; see docs/campaigns/pipe-bundle1/NATIVE_COMPILE_SANDBOX.md in trio-game-forge);
  * the evidence row carries ONLY digests and typed codes — never logs, stdout, or compiler text.

Typed codes (row `code`):  OK | JUDGE_SCRIPT_TAMPERED | JUDGE_CONFIG_TAMPERED | JUDGE_QUALIFICATION_TAMPERED | JUDGE_COPY_MISMATCH |
NATIVE_TRUSTED_REPO_REQUIRED | NATIVE_CONFIG_REQUIRED | NATIVE_VENDOR_MISSING | NATIVE_EDITOR_RUNNING | NATIVE_COMPILE_LOCK_HELD |
NATIVE_COMPILE_FAILED | NATIVE_COMPILE_NO_SUCCESS_MARKER | NATIVE_COMPILE_ERRORS | NATIVE_COMPILE_TIMEOUT | NATIVE_AUTOMATION_FAILED |
NATIVE_AUTOMATION_NO_RESULT | NATIVE_AUTOMATION_TEST_MISSING | NATIVE_AUTOMATION_TEST_FAILED | NATIVE_AUTOMATION_FAILURE_MARKER |
NATIVE_AUTOMATION_TIMEOUT | NATIVE_JUDGE_ERROR.
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "NATIVE_COMPILE_ID", "NATIVE_AUTOMATION_ID", "NATIVE_VALIDATOR_IDS", "JUDGE_PATTERNS", "NativeConfig",
    "editor_running", "judge_protected_paths", "judge_tamper", "run_native_validator", "with_judge_protection",
]

NATIVE_COMPILE_ID = "native-compile"
NATIVE_AUTOMATION_ID = "native-automation-nullrhi"
NATIVE_VALIDATOR_IDS = (NATIVE_COMPILE_ID, NATIVE_AUTOMATION_ID)

#: (fnmatch pattern over the repo-relative path, typed code). The judges and their configuration.
JUDGE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("scripts/trio_native_*.py", "JUDGE_SCRIPT_TAMPERED"),
    ("scripts/trio_validators.py", "JUDGE_SCRIPT_TAMPERED"),
    ("scripts/trio_log_markers.py", "JUDGE_SCRIPT_TAMPERED"),
    ("scripts/ue.sh", "JUDGE_SCRIPT_TAMPERED"),
    ("pytest.ini", "JUDGE_CONFIG_TAMPERED"),
    ("pipeline.config.json", "JUDGE_QUALIFICATION_TAMPERED"),
    ("pipeline.operator.pub", "JUDGE_QUALIFICATION_TAMPERED"),
)
#: refusal precedence when several judge paths changed
_CODE_ORDER = ("JUDGE_SCRIPT_TAMPERED", "JUDGE_QUALIFICATION_TAMPERED", "JUDGE_CONFIG_TAMPERED")
_EDITOR_NAMES = ("UnrealEditor", "UnrealEditor-Cmd")
_FAIL_LINE = re.compile(r"^FAIL ([A-Z_]+)\b", re.MULTILINE)
#: the ONLY environment a native judge sees (no credentials, no tokens)
_ENV_ALLOW = ("PATH", "HOME", "LANG", "LC_ALL", "UE_ENGINE_ROOT", "TRIO_CACHE_DIR", "TRIO_BUILD_DIR")


@dataclass(frozen=True)
class NativeConfig:
    engine_root: str
    heavy_lock: str = "/mnt/ue/Cache/trio-heavy.lock"
    #: [(source dir, destination relative to the project)] — vendor plugin SOURCES provisioned into the judged copy
    vendor_plugins: tuple[tuple[str, str], ...] = ()
    compile_timeout_s: int = 1800
    automation_timeout_s: int = 1800
    max_parallel: int = 8
    python: str = "python3"
    #: automation only: `Automation RunTests <filter>` and the test NAMES (last path segment) that must report Success
    automation_filter: str = "TRIO.WorldForge"
    automation_expect: tuple[str, ...] = ()


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _git(repo: str, *args: str, binary: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "credential.helper=", "-C", repo, *args], capture_output=True, text=not binary)


def _base_tree(repo: str, base_commit: str) -> list[str]:
    out = _git(repo, "ls-tree", "-r", "--name-only", base_commit)
    if out.returncode != 0:
        raise RuntimeError(f"base commit unreadable: {out.stderr.strip()[:120]}")
    return [line for line in out.stdout.splitlines() if line]


def _code_for(path: str) -> str | None:
    for pattern, code in JUDGE_PATTERNS:
        if fnmatch.fnmatchcase(path, pattern):
            return code
    return None


def judge_protected_paths(trusted_repo: str, base_commit: str) -> list[str]:
    """The EXACT judge paths at the trusted base (patterns expanded) — feed these to the scope gate's `protectedPaths`."""
    return sorted(p for p in _base_tree(trusted_repo, base_commit) if _code_for(p) is not None)


def with_judge_protection(contract: Mapping[str, Any], trusted_repo: str, base_commit: str) -> dict[str, Any]:
    """A scope contract whose `protectedPaths` additionally contains every judge path (a candidate cannot edit its own judges)."""
    merged = dict(contract)
    merged["protectedPaths"] = sorted(set(contract.get("protectedPaths") or []) | set(judge_protected_paths(trusted_repo, base_commit)))
    return merged


def _workspace_judge_files(workspace_dir: str) -> list[str]:
    found: list[str] = []
    root = Path(workspace_dir)
    for pattern, _ in JUDGE_PATTERNS:
        parent = Path(pattern).parent
        base = root / parent
        if base.is_dir():
            for f in base.iterdir():
                rel = str(parent / f.name) if str(parent) != "." else f.name
                if fnmatch.fnmatchcase(rel, pattern) and (f.is_file() or f.is_symlink()):
                    found.append(rel)
    return sorted(set(found))


def judge_tamper(workspace_dir: str, trusted_repo: str, base_commit: str) -> list[tuple[str, str]]:
    """[(path, typed code)] for every judge path where the candidate workspace differs from the trusted base (edit, add, delete, symlink)."""
    base_paths = set(judge_protected_paths(trusted_repo, base_commit))
    candidates = base_paths | set(_workspace_judge_files(workspace_dir))
    changed: list[tuple[str, str]] = []
    for rel in sorted(candidates):
        code = _code_for(rel) or "JUDGE_SCRIPT_TAMPERED"
        f = Path(workspace_dir) / rel
        if f.is_symlink():
            changed.append((rel, code)); continue
        in_base = rel in base_paths
        if not f.is_file():
            if in_base:
                changed.append((rel, code))
            continue
        if not in_base:
            changed.append((rel, code)); continue
        blob = _git(trusted_repo, "cat-file", "blob", f"{base_commit}:{rel}", binary=True)
        if blob.returncode != 0 or blob.stdout != f.read_bytes():
            changed.append((rel, code))
    return changed


def editor_running() -> bool:
    """True iff a process whose EXECUTABLE (argv[0] basename) is UnrealEditor/UnrealEditor-Cmd exists. A mention in an argument is not an editor."""
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            raw = Path(f"/proc/{name}/cmdline").read_bytes()
        except OSError:
            continue
        argv = [a for a in raw.split(b"\0") if a]
        if argv and Path(argv[0].decode("utf-8", "replace")).name in _EDITOR_NAMES:
            return True
    return False


def _scratch_root() -> Path:
    from .validator_evidence import host_tests_scratch_root

    return host_tests_scratch_root()


def _copy_workspace(src: str, dst: Path) -> None:
    shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns(".git", "Intermediate", "Binaries", "Saved", "DerivedDataCache"))


def _export_trusted(trusted_repo: str, base_commit: str, paths: Sequence[str], into: Path) -> None:
    """`git archive` of EXACTLY the judge paths at the base commit (a Runner-owned directory, never the candidate)."""
    into.mkdir(parents=True, exist_ok=True)
    if not paths:
        return
    archive = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-C", trusted_repo, "archive", "--format=tar", base_commit, "--", *paths], capture_output=True)
    if archive.returncode != 0:
        raise RuntimeError("git archive of the judge paths failed")
    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tf:
        for member in tf.getmembers():
            if member.issym() or member.islnk() or not member.name or member.name.startswith(("/", "..")) or ".." in member.name.split("/"):
                raise RuntimeError("unsafe member in the trusted judge archive")
        tf.extractall(into)


def _provision_vendor(plugins: Iterable[tuple[str, str]], project: Path) -> str | None:
    for src, dest in plugins:
        if not Path(src).is_dir():
            return "NATIVE_VENDOR_MISSING"
        target = project / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, target, symlinks=True, ignore=shutil.ignore_patterns("Intermediate", "Binaries", "Saved", "DerivedDataCache"))
    return None


def _init_judged_repo(judged: Path) -> None:
    """A Runner-owned git repo (hooks off) at the judged copy's root, committing everything.

    UBT builds differently without a git work tree (diagnosed 2026-10-02: the SAME tree that compiles inside a git repo fails with
    `-Werror` unreachable-code errors in the vendor plugin when `.git` is absent). The candidate's own `.git` is never copied.
    """
    base = ["git", "-c", "core.hooksPath=/dev/null", "-c", "credential.helper=", "-c", "user.name=runner", "-c", "user.email=runner@localhost"]
    for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "--allow-empty", "-m", "judged"]):
        subprocess.run([*base, "-C", str(judged), *args], capture_output=True)


def _env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k in _ENV_ALLOW}


def _sanitize_repair(text: str, cap: int = 4000) -> str:
    """A SANITIZED repair summary for the NEXT attempt's prompt: only typed error/result lines,
    repo-relative paths, no absolute scratch paths and no long digests. Written to a
    driver-readable repair dir (TRIO_NATIVE_REPAIR_DIR); never part of the evidence row."""
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if "error:" in stripped or stripped.startswith("FAIL ") or stripped.startswith("Result:"):
            s = re.sub(r"(?:/[^\s:]*/)*([A-Za-z0-9_.+-]+\.(?:cpp|cc|h|hpp|py|ini|json|cs))", r"\1", stripped)
            s = re.sub(r"\b[0-9a-f]{16,}\b", "<hash>", s)
            out.append(s[:300])
    return "\n".join(out)[:cap]


def _row(validator_id: str, outcome: str, code: str, *, exit_code: int | None = None, out: bytes = b"", err: bytes = b"", log: bytes | None = None,
         seconds: float = 0.0, tampered: Sequence[str] = ()) -> dict[str, Any]:
    """The evidence row: digests and a typed code ONLY."""
    row: dict[str, Any] = {
        "validator_id": validator_id, "outcome": outcome, "code": code, "exit_code": exit_code,
        "stdout_digest": _sha(out), "stderr_digest": _sha(err), "log_digest": _sha(log) if log is not None else None, "seconds": round(seconds, 2),
    }
    if tampered:
        row["tampered_paths"] = sorted(tampered)
    return row


def _run(argv: list[str], cwd: Path, timeout_s: int) -> tuple[int | None, bytes, bytes, bool]:
    proc = subprocess.Popen(argv, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, env=_env())
    try:
        out, err = proc.communicate(timeout=timeout_s)
        return proc.returncode, out, err, False
    except subprocess.TimeoutExpired:
        for sig in (signal.SIGTERM, signal.SIGKILL):          # only the process group WE started
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except (ProcessLookupError, PermissionError):
                break
            try:
                out, err = proc.communicate(timeout=20)
                return None, out, err, True
            except subprocess.TimeoutExpired:
                continue
        out, err = proc.communicate()
        return None, out, err, True


def run_native_validator(
    validator_id: str, *, workspace_dir: str, trusted_repo: str | None, base_commit: str, config: NativeConfig | None,
) -> dict[str, Any]:
    """One native validator. Returns the digest-and-code evidence row (never raises for a judged outcome)."""
    if validator_id not in NATIVE_VALIDATOR_IDS:
        return _row(validator_id, "refused", "NATIVE_JUDGE_ERROR")
    if trusted_repo is None:
        return _row(validator_id, "refused", "NATIVE_TRUSTED_REPO_REQUIRED")
    if config is None:
        return _row(validator_id, "refused", "NATIVE_CONFIG_REQUIRED")
    if not os.path.isabs(workspace_dir) or not os.path.isdir(workspace_dir):
        # ported from agent/t2-validators: refuse a relative/nonexistent workspace before any judge work
        return _row(validator_id, "refused", "NATIVE_JUDGE_ERROR")
    started = time.time()
    try:
        tampered = judge_tamper(workspace_dir, trusted_repo, base_commit)
    except RuntimeError:
        return _row(validator_id, "refused", "NATIVE_JUDGE_ERROR")
    if tampered:
        code = next(c for c in _CODE_ORDER if any(code == c for _, code in tampered))
        return _row(validator_id, "refused", code, tampered=[p for p, _ in tampered])
    if editor_running():
        return _row(validator_id, "refused", "NATIVE_EDITOR_RUNNING")
    scratch = Path(tempfile.mkdtemp(prefix="native-", dir=str(_scratch_root())))
    try:
        judged = scratch / "judged"
        trusted = scratch / "trusted"
        try:
            _copy_workspace(workspace_dir, judged)
            judge_paths = judge_protected_paths(trusted_repo, base_commit)
            _export_trusted(trusted_repo, base_commit, judge_paths, trusted)
            for rel in judge_paths:                      # the judged copy runs the TRUSTED bytes
                dest = judged / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(trusted / rel, dest)
                shutil.copymode(trusted / rel, dest)
        except (RuntimeError, OSError):
            return _row(validator_id, "refused", "NATIVE_JUDGE_ERROR")
        script_rel = "scripts/trio_native_compile.py" if validator_id == NATIVE_COMPILE_ID else "scripts/trio_native_automation.py"
        script = judged / script_rel
        blob = _git(trusted_repo, "cat-file", "blob", f"{base_commit}:{script_rel}", binary=True)
        if blob.returncode != 0 or not script.is_file() or script.read_bytes() != blob.stdout:
            return _row(validator_id, "refused", "JUDGE_COPY_MISMATCH" if blob.returncode == 0 else "NATIVE_JUDGE_ERROR")
        if _provision_vendor(config.vendor_plugins, judged) is not None:
            return _row(validator_id, "refused", "NATIVE_VENDOR_MISSING")
        _init_judged_repo(judged)
        log_file = scratch / "native.log"
        argv = [config.python, str(script), "--sandbox", "--engine", config.engine_root, "--lock", config.heavy_lock, "--log-out", str(log_file)]
        if validator_id == NATIVE_COMPILE_ID:
            argv += ["--max-parallel", str(config.max_parallel), "--timeout", str(config.compile_timeout_s)]
            timeout_s, timeout_code, fail_code = config.compile_timeout_s + 120, "NATIVE_COMPILE_TIMEOUT", "NATIVE_COMPILE_FAILED"
        else:
            argv += ["--tests", config.automation_filter, "--timeout", str(config.automation_timeout_s)]
            for name in config.automation_expect:
                argv += ["--expect", name]
            timeout_s, timeout_code, fail_code = config.automation_timeout_s + 120, "NATIVE_AUTOMATION_TIMEOUT", "NATIVE_AUTOMATION_FAILED"
        rc, out, err, timed_out = _run(argv, judged, timeout_s)
        log = log_file.read_bytes() if log_file.is_file() else None
        seconds = time.time() - started
        debug_dir = os.environ.get("TRIO_NATIVE_DEBUG_DIR")                 # operator-only: keep the raw judge output OUTSIDE the evidence (never in the row)
        if debug_dir:
            d = Path(debug_dir) / f"{validator_id}-{int(started)}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "stdout.txt").write_bytes(out); (d / "stderr.txt").write_bytes(err)
            if log is not None:
                (d / "native.log").write_bytes(log)
        repair_dir = os.environ.get("TRIO_NATIVE_REPAIR_DIR")             # driver repair channel: SANITIZED error summary only
        if repair_dir:
            rd = Path(repair_dir); rd.mkdir(parents=True, exist_ok=True)
            (rd / f"{validator_id}.repair.txt").write_text(
                _sanitize_repair(err.decode("utf-8", "replace") + "\n" + (log.decode("utf-8", "replace") if log else "")))
        if timed_out:
            return _row(validator_id, "failed", timeout_code, out=out, err=err, log=log, seconds=seconds)
        if rc == 0:
            return _row(validator_id, "passed", "OK", exit_code=0, out=out, err=err, log=log, seconds=seconds)
        match = _FAIL_LINE.search(err.decode("utf-8", "replace"))
        code = match.group(1) if match else fail_code
        outcome = "refused" if code in ("NATIVE_COMPILE_LOCK_HELD", "NATIVE_COMPILE_EDITOR_RUNNING", "NATIVE_COMPILE_SANDBOX_UNAVAILABLE") else "failed"
        if code == "NATIVE_COMPILE_EDITOR_RUNNING":
            code = "NATIVE_EDITOR_RUNNING"
        return _row(validator_id, outcome, code, exit_code=rc, out=out, err=err, log=log, seconds=seconds)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
