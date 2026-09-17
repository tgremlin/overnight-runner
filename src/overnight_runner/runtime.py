"""Runtime helpers: lock, PAUSED sentinel, runtime fingerprint."""
from __future__ import annotations

import fcntl
import hashlib
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .safety import sha256_bytes


DEFAULT_STATE_DIR_FALLBACK = Path.home() / ".local" / "state" / "overnight-runner"


def state_dir() -> Path:
    """Read OVERNIGHT_STATE_DIR at call time so monkeypatch works."""
    p = Path(os.environ.get("OVERNIGHT_STATE_DIR", str(DEFAULT_STATE_DIR_FALLBACK)))
    p.mkdir(parents=True, exist_ok=True)
    return p


def paused_path() -> Path:
    return state_dir() / "PAUSED"


def lock_path() -> Path:
    return state_dir() / "runner.lock"


def is_paused() -> bool:
    return paused_path().exists()


def require_not_paused() -> None:
    if is_paused():
        from .safety import SafetyError
        raise SafetyError("PAUSED sentinel present; aborting")


@contextmanager
def runner_lock(timeout_seconds: float = 0.0):
    """Exclusive non-blocking lock. If held, raises immediately.

    We use flock on the lockfile. Released automatically when fd is closed.
    """
    lp = lock_path()
    lp.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lp), os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as e:
            raise RuntimeError("runner lock already held by another process") from e
        try:
            os.write(fd, f"{os.getpid()}\n".encode())
            os.fsync(fd)
        except OSError:
            pass
        yield
    finally:
        if locked:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        try:
            os.close(fd)
        except OSError:
            pass


# ----------------------------- Runtime fingerprint -----------------------------

@dataclass(frozen=True)
class FingerprintResult:
    sha256: str
    files: int


# File extensions that count as runtime source for the security fingerprint.
# Anything else (pycache, .pyc, etc.) is excluded so regenerating bytecode
# does not change the fingerprint.
_RUNTIME_SOURCE_SUFFIXES = {".py", ".toml", ".md", ".txt", ".json", ".yaml", ".yml"}
# Directories that are NEVER security-relevant runtime source.
_EXCLUDE_DIR_NAMES = {"__pycache__", ".git", ".tox", "node_modules", ".venv", "venv"}


def _is_runtime_source(p: Path) -> bool:
    if not p.is_file():
        return False
    if any(part in _EXCLUDE_DIR_NAMES for part in p.parts):
        return False
    return p.suffix.lower() in _RUNTIME_SOURCE_SUFFIXES or p.name in {"config.toml", "prompts"}


def runtime_fingerprint(file_roots: list[Path]) -> FingerprintResult:
    """Deterministic SHA-256 over (sorted_relative_path, file_sha256) pairs.

    Used to detect changes in:
      - overnight_runner source files (.py)
      - top-level config files (config.toml)
      - prompts/* (text content)

    Excludes: __pycache__/, *.pyc, .git/, virtualenvs, node_modules.
    """
    pairs: list[tuple[str, str]] = []
    for root in file_roots:
        if not root.exists():
            continue
        if root.is_file():
            if _is_runtime_source(root):
                pairs.append((root.name, sha256_bytes(root.read_bytes())))
            continue
        for p in sorted(root.rglob("*")):
            if _is_runtime_source(p):
                # Use POSIX relative path so fingerprint is independent of
                # installation prefix.
                rel = p.relative_to(root).as_posix()
                pairs.append((rel, sha256_bytes(p.read_bytes())))
    h = hashlib.sha256()
    for rel, sha in pairs:
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(sha.encode("utf-8"))
        h.update(b"\n")
    return FingerprintResult(sha256=h.hexdigest(), files=len(pairs))
