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


DEFAULT_STATE_DIR = Path(os.environ.get("OVERNIGHT_STATE_DIR", str(Path.home() / ".local" / "state" / "overnight-runner")))


def state_dir() -> Path:
    p = DEFAULT_STATE_DIR
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
        raise RuntimeError("PAUSED sentinel present; aborting")


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


def runtime_fingerprint(file_roots: list[Path]) -> FingerprintResult:
    """Deterministic SHA-256 over (sorted_path, file_sha256) pairs.

    Used to detect changes in:
      - schemas.py, broker.py, worker.py, ollama_client.py, config.toml, prompts/*
    """
    pairs: list[tuple[str, str]] = []
    for root in file_roots:
        if not root.exists():
            continue
        if root.is_file():
            try:
                rel = str(root)
                pairs.append((rel, sha256_bytes(root.read_bytes())))
            except OSError:
                continue
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            try:
                rel = str(p)
                pairs.append((rel, sha256_bytes(p.read_bytes())))
            except OSError:
                continue
    h = hashlib.sha256()
    for rel, sha in pairs:
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(sha.encode("utf-8"))
        h.update(b"\n")
    return FingerprintResult(sha256=h.hexdigest(), files=len(pairs))
