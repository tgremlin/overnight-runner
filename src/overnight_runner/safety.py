"""Repo/path safety layer.

All filesystem operations on the target repository MUST go through these helpers.

Invariants:
- Paths must resolve into the repo root, with no ../ traversal.
- Symlink escapes are detected by realpath comparison.
- Writes may only land on authorized paths.
- .git is never writable through the runner.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


# Files / dirs that the runner never writes (relative to repo root).
DEFAULT_PROTECTED_RELATIVE = {
    ".git",
    ".git/",
    "vendor/",
    "third_party/",
    "node_modules/",
    "__pycache__/",
}


class SafetyError(Exception):
    """Raised on any safety violation. Caller should translate to BLOCKED / REVIEW_REQUIRED."""


@dataclass(frozen=True)
class ResolvedPath:
    raw: str
    absolute: Path
    relative_to_repo: Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def resolve_repo_path(repo_root: Path, raw_path: str) -> ResolvedPath:
    """Resolve a model/path-supplied path against repo_root.

    Rules:
      - No absolute paths.
      - No '..' segments (after normalising, segments must not contain '..').
      - The resolved realpath must stay inside repo_root's realpath.
      - Symlinks pointing outside are rejected.
    """
    if not raw_path or raw_path.startswith("/"):
        raise SafetyError(f"absolute paths not allowed: {raw_path!r}")
    # Strip leading ./ for clarity, but reject ..
    p = Path(raw_path)
    parts = p.parts
    if any(part == ".." for part in parts):
        raise SafetyError(f"path traversal not allowed: {raw_path!r}")

    abs_repo = repo_root.resolve(strict=False)
    candidate = (abs_repo / raw_path).resolve(strict=False)
    try:
        candidate.relative_to(abs_repo)
    except ValueError:
        raise SafetyError(f"path escapes repo: {raw_path!r}") from None

    # If the candidate already exists, ensure its realpath is contained.
    if candidate.exists():
        rp = candidate.resolve(strict=True)
        try:
            rp.relative_to(abs_repo)
        except ValueError:
            raise SafetyError(f"symlink escape detected: {raw_path!r}") from None

    return ResolvedPath(
        raw=raw_path,
        absolute=candidate,
        relative_to_repo=Path(*candidate.relative_to(abs_repo).parts),
    )


def is_protected(repo_root: Path, rel: Path, protected_relative: set[str]) -> bool:
    """A path is protected if it equals or starts with one of the protected prefixes."""
    s = rel.as_posix()
    for prefix in protected_relative:
        p = prefix.rstrip("/")
        if s == p or s.startswith(p + "/"):
            return True
    return False


def ensure_writable(repo_root: Path, rel: Path, protected_relative: set[str]) -> None:
    if is_protected(repo_root, rel, protected_relative):
        raise SafetyError(f"protected path not writable: {rel.as_posix()}")
    # .git absolute guard even if protected set is empty
    parts = rel.parts
    if ".git" in parts:
        raise SafetyError(f".git path not writable: {rel.as_posix()}")


# ----------------------------- Git helpers -----------------------------

def git_head(repo_root: Path) -> str:
    """Return current HEAD commit SHA, or '' if no commits / not a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except subprocess.CalledProcessError:
        return ""
    except FileNotFoundError:
        return ""


def git_is_clean(repo_root: Path) -> bool:
    """True iff `git status --porcelain` is empty."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        # If git fails, treat as not-clean (safer).
        return False
    return out.stdout.strip() == ""


def git_worktree_sha(repo_root: Path) -> str:
    """Hash of the entire working-tree state (HEAD + dirty files content).

    Used to detect unauthorized changes between approval and execution.
    """
    files: list[str] = []
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repo_root,
            capture_output=True,
            check=True,
        )
        files = [f for f in out.stdout.split(b"\x00") if f]
    except subprocess.CalledProcessError:
        pass

    h = hashlib.sha256()
    for raw in sorted(files):
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            continue
        h.update(text.encode("utf-8"))
        h.update(b"\x00")
        full = repo_root / Path(text)
        try:
            h.update(full.read_bytes())
        except OSError:
            h.update(b"<unreadable>")
    return h.hexdigest()


def git_init_empty(repo_root: Path) -> None:
    """Initialise an empty git repo at repo_root (idempotent)."""
    if (repo_root / ".git").exists():
        return
    repo_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "runner@example"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Runner"], cwd=repo_root, check=True, capture_output=True)


def git_commit_all(repo_root: Path, message: str) -> str:
    """For TESTS ONLY. The runner itself NEVER calls this."""
    subprocess.run(["git", "add", "-A"], cwd=repo_root, check=True, capture_output=True)
    out = subprocess.run(
        ["git", "commit", "-m", message, "--allow-empty"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
