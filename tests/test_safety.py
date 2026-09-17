"""Safety layer tests."""
import os
from pathlib import Path

import pytest

from overnight_runner.safety import (
    SafetyError,
    git_commit_all,
    git_head,
    git_init_empty,
    git_is_clean,
    git_worktree_sha,
    is_protected,
    resolve_repo_path,
    sha256_file,
)


def test_resolve_repo_path_inside(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    rp = resolve_repo_path(tmp_path, "sub/file.txt")
    assert rp.absolute == (tmp_path / "sub" / "file.txt").resolve()


def test_resolve_repo_path_traversal_rejected(tmp_path: Path):
    with pytest.raises(SafetyError):
        resolve_repo_path(tmp_path, "../etc/passwd")
    with pytest.raises(SafetyError):
        resolve_repo_path(tmp_path, "a/../../b")


def test_resolve_repo_path_absolute_rejected(tmp_path: Path):
    with pytest.raises(SafetyError):
        resolve_repo_path(tmp_path, "/etc/passwd")


def test_resolve_repo_path_symlink_escape(tmp_path: Path):
    target = tmp_path.parent / "outside.txt"
    target.write_text("hi")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unsupported")
    with pytest.raises(SafetyError):
        resolve_repo_path(tmp_path, "link.txt")


def test_is_protected_matches():
    assert is_protected(Path("/x"), Path(".git/HEAD"), {".git"})
    assert is_protected(Path("/x"), Path("vendor/foo"), {"vendor/"})
    assert not is_protected(Path("/x"), Path("src/foo.py"), {"vendor/"})


def test_sha256_file(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"hello")
    # sha256("hello")
    assert sha256_file(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_git_init_and_clean(tmp_path: Path):
    git_init_empty(tmp_path)
    (tmp_path / "a.txt").write_text("hi\n")
    git_commit_all(tmp_path, "init")
    assert git_is_clean(tmp_path)
    assert len(git_head(tmp_path)) == 40


def test_git_worktree_sha_changes(tmp_path: Path):
    git_init_empty(tmp_path)
    (tmp_path / "a.txt").write_text("v1")
    git_commit_all(tmp_path, "v1")
    s1 = git_worktree_sha(tmp_path)
    (tmp_path / "a.txt").write_text("v2")
    s2 = git_worktree_sha(tmp_path)
    assert s1 != s2
