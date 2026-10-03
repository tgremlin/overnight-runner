"""§T2 validator tests — hermetic (no bwrap, no editor, no engine).

The rows assert the CONTROLS, not a real build: the trusted validator bytes come from
the merge-base (never the candidate), the editor guard and heavy lock refuse, the
result carries digests + a typed code and NO raw output, and a digest mismatch refuses.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from overnight_runner.t2_validators import (
    T2_PROTECTED_PATHS,
    copy_trusted_validators,
    heavy_lock,
    run_t2_validator,
)

MERGE_BASE = "5074d47cef76da9f73d0eda626731c6510ff70e4"
TRUSTED_BYTES = b"print('trusted validator')\n"


def _git_fake(records, *, missing=False):
    def run(argv, **kwargs):
        records.append(list(argv))
        if "ls-tree" in argv:
            return SimpleNamespace(returncode=0, stdout="scripts/trio_validators.py\nscripts/trio_native_compile.py\nscripts/trio_native_harness.py\n", stderr="")
        if "show" in argv:
            if missing:
                return SimpleNamespace(returncode=1, stdout=b"", stderr="missing")
            return SimpleNamespace(returncode=0, stdout=TRUSTED_BYTES, stderr=b"")
        return SimpleNamespace(returncode=1, stdout="", stderr="unexpected")
    return run


def test_trusted_validators_come_from_the_merge_base_not_the_candidate(tmp_path):
    records: list[list[str]] = []
    ok, code, digests = copy_trusted_validators(
        target_repo="/target/repo", merge_base=MERGE_BASE, dest_dir=str(tmp_path / "t"), run=_git_fake(records),
    )
    assert ok and code == ""
    show_calls = [r for r in records if "show" in r]
    assert show_calls, "expected git show calls"
    for call in show_calls:
        assert MERGE_BASE in " ".join(call)
        assert "/target/repo" in call
    # the candidate workspace is never consulted; each path is in a `sha:path` arg
    assert any("scripts/trio_validators.py" in arg for call in show_calls for arg in call)
    assert all(":scripts/trio_native_" in arg or ":scripts/trio_validators.py" in arg
               for call in show_calls for arg in call if ":" in arg)
    expected = "sha256:" + hashlib.sha256(TRUSTED_BYTES).hexdigest()
    assert digests["scripts/trio_validators.py"] == expected
    assert (tmp_path / "t" / "trio_validators.py").read_bytes() == TRUSTED_BYTES


def test_missing_trusted_source_refuses(tmp_path):
    ok, code, _ = copy_trusted_validators(
        target_repo="/target/repo", merge_base=MERGE_BASE, dest_dir=str(tmp_path / "t"),
        run=_git_fake([], missing=True),
    )
    assert not ok and code == "T2_SOURCE_MISSING"


def test_editor_guard_refuses(tmp_path):
    r = run_t2_validator(
        validator_id="native-compile", repo_root=str(tmp_path), merge_base=MERGE_BASE,
        target_repo="/target/repo", lock_path=str(tmp_path / "lock"),
        is_editor_running=lambda: True, runner=object(),
    )
    assert r.outcome == "refused" and r.code == "T2_EDITOR_RUNNING"


def test_relative_workspace_refused(tmp_path):
    r = run_t2_validator(
        validator_id="native-compile", repo_root="relative/path", merge_base=MERGE_BASE,
        target_repo="/target/repo", lock_path=str(tmp_path / "lock"), runner=object(),
    )
    assert r.outcome == "refused" and r.code == "T2_WORKSPACE_UNSAFE"


def test_heavy_lock_contention_refuses(tmp_path):
    lock = str(tmp_path / "heavy.lock")
    with heavy_lock(lock):
        r = run_t2_validator(
            validator_id="native-compile", repo_root=str(tmp_path), merge_base=MERGE_BASE,
            target_repo="/target/repo", lock_path=lock, is_editor_running=lambda: False, runner=object(),
        )
    assert r.outcome == "refused" and r.code == "T2_LOCK_HELD"


class _FakeProc:
    def __init__(self, out: bytes, err: bytes, code: int):
        self._out, self._err, self.returncode, self.pid = out, err, code, 999999

    def communicate(self, timeout=None):
        return self._out, self._err


def _fake_runner(out: bytes, code: int):
    def spawn(argv, **kwargs):
        return _FakeProc(out, b"", code)
    return spawn


def _run(tmp_path, *, out: bytes, code: int, expected=None):
    return run_t2_validator(
        validator_id="native-compile", repo_root=str(tmp_path), merge_base=MERGE_BASE,
        target_repo="/target/repo", lock_path=str(tmp_path / "lock"), runner=_fake_runner(out, code),
        run_git=_git_fake([]), is_editor_running=lambda: False, expected=expected,
    )


def test_result_carries_digests_and_typed_code_only(tmp_path):
    r = _run(tmp_path, out=b"random build output, no marker", code=0)
    assert r.outcome == "failed" and r.code == "T2_NATIVE_COMPILE_NO_SUCCESS_MARKER"
    evidence = r.as_evidence()
    assert set(evidence) >= {"stdout_digest", "stderr_digest", "code"}
    # NEVER raw output
    assert "stdout" not in evidence and "stderr" not in evidence
    assert evidence["stdout_digest"].startswith("sha256:")


def test_success_marker_passes(tmp_path):
    r = _run(tmp_path, out=b"BuildGraph ... Result: Succeeded\n", code=0)
    assert r.outcome == "passed" and r.code == ""


def test_error_lines_are_their_own_code(tmp_path):
    r = _run(tmp_path, out=b"Foo.cpp(1): error: boom\n", code=1)
    assert r.outcome == "failed" and r.code == "T2_NATIVE_COMPILE_ERRORS"


def test_digest_mismatch_refuses(tmp_path):
    r = _run(tmp_path, out=b"Result: Succeeded\n", code=0,
             expected={"scripts/trio_validators.py": "sha256:" + "0" * 64})
    assert r.outcome == "refused" and r.code == "T2_DIGEST_MISMATCH"


def test_protected_paths_extended():
    assert "scripts/trio_validators.py" in T2_PROTECTED_PATHS
    assert any(p.startswith("scripts/trio_native_") for p in T2_PROTECTED_PATHS)
