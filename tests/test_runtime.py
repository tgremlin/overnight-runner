"""Runtime tests: lock, PAUSED, fingerprint."""
import os
import time
from pathlib import Path

import pytest

from overnight_runner.runtime import (
    is_paused,
    lock_path,
    paused_path,
    require_not_paused,
    runner_lock,
    runtime_fingerprint,
    state_dir,
)


def test_lock_excludes(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OVERNIGHT_STATE_DIR", str(tmp_path))
    with runner_lock():
        with pytest.raises(RuntimeError):
            with runner_lock():
                pass


def test_lock_releases(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OVERNIGHT_STATE_DIR", str(tmp_path))
    with runner_lock():
        pass
    with runner_lock():
        pass


def test_paused_sentinel(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OVERNIGHT_STATE_DIR", str(tmp_path))
    pp = paused_path()
    assert not is_paused()
    require_not_paused()
    pp.write_text("paused\n")
    assert is_paused()
    with pytest.raises(RuntimeError):
        require_not_paused()
    pp.unlink()
    assert not is_paused()


def test_runtime_fingerprint_changes_with_content(tmp_path: Path):
    a = tmp_path / "a.py"
    a.write_text("v1")
    b = tmp_path / "b.py"
    b.write_text("v1")
    fp1 = runtime_fingerprint([a, b])
    b.write_text("v2")
    fp2 = runtime_fingerprint([a, b])
    assert fp1.sha256 != fp2.sha256
    assert fp1.files == 2
