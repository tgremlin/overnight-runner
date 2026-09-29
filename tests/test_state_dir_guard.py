"""§ACT1-0 — an unset or REAL `OVERNIGHT_STATE_DIR` is a hard refusal.

Rows required by the kickoff: unset → refused; the real path (and a SYMLINK to it)
→ refused; a temp path → passes. Plus positive controls, and the same refusal on
the Runner-owned validator-evidence / host-tests scratch roots.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from overnight_runner.state_dir_guard import (
    ACTIVATION_FLAG,
    STATE_DIR_ENV,
    STATE_DIR_UNSET_OR_REAL,
    StateDirRefusal,
    assert_usable_state_dir,
    real_state_paths,
    state_dir_is_usable,
)
from overnight_runner.validator_evidence import evidence_root, host_tests_scratch_root, scratch_root


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch) -> Path:
    """A hermetic `~` so the rows never read the operator's real state dir."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def test_unset_is_refused(fake_home, monkeypatch):
    monkeypatch.delenv(STATE_DIR_ENV, raising=False)
    with pytest.raises(StateDirRefusal) as excinfo:
        assert_usable_state_dir(home=fake_home)
    assert excinfo.value.code == STATE_DIR_UNSET_OR_REAL
    assert STATE_DIR_UNSET_OR_REAL in str(excinfo.value)
    assert state_dir_is_usable(home=fake_home) is False


def test_empty_is_refused(fake_home):
    with pytest.raises(StateDirRefusal):
        assert_usable_state_dir("   ", home=fake_home)


def test_the_real_state_dir_is_refused(fake_home):
    for real in real_state_paths(fake_home):
        with pytest.raises(StateDirRefusal) as excinfo:
            assert_usable_state_dir(str(real), home=fake_home)
        assert excinfo.value.code == STATE_DIR_UNSET_OR_REAL


def test_a_symlink_to_the_real_state_dir_is_refused(fake_home):
    link = fake_home.parent / "runner-state-link"
    link.symlink_to(fake_home / ".trio" / "runner-state")
    with pytest.raises(StateDirRefusal) as excinfo:
        assert_usable_state_dir(str(link), home=fake_home)
    assert excinfo.value.code == STATE_DIR_UNSET_OR_REAL


def test_a_temp_path_passes(fake_home):
    temp = fake_home.parent / "state"
    assert assert_usable_state_dir(str(temp), home=fake_home) == temp
    assert state_dir_is_usable(str(temp), home=fake_home) is True


def test_only_the_documented_activation_flag_allows_the_real_dir(fake_home, monkeypatch):
    """POSITIVE CONTROL for the escape hatch: it is the flag, not the path, that opens it."""
    real = real_state_paths(fake_home)[1]
    monkeypatch.setenv(ACTIVATION_FLAG, "1")
    assert assert_usable_state_dir(str(real), home=fake_home) == real
    monkeypatch.setenv(ACTIVATION_FLAG, "0")
    with pytest.raises(StateDirRefusal):
        assert_usable_state_dir(str(real), home=fake_home)


def test_runtime_state_dir_refuses_and_writes_nothing(fake_home, monkeypatch):
    """End to end: the fallback is gone and no real directory is created."""
    from overnight_runner.runtime import state_dir

    monkeypatch.delenv(STATE_DIR_ENV, raising=False)
    with pytest.raises(StateDirRefusal):
        state_dir()
    assert not (fake_home / ".trio" / "runner-state").exists()
    assert not (fake_home / ".local" / "state" / "overnight-runner").exists()

    temp = fake_home.parent / "state"
    monkeypatch.setenv(STATE_DIR_ENV, str(temp))
    assert state_dir() == temp
    assert temp.is_dir()


def test_the_scratch_roots_refuse_an_unset_state_dir(fake_home, monkeypatch):
    """The Runner-owned scratch roots are Runner-side entry points too."""
    monkeypatch.delenv(STATE_DIR_ENV, raising=False)
    for root_fn in (scratch_root, evidence_root, host_tests_scratch_root):
        with pytest.raises(StateDirRefusal) as excinfo:
            root_fn()
        assert excinfo.value.code == STATE_DIR_UNSET_OR_REAL


def test_the_scratch_roots_refuse_a_root_inside_the_real_state_dir(fake_home, monkeypatch):
    """Even an explicit evidence override cannot put the scratch root in the real dir."""
    monkeypatch.setenv(STATE_DIR_ENV, str(fake_home.parent / "state"))
    monkeypatch.setenv("TRIO_RUNNER_EVIDENCE_ROOT", str(real_state_paths(fake_home)[1] / "validator-evidence"))
    with pytest.raises(StateDirRefusal):
        evidence_root()


def test_the_scratch_roots_pass_with_an_isolated_state_dir(fake_home, monkeypatch):
    """POSITIVE CONTROL: a real temp state dir is all the scratch roots need."""
    monkeypatch.setenv(STATE_DIR_ENV, str(fake_home.parent / "state"))
    monkeypatch.setenv("TRIO_RUNNER_EVIDENCE_ROOT", str(fake_home.parent / "runner-scratch"))
    assert evidence_root().is_dir()
    assert host_tests_scratch_root().is_dir()
    assert str(evidence_root()).startswith(str(fake_home.parent))
    assert os.environ[STATE_DIR_ENV] != str(real_state_paths(fake_home)[0])
