"""Pytest configuration: enable campaign-v2 by default for the test
session. Each test sets OVERNIGHT_STATE_DIR via setUp; this file
unconditionally enables the campaign-v2 feature gate so the
existing P06 tests continue to work. Real deployments leave the
gate disabled.
"""
import os
import sys

import pytest
from pathlib import Path

# Enable campaign-v2 for the entire pytest session.
os.environ["TR_P06_CAMPAIGN_V2"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(autouse=True)
def _isolated_runner_dirs(tmp_path, monkeypatch):
    """§EXEC3 fix 2 — every test gets its OWN Runner scratch root.

    The shared root used to accumulate one directory per evidence run, and the first
    version of that root was nested in the state dir, so a test that left
    OVERNIGHT_STATE_DIR unset wrote into the configured — possibly REAL — state
    directory. Both are now impossible: the scratch root is a per-test tmp_path, and
    the Runner's default root is never the state dir. The state dir itself is
    deliberately left alone here: the approval-binding rows need the bindings the
    session already has.
    """
    from overnight_runner.validator_evidence import EVIDENCE_ROOT_ENV

    monkeypatch.setenv(EVIDENCE_ROOT_ENV, str(tmp_path / "runner-scratch"))


def pytest_collection_modifyitems(config, items):
    """Pass-through hook so a future sub-dir's conftest inherits the
    sys.path entry."""
    return None
