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
    """§EXEC3 fix 2 — every test is isolated from the shared Runner dirs.

    Two things this guarantees, both of which a test suite must never get wrong:
      * the Runner's OWN evidence/scratch root is per test and inside pytest's
        tmp_path, so a test cannot accumulate directories in the shared root;
      * the state dir is per test, so no test can write into the REAL state
        directory when it forgets to set OVERNIGHT_STATE_DIR.
    """
    from overnight_runner.validator_evidence import EVIDENCE_ROOT_ENV

    monkeypatch.setenv("OVERNIGHT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv(EVIDENCE_ROOT_ENV, str(tmp_path / "runner-scratch"))


def pytest_collection_modifyitems(config, items):
    """Pass-through hook so a future sub-dir's conftest inherits the
    sys.path entry."""
    return None
