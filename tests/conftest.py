"""Pytest configuration: enable campaign-v2 by default for the test
session. Each test sets OVERNIGHT_STATE_DIR via setUp; this file
unconditionally enables the campaign-v2 feature gate so the
existing P06 tests continue to work. Real deployments leave the
gate disabled.
"""
import os
import sys
from pathlib import Path

# Enable campaign-v2 for the entire pytest session.
os.environ["TR_P06_CAMPAIGN_V2"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def pytest_collection_modifyitems(config, items):
    """Pass-through hook so a future sub-dir's conftest inherits the
    sys.path entry."""
    return None
