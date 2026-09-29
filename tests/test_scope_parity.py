"""OV4-2 — scope-gate parity, Runner (Python) side.

Runs the SHARED fixture set through the Runner's own port and requires the same
(ok, codes) the TypeScript gate produces, so the Runner's recomputation cannot
drift from the compiler's gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from overnight_runner.scope_gate import evaluate_scope

FIXTURES = json.loads((Path(__file__).parent / "parity" / "scope-fixtures.json").read_text(encoding="utf-8"))


def test_the_shared_fixture_set_is_present() -> None:
    assert FIXTURES["schemaVersion"] == "trio.scope-parity.v1"
    assert len(FIXTURES["fixtures"]) >= 30


@pytest.mark.parametrize("fixture", FIXTURES["fixtures"], ids=[f["name"] for f in FIXTURES["fixtures"]])
def test_parity_with_the_typescript_gate(fixture: dict) -> None:
    ok, refusals = evaluate_scope(fixture["contract"], fixture["changes"], fixture["entries"])
    assert {"ok": ok, "codes": sorted({r.code for r in refusals})} == fixture["expected"]
