"""M5/H1 proposal tests (additive; run against the proposal clone).

Covers patch (a) verified plan registration, patch (d) the normalized error-class
vocabulary + capacity state, and patch (f) EFFECT_UNKNOWN handling.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from overnight_runner.db import Database
from overnight_runner.execution_state import (
    EXECUTION_STATE_SCHEMA_VERSION,
    ProfileCapacityState,
    ProfileRuntimeState,
    RunnerErrorClass,
    apply_runtime_outcome,
    fallback_disposition,
    probe_candidates,
    resolve_reset_time,
    unknown_effect_action,
)
from overnight_runner.plans import (
    PlanVerificationError,
    plan_projection_digest,
    register_plan,
    register_plan_verified,
)
from overnight_runner.safety import SafetyError


def _db() -> Database:
    tmp = tempfile.TemporaryDirectory(prefix="m5-proposal-")
    return Database(Path(tmp.name) / "t.sqlite")


def test_plan_projection_digest_is_unchanged() -> None:
    # the extracted formula must equal the original register_plan digest, including
    # the golden vector recorded by the forge compiler tests
    db = _db()
    digest = register_plan(
        db,
        plan_id="P",
        approved_artifact_id="a" * 64,
        work_package_criterion_ids={"b": {"y", "x"}, "a": {"m", "k"}},
    )
    assert digest == "4e2d8ad033c41bdc7edca7a73ae7af66c1f8c59b130355aacaffcf66ba2fa981"
    assert plan_projection_digest("P", {"b": {"y", "x"}, "a": {"m", "k"}}) == digest


def _artifact(tmp_path, plan_id="Q", wpci=None):
    wpci = wpci if wpci is not None else {"pkg-1": ["crit-1"]}
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({"plan_id": plan_id, "work_package_criterion_ids": wpci}), encoding="utf-8")
    return str(path)


def test_register_plan_verified_accepts_a_matching_artifact(tmp_path) -> None:
    db = _db()
    wpci = {"pkg-1": {"crit-1"}}
    digest = register_plan_verified(
        db,
        plan_id="Q",
        approved_artifact_id="b" * 64,
        work_package_criterion_ids=wpci,
        artifact_path=_artifact(tmp_path),
    )
    assert digest == plan_projection_digest("Q", wpci)


def test_register_plan_verified_refuses_a_mismatched_artifact(tmp_path) -> None:
    db = _db()
    with pytest.raises(PlanVerificationError) as exc:
        register_plan_verified(
            db,
            plan_id="Q",
            approved_artifact_id="b" * 64,
            work_package_criterion_ids={"pkg-1": {"crit-1"}},
            artifact_path=_artifact(tmp_path, wpci={"pkg-1": ["crit-2"]}),
        )
    assert exc.value.code == "ARTIFACT_MISMATCH"
    from overnight_runner.plans import load_plan_digest
    assert load_plan_digest(db, "Q") is None


def test_register_plan_verified_refuses_a_missing_or_wrong_artifact(tmp_path) -> None:
    db = _db()
    with pytest.raises(PlanVerificationError) as missing:
        register_plan_verified(
            db, plan_id="Q", approved_artifact_id="b" * 64,
            work_package_criterion_ids={"pkg-1": {"crit-1"}},
            artifact_path=str(tmp_path / "nope.json"),
        )
    assert missing.value.code == "ARTIFACT_UNREADABLE"
    with pytest.raises(PlanVerificationError) as wrongplan:
        register_plan_verified(
            db, plan_id="Q", approved_artifact_id="b" * 64,
            work_package_criterion_ids={"pkg-1": {"crit-1"}},
            artifact_path=_artifact(tmp_path, plan_id="OTHER"),
        )
    assert wrongplan.value.code == "ARTIFACT_MISMATCH"


def test_register_plan_verified_refuses_a_wrong_pinned_digest(tmp_path) -> None:
    db = _db()
    with pytest.raises(PlanVerificationError) as exc:
        register_plan_verified(
            db, plan_id="Q", approved_artifact_id="b" * 64,
            work_package_criterion_ids={"pkg-1": {"crit-1"}},
            artifact_path=_artifact(tmp_path),
            expected_plan_digest="0" * 64,
        )
    assert exc.value.code == "PROJECTION_DIGEST_MISMATCH"


def test_fallback_disposition_and_unknown_effect() -> None:
    assert fallback_disposition(RunnerErrorClass.RATE_LIMITED) == "availability"
    assert fallback_disposition(RunnerErrorClass.AUTH_FAILED) == "availability"
    assert fallback_disposition(RunnerErrorClass.INVALID_REQUEST) == "stop"
    assert fallback_disposition(RunnerErrorClass.MAX_TURNS) == "stop"
    assert fallback_disposition(RunnerErrorClass.EFFECT_UNKNOWN) == "unknown-effect"
    assert unknown_effect_action() == "stop"


def test_runtime_state_capacity_wait_cooldown_and_probe() -> None:
    state = ProfileRuntimeState(profile_id="p1")
    assert state.schema_version == EXECUTION_STATE_SCHEMA_VERSION

    waiting = apply_runtime_outcome(state, RunnerErrorClass.QUOTA_EXHAUSTED, 1000, reset_at=2000)
    assert waiting.capacity_state == ProfileCapacityState.CAPACITY_WAIT
    assert waiting.cooldown_until == 2000
    assert waiting.probe_required is True

    # before expiry: not a probe candidate
    assert probe_candidates([waiting], ["p1"], 1500) == []
    # after expiry: a probe candidate, still skipped until probed
    assert probe_candidates([waiting], ["p1"], 2500) == ["p1"]

    # a successful probe restores READY
    restored = apply_runtime_outcome(waiting, None, 2600)
    assert restored.capacity_state == ProfileCapacityState.READY
    assert restored.probe_required is False

    # no reset time → the bounded default backoff
    defaulted = apply_runtime_outcome(state, RunnerErrorClass.RATE_LIMITED, 5000)
    assert defaulted.cooldown_until == 5000 + 10 * 60_000


def test_effect_unknown_marks_probe_required_without_a_cooldown() -> None:
    state = ProfileRuntimeState(profile_id="p1")
    unknown = apply_runtime_outcome(state, RunnerErrorClass.EFFECT_UNKNOWN, 10)
    assert unknown.capacity_state == ProfileCapacityState.CAPACITY_WAIT
    assert unknown.cooldown_until is None
    assert unknown.probe_required is True
    assert probe_candidates([unknown], ["p1"], 10_000_000) == []


def test_resolve_reset_time_only_understands_the_documented_form() -> None:
    assert resolve_reset_time("Plan usage limit reached; resets in 15m", 1000) == 1000 + 15 * 60_000
    assert resolve_reset_time("no reset information here", 1000) is None
    assert resolve_reset_time(None, 1000) is None
