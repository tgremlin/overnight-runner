"""M5/H1 proposal tests: (b) handler entry point, (c) execution class, (d) wiring,
(e) intake. Additive; run against the proposal clone.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from overnight_runner.admission import derive_admission
from overnight_runner.admission_trust_boundary import (
    TRUST_BOUNDARY_FLAG,
    apply_trust_boundary,
    trust_boundary_enabled,
)
from overnight_runner.execution_class import (
    ExecutionClass,
    admit_execution_class,
    record_runtime_failure,
)
from overnight_runner.execution_state import (
    ProfileCapacityState,
    ProfileRuntimeState,
    RunnerErrorClass,
)
from overnight_runner.handlers import admit_chunk_extension, load_mapping
from overnight_runner.intake import intake_scope_gate, intake_validator_evidence
from overnight_runner.workspace_snapshot import snapshot_workspace
from overnight_runner.plan_verify import content_digest

MAPPING = {"version": "trio.runner-mapping.v1", "hostActions": ["write_workspace_evidence", "run_pytest"], "validators": ["py_compile", "pytest"]}


class Spec:
    """Stands in for the Runner's ChunkSpec."""

    def __init__(self, chunk_id: str, idempotency_key: str) -> None:
        self.chunk_id = chunk_id
        self.idempotency_key = idempotency_key


def sidecar(chunk_id="CT-1", artifact="a" * 64, key="idem-1", **over):
    record = {
        "schema_version": "trio.chunk-extension.v1",
        "candidate_fingerprint": artifact,
        "chunk_id": chunk_id,
        "contract_id": chunk_id,
        "idempotency_key": key,
        **over,
    }
    record["digest"] = content_digest({k: v for k, v in record.items() if k != "digest"})
    return record


def test_b_admits_a_verified_extension_with_allowlisted_effects() -> None:
    decision = admit_chunk_extension(
        sidecar=sidecar(host_actions=["run_pytest"], validator_ids=["pytest"]),
        chunk_spec=Spec("CT-1", "idem-1"),
        registered_approved_artifact_id="a" * 64,
        mapping=MAPPING,
    )
    assert decision.ok, decision


def test_b_refuses_an_unverified_extension_and_says_stop() -> None:
    bad = sidecar()
    bad["digest"] = "0" * 64
    decision = admit_chunk_extension(sidecar=bad, chunk_spec=Spec("CT-1", "idem-1"), registered_approved_artifact_id="a" * 64, mapping=MAPPING)
    assert not decision.ok
    assert decision.code == "EXTENSION_DIGEST_MISMATCH"
    # (f) wired for real: the action is stop
    assert decision.action == "stop"


def test_b_refuses_unknown_host_actions_and_validators() -> None:
    unknown_action = admit_chunk_extension(sidecar=sidecar(host_actions=["rm_rf"]), chunk_spec=Spec("CT-1", "idem-1"), registered_approved_artifact_id="a" * 64, mapping=MAPPING)
    assert (unknown_action.ok, unknown_action.code) == (False, "HOST_ACTION_NOT_ALLOWED")
    unknown_validator = admit_chunk_extension(sidecar=sidecar(validator_ids=["bash"]), chunk_spec=Spec("CT-1", "idem-1"), registered_approved_artifact_id="a" * 64, mapping=MAPPING)
    assert (unknown_validator.ok, unknown_validator.code) == (False, "VALIDATOR_NOT_ALLOWED")


def test_b_refuses_a_sidecar_for_another_artifact() -> None:
    decision = admit_chunk_extension(sidecar=sidecar(artifact="b" * 64), chunk_spec=Spec("CT-1", "idem-1"), registered_approved_artifact_id="a" * 64, mapping=MAPPING)
    assert (decision.ok, decision.code) == (False, "EXTENSION_ARTIFACT_MISMATCH")


def test_b_load_mapping_validates_the_version(tmp_path) -> None:
    good = tmp_path / "m.json"
    good.write_text(json.dumps(MAPPING), encoding="utf-8")
    assert load_mapping(str(good))["hostActions"] == MAPPING["hostActions"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": "nope"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_mapping(str(bad))


def test_c_source_only_and_host_tests_are_enabled() -> None:
    assert admit_execution_class(ExecutionClass.SOURCE_ONLY).ok
    assert admit_execution_class(ExecutionClass.HOST_TESTS).ok


def test_c_native_render_and_automation_are_blocked_with_reason() -> None:
    for cls in (ExecutionClass.NATIVE_COMPILE, ExecutionClass.RENDERED_EDITOR, ExecutionClass.STANDALONE_AUTOMATION):
        decision = admit_execution_class(cls)
        assert not decision.ok
        assert decision.code == "UNSUPPORTED_ON_THIS_HOST_STAGE"
        assert decision.detail


def test_c_human_review_needs_a_gate() -> None:
    decision = admit_execution_class(ExecutionClass.HUMAN_REVIEW)
    assert (decision.ok, decision.needs_human_gate) == (False, True)


def test_d_wired_failure_records_capacity_wait_or_stops() -> None:
    state = ProfileRuntimeState(profile_id="p1")
    waiting, action = record_runtime_failure(state, RunnerErrorClass.QUOTA_EXHAUSTED, 1000, reset_at=2000)
    assert action == "failover"
    assert waiting.capacity_state == ProfileCapacityState.CAPACITY_WAIT
    assert waiting.cooldown_until == 2000
    stopped, action2 = record_runtime_failure(state, RunnerErrorClass.EFFECT_UNKNOWN, 1000)
    assert action2 == "stop"
    assert stopped.probe_required is True


def _ws(tmp_path, files=None):
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True, exist_ok=True)
    (ws / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    for rel, content in (files or {}).items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return str(ws)


CONTRACT = {"allowedWritePaths": ["src/a.py"], "protectedPaths": ["secrets"]}


def _digest_for(ws, pre, contract=CONTRACT):
    """Run the recomputation once to learn the honest digest."""
    decision = intake_scope_gate(submitted={"allowed": True, "digest": "sha256:seed"}, workspace_dir=ws, pre_snapshot=pre, contract=contract)
    return decision.digest


def test_e_recomputes_the_change_set_from_the_workspace(tmp_path) -> None:
    ws = _ws(tmp_path)
    pre = snapshot_workspace(ws)
    (Path(ws) / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
    digest = _digest_for(ws, pre)
    ok = intake_scope_gate(submitted={"allowed": True, "digest": digest}, workspace_dir=ws, pre_snapshot=pre, contract=CONTRACT)
    assert ok.ok, ok
    assert ok.recomputed["changes"] == 1


def test_e_refuses_a_change_the_caller_never_mentioned(tmp_path) -> None:
    ws = _ws(tmp_path)
    pre = snapshot_workspace(ws)
    (Path(ws) / "evil.py").write_text("y = 1\n", encoding="utf-8")
    digest = _digest_for(ws, pre)
    # even with a truthful digest, the recomputed refusal wins
    decision = intake_scope_gate(submitted={"allowed": True, "digest": digest}, workspace_dir=ws, pre_snapshot=pre, contract=CONTRACT)
    assert not decision.ok
    assert decision.code == "SCOPE_WRITE_OUT_OF_SCOPE"


def test_e_ignores_a_caller_supplied_change_list(tmp_path) -> None:
    ws = _ws(tmp_path)
    pre = snapshot_workspace(ws)
    (Path(ws) / "evil.py").write_text("y = 1\n", encoding="utf-8")
    digest = _digest_for(ws, pre)
    decision = intake_scope_gate(
        submitted={"allowed": True, "digest": digest, "changes": [{"kind": "modified", "path": "src/a.py"}]},
        workspace_dir=ws, pre_snapshot=pre, contract=CONTRACT,
    )
    assert not decision.ok and decision.code == "SCOPE_WRITE_OUT_OF_SCOPE"


def test_e_refuses_a_candidate_created_git_directory(tmp_path) -> None:
    ws = _ws(tmp_path)
    pre = snapshot_workspace(ws)
    (Path(ws) / ".git" / "hooks").mkdir(parents=True, exist_ok=True)
    (Path(ws) / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    digest = _digest_for(ws, pre)
    decision = intake_scope_gate(submitted={"allowed": True, "digest": digest}, workspace_dir=ws, pre_snapshot=pre, contract=CONTRACT)
    assert not decision.ok
    assert "SCOPE_GIT_INTERNAL" in (decision.recomputed or {}).get("codes", []) or decision.code == "SCOPE_GIT_INTERNAL"


def test_e_missing_or_none_digests_fail_on_either_side(tmp_path) -> None:
    ws = _ws(tmp_path)
    pre = snapshot_workspace(ws)
    assert intake_scope_gate(submitted=None, workspace_dir=ws, pre_snapshot=pre, contract=CONTRACT).code == "SCOPE_EVIDENCE_MISSING"
    assert intake_scope_gate(submitted={"allowed": True}, workspace_dir=ws, pre_snapshot=pre, contract=CONTRACT).code == "EVIDENCE_DIGEST_MISSING"
    assert intake_scope_gate(submitted={"allowed": True, "digest": None}, workspace_dir=ws, pre_snapshot=pre, contract=CONTRACT).code == "EVIDENCE_DIGEST_MISSING"
    assert intake_scope_gate(submitted={"allowed": True, "digest": "None"}, workspace_dir=ws, pre_snapshot=pre, contract=CONTRACT).code == "EVIDENCE_DIGEST_MISSING"
    # the validator side, both directions
    assert intake_validator_evidence(submitted={"validator_results": [], "digest": None}, recomputed={"digest": "sha256:a", "passed": True}).code == "EVIDENCE_DIGEST_MISSING"
    assert intake_validator_evidence(submitted={"validator_results": [], "digest": "sha256:a"}, recomputed={"digest": None, "passed": True}).code == "EVIDENCE_DIGEST_MISSING"
    assert intake_validator_evidence(submitted={"validator_results": [], "digest": "sha256:a"}, recomputed=None).code == "EVIDENCE_DIGEST_MISSING"


def test_e_a_submitted_refusal_is_never_ok_true(tmp_path) -> None:
    ws = _ws(tmp_path)
    pre = snapshot_workspace(ws)
    (Path(ws) / "src" / "a.py").write_text("x = 3\n", encoding="utf-8")
    digest = _digest_for(ws, pre)
    decision = intake_scope_gate(submitted={"allowed": False, "digest": digest}, workspace_dir=ws, pre_snapshot=pre, contract=CONTRACT)
    assert not decision.ok
    assert decision.code == "SCOPE_SUBMITTED_REFUSAL"
    # the validator side too
    assert not intake_validator_evidence(submitted={"validator_results": [], "digest": "sha256:a", "passed": False}, recomputed={"digest": "sha256:a", "passed": False}).ok


def test_e_recomputes_validator_evidence() -> None:
    recomputed = {"digest": "sha256:abc", "passed": False}
    assert intake_validator_evidence(submitted={"digest": "sha256:abc", "passed": True, "validator_results": []}, recomputed=recomputed).code == "EVIDENCE_DISAGREES"
    assert intake_validator_evidence(submitted={"digest": "sha256:zzz", "passed": False, "validator_results": []}, recomputed=recomputed).code == "EVIDENCE_DISAGREES"
    assert intake_validator_evidence(submitted={"digest": "sha256:abc"}, recomputed=recomputed).code == "EVIDENCE_INCOMPLETE"
    assert intake_validator_evidence(submitted=None, recomputed=recomputed).code == "EVIDENCE_MISSING"
    assert intake_validator_evidence(submitted={"digest": "sha256:aaa", "passed": True, "validator_results": []}, recomputed={"digest": "sha256:aaa", "passed": True}).ok


# --------------------------------------------------------------------------- #
# §OV4-3: the admission trust boundary, behind a versioned default-safe flag
# --------------------------------------------------------------------------- #
def test_ov43_the_flag_is_versioned_and_default_safe() -> None:
    assert trust_boundary_enabled(None) is False
    assert trust_boundary_enabled("") is False
    assert trust_boundary_enabled("true") is False
    assert trust_boundary_enabled("trio.admission-trust-boundary") is False
    assert trust_boundary_enabled(TRUST_BOUNDARY_FLAG) is True


def test_ov43_off_means_nothing_changes(tmp_path) -> None:
    decision = apply_trust_boundary()  # no inputs at all
    assert decision.ok and decision.applied is False
    # every unwired shape is fine while the flag is off
    assert apply_trust_boundary(execution_class=None, workspace_dir=None, flag=None).ok


def test_ov43_on_refuses_unwired_inputs() -> None:
    on = TRUST_BOUNDARY_FLAG
    d1 = apply_trust_boundary(flag=on)
    assert (d1.ok, d1.code, d1.applied) == (False, "UNWIRED_EXECUTION_CLASS", True)
    d2 = apply_trust_boundary(execution_class=ExecutionClass.HOST_TESTS, flag=on)
    assert (d2.ok, d2.code) == (False, "UNWIRED_WORKSPACE")
    d3 = apply_trust_boundary(execution_class=ExecutionClass.NATIVE_COMPILE, flag=on)
    assert (d3.ok, d3.code) == (False, "UNSUPPORTED_ON_THIS_HOST_STAGE")
    assert apply_trust_boundary(execution_class="teleport", flag=on).code == "UNKNOWN_EXECUTION_CLASS"


def test_ov43_on_refuses_missing_evidence(tmp_path) -> None:
    ws = _ws(tmp_path)
    pre = snapshot_workspace(ws)
    (Path(ws) / "src" / "a.py").write_text("x = 9\n", encoding="utf-8")
    digest = _digest_for(ws, pre)
    decision = apply_trust_boundary(
        execution_class=ExecutionClass.HOST_TESTS, workspace_dir=ws, pre_snapshot=pre, contract=CONTRACT,
        submitted_scope={"allowed": True, "digest": digest}, submitted_evidence=None, flag=TRUST_BOUNDARY_FLAG,
    )
    assert (decision.ok, decision.code) == (False, "UNWIRED_VALIDATOR_EVIDENCE")


def test_ov43_on_admits_a_fully_wired_clean_run(tmp_path) -> None:
    ws = _ws(tmp_path)
    pre = snapshot_workspace(ws)
    (Path(ws) / "src" / "a.py").write_text("x = 10\n", encoding="utf-8")
    digest = _digest_for(ws, pre)
    evidence = {"digest": "sha256:ev", "passed": True}
    decision = apply_trust_boundary(
        execution_class=ExecutionClass.HOST_TESTS, workspace_dir=ws, pre_snapshot=pre, contract=CONTRACT,
        submitted_scope={"allowed": True, "digest": digest},
        submitted_evidence={"digest": "sha256:ev", "passed": True, "validator_results": []},
        recomputed_evidence=evidence, flag=TRUST_BOUNDARY_FLAG,
    )
    assert decision.ok, decision
    assert decision.applied is True


def test_ov43_derive_admission_is_unchanged_when_the_flag_is_off() -> None:
    import inspect
    sig = inspect.signature(derive_admission)
    # the new parameters are OPTIONAL, so an existing call site is untouched
    assert sig.parameters["trust_boundary_flag"].default is None
    assert sig.parameters["trust_boundary_inputs"].default is None
