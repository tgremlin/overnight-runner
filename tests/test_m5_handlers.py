"""M5/H1 proposal tests: (b) handler entry point, (c) execution class, (d) wiring,
(e) intake. Additive; run against the proposal clone.
"""
from __future__ import annotations

import json

import pytest

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


def test_e_recomputes_the_scope_decision_and_refuses_a_false_pass() -> None:
    changes = [{"kind": "modified", "path": "src/a.py"}, {"kind": "added", "path": "src/evil.py"}]
    decision = intake_scope_gate(submitted={"allowed": True}, changes=changes, allowed_write_paths=["src/a.py"])
    assert (decision.ok, decision.code) == (False, "SCOPE_EVIDENCE_DISAGREES")
    ok = intake_scope_gate(submitted={"allowed": True}, changes=[{"kind": "modified", "path": "src/a.py"}], allowed_write_paths=["src/a.py"])
    assert ok.ok
    # a submitted refusal is fine (the Runner only tightens)
    assert intake_scope_gate(submitted={"allowed": False}, changes=changes, allowed_write_paths=["src/a.py"]).ok
    # no submission at all is a refusal, never a default-allow
    assert intake_scope_gate(submitted=None, changes=[], allowed_write_paths=[]).code == "SCOPE_EVIDENCE_MISSING"


def test_e_recomputes_validator_evidence() -> None:
    recomputed = {"digest": "sha256:abc", "passed": False}
    assert intake_validator_evidence(submitted={"digest": "sha256:abc", "passed": False, "validator_results": []}, recomputed=recomputed).ok
    # a submitted PASS the recomputation does not support
    forged = intake_validator_evidence(submitted={"digest": "sha256:abc", "passed": True, "validator_results": []}, recomputed=recomputed)
    assert (forged.ok, forged.code) == (False, "EVIDENCE_DISAGREES")
    # a digest mismatch
    mismatched = intake_validator_evidence(submitted={"digest": "sha256:zzz", "validator_results": []}, recomputed=recomputed)
    assert (mismatched.ok, mismatched.code) == (False, "EVIDENCE_DISAGREES")
    # missing required fields
    assert intake_validator_evidence(submitted={"digest": "sha256:abc"}, recomputed=recomputed).code == "EVIDENCE_INCOMPLETE"
    assert intake_validator_evidence(submitted=None, recomputed=recomputed).code == "EVIDENCE_MISSING"
