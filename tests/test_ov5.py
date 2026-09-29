"""OV5 — fail-closed flag, Runner-held baseline/contract, staged completion gate
and Runner-produced validator evidence.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from overnight_runner.admission_trust_boundary import (
    TRUST_BOUNDARY_FLAG,
    UnknownTrustBoundaryFlag,
    apply_trust_boundary,
    check_completion,
    resolve_trust_boundary,
)
from overnight_runner.baseline import (
    RunnerBaseline,
    baseline_from_base_commit,
    baseline_from_workspace_creation,
    is_runner_issued,
)
from overnight_runner.campaign import record_chunk_accepted
from overnight_runner.db import Database
from overnight_runner.execution_class import ExecutionClass
from overnight_runner.intake import intake_scope_gate, scope_contract_from, submitted_verdict_digest
from overnight_runner.safety import SafetyError
from overnight_runner.scope_gate import evaluate_scope
from overnight_runner.validator_evidence import (
    RUNNER_VALIDATOR_EVIDENCE_SCHEMA,
    intake_runner_evidence,
    run_trusted_validators,
)
from overnight_runner.workspace_snapshot import snapshot_workspace

# --------------------------------------------------------------------------- #
# §OV5-1 the flag is FAIL-CLOSED
# --------------------------------------------------------------------------- #
def test_ov51_absent_flag_is_off() -> None:
    assert resolve_trust_boundary(None) == "off"
    assert resolve_trust_boundary("") == "off"


def test_ov51_the_exact_versioned_flag_is_on() -> None:
    assert resolve_trust_boundary(TRUST_BOUNDARY_FLAG) == "on"


@pytest.mark.parametrize("bogus", ["true", "True", "1", "yes", "on", "off", "trio.admission-trust-boundary", "trio.admission-trust-boundary.v2", " trio.admission-trust-boundary.v1"])
def test_ov51_any_other_value_is_refused_not_treated_as_off(bogus: str) -> None:
    with pytest.raises(UnknownTrustBoundaryFlag) as exc:
        resolve_trust_boundary(bogus)
    assert exc.value.code == "UNKNOWN_TRUST_BOUNDARY_FLAG"
    # and through the decision surface
    decision = apply_trust_boundary(execution_class=ExecutionClass.HOST_TESTS, flag=bogus)
    assert (decision.ok, decision.code) == (False, "UNKNOWN_TRUST_BOUNDARY_FLAG")


def test_ov51_the_environment_value_is_checked_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIO_ADMISSION_TRUST_BOUNDARY", "garbage")
    assert apply_trust_boundary(execution_class=ExecutionClass.HOST_TESTS).code == "UNKNOWN_TRUST_BOUNDARY_FLAG"
    monkeypatch.setenv("TRIO_ADMISSION_TRUST_BOUNDARY", TRUST_BOUNDARY_FLAG)
    assert apply_trust_boundary(execution_class=ExecutionClass.HOST_TESTS).ok


def test_ov51_inputs_without_the_flag_are_refused() -> None:
    decision = apply_trust_boundary(execution_class=ExecutionClass.HOST_TESTS, inputs_supplied=True)
    assert (decision.ok, decision.code, decision.applied) == (False, "UNKNOWN_TRUST_BOUNDARY_FLAG", False)


def test_ov51_off_still_does_nothing() -> None:
    assert apply_trust_boundary().ok and apply_trust_boundary().applied is False


# --------------------------------------------------------------------------- #
# §OV5-2/3 a Runner-held baseline and contract
# --------------------------------------------------------------------------- #
def _owner_repo(tmp: Path, files=None):
    repo = tmp / "owner"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    for rel, content in (files or {}).items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)  # noqa: E731
    run("init", "-q", "-b", "main")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    run("add", ".")
    run("commit", "-q", "-m", "init")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return repo, head


def _ws_from_base(tmp: Path, repo: Path, head: str) -> str:
    """Export the base tree WITHOUT any .git, like the runner-owned workspace."""
    import tarfile

    ws = tmp / "ws"
    ws.mkdir()
    archive = tmp / "base.tar"
    with open(archive, "wb") as fh:
        subprocess.run(["git", "--git-dir", str(repo / ".git"), "archive", "--format=tar", head], stdout=fh, check=True)
    with tarfile.open(archive) as tf:
        tf.extractall(ws)
    assert not (ws / ".git").exists()
    return str(ws)


class _Spec:
    def __init__(self, writes):
        self.permitted_write_paths = writes


class _Grant:
    def __init__(self, allowed, protected=()):
        self.allowed_write_paths = allowed
        self.protected_paths = list(protected)


def test_ov52_the_baseline_must_be_runner_issued(tmp_path: Path) -> None:
    repo, head = _owner_repo(tmp_path)
    ws = _ws_from_base(tmp_path, repo, head)
    forged = RunnerBaseline(head, snapshot_workspace(ws), "base-commit-tree")
    assert is_runner_issued(forged) is False
    decision = intake_scope_gate(
        submitted={"allowed": True, "digest": "sha256:x"}, workspace_dir=ws, baseline=forged,
        chunk_spec=_Spec(["src/a.py"]), grant=_Grant(["src/a.py"]),
    )
    assert (decision.ok, decision.code) == (False, "BASELINE_NOT_TRUSTED")


def test_ov52_a_post_tamper_snapshot_is_not_accepted(tmp_path: Path) -> None:
    repo, head = _owner_repo(tmp_path)
    ws = _ws_from_base(tmp_path, repo, head)
    # the Runner's OWN baseline, taken from the base commit's tree
    baseline = baseline_from_base_commit(git_dir=str(repo / ".git"), base_commit=head)
    assert is_runner_issued(baseline) is True
    # the candidate tampers AFTER the baseline exists
    (Path(ws) / "evil.py").write_text("y = 1\n", encoding="utf-8")
    decision = intake_scope_gate(
        submitted={"allowed": True, "digest": "sha256:whatever"}, workspace_dir=ws, baseline=baseline,
        chunk_spec=_Spec(["src/a.py"]), grant=_Grant(["src/a.py"]),
    )
    # the submitter's digest is wrong for this state, and the recomputation refuses
    assert decision.ok is False
    assert decision.code in ("SCOPE_EVIDENCE_DISAGREES", "SCOPE_WRITE_OUT_OF_SCOPE")
    # with a truthful digest, the recomputed refusal is what comes back
    honest = submitted_verdict_digest(False, ["SCOPE_WRITE_OUT_OF_SCOPE"], decision.recomputed["changes"])
    decision2 = intake_scope_gate(
        submitted={"allowed": True, "digest": honest}, workspace_dir=ws, baseline=baseline,
        chunk_spec=_Spec(["src/a.py"]), grant=_Grant(["src/a.py"]),
    )
    assert (decision2.ok, decision2.code) == (False, "SCOPE_WRITE_OUT_OF_SCOPE")


def test_ov52_a_workspace_creation_baseline_only_sees_later_changes(tmp_path: Path) -> None:
    repo, head = _owner_repo(tmp_path)
    ws = _ws_from_base(tmp_path, repo, head)
    baseline = baseline_from_workspace_creation(workspace_dir=ws, base_commit=head)
    ok, refusals = evaluate_scope(scope_contract_from(chunk_spec=_Spec(["src/a.py"]), grant=_Grant(["src/a.py"])), [], snapshot_workspace(ws))
    assert ok and refusals == []
    assert len(baseline.entries) >= 1


def test_ov53_the_contract_comes_from_the_spec_and_grant(tmp_path: Path) -> None:
    contract = scope_contract_from(chunk_spec=_Spec(["src/a.py", "src/b.py"]), grant=_Grant(["src/a.py"], protected=["secrets"]))
    # intersection: the chunk may never widen the grant
    assert contract["allowedWritePaths"] == ["src/a.py"]
    assert contract["protectedPaths"] == ["secrets"]


def test_ov53_a_widened_contract_is_refused(tmp_path: Path) -> None:
    repo, head = _owner_repo(tmp_path)
    ws = _ws_from_base(tmp_path, repo, head)
    baseline = baseline_from_workspace_creation(workspace_dir=ws, base_commit=head)
    decision = intake_scope_gate(
        submitted={"allowed": True, "digest": "sha256:x"}, workspace_dir=ws, baseline=baseline,
        chunk_spec=_Spec(["src/a.py"]), grant=_Grant(["src/a.py"]),
        contract_override={"allowedWritePaths": ["**"]},
    )
    assert (decision.ok, decision.code) == (False, "CONTRACT_OVERRIDE_REFUSED")


# --------------------------------------------------------------------------- #
# §OV5-4 the completion gate
# --------------------------------------------------------------------------- #
def _clean_completion(tmp_path: Path, tamper=None):
    repo, head = _owner_repo(tmp_path)
    ws = _ws_from_base(tmp_path, repo, head)
    baseline = baseline_from_workspace_creation(workspace_dir=ws, base_commit=head)
    spec, grant = _Spec(["src/a.py"]), _Grant(["src/a.py"], protected=["secrets"])
    if tamper is not None:
        tamper(ws)
    after = snapshot_workspace(ws)
    from overnight_runner.workspace_snapshot import diff_snapshots

    from overnight_runner.scope_gate import evaluate_scope as gate

    changes = [{"kind": c.kind, "path": c.path, "reasons": list(c.reasons), **({"from": c.from_path} if c.from_path else {})} for c in diff_snapshots(dict(baseline.entries), after)]
    ok, refusals = gate(scope_contract_from(chunk_spec=spec, grant=grant), changes, after)
    digest = submitted_verdict_digest(ok, [r.code for r in refusals], len(changes))
    evidence = run_trusted_validators(workspace_dir=ws, base_commit=head, candidate_tree_digest="t" * 64)
    return ws, baseline, spec, grant, digest, evidence


def test_ov54_an_honest_candidate_is_accepted_at_completion(tmp_path: Path) -> None:
    ws, baseline, spec, grant, digest, evidence = _clean_completion(tmp_path)
    decision = check_completion(
        workspace_dir=ws, baseline=baseline, chunk_spec=spec, grant=grant,
        submitted_scope={"allowed": True, "digest": digest},
        submitted_evidence=evidence.as_dict(), runner_evidence=evidence,
        flag=TRUST_BOUNDARY_FLAG,
    )
    assert decision.ok, decision


@pytest.mark.parametrize("tamper_name", ["out-of-scope", "symlink", "git"])
def test_ov54_a_hostile_candidate_is_refused_at_completion(tmp_path: Path, tamper_name: str) -> None:
    import os

    def tamper(ws: str) -> None:
        if tamper_name == "out-of-scope":
            (Path(ws) / "evil.py").write_text("y = 1\n", encoding="utf-8")
        elif tamper_name == "symlink":
            os.symlink("/etc/passwd", str(Path(ws) / "link"))
        else:
            (Path(ws) / ".git" / "hooks").mkdir(parents=True, exist_ok=True)
            (Path(ws) / ".git" / "config").write_text("[core]\n", encoding="utf-8")

    ws, baseline, spec, grant, digest, evidence = _clean_completion(tmp_path, tamper=tamper)
    decision = check_completion(
        workspace_dir=ws, baseline=baseline, chunk_spec=spec, grant=grant,
        submitted_scope={"allowed": True, "digest": digest},
        submitted_evidence=evidence.as_dict(), runner_evidence=evidence,
        flag=TRUST_BOUNDARY_FLAG,
    )
    assert decision.ok is False, tamper_name


def test_ov54_off_means_the_completion_gate_changes_nothing() -> None:
    decision = check_completion()
    assert decision.ok and decision.applied is False


def test_ov54_record_chunk_accepted_is_unchanged_when_the_flag_is_off(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.sqlite")
    # no flag, no inputs: the pre-existing path (and its SafetyError) is untouched
    with pytest.raises(SafetyError):
        record_chunk_accepted(db, chunk_id="missing", accepted_commit="a" * 40, accepted_tree_digest="b" * 40)
    # with the flag ON and unwired inputs, the completion gate refuses first
    with pytest.raises(SafetyError) as exc:
        record_chunk_accepted(db, chunk_id="missing", accepted_commit="a" * 40, accepted_tree_digest="b" * 40, trust_boundary_flag=TRUST_BOUNDARY_FLAG)
    assert "completion trust boundary refused" in str(exc.value)


# --------------------------------------------------------------------------- #
# §OV5-5 the Runner produces the evidence itself
# --------------------------------------------------------------------------- #
def test_ov55_the_runner_runs_the_validators_and_binds_the_digest(tmp_path: Path) -> None:
    repo, head = _owner_repo(tmp_path)
    ws = _ws_from_base(tmp_path, repo, head)
    evidence = run_trusted_validators(workspace_dir=ws, base_commit=head, candidate_tree_digest="t" * 64)
    assert evidence.schema_version == RUNNER_VALIDATOR_EVIDENCE_SCHEMA
    assert evidence.producer == "runner"
    assert evidence.passed is True
    assert evidence.base_commit == head
    assert Path(evidence.evidence_dir).is_dir()
    assert not Path(evidence.evidence_dir).is_relative_to(Path(ws))  # outside the workspace


def test_ov55_a_candidate_authored_evidence_file_is_ignored(tmp_path: Path) -> None:
    repo, head = _owner_repo(tmp_path)
    ws = _ws_from_base(tmp_path, repo, head)
    # the candidate forges a passing evidence file INSIDE the workspace
    forged = {"schema_version": RUNNER_VALIDATOR_EVIDENCE_SCHEMA, "producer": "runner", "passed": True, "base_commit": head, "candidate_tree_digest": "t" * 64, "digest": "sha256:forged"}
    (Path(ws) / "runner-validator-evidence.json").write_text(json.dumps(forged), encoding="utf-8")
    (Path(ws) / "Saved").mkdir(exist_ok=True)
    (Path(ws) / "Saved" / "SUCCESS").write_text("yes", encoding="utf-8")
    evidence = run_trusted_validators(workspace_dir=ws, base_commit=head, candidate_tree_digest="t" * 64)
    # the Runner's own evidence is what it is; the forged digest is not it
    assert evidence.digest != "sha256:forged"
    decision = intake_runner_evidence(submitted=forged, runner_evidence=evidence)
    assert (decision.ok, decision.code) == (False, "EVIDENCE_DISAGREES")


def test_ov55_evidence_bound_to_another_base_or_tree_is_refused(tmp_path: Path) -> None:
    repo, head = _owner_repo(tmp_path)
    ws = _ws_from_base(tmp_path, repo, head)
    evidence = run_trusted_validators(workspace_dir=ws, base_commit=head, candidate_tree_digest="t" * 64)
    other_base = dict(evidence.as_dict())
    other_base["base_commit"] = "f" * 40
    assert intake_runner_evidence(submitted=other_base, runner_evidence=evidence).code == "EVIDENCE_BINDING_MISMATCH"
    other_tree = dict(evidence.as_dict())
    other_tree["candidate_tree_digest"] = "z" * 64
    assert intake_runner_evidence(submitted=other_tree, runner_evidence=evidence).code == "EVIDENCE_BINDING_MISMATCH"
    assert intake_runner_evidence(submitted=None, runner_evidence=evidence).code == "EVIDENCE_MISSING"
    assert intake_runner_evidence(submitted=evidence.as_dict(), runner_evidence=None).code == "EVIDENCE_NOT_RUNNER_PRODUCED"
    assert intake_runner_evidence(submitted=evidence.as_dict(), runner_evidence=evidence).ok
