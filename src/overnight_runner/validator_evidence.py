"""OV5-5 — Runner-run validator evidence.

The Runner does NOT take `recomputed_evidence` from a caller. It runs the trusted
validator command itself, writes the evidence into a Runner-created directory
OUTSIDE the workspace, and binds the evidence digest to the base commit and the
candidate tree — so a candidate-authored evidence file can neither be read nor
pass as authority.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .evidence_util import digest_of as _digest_of
from .evidence_util import usable_digest as _usable_digest

__all__ = [
    "RUNNER_VALIDATOR_EVIDENCE_SCHEMA",
    "RunnerValidatorEvidence",
    "intake_runner_evidence",
    "run_trusted_validators",
]

RUNNER_VALIDATOR_EVIDENCE_SCHEMA = "trio.runner-validator-evidence.v1"

#: The FIXED validator commands. No shell, no caller-supplied argv.
PY_SYNTAX_CHECK = (
    "import ast,pathlib,sys\n"
    "bad=[]\n"
    "for f in sorted(pathlib.Path('.').rglob('*.py')):\n"
    "    try:\n"
    "        ast.parse(f.read_text(encoding='utf-8'), str(f))\n"
    "    except SyntaxError as e:\n"
    "        bad.append(str(f)); print(e)\n"
    "sys.exit(1 if bad else 0)\n"
)
VALIDATOR_COMMANDS: Mapping[str, list[str]] = {
    "py_compile": ["python3", "-B", "-c", PY_SYNTAX_CHECK],
}

#: §OV6-3: `host-tests` is NOT a plain subprocess — the Runner runs it inside the
#: bwrap confinement (read-only workspace copy + pinned toolchain root).
HOST_TESTS_VALIDATOR_ID = "host-tests"


@dataclass(frozen=True)
class IntakeDecision:
    """Structurally identical to intake.IntakeDecision; declared here to keep the
    intake module free of a back-import."""

    ok: bool
    code: str = ""
    detail: str = ""
    recomputed: Any = None
    digest: str = ""


@dataclass(frozen=True)
class RunnerValidatorEvidence:
    schema_version: str
    producer: str
    evidence_dir: str
    base_commit: str
    candidate_tree_digest: str
    validator_ids: tuple[str, ...]
    results: tuple[Mapping[str, Any], ...]
    passed: bool
    digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "producer": self.producer,
            "evidence_dir": self.evidence_dir, "base_commit": self.base_commit,
            "candidate_tree_digest": self.candidate_tree_digest,
            "validator_ids": list(self.validator_ids), "results": [dict(r) for r in self.results],
            "passed": self.passed, "digest": self.digest,
        }


def run_trusted_validators(
    *,
    workspace_dir: str,
    base_commit: str,
    candidate_tree_digest: str,
    validator_ids: Iterable[str] = ("py_compile",),
    timeout_s: int = 120,
) -> RunnerValidatorEvidence:
    """Run the fixed validators on the workspace and mint Runner-owned evidence."""
    # The Runner CREATES the evidence path; nothing the candidate wrote is read.
    evidence_dir = tempfile.mkdtemp(prefix="trio-runner-evidence-")
    results: list[dict[str, Any]] = []
    for validator_id in validator_ids:
        if validator_id == HOST_TESTS_VALIDATOR_ID:
            from .host_tests import run_host_tests

            host = run_host_tests(workspace_dir=workspace_dir, timeout_s=timeout_s)
            results.append({
                "validator_id": validator_id,
                "outcome": host.outcome,
                "exit_code": host.exit_code,
                "stdout_digest": _digest_of(host.stdout),
                "stderr_digest": _digest_of(host.stderr),
            })
            continue
        argv = VALIDATOR_COMMANDS.get(validator_id)
        if argv is None:
            results.append({"validator_id": validator_id, "outcome": "unknown", "exit_code": None})
            continue
        proc = subprocess.run(argv, cwd=workspace_dir, capture_output=True, timeout=timeout_s, check=False)
        results.append({
            "validator_id": validator_id,
            "outcome": "passed" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
            "stdout_digest": _digest_of(proc.stdout.decode(errors="replace")),
            "stderr_digest": _digest_of(proc.stderr.decode(errors="replace")),
        })
    passed = bool(results) and all(r["outcome"] == "passed" for r in results)
    body = {
        "schema_version": RUNNER_VALIDATOR_EVIDENCE_SCHEMA,
        "producer": "runner",
        "evidence_dir": evidence_dir,
        "base_commit": base_commit,
        "candidate_tree_digest": candidate_tree_digest,
        "validator_ids": list(validator_ids),
        "results": results,
        "passed": passed,
    }
    body["digest"] = _digest_of(body)
    (Path(evidence_dir) / "runner-validator-evidence.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    return RunnerValidatorEvidence(
        schema_version=RUNNER_VALIDATOR_EVIDENCE_SCHEMA, producer="runner", evidence_dir=evidence_dir,
        base_commit=base_commit, candidate_tree_digest=candidate_tree_digest,
        validator_ids=tuple(validator_ids), results=tuple(results), passed=passed, digest=body["digest"],
    )


def intake_runner_evidence(
    *,
    submitted: Any,
    runner_evidence: RunnerValidatorEvidence | None,
    current_tree_digest: str | None = None,
) -> IntakeDecision:
    """Compare a submitted evidence document with the Runner's OWN run."""
    if not isinstance(runner_evidence, RunnerValidatorEvidence):
        return IntakeDecision(False, "EVIDENCE_NOT_RUNNER_PRODUCED", "the Runner has no evidence it produced itself")
    if not isinstance(submitted, Mapping):
        return IntakeDecision(False, "EVIDENCE_MISSING", "no submitted validator evidence")
    if not _usable_digest(submitted.get("digest")) or not _usable_digest(runner_evidence.digest):
        return IntakeDecision(False, "EVIDENCE_DIGEST_MISSING", "a digest is missing on one side")
    # Binding: the submitted document must name the SAME base commit and tree.
    if submitted.get("base_commit") != runner_evidence.base_commit:
        return IntakeDecision(False, "EVIDENCE_BINDING_MISMATCH", "the submitted evidence is bound to a different base commit", digest=runner_evidence.digest)
    if submitted.get("candidate_tree_digest") != runner_evidence.candidate_tree_digest:
        return IntakeDecision(False, "EVIDENCE_BINDING_MISMATCH", "the submitted evidence is bound to a different candidate tree", digest=runner_evidence.digest)
    if submitted.get("digest") != runner_evidence.digest:
        return IntakeDecision(False, "EVIDENCE_DISAGREES", "the submitted evidence digest is not the Runner's", digest=runner_evidence.digest)
    # §OV6-2: the evidence must have been run on the tree the Runner holds NOW
    # (only checked when the caller supplies the tree it actually holds).
    if current_tree_digest is not None and runner_evidence.candidate_tree_digest != current_tree_digest:
        return IntakeDecision(False, "EVIDENCE_BINDING_MISMATCH", "the Runner's evidence was produced on a different candidate tree", digest=runner_evidence.digest)
    if submitted.get("passed") is not True:
        return IntakeDecision(False, "EVIDENCE_SUBMITTED_REFUSAL", "the submitted evidence refuses; a refusal is never ok=True", digest=runner_evidence.digest)
    if runner_evidence.passed is not True:
        return IntakeDecision(False, "EVIDENCE_RUNNER_FAILED", "the Runner's own validator run did not pass", digest=runner_evidence.digest)
    return IntakeDecision(True, digest=runner_evidence.digest)
