"""OV5-5 — Runner-run validator evidence.

The Runner does NOT take `recomputed_evidence` from a caller. It runs the trusted
validator command itself, writes the evidence into a Runner-created directory
OUTSIDE the workspace, and binds the evidence digest to the base commit and the
candidate tree — so a candidate-authored evidence file can neither be read nor
pass as authority.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .evidence_util import digest_of as _digest_of
from .evidence_util import usable_digest as _usable_digest

__all__ = [
    "RUNNER_VALIDATOR_EVIDENCE_SCHEMA",
    "RunnerValidatorEvidence",
    "discard_runner_evidence",
    "evidence_root",
    "host_tests_scratch_root",
    "scratch_root",
    "intake_runner_evidence",
    "prune_runner_evidence",
    "run_trusted_validators",
]

#: §EXEC3 fix 2 — the Runner-owned root for validator evidence. It used to be a bare
#: `tempfile.mkdtemp()` in the system temp dir, and NOTHING removed it: a campaign left
#: one `trio-runner-evidence-*` directory per validator run (939 of them after one
#: campaign), which exhausted the temp filesystem's INODES.
EVIDENCE_ROOT_ENV = "TRIO_RUNNER_EVIDENCE_ROOT"


def scratch_root() -> Path:
    """The Runner-owned scratch root: evidence and host-tests copies live here.

    It is deliberately NOT the state dir. Nesting it in the state dir wrote into the
    operator's real state directory whenever a caller did not set `OVERNIGHT_STATE_DIR`
    (the qualification runs the Runner against the real one), which the campaign's
    "never touch the real state directory" rule forbids. One stable root, with the
    per-run directories discarded after intake and pruned if a run dies.
    """
    root = Path(tempfile.gettempdir()) / "trio-runner-scratch"
    root.mkdir(parents=True, exist_ok=True)
    return root


def evidence_root() -> Path:
    """The Runner-owned directory validator evidence is created under."""
    override = os.environ.get(EVIDENCE_ROOT_ENV)
    root = Path(override) if override else scratch_root() / "validator-evidence"
    root.mkdir(parents=True, exist_ok=True)
    return root


def host_tests_scratch_root() -> Path:
    """§EXEC3 fix 2 — the Runner-owned root for host-tests scratch copies."""
    override = os.environ.get(EVIDENCE_ROOT_ENV)
    root = Path(override).parent / "host-tests-scratch" if override else scratch_root() / "host-tests-scratch"
    root.mkdir(parents=True, exist_ok=True)
    return root


def prune_runner_evidence(max_age_s: float = 86_400.0, root: Path | None = None) -> int:
    """Retention: drop evidence directories a crashed run left behind. Returns the count."""
    target = evidence_root() if root is None else root
    cutoff = time.time() - max_age_s
    removed = 0
    if not target.is_dir():
        return 0
    for child in target.iterdir():
        if not child.is_dir() or child.name in {".keep"}:
            continue
        try:
            if child.stat().st_mtime > cutoff:
                continue
        except OSError:
            continue
        shutil.rmtree(child, ignore_errors=True)
        removed += 1
    return removed


def discard_runner_evidence(evidence: RunnerValidatorEvidence | None) -> bool:
    """Delete a Runner-owned evidence directory once its intake decision is made.

    Containment is enforced: only a direct child of `evidence_root()` is ever removed,
    so a record carrying any other path cannot turn this into an arbitrary delete.
    """
    if evidence is None:
        return False
    directory = Path(evidence.evidence_dir)
    root = evidence_root()
    try:
        if directory.parent.resolve() != root.resolve():
            return False
    except OSError:
        return False
    shutil.rmtree(directory, ignore_errors=True)
    return not directory.exists()

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


_PRUNED_THIS_PROCESS = False


def _prune_once() -> None:
    """Retention, once per process: drop evidence a crashed run left behind."""
    global _PRUNED_THIS_PROCESS
    if _PRUNED_THIS_PROCESS:
        return
    _PRUNED_THIS_PROCESS = True
    try:
        prune_runner_evidence()
    except OSError:
        pass


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
    # §EXEC3 fix 2: inside the Runner-owned root (the state dir by default), NOT the
    # bare system temp dir, so the directory is retained and prunable instead of leaked.
    _prune_once()
    evidence_dir = tempfile.mkdtemp(prefix="evidence-", dir=str(evidence_root()))
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
