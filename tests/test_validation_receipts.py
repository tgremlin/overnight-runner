"""P05-A06 TRUSTED VALIDATOR RECEIPT TESTS.

The user-visible V1 suite is preserved (110 tests). The tests in this
module are additive and exercise the runner-owned ``validation``
receipts that are minted at the validator boundary (NOT the
``mutation/apply`` receipts that are minted at apply time).

Each test must be PROVING a specific P05-A06 contract requirement:

  - actual required validator PASS -> trusted receipt accepted
  - actual required validator FAIL -> receipt records FAIL, not PASS
  - forged opaque receipt id rejected
  - wrong validator/profile rejected
  - wrong candidate snapshot/tree rejected
  - wrong chunk/request rejected
  - mutation receipt cannot satisfy a required validation receipt
  - validator receipt for candidate A cannot satisfy candidate B
  - worker has no validation-receipt mint capability

The worker is the Pydantic-forge ``WorkerOrchestrator`` consumed by
external code; the runner owns the mint surface via
``Worker._build_broker`` (mutation) and ``Worker._finalise`` /
``_maybe_mint_validation_receipt`` (validation).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import pytest

from overnight_runner.broker import (
    Broker,
    CommandSpec,
    default_registry,
)
from overnight_runner.worker import Worker
from overnight_runner.schemas import TaskManifest
from overnight_runner.safety import (
    git_commit_all,
    git_init_empty,
    sha256_file,
)
from overnight_runner.receipts import (
    KIND_MUTATION_APPLY,
    KIND_VALIDATION,
    receipts_enabled,
    mint_mutation_receipt,
    mint_validation_receipt,
    verify_receipt,
)


# ----------------------------- Fixtures -----------------------------

@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Isolated OVERNIGHT_STATE_DIR so receipt tests are hermetic."""
    sd = tmp_path / "state"
    sd.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OVERNIGHT_STATE_DIR", str(sd))
    monkeypatch.setenv("OVERNIGHT_RECEIPTS", "1")
    yield tmp_path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git_init_empty(repo)
    (repo / "hello.py").write_text("def hello():\n    return 'old'\n")
    (repo / "test_hello.py").write_text("from hello import hello\nassert hello() == 'old'\n")
    git_commit_all(repo, "init")
    return repo


def _make_run_validator_registry(passing_cmd: str = "echo_passing") -> "default_registry-like":
    from overnight_runner.broker import CommandRegistry
    reg = CommandRegistry()
    # ``true`` / ``false`` are non-mutating, repo-independent, and never
    # touch the filesystem, which is exactly what we want for receipts
    # tests that compare worktree fingerprints around validator runs.
    reg.register(CommandSpec(passing_cmd, ["true"], "repo", 10, "read"))
    reg.register(CommandSpec("echo_failing", ["false"], "repo", 10, "read"))
    reg.register(CommandSpec("noop", ["true"], "repo", 5, "none"))
    reg.register(CommandSpec("pytest_runner_tests", ["python3", "-m", "pytest", "-q", "-x"], "repo", 300, "read"))
    reg.register(CommandSpec("git_diff_check", ["git", "--no-pager", "diff", "--no-color"], "repo", 30, "read"))
    reg.register(CommandSpec("git_status", ["git", "status", "--porcelain"], "repo", 10, "read"))
    return reg


# ----------------------------- 1. PASS -> receipt accepted -----------------------------

class TestActualValidatorPass(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._old_state = os.environ.get("OVERNIGHT_STATE_DIR")
        self._old_flag = os.environ.get("OVERNIGHT_RECEIPTS")
        os.environ["OVERNIGHT_STATE_DIR"] = self._tmp
        os.environ["OVERNIGHT_RECEIPTS"] = "1"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("OVERNIGHT_STATE_DIR", None)
        os.environ.pop("OVERNIGHT_RECEIPTS", None)
        if self._old_state is not None:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state
        if self._old_flag is not None:
            os.environ["OVERNIGHT_RECEIPTS"] = self._old_flag

    def test_actual_validator_pass_mints_trusted_receipt(self):
        """Required validator runs against the actual candidate, PASSES, and
        runner mints a validation receipt that verifies against the
        EXPECTED identity (validator/command/candidate snapshot)."""
        reg = _make_run_validator_registry()
        repo = Path(self._tmp) / "repo"
        repo.mkdir()
        git_init_empty(repo)
        (repo / "hello.py").write_text("def hello():\n    return 'ok'\n")
        (repo / "test_hello.py").write_text("from hello import hello\nassert hello() == 'ok'\n")
        git_commit_all(repo, "init")

        # Capture candidate snapshot (must match what validator saw).
        from overnight_runner.safety import git_worktree_sha
        snapshot = git_worktree_sha(repo)
        reg = _make_run_validator_registry()
        b = Broker(
            repo_root=repo,
            registry=reg,
            allowed_write_paths=["hello.py"],
            allowed_read_paths=["hello.py", "test_hello.py"],
        )

        # Bypass the chat; force finalise path through a synthetic call.
        w = Worker(client=_NoChat(), registry=reg)
        w.receipts_mint_mutation = lambda payload: mint_mutation_receipt(**payload)
        sha = sha256_file(repo / "hello.py")
        from overnight_runner.schemas import (
            ReplaceExactArgs, ToolCall,
        )
        out = b.handle(ToolCall(call_id="c1", args=ReplaceExactArgs(
            path="hello.py", expected_sha256=sha,
            old_text="return 'ok'", new_text="return 'OK'",
        )))
        b.apply_proposal(out["proposal_id"])
        # Snapshot AFTER the apply to match what the validator saw.
        post_apply_snapshot = git_worktree_sha(repo)

        # Use a /tmp artifact dir to avoid drift in the repo worktree.
        from overnight_runner.worker import _finalise
        from overnight_runner.schemas import Disposition
        manifest = _make_manifest(repo, "passes", required_validator_ids=["echo_passing"])
        rid_holder: list[str] = []
        def on_val(payload):
            rid = mint_validation_receipt(
                validator_id=payload["validator_id"],
                validator_command=payload["validator_command"],
                candidate_snapshot_digest=payload["candidate_snapshot_digest"],
                candidate_tree_state=payload.get("candidate_tree_state", "post-apply"),
                exit_code=payload.get("exit_code", 0),
                outcome=payload.get("outcome", "pass"),
                detail=payload.get("detail", ""),
                env_digest="env-" + "f"*64,
                profile_digest="prof-" + "f"*64,
            )
            rid_holder.append(rid)
            return rid
        art = Path(self._tmp) / "artifacts"
        status, reason, text, receipts = _finalise(
            manifest, b, Disposition.DONE, art,
            applied_proposals=["c1"],
            on_validation_receipt=on_val,
            env_digest="env-" + "f"*64, profile_digest="prof-" + "f"*64,
        )
        self.assertEqual(status, "PASSED", f"expected PASSED, got {status}: {text}")
        self.assertEqual(len(receipts), 1, "exactly one validation receipt expected")
        rid = receipts[0]
        # Trusted verification against expected identity using the
        # candidate snapshot the validator actually examined
        # (post-apply).
        self.assertTrue(
            verify_receipt(
                rid, expected_kind=KIND_VALIDATION,
                candidate_snapshot_digest=post_apply_snapshot,
                validator_command="echo_passing",
                validator_id="echo_passing",
            ),
            "PASS receipt must verify against expected identity",
        )


# ----------------------------- 2. FAIL -> receipt records FAIL -----------------------------

class TestActualValidatorFail(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._old_state = os.environ.get("OVERNIGHT_STATE_DIR")
        self._old_flag = os.environ.get("OVERNIGHT_RECEIPTS")
        os.environ["OVERNIGHT_STATE_DIR"] = self._tmp
        os.environ["OVERNIGHT_RECEIPTS"] = "1"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("OVERNIGHT_STATE_DIR", None)
        os.environ.pop("OVERNIGHT_RECEIPTS", None)
        if self._old_state is not None:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state
        if self._old_flag is not None:
            os.environ["OVERNIGHT_RECEIPTS"] = self._old_flag

    def test_actual_validator_fail_records_fail_outcome(self):
        """When a required validator FAILS, the receipt records outcome='fail'
        and verify_receipt(rid, outcome='fail') is True (it really did fail).
        The task itself is marked FAILED, not PASSED — proving that
        a successful apply does NOT imply validator success."""
        reg = _make_run_validator_registry()
        repo = Path(self._tmp) / "repo"
        repo.mkdir()
        git_init_empty(repo)
        (repo / "hello.py").write_text("def hello():\n    return 'ok'\n")
        (repo / "test_hello.py").write_text("from hello import hello\nassert hello() == 'ok'\n")
        git_commit_all(repo, "init")

        from overnight_runner.worker import _finalise
        from overnight_runner.schemas import Disposition, ReplaceExactArgs, ToolCall

        sha = sha256_file(repo / "hello.py")
        b = Broker(repo_root=repo, registry=reg,
                   allowed_write_paths=["hello.py"],
                   allowed_read_paths=["hello.py", "test_hello.py"])
        out = b.handle(ToolCall(call_id="c1", args=ReplaceExactArgs(
            path="hello.py", expected_sha256=sha,
            old_text="return 'ok'", new_text="return 'OK'",
        )))
        b.apply_proposal(out["proposal_id"])

        manifest = _make_manifest(repo, "fails", required_validator_ids=["echo_failing"])
        rid_holder: list[str] = []
        def on_val(payload):
            rid = mint_validation_receipt(
                validator_id=payload["validator_id"],
                validator_command=payload["validator_command"],
                candidate_snapshot_digest=payload["candidate_snapshot_digest"],
                candidate_tree_state=payload.get("candidate_tree_state", "post-apply"),
                exit_code=payload.get("exit_code", 2),
                outcome=payload.get("outcome", "fail"),
                detail=payload.get("detail", ""),
                env_digest="env-" + "f"*64,
                profile_digest="prof-" + "f"*64,
            )
            rid_holder.append(rid)
            return rid
        art = Path(self._tmp) / "artifacts"
        status, reason, text, receipts = _finalise(
            manifest, b, Disposition.DONE, art,
            applied_proposals=["c1"],
            on_validation_receipt=on_val,
            env_digest="env-" + "f"*64, profile_digest="prof-" + "f"*64,
        )
        # Task FAILED because validator failed.
        self.assertEqual(status, "FAILED")
        # The FAIL details live on reason_text (the per-validator message
        # joined by ``; ``); reason_code is the classification.
        self.assertEqual(reason, "VALIDATORS_FAILED")
        self.assertIn("echo_failing", text or "")
        # Receipt MUST record FAIL outcome, not PASS.
        self.assertEqual(len(receipts), 1)
        rid = receipts[0]
        self.assertTrue(
            verify_receipt(rid, expected_kind=KIND_VALIDATION, outcome="fail"),
            "FAIL receipt must verify with outcome='fail'",
        )
        # And it MUST NOT verify with outcome='pass' (that would mean the
        # runner lied).
        self.assertFalse(
            verify_receipt(rid, expected_kind=KIND_VALIDATION, outcome="pass"),
            "FAIL receipt must NOT verify with outcome='pass'",
        )


# ----------------------------- 3. Forged opaque receipt id rejected -----------------------------

class TestForgedOpaqueReceiptRejected(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._old_state = os.environ.get("OVERNIGHT_STATE_DIR")
        self._old_flag = os.environ.get("OVERNIGHT_RECEIPTS")
        os.environ["OVERNIGHT_STATE_DIR"] = self._tmp
        os.environ["OVERNIGHT_RECEIPTS"] = "1"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("OVERNIGHT_STATE_DIR", None)
        os.environ.pop("OVERNIGHT_RECEIPTS", None)
        if self._old_state is not None:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state
        if self._old_flag is not None:
            os.environ["OVERNIGHT_RECEIPTS"] = self._old_flag

    def test_forged_receipt_id_rejected(self):
        """A forged opaque id (never issued by the runner) MUST NOT verify."""
        # Quietly confirm no record exists for the fabricated id.
        self.assertFalse(
            verify_receipt(
                "rec-val-forged-by-attacker",
                expected_kind=KIND_VALIDATION,
                candidate_snapshot_digest="a" * 64,
                validator_id="anything",
            ),
            "forged receipt id must not verify",
        )

    def test_path_traversal_id_rejected(self):
        """An id with '..' or '/' MUST be refused (no crash; no leak)."""
        self.assertFalse(
            verify_receipt("../escape", expected_kind=KIND_VALIDATION,
                           candidate_snapshot_digest="a" * 64)
        )
        self.assertFalse(
            verify_receipt("rec-val/../../etc", expected_kind=KIND_VALIDATION,
                           candidate_snapshot_digest="a" * 64)
        )
        # Empty id and oversize id also refused.
        self.assertFalse(verify_receipt("", expected_kind=KIND_VALIDATION))
        self.assertFalse(verify_receipt("a" * 1024, expected_kind=KIND_VALIDATION))


# ----------------------------- 4. Wrong validator/profile rejected -----------------------------

class TestWrongValidatorRejected(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._old_state = os.environ.get("OVERNIGHT_STATE_DIR")
        self._old_flag = os.environ.get("OVERNIGHT_RECEIPTS")
        os.environ["OVERNIGHT_STATE_DIR"] = self._tmp
        os.environ["OVERNIGHT_RECEIPTS"] = "1"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("OVERNIGHT_STATE_DIR", None)
        os.environ.pop("OVERNIGHT_RECEIPTS", None)
        if self._old_state is not None:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state
        if self._old_flag is not None:
            os.environ["OVERNIGHT_RECEIPTS"] = self._old_flag

    def test_wrong_validator_id_rejected(self):
        rid = mint_validation_receipt(
            validator_id="v-a", validator_command="echo_passing",
            candidate_snapshot_digest="a" * 64, outcome="pass", detail="ok",
            env_digest="env-" + "f" * 64,
        )
        # Same id, but caller expects a different validator -> False.
        self.assertFalse(
            verify_receipt(rid, expected_kind=KIND_VALIDATION,
                           validator_id="v-other"),
            "wrong validator_id must not verify",
        )
        # Correct validator verifies.
        self.assertTrue(
            verify_receipt(rid, expected_kind=KIND_VALIDATION,
                           validator_id="v-a", validator_command="echo_passing"),
        )

    def test_wrong_validator_command_rejected(self):
        rid = mint_validation_receipt(
            validator_id="v-a", validator_command="echo_passing",
            candidate_snapshot_digest="a" * 64, outcome="pass", detail="ok",
            env_digest="env-" + "f" * 64,
        )
        # Caller verifies against a different command (different "profile").
        self.assertFalse(
            verify_receipt(rid, expected_kind=KIND_VALIDATION,
                           validator_id="v-a",
                           validator_command="echo_failing"),
            "wrong validator_command must not verify",
        )


# ----------------------------- 5. Wrong candidate snapshot/tree rejected -----------------------------

class TestWrongSnapshotRejected(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._old_state = os.environ.get("OVERNIGHT_STATE_DIR")
        self._old_flag = os.environ.get("OVERNIGHT_RECEIPTS")
        os.environ["OVERNIGHT_STATE_DIR"] = self._tmp
        os.environ["OVERNIGHT_RECEIPTS"] = "1"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("OVERNIGHT_STATE_DIR", None)
        os.environ.pop("OVERNIGHT_RECEIPTS", None)
        if self._old_state is not None:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state
        if self._old_flag is not None:
            os.environ["OVERNIGHT_RECEIPTS"] = self._old_flag

    def test_wrong_candidate_snapshot_rejected(self):
        snapshot_a = "a" * 64
        snapshot_b = "b" * 64
        rid = mint_validation_receipt(
            validator_id="v-a", validator_command="echo_passing",
            candidate_snapshot_digest=snapshot_a, outcome="pass",
            detail="ok", env_digest="env-" + "f" * 64,
        )
        # Correct snapshot verifies.
        self.assertTrue(
            verify_receipt(rid, expected_kind=KIND_VALIDATION,
                           validator_id="v-a",
                           candidate_snapshot_digest=snapshot_a),
        )
        # Wrong snapshot does NOT.
        self.assertFalse(
            verify_receipt(rid, expected_kind=KIND_VALIDATION,
                           validator_id="v-a",
                           candidate_snapshot_digest=snapshot_b),
            "wrong candidate_snapshot_digest must not verify",
        )


# ----------------------------- 6. Wrong chunk/request rejected -----------------------------

class TestWrongIdentityRejected(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._old_state = os.environ.get("OVERNIGHT_STATE_DIR")
        self._old_flag = os.environ.get("OVERNIGHT_RECEIPTS")
        os.environ["OVERNIGHT_STATE_DIR"] = self._tmp
        os.environ["OVERNIGHT_RECEIPTS"] = "1"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("OVERNIGHT_STATE_DIR", None)
        os.environ.pop("OVERNIGHT_RECEIPTS", None)
        if self._old_state is not None:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state
        if self._old_flag is not None:
            os.environ["OVERNIGHT_RECEIPTS"] = self._old_flag

    def test_wrong_chunk_or_request_rejected(self):
        rid = mint_validation_receipt(
            validator_id="v-a", validator_command="echo_passing",
            candidate_snapshot_digest="a" * 64,
            chunk_id="chunk-1", request_id="req-1",
            outcome="pass", detail="ok",
            env_digest="env-" + "f" * 64,
        )
        # Wrong chunk rejected.
        self.assertFalse(
            verify_receipt(rid, expected_kind=KIND_VALIDATION,
                           candidate_snapshot_digest="a" * 64,
                           chunk_id="chunk-2", request_id="req-1"),
            "wrong chunk_id must not verify",
        )
        # Wrong request rejected.
        self.assertFalse(
            verify_receipt(rid, expected_kind=KIND_VALIDATION,
                           candidate_snapshot_digest="a" * 64,
                           chunk_id="chunk-1", request_id="req-2"),
            "wrong request_id must not verify",
        )
        # Correct identity verifies.
        self.assertTrue(
            verify_receipt(rid, expected_kind=KIND_VALIDATION,
                           candidate_snapshot_digest="a" * 64,
                           chunk_id="chunk-1", request_id="req-1"),
        )


# ----------------------------- 7. Mutation receipt cannot satisfy a required validation receipt -----------------------------

class TestMutationCannotSatisfyValidation(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._old_state = os.environ.get("OVERNIGHT_STATE_DIR")
        self._old_flag = os.environ.get("OVERNIGHT_RECEIPTS")
        os.environ["OVERNIGHT_STATE_DIR"] = self._tmp
        os.environ["OVERNIGHT_RECEIPTS"] = "1"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("OVERNIGHT_STATE_DIR", None)
        os.environ.pop("OVERNIGHT_RECEIPTS", None)
        if self._old_state is not None:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state
        if self._old_flag is not None:
            os.environ["OVERNIGHT_RECEIPTS"] = self._old_flag

    def test_mutation_receipt_does_not_satisfy_validation(self):
        """A mutation/apply receipt is structurally incapable of satisfying a
        required validation receipt — the kinds are not interchangeable."""
        mut_rid = mint_mutation_receipt(
            proposal_id="prop-1",
            path="hello.py", op="replace_file",
            pre_sha256="a" * 64, post_sha256="b" * 64,
            bytes_written=10,
            candidate_snapshot_digest="c" * 64,
        )
        # Even with the same proposal_id, asking for validation kind FAILs.
        self.assertFalse(
            verify_receipt(mut_rid, expected_kind=KIND_VALIDATION,
                           candidate_snapshot_digest="c" * 64,
                           validator_id="v-a", validator_command="cmd",
                           chunk_id="chunk-1", request_id="req-1"),
            "mutation/apply receipt must NOT satisfy a validation ask",
        )
        # And the reverse: a validation receipt must NOT satisfy a
        # mutation/apply ask (no pre_sha binding).
        val_rid = mint_validation_receipt(
            validator_id="v-a", validator_command="cmd",
            candidate_snapshot_digest="c" * 64, outcome="pass",
            detail="ok", env_digest="env-" + "f" * 64,
        )
        self.assertFalse(
            verify_receipt(val_rid, expected_kind=KIND_MUTATION_APPLY,
                           proposal_id="prop-1"),
            "validation receipt must NOT satisfy a mutation/apply ask",
        )


# ----------------------------- 8. Validator receipt for candidate A cannot satisfy candidate B -----------------------------

class TestValidatorReceiptCandidateABinding(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._old_state = os.environ.get("OVERNIGHT_STATE_DIR")
        self._old_flag = os.environ.get("OVERNIGHT_RECEIPTS")
        os.environ["OVERNIGHT_STATE_DIR"] = self._tmp
        os.environ["OVERNIGHT_RECEIPTS"] = "1"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("OVERNIGHT_STATE_DIR", None)
        os.environ.pop("OVERNIGHT_RECEIPTS", None)
        if self._old_state is not None:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state
        if self._old_flag is not None:
            os.environ["OVERNIGHT_RECEIPTS"] = self._old_flag

    def test_receipt_for_candidate_a_rejected_for_candidate_b(self):
        rid_a = mint_validation_receipt(
            validator_id="v-a", validator_command="cmd",
            candidate_snapshot_digest="a" * 64, outcome="pass",
            detail="for A", env_digest="env-" + "f" * 64,
        )
        # Same validator, but a different candidate snapshot (B). MUST NOT
        # pass — the receipt is bound to A.
        self.assertFalse(
            verify_receipt(rid_a, expected_kind=KIND_VALIDATION,
                           validator_id="v-a", validator_command="cmd",
                           candidate_snapshot_digest="b" * 64),
            "receipt for candidate A must not satisfy candidate B",
        )
        # And it MUST verify for A.
        self.assertTrue(
            verify_receipt(rid_a, expected_kind=KIND_VALIDATION,
                           validator_id="v-a", validator_command="cmd",
                           candidate_snapshot_digest="a" * 64),
        )


# ----------------------------- 9. Worker has no validation-receipt mint capability -----------------------------

class TestWorkerHasNoMintCapability(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._old_state = os.environ.get("OVERNIGHT_STATE_DIR")
        self._old_flag = os.environ.get("OVERNIGHT_RECEIPTS")
        os.environ["OVERNIGHT_STATE_DIR"] = self._tmp
        os.environ["OVERNIGHT_RECEIPTS"] = "1"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("OVERNIGHT_STATE_DIR", None)
        os.environ.pop("OVERNIGHT_RECEIPTS", None)
        if self._old_state is not None:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state
        if self._old_flag is not None:
            os.environ["OVERNIGHT_RECEIPTS"] = self._old_flag

    def test_worker_exposes_no_mint_surface(self):
        """The forge worker (trio-workers) only depends on BrokerPort. The
        runner-owned receipts module has mint functions, but those are
        not exposed on any class the forge worker depends on."""
        from overnight_runner.broker import Broker
        b = Broker(repo_root=Path(self._tmp))
        # The Broker surface exposes no ``mint`` method; only the
        # runner-owned module exports mint functions.
        self.assertFalse(hasattr(b, "mint_receipt"))
        self.assertFalse(hasattr(b, "mint_validation_receipt"))
        self.assertFalse(hasattr(b, "mint_mutation_receipt"))
        # The Worker class exposes no mint method either (forge workers
        # only supply callbacks via Worker.run kwargs).
        w = Worker(client=_NoChat())
        self.assertFalse(hasattr(w, "mint_receipt"))
        self.assertFalse(hasattr(w, "mint_validation_receipt"))
        self.assertFalse(hasattr(w, "mint_mutation_receipt"))

    def test_receipts_disabled_when_flag_off(self):
        """When OVERNIGHT_RECEIPTS is unset, mint returns '' and verify
        returns False (the worker cannot satisfy admission by forcing a
        mint against an empty store)."""
        os.environ.pop("OVERNIGHT_RECEIPTS", None)
        rid = mint_validation_receipt(
            validator_id="v-a", validator_command="cmd",
            candidate_snapshot_digest="a" * 64, outcome="pass",
            detail="ok", env_digest="env-" + "f" * 64,
        )
        self.assertEqual(rid, "", "mint must be no-op when receipts disabled")
        # Empty store => verify False for any id.
        self.assertFalse(verify_receipt("rec-val-anything", expected_kind=KIND_VALIDATION))


# ----------------------------- 10. Existing mutation-receipt tests stay green -----------------------------

class TestMutationReceiptAdditive(unittest.TestCase):
    """Preserve the existing mutation/apply receipt tests so that the new
    validation receipt mode is purely additive."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._old_state = os.environ.get("OVERNIGHT_STATE_DIR")
        self._old_flag = os.environ.get("OVERNIGHT_RECEIPTS")
        os.environ["OVERNIGHT_STATE_DIR"] = self._tmp
        os.environ["OVERNIGHT_RECEIPTS"] = "1"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("OVERNIGHT_STATE_DIR", None)
        os.environ.pop("OVERNIGHT_RECEIPTS", None)
        if self._old_state is not None:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state
        if self._old_flag is not None:
            os.environ["OVERNIGHT_RECEIPTS"] = self._old_flag

    def test_mutation_receipt_mint_and_verify(self):
        rid = mint_mutation_receipt(
            proposal_id="prop-1",
            path="hello.py", op="replace_file",
            pre_sha256="a" * 64, post_sha256="b" * 64,
            bytes_written=10, candidate_snapshot_digest="c" * 64,
        )
        self.assertTrue(rid.startswith("rec-mut-"))
        # The mutation/apply receipt verifies ONLY with the mutation kind.
        self.assertTrue(
            verify_receipt(rid, expected_kind=KIND_MUTATION_APPLY,
                           proposal_id="prop-1"),
        )
        self.assertFalse(
            verify_receipt(rid, expected_kind=KIND_VALIDATION),
            "mutation receipt must not satisfy a validation ask",
        )


# ----------------------------- 11. Built-in validator branches in _finalise -----------------------------

class TestBuiltinValidatorsMintReceipts(unittest.TestCase):
    """Regression: the built-in validator branches in ``_finalise``
    (``no_op`` / ``noop`` and ``python_compile``) MUST capture the
    candidate snapshot BEFORE invoking the validator and mint a
    ``kind=validation`` receipt bound to that snapshot.

    These tests exercise ``_finalise`` end-to-end (they MUST NOT
    shortcut the boundary by calling ``mint_validation_receipt``
    directly) so we prove the actual required-validator path creates
    the receipt.

    The shared ``mint_through_callback`` helper wires
    ``on_validation_receipt`` to the runner-owned ``mint_validation_receipt``
    via the live ``overnight_runner.receipts`` module so the durable
    store path is exercised exactly as it would be in production.
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._old_state = os.environ.get("OVERNIGHT_STATE_DIR")
        self._old_flag = os.environ.get("OVERNIGHT_RECEIPTS")
        os.environ["OVERNIGHT_STATE_DIR"] = self._tmp
        os.environ["OVERNIGHT_RECEIPTS"] = "1"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("OVERNIGHT_STATE_DIR", None)
        os.environ.pop("OVERNIGHT_RECEIPTS", None)
        if self._old_state is not None:
            os.environ["OVERNIGHT_STATE_DIR"] = self._old_state
        if self._old_flag is not None:
            os.environ["OVERNIGHT_RECEIPTS"] = self._old_flag

    # ---------- 1. required ``no_op`` validator -> exactly one PASS validation receipt ----------

    def test_required_noop_mints_pass_validation_receipt(self):
        """A required ``no_op`` validator that runs through ``_finalise``
        produces exactly one ``kind=validation`` trusted receipt; the
        receipt verifies against the pre-validator candidate snapshot
        and records ``outcome=pass``."""
        from overnight_runner.safety import git_init_empty, git_commit_all, git_worktree_sha, sha256_file
        from overnight_runner.schemas import ReplaceExactArgs, ToolCall, Disposition
        from overnight_runner.worker import _finalise

        repo = Path(self._tmp) / "repo"
        repo.mkdir()
        git_init_empty(repo)
        (repo / "hello.py").write_text("def hello():\n    return 'ok'\n")
        (repo / "test_hello.py").write_text(
            "from hello import hello\nassert hello() == 'ok'\n"
        )
        git_commit_all(repo, "init")

        # Set up a real broker so the apply boundary can resolve.
        reg = default_registry()
        b = Broker(
            repo_root=repo,
            registry=reg,
            allowed_write_paths=["hello.py"],
            allowed_read_paths=["hello.py", "test_hello.py"],
        )
        sha = sha256_file(repo / "hello.py")
        # Apply a small in-place change so applied_proposals is
        # populated AND the candidate snapshot reflects the post-apply
        # state. The validator examines THIS state, so the receipt
        # must bind to ``git_worktree_sha(repo)`` captured AFTER
        # ``apply_proposal``.
        out = b.handle(ToolCall(call_id="c1", args=ReplaceExactArgs(
            path="hello.py", expected_sha256=sha,
            old_text="return 'ok'", new_text="return 'OK'",
        )))
        b.apply_proposal(out["proposal_id"])
        expected_snapshot = git_worktree_sha(repo)

        # Use a /tmp artifact dir to avoid further drift in the repo
        # worktree (artifact_dir is a sibling, not inside repo).
        art = Path(self._tmp) / "artifacts"

        # Wire the runner-owned receipts module as the mint callback so
        # the receipt lands in the durable evidence store (not in
        # memory). This proves ``_finalise`` itself produced the receipt
        # — we only forward ``payload`` to ``mint_validation_receipt``.
        from overnight_runner import receipts as rm
        def on_val(payload):
            return rm.mint_validation_receipt(
                validator_id=payload["validator_id"],
                validator_command=payload["validator_command"],
                candidate_snapshot_digest=payload["candidate_snapshot_digest"],
                candidate_tree_state=payload.get("candidate_tree_state", "post-apply"),
                proposal_id=payload.get("proposal_id", ""),
                chunk_id=payload.get("chunk_id", ""),
                request_id=payload.get("request_id", ""),
                patch_ref=payload.get("patch_ref", ""),
                patch_sha256=payload.get("patch_sha256", ""),
                exit_code=payload.get("exit_code", 0),
                signal_name=payload.get("signal_name", ""),
                timed_out=payload.get("timed_out", False),
                outcome=payload.get("outcome", "pass"),
                detail=payload.get("detail", ""),
                raw_artifact_ref=payload.get("raw_artifact_ref", ""),
                env_digest=payload.get("env_digest", ""),
                profile_digest=payload.get("profile_digest", ""),
            )

        manifest = _make_manifest(
            repo, "noop-finalise",
            required_validator_ids=["no_op"],
            execution_class="source_mutation",
        )
        status, reason, text, receipts = _finalise(
            manifest, b, Disposition.DONE, art,
            applied_proposals=[out["proposal_id"]],
            on_validation_receipt=on_val,
            env_digest="env-" + "f" * 64,
            profile_digest="prof-" + "f" * 64,
        )

        # 1. ``_finalise`` reports PASS for the no_op validator.
        self.assertEqual(status, "PASSED",
                         f"no_op validator must pass; got {status}: {text}")
        # 2. Exactly one trusted validation receipt was minted.
        self.assertEqual(len(receipts), 1, "exactly one validation receipt")
        rid = receipts[0]
        self.assertTrue(rid.startswith("rec-val-"),
                        f"validation receipt id expected, got {rid}")
        # 3. The receipt verifies against the pre-validator snapshot.
        self.assertTrue(
            verify_receipt(
                rid, expected_kind=KIND_VALIDATION,
                validator_id="no_op",
                validator_command="no_op",
                candidate_snapshot_digest=expected_snapshot,
                outcome="pass",
            ),
            "no_op PASS receipt must verify against expected identity",
        )

    # ---------- 2. required ``python_compile`` PASS -> one validation receipt ----------

    def test_required_python_compile_pass_mints_validation_receipt(self):
        """A required ``python_compile`` validator that PASSES produces
        exactly one trusted validation receipt with non-empty
        candidate snapshot and ``outcome=pass``."""
        from overnight_runner.safety import git_init_empty, git_commit_all, git_worktree_sha
        from overnight_runner.worker import _finalise
        from overnight_runner.schemas import Disposition

        repo = Path(self._tmp) / "repo"
        repo.mkdir()
        git_init_empty(repo)
        (repo / "hello.py").write_text("def hello():\n    return 'ok'\n")
        git_commit_all(repo, "init")

        expected_snapshot = git_worktree_sha(repo)
        reg = default_registry()
        # python_compile path is the special built-in in _finalise, so
        # we do NOT register it; required_validator_ids=["python_compile"]
        # below forces the special branch.
        b = Broker(
            repo_root=repo,
            registry=reg,
            allowed_write_paths=["hello.py"],
            allowed_read_paths=["hello.py"],
        )

        from overnight_runner import receipts as rm
        def on_val(payload):
            return rm.mint_validation_receipt(
                validator_id=payload["validator_id"],
                validator_command=payload["validator_command"],
                candidate_snapshot_digest=payload["candidate_snapshot_digest"],
                candidate_tree_state=payload.get("candidate_tree_state", "post-apply"),
                exit_code=payload.get("exit_code", 0),
                outcome=payload.get("outcome", "pass"),
                detail=payload.get("detail", ""),
                env_digest=payload.get("env_digest", ""),
                profile_digest=payload.get("profile_digest", ""),
            )

        manifest = _make_manifest(
            repo, "pyc-pass",
            required_validator_ids=["python_compile"],
            execution_class="source_mutation",
            write_paths=["hello.py"],
        )
        # ``_make_manifest`` always includes a write_paths entry of
        # ``hello.py`` and creates the manifest with execution_class
        # source_mutation by default.
        art = Path(self._tmp) / "artifacts"
        status, reason, text, receipts = _finalise(
            manifest, b, Disposition.DONE, art,
            applied_proposals=["c-py-pass"],
            on_validation_receipt=on_val,
            env_digest="env-" + "f" * 64,
            profile_digest="prof-" + "f" * 64,
        )
        self.assertEqual(status, "PASSED",
                         f"python_compile must pass on valid source; got {status}: {text}")
        self.assertEqual(len(receipts), 1, "exactly one validation receipt")
        rid = receipts[0]
        self.assertTrue(rid.startswith("rec-val-"))
        # candidate snapshot is non-empty and matches the pre-validator state.
        rec = rm.load_receipt(rid)
        self.assertTrue(rec is not None and rec.get("candidate_snapshot_digest"),
                        "candidate_snapshot_digest must be present and non-empty")
        self.assertEqual(rec["candidate_snapshot_digest"], expected_snapshot)
        self.assertTrue(
            verify_receipt(
                rid, expected_kind=KIND_VALIDATION,
                validator_id="python_compile",
                validator_command="python_compile",
                candidate_snapshot_digest=expected_snapshot,
                outcome="pass",
            ),
        )

    # ---------- 3. required ``python_compile`` FAIL -> receipt still exists, outcome=FAIL ----------

    def test_required_python_compile_fail_mints_failure_receipt(self):
        """A required ``python_compile`` validator that FAILS (broken
        Python source) still produces a trusted validation receipt; the
        receipt outcome is ``fail`` and the candidate snapshot binding is
        preserved."""
        from overnight_runner.safety import git_init_empty, git_commit_all, git_worktree_sha
        from overnight_runner.worker import _finalise
        from overnight_runner.schemas import Disposition

        repo = Path(self._tmp) / "repo"
        repo.mkdir()
        git_init_empty(repo)
        # Broken Python: a function declaration that lacks ``:``.
        (repo / "hello.py").write_text("def hello()\n    return 'ok'\n")
        git_commit_all(repo, "init")

        expected_snapshot = git_worktree_sha(repo)
        reg = default_registry()
        b = Broker(
            repo_root=repo,
            registry=reg,
            allowed_write_paths=["hello.py"],
            allowed_read_paths=["hello.py"],
        )

        minted: list[dict] = []
        from overnight_runner import receipts as rm
        def on_val(payload):
            minted.append(dict(payload))
            return rm.mint_validation_receipt(
                validator_id=payload["validator_id"],
                validator_command=payload["validator_command"],
                candidate_snapshot_digest=payload["candidate_snapshot_digest"],
                candidate_tree_state=payload.get("candidate_tree_state", "post-apply"),
                exit_code=payload.get("exit_code", 1),
                outcome=payload.get("outcome", "fail"),
                detail=payload.get("detail", ""),
                env_digest=payload.get("env_digest", ""),
                profile_digest=payload.get("profile_digest", ""),
            )

        manifest = _make_manifest(
            repo, "pyc-fail",
            required_validator_ids=["python_compile"],
            execution_class="source_mutation",
            write_paths=["hello.py"],
        )
        art = Path(self._tmp) / "artifacts"
        status, reason, text, receipts = _finalise(
            manifest, b, Disposition.DONE, art,
            applied_proposals=["c-py-fail"],
            on_validation_receipt=on_val,
            env_digest="env-" + "f" * 64,
            profile_digest="prof-" + "f" * 64,
        )
        # The task MUST FAIL because the validator failed.
        self.assertEqual(status, "FAILED")
        # reason_code is the classification; reason_text carries the
        # per-validator detail (joined with ``; ``).
        self.assertEqual(reason, "VALIDATORS_FAILED")
        self.assertIn("python_compile", text or "")
        # ... but a trusted validation receipt still exists.
        self.assertEqual(len(receipts), 1,
                         "FAIL path must still mint exactly one receipt")
        rid = receipts[0]
        self.assertTrue(rid.startswith("rec-val-"))
        # Same candidate snapshot binding as the pre-validator state.
        rec = rm.load_receipt(rid)
        self.assertTrue(rec is not None)
        self.assertEqual(rec["candidate_snapshot_digest"], expected_snapshot)
        self.assertEqual(rec["outcome"], "fail")
        # Verify: outcome=fail MUST verify; outcome=pass MUST NOT.
        self.assertTrue(
            verify_receipt(
                rid, expected_kind=KIND_VALIDATION,
                validator_id="python_compile",
                validator_command="python_compile",
                candidate_snapshot_digest=expected_snapshot,
                outcome="fail",
            ),
        )
        self.assertFalse(
            verify_receipt(
                rid, expected_kind=KIND_VALIDATION,
                candidate_snapshot_digest=expected_snapshot,
                outcome="pass",
            ),
            "FAIL receipt must NOT verify with outcome=pass",
        )

    # ---------- 4. wrong candidate snapshot for either built-in receipt is rejected ----------

    def test_wrong_snapshot_rejected_for_builtin_receipt(self):
        """A receipt produced by the ``no_op`` path is rejected when the
        caller supplies the wrong candidate snapshot. This proves the
        receipt IS bound to the candidate state the validator actually
        examined."""
        from overnight_runner.safety import git_init_empty, git_commit_all, git_worktree_sha
        from overnight_runner.worker import _finalise
        from overnight_runner.schemas import Disposition

        repo = Path(self._tmp) / "repo"
        repo.mkdir()
        git_init_empty(repo)
        (repo / "hello.py").write_text("def hello():\n    return 'ok'\n")
        git_commit_all(repo, "init")

        reg = default_registry()
        b = Broker(repo_root=repo, registry=reg,
                   allowed_write_paths=["hello.py"],
                   allowed_read_paths=["hello.py"])

        from overnight_runner import receipts as rm
        def on_val(payload):
            return rm.mint_validation_receipt(
                validator_id=payload["validator_id"],
                validator_command=payload["validator_command"],
                candidate_snapshot_digest=payload["candidate_snapshot_digest"],
                candidate_tree_state=payload.get("candidate_tree_state", "post-apply"),
                exit_code=payload.get("exit_code", 0),
                outcome=payload.get("outcome", "pass"),
                detail=payload.get("detail", ""),
                env_digest=payload.get("env_digest", ""),
                profile_digest=payload.get("profile_digest", ""),
            )

        manifest = _make_manifest(
            repo, "noop-snap",
            required_validator_ids=["no_op"],
            execution_class="source_mutation",
        )
        art = Path(self._tmp) / "artifacts"
        status, reason, text, receipts = _finalise(
            manifest, b, Disposition.DONE, art,
            applied_proposals=["c-noop-snap"],
            on_validation_receipt=on_val,
            env_digest="env-" + "f" * 64,
            profile_digest="prof-" + "f" * 64,
        )
        self.assertEqual(status, "PASSED")
        self.assertEqual(len(receipts), 1)
        rid = receipts[0]
        # Wrong snapshot must NOT verify; correct snapshot verifies.
        self.assertFalse(
            verify_receipt(rid, expected_kind=KIND_VALIDATION,
                           validator_id="no_op",
                           candidate_snapshot_digest="0" * 64),
            "wrong candidate snapshot must not verify",
        )
        correct_snapshot = git_worktree_sha(repo)
        self.assertTrue(
            verify_receipt(rid, expected_kind=KIND_VALIDATION,
                           validator_id="no_op",
                           candidate_snapshot_digest=correct_snapshot),
        )


# ----------------------------- helpers -----------------------------

class _NoChat:
    """Stub Ollama client used only for class construction (no chat calls)."""

    def chat(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("Ollama must not be called by receipt tests")


def _make_manifest(repo: Path, tag: str, **kw) -> TaskManifest:
    write_paths = kw.get("write_paths", ["hello.py"])
    base = {
        "schema_version": "1.0", "task_id": f"rec-{tag}", "title": "rec",
        "execution_class": kw.get("execution_class", "source_mutation"),
        "objective": "receipt",
        "repo": {"path": str(repo)},
        "paths": {"write_paths": write_paths, "read_paths": ["hello.py"]},
        "commands": {"required_validator_ids": kw.get("required_validator_ids", ["noop"])},
    }
    if "extra_validator" in kw:
        base["commands"]["required_validator_ids"].append(kw["extra_validator"])
    return TaskManifest.model_validate(base)
