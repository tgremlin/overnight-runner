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


# ----------------------------- helpers -----------------------------

class _NoChat:
    """Stub Ollama client used only for class construction (no chat calls)."""

    def chat(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("Ollama must not be called by receipt tests")


def _make_manifest(repo: Path, tag: str, **kw) -> TaskManifest:
    base = {
        "schema_version": "1.0", "task_id": f"rec-{tag}", "title": "rec",
        "execution_class": "source_mutation", "objective": "receipt",
        "repo": {"path": str(repo)},
        "paths": {"write_paths": ["hello.py"], "read_paths": ["hello.py"]},
        "commands": {"required_validator_ids": kw.get("required_validator_ids", ["noop"])},
    }
    if "extra_validator" in kw:
        base["commands"]["required_validator_ids"].append(kw["extra_validator"])
    return TaskManifest.model_validate(base)
