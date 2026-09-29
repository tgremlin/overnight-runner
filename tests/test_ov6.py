"""OV6 — the env-flag chokepoint, the Runner-computed tree digest, and the
sandboxed host-tests validator.
"""
from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

import pytest

from overnight_runner.admission_trust_boundary import TRUST_BOUNDARY_ENV, TRUST_BOUNDARY_FLAG, check_completion
from overnight_runner.campaign import record_chunk_accepted
from overnight_runner.db import Database
from overnight_runner.host_tests import (
    DEFAULT_TOOLCHAIN_ROOT,
    build_host_tests_bwrap,
    run_host_tests,
)
from overnight_runner.safety import SafetyError
from overnight_runner.validator_evidence import run_trusted_validators
from overnight_runner.workspace_snapshot import candidate_tree_digest

BWRAP_OK = subprocess.run(["bwrap", "--version"], capture_output=True).returncode == 0
TOOLCHAIN_OK = (Path(DEFAULT_TOOLCHAIN_ROOT) / "bin" / "python").exists()


def _owner(tmp: Path, files=None) -> tuple[Path, str]:
    repo = tmp / "owner"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    for rel, content in (files or {}).items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)  # noqa: E731
    run("init", "-q", "-b", "main")
    run("config", "user.email", "t@e.com")
    run("config", "user.name", "t")
    run("add", ".")
    run("commit", "-q", "-m", "i")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return repo, head


def _ws(tmp: Path, repo: Path, head: str) -> str:
    ws = tmp / "ws"
    ws.mkdir()
    archive = tmp / "b.tar"
    with open(archive, "wb") as fh:
        subprocess.run(["git", "--git-dir", str(repo / ".git"), "archive", "--format=tar", head], stdout=fh, check=True)
    with tarfile.open(archive) as tf:
        tf.extractall(ws)
    return str(ws)


# --------------------------------------------------------------------------- #
# §OV6-1 the environment flag reaches the chokepoint with NO kwargs
# --------------------------------------------------------------------------- #
def test_ov61_env_unset_is_a_no_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRUST_BOUNDARY_ENV, raising=False)
    db = Database(tmp_path / "t.sqlite")
    # the pre-existing behaviour is what raises here (no such chunk), NOT the boundary
    with pytest.raises(SafetyError) as exc:
        record_chunk_accepted(db, chunk_id="missing", accepted_commit="a" * 40, accepted_tree_digest="b" * 40)
    assert "completion trust boundary refused" not in str(exc.value)


def test_ov61_env_on_with_no_kwargs_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TRUST_BOUNDARY_ENV, TRUST_BOUNDARY_FLAG)
    db = Database(tmp_path / "t.sqlite")
    with pytest.raises(SafetyError) as exc:
        # NO kwargs at all: the environment variable alone must enforce
        record_chunk_accepted(db, chunk_id="missing", accepted_commit="a" * 40, accepted_tree_digest="b" * 40)
    assert "UNWIRED_WORKSPACE" in str(exc.value)


def test_ov61_env_foreign_value_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TRUST_BOUNDARY_ENV, "on")
    db = Database(tmp_path / "t.sqlite")
    with pytest.raises(SafetyError) as exc:
        record_chunk_accepted(db, chunk_id="missing", accepted_commit="a" * 40, accepted_tree_digest="b" * 40)
    assert "UNKNOWN_TRUST_BOUNDARY_FLAG" in str(exc.value)


def test_ov61_explicit_flag_off_still_no_ops(tmp_path: Path) -> None:
    decision = check_completion(flag="")
    assert decision.ok and decision.applied is False


# --------------------------------------------------------------------------- #
# §OV6-2 the Runner computes the candidate tree digest itself
# --------------------------------------------------------------------------- #
def test_ov62_the_tree_digest_is_canonical_and_stable(tmp_path: Path) -> None:
    repo, head = _owner(tmp_path)
    ws = _ws(tmp_path, repo, head)
    first = candidate_tree_digest(ws)
    assert first == candidate_tree_digest(ws)  # deterministic
    (Path(ws) / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert candidate_tree_digest(ws) != first  # content-sensitive


def test_ov62_a_mismatched_accepted_tree_digest_is_refused(tmp_path: Path) -> None:
    repo, head = _owner(tmp_path)
    ws = _ws(tmp_path, repo, head)
    from overnight_runner.baseline import baseline_from_workspace_creation

    baseline = baseline_from_workspace_creation(workspace_dir=ws, base_commit=head)

    class Spec:
        permitted_write_paths = ["src/a.py"]

    class Grant:
        allowed_write_paths = ["src/a.py"]
        protected_paths = []

    decision = check_completion(
        workspace_dir=ws, baseline=baseline, chunk_spec=Spec(), grant=Grant(),
        accepted_tree_digest="0" * 64, flag=TRUST_BOUNDARY_FLAG,
    )
    assert (decision.ok, decision.code) == (False, "TREE_DIGEST_MISMATCH")


def test_ov62_evidence_from_a_different_tree_is_refused(tmp_path: Path) -> None:
    repo, head = _owner(tmp_path)
    ws = _ws(tmp_path, repo, head)
    evidence = run_trusted_validators(workspace_dir=ws, base_commit=head, candidate_tree_digest="t" * 64)
    # the Runner's evidence claims tree "t"*64, but the workspace it holds is not it
    assert evidence.candidate_tree_digest == "t" * 64
    from overnight_runner.validator_evidence import intake_runner_evidence

    decision = intake_runner_evidence(submitted=evidence.as_dict(), runner_evidence=evidence, current_tree_digest=candidate_tree_digest(ws))
    assert (decision.ok, decision.code) == (False, "EVIDENCE_BINDING_MISMATCH")


# --------------------------------------------------------------------------- #
# §OV6-3 sandboxed host-tests
# --------------------------------------------------------------------------- #
def test_ov63_the_bwrap_argv_follows_the_ts_rules() -> None:
    argv = build_host_tests_bwrap(workspace_dir="/w", scratch_dir="/s", toolchain_python="/t/bin/python")
    joined = " ".join(argv)
    assert argv[0] == "bwrap"
    assert "--ro-bind /w /w" in joined          # the workspace is READ-ONLY
    assert "--bind /s /s" in joined             # the only read-write bind is the scratch copy
    assert "--unshare-net" in argv              # no network
    assert "--clearenv" in argv
    assert joined.endswith("/t/bin/python -B -m pytest -q")
    assert "--die-with-parent" in argv


@pytest.mark.skipif(not (BWRAP_OK and TOOLCHAIN_OK), reason="needs bwrap and the pinned toolchain")
def test_ov63_a_passing_test_passes(tmp_path: Path) -> None:
    repo, head = _owner(tmp_path)
    ws = _ws(tmp_path, repo, head)
    (Path(ws) / "test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    result = run_host_tests(workspace_dir=ws, timeout_s=60)
    assert result.outcome == "passed", result.stderr[-400:]


@pytest.mark.skipif(not (BWRAP_OK and TOOLCHAIN_OK), reason="needs bwrap and the pinned toolchain")
def test_ov63_writes_to_the_workspace_are_discarded(tmp_path: Path) -> None:
    repo, head = _owner(tmp_path)
    ws = _ws(tmp_path, repo, head)
    before = candidate_tree_digest(ws)
    # the sandbox runs on a Runner-owned COPY, so a write succeeds there and is
    # discarded with the copy; the candidate workspace must be untouched.
    (Path(ws) / "test_write.py").write_text(
        "import pathlib\n"
        "def test_write():\n"
        "    pathlib.Path('written.txt').write_text('x')\n"
        "    assert pathlib.Path('written.txt').exists()\n",
        encoding="utf-8",
    )
    result = run_host_tests(workspace_dir=ws, timeout_s=60)
    assert result.outcome == "passed", result.stderr[-400:]
    assert not (Path(ws) / "written.txt").exists()          # discarded
    assert not (Path(ws) / ".pytest_cache").exists()        # discarded
    after_without_test_file = candidate_tree_digest(ws)
    assert after_without_test_file != before  # only the test file itself was added
    (Path(ws) / "test_write.py").unlink()
    assert candidate_tree_digest(ws) == before


@pytest.mark.skipif(not (BWRAP_OK and TOOLCHAIN_OK), reason="needs bwrap and the pinned toolchain")
def test_ov63_a_network_attempt_fails(tmp_path: Path) -> None:
    repo, head = _owner(tmp_path)
    ws = _ws(tmp_path, repo, head)
    (Path(ws) / "test_net.py").write_text(
        "import socket\n"
        "def test_net():\n"
        "    try:\n"
        "        socket.create_connection(('1.1.1.1', 80), timeout=3)\n"
        "    except OSError:\n"
        "        return\n"
        "    raise AssertionError('the sandbox had network access')\n",
        encoding="utf-8",
    )
    result = run_host_tests(workspace_dir=ws, timeout_s=60)
    assert result.outcome == "passed", result.stderr[-400:]


@pytest.mark.skipif(not (BWRAP_OK and TOOLCHAIN_OK), reason="needs bwrap and the pinned toolchain")
def test_ov63_a_hang_is_killed_with_no_orphan(tmp_path: Path) -> None:
    repo, head = _owner(tmp_path)
    ws = _ws(tmp_path, repo, head)
    (Path(ws) / "test_slow.py").write_text("import time\ndef test_slow():\n    time.sleep(60)\n", encoding="utf-8")
    result = run_host_tests(workspace_dir=ws, timeout_s=3)
    assert result.outcome == "timeout"
    # no orphan survives: the process group was killed. Match on the Runner's own
    # scratch path, which appears in the sandboxed argv (never on our own pytest).
    assert result.scratch_dir != ""
    leftovers = subprocess.run(["pgrep", "-f", result.scratch_dir], capture_output=True, text=True).stdout.strip()
    assert leftovers == "", f"orphans: {leftovers}"
