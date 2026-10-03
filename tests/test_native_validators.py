"""M5 candidate — the native-compile / native-automation-nullrhi trusted validators.

The judge scripts here are STUBS that honour the real CLI contract (`--sandbox --engine --lock --log-out ...`, `FAIL <CODE>  --  <detail>` on
stderr, exit 0/1). What is under test is the RUNNER's behaviour: the judge is a TRUSTED copy from the base commit, a candidate cannot edit its
own judges, the heavy lock and the editor guard are honoured, and the evidence carries digests and typed codes only.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from overnight_runner.native_validators import (
    JUDGE_PATTERNS, NATIVE_AUTOMATION_ID, NATIVE_COMPILE_ID, NativeConfig, judge_protected_paths, judge_tamper, run_native_validator, with_judge_protection,
)
from overnight_runner.scope_gate import evaluate_scope
from overnight_runner.validator_evidence import discard_runner_evidence, run_trusted_validators

STUB_COMPILE = r'''#!/usr/bin/env python3
import argparse, fcntl, json, os, sys
from pathlib import Path
p = argparse.ArgumentParser()
for a in ("--engine", "--lock", "--log-out", "--max-parallel", "--timeout"): p.add_argument(a)
p.add_argument("--sandbox", action="store_true")
a = p.parse_args()
repo = Path(__file__).resolve().parent.parent
fh = open(a.lock, "a+")
try: fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    print("FAIL NATIVE_COMPILE_LOCK_HELD  --  another heavy step holds the resource lock", file=sys.stderr); sys.exit(1)
if any("KEY" in k or "TOKEN" in k for k in os.environ):
    print("FAIL LEAKED_ENV  --  a credential variable reached the judge", file=sys.stderr); sys.exit(1)
src = "".join(f.read_text() for f in sorted(repo.glob("Source/**/*.cpp")))
Path(a.log_out).write_text(f"ran-from={Path(__file__).resolve()}\nsandbox={a.sandbox}\n" + ("Foo.cpp(1): error: BROKEN\n" if "BROKEN" in src else "Result: Succeeded\n"))
if "BROKEN" in src:
    print("FAIL NATIVE_COMPILE_FAILED  --  UBT exited 6", file=sys.stderr); sys.exit(1)
print(json.dumps({"exitCode": 0}))
'''
STUB_AUTOMATION = r'''#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
p = argparse.ArgumentParser()
for a in ("--engine", "--lock", "--log-out", "--timeout", "--tests"): p.add_argument(a)
p.add_argument("--expect", action="append", default=[]); p.add_argument("--sandbox", action="store_true")
a = p.parse_args()
repo = Path(__file__).resolve().parent.parent
src = "".join(f.read_text() for f in sorted(repo.glob("Source/**/*.cpp")))
Path(a.log_out).write_text(f"ran-from={Path(__file__).resolve()}\n")
if "AUTOFAIL" in src:
    print("FAIL NATIVE_AUTOMATION_TEST_FAILED  --  1 test(s) not Success", file=sys.stderr); sys.exit(1)
print(json.dumps({"exitCode": 0}))
'''
FILES = {
    "scripts/trio_native_compile.py": STUB_COMPILE, "scripts/trio_native_automation.py": STUB_AUTOMATION,
    "scripts/trio_validators.py": "# judge\n", "scripts/trio_log_markers.py": "# judge\n", "scripts/ue.sh": "#!/bin/sh\n",
    "pytest.ini": "[pytest]\ntestpaths = scripts\n", "pipeline.config.json": '{"qualification": {"args": ["-m", "pytest", "scripts/test_a.py"]}}\n',
    "scripts/test_a.py": "def test_a():\n    assert True\n", "Source/Mod/A.cpp": "int a = 1;\n",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture()
def world(tmp_path: Path):
    repo = tmp_path / "trusted"
    for rel, text in FILES.items():
        f = repo / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    _git(repo, "init", "-q", "-b", "main"); _git(repo, "config", "user.email", "t@e.com"); _git(repo, "config", "user.name", "t")
    _git(repo, "add", "."); _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    ws = tmp_path / "ws"
    shutil.copytree(repo, ws, ignore=shutil.ignore_patterns(".git"))
    cfg = NativeConfig(engine_root=str(tmp_path / "engine"), heavy_lock=str(tmp_path / "heavy.lock"), automation_expect=("A",))
    return {"repo": str(repo), "base": base, "ws": ws, "cfg": cfg, "tmp": tmp_path}


def run(w, vid=NATIVE_COMPILE_ID):
    return run_native_validator(vid, workspace_dir=str(w["ws"]), trusted_repo=w["repo"], base_commit=w["base"], config=w["cfg"])


# ---------------------------------------------------------------------------- positive controls
def test_a_candidate_that_passes_is_accepted_with_digests_only(world):
    (world["ws"] / "Source/Mod/B.cpp").write_text("int b = 2;\n")
    row = run(world)
    assert (row["outcome"], row["code"], row["exit_code"]) == ("passed", "OK", 0)
    assert set(row) == {"validator_id", "outcome", "code", "exit_code", "stdout_digest", "stderr_digest", "log_digest", "seconds"}     # no raw text anywhere
    for k in ("stdout_digest", "stderr_digest", "log_digest"):
        assert row[k].startswith("sha256:") and len(row[k]) == 71
    assert run(world, NATIVE_AUTOMATION_ID)["outcome"] == "passed"


def test_the_judge_that_runs_is_the_trusted_base_copy_never_the_candidates(world):
    row = run(world)
    assert row["outcome"] == "passed"
    # the stub wrote `ran-from=` into the log; recompute the digest it would have: the path is Runner scratch, not the candidate workspace
    assert not (world["ws"] / "Intermediate").exists() and not list(world["ws"].rglob("native.log"))           # the candidate tree was copied, never built in
    blob = subprocess.run(["git", "-C", world["repo"], "cat-file", "blob", f"{world['base']}:scripts/trio_native_compile.py"], capture_output=True).stdout
    assert blob == STUB_COMPILE.encode()


# ---------------------------------------------------------------------------- tamper matrix: each refused with a typed code BEFORE anything runs
@pytest.mark.parametrize("path, content, code", [
    ("scripts/trio_native_compile.py", "import sys\nsys.exit(0)  # always green\n", "JUDGE_SCRIPT_TAMPERED"),
    ("scripts/trio_native_automation.py", "import sys\nsys.exit(0)\n", "JUDGE_SCRIPT_TAMPERED"),
    ("scripts/trio_validators.py", "# weakened\n", "JUDGE_SCRIPT_TAMPERED"),
    ("scripts/trio_native_extra.py", "# a new judge\n", "JUDGE_SCRIPT_TAMPERED"),
    ("pytest.ini", "[pytest]\ntestpaths = nothing\n", "JUDGE_CONFIG_TAMPERED"),
    ("pipeline.config.json", '{"qualification": {"args": ["-m", "pytest", "scripts/test_a.py", "scripts/test_new_always_passes.py"]}}\n', "JUDGE_QUALIFICATION_TAMPERED"),
])
def test_a_candidate_cannot_edit_its_own_judges(world, path, content, code):
    (world["ws"] / path).write_text(content)
    for vid in (NATIVE_COMPILE_ID, NATIVE_AUTOMATION_ID):
        row = run(world, vid)
        assert (row["outcome"], row["code"]) == ("refused", code)
        assert row["log_digest"] is None and row["exit_code"] is None                       # nothing was executed
        assert row["tampered_paths"] == [path]


def test_deleting_or_replacing_a_judge_by_symlink_is_tampering_too(world):
    (world["ws"] / "pytest.ini").unlink()
    assert run(world)["code"] == "JUDGE_CONFIG_TAMPERED"
    (world["ws"] / "pytest.ini").symlink_to("/etc/passwd")
    assert run(world)["code"] == "JUDGE_CONFIG_TAMPERED"


def test_precedence_when_several_judges_change(world):
    (world["ws"] / "pytest.ini").write_text("x\n"); (world["ws"] / "scripts/trio_validators.py").write_text("x\n")
    row = run(world)
    assert row["code"] == "JUDGE_SCRIPT_TAMPERED" and row["tampered_paths"] == ["pytest.ini", "scripts/trio_validators.py"]


# ---------------------------------------------------------------------------- judged outcomes
def test_a_candidate_that_breaks_the_build_is_NATIVE_COMPILE_FAILED(world):
    (world["ws"] / "Source/Mod/A.cpp").write_text("int a = BROKEN;\n")
    row = run(world)
    assert (row["outcome"], row["code"]) == ("failed", "NATIVE_COMPILE_FAILED") and row["exit_code"] == 1
    assert row["log_digest"] is not None


def test_a_failed_judge_writes_a_sanitized_repair_summary(world, tmp_path, monkeypatch):
    # the evidence row stays digests+codes only; the SANITIZED summary goes to the driver repair dir
    monkeypatch.setenv("TRIO_NATIVE_REPAIR_DIR", str(tmp_path / "repair"))
    (world["ws"] / "Source/Mod/A.cpp").write_text("int a = BROKEN;\n")
    assert run(world)["outcome"] == "failed"
    text = (tmp_path / "repair" / "native-compile.repair.txt").read_text(encoding="utf-8")
    assert "FAIL NATIVE_COMPILE_FAILED" in text or "error:" in text
    assert "/mnt/" not in text and "/tmp/" not in text and "pytest-" not in text     # sanitized: no absolute scratch paths


def test_a_failing_automation_test_is_typed(world):
    (world["ws"] / "Source/Mod/A.cpp").write_text("// AUTOFAIL\n")
    assert (run(world, NATIVE_AUTOMATION_ID)["outcome"], run(world, NATIVE_AUTOMATION_ID)["code"]) == ("failed", "NATIVE_AUTOMATION_TEST_FAILED")


def test_the_heavy_lock_is_honoured_and_a_held_lock_is_a_refusal(world):
    fh = open(world["cfg"].heavy_lock, "a+")
    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        row = run(world)
        assert (row["outcome"], row["code"]) == ("refused", "NATIVE_COMPILE_LOCK_HELD")
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN); fh.close()
    assert run(world)["outcome"] == "passed"                                                # released -> runs


def test_the_editor_guard_matches_the_executable_not_a_mention(world, tmp_path):
    mention = subprocess.Popen(["python3", "-c", "import time; time.sleep(30)  # UnrealEditor"])
    try:
        assert run(world)["outcome"] == "passed"
        fake = tmp_path / "UnrealEditor"
        shutil.copy(shutil.which("sleep"), fake)
        editor = subprocess.Popen([str(fake), "30"])
        try:
            time.sleep(0.2)
            assert (run(world)["outcome"], run(world)["code"]) == ("refused", "NATIVE_EDITOR_RUNNING")
        finally:
            editor.kill(); editor.wait()
    finally:
        mention.kill(); mention.wait()


def test_the_judge_sees_no_credentials(world, monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-should-never-reach-the-judge")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_should_never_reach_the_judge")
    row = run(world)                                  # the stub FAILS with LEAKED_ENV if any KEY/TOKEN variable is visible to it
    assert (row["outcome"], row["code"]) == ("passed", "OK")


def test_missing_inputs_are_typed(world):
    assert run_native_validator(NATIVE_COMPILE_ID, workspace_dir=str(world["ws"]), trusted_repo=None, base_commit=world["base"], config=world["cfg"])["code"] == "NATIVE_TRUSTED_REPO_REQUIRED"
    assert run_native_validator(NATIVE_COMPILE_ID, workspace_dir=str(world["ws"]), trusted_repo=world["repo"], base_commit=world["base"], config=None)["code"] == "NATIVE_CONFIG_REQUIRED"
    assert run_native_validator(NATIVE_COMPILE_ID, workspace_dir=str(world["ws"]), trusted_repo=world["repo"], base_commit="0" * 40, config=world["cfg"])["code"] == "NATIVE_JUDGE_ERROR"
    cfg = NativeConfig(engine_root="/e", heavy_lock=world["cfg"].heavy_lock, vendor_plugins=((str(world["tmp"] / "absent-plugin"), "Plugins/Marketplace/X"),))
    assert run_native_validator(NATIVE_COMPILE_ID, workspace_dir=str(world["ws"]), trusted_repo=world["repo"], base_commit=world["base"], config=cfg)["code"] == "NATIVE_VENDOR_MISSING"


def test_a_relative_or_absent_workspace_is_refused(world):
    # ported from agent/t2-validators: never resolve a relative/absent workspace
    assert run_native_validator(NATIVE_COMPILE_ID, workspace_dir="relative/ws", trusted_repo=world["repo"], base_commit=world["base"], config=world["cfg"])["code"] == "NATIVE_JUDGE_ERROR"
    assert run_native_validator(NATIVE_COMPILE_ID, workspace_dir=str(world["tmp"] / "absent"), trusted_repo=world["repo"], base_commit=world["base"], config=world["cfg"])["code"] == "NATIVE_JUDGE_ERROR"


def test_vendor_plugin_sources_are_provisioned_into_the_judged_copy_only(world):
    plugin = world["tmp"] / "acf"; (plugin / "Source").mkdir(parents=True); (plugin / "Source/X.cpp").write_text("int x;\n"); (plugin / "Intermediate").mkdir(); (plugin / "Intermediate/skip").write_text("1")
    cfg = NativeConfig(engine_root="/e", heavy_lock=world["cfg"].heavy_lock, vendor_plugins=((str(plugin), "Plugins/Marketplace/ACF"),))
    row = run_native_validator(NATIVE_COMPILE_ID, workspace_dir=str(world["ws"]), trusted_repo=world["repo"], base_commit=world["base"], config=cfg)
    assert row["outcome"] == "passed"
    assert not (world["ws"] / "Plugins").exists()                                            # the candidate workspace never receives vendor files


# ---------------------------------------------------------------------------- the Runner integration and the scope gate
def test_run_trusted_validators_dispatches_the_native_ids_and_fails_closed(world):
    from overnight_runner.workspace_snapshot import candidate_tree_digest

    tree = candidate_tree_digest(str(world["ws"]))
    ev = run_trusted_validators(workspace_dir=str(world["ws"]), base_commit=world["base"], candidate_tree_digest=tree, validator_ids=("native-compile",),
                                trusted_repo=world["repo"], native_config=world["cfg"])
    assert ev.passed and ev.results[0]["code"] == "OK"
    discard_runner_evidence(ev)
    (world["ws"] / "pytest.ini").write_text("[pytest]\n")
    ev2 = run_trusted_validators(workspace_dir=str(world["ws"]), base_commit=world["base"], candidate_tree_digest=candidate_tree_digest(str(world["ws"])),
                                 validator_ids=("native-compile",), trusted_repo=world["repo"], native_config=world["cfg"])
    assert not ev2.passed and ev2.results[0]["code"] == "JUDGE_CONFIG_TAMPERED"
    discard_runner_evidence(ev2)
    ev3 = run_trusted_validators(workspace_dir=str(world["ws"]), base_commit=world["base"], candidate_tree_digest=tree, validator_ids=("native-compile",))   # no trusted repo
    assert not ev3.passed and ev3.results[0]["code"] == "NATIVE_TRUSTED_REPO_REQUIRED"
    discard_runner_evidence(ev3)


def test_the_scope_gate_protects_every_judge_path_but_not_the_chunks_own_tests(world):
    protected = judge_protected_paths(world["repo"], world["base"])
    assert protected == sorted(["pipeline.config.json", "pytest.ini", "scripts/trio_log_markers.py", "scripts/trio_native_automation.py", "scripts/trio_native_compile.py", "scripts/trio_validators.py", "scripts/ue.sh"])
    contract = with_judge_protection({"allowedWritePrefixes": ["scripts", "Source", "pytest.ini", "pipeline.config.json"], "protectedPaths": ["Config/Trio"]}, world["repo"], world["base"])
    assert "Config/Trio" in contract["protectedPaths"] and "pytest.ini" in contract["protectedPaths"]
    def verdict(path):
        ok, refusals = evaluate_scope(contract, [{"path": path, "kind": "modified"}], {path: {"type": "file", "mode": 0o644, "size": 3, "nlink": 1}})
        return ok, sorted(r.code for r in refusals)
    for judge_path in protected:
        assert verdict(judge_path) == (False, ["SCOPE_PROTECTED_WRITE"]), judge_path
    assert verdict("scripts/test_a.py") == (True, [])                                         # a chunk's OWN test stays writable
    assert verdict("Source/Mod/A.cpp") == (True, [])


def test_the_judge_pattern_table_is_exactly_the_documented_set():
    assert [p for p, _ in JUDGE_PATTERNS] == ["scripts/trio_native_*.py", "scripts/trio_validators.py", "scripts/trio_log_markers.py", "scripts/ue.sh", "pytest.ini", "pipeline.config.json", "pipeline.operator.pub"]
    assert judge_tamper.__name__ == "judge_tamper"


def test_the_judged_copy_is_a_runner_owned_git_repo_never_the_candidates(world):
    """UBT needs a git work tree (a tree that builds inside one fails without it). The Runner makes its OWN repo; the candidate's .git is never copied."""
    (world["ws"] / ".git").mkdir()
    (world["ws"] / ".git" / "HOOKS_MUST_NOT_RUN").write_text("candidate git internals")
    probe = world["tmp"] / "probe_stub.py"
    # reuse the trusted stub contract: a build that succeeds only if the judged copy is a git repo whose HEAD exists and has no candidate files in .git
    stub = STUB_COMPILE.replace('src = "".join(', 'import subprocess\nif subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "HEAD"], capture_output=True).returncode != 0 or (repo / ".git" / "HOOKS_MUST_NOT_RUN").exists():\n    print("FAIL NATIVE_COMPILE_FAILED  --  no runner-owned git repo", file=sys.stderr); sys.exit(1)\nsrc = "".join(', 1)
    repo = Path(world["repo"])
    (repo / "scripts/trio_native_compile.py").write_text(stub)
    _git(repo, "add", "."); _git(repo, "commit", "-q", "-m", "stub with git probe")
    world["base"] = _git(repo, "rev-parse", "HEAD")
    shutil.copy(repo / "scripts/trio_native_compile.py", world["ws"] / "scripts/trio_native_compile.py")
    assert run(world)["outcome"] == "passed"
    assert probe is not None
