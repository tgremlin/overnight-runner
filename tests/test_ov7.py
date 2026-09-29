"""OV7 — capped host-tests output, the Python bwrap path refusals, and the pure
credential-exposure builders.
"""
from __future__ import annotations

import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from overnight_runner.credential_exposure import (
    KNOWN_AUTH_LOCATIONS,
    KNOWN_AUTH_VARIABLES,
    CredentialRefusal,
    assert_no_secret_in_argv,
    build_auth_file_bind,
    build_env_allowlist_variable,
)
from overnight_runner.host_tests import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_TOOLCHAIN_ROOT,
    build_host_tests_bwrap,
    cap_output,
    run_host_tests,
)

BWRAP_OK = subprocess.run(["bwrap", "--version"], capture_output=True).returncode == 0
TOOLCHAIN_OK = (Path(DEFAULT_TOOLCHAIN_ROOT) / "bin" / "python").exists()


def _owner(tmp: Path) -> tuple[Path, str]:
    repo = tmp / "owner"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
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
# §OV7-1 capped output
# --------------------------------------------------------------------------- #
def test_ov71_cap_output_truncates_and_flags() -> None:
    assert cap_output(b"x" * 10, 4) == (b"xxxx", True)
    assert cap_output(b"xx", 4) == (b"xx", False)
    assert cap_output(b"", 4) == (b"", False)
    assert DEFAULT_MAX_OUTPUT_BYTES == 64 * 1024


@pytest.mark.skipif(not (BWRAP_OK and TOOLCHAIN_OK), reason="needs bwrap and the pinned toolchain")
def test_ov71_a_flooding_test_is_capped_with_a_digest_over_the_capped_bytes(tmp_path: Path) -> None:
    repo, head = _owner(tmp_path)
    ws = _ws(tmp_path, repo, head)
    # ~400 KiB of stdout emerging through a failing assertion (pytest echoes the
    # captured stream on failure, so the bytes reach the process stdout).
    (Path(ws) / "test_flood.py").write_text(
        "import sys\n"
        "def test_flood():\n"
        "    sys.stdout.write('A' * (400 * 1024))\n"
        "    sys.stdout.flush()\n"
        "    assert False, 'flood'\n",
        encoding="utf-8",
    )
    result = run_host_tests(workspace_dir=ws, timeout_s=90)
    assert result.outcome == "failed", result.stdout[-300:]
    assert result.stdout_truncated is True
    assert result.stdout_bytes == DEFAULT_MAX_OUTPUT_BYTES
    assert len(result.stdout) == DEFAULT_MAX_OUTPUT_BYTES
    # the digest is over the CAPPED bytes, and it is a real sha256
    import hashlib

    assert result.stdout_digest == "sha256:" + hashlib.sha256(result.stdout.encode()).hexdigest()
    assert result.stdout_digest.startswith("sha256:")
    assert result.stderr_truncated is False


@pytest.mark.skipif(not (BWRAP_OK and TOOLCHAIN_OK), reason="needs bwrap and the pinned toolchain")
def test_ov71_a_custom_cap_is_honoured(tmp_path: Path) -> None:
    repo, head = _owner(tmp_path)
    ws = _ws(tmp_path, repo, head)
    (Path(ws) / "test_flood.py").write_text("import sys\ndef test_f():\n    sys.stdout.write('A' * 8192)\n    assert False\n", encoding="utf-8")
    result = run_host_tests(workspace_dir=ws, timeout_s=90, max_output_bytes=128)
    assert result.stdout_truncated is True
    assert result.stdout_bytes == 128


# --------------------------------------------------------------------------- #
# §OV7-2 the Python bwrap path refusals (mirroring the TS rules)
# --------------------------------------------------------------------------- #
HOME = os.path.expanduser("~")


def test_ov72_a_relative_workspace_or_scratch_is_refused() -> None:
    with pytest.raises(ValueError, match="relative workspaceDir"):
        build_host_tests_bwrap(workspace_dir="relative/ws", scratch_dir="/s", toolchain_python="/t/p")
    with pytest.raises(ValueError, match="relative scratch dir"):
        build_host_tests_bwrap(workspace_dir="/w", scratch_dir="rel", toolchain_python="/t/p")


def test_ov72_root_and_home_are_refused() -> None:
    with pytest.raises(ValueError, match=r"binds / read-write"):
        build_host_tests_bwrap(workspace_dir="/", scratch_dir="/s", toolchain_python="/t/p")
    with pytest.raises(ValueError, match="real HOME"):
        build_host_tests_bwrap(workspace_dir=HOME, scratch_dir="/s", toolchain_python="/t/p")


def test_ov72_a_workspace_that_is_an_ancestor_of_a_forbidden_path_is_refused() -> None:
    with pytest.raises(ValueError, match="ancestor of, or equal to, a forbidden path"):
        build_host_tests_bwrap(workspace_dir="/tmp", scratch_dir="/s", toolchain_python="/t/p", forbidden_paths=("/tmp/below",))
    with pytest.raises(ValueError, match="ancestor of, or equal to, a forbidden path"):
        build_host_tests_bwrap(workspace_dir="/tmp/same", scratch_dir="/s", toolchain_python="/t/p", forbidden_paths=("/tmp/same",))


def test_ov72_reserved_roots_and_home_related_roots_are_refused() -> None:
    for reserved in ("/", "/home", "/root", "/mnt", "/run/user"):
        with pytest.raises(ValueError, match="reserved root"):
            build_host_tests_bwrap(workspace_dir="/w", scratch_dir="/s", toolchain_python="/t/p", roots=[reserved])
    with pytest.raises(ValueError, match="under the real HOME"):
        build_host_tests_bwrap(workspace_dir="/w", scratch_dir="/s", toolchain_python="/t/p", roots=[os.path.join(HOME, ".cache")])
    # the RESERVED check runs first, exactly as in the TS gate
    with pytest.raises(ValueError, match="reserved root"):
        build_host_tests_bwrap(workspace_dir="/w", scratch_dir="/s", toolchain_python="/t/p", roots=[os.path.dirname(HOME)])
    with pytest.raises(ValueError, match="ancestor of a forbidden path"):
        build_host_tests_bwrap(workspace_dir="/w", scratch_dir="/s", toolchain_python="/t/p", forbidden_paths=("/opt/f",), roots=["/opt"])
    with pytest.raises(ValueError, match="under a forbidden path"):
        build_host_tests_bwrap(workspace_dir="/w", scratch_dir="/s", toolchain_python="/t/p", forbidden_paths=("/opt",), roots=["/opt/f"])


def test_ov72_the_legitimate_shape_still_builds() -> None:
    argv = build_host_tests_bwrap(workspace_dir="/w", scratch_dir="/s", toolchain_python="/t/bin/python")
    assert argv[0] == "bwrap" and "--ro-bind /w /w" in " ".join(argv)
    with pytest.raises(ValueError, match="scratch dir IS the workspace"):
        build_host_tests_bwrap(workspace_dir="/w", scratch_dir="/w", toolchain_python="/t/p")


# --------------------------------------------------------------------------- #
# §OV7-3 credential exposure (pure builders; no real credential)
# --------------------------------------------------------------------------- #
def test_ov73_a_read_only_bind_of_the_known_auth_file_is_built() -> None:
    known = KNOWN_AUTH_LOCATIONS["opencode"].replace("~", "/home/sandbox")
    args = build_auth_file_bind(tool="opencode", auth_path=known, stat_kind="file")
    assert args.argv == ("--ro-bind", known, known)
    assert args.env_allow == ()
    assert args.mechanism == "auth-file-readonly-bind"


def test_ov73_a_directory_bind_is_refused() -> None:
    known = KNOWN_AUTH_LOCATIONS["opencode"].replace("~", "/home/sandbox")
    with pytest.raises(CredentialRefusal) as exc:
        build_auth_file_bind(tool="opencode", auth_path=known, stat_kind="dir")
    assert exc.value.code == "AUTH_BIND_IS_DIRECTORY"


@pytest.mark.parametrize("kind,code", [("symlink", "AUTH_BIND_IS_SYMLINK"), ("missing", "AUTH_BIND_NOT_A_FILE")])
def test_ov73_a_non_regular_file_is_refused(kind: str, code: str) -> None:
    known = KNOWN_AUTH_LOCATIONS["commandcode"].replace("~", "/home/sandbox")
    with pytest.raises(CredentialRefusal) as exc:
        build_auth_file_bind(tool="commandcode", auth_path=known, stat_kind=kind)
    assert exc.value.code == code


def test_ov73_a_path_outside_the_known_auth_location_is_refused() -> None:
    for bad in ("/home/sandbox/.ssh/id_rsa", "/home/sandbox/.config/opencode", "/tmp/auth.json"):
        with pytest.raises(CredentialRefusal) as exc:
            build_auth_file_bind(tool="opencode", auth_path=bad, stat_kind="file")
        assert exc.value.code == "AUTH_PATH_OUTSIDE_KNOWN_LOCATION"
    with pytest.raises(CredentialRefusal) as exc:
        build_auth_file_bind(tool="unknown-tool", auth_path="/x", stat_kind="file")
    assert exc.value.code == "UNKNOWN_TOOL"


def test_ov73_the_env_allowlist_names_one_variable_and_carries_no_value() -> None:
    args = build_env_allowlist_variable(tool="opencode", variable=KNOWN_AUTH_VARIABLES["opencode"])
    assert args.env_allow == ("OPENCODE_API_KEY",)
    assert "<from-environment>" in args.argv  # the VALUE is never on the command line
    assert "OPENCODE_API_KEY" in args.argv
    assert args.mechanism == "env-allowlist-variable"


def test_ov73_an_unknown_or_wildcard_variable_is_refused() -> None:
    with pytest.raises(CredentialRefusal) as exc:
        build_env_allowlist_variable(tool="opencode", variable="AWS_SECRET_ACCESS_KEY")
    assert exc.value.code == "ENV_VARIABLE_NOT_KNOWN"
    with pytest.raises(CredentialRefusal) as exc:
        build_env_allowlist_variable(tool="commandcode", variable="COMMANDCODE_API_KEY*")
    assert exc.value.code == "ENV_VARIABLE_NOT_KNOWN"
    with pytest.raises(CredentialRefusal) as exc:
        build_env_allowlist_variable(tool="nope", variable="X")
    assert exc.value.code == "UNKNOWN_TOOL"


def test_ov73_a_secret_looking_argv_element_is_refused() -> None:
    for secret in ("sk-abcdefghijklmnopqrstuvwx", "ghp_aaaaaaaaaaaaaaaaaaaa", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop", "api_key=abcdefgh"):
        with pytest.raises(CredentialRefusal) as exc:
            assert_no_secret_in_argv(["--setenv", "SOME_KEY", secret])
        assert exc.value.code == "CREDENTIAL_IN_ARGV"
    # a positive control: ordinary argv is fine
    assert_no_secret_in_argv(["bwrap", "--ro-bind", "/w", "/w"])
    with pytest.raises(CredentialRefusal) as exc:
        build_env_allowlist_variable(tool="opencode", variable="OPENCODE_API_KEY", extra_argv=["--token", "sk-abcdefghijklmnopqrstuvwx"])
    assert exc.value.code == "CREDENTIAL_IN_ARGV"


def test_ov73_no_real_credential_file_or_hosted_call_is_needed() -> None:
    # the builders are pure: nothing here reads a credential or touches the network
    source = (Path(__file__).parent.parent / "src" / "overnight_runner" / "credential_exposure.py").read_text(encoding="utf-8")
    assert "open(" not in source
    assert not any(word in source for word in ("requests.", "urllib.", "http://", "https://", "socket."))
