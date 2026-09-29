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
    AUTH_LOCATIONS_PATH,
    AUTH_LOCATIONS_SCHEMA_VERSION,
    KNOWN_AUTH_LOCATIONS,
    KNOWN_AUTH_VARIABLES,
    CredentialRefusal,
    assert_no_secret_in_argv,
    build_auth_file_bind,
    build_env_allowlist_variable,
    load_auth_locations,
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
# §OV7-3 / §OV8 credential exposure (pure builders; no real credential)
# --------------------------------------------------------------------------- #
import json  # noqa: E402

HOST_HOME = "/home/operator"
SANDBOX_HOME = "/home/sandbox"
#: A VERIFIED location table, as M3 would write it after observing the tools.
VERIFIED = {
    "opencode": {"location": "~/.local/share/opencode/auth.json", "verified": True},
    "commandcode": {"location": "~/.commandcode/auth.json", "verified": True},
}


def _paths(tool: str) -> tuple[str, str]:
    location = VERIFIED[tool]["location"]
    return location.replace("~", HOST_HOME), location.replace("~", SANDBOX_HOME)


def test_ov81_source_and_destination_are_separate_and_differ() -> None:
    source, destination = _paths("opencode")
    args = build_auth_file_bind(
        tool="opencode", source_path=source, destination_path=destination,
        stat_kind="file", host_home=HOST_HOME, locations=VERIFIED,
    )
    assert args.argv == ("--ro-bind", source, destination)
    assert args.source_path == source and args.destination_path == destination
    assert source != destination  # the host layout is never reused as the sandbox path


def test_ov81_the_destination_must_be_under_the_sandbox_home() -> None:
    source, _ = _paths("opencode")
    for bad in ("/tmp/auth.json", "/home/other/.local/share/opencode/auth.json", "/home/sandboxx/a"):
        with pytest.raises(CredentialRefusal) as exc:
            build_auth_file_bind(
                tool="opencode", source_path=source, destination_path=bad,
                stat_kind="file", host_home=HOST_HOME, locations=VERIFIED,
            )
        assert exc.value.code in ("AUTH_DESTINATION_OUTSIDE_SANDBOX_HOME", "AUTH_DESTINATION_MISMATCH")


def test_ov81_the_sandbox_home_must_match() -> None:
    source, destination = _paths("opencode")
    # a destination under a DIFFERENT sandbox home is refused
    with pytest.raises(CredentialRefusal) as exc:
        build_auth_file_bind(
            tool="opencode", source_path=source, destination_path=destination,
            stat_kind="file", host_home=HOST_HOME, sandbox_home="/home/elsewhere", locations=VERIFIED,
        )
    # the destination is not under THAT sandbox home, so it is refused (the
    # outside-home check runs first; either refusal means "the home must match")
    assert exc.value.code in ("AUTH_DESTINATION_OUTSIDE_SANDBOX_HOME", "AUTH_DESTINATION_MISMATCH")
    # ...and the matching one is accepted
    assert build_auth_file_bind(
        tool="opencode", source_path=source, destination_path=destination,
        stat_kind="file", host_home=HOST_HOME, sandbox_home=SANDBOX_HOME, locations=VERIFIED,
    ).destination_path.startswith(SANDBOX_HOME + "/")


def test_ov81_the_old_same_path_form_is_refused() -> None:
    source, _ = _paths("opencode")
    with pytest.raises(CredentialRefusal) as exc:
        build_auth_file_bind(
            tool="opencode", source_path=source, destination_path=source,
            stat_kind="file", host_home=HOST_HOME, locations=VERIFIED,
        )
    assert exc.value.code == "AUTH_BIND_SAME_PATH"
    # and the destination is not optional either
    with pytest.raises(CredentialRefusal) as exc2:
        build_auth_file_bind(tool="opencode", source_path=source, destination_path=None, stat_kind="file", host_home=HOST_HOME, locations=VERIFIED)
    assert exc2.value.code == "AUTH_DESTINATION_REQUIRED"


def test_ov82_locations_are_unverified_by_default_and_refuse() -> None:
    assert all(entry["verified"] is False for entry in KNOWN_AUTH_LOCATIONS.values())
    source = KNOWN_AUTH_LOCATIONS["opencode"]["location"].replace("~", HOST_HOME)
    destination = KNOWN_AUTH_LOCATIONS["opencode"]["location"].replace("~", SANDBOX_HOME)
    with pytest.raises(CredentialRefusal) as exc:
        # no location table supplied: the shipped defaults are unverified
        build_auth_file_bind(tool="opencode", source_path=source, destination_path=destination, stat_kind="file", host_home=HOST_HOME)
    assert exc.value.code == "AUTH_LOCATION_UNVERIFIED"
    assert AUTH_LOCATIONS_PATH in str(exc.value)          # actionable
    assert "M3" in str(exc.value)                         # says who fills it in


def test_ov82_only_a_versioned_data_file_can_mark_a_location_verified(tmp_path: Path) -> None:
    good = tmp_path / "auth-locations.v1.json"
    good.write_text(json.dumps({"schema_version": AUTH_LOCATIONS_SCHEMA_VERSION, "locations": VERIFIED}), encoding="utf-8")
    loaded = load_auth_locations(good)
    assert loaded["opencode"]["verified"] is True
    source, destination = _paths("opencode")
    assert build_auth_file_bind(tool="opencode", source_path=source, destination_path=destination, stat_kind="file", host_home=HOST_HOME, locations=loaded).argv[0] == "--ro-bind"
    # a partial file: the OTHER tool is still unverified and refuses
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"schema_version": AUTH_LOCATIONS_SCHEMA_VERSION, "locations": {"opencode": VERIFIED["opencode"]}}), encoding="utf-8")
    with pytest.raises(CredentialRefusal) as exc:
        build_auth_file_bind(tool="commandcode", source_path="/x", destination_path="/home/sandbox/x", stat_kind="file", host_home=HOST_HOME, locations=load_auth_locations(partial))
    assert exc.value.code in ("UNKNOWN_TOOL", "AUTH_LOCATION_UNVERIFIED")


def test_ov82_a_wrong_schema_is_refused(tmp_path: Path) -> None:
    for body in ('{"schema_version":"nope","locations":{}}', "[]", "not json"):
        bad = tmp_path / "bad.json"
        bad.write_text(body, encoding="utf-8")
        with pytest.raises(CredentialRefusal) as exc:
            load_auth_locations(bad)
        assert exc.value.code in ("AUTH_LOCATIONS_SCHEMA", "AUTH_LOCATIONS_UNREADABLE")
    with pytest.raises(CredentialRefusal) as exc:
        load_auth_locations(tmp_path / "missing.json")
    assert exc.value.code == "AUTH_LOCATIONS_UNREADABLE"


def test_ov7_metadata_refusals_still_hold_with_a_verified_location() -> None:
    source, destination = _paths("commandcode")
    for kind, code in (("dir", "AUTH_BIND_IS_DIRECTORY"), ("symlink", "AUTH_BIND_IS_SYMLINK"), ("missing", "AUTH_BIND_NOT_A_FILE")):
        with pytest.raises(CredentialRefusal) as exc:
            build_auth_file_bind(tool="commandcode", source_path=source, destination_path=destination, stat_kind=kind, host_home=HOST_HOME, locations=VERIFIED)
        assert exc.value.code == code
    with pytest.raises(CredentialRefusal) as exc:
        build_auth_file_bind(tool="commandcode", source_path="/home/operator/.ssh/id_rsa", destination_path=destination, stat_kind="file", host_home=HOST_HOME, locations=VERIFIED)
    assert exc.value.code == "AUTH_PATH_OUTSIDE_KNOWN_LOCATION"


def test_ov73_the_env_allowlist_names_one_variable_and_carries_no_value() -> None:
    args = build_env_allowlist_variable(tool="opencode", variable=KNOWN_AUTH_VARIABLES["opencode"])
    assert args.env_allow == ("OPENCODE_API_KEY",)
    assert "<from-environment>" in args.argv
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
    assert_no_secret_in_argv(["bwrap", "--ro-bind", "/w", "/w"])
    with pytest.raises(CredentialRefusal) as exc:
        build_env_allowlist_variable(tool="opencode", variable="OPENCODE_API_KEY", extra_argv=["--token", "sk-abcdefghijklmnopqrstuvwx"])
    assert exc.value.code == "CREDENTIAL_IN_ARGV"


def test_ov73_no_real_credential_file_or_hosted_call_is_needed() -> None:
    source = (Path(__file__).parent.parent / "src" / "overnight_runner" / "credential_exposure.py").read_text(encoding="utf-8")
    assert "open(" not in source
    assert not any(word in source for word in ("requests.", "urllib.", "http://", "https://", "socket."))
