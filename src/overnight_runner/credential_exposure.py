"""OV7-3 — credential exposure for hosted harnesses: PURE ARG BUILDERS ONLY.

The M3 qualification needs a hosted harness to authenticate inside the sandbox.
This module offers exactly two mechanisms, and neither puts a secret in a command
line:

  * `build_auth_file_bind` — a READ-ONLY bind of ONE named auth file that lives at
    the tool's known auth location; and
  * `build_env_allowlist_variable` — an env allowlist entry naming ONE variable
    whose VALUE is supplied at spawn time, never on the command line.

Every builder is pure: it decides from arguments and injected facts, so it can be
exercised without touching the filesystem, without a real credential and without
any hosted call. Refusals are typed.

Refused:
  * a bind of anything that is not a regular file (a directory, a symlink, a
    missing path);
  * a bind of a path outside the tool's KNOWN auth location;
  * an env variable not in the tool's known set;
  * ANY secret-looking value in the argv;
  * more than one credential, and any wildcard.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

__all__ = [
    "KNOWN_AUTH_LOCATIONS",
    "KNOWN_AUTH_VARIABLES",
    "SECRET_ARGV_PATTERNS",
    "CredentialRefusal",
    "assert_no_secret_in_argv",
    "build_auth_file_bind",
    "build_env_allowlist_variable",
]

#: The tool's known auth file LOCATION. A bind may only name the file itself.
KNOWN_AUTH_LOCATIONS: Mapping[str, str] = {
    "opencode": "~/.local/share/opencode/auth.json",
    "commandcode": "~/.commandcode/auth.json",
}

#: The tool's known credential VARIABLE name. One variable, by name.
KNOWN_AUTH_VARIABLES: Mapping[str, str] = {
    "opencode": "OPENCODE_API_KEY",
    "commandcode": "COMMANDCODE_API_KEY",
}

#: Secret shapes that must never appear in an argv element (the TS runner's set,
#: plus the shapes our own tooling emits).
SECRET_ARGV_PATTERNS: Sequence[re.Pattern] = (
    re.compile(r"(?:^|[^A-Za-z0-9])(?:sk|pk|rk)-[A-Za-z0-9]{16,}"),
    re.compile(r"ghp_|gho_|github_pat_"),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),  # a JWT
    re.compile(r"(?i)(?:api[_-]?key|token|secret|password)\s*[=:]\s*\S{8,}"),
)


class CredentialRefusal(ValueError):
    """A credential exposure the builders refuse. Typed by `code`."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class CredentialArgs:
    """The argv fragments plus the env allowlist a caller must apply."""

    argv: tuple[str, ...]
    env_allow: tuple[str, ...]
    mechanism: str


def assert_no_secret_in_argv(argv: Iterable[str]) -> None:
    """Refuse ANY argv element that looks like a secret."""
    for element in argv:
        for pattern in SECRET_ARGV_PATTERNS:
            if pattern.search(element):
                raise CredentialRefusal(
                    "CREDENTIAL_IN_ARGV",
                    "an argv element looks like a secret; secrets never go on a command line",
                )


def _known_auth_path(tool: str, path: str, home: str) -> str:
    if tool not in KNOWN_AUTH_LOCATIONS:
        raise CredentialRefusal("UNKNOWN_TOOL", f"no known auth location for {tool!r}")
    known = KNOWN_AUTH_LOCATIONS[tool].replace("~", home)
    if path != known:
        raise CredentialRefusal(
            "AUTH_PATH_OUTSIDE_KNOWN_LOCATION",
            f"{path!r} is not the known auth file for {tool!r} ({known})",
        )
    return known


def build_auth_file_bind(
    *,
    tool: str,
    auth_path: str,
    stat_kind: str,
    home: str = "/home/sandbox",
) -> CredentialArgs:
    """A read-only bind of ONE named auth file.

    `stat_kind` is the TRUSTED caller's observation: 'file' | 'dir' | 'symlink' |
    'missing'. Anything but a regular file is refused, and the path must be the
    tool's known auth file exactly.
    """
    known = _known_auth_path(tool, auth_path, home)
    if stat_kind == "dir":
        raise CredentialRefusal("AUTH_BIND_IS_DIRECTORY", f"{known} is a directory; bind the auth FILE, never its directory")
    if stat_kind == "symlink":
        raise CredentialRefusal("AUTH_BIND_IS_SYMLINK", f"{known} is a symlink; bind a regular file")
    if stat_kind != "file":
        raise CredentialRefusal("AUTH_BIND_NOT_A_FILE", f"{known} is not a regular file (stat_kind={stat_kind!r})")
    argv = ("--ro-bind", known, known)
    assert_no_secret_in_argv(argv)
    return CredentialArgs(argv=argv, env_allow=(), mechanism="auth-file-readonly-bind")


def build_env_allowlist_variable(*, tool: str, variable: str, extra_argv: Sequence[str] = ()) -> CredentialArgs:
    """An env allowlist entry naming ONE variable; the VALUE is never in argv."""
    if tool not in KNOWN_AUTH_VARIABLES:
        raise CredentialRefusal("UNKNOWN_TOOL", f"no known credential variable for {tool!r}")
    known = KNOWN_AUTH_VARIABLES[tool]
    if variable != known:
        raise CredentialRefusal(
            "ENV_VARIABLE_NOT_KNOWN",
            f"{variable!r} is not the known credential variable for {tool!r} ({known})",
        )
    if variable.endswith("*") or "*" in variable:
        raise CredentialRefusal("ENV_VARIABLE_WILDCARD", "a wildcard credential name is refused")
    # The value is supplied by the runner at spawn time, so the argv carries only
    # the variable NAME plus whatever the caller asked for.
    argv = tuple(["--setenv", known, "<from-environment>", *extra_argv])
    assert_no_secret_in_argv(argv)
    return CredentialArgs(argv=argv, env_allow=(known,), mechanism="env-allowlist-variable")
