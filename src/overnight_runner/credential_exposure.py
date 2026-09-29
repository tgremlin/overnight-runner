"""OV7-3 / OV8 — credential exposure for hosted harnesses: PURE ARG BUILDERS ONLY.

Two mechanisms, and no secret ever on a command line:

  * `build_auth_file_bind` — a READ-ONLY bind of ONE named auth file, from its
    HOST source path to a SANDBOX destination path under the sandbox HOME; and
  * `build_env_allowlist_variable` — an env allowlist entry naming ONE variable
    whose VALUE is supplied at spawn time.

§OV8-2: the tools' auth LOCATIONS are UNVERIFIED by default. They may only be
overridden through a versioned data file that the M3 session fills in with
OBSERVED locations (`trio.m3-auth-locations.v1`); while any location in use is
still unverified, a bind is REFUSED with `AUTH_LOCATION_UNVERIFIED` and an
actionable message.

Every builder is pure: it decides from arguments and injected facts, so it can be
exercised without a filesystem, without a real credential and without any hosted
call.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "AUTH_LOCATIONS_SCHEMA_VERSION",
    "KNOWN_AUTH_LOCATIONS",
    "KNOWN_AUTH_VARIABLES",
    "SECRET_ARGV_PATTERNS",
    "CredentialRefusal",
    "assert_no_secret_in_argv",
    "build_auth_file_bind",
    "build_env_allowlist_variable",
    "load_auth_locations",
]

#: §OV8-2 — the only accepted shape for an operator-supplied location file.
AUTH_LOCATIONS_SCHEMA_VERSION = "trio.m3-auth-locations.v1"
AUTH_LOCATIONS_PATH = "tools/qualification/auth-locations.v1.json"

#: The tools' auth locations as shipped: UNVERIFIED. Each value is the location
#: M3 must OBSERVE; until the data file says otherwise, nothing may use it.
KNOWN_AUTH_LOCATIONS: Mapping[str, Mapping[str, Any]] = {
    "opencode": {"location": "~/.local/share/opencode/auth.json", "verified": False},
    "commandcode": {"location": "~/.commandcode/auth.json", "verified": False},
}

#: The tool's known credential VARIABLE name. One variable, by name.
KNOWN_AUTH_VARIABLES: Mapping[str, str] = {
    "opencode": "OPENCODE_API_KEY",
    "commandcode": "COMMANDCODE_API_KEY",
}

#: Secret shapes that must never appear in an argv element.
SECRET_ARGV_PATTERNS: Sequence[re.Pattern] = (
    re.compile(r"(?:^|[^A-Za-z0-9])(?:sk|pk|rk)-[A-Za-z0-9]{16,}"),
    re.compile(r"ghp_|gho_|github_pat_"),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),  # a JWT
    re.compile(r"(?i)(?:api[_-]?key|token|secret|password)\s*[=:]\s*\S{8,}"),
)

DEFAULT_SANDBOX_HOME = "/home/sandbox"


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
    source_path: str = ""
    destination_path: str = ""


def assert_no_secret_in_argv(argv: Iterable[str]) -> None:
    """Refuse ANY argv element that looks like a secret."""
    for element in argv:
        for pattern in SECRET_ARGV_PATTERNS:
            if pattern.search(element):
                raise CredentialRefusal(
                    "CREDENTIAL_IN_ARGV",
                    "an argv element looks like a secret; secrets never go on a command line",
                )


def load_auth_locations(path: str | Path) -> dict[str, dict[str, Any]]:
    """§OV8-2 — load the versioned location file M3 fills in with observations.

    Only this versioned shape is accepted; anything else is refused rather than
    guessed at.
    """
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CredentialRefusal("AUTH_LOCATIONS_UNREADABLE", f"cannot load {path!r}: {exc}") from exc
    if not isinstance(document, Mapping) or document.get("schema_version") != AUTH_LOCATIONS_SCHEMA_VERSION:
        raise CredentialRefusal(
            "AUTH_LOCATIONS_SCHEMA",
            f"not a {AUTH_LOCATIONS_SCHEMA_VERSION} document: {path!r}",
        )
    locations = document.get("locations")
    if not isinstance(locations, Mapping):
        raise CredentialRefusal("AUTH_LOCATIONS_SCHEMA", "the document carries no locations map")
    out: dict[str, dict[str, Any]] = {}
    for tool, entry in locations.items():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("location"), str):
            raise CredentialRefusal("AUTH_LOCATIONS_SCHEMA", f"location for {tool!r} is malformed")
        out[str(tool)] = {"location": entry["location"], "verified": entry.get("verified") is True}
    return out


def _location_for(tool: str, locations: Mapping[str, Mapping[str, Any]] | None) -> tuple[str, bool]:
    """The tool's host location and whether it is VERIFIED."""
    table = locations if locations is not None else KNOWN_AUTH_LOCATIONS
    if tool not in table:
        raise CredentialRefusal("UNKNOWN_TOOL", f"no known auth location for {tool!r}")
    entry = table[tool]
    if isinstance(entry, str):  # a bare string means "unverified"
        return entry, False
    return str(entry.get("location", "")), entry.get("verified") is True


def build_auth_file_bind(
    *,
    tool: str,
    source_path: str,
    destination_path: str | None = None,
    stat_kind: str,
    host_home: str,
    sandbox_home: str = DEFAULT_SANDBOX_HOME,
    locations: Mapping[str, Mapping[str, Any]] | None = None,
) -> CredentialArgs:
    """A read-only bind of the tool's auth file: HOST source -> SANDBOX destination.

    §OV8-1: the source and the destination are SEPARATE. The source must be the
    tool's known host auth file and the destination must be the SAME relative path
    under `sandbox_home`; binding the host path to itself (the old form) is refused.

    §OV8-2: while the location is still marked unverified, the bind is refused.
    """
    location, verified = _location_for(tool, locations)
    if not verified:
        raise CredentialRefusal(
            "AUTH_LOCATION_UNVERIFIED",
            f"the auth location for {tool!r} ({location}) is UNVERIFIED; run the M3 session, "
            f"record the observed path, and write it to {AUTH_LOCATIONS_PATH} as "
            f"{AUTH_LOCATIONS_SCHEMA_VERSION} with verified:true before building a bind",
        )
    known = location.replace("~", host_home)
    if source_path != known:
        raise CredentialRefusal(
            "AUTH_PATH_OUTSIDE_KNOWN_LOCATION",
            f"{source_path!r} is not the known host auth file for {tool!r} ({known})",
        )
    if stat_kind == "dir":
        raise CredentialRefusal("AUTH_BIND_IS_DIRECTORY", f"{known} is a directory; bind the auth FILE, never its directory")
    if stat_kind == "symlink":
        raise CredentialRefusal("AUTH_BIND_IS_SYMLINK", f"{known} is a symlink; bind a regular file")
    if stat_kind != "file":
        raise CredentialRefusal("AUTH_BIND_NOT_A_FILE", f"{known} is not a regular file (stat_kind={stat_kind!r})")

    if destination_path is None or destination_path == "":
        raise CredentialRefusal(
            "AUTH_DESTINATION_REQUIRED",
            "the sandbox destination path is required and must be a separate argument",
        )
    if source_path == destination_path:
        raise CredentialRefusal(
            "AUTH_BIND_SAME_PATH",
            "the source and destination must differ; binding a host path onto itself exposes the host layout",
        )
    sandbox_known = location.replace("~", sandbox_home)
    if not destination_path.startswith(sandbox_home.rstrip("/") + "/"):
        raise CredentialRefusal(
            "AUTH_DESTINATION_OUTSIDE_SANDBOX_HOME",
            f"{destination_path!r} is not under the sandbox home {sandbox_home}",
        )
    if destination_path != sandbox_known:
        raise CredentialRefusal(
            "AUTH_DESTINATION_MISMATCH",
            f"{destination_path!r} is not the tool's location under the sandbox home ({sandbox_known})",
        )

    argv = ("--ro-bind", source_path, destination_path)
    assert_no_secret_in_argv(argv)
    return CredentialArgs(
        argv=argv, env_allow=(), mechanism="auth-file-readonly-bind",
        source_path=source_path, destination_path=destination_path,
    )


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
    if "*" in variable:
        raise CredentialRefusal("ENV_VARIABLE_WILDCARD", "a wildcard credential name is refused")
    argv = tuple(["--setenv", known, "<from-environment>", *extra_argv])
    assert_no_secret_in_argv(argv)
    return CredentialArgs(argv=argv, env_allow=(known,), mechanism="env-allowlist-variable")
