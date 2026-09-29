"""OV4-2 — the Runner's own scope gate (parity with the compiler's TS gate).

A port of `evaluateScope` from `packages/candidate-workspace/src/scope.ts`. The
Runner RECOMPUTES with this; it never trusts a submitted verdict or a submitted
change set. `tests/test_scope_parity.py` runs the shared fixture set against both
implementations and requires identical code sets.

No IO, no clock, no Runner state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["ScopeRefusal", "canonical_scope_path", "evaluate_scope"]

DEFAULT_MAX_WRITE_PATHS = 4096
DEFAULT_MAX_PATH_LENGTH = 255
DEFAULT_MAX_FILE_BYTES = 8 * 1024 * 1024
DEFAULT_LICENSED_PREFIXES = ("/mnt/storage/TRIO-Forge/licensed", "Content/Synty")


@dataclass(frozen=True)
class ScopeRefusal:
    code: str
    path: str
    detail: str = ""


def canonical_scope_path(raw: Any) -> tuple[bool, str, str]:
    """(ok, path, reason) — refuses absolute/traversal/NUL/backslash/repeats."""
    if not isinstance(raw, str) or raw == "":
        return False, "", "empty path"
    if "\x00" in raw:
        return False, raw, "NUL byte"
    if "\\" in raw:
        return False, raw, "backslash separator"
    if raw.startswith("/"):
        return False, raw, "absolute path"
    segments = raw.split("/")
    if any(s == "" for s in segments):
        return False, raw, "empty segment (leading/repeated/trailing separator)"
    if any(s == "." for s in segments):
        return False, raw, "`.` segment"
    if any(s == ".." for s in segments):
        return False, raw, "`..` segment"
    return True, "/".join(segments), ""


def _entry_field(entry: Any, name: str, default: Any = None) -> Any:
    """Read a snapshot entry as either a Mapping or a SnapshotEntry dataclass."""
    if entry is None:
        return default
    if isinstance(entry, Mapping):
        return entry.get(name, default)
    snake = "".join("_" + c.lower() if c.isupper() else c for c in name)
    return getattr(entry, snake, getattr(entry, name, default))


def _matches(rule: str, path: str, case_sensitive: bool) -> bool:
    fold = (lambda s: s) if case_sensitive else (lambda s: s.lower())
    r = fold(rule.rstrip("/"))
    p = fold(path)
    return p == r or p.startswith(r + "/")


def _is_git_internal(path: str) -> bool:
    return any(seg == ".git" or seg.startswith(".git") for seg in path.split("/"))


def _protected(contract: Mapping[str, Any], path: str, case_sensitive: bool) -> bool:
    return any(_matches(r, path, case_sensitive) for r in contract.get("protectedPaths") or [])


def _licensed(contract: Mapping[str, Any], path: str, case_sensitive: bool) -> bool:
    prefixes = contract.get("licensedPathPrefixes") or list(DEFAULT_LICENSED_PREFIXES)
    return any(_matches(r, path, case_sensitive) for r in prefixes)


def evaluate_scope(
    contract: Mapping[str, Any],
    change_set: Iterable[Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, list[ScopeRefusal]]:
    """PASS or typed refusals. `after` is the TRUSTED snapshot (path -> entry)."""
    refusals: list[ScopeRefusal] = []
    case_sensitive = contract.get("caseSensitive", True) is not False
    max_paths = contract.get("maxWritePaths", DEFAULT_MAX_WRITE_PATHS)
    max_path_len = contract.get("maxPathLength", DEFAULT_MAX_PATH_LENGTH)
    max_file_bytes = contract.get("maxFileBytes", DEFAULT_MAX_FILE_BYTES)
    allowed = list(contract.get("allowedWritePaths") or [])
    allowed_prefixes = list(contract.get("allowedWritePrefixes") or [])
    declared_exec = list(contract.get("declaredExecutablePaths") or [])
    outputs = contract.get("outputs", None)
    canonical: list[str] = []

    for change in change_set:
        raw_path = change.get("path")
        ok, path, reason = canonical_scope_path(raw_path)
        if not ok:
            refusals.append(ScopeRefusal("SCOPE_PATH_INVALID", str(raw_path), f"path is not a valid repo-relative reference ({reason})"))
            continue
        canonical.append(path)
        kind = change.get("kind")
        if kind not in ("added", "modified", "deleted", "renamed"):
            refusals.append(ScopeRefusal("SCOPE_UNKNOWN_CHANGE_KIND", path, f"unknown change kind {kind!r}"))
        entry = after.get(path)
        if len(path) > max_path_len:
            refusals.append(ScopeRefusal("SCOPE_PATH_TOO_LONG", path, f"path exceeds {max_path_len} characters"))
        # §OV4-1 / parity: any .git* entry is refused outright.
        if _is_git_internal(path):
            refusals.append(ScopeRefusal("SCOPE_GIT_INTERNAL", path, "a .git/.git* path may not be created or changed by a candidate"))
        if path == ".gitmodules":
            refusals.append(ScopeRefusal("SCOPE_SUBMODULE_CHANGE", path, "a submodule declaration may not be changed by a candidate"))
        if _licensed(contract, path, case_sensitive):
            refusals.append(ScopeRefusal("SCOPE_LICENSED_PATH", path, "licensed-asset path may not be written"))
        for forbidden in contract.get("forbiddenScope") or []:
            if _matches(forbidden, path, case_sensitive):
                refusals.append(ScopeRefusal("SCOPE_FORBIDDEN_SCOPE", path, f"write falls under forbidden scope {forbidden!r}"))
        if _protected(contract, path, case_sensitive):
            refusals.append(ScopeRefusal("SCOPE_PROTECTED_WRITE", path, "write touches a protected path"))

        is_allowed = path in allowed or any(_matches(p, path, case_sensitive) for p in allowed_prefixes)
        if kind != "deleted" and not is_allowed:
            refusals.append(ScopeRefusal("SCOPE_WRITE_OUT_OF_SCOPE", path, "write is outside allowedWritePaths"))
        if kind == "deleted" and not is_allowed:
            refusals.append(ScopeRefusal("SCOPE_DELETE_OUT_OF_SCOPE", path, "deletion is outside allowedWritePaths"))
        if kind == "renamed" and change.get("from") is not None:
            ok_from, from_path, _ = canonical_scope_path(change.get("from"))
            if ok_from:
                from_protected = _protected(contract, from_path, case_sensitive)
                to_protected = _protected(contract, path, case_sensitive)
                if from_protected and not to_protected:
                    refusals.append(ScopeRefusal("SCOPE_PROTECTED_RENAME_OUT", path, f"rename moves a protected path out ({from_path})"))
                if not from_protected and to_protected:
                    refusals.append(ScopeRefusal("SCOPE_PROTECTED_RENAME_IN", path, f"rename moves a path into a protected path ({from_path})"))
                from_allowed = from_path in allowed or any(_matches(p, from_path, case_sensitive) for p in allowed_prefixes)
                if not from_allowed:
                    refusals.append(ScopeRefusal("SCOPE_RENAME_OUT_OF_SCOPE", path, f"rename source {from_path} is outside allowedWritePaths"))
        if kind == "deleted" and _protected(contract, path, case_sensitive):
            refusals.append(ScopeRefusal("SCOPE_PROTECTED_DELETE", path, "deletion of a protected file"))

        if entry is not None:
            if _entry_field(entry, "type") == "symlink":
                refusals.append(ScopeRefusal("SCOPE_SYMLINK_CREATED", path, "a candidate may not create a symlink"))
                target = _entry_field(entry, "linkTarget", "") or ""
                if target.startswith("/") or ".." in target.split("/"):
                    refusals.append(ScopeRefusal("SCOPE_SYMLINK_ESCAPE", path, f"symlink target escapes the workspace ({target})"))
            if int(_entry_field(entry, "nlink", 1) or 1) > 1:
                refusals.append(ScopeRefusal("SCOPE_HARDLINK", path, "a changed file has more than one hardlink"))
            if int(_entry_field(entry, "size", 0) or 0) > max_file_bytes:
                refusals.append(ScopeRefusal("SCOPE_FILE_TOO_LARGE", path, f"file is {_entry_field(entry, 'size')} bytes (> {max_file_bytes})"))
            executable = (int(_entry_field(entry, "mode", 0) or 0) & 0o111) != 0
            if kind == "added" and executable and path not in declared_exec:
                refusals.append(ScopeRefusal("SCOPE_UNEXPECTED_EXEC_BIT", path, "a newly added file is executable and no declaration allows it"))
            if kind == "modified" and "mode" in (change.get("reasons") or ()) and contract.get("allowModeChanges") is not True:
                refusals.append(ScopeRefusal("SCOPE_MODE_CHANGE", path, "exec/mode bit changed without being declared"))
        if kind == "added" and outputs is not None and path not in outputs:
            refusals.append(ScopeRefusal("SCOPE_UNDECLARED_OUTPUT", path, "new file is not one of the contract outputs"))

    if case_sensitive is False:
        folded: dict[str, str] = {p.lower(): p for p in after}
        for path in canonical:
            key = path.lower()
            prior = folded.get(key)
            if prior is not None and prior != path:
                refusals.append(ScopeRefusal("SCOPE_CASE_COLLISION", path, f"case-fold collision with {prior}"))
            folded[key] = path

    if len(canonical) > max_paths:
        refusals.append(ScopeRefusal("SCOPE_TOO_MANY_WRITES", "*", f"{len(canonical)} changed paths exceeds the {max_paths} cap"))
    return (len(refusals) == 0, refusals)
