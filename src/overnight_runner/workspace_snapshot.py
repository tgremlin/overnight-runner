"""OV4-2 — trusted workspace snapshot + change set (Runner side).

A Python port of the compiler's `snapshotWorkspace`/`diffSnapshots`
(`packages/candidate-workspace/src/index.ts`) so the Runner can compute a change
set from a workspace IT holds, instead of trusting a caller-supplied list.

Parity is enforced by `tests/test_scope_parity.py` against the shared fixture set
`tests/parity/scope-fixtures.json`, which the TypeScript gate is also run on.

No IO beyond reading the workspace; no clock; no Runner state.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ChangeEntry",
    "SnapshotEntry",
    "candidate_tree_digest",
    "diff_snapshots",
    "snapshot_workspace",
]

MAX_HASH_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class SnapshotEntry:
    path: str
    type: str  # 'file' | 'symlink'
    mode: int
    size: int
    digest: str
    binary: bool
    link_target: str | None
    nlink: int

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path, "type": self.type, "mode": self.mode, "size": self.size,
            "digest": self.digest, "binary": self.binary, "linkTarget": self.link_target,
            "nlink": self.nlink,
        }


@dataclass(frozen=True)
class ChangeEntry:
    kind: str  # added | modified | deleted | renamed | <anything else = unknown>
    path: str
    reasons: tuple[str, ...] = ()
    from_path: str | None = None


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def snapshot_workspace(root: str) -> dict[str, SnapshotEntry]:
    """Walk the workspace. `.git` is NOT skipped (OV4-1)."""
    entries: dict[str, SnapshotEntry] = {}
    base = os.path.realpath(root)

    def walk(rel: str) -> None:
        abs_dir = base if rel == "" else os.path.join(base, rel)
        for name in sorted(os.listdir(abs_dir)):
            child_rel = name if rel == "" else f"{rel}/{name}"
            st = os.lstat(os.path.join(base, child_rel))
            if os.path.isdir(os.path.join(base, child_rel)) and not os.path.islink(os.path.join(base, child_rel)):
                walk(child_rel)
            elif os.path.islink(os.path.join(base, child_rel)):
                target = os.readlink(os.path.join(base, child_rel))
                entries[child_rel] = SnapshotEntry(child_rel, "symlink", st.st_mode & 0o777, 0, _digest(target.encode()), False, target, st.st_nlink)
            elif os.path.isfile(os.path.join(base, child_rel)):
                size = st.st_size
                data = b""
                if size <= MAX_HASH_BYTES:
                    with open(os.path.join(base, child_rel), "rb") as fh:
                        data = fh.read()
                entries[child_rel] = SnapshotEntry(child_rel, "file", st.st_mode & 0o777, size, _digest(data), b"\x00" in data, None, st.st_nlink)

    walk("")
    return entries


def diff_snapshots(before: dict[str, SnapshotEntry], after: dict[str, SnapshotEntry]) -> list[ChangeEntry]:
    """Diff two trusted snapshots (adds/modifies/deletes/renames), as in TS."""
    changes: list[ChangeEntry] = []
    removed = [before[p] for p in before if p not in after]
    for path in sorted(after):
        a = after[path]
        b = before.get(path)
        if b is None:
            match = next((r for r in removed if r.type == a.type and r.size == a.size and r.digest == a.digest), None)
            if match is not None:
                removed.remove(match)
                changes.append(ChangeEntry("renamed", path, ("rename",), match.path))
            else:
                changes.append(ChangeEntry("added", path, ("new file",)))
            continue
        reasons: list[str] = []
        if b.digest != a.digest or b.size != a.size:
            reasons.append("content")
        if b.mode != a.mode:
            reasons.append("mode")
        if b.type != a.type:
            reasons.append("type")
        if b.type == "symlink" and a.type == "symlink" and b.link_target != a.link_target:
            reasons.append("symlink-target")
        if reasons:
            changes.append(ChangeEntry("modified", path, tuple(reasons)))
    for r in removed:
        changes.append(ChangeEntry("deleted", r.path, ("deleted",)))
    return sorted(changes, key=lambda c: c.path)


def candidate_tree_digest(root: str) -> str:
    """§OV6-2 — the canonical digest of the tree the Runner holds.

    Same algorithm as the trusted snapshot (path, type, mode, size, content
    digest, nlink, symlink target), sorted by path, hashed over canonical JSON.
    The digest therefore covers exactly what the change set is computed from.
    """
    entries = snapshot_workspace(root)
    payload = [
        {
            "path": e.path, "type": e.type, "mode": e.mode, "size": e.size,
            "digest": e.digest, "nlink": e.nlink, "link_target": e.link_target,
        }
        for _, e in sorted(entries.items())
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
