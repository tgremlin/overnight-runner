"""OV5-2 — a Runner-HELD baseline snapshot.

`intake_scope_gate` must not accept a caller-supplied pre-snapshot: a caller that
tampered with the workspace could hand over a post-tamper snapshot and hide the
change. Baselines come only from here:

  * `baseline_from_base_commit` materialises the BASE COMMIT's tree with git
    OUTSIDE the workspace and snapshots that; or
  * `baseline_from_workspace_creation` snapshots the workspace at creation time.

Issued baselines are recorded in a private registry, so a hand-built
`RunnerBaseline` is detectable and refused (`BASELINE_NOT_TRUSTED`).
"""
from __future__ import annotations

import subprocess
import tarfile
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .workspace_snapshot import SnapshotEntry, snapshot_workspace

__all__ = [
    "RunnerBaseline",
    "baseline_from_base_commit",
    "baseline_from_workspace_creation",
    "is_runner_issued",
]

#: Tokens of the baselines THIS process issued. A hand-built RunnerBaseline has a
#: fresh token that is not in here, so it is detectable.
_ISSUED_TOKENS: "set[str]" = set()


@dataclass(frozen=True)
class RunnerBaseline:
    """A baseline the RUNNER produced. Construct it through the factories only."""

    base_commit: str
    entries: Mapping[str, SnapshotEntry]
    source: str  # 'base-commit-tree' | 'workspace-creation'
    token: str = field(default_factory=lambda: uuid.uuid4().hex, compare=False, repr=False)


def _issue(baseline: RunnerBaseline) -> RunnerBaseline:
    _ISSUED_TOKENS.add(baseline.token)
    return baseline


def is_runner_issued(baseline: object) -> bool:
    return isinstance(baseline, RunnerBaseline) and baseline.token in _ISSUED_TOKENS


def baseline_from_base_commit(*, git_dir: str, base_commit: str) -> RunnerBaseline:
    """Materialise the base commit's tree OUTSIDE the workspace and snapshot it."""
    with tempfile.TemporaryDirectory(prefix="trio-runner-baseline-") as tmp:
        archive = Path(tmp) / "base.tar"
        # git runs against the OUTSIDE git dir, never against candidate metadata.
        with open(archive, "wb") as fh:
            proc = subprocess.run(
                ["git", "--git-dir", git_dir, "archive", "--format=tar", base_commit],
                stdout=fh, stderr=subprocess.PIPE, check=False,
            )
        if proc.returncode != 0:
            raise RuntimeError(f"cannot materialise base tree {base_commit}: {proc.stderr.decode(errors='replace').strip()}")
        extract = Path(tmp) / "tree"
        extract.mkdir()
        with tarfile.open(archive) as tf:
            tf.extractall(extract)  # noqa: S202 - trusted, runner-created archive
        entries = snapshot_workspace(str(extract))
    return _issue(RunnerBaseline(base_commit, entries, "base-commit-tree"))


def baseline_from_workspace_creation(*, workspace_dir: str, base_commit: str = "") -> RunnerBaseline:
    """A snapshot taken by the Runner at workspace-creation time."""
    entries = snapshot_workspace(workspace_dir)
    return _issue(RunnerBaseline(base_commit, entries, "workspace-creation"))
