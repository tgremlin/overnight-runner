"""§ACT1-0 — an unset or REAL `OVERNIGHT_STATE_DIR` is a hard refusal.

EXEC4 briefly wrote stray directories into the operator's REAL state directory
because `OVERNIGHT_STATE_DIR` was unset and `runtime.state_dir()` silently fell
back to `~/.local/state/overnight-runner` (the target of `~/.trio/runner-state`).
The fallback is what made that breach possible, so it is gone.

Every Runner-side entry point that needs a state directory (or a Runner-owned
scratch root) now REFUSES with the typed code `STATE_DIR_UNSET_OR_REAL` unless
`OVERNIGHT_STATE_DIR` is set to a directory that is neither empty/unset nor the
operator's real state directory (or its target).

The only escape is the documented activation flag
`TRIO_STATE_DIR_IS_ACTIVATED_REAL_RUN=1`, which states that the run IS the
operator's activated real run. This campaign never sets it; activation stays the
operator's explicit command.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "ACTIVATION_FLAG",
    "STATE_DIR_ENV",
    "STATE_DIR_UNSET_OR_REAL",
    "StateDirRefusal",
    "assert_usable_state_dir",
    "real_state_paths",
    "state_dir_is_usable",
]

STATE_DIR_ENV = "OVERNIGHT_STATE_DIR"
ACTIVATION_FLAG = "TRIO_STATE_DIR_IS_ACTIVATED_REAL_RUN"
STATE_DIR_UNSET_OR_REAL = "STATE_DIR_UNSET_OR_REAL"


class StateDirRefusal(RuntimeError):
    """Typed refusal: the state directory is unset or is the real one."""

    code = STATE_DIR_UNSET_OR_REAL

    def __init__(self, detail: str) -> None:
        super().__init__(f"{STATE_DIR_UNSET_OR_REAL}: {detail}")
        self.detail = detail


def real_state_paths(home: str | os.PathLike[str] | None = None) -> tuple[Path, ...]:
    """The operator's real state directory AND its target, both resolved.

    `~/.trio/runner-state` is a symlink to `~/.local/state/overnight-runner`, so
    the two spellings resolve to the same place; a caller may not use either.
    """
    base = Path(home) if home is not None else Path(os.path.expanduser("~"))
    return (
        (base / ".trio" / "runner-state").resolve(),
        (base / ".local" / "state" / "overnight-runner").resolve(),
    )


def _activated() -> bool:
    return os.environ.get(ACTIVATION_FLAG, "") == "1"


def assert_usable_state_dir(raw: str | None = None, home: str | os.PathLike[str] | None = None) -> Path:
    """Return the usable state directory, or raise `StateDirRefusal`.

    `raw` defaults to the live `OVERNIGHT_STATE_DIR`, read at call time so tests and
    callers can monkeypatch it.
    """
    if raw is None:
        raw = os.environ.get(STATE_DIR_ENV)
    activated = _activated()
    if raw is None or raw.strip() == "":
        if activated:
            # Only the operator's explicit activated real run may use the real dir.
            return real_state_paths(home)[1]
        raise StateDirRefusal(
            f"{STATE_DIR_ENV} is unset; refusing to fall back to the operator's real state dir"
        )
    candidate = Path(raw).expanduser()
    if not activated and candidate.resolve() in real_state_paths(home):
        raise StateDirRefusal(
            f"{STATE_DIR_ENV} resolves to the operator's real state dir ({candidate.resolve()})"
        )
    return candidate


def state_dir_is_usable(raw: str | None = None, home: str | os.PathLike[str] | None = None) -> bool:
    """Boolean form, for callers that must not raise."""
    try:
        assert_usable_state_dir(raw, home)
        return True
    except StateDirRefusal:
        return False
