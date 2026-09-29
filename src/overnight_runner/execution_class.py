"""M5/H1 proposal (c) + (d) — execution-class admission.

ADDITIVE new module. `source-only` and `host-tests` are ENABLED; the
native/render/automation classes stay blocked-with-reason (no Unreal on this
host); `human-review` requires a human gate. `(d)` is wired here for real: a
blocked class carries the normalized `RunnerErrorClass` that names why, and a
runtime failure is recorded with the capacity-state transition.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .execution_state import (
    ProfileRuntimeState,
    RunnerErrorClass,
    apply_runtime_outcome,
    fallback_disposition,
)

__all__ = [
    "ExecutionClass",
    "ExecutionClassDecision",
    "admit_execution_class",
    "record_runtime_failure",
]


class ExecutionClass(str, Enum):
    SOURCE_ONLY = "source-only"
    HOST_TESTS = "host-tests"
    NATIVE_COMPILE = "native-compile"
    RENDERED_EDITOR = "rendered-editor"
    STANDALONE_AUTOMATION = "standalone-automation"
    HUMAN_REVIEW = "human-review"


_ENABLED = {ExecutionClass.SOURCE_ONLY, ExecutionClass.HOST_TESTS}
_BLOCKED = {
    ExecutionClass.NATIVE_COMPILE: "requires an Unreal toolchain that this host stage does not provide",
    ExecutionClass.RENDERED_EDITOR: "requires an editor/GPU slot that this host stage does not provide",
    ExecutionClass.STANDALONE_AUTOMATION: "requires a standalone packaged build that this host stage does not provide",
}


@dataclass(frozen=True)
class ExecutionClassDecision:
    ok: bool
    execution_class: ExecutionClass
    code: str = ""
    detail: str = ""
    needs_human_gate: bool = False


def admit_execution_class(execution_class: ExecutionClass) -> ExecutionClassDecision:
    """Allow `source-only`/`host-tests`; block the rest with a specific reason."""
    if execution_class in _ENABLED:
        return ExecutionClassDecision(True, execution_class)
    if execution_class == ExecutionClass.HUMAN_REVIEW:
        return ExecutionClassDecision(
            False, execution_class, "NEEDS_HUMAN_REVIEW",
            "human-review requires a human gate disposition before it can run",
            True,
        )
    return ExecutionClassDecision(
        False, execution_class, "UNSUPPORTED_ON_THIS_HOST_STAGE", _BLOCKED[execution_class],
    )


def record_runtime_failure(
    state: ProfileRuntimeState,
    error_class: RunnerErrorClass,
    now: int,
    *,
    reset_at: Optional[int] = None,
) -> tuple[ProfileRuntimeState, str]:
    """Apply a failure and return (new state, what to do next).

    Uses the (d) vocabulary and transitions: an availability failure enters
    CAPACITY_WAIT with a cooldown; an EFFECT_UNKNOWN blocks fallback entirely.
    """
    updated = apply_runtime_outcome(state, error_class, now, reset_at=reset_at)
    disposition = fallback_disposition(error_class)
    if disposition == "availability":
        return updated, "failover"
    if disposition == "unknown-effect":
        return updated, "stop"
    return updated, "stop"
