"""M5/H1 proposal — normalized error classes + profile capacity state.

ADDITIVE and VERSIONED. New module (no existing module is edited), so the frozen
Runner's behaviour and tests are unchanged. It gives the execution layer the
vocabulary the compiler/execution packages already use:

  * a normalized error-class enum (16 classes);
  * a separate mutable ``ProfileRuntimeState`` with capacity-wait/cooldown/probe;
  * the fallback disposition rule (availability | stop | unknown-effect), with
    ``EFFECT_UNKNOWN`` BLOCKING fallback;
  * reset-time parsing from the documented harness text (never guessed).

The immutable qualification identity is NOT modelled here; only the mutable
runtime state, so nothing can store runtime state on an identity.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Iterable, Optional

from pydantic import Field

from .campaign_schemas import StrictBase

EXECUTION_STATE_SCHEMA_VERSION = "trio.execution-state.v1"

DEFAULT_COOLDOWN_MS = 10 * 60_000


class RunnerErrorClass(str, Enum):
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    TEMPORARY_OVERLOAD = "TEMPORARY_OVERLOAD"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    CONNECTIVITY_FAILURE = "CONNECTIVITY_FAILURE"
    AUTH_FAILED = "AUTH_FAILED"
    PAYMENT_REQUIRED = "PAYMENT_REQUIRED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    INVALID_REQUEST = "INVALID_REQUEST"
    MAX_TURNS = "MAX_TURNS"
    NO_RESPONSE = "NO_RESPONSE"
    HARNESS_FAILURE = "HARNESS_FAILURE"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    EFFECT_UNKNOWN = "EFFECT_UNKNOWN"


class ProfileCapacityState(str, Enum):
    READY = "READY"
    CAPACITY_WAIT = "CAPACITY_WAIT"


class ProfileHealth(str, Enum):
    HEALTHY = "healthy"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class ProfileRuntimeState(StrictBase):
    """MUTABLE per-profile runtime state (never part of a qualification identity)."""

    schema_version: str = Field(default=EXECUTION_STATE_SCHEMA_VERSION)
    profile_id: str = Field(min_length=1, max_length=128)
    health: ProfileHealth = ProfileHealth.UNKNOWN
    capacity_state: ProfileCapacityState = ProfileCapacityState.READY
    cooldown_until: Optional[int] = None
    last_error_class: Optional[RunnerErrorClass] = None
    last_probe_at: Optional[int] = None
    remaining_quota: Optional[int] = None
    reset_time: Optional[int] = None
    last_success_at: Optional[int] = None
    consecutive_failures: int = Field(default=0, ge=0)
    probe_required: bool = False


# --------------------------------------------------------------------------- //
# Fallback disposition (decision D2): only provider/runtime AVAILABILITY falls back
# --------------------------------------------------------------------------- //
_AVAILABILITY = {
    RunnerErrorClass.RATE_LIMITED,
    RunnerErrorClass.QUOTA_EXHAUSTED,
    RunnerErrorClass.TEMPORARY_OVERLOAD,
    RunnerErrorClass.PROVIDER_UNAVAILABLE,
    RunnerErrorClass.CONNECTIVITY_FAILURE,
    RunnerErrorClass.AUTH_FAILED,
    RunnerErrorClass.PAYMENT_REQUIRED,
    RunnerErrorClass.UNSUPPORTED_MODEL,
}


def fallback_disposition(error_class: RunnerErrorClass) -> str:
    """Return ``availability`` | ``stop`` | ``unknown-effect``."""
    if error_class == RunnerErrorClass.EFFECT_UNKNOWN:
        return "unknown-effect"
    if error_class in _AVAILABILITY:
        return "availability"
    return "stop"


_RESET_IN = re.compile(r"reset(?:s|ting)?\s+(?:at|in)\s+(\d+)m", re.IGNORECASE)


def resolve_reset_time(text: Optional[str], now: int) -> Optional[int]:
    """Parse a documented reset time from harness text; never guess.

    Only the documented ``resets in <N>m`` form is understood. Anything else is
    ``None`` (the caller then uses the bounded default backoff).
    """
    if not text:
        return None
    m = _RESET_IN.search(text)
    if m is None:
        return None
    return now + int(m.group(1)) * 60_000


def apply_runtime_outcome(
    state: ProfileRuntimeState,
    error_class: Optional[RunnerErrorClass],
    now: int,
    *,
    reset_at: Optional[int] = None,
    default_cooldown_ms: int = DEFAULT_COOLDOWN_MS,
) -> ProfileRuntimeState:
    """Pure state transition. Success resets; availability marks CAPACITY_WAIT."""
    if error_class is None:
        return state.model_copy(
            update={
                "health": ProfileHealth.HEALTHY,
                "capacity_state": ProfileCapacityState.READY,
                "cooldown_until": None,
                "last_error_class": None,
                "last_success_at": now,
                "consecutive_failures": 0,
                "probe_required": False,
            }
        )
    disposition = fallback_disposition(error_class)
    failures = state.consecutive_failures + 1
    if disposition == "availability":
        return state.model_copy(
            update={
                "health": ProfileHealth.UNAVAILABLE,
                "capacity_state": ProfileCapacityState.CAPACITY_WAIT,
                "cooldown_until": reset_at if reset_at is not None else now + default_cooldown_ms,
                "reset_time": reset_at,
                "last_error_class": error_class,
                "consecutive_failures": failures,
                "probe_required": True,
            }
        )
    if disposition == "unknown-effect":
        return state.model_copy(
            update={
                "health": ProfileHealth.UNAVAILABLE,
                "capacity_state": ProfileCapacityState.CAPACITY_WAIT,
                "cooldown_until": None,
                "last_error_class": error_class,
                "consecutive_failures": failures,
                "probe_required": True,
            }
        )
    # 'stop' classes do not mark the profile unavailable
    return state.model_copy(update={"last_error_class": error_class, "consecutive_failures": failures})


def probe_candidates(
    states: Iterable[ProfileRuntimeState], order: list[str], now: int
) -> list[str]:
    """Profiles whose cooldown expired and which still require a probe."""
    by_id = {s.profile_id: s for s in states}
    out: list[str] = []
    for pid in order:
        s = by_id.get(pid)
        if s is None:
            continue
        if (
            s.capacity_state == ProfileCapacityState.CAPACITY_WAIT
            and s.probe_required
            and s.cooldown_until is not None
            and now >= s.cooldown_until
        ):
            out.append(pid)
    return out


def unknown_effect_action() -> str:
    """``EFFECT_UNKNOWN`` blocks fallback: the only action is to stop/reconcile."""
    return "stop"
