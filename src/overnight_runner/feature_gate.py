"""P06 — External feature/profile gate.

campaign-v2 is opt-in via an externally visible gate. Default is
disabled. The runtime consults ``feature_gate.campaign_v2_enabled()``
and raises ``SafetyError`` if campaign-v2 activity is attempted while
the gate is closed.

Operators may enable campaign-v2 by:

  1. Setting the ``TR_P06_CAMPAIGN_V2=1`` environment variable, OR
  2. Setting ``OVERNIGHT_CAMPAIGN_PROFILE=internal-test`` env var,
     OR
  3. Writing a configuration file at
     ``<state_dir>/config/campaign_v2.enabled`` containing the
     exact string ``"1"``.

The gate inspects ALL three sources and is fail-closed: ANY of them
must explicitly enable campaign-v2 for it to be enabled. Absence of
all three signals disables campaign-v2.

The gate is consulted at:
  - ``derive_admission`` (P06-A01/A04)
  - ``create_campaign`` (P06 lifecycle)
  - ``compare_and_swap_advance`` (P06-A04)

V1 code does NOT consult this gate. V1 paths are unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

from .safety import SafetyError


def _read_state_dir_config() -> bool:
    try:
        from .runtime import state_dir
        sd = state_dir()
        sentinel = sd / "config" / "campaign_v2.enabled"
        if sentinel.exists() and sentinel.read_text().strip() == "1":
            return True
    except Exception:
        return False
    return False


def campaign_v2_enabled() -> bool:
    """Return True iff an explicit enable signal is present.

    ANY of the following enables campaign-v2:
      - TR_P06_CAMPAIGN_V2 env var set to a truthy value ("1", "true",
        "yes", "on")
      - OVERNIGHT_CAMPAIGN_PROFILE env var set to a recognised profile
      - <state_dir>/config/campaign_v2.enabled contains "1"

    Fail-closed default is False.
    """
    if os.environ.get("TR_P06_CAMPAIGN_V2", "").lower() in ("1", "true", "yes", "on"):
        return True
    profile = os.environ.get("OVERNIGHT_CAMPAIGN_PROFILE", "")
    if profile in ("internal-test", "campaign-v2-test"):
        return True
    if _read_state_dir_config():
        return True
    return False


def require_campaign_v2(action: str) -> None:
    """Raise ``SafetyError`` if campaign-v2 is not enabled for ``action``.

    Use this at the top of every campaign-v2 entry point (admission,
    campaign creation, integration CAS). The error message names the
    action so operators know which surface rejected them.
    """
    if not campaign_v2_enabled():
        raise SafetyError(
            f"campaign-v2 is DISABLED (default). Set TR_P06_CAMPAIGN_V2=1 "
            f"or OVERNIGHT_CAMPAIGN_PROFILE=internal-test or write "
            f"<state_dir>/config/campaign_v2.enabled with '1' to perform "
            f"the action: {action!r}"
        )


__all__ = ["campaign_v2_enabled", "require_campaign_v2"]
