"""OV5 — canonical digests shared by the intake and the Runner's own evidence."""
from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["digest_of", "usable_digest"]


def digest_of(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def usable_digest(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != "" and value not in ("None", "null")
