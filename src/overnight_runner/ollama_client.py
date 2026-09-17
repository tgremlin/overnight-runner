"""Ollama HTTP client.

Direct /api/chat. No frameworks. Bound num_ctx, num_predict, temperature, seed.
Records timing/token metrics when provided by Ollama.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .schemas import ModelProfile


DEFAULT_HOST = "http://127.0.0.1:11434"


@dataclass
class OllamaMetrics:
    prompt_eval_count: int = 0
    eval_count: int = 0
    total_duration_ns: int = 0
    eval_duration_ns: int = 0
    prompt_eval_duration_ns: int = 0
    load_duration_ns: int = 0


@dataclass
class ChatResult:
    content: str
    tool_calls: list[dict[str, Any]]
    metrics: OllamaMetrics
    raw: dict[str, Any]


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str = DEFAULT_HOST, timeout_seconds: float = 60.0) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout_seconds

    def chat(
        self,
        profile: ModelProfile,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        """One /api/chat call. Stream=False to keep logic simple."""
        body: dict[str, Any] = {
            "model": profile.model_name,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": profile.num_ctx,
                "num_predict": profile.num_predict,
                "temperature": profile.temperature,
            },
        }
        if profile.seed is not None:
            body["options"]["seed"] = profile.seed
        if tools:
            body["tools"] = tools

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise OllamaError(f"Ollama request failed: {e}") from e
        except json.JSONDecodeError as e:
            raise OllamaError(f"Ollama returned non-JSON: {e}") from e

        msg = raw.get("message") or {}
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        m = OllamaMetrics()
        if "prompt_eval_count" in raw:
            m.prompt_eval_count = int(raw.get("prompt_eval_count", 0))
        if "eval_count" in raw:
            m.eval_count = int(raw.get("eval_count", 0))
        m.total_duration_ns = int(raw.get("total_duration", 0))
        m.eval_duration_ns = int(raw.get("eval_duration", 0))
        m.prompt_eval_duration_ns = int(raw.get("prompt_eval_duration", 0))
        m.load_duration_ns = int(raw.get("load_duration", 0))
        return ChatResult(content=content, tool_calls=tool_calls, metrics=m, raw=raw)

    def model_digest(self, model_name: str) -> str | None:
        """Return the digest of an installed model via /api/tags.

        /api/show does not expose digest at the top level; the digest is
        present per-model in the /api/tags listing.
        """
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for m in data.get("models", []):
                if m.get("name") == model_name:
                    return m.get("digest")
            return None
        except Exception:
            return None


def model_supports_tools(model_name: str, host: str = DEFAULT_HOST) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for m in data.get("models", []):
            if m.get("name") == model_name:
                caps = m.get("capabilities") or []
                return "tools" in caps
    except Exception:
        return False
    return False
