"""Shared request-parameter policy for reasoning models."""

from __future__ import annotations

__all__ = ["is_reasoning_model", "reasoning_safe_temperature"]

_REASONING_MODEL_MARKERS = (
    "gpt-5",
    "kimi-k2.5",
    "kimi-k2.6",
    "kimi-k2.7",
    "kimi-k3",
)


def is_reasoning_model(model: object) -> bool:
    """Return whether a model uses the constrained reasoning request policy."""
    normalized = str(model or "").casefold().replace("/", "-")
    return any(marker in normalized for marker in _REASONING_MODEL_MARKERS)


def reasoning_safe_temperature(model: object, requested: float = 1.0) -> float:
    """Force temperature 1 for reasoning models and preserve it otherwise."""
    return 1 if is_reasoning_model(model) else requested
