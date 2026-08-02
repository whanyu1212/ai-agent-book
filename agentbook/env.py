"""Small shared readers for environment-based experiment configuration."""

from __future__ import annotations

import os
from collections.abc import Callable

__all__ = ["read_int_env"]


def read_int_env(
    name: str,
    default: int,
    *,
    on_invalid: Callable[[str, str, int], None],
) -> int:
    """Read an integer environment variable, warning and falling back if malformed."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        on_invalid(name, raw, default)
        return default
