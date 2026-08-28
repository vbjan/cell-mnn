"""
One-line helpers for stating invariants on data the user brought in.
Both build their message eagerly, which is free at load time and wasteful in a
training loop. Use them for user data, not for hot paths.
"""

from collections.abc import Collection
from typing import Any


def require(condition: object, message: str) -> None:
    """
    Raise `ValueError(message)` unless `condition` is truthy.
    """
    if not condition:
        raise ValueError(message)


def require_key(key: str, options: Collection[Any], where: str) -> None:
    """
    `key` must be present in `options`; otherwise say what was available.
    """
    if key not in options:
        raise KeyError(f"{where} has no {key!r}; available: {sorted(options)}")
