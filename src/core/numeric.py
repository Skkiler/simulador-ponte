from __future__ import annotations

import math
from typing import Any, Iterable


def safe_float(value: Any, default: float | None = None) -> float | None:
    """Convert arbitrary values to finite float, else return default."""
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int | None = None) -> int | None:
    """Convert arbitrary values to int, else return default."""
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_sort_key(value: Any, default: float = 1.0e99) -> float:
    """Sort key that sends invalid values to the end in ascending sorts."""
    clean = safe_float(value, None)
    return default if clean is None else clean


def numeric_values(values: Iterable[Any]) -> list[float]:
    """Filter numeric finite values from an iterable."""
    out: list[float] = []
    for value in values:
        clean = safe_float(value, None)
        if clean is not None:
            out.append(clean)
    return out
