from __future__ import annotations

from typing import Any

from src.core.numeric import safe_float


def safety_label(value: Any) -> str:
    """Human-readable label for safety factors."""
    clean = safe_float(value, None)
    if clean is None:
        return "sem solicitação"
    return f"{clean:.3f}"


def risk_from_fs(value: Any) -> str:
    """Risk classification from minimum safety factor."""
    fs = safe_float(value, None)
    if fs is None:
        return "OK"
    if fs < 1.0:
        return "CRITICAL"
    if fs < 2.0:
        return "LOW_MARGIN"
    return "OK"
