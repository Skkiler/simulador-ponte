from __future__ import annotations

from src.core.numeric import numeric_values, safe_float, safe_sort_key


def test_safe_float_handles_invalid_values() -> None:
    assert safe_float(None, 1.5) == 1.5
    assert safe_float("", 2.5) == 2.5
    assert safe_float("  ", 3.5) == 3.5
    assert safe_float("nan", 4.5) == 4.5
    assert safe_float("inf", 5.5) == 5.5
    assert safe_float("abc", 6.5) == 6.5
    assert safe_float("12.3", 0.0) == 12.3


def test_safe_sort_key_and_numeric_values() -> None:
    vals = [None, "3.0", "x", 1, float("nan"), 2.5]
    assert numeric_values(vals) == [3.0, 1.0, 2.5]
    assert safe_sort_key(None) > 1e90
    assert safe_sort_key("2.0") == 2.0
