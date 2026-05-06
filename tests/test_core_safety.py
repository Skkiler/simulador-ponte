from __future__ import annotations

from src.core.safety import risk_from_fs, safety_label


def test_safety_label_and_risk_from_fs() -> None:
    assert safety_label(None) == "sem solicitação"
    assert safety_label(1.23456) == "1.235"

    assert risk_from_fs(None) == "OK"
    assert risk_from_fs(0.99) == "CRITICAL"
    assert risk_from_fs(1.50) == "LOW_MARGIN"
    assert risk_from_fs(2.00) == "OK"
