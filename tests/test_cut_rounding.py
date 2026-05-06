from __future__ import annotations

from src.services.stick_detail_service import StickDetailService


def test_floor_to_cut_increment_5mm() -> None:
    svc = StickDetailService()
    assert svc.floor_to_cut_increment(102.0, increment_mm=5.0, min_value_mm=5.0) == 100.0
    assert svc.floor_to_cut_increment(113.0, increment_mm=5.0, min_value_mm=5.0) == 110.0
    assert svc.floor_to_cut_increment(87.9, increment_mm=5.0, min_value_mm=5.0) == 85.0
    assert svc.floor_to_cut_increment(2.0, increment_mm=5.0, min_value_mm=5.0) == 5.0

