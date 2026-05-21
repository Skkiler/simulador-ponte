from __future__ import annotations

import pytest

from src.services.config_service import ConfigService


def test_config_normalize_load_distribution_and_defaults(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["bridge"]["load_distribution_x_mm"] = []
    normalized = ConfigService().normalize(cfg)

    xs = normalized["bridge"]["load_distribution_x_mm"]
    assert xs
    assert xs == sorted(set(xs))
    assert normalized["analysis"]["planner_max_sticks_per_group_by_group"]["top_chord"] == 20


def test_config_normalize_rejects_invalid_ranges(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["planner"]["span_min_mm"] = 1300.0
    cfg["planner"]["span_max_mm"] = 1200.0
    with pytest.raises(ValueError, match="planner\\.span_min_mm"):
        ConfigService().normalize(cfg)


def test_config_normalize_strength_first_defaults(base_cfg: dict) -> None:
    normalized = ConfigService().normalize(base_cfg)
    ms = normalized["member_sizing"]

    assert ms["plane_bracing_efficiency_allow_strength_loss_below_target"] is False
    assert ms["final_strength_push_min_actual_break_gain_kgf"] == 0.0


def test_no_crossing_policy_removes_x_from_secondary_planner_domains(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["detail_model"]["x_bracing_crossing_policy"] = "warren_no_crossing"
    cfg["planner"]["consider_internal_trusses"] = ["X", "Pratt_symmetric"]
    cfg["planner"]["consider_top_chord_trusses"] = ["X", "Warren_symmetric"]
    cfg["planner"]["consider_bottom_chord_trusses"] = ["X", "Howe_inverted"]

    normalized = ConfigService().normalize(cfg)

    for key in (
        "consider_internal_trusses",
        "consider_top_chord_trusses",
        "consider_bottom_chord_trusses",
    ):
        assert "X" not in normalized["planner"][key]
