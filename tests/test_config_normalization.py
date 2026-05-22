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


def test_config_normalize_closes_8020_and_even_primary_boxes(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg.setdefault("multi_loadcase_screening", {})["strength_governing_cases"] = ["center", "torsion_70_30"]
    cfg.setdefault("member_sticks_by_group", {})["top_chord"] = 7
    cfg.setdefault("planner", {}).setdefault("local_sizing", {}).setdefault("min_sticks_primary_member_by_group", {})["top_chord"] = 7
    cfg.setdefault("analysis", {}).setdefault("planner_min_sticks_per_group_by_group", {})["top_chord"] = 7

    normalized = ConfigService().normalize(cfg)

    assert "torsion_80_20" in normalized["multi_loadcase_screening"]["strength_governing_cases"]
    assert normalized["member_sticks_by_group"]["top_chord"] == 8
    assert normalized["planner"]["local_sizing"]["min_sticks_primary_member_by_group"]["top_chord"] == 8
    assert normalized["analysis"]["planner_min_sticks_per_group_by_group"]["top_chord"] == 8
    assert normalized["analysis"]["use_quarter_model"] is False


def test_top_profile_domain_is_strict_when_user_selects_flat_only(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["bridge"]["top_profile"] = "parker_plateau"
    cfg["bridge"]["center_height_mm"] = 300.0
    cfg["bridge"]["end_height_mm"] = 100.0
    cfg["planner"]["consider_top_profiles"] = ["flat"]

    normalized = ConfigService().normalize(cfg)

    assert normalized["planner"]["consider_top_profiles"] == ["flat"]
    assert normalized["bridge"]["top_profile"] == "flat"
    assert normalized["bridge"]["end_height_mm"] == normalized["bridge"]["center_height_mm"]


def test_safe_default_detailing_avoids_unverified_miter_and_visual_offsets(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg.get("detail_model", {}).pop("angled_end_cuts_enabled", None)
    cfg.get("detail_model", {}).pop("visual_beveled_end_cuts", None)
    cfg.get("detail_model", {}).pop("piece_view_mounted_connection_offset_scale", None)

    normalized = ConfigService().normalize(cfg)

    assert normalized["detail_model"]["angled_end_cuts_enabled"] is False
    assert normalized["detail_model"]["visual_beveled_end_cuts"] is False
    assert normalized["detail_model"]["piece_view_mounted_connection_offset_scale"] == 0.0


def test_nonflat_profile_normalization_creates_real_height_difference(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["planner"]["consider_top_profiles"] = ["parker_plateau"]
    cfg["bridge"]["top_profile"] = "flat"
    cfg["bridge"]["center_height_mm"] = 300.0
    cfg["bridge"]["end_height_mm"] = 300.0

    normalized = ConfigService().normalize(cfg)

    assert normalized["bridge"]["top_profile"] == "parker_plateau"
    assert normalized["bridge"]["end_height_mm"] < normalized["bridge"]["center_height_mm"]
    assert normalized["bridge"]["end_height_mm"] == 105.0


def test_simple_buildable_mode_defaults_to_butt_splints_not_same_axis_overlap(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["planner"]["design_mode"] = "simple_buildable_bridge"
    cfg["detail_model"].pop("splice_mode", None)

    normalized = ConfigService().normalize(cfg)

    assert normalized["detail_model"]["splice_mode"] == "butt_with_splints"
    assert normalized["detail_model"]["angled_end_cuts_enabled"] is False
