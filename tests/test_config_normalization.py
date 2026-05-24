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

    assert normalized["detail_model"]["angled_end_cuts_enabled"] is True
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
    assert normalized["detail_model"]["angled_end_cuts_enabled"] is True
    assert normalized["detail_model"].get("node_lap_physical_offset_enabled") is True
    assert float(normalized["detail_model"].get("node_lap_visual_side_offset_mm", 0.0)) <= 8.0
    assert "top_transverse" in normalized["detail_model"].get("node_lap_visual_side_offset_groups", [])


def test_legacy_simple_layouts_are_repaired_to_compact_buildable_sections(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["member_sticks_by_group"]["top_chord"] = 6
    cfg["member_sticks_by_group"]["bottom_chord"] = 3
    cfg["member_sticks_by_group"]["vertical"] = 3
    cfg["section_layout_by_group"]["top_chord"] = {"layout": "stacked_edge"}
    cfg["section_layout_by_group"]["bottom_chord"] = {"layout": "stacked_edge"}
    cfg["section_layout_by_group"]["vertical"] = {"layout": "side_by_side"}
    cfg["detail_model"]["node_lap_visual_side_offset_mm"] = 25.0
    cfg["detail_model"]["node_lap_visual_side_offset_max_mm"] = 6.0

    normalized = ConfigService().normalize(cfg)

    assert normalized["analysis"]["target_min_fs"] >= 1.5
    assert normalized["detail_model"]["node_lap_visual_side_offset_mm"] == 6.0
    for group in ("top_chord", "bottom_chord", "vertical"):
        layout = normalized["section_layout_by_group"][group]
        assert layout["layout"] == "contact_box"
        assert layout["stick_orientation"] == "edge"
        assert layout["box_extra_stick_strategy"] == "balanced"


def test_butt_splint_defaults_are_mass_conservative_not_visual_exploded(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["detail_model"].pop("reinforcement_length_mm", None)
    cfg["detail_model"].pop("reinforcement_sticks_per_splice", None)
    cfg["detail_model"].pop("node_lap_visual_side_offset_mm", None)

    normalized = ConfigService().normalize(cfg)

    assert normalized["detail_model"]["reinforcement_length_mm"] == 25.0
    assert normalized["detail_model"]["reinforcement_sticks_per_splice"] == 1
    assert normalized["detail_model"]["node_lap_visual_side_offset_mm"] >= 4.0
    assert normalized["detail_model"].get("side_lap_groups_skip_axis_setback") is True


def test_legacy_heavy_butt_splints_are_capped_for_mass(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["detail_model"]["splice_mode"] = "butt_with_splints"
    cfg["detail_model"]["reinforcement_length_mm"] = 55.0
    cfg["detail_model"]["reinforcement_sticks_per_splice"] = 4

    normalized = ConfigService().normalize(cfg)

    assert normalized["detail_model"]["reinforcement_length_mm"] == 25.0
    assert normalized["detail_model"]["reinforcement_sticks_per_splice"] == 1


def test_side_lap_groups_skip_axis_setback_by_default(base_cfg: dict) -> None:
    normalized = ConfigService().normalize(base_cfg)

    detail = normalized["detail_model"]
    assert detail["side_lap_groups_skip_axis_setback"] is True
    assert "vertical" in detail["side_lap_no_axis_setback_groups"]
    assert "diagonal" in detail["side_lap_no_axis_setback_groups"]


def test_min_constructive_piece_length_accepts_20mm(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["detail_model"].pop("min_constructive_piece_length_mm", None)

    normalized = ConfigService().normalize(cfg)

    assert normalized["detail_model"]["min_constructive_piece_length_mm"] == 20.0


def test_explicit_legacy_40mm_piece_minimum_is_not_forced(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["detail_model"]["min_constructive_piece_length_mm"] = 20.0

    normalized = ConfigService().normalize(cfg)

    assert normalized["detail_model"]["min_constructive_piece_length_mm"] == 20.0


def test_explicit_legacy_40mm_piece_minimum_is_capped_to_20mm(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["detail_model"]["min_constructive_piece_length_mm"] = 40.0

    normalized = ConfigService().normalize(cfg)

    assert normalized["detail_model"]["min_constructive_piece_length_mm"] == 20.0


def test_as_built_face_lap_tolerance_defaults_to_mountable_contact(base_cfg: dict) -> None:
    normalized = ConfigService().normalize(base_cfg)

    detail = normalized["detail_model"]
    assert detail["as_built_ignore_face_lap_tolerance"] is True
    assert 0.0 < detail["as_built_face_contact_tolerance_mm"] <= 1.6
    assert detail["auto_mountable_layer_offsets"] is True
    assert detail["node_lap_visual_side_offset_mm"] >= 4.0
    assert detail["node_lap_visual_side_offset_max_mm"] <= 24.0


def test_contact_stack_offsets_are_not_exploded_by_default(base_cfg: dict) -> None:
    normalized = ConfigService().normalize(base_cfg)

    detail = normalized["detail_model"]
    assert detail["node_lap_physical_offset_model"] == "contact_stack_not_exploded"
    stack = detail["contact_stack_offsets_mm"]
    assert stack["vertical_y"] == 10.0
    assert stack["diagonal_y"] == 12.0
    assert stack["top_transverse_z"] == 4.8
    assert stack["cross_frame_bracing_x"] == 0.0
    assert detail["node_lap_visual_side_offset_max_mm"] <= 24.0


def test_contact_stack_caps_face_contact_tolerance(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["detail_model"]["node_lap_physical_offset_model"] = "contact_stack_not_exploded"
    cfg["detail_model"]["as_built_face_contact_tolerance_mm"] = 4.5

    normalized = ConfigService().normalize(cfg)

    assert normalized["detail_model"]["as_built_face_contact_tolerance_mm"] == 1.6


def test_cross_frame_uses_trim_not_fake_offset_by_default(base_cfg: dict) -> None:
    normalized = ConfigService().normalize(base_cfg)

    detail = normalized["detail_model"]
    assert "cross_frame_bracing" not in detail["side_lap_no_axis_setback_groups"]
    assert detail["joint_min_setback_by_group"]["cross_frame_bracing"] >= 8.0
    assert detail["contact_stack_offsets_mm"]["cross_frame_bracing_x"] == 0.0


def test_contact_stack_offsets_are_mounting_not_exploded(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["detail_model"]["node_lap_physical_offset_model"] = "contact_stack_not_exploded"
    cfg["detail_model"]["contact_stack_offsets_mm"] = {
        "vertical_y": 11.5,
        "diagonal_y": 21.5,
        "top_bracing_y": 25.5,
        "bottom_bracing_y": 25.5,
        "top_transverse_z": 6.0,
        "bottom_transverse_z": -6.0,
        "support_pad_z": -5.0,
        "cross_frame_bracing_x": 10.0,
    }

    normalized = ConfigService().normalize(cfg)
    stack = normalized["detail_model"]["contact_stack_offsets_mm"]

    assert stack["vertical_y"] == 10.0
    assert stack["diagonal_y"] == 12.0
    assert stack["top_bracing_y"] == 14.0
    assert stack["bottom_bracing_y"] == 14.0
    assert stack["top_transverse_z"] == 4.8
    assert stack["bottom_transverse_z"] == -4.8
    assert stack["support_pad_z"] == -4.8
    assert stack["cross_frame_bracing_x"] == 0.0
    assert normalized["detail_model"]["node_lap_visual_side_offset_max_mm"] <= 14.0
