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
    assert normalized["analysis"]["planner_max_sticks_per_group_by_group"]["bottom_chord"] == 2
    assert normalized["section_layout_by_group"]["bottom_chord"]["layout"] == "tee_top"


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


def test_config_normalize_closes_8020_without_forcing_odd_solid_laminates_to_hollow_boxes(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg.setdefault("multi_loadcase_screening", {})["strength_governing_cases"] = ["center", "torsion_70_30"]
    cfg.setdefault("detail_model", {})["top_chord_closed_sandwich_enabled"] = False
    cfg.setdefault("member_sticks_by_group", {})["top_chord"] = 7
    cfg.setdefault("planner", {}).setdefault("local_sizing", {}).setdefault("min_sticks_primary_member_by_group", {})["top_chord"] = 7
    cfg.setdefault("analysis", {}).setdefault("planner_min_sticks_per_group_by_group", {})["top_chord"] = 7

    normalized = ConfigService().normalize(cfg)

    assert "torsion_80_20" in normalized["multi_loadcase_screening"]["strength_governing_cases"]
    assert normalized["member_sticks_by_group"]["top_chord"] == 7
    assert normalized["planner"]["local_sizing"]["min_sticks_primary_member_by_group"]["top_chord"] == 7
    assert normalized["analysis"]["planner_min_sticks_per_group_by_group"]["top_chord"] == 7
    assert normalized["section_layout_by_group"]["top_chord"]["layout"] == "solid_face_laminated_flat"
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
    cfg["detail_model"]["top_chord_closed_sandwich_enabled"] = False
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
    top_layout = normalized["section_layout_by_group"]["top_chord"]
    vertical_layout = normalized["section_layout_by_group"]["vertical"]
    assert top_layout["layout"] == "solid_face_laminated_flat"
    assert top_layout["stick_orientation"] == "flat"
    assert top_layout["joint_quality"] == "face_laminated"
    assert vertical_layout["layout"] == "solid_face_laminated_edge"
    assert vertical_layout["stick_orientation"] == "edge"
    assert vertical_layout["joint_quality"] == "face_laminated"
    bottom_layout = normalized["section_layout_by_group"]["bottom_chord"]
    assert normalized["member_sticks_by_group"]["bottom_chord"] == 2
    assert bottom_layout["layout"] == "tee_top"
    assert bottom_layout["stick_orientation"] == "mixed"


def test_butt_splint_defaults_use_single_anchored_face_tala_not_visual_exploded(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["detail_model"].pop("reinforcement_length_mm", None)
    cfg["detail_model"].pop("reinforcement_sticks_per_splice", None)
    cfg["detail_model"].pop("node_lap_visual_side_offset_mm", None)

    normalized = ConfigService().normalize(cfg)

    assert normalized["detail_model"]["reinforcement_length_mm"] == 25.0
    assert normalized["detail_model"]["reinforcement_sticks_per_splice"] == 1
    assert normalized["detail_model"]["splints_per_splice_by_group"]["diagonal"] == 1
    assert normalized["detail_model"]["splint_length_mm_by_group"]["diagonal"] == 40.0
    assert normalized["detail_model"]["node_lap_visual_side_offset_mm"] >= 4.0
    assert normalized["detail_model"].get("side_lap_groups_skip_axis_setback") is True


def test_legacy_heavy_butt_splints_are_replaced_by_one_long_face_tala(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["detail_model"]["splice_mode"] = "butt_with_splints"
    cfg["detail_model"]["reinforcement_length_mm"] = 55.0
    cfg["detail_model"]["reinforcement_sticks_per_splice"] = 4

    normalized = ConfigService().normalize(cfg)

    assert normalized["detail_model"]["reinforcement_length_mm"] == 25.0
    assert normalized["detail_model"]["reinforcement_sticks_per_splice"] == 1
    assert normalized["detail_model"]["splints_per_splice_by_group"]["diagonal"] == 1
    assert normalized["detail_model"]["splint_length_mm_by_group"]["diagonal"] == 40.0


def test_side_lap_groups_skip_axis_setback_by_default(base_cfg: dict) -> None:
    normalized = ConfigService().normalize(base_cfg)

    detail = normalized["detail_model"]
    assert detail["side_lap_groups_skip_axis_setback"] is True
    assert "vertical" not in detail["side_lap_no_axis_setback_groups"]
    assert "diagonal" in detail["side_lap_no_axis_setback_groups"]
    assert detail["vertical_seated_on_bottom_flange_enabled"] is True


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
    assert stack["vertical_y"] == 0.0
    assert stack["diagonal_y"] == 5.75
    assert stack["top_transverse_z"] == 4.8
    assert stack["cross_frame_bracing_x"] == 0.0
    assert detail["node_lap_visual_side_offset_max_mm"] <= 24.0


def test_contact_stack_caps_face_contact_tolerance(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["detail_model"]["node_lap_physical_offset_model"] = "contact_stack_not_exploded"
    cfg["detail_model"]["as_built_face_contact_tolerance_mm"] = 4.5

    normalized = ConfigService().normalize(cfg)

    assert normalized["detail_model"]["as_built_face_contact_tolerance_mm"] == 1.6


def test_cross_frame_reaches_montante_face_without_axis_gap_by_default(base_cfg: dict) -> None:
    normalized = ConfigService().normalize(base_cfg)

    detail = normalized["detail_model"]
    assert "cross_frame_bracing" in detail["side_lap_no_axis_setback_groups"]
    assert detail["joint_min_setback_by_group"]["cross_frame_bracing"] == 0.0
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

    assert stack["vertical_y"] == 0.0
    assert stack["diagonal_y"] == 12.0
    assert stack["top_bracing_y"] == 14.0
    assert stack["bottom_bracing_y"] == 14.0
    assert stack["top_transverse_z"] == 4.8
    assert stack["bottom_transverse_z"] == -4.8
    assert stack["support_pad_z"] == -4.8
    assert stack["cross_frame_bracing_x"] == 0.0
    assert normalized["detail_model"]["node_lap_visual_side_offset_max_mm"] <= 14.0


def test_project_domain_keeps_chord_planes_transverse_only_and_internal_zigzag(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["bridge"]["include_top_x_bracing"] = True
    cfg["bridge"]["include_bottom_x_bracing"] = True
    cfg["bridge"]["cross_frame_truss_type"] = "Pratt_symmetric"

    normalized = ConfigService().normalize(cfg)

    assert normalized["bridge"]["include_top_x_bracing"] is False
    assert normalized["bridge"]["include_bottom_x_bracing"] is False
    assert normalized["bridge"]["include_cross_frame_bracing"] is True
    assert normalized["bridge"]["top_chord_truss_type"] == "none"
    assert normalized["bridge"]["bottom_chord_truss_type"] == "none"
    assert normalized["bridge"]["cross_frame_truss_type"] == "Warren_symmetric"


def test_loaded_top_stations_receive_more_transverse_sticks_without_chord_zigzag(base_cfg: dict) -> None:
    from src.services.geometry_service import GeometryService

    normalized = ConfigService().normalize(base_cfg)
    nodes, members, _, loads = GeometryService().generate(normalized)
    by_id = {n.id: n for n in nodes}

    assert not [m for m in members if m.group in {"top_bracing", "bottom_bracing"}]
    loaded_x = {
        round(by_id[l.node_id].x, 6)
        for l in loads
        if by_id[l.node_id].level == "top" and abs(l.Fz) > 1.0e-9
    }
    ties = [m for m in members if m.group == "top_transverse"]
    loaded_ties = [m for m in ties if round(by_id[m.i].x, 6) in loaded_x]
    regular_ties = [m for m in ties if round(by_id[m.i].x, 6) not in loaded_x]
    assert loaded_ties and regular_ties
    if normalized["detail_model"].get("single_piece_chord_transverse_ties_enabled", False):
        assert all(m.n_sticks == 1 for m in ties)
        assert normalized["bridge"]["width_mm"] <= normalized["detail_model"]["single_piece_chord_transverse_max_centerline_span_mm"]
    else:
        assert min(m.n_sticks for m in loaded_ties) > max(m.n_sticks for m in regular_ties)

    cross = sorted(
        [m for m in members if m.group == "cross_frame_bracing" and abs(by_id[m.i].x - by_id[m.j].x) > 1.0e-9],
        key=lambda m: min(by_id[m.i].x, by_id[m.j].x),
    )
    left_levels = []
    for m in cross:
        n1, n2 = by_id[m.i], by_id[m.j]
        left_levels.append((n1 if n1.y < n2.y else n2).level)
    assert all(a != b for a, b in zip(left_levels, left_levels[1:]))


def test_fixed_design_inputs_preserve_validated_geometry_and_member_overrides(base_cfg: dict) -> None:
    cfg = ConfigService().normalize(base_cfg)
    original_bridge = {
        key: cfg["bridge"].get(key)
        for key in (
            "panel_mm",
            "center_height_mm",
            "end_height_mm",
            "top_profile",
            "side_truss_type",
            "internal_truss_type",
            "top_chord_truss_type",
            "bottom_chord_truss_type",
        )
    }
    original_overrides = dict(cfg.get("member_sticks_by_id", {}))

    frozen = ConfigService().from_fixed_design_inputs(
        cfg,
        target_load_kgf=cfg["planner"]["target_load_kgf"],
        target_breaking_load_kgf=cfg["planner"]["target_breaking_load_kgf"],
        max_bridge_mass_g=cfg["planner"]["max_bridge_mass_g"],
        target_bridge_mass_g=cfg["planner"]["target_bridge_mass_g"],
        E_MPa=cfg["material"]["E_MPa"],
        stick_length_mm=cfg["material"]["stick_length_mm"],
        stick_width_mm=cfg["material"]["stick_width_mm"],
        stick_thickness_mm=cfg["material"]["stick_thickness_mm"],
        stick_mass_g=cfg["material"]["stick_mass_g"],
        tension_capacity_per_stick_kgf=cfg["material"]["tension_capacity_per_stick_kgf"],
        compression_capacity_one_stick_kgf=cfg["material"]["compression_capacity_one_stick_kgf"],
        compression_capacity_two_sticks_kgf=cfg["material"]["compression_capacity_two_sticks_kgf"],
        glue_shear_strength_MPa=cfg["detail_model"]["glue_shear_strength_MPa"],
        overlap_length_mm=cfg["detail_model"]["overlap_length_mm"],
        target_min_fs=cfg["analysis"]["target_min_fs"],
    )

    assert {key: frozen["bridge"].get(key) for key in original_bridge} == original_bridge
    assert frozen.get("member_sticks_by_id", {}) == original_overrides
    assert frozen["analysis"]["optimize_variants"] is False
    assert frozen["analysis"]["active_planner_enabled"] is False


def test_face_connection_policy_preserves_overlap_and_forbids_hollow_primary_sections(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["detail_model"]["splice_mode"] = "overlap"
    normalized = ConfigService().normalize(cfg)
    assert normalized["detail_model"]["splice_mode"] == "overlap"
    assert normalized["detail_model"]["forbid_plain_butt_connections"] is True
    assert normalized["section_layout_by_group"]["top_chord"]["layout"] == "closed_sandwich_4core_2caps_2covers"
    assert normalized["section_layout_by_group"]["vertical"]["layout"].startswith("solid_face_laminated")
    assert normalized["section_layout_by_group"]["bottom_chord"]["layout"] == "tee_top"


def test_bottom_tee_face_overlap_fabrication_is_locked_to_fourteen_web_sticks_per_side(base_cfg: dict) -> None:
    normalized = ConfigService().normalize(base_cfg)
    detail = normalized["detail_model"]
    assert detail["bottom_chord_tee_web_splice_mode"] == "face_overlap"
    assert detail["bottom_chord_tee_web_overlap_mm"] == 20.0
    assert detail["bottom_chord_tee_web_piece_count_per_side"] == 14


def test_closed_sandwich_upper_chord_and_built_post_layouts_are_locked(base_cfg: dict) -> None:
    from src.services.geometry_service import GeometryService

    normalized = ConfigService().normalize(base_cfg)
    assert normalized["member_sticks_by_group"]["top_chord"] == 8
    assert normalized["section_layout_by_group"]["top_chord"]["layout"] == "closed_sandwich_4core_2caps_2covers"
    assert normalized["analysis"]["clean_output_root_before_run"] is True

    nodes, members, _, _ = GeometryService().generate(normalized)
    by_id = {n.id: n for n in nodes}
    upper = [m for m in members if m.group == "top_chord"]
    assert upper and {m.n_sticks for m in upper} == {8, 10}
    assert {m.layout for m in upper} == {"closed_sandwich_4core_2caps_2covers"}
    upper_mid = {round(0.5 * (by_id[m.i].x + by_id[m.j].x), 6): m.n_sticks for m in upper if by_id[m.i].y < 0.0}
    assert set(upper_mid) == {-50.0, 60.0, 180.0, 300.0, 420.0, 540.0, 660.0, 780.0, 900.0, 1020.0, 1140.0, 1250.0}
    assert {upper_mid[x] for x in (-50.0, 60.0, 180.0, 420.0, 780.0, 1020.0, 1140.0, 1250.0)} == {8}
    assert {upper_mid[x] for x in (300.0, 540.0, 660.0, 900.0)} == {10}

    built_posts = [
        m for m in members
        if m.group == "vertical"
        and round(0.5 * (by_id[m.i].x + by_id[m.j].x), 6) in {0.0, 1200.0}
    ]
    assert len(built_posts) == 4
    assert {m.n_sticks for m in built_posts} == {6}
    assert {m.layout for m in built_posts} == {"closed_sandwich_4core_2caps"}


def test_sandwich_members_reject_odd_or_underbuilt_local_sizing(base_cfg: dict) -> None:
    from src.services.geometry_service import GeometryService

    normalized = ConfigService().normalize(base_cfg)
    # A primeira barra do banzo superior e o montante construído em x=0
    # recebem overrides impossíveis de fabricar; a geometria deve preservá-los
    # como unidades sanduíche completas e simétricas.
    nodes0, members0, _, _ = GeometryService().generate(normalized)
    nodes_by_id = {n.id: n for n in nodes0}
    top_id = next(m.id for m in members0 if m.group == "top_chord")
    post_id = next(
        m.id for m in members0
        if m.group == "vertical"
        and abs(0.5 * (nodes_by_id[m.i].x + nodes_by_id[m.j].x)) <= 1.0e-9
    )
    normalized["member_sticks_by_id"] = {str(top_id): 7, str(post_id): 5}
    _, members, _, _ = GeometryService().generate(normalized)
    by_id = {m.id: m for m in members}
    assert by_id[top_id].layout == "closed_sandwich_4core_2caps_2covers"
    assert by_id[top_id].n_sticks == 8
    assert by_id[post_id].layout == "closed_sandwich_4core_2caps"
    assert by_id[post_id].n_sticks == 6


def test_single_piece_transverse_tie_control_reduces_infeasible_width(base_cfg: dict) -> None:
    base_cfg["bridge"]["width_mm"] = 150.0
    base_cfg["detail_model"]["single_piece_chord_transverse_ties_enabled"] = True
    base_cfg["detail_model"]["single_piece_chord_transverse_max_centerline_span_mm"] = 115.0
    normalized = ConfigService().normalize(base_cfg)
    assert normalized["bridge"]["width_mm"] == 115.0
    assert normalized["detail_model"]["loaded_top_transverse_reinforcement_enabled"] is False


def test_top_chord_seated_geometry_raises_top_nodes(base_cfg: dict) -> None:
    from src.services.geometry_service import GeometryService
    base_cfg["detail_model"]["top_chord_seated_above_vertical_enabled"] = True
    base_cfg["detail_model"]["top_chord_centroid_seat_raise_mm"] = 6.5
    normalized = ConfigService().normalize(base_cfg)
    svc = GeometryService()
    assert svc.fabricated_top_node_height(normalized, 600.0) == svc.top_height(normalized, 600.0) + 6.5


def test_face_connection_policy_accepts_one_long_diagonal_splint(base_cfg: dict) -> None:
    base_cfg["detail_model"]["physical_connection_policy_enabled"] = True
    base_cfg["detail_model"]["minimum_face_splints_per_butt_joint"] = 1
    base_cfg["detail_model"]["splints_per_splice_by_group"] = {"diagonal": 1}
    base_cfg["detail_model"]["splint_length_mm_by_group"] = {"diagonal": 40.0}
    normalized = ConfigService().normalize(base_cfg)
    assert normalized["detail_model"]["minimum_face_splints_per_butt_joint"] == 1
    assert normalized["detail_model"]["splints_per_splice_by_group"]["diagonal"] == 1
    assert normalized["detail_model"]["splint_length_mm_by_group"]["diagonal"] == 40.0
