from __future__ import annotations

from src.services.stick_detail_service import StickDetailService


def test_floor_to_cut_increment_5mm() -> None:
    svc = StickDetailService()
    assert svc.floor_to_cut_increment(102.0, increment_mm=5.0, min_value_mm=5.0) == 100.0
    assert svc.floor_to_cut_increment(113.0, increment_mm=5.0, min_value_mm=5.0) == 110.0
    assert svc.floor_to_cut_increment(87.9, increment_mm=5.0, min_value_mm=5.0) == 85.0
    assert svc.floor_to_cut_increment(2.0, increment_mm=5.0, min_value_mm=5.0) == 5.0



def test_ceil_to_cut_increment_never_under_reports_piece_length() -> None:
    svc = StickDetailService()
    assert svc.ceil_to_cut_increment(102.0, increment_mm=5.0, min_value_mm=5.0, max_value_mm=120.0) == 105.0
    assert svc.ceil_to_cut_increment(113.0, increment_mm=5.0, min_value_mm=5.0, max_value_mm=120.0) == 115.0
    assert svc.ceil_to_cut_increment(119.2, increment_mm=5.0, min_value_mm=5.0, max_value_mm=120.0) == 120.0
    assert svc.ceil_to_cut_increment(2.0, increment_mm=5.0, min_value_mm=5.0, max_value_mm=120.0) == 5.0


def test_piece_intervals_avoid_short_terminal_fragments() -> None:
    intervals = StickDetailService._piece_intervals(
        125.0,
        stick_len=120.0,
        overlap=30.0,
        min_constructive_piece_length_mm=40.0,
    )

    assert intervals[-1] == (85.0, 125.0, 40.0)
    assert min(b - a for a, b, _ in intervals) >= 40.0

from src.domain.models import Member, Node


def _dummy_member(mid: int, i: int, j: int, group: str, n_sticks: int, length: float) -> Member:
    return Member(
        id=mid,
        i=i,
        j=j,
        group=group,
        n_sticks=n_sticks,
        A=10.5 * n_sticks,
        Asy=10.5 * n_sticks,
        Asz=10.5 * n_sticks,
        Iy=1.0,
        Iz=1.0,
        J=1.0,
        E=6000.0,
        G=500.0,
        Ky=1.0,
        Kz=1.0,
        L=length,
    )


def test_miter_cuts_are_terminal_only_not_internal_splices(tmp_path) -> None:
    nodes = [
        Node(1, 0.0, -75.0, 0.0, "bottom", "L", 0.0),
        Node(2, 0.0, -75.0, 200.0, "top", "L", 0.0),
        Node(3, 100.0, -75.0, 245.0, "top", "L", 100.0),
    ]
    members = [
        _dummy_member(1, 1, 2, "vertical", 1, 200.0),
        _dummy_member(2, 2, 3, "top_chord", 2, 109.7),
    ]
    cfg = {
        "material": {
            "stick_length_mm": 120.0,
            "stick_width_mm": 7.0,
            "stick_thickness_mm": 1.5,
            "stick_mass_g": 1.4,
            "mass_limit_g": 1000.0,
        },
        "detail_model": {
            "enabled": True,
            "overlap_length_mm": 30.0,
            "strict_cut_length": True,
            "cut_increment_mm": 5.0,
            "node_face_lap_enabled": True,
            "joint_face_setback_enabled": False,
            "angled_end_cuts_enabled": True,
            "miter_cut_min_host_slope_deg": 7.5,
            "min_constructive_piece_length_mm": 40.0,
            "node_lap_visual_side_offset_enabled": False,
        },
        "section_layout_by_group": {
            "vertical": {"layout": "single"},
            "top_chord": {"layout": "stacked"},
        },
    }
    detailed = StickDetailService().analyze(
        cfg,
        nodes,
        members,
        member_results=[{"member_id": 1, "N_N": -10.0}, {"member_id": 2, "N_N": -20.0}],
        member_checks=[{"member_id": 1, "FS_min": 2.0}, {"member_id": 2, "FS_min": 2.0}],
        out_dir=tmp_path,
    )
    vertical_pieces = [r for r in detailed["stick_pieces"] if r["member_id"] == 1]
    assert len(vertical_pieces) > 1
    assert any(r["miter_cut_end_required"] for r in vertical_pieces)
    for r in vertical_pieces[:-1]:
        assert r["miter_cut_end_required"] is False
        assert r["miter_cut_end_angle_deg"] == 90.0


def test_terminal_miter_records_skew_and_host_metadata(tmp_path) -> None:
    nodes = [
        Node(1, 0.0, -75.0, 0.0, "bottom", "L", 0.0),
        Node(2, 0.0, -75.0, 200.0, "top", "L", 0.0),
        Node(3, 100.0, -75.0, 245.0, "top", "L", 100.0),
    ]
    members = [
        _dummy_member(1, 1, 2, "vertical", 1, 200.0),
        _dummy_member(2, 2, 3, "top_chord", 2, 109.7),
    ]
    cfg = {
        "material": {
            "stick_length_mm": 120.0,
            "stick_width_mm": 7.0,
            "stick_thickness_mm": 1.5,
            "stick_mass_g": 1.4,
            "mass_limit_g": 1000.0,
        },
        "detail_model": {
            "enabled": True,
            "overlap_length_mm": 30.0,
            "strict_cut_length": True,
            "cut_increment_mm": 5.0,
            "node_face_lap_enabled": True,
            "joint_face_setback_enabled": False,
            "angled_end_cuts_enabled": True,
            "miter_cut_min_host_slope_deg": 7.5,
            "min_constructive_piece_length_mm": 40.0,
            "node_lap_visual_side_offset_enabled": False,
        },
        "section_layout_by_group": {
            "vertical": {"layout": "single"},
            "top_chord": {"layout": "stacked"},
        },
    }
    detailed = StickDetailService().analyze(
        cfg,
        nodes,
        members,
        member_results=[{"member_id": 1, "N_N": -10.0}, {"member_id": 2, "N_N": -20.0}],
        member_checks=[{"member_id": 1, "FS_min": 2.0}, {"member_id": 2, "FS_min": 2.0}],
        out_dir=tmp_path,
    )
    beveled = [r for r in detailed["stick_pieces"] if r["member_id"] == 1 and r["miter_cut_end_required"]]
    assert beveled
    assert {r["miter_cut_end_host_group"] for r in beveled} == {"top_chord"}
    assert {r["miter_cut_end_skew_sign"] for r in beveled} <= {-1.0, 1.0}


def test_internal_cross_frame_diagonal_cuts_both_ends_against_vertical_hosts(tmp_path) -> None:
    nodes = [
        Node(1, 0.0, -75.0, 0.0, "bottom", "L", 0.0),
        Node(2, 0.0, -75.0, 200.0, "top", "L", 0.0),
        Node(3, 0.0, 75.0, 0.0, "bottom", "R", 0.0),
        Node(4, 0.0, 75.0, 200.0, "top", "R", 0.0),
    ]
    members = [
        _dummy_member(1, 1, 4, "cross_frame_bracing", 1, 250.0),
        _dummy_member(2, 1, 2, "vertical", 1, 200.0),
        _dummy_member(3, 3, 4, "vertical", 1, 200.0),
    ]
    cfg = {
        "material": {
            "stick_length_mm": 120.0,
            "stick_width_mm": 7.0,
            "stick_thickness_mm": 1.5,
            "stick_mass_g": 1.4,
            "mass_limit_g": 1000.0,
        },
        "detail_model": {
            "enabled": True,
            "overlap_length_mm": 30.0,
            "strict_cut_length": True,
            "cut_increment_mm": 5.0,
            "node_face_lap_enabled": True,
            "joint_face_setback_enabled": False,
            "angled_end_cuts_enabled": True,
            "miter_cut_min_host_slope_deg": 4.0,
            "miter_cut_terminal_groups": ["cross_frame_bracing"],
            "miter_cut_host_groups": ["vertical"],
            "miter_cut_primary_host_priority_by_member_group": {"cross_frame_bracing": ["vertical"]},
            "miter_cut_required_both_ends_groups": ["cross_frame_bracing"],
            "min_constructive_piece_length_mm": 20.0,
            "node_lap_visual_side_offset_enabled": False,
        },
        "section_layout_by_group": {
            "vertical": {"layout": "single"},
            "cross_frame_bracing": {"layout": "single"},
        },
    }
    detailed = StickDetailService().analyze(
        cfg,
        nodes,
        members,
        member_results=[{"member_id": 1, "N_N": 10.0}, {"member_id": 2, "N_N": -10.0}, {"member_id": 3, "N_N": -10.0}],
        member_checks=[{"member_id": 1, "FS_min": 2.0}, {"member_id": 2, "FS_min": 2.0}, {"member_id": 3, "FS_min": 2.0}],
        out_dir=tmp_path,
    )
    cross_pieces = [r for r in detailed["stick_pieces"] if r["member_id"] == 1]
    assert any(r["miter_cut_start_required"] for r in cross_pieces)
    assert any(r["miter_cut_end_required"] for r in cross_pieces)
    assert {r["miter_cut_start_host_group"] for r in cross_pieces if r["miter_cut_start_required"]} == {"vertical"}
    assert {r["miter_cut_end_host_group"] for r in cross_pieces if r["miter_cut_end_required"]} == {"vertical"}


def test_diagonal_butt_splice_renders_two_face_splint_pieces(tmp_path) -> None:
    nodes = [Node(1, 0.0, -75.0, 0.0, "bottom", "L", 0.0), Node(2, 150.0, -75.0, 180.0, "top", "L", 150.0)]
    members = [_dummy_member(1, 1, 2, "diagonal", 1, 234.3)]
    cfg = {
        "material": {"stick_length_mm": 120.0, "stick_width_mm": 7.0, "stick_thickness_mm": 1.5, "stick_mass_g": 1.4, "mass_limit_g": 1000.0},
        "detail_model": {
            "enabled": True, "overlap_length_mm": 20.0, "strict_cut_length": True, "cut_increment_mm": 5.0,
            "joint_face_setback_enabled": False, "node_face_lap_enabled": False, "angled_end_cuts_enabled": False,
            "min_constructive_piece_length_mm": 20.0, "splice_mode_by_group": {"diagonal": "butt_with_splints"},
            "minimum_face_splints_per_butt_joint": 2, "splints_per_splice_by_group": {"diagonal": 2},
            "splint_length_mm_by_group": {"diagonal": 25.0}, "render_physical_face_splints_enabled": True,
            "render_physical_splints_by_group": ["diagonal"],
        },
        "section_layout_by_group": {"diagonal": {"layout": "stacked_edge", "stick_orientation": "edge"}},
    }
    detailed = StickDetailService().analyze(cfg, nodes, members, [{"member_id": 1, "N_N": 20.0}], [{"member_id": 1, "FS_min": 2.0}], tmp_path)
    main = [r for r in detailed["stick_pieces"] if r["member_group"] == "diagonal"]
    talas = [r for r in detailed["stick_pieces"] if r["member_group"] == "diagonal_splint"]
    joints = [r for r in detailed["glue_joints"] if r["member_group"] == "diagonal" and r["joint_type"] == "butt_with_splints"]
    assert len(main) >= 2
    assert all(min(float(a["s1_mm"]), float(b["s1_mm"])) - max(float(a["s0_mm"]), float(b["s0_mm"])) <= 1e-6 for a, b in zip(main[:-1], main[1:]))
    assert joints and all(int(j["splints_per_splice"]) == 2 for j in joints)
    assert len(talas) == 2 * len(joints)


def test_top_chord_transition_miter_can_flip_to_inclination_side() -> None:
    nodes = {
        1: Node(1, 0.0, -75.0, 200.0, "top", "L", 0.0),
        2: Node(2, 100.0, -75.0, 245.0, "top", "L", 100.0),
        3: Node(3, 200.0, -75.0, 245.0, "top", "L", 200.0),
    }
    m1 = _dummy_member(1, 1, 2, "top_chord", 8, 109.7)
    m2 = _dummy_member(2, 2, 3, "top_chord", 8, 100.0)
    env = {2: [(1, 12.0), (2, 12.0)]}
    groups = {1: "top_chord", 2: "top_chord"}
    base = StickDetailService._terminal_end_cut_spec(member=m2, node_id=2, nodes_by_id=nodes, members_by_id={1:m1,2:m2}, node_member_envelopes=env, member_group_by_id=groups, detail={"angled_end_cuts_enabled": True, "miter_cut_terminal_groups": ["top_chord"], "miter_cut_host_groups": ["top_chord"], "miter_cut_min_host_slope_deg": 4.0}, fallback_axis=(1.0,0.0,0.0), local_bevel_axis=(0.0,0.0,1.0))
    flipped = StickDetailService._terminal_end_cut_spec(member=m2, node_id=2, nodes_by_id=nodes, members_by_id={1:m1,2:m2}, node_member_envelopes=env, member_group_by_id=groups, detail={"angled_end_cuts_enabled": True, "miter_cut_terminal_groups": ["top_chord"], "miter_cut_host_groups": ["top_chord"], "miter_cut_min_host_slope_deg": 4.0, "top_chord_transition_miter_reverse_side_enabled": True}, fallback_axis=(1.0,0.0,0.0), local_bevel_axis=(0.0,0.0,1.0))
    assert base["host_relation"] == "same_chord_half_miter"
    assert flipped["skew_sign"] == -base["skew_sign"]


def test_vertical_top_piece_is_trimmed_to_sandwich_underside(tmp_path) -> None:
    nodes = [
        Node(1, 0.0, -50.0, 0.0, "bottom", "L", 0.0),
        Node(2, 0.0, -50.0, 206.5, "top", "L", 0.0),
        Node(3, 100.0, -50.0, 251.5, "top", "L", 100.0),
    ]
    members = [
        _dummy_member(1, 1, 2, "vertical", 1, 206.5),
        _dummy_member(2, 2, 3, "top_chord", 1, 109.7),
    ]
    cfg = {
        "material": {"stick_length_mm": 120.0, "stick_width_mm": 7.0, "stick_thickness_mm": 1.5, "stick_mass_g": 1.4, "mass_limit_g": 1000.0},
        "detail_model": {
            "enabled": True, "overlap_length_mm": 30.0, "strict_cut_length": True, "cut_increment_mm": 5.0,
            "node_face_lap_enabled": True, "joint_face_setback_enabled": True,
            "joint_setback_groups": ["vertical"], "joint_min_setback_mm": 1.75,
            "angled_end_cuts_enabled": True, "miter_cut_min_host_slope_deg": 7.5,
            "min_constructive_piece_length_mm": 20.0, "node_lap_visual_side_offset_enabled": False,
            "vertical_trim_to_top_chord_underside_enabled": True, "vertical_top_chord_underside_trim_mm": 6.5,
        },
        "section_layout_by_group": {"vertical": {"layout": "single"}, "top_chord": {"layout": "single"}},
    }
    detailed = StickDetailService().analyze(
        cfg, nodes, members,
        member_results=[{"member_id": 1, "N_N": -10.0}, {"member_id": 2, "N_N": -20.0}],
        member_checks=[{"member_id": 1, "FS_min": 2.0}, {"member_id": 2, "FS_min": 2.0}],
        out_dir=tmp_path,
    )
    vertical_pieces = [r for r in detailed["stick_pieces"] if r["member_id"] == 1]
    assert vertical_pieces
    assert all(float(r["joint_end_setback_mm"]) >= 6.5 for r in vertical_pieces)
    assert max(float(r["z1_mm"]) for r in vertical_pieces) <= 200.0 + 1.0e-9
    assert any(r["miter_cut_end_required"] for r in vertical_pieces)


def test_diagonal_butt_splice_renders_one_40mm_face_splint_piece(tmp_path) -> None:
    nodes = [Node(1, 0.0, -75.0, 0.0, "bottom", "L", 0.0), Node(2, 150.0, -75.0, 180.0, "top", "L", 150.0)]
    members = [_dummy_member(1, 1, 2, "diagonal", 1, 234.3)]
    cfg = {
        "material": {"stick_length_mm": 120.0, "stick_width_mm": 7.0, "stick_thickness_mm": 1.5, "stick_mass_g": 1.4, "mass_limit_g": 1000.0},
        "detail_model": {
            "enabled": True, "overlap_length_mm": 20.0, "strict_cut_length": True, "cut_increment_mm": 5.0,
            "joint_face_setback_enabled": False, "node_face_lap_enabled": False, "angled_end_cuts_enabled": False,
            "min_constructive_piece_length_mm": 20.0, "splice_mode_by_group": {"diagonal": "butt_with_splints"},
            "minimum_face_splints_per_butt_joint": 1, "splints_per_splice_by_group": {"diagonal": 1},
            "splint_length_mm_by_group": {"diagonal": 40.0}, "render_physical_face_splints_enabled": True,
            "render_physical_splints_by_group": ["diagonal"],
            "physical_connection_policy_enabled": True,
        },
        "section_layout_by_group": {"diagonal": {"layout": "stacked_edge", "stick_orientation": "edge"}},
    }
    detailed = StickDetailService().analyze(cfg, nodes, members, [{"member_id": 1, "N_N": 20.0}], [{"member_id": 1, "FS_min": 2.0}], tmp_path)
    talas = [r for r in detailed["stick_pieces"] if r["member_group"] == "diagonal_splint"]
    joints = [r for r in detailed["glue_joints"] if r["member_group"] == "diagonal" and r["joint_type"] == "butt_with_splints"]
    assert joints and all(int(j["splints_per_splice"]) == 1 for j in joints)
    assert all(float(j["splint_length_mm"]) == 40.0 for j in joints)
    assert len(talas) == len(joints)


def test_top_chord_outer_transition_uses_opposite_miter_side_from_plateau_transition() -> None:
    nodes = {
        1: Node(1, -100.0, -57.5, 100.0, "top", "L", -100.0),
        2: Node(2, 0.0, -57.5, 100.0, "top", "L", 0.0),
        3: Node(3, 100.0, -57.5, 145.0, "top", "L", 100.0),
    }
    outer = _dummy_member(1, 1, 2, "top_chord", 6, 100.0)
    slope = _dummy_member(2, 2, 3, "top_chord", 6, 109.7)
    env = {2: [(1, 12.0), (2, 12.0)]}
    groups = {1: "top_chord", 2: "top_chord"}
    regular = StickDetailService._terminal_end_cut_spec(
        member=outer, node_id=2, nodes_by_id=nodes, members_by_id={1: outer, 2: slope},
        node_member_envelopes=env, member_group_by_id=groups,
        detail={"angled_end_cuts_enabled": True, "miter_cut_terminal_groups": ["top_chord"], "miter_cut_host_groups": ["top_chord"], "miter_cut_min_host_slope_deg": 4.0, "top_chord_transition_miter_reverse_side_enabled": True},
        fallback_axis=(1.0, 0.0, 0.0), local_bevel_axis=(0.0, 0.0, 1.0),
    )
    outer_corrected = StickDetailService._terminal_end_cut_spec(
        member=outer, node_id=2, nodes_by_id=nodes, members_by_id={1: outer, 2: slope},
        node_member_envelopes=env, member_group_by_id=groups,
        detail={"angled_end_cuts_enabled": True, "miter_cut_terminal_groups": ["top_chord"], "miter_cut_host_groups": ["top_chord"], "miter_cut_min_host_slope_deg": 4.0, "top_chord_transition_miter_reverse_side_enabled": True, "top_chord_outer_transition_miter_flip_enabled": True, "top_chord_outer_transition_nodes_x_mm": [0.0, 1200.0]},
        fallback_axis=(1.0, 0.0, 0.0), local_bevel_axis=(0.0, 0.0, 1.0),
    )
    assert regular["host_relation"] == "same_chord_half_miter"
    assert outer_corrected["skew_sign"] == -regular["skew_sign"]


def test_outer_top_chord_has_single_inward_receiving_cut_and_square_free_end() -> None:
    nodes = {
        1: Node(1, -100.0, -57.5, 100.0, "top", "L", -100.0),
        2: Node(2, 0.0, -57.5, 100.0, "top", "L", 0.0),
        3: Node(3, 100.0, -57.5, 144.0, "top", "L", 100.0),
        4: Node(4, -100.0, -57.5, 0.0, "bottom", "L", -100.0),
    }
    outer = _dummy_member(2, 1, 2, "top_chord", 6, 100.0)
    ramp = _dummy_member(4, 2, 3, "top_chord", 6, 109.2)
    vertical = _dummy_member(29, 4, 1, "vertical", 3, 100.0)
    envelopes = {1: [(2, 8.5), (29, 8.5)], 2: [(2, 8.5), (4, 8.5)]}
    groups = {2: "top_chord", 4: "top_chord", 29: "vertical"}
    detail = {
        "angled_end_cuts_enabled": True,
        "miter_cut_terminal_groups": ["top_chord"],
        "miter_cut_host_groups": ["top_chord", "vertical"],
        "miter_cut_min_host_slope_deg": 4.0,
        "end_cut_angle_increment_deg": 5.0,
        "top_chord_outer_transition_single_receiving_cut_enabled": True,
        "top_chord_outer_transition_nodes_x_mm": [0.0, 1200.0],
    }
    free = StickDetailService._terminal_end_cut_spec(
        member=outer, node_id=1, nodes_by_id=nodes, members_by_id={2: outer, 4: ramp, 29: vertical},
        node_member_envelopes=envelopes, member_group_by_id=groups, detail=detail,
        fallback_axis=(1.0, 0.0, 0.0), local_bevel_axis=(0.0, 0.0, 1.0),
    )
    inward = StickDetailService._terminal_end_cut_spec(
        member=outer, node_id=2, nodes_by_id=nodes, members_by_id={2: outer, 4: ramp},
        node_member_envelopes=envelopes, member_group_by_id=groups, detail=detail,
        fallback_axis=(1.0, 0.0, 0.0), local_bevel_axis=(0.0, 0.0, 1.0),
    )
    mating = StickDetailService._terminal_end_cut_spec(
        member=ramp, node_id=2, nodes_by_id=nodes, members_by_id={2: outer, 4: ramp},
        node_member_envelopes=envelopes, member_group_by_id=groups, detail=detail,
        fallback_axis=(1.0, 0.0, 0.0), local_bevel_axis=(0.0, 0.0, 1.0),
    )
    assert free["angle_deg"] == 90.0
    assert free["host_relation"] == "outer_top_chord_square_free_end"
    assert inward["host_relation"] == "outer_top_chord_single_receiving_bevel"
    assert inward["angle_deg"] < 90.0
    assert inward["shop_reference_angle_deg"] > 90.0
    assert mating["angle_deg"] == 90.0
    assert mating["host_relation"] == "outer_top_chord_mating_square_end"


def test_collinear_top_chord_ramp_divisions_remain_square() -> None:
    nodes = {
        1: Node(1, 0.0, -57.5, 100.0, "top", "L", 0.0),
        2: Node(2, 100.0, -57.5, 144.0, "top", "L", 100.0),
        3: Node(3, 200.0, -57.5, 188.0, "top", "L", 200.0),
    }
    first = _dummy_member(4, 1, 2, "top_chord", 6, 109.2)
    second = _dummy_member(6, 2, 3, "top_chord", 6, 109.2)
    spec = StickDetailService._terminal_end_cut_spec(
        member=first, node_id=2, nodes_by_id=nodes, members_by_id={4: first, 6: second},
        node_member_envelopes={2: [(4, 8.5), (6, 8.5)]}, member_group_by_id={4: "top_chord", 6: "top_chord"},
        detail={"angled_end_cuts_enabled": True, "miter_cut_terminal_groups": ["top_chord"], "miter_cut_host_groups": ["top_chord"], "miter_cut_min_host_slope_deg": 4.0},
        fallback_axis=(1.0, 0.0, 0.0), local_bevel_axis=(0.0, 0.0, 1.0),
    )
    assert spec["angle_deg"] == 90.0
    assert spec["host_relation"] == "same_chord_collinear_square"
