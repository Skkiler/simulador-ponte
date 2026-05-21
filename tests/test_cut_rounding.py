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
