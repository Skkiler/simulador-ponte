from __future__ import annotations

from src.domain.models import Member, Node
from src.services.piece_inspection_service import PieceInspectionService
from src.services.config_service import ConfigService
from src.services.stick_detail_service import StickDetailService


def _vertical_member(length: float = 146.0, sticks: int = 4) -> tuple[list[Node], list[Member]]:
    nodes = [
        Node(1, 0.0, -57.5, 0.0, "bottom", "L", 0.0),
        Node(2, 0.0, -57.5, length, "top", "L", 0.0),
    ]
    members = [
        Member(1, 1, 2, "vertical", sticks, 10.5 * sticks, 10.5 * sticks, 10.5 * sticks,
               171.5, 90.5625, 91.72, 6000.0, 500.0, 0.8, 0.8, length,
               "solid_face_laminated_edge", "edge")
    ]
    return nodes, members


def _cfg() -> dict:
    return {
        "material": {
            "stick_length_mm": 120.0,
            "stick_width_mm": 7.0,
            "stick_thickness_mm": 1.5,
            "stick_mass_g": 1.4,
            "mass_limit_g": 1000.0,
            "glue_cure_solids_fraction": 0.55,
        },
        "analysis": {"acceptance_min_glue_fs": 1.0},
        "detail_model": {
            "enabled": True,
            "physical_connection_policy_enabled": True,
            "overlap_length_mm": 20.0,
            "strict_cut_length": True,
            "cut_increment_mm": 5.0,
            "max_cut_length_mm": 120.0,
            "min_constructive_piece_length_mm": 20.0,
            "joint_face_setback_enabled": False,
            "node_face_lap_enabled": False,
            "angled_end_cuts_enabled": False,
            "splice_stagger_enabled": True,
            "forbid_same_lane_axial_interpenetration": True,
            "same_lane_interpenetration_audit_enabled": True,
            "nonoverlap_staggered_laminate_groups": ["vertical"],
            "laminated_butt_bridge_overlap_mm": 20.0,
            "laminated_butt_joint_stagger_mm": 25.0,
            "solid_face_laminated_continuous_glue_mass_enabled": True,
        },
        "section_layout_by_group": {
            "vertical": {"layout": "solid_face_laminated_edge", "stick_orientation": "edge"},
        },
    }


def test_staggered_laminate_segments_are_nonoverlapping_and_constructible(tmp_path) -> None:
    nodes, members = _vertical_member()
    detailed = StickDetailService().analyze(
        _cfg(), nodes, members,
        member_results=[{"member_id": 1, "N_N": -20.0}],
        member_checks=[{"member_id": 1, "FS_min": 2.0}],
        out_dir=tmp_path,
    )
    pieces = detailed["stick_pieces"]
    assert detailed["summary"]["same_lane_axial_interpenetration_count"] == 0
    assert all(float(p["geometric_piece_length_mm"]) >= 20.0 for p in pieces)
    assert all(float(p["geometric_piece_length_mm"]) <= 120.0 for p in pieces)
    for lane in {int(p["lane"]) for p in pieces}:
        lane_pieces = sorted([p for p in pieces if int(p["lane"]) == lane], key=lambda p: float(p["s0_mm"]))
        for left, right in zip(lane_pieces[:-1], lane_pieces[1:]):
            assert float(left["s1_mm"]) <= float(right["s0_mm"]) + 1.0e-9


def test_member_summary_reports_assembled_envelope_dimensions() -> None:
    pieces = [{
        "stick_id": "M001-L01-P01", "member_id": 1, "member_group": "vertical", "lane": 1,
        "piece_index": 1, "mass_g": 1.0, "n_sticks": 4, "shop_cut_length_mm": 70.0,
        "installed_length_mm": 68.0, "layout": "solid_face_laminated_edge",
        "assembled_member_length_mm": 137.25, "assembled_member_width_mm": 6.0,
        "assembled_member_thickness_mm": 7.0,
        "longitudinal_splice_model": "staggered_butt_bridged_by_adjacent_face_lamina",
    }]
    info = PieceInspectionService.inspect_member(1, pieces, [], [{"member_id": 1, "FS_design": 2.0}])
    summary = {row["item"]: row["valor"] for row in PieceInspectionService.member_summary_rows(info)}
    assert summary["Comprimento total montado"] == "137.25 mm"
    assert summary["Largura total montada"] == "6.00 mm"
    assert summary["Espessura total montada"] == "7.00 mm"
    assert "staggered_butt" in summary["Modelo longitudinal"]


def test_eleven_main_span_stations_keep_restored_support_closure_frames(base_cfg: dict) -> None:
    from src.services.geometry_service import GeometryService
    cfg = ConfigService().normalize(base_cfg)
    nodes, members, _, _ = GeometryService().generate(cfg)
    by = {n.id: n for n in nodes}
    left_vertical_x = sorted({round(0.5 * (by[m.i].x + by[m.j].x), 6) for m in members if m.group == "vertical" and by[m.i].y < 0.0})
    assert left_vertical_x == [-100.0] + [float(v) for v in range(0, 1201, 120)] + [1300.0]
    assert len([x for x in left_vertical_x if 0.0 <= x <= 1200.0]) == 11
    top_ranges = {(round(min(by[m.i].x, by[m.j].x), 6), round(max(by[m.i].x, by[m.j].x), 6)) for m in members if m.group == "top_chord" and by[m.i].y < 0.0}
    assert (-100.0, 0.0) in top_ranges
    assert (1200.0, 1300.0) in top_ranges
    assert cfg["bridge"]["top_chord_exclude_support_overhang_panels"] is False
    assert cfg["bridge"]["vertical_exclude_support_overhang_stations"] is False


def test_top_chord_transition_external_cover_uses_relief_not_collision(base_cfg: dict, tmp_path) -> None:
    from src.services.geometry_service import GeometryService
    cfg = ConfigService().normalize(base_cfg)
    nodes, members, _, _ = GeometryService().generate(cfg)
    detailed = StickDetailService().analyze(cfg, nodes, members, [], [], tmp_path)
    relief = [r for r in detailed["stick_pieces"] if r.get("member_group") == "top_chord" and float(r.get("transition_cover_relief_applied_mm", 0.0) or 0.0) > 0.0]
    assert len(relief) == 8
    assert {str(r.get("sandwich_lane_role")) for r in relief} == {"capa_externa_inferior_1", "capa_externa_inferior_2"}
    assert {round(float(r["transition_cover_relief_applied_mm"]), 6) for r in relief} == {4.0}
    assert all(float(r["geometric_piece_length_mm"]) >= 20.0 for r in relief)


def test_piece_rows_export_assembled_dimensions_for_hover(tmp_path) -> None:
    nodes, members = _vertical_member()
    detailed = StickDetailService().analyze(
        _cfg(), nodes, members,
        member_results=[{"member_id": 1, "N_N": -20.0}],
        member_checks=[{"member_id": 1, "FS_min": 2.0}],
        out_dir=tmp_path,
    )
    row = detailed["stick_pieces"][0]
    assert float(row["assembled_member_length_mm"]) > 0.0
    assert float(row["assembled_member_width_mm"]) > 0.0
    assert float(row["assembled_member_thickness_mm"]) > 0.0
    assert row["longitudinal_splice_model"] == "staggered_butt_bridged_by_adjacent_face_lamina"
