from __future__ import annotations

import pytest

from src.domain.models import Node
from src.services.stick_detail_service import StickDetailService


def test_bottom_x_bracing_diagonals_are_assigned_opposite_z_layers() -> None:
    n1 = Node(1, 0.0, -50.0, 0.0, "bottom", "L", 0.0)
    n2 = Node(2, 100.0, 50.0, 0.0, "bottom", "R", 100.0)
    n3 = Node(3, 0.0, 50.0, 0.0, "bottom", "R", 0.0)
    n4 = Node(4, 100.0, -50.0, 0.0, "bottom", "L", 100.0)

    detail = {"x_bracing_crossing_policy": "alternate_layers"}
    a = StickDetailService._x_bracing_layer_offset("bottom_bracing", n1, n2, stick_thickness_mm=1.5, detail=detail)
    b = StickDetailService._x_bracing_layer_offset("bottom_bracing", n3, n4, stick_thickness_mm=1.5, detail=detail)

    assert a["handling"] == "alternate_front_back_layer_no_midspan_joint"
    assert b["handling"] == "alternate_front_back_layer_no_midspan_joint"
    assert a["midspan_connected"] is False
    assert b["midspan_connected"] is False
    assert a["offset"][2] == pytest.approx(-b["offset"][2])
    assert abs(a["offset"][2] - b["offset"][2]) > 1.5


def test_cross_frame_x_bracing_diagonals_are_assigned_opposite_x_layers() -> None:
    n1 = Node(1, 100.0, -50.0, 0.0, "bottom", "L", 100.0)
    n2 = Node(2, 100.0, 50.0, 100.0, "top", "R", 100.0)
    n3 = Node(3, 100.0, -50.0, 100.0, "top", "L", 100.0)
    n4 = Node(4, 100.0, 50.0, 0.0, "bottom", "R", 100.0)

    detail = {"x_bracing_crossing_policy": "alternate_layers"}
    a = StickDetailService._x_bracing_layer_offset("cross_frame_bracing", n1, n2, stick_thickness_mm=1.5, detail=detail)
    b = StickDetailService._x_bracing_layer_offset("cross_frame_bracing", n3, n4, stick_thickness_mm=1.5, detail=detail)

    assert a["plane"] == "crossframe_yz"
    assert b["plane"] == "crossframe_yz"
    assert a["offset"][0] == pytest.approx(-b["offset"][0])
    assert abs(a["offset"][0] - b["offset"][0]) > 1.5


def test_default_x_policy_uses_layered_x_without_midspan_joint() -> None:
    n1 = Node(1, 0.0, -50.0, 0.0, "bottom", "L", 0.0)
    n2 = Node(2, 100.0, 50.0, 0.0, "bottom", "R", 100.0)

    r = StickDetailService._x_bracing_layer_offset("bottom_bracing", n1, n2, stick_thickness_mm=1.5, detail={})

    assert r["handling"] == "alternate_front_back_layer_no_midspan_joint"
    assert abs(r["offset"][2]) > 0.5
    assert r["midspan_connected"] is False


def test_geometry_converts_secondary_x_bracing_to_alternating_single_diagonals() -> None:
    from src.services.config_service import ConfigService
    from src.services.geometry_service import GeometryService

    cfg = ConfigService().normalize(
        {
            "bridge": {
                "span_mm": 1200.0,
                "width_mm": 120.0,
                "height_mm": 250.0,
                "panel_length_mm": 200.0,
                "left_support_overhang_mm": 50.0,
                "right_support_overhang_mm": 50.0,
                "top_chord_profile": "flat",
                "end_height_mm": 250.0,
                "center_height_mm": 250.0,
                "bottom_chord_truss_type": "X",
                "top_chord_truss_type": "X",
                "internal_truss_type": "X",
                "include_bottom_x_bracing": True,
                "include_top_x_bracing": True,
                "include_cross_frame_bracing": True,
            },
            "material": {"stick_length_mm": 120.0, "stick_width_mm": 7.0, "stick_thickness_mm": 1.5},
            "load": {"design_load_kgf": 80.0},
            "detail_model": {"x_bracing_crossing_policy": "single_diagonal_no_crossing"},
        }
    )
    _, members, _, _ = GeometryService().generate(cfg)
    by_group = {}
    for m in members:
        by_group.setdefault(m.group, 0)
        by_group[m.group] += 1

    # 7 stations = 6 panels.  X would create 12 members per plane; the physical
    # no-crossing policy must create one alternating diagonal per panel.
    assert by_group.get("bottom_bracing") == 14
    assert by_group.get("top_bracing") == 14
    assert by_group.get("cross_frame_bracing") == 15


def test_split_midpoint_policy_keeps_x_geometry_and_marks_midspan_joint() -> None:
    from src.services.config_service import ConfigService
    from src.services.geometry_service import GeometryService

    cfg = ConfigService().normalize(
        {
            "bridge": {
                "span_mm": 1200.0,
                "width_mm": 120.0,
                "height_mm": 250.0,
                "panel_length_mm": 200.0,
                "left_support_overhang_mm": 50.0,
                "right_support_overhang_mm": 50.0,
                "top_chord_profile": "flat",
                "end_height_mm": 250.0,
                "center_height_mm": 250.0,
                "bottom_chord_truss_type": "X",
                "top_chord_truss_type": "X",
                "internal_truss_type": "X",
                "include_bottom_x_bracing": True,
                "include_top_x_bracing": True,
                "include_cross_frame_bracing": True,
            },
            "material": {"stick_length_mm": 120.0, "stick_width_mm": 7.0, "stick_thickness_mm": 1.5},
            "load": {"design_load_kgf": 80.0},
            "detail_model": {"x_bracing_crossing_policy": "split_midpoint_lap_joint"},
        }
    )
    _, members, _, _ = GeometryService().generate(cfg)
    by_group = {}
    for m in members:
        by_group.setdefault(m.group, 0)
        by_group[m.group] += 1

    # X remains structurally active: two diagonals per panel/station, not a
    # single Warren-style diagonal.  The crossing is solved in fabrication by
    # splitting/cutting at midspan, not by deleting one leg of the X.
    assert by_group.get("bottom_bracing") == 28
    assert by_group.get("top_bracing") == 28
    assert by_group.get("cross_frame_bracing") == 30

    n1 = Node(1, 0.0, -50.0, 0.0, "bottom", "L", 0.0)
    n2 = Node(2, 100.0, 50.0, 0.0, "bottom", "R", 100.0)
    r = StickDetailService._x_bracing_layer_offset(
        "bottom_bracing",
        n1,
        n2,
        stick_thickness_mm=1.5,
        detail={"x_bracing_crossing_policy": "split_midpoint_lap_joint"},
    )
    assert r["handling"] == "split_midpoint_lap_joint"
    assert r["midspan_connected"] is True
    assert r["offset"] == pytest.approx((0.0, 0.0, 0.0))
