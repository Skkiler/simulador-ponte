from __future__ import annotations

import pytest

from src.domain.models import Node
from src.services.stick_detail_service import StickDetailService


def test_bottom_x_bracing_diagonals_are_assigned_opposite_z_layers() -> None:
    n1 = Node(1, 0.0, -50.0, 0.0, "bottom", "L", 0.0)
    n2 = Node(2, 100.0, 50.0, 0.0, "bottom", "R", 100.0)
    n3 = Node(3, 0.0, 50.0, 0.0, "bottom", "R", 0.0)
    n4 = Node(4, 100.0, -50.0, 0.0, "bottom", "L", 100.0)

    a = StickDetailService._x_bracing_layer_offset("bottom_bracing", n1, n2, stick_thickness_mm=1.5, detail={})
    b = StickDetailService._x_bracing_layer_offset("bottom_bracing", n3, n4, stick_thickness_mm=1.5, detail={})

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

    a = StickDetailService._x_bracing_layer_offset("cross_frame_bracing", n1, n2, stick_thickness_mm=1.5, detail={})
    b = StickDetailService._x_bracing_layer_offset("cross_frame_bracing", n3, n4, stick_thickness_mm=1.5, detail={})

    assert a["plane"] == "crossframe_yz"
    assert b["plane"] == "crossframe_yz"
    assert a["offset"][0] == pytest.approx(-b["offset"][0])
    assert abs(a["offset"][0] - b["offset"][0]) > 1.5
