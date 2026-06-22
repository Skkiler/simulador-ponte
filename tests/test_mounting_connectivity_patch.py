from __future__ import annotations

import pytest

from src.domain.models import Member, Node
from src.services.config_service import ConfigService
from src.services.stick_detail_service import StickDetailService


def _node(node_id: int, x: float, y: float, z: float) -> Node:
    return Node(node_id, x, y, z, "bottom", "L", x)


def _member(group: str) -> Member:
    return Member(
        id=1,
        i=1,
        j=2,
        group=group,
        n_sticks=1,
        A=1.0,
        Asy=1.0,
        Asz=1.0,
        Iy=1.0,
        Iz=1.0,
        J=1.0,
        E=1.0,
        G=1.0,
        Ky=1.0,
        Kz=1.0,
        L=100.0,
    )


def test_explicit_zero_contact_stack_offset_is_preserved(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["detail_model"]["node_lap_physical_offset_model"] = "contact_stack_not_exploded"
    cfg["detail_model"]["contact_stack_offsets_mm"]["cross_frame_bracing_x"] = 0.0

    normalized = ConfigService().normalize(cfg)

    assert normalized["detail_model"]["contact_stack_offsets_mm"]["cross_frame_bracing_x"] == 0.0


def test_chords_keep_node_axis_continuity_even_when_legacy_setback_group_is_present() -> None:
    detail = {
        "joint_face_setback_enabled": True,
        "continuous_chord_axis_setback_disabled": True,
        "joint_setback_groups": ["top_chord", "bottom_chord"],
        "joint_min_setback_mm": 1.75,
    }

    assert StickDetailService._joint_face_setbacks(
        _member("bottom_chord"),
        100.0,
        node_member_envelopes={},
        detail=detail,
        stick_w=7.0,
        stick_t=1.5,
        min_constructive_piece_length_mm=20.0,
    ) == pytest.approx((0.0, 0.0))
    assert StickDetailService._joint_face_setbacks(
        _member("top_chord"),
        100.0,
        node_member_envelopes={},
        detail=detail,
        stick_w=7.0,
        stick_t=1.5,
        min_constructive_piece_length_mm=20.0,
    ) == pytest.approx((0.0, 0.0))


def test_cross_frame_station_diagonal_uses_opposite_montante_contact_layer() -> None:
    ni = _node(1, 100.0, -50.0, 0.0)
    nj = _node(2, 100.0, 50.0, 100.0)

    offset = StickDetailService._node_lap_visual_side_offset(
        member=_member("cross_frame_bracing"), ni=ni, nj=nj, detail={}
    )

    # O diafragma interno é colado na face longitudinal oposta à diagonal
    # 3D; 1,6 mm representa a camada de contato, não uma peça flutuante.
    assert offset == pytest.approx((-1.6, 0.0, 0.0))


def test_vertical_can_be_seated_on_tee_flange_without_lateral_displacement() -> None:
    ni = _node(1, 100.0, -50.0, 0.0)
    nj = _node(2, 100.0, -50.0, 100.0)
    detail = {
        "node_lap_visual_side_offset_enabled": True,
        "node_lap_visual_side_offset_groups": ["vertical", "diagonal"],
        "contact_stack_offsets_mm": {"vertical_y": 0.0, "diagonal_y": 5.75},
    }

    vertical_offset = StickDetailService._node_lap_visual_side_offset(
        member=_member("vertical"), ni=ni, nj=nj, detail=detail
    )
    diagonal_offset = StickDetailService._node_lap_visual_side_offset(
        member=_member("diagonal"), ni=ni, nj=nj, detail=detail
    )

    assert vertical_offset == pytest.approx((0.0, 0.0, 0.0))
    assert diagonal_offset == pytest.approx((0.0, -5.75, 0.0))
