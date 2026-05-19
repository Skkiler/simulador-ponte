from __future__ import annotations

import pytest

from src.services.section_service import SectionService


MAT = {
    "stick_width_mm": 7.0,
    "stick_thickness_mm": 1.5,
    "compression_capacity_one_stick_N": 39.2266,
    "compression_capacity_two_sticks_N": 107.87315,
    "tension_capacity_per_stick_N": 706.0788,
}


def test_balanced_box_keeps_odd_count_centroid_on_member_axis() -> None:
    sec = SectionService.composite_section(
        5,
        MAT,
        {
            "layout": "box",
            "stick_orientation": "edge",
            "spacing_y_mm": 28.0,
            "spacing_z_mm": 28.0,
        },
    )

    assert sec["centroid_y_mm"] == pytest.approx(0.0)
    assert sec["centroid_z_mm"] == pytest.approx(0.0)
    assert (0.0, 0.0) in sec["stick_positions_yz"]


def test_contact_box_for_six_sticks_is_centered_and_buildable() -> None:
    sec = SectionService.composite_section(
        6,
        MAT,
        {
            "layout": "box",
            "stick_orientation": "edge",
            "spacing_y_mm": 28.0,
            "spacing_z_mm": 28.0,
        },
    )

    assert sec["layout"] == "contact_box"
    assert sec["centroid_y_mm"] == pytest.approx(0.0)
    assert sec["centroid_z_mm"] == pytest.approx(0.0)
    assert sec["buckling_I_critical_mm4"] > 200.0
    assert set(sec["stick_orientations"]) == {"flat", "edge"}


def test_three_stick_box_request_becomes_connected_tee_not_box() -> None:
    sec = SectionService.composite_section(
        3,
        MAT,
        {
            "layout": "box",
            "stick_orientation": "edge",
            "spacing_y_mm": 28.0,
            "spacing_z_mm": 28.0,
        },
    )

    assert sec["layout"] == "tee3"
    assert sec["centroid_y_mm"] == pytest.approx(0.0)
    assert sec["centroid_z_mm"] == pytest.approx(0.0)
    assert set(sec["stick_orientations"]) == {"flat", "edge"}


def test_legacy_corner_cycle_is_explicit_and_eccentric() -> None:
    sec = SectionService.composite_section(
        5,
        MAT,
        {
            "layout": "box",
            "stick_orientation": "edge",
            "spacing_y_mm": 28.0,
            "spacing_z_mm": 28.0,
            "box_extra_stick_strategy": "corner_cycle",
        },
    )

    assert abs(sec["centroid_y_mm"]) > 1.0
    assert abs(sec["centroid_z_mm"]) > 1.0


def test_unmodeled_laced_box_is_demoted_to_contact_box() -> None:
    sec = SectionService.composite_section(
        4,
        MAT,
        {
            "layout": "box",
            "stick_orientation": "edge",
            "spacing_y_mm": 28.0,
            "spacing_z_mm": 28.0,
            "box_extra_stick_strategy": "laced_box",
        },
    )

    assert sec["layout"] == "contact_box"
    assert sec["section_connection_model"] == "four_side_contact_box_with_face_side_glue"
    assert max(abs(y) for y, _ in sec["stick_positions_yz"]) < 7.0
    assert max(abs(z) for _, z in sec["stick_positions_yz"]) < 7.0
