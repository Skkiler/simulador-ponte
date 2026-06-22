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
    # Com número ímpar de palitos acima de 4, uma seção sem interpenetração pode
    # ficar levemente excêntrica. O requisito de realismo aqui é não usar um
    # palito central ocupando o mesmo volume dos caps/webs.
    assert abs(float(sec["centroid_z_mm"])) <= 1.0
    assert sec["no_internal_overlap"] is True


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


def _rectangles_for_section(sec):
    rects = []
    for (y, z), yd, zd in zip(
        sec["stick_positions_yz"],
        sec["stick_width_y_mm_by_lane"],
        sec["stick_height_z_mm_by_lane"],
    ):
        rects.append((y - yd / 2, y + yd / 2, z - zd / 2, z + zd / 2))
    return rects


def _overlap_area(a, b):
    oy = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    oz = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    return oy * oz


def test_contact_box_sections_do_not_interpenetrate_stick_volumes() -> None:
    for n in (3, 4, 5, 6, 7, 8):
        sec = SectionService.composite_section(
            n,
            MAT,
            {
                "layout": "box",
                "stick_orientation": "edge",
                "spacing_y_mm": 24.0,
                "spacing_z_mm": 24.0,
            },
        )
        rects = _rectangles_for_section(sec)
        for i, a in enumerate(rects):
            for b in rects[i + 1:]:
                assert _overlap_area(a, b) == pytest.approx(0.0)


def test_simple_layout_aliases_expose_buildability_flags() -> None:
    sec = SectionService.composite_section(
        4,
        MAT,
        {
            "layout": "laminated_rectangular",
            "stick_orientation": "flat",
        },
    )
    assert sec["section_buildable"] is True
    assert sec["no_internal_overlap"] is True
    assert sec["composite_action_eta_I"] == sec["eta_I"]

    sec_box = SectionService.composite_section(
        4,
        MAT,
        {
            "layout": "simple_box_with_real_spacers",
            "stick_orientation": "edge",
        },
    )
    assert sec_box["layout"] == "contact_box"
    assert sec_box["section_buildable"] is True


def test_upper_tee_bottom_chord_is_buildable_and_stiff_in_vertical_plane() -> None:
    sec_t = SectionService.composite_section(
        2,
        MAT,
        {"layout": "tee_top", "stick_orientation": "mixed"},
    )
    sec_flat = SectionService.composite_section(
        1,
        MAT,
        {"layout": "single", "stick_orientation": "flat"},
    )

    assert sec_t["layout"] == "tee_top"
    assert sec_t["stick_orientations"] == ["edge", "flat"]
    assert sec_t["centroid_z_mm"] > 0.0
    assert sec_t["no_internal_overlap"] is True
    assert sec_t["section_connection_model"] == "upper_T_continuous_web_flange_contact"
    assert sec_t["Iy"] > 50.0 * sec_flat["Iy"]


def test_closed_sandwich_eight_sticks_has_face_contact_without_internal_overlap() -> None:
    sec = SectionService.composite_section(
        8,
        MAT,
        {"layout": "closed_sandwich_4core_2caps_2covers", "joint_quality": "face_laminated"},
    )
    assert sec["layout"] == "closed_sandwich_4core_2caps_2covers"
    assert sec["section_connection_model"] == "closed_face_sandwich_core_caps_external_covers"
    assert sec["no_internal_overlap"] is True
    assert sec["stick_roles"].count("nucleo_central") == 4
    assert "capa_externa_superior_1" in sec["stick_roles"]
    assert "capa_externa_inferior_1" in sec["stick_roles"]
    assert sec["Iy"] > SectionService.composite_section(8, MAT, {"layout": "solid_face_laminated_edge", "stick_orientation": "edge"})["Iy"]


def test_closed_sandwich_six_sticks_models_built_post_core_and_caps() -> None:
    sec = SectionService.composite_section(6, MAT, {"layout": "closed_sandwich_4core_2caps"})
    assert sec["layout"] == "closed_sandwich_4core_2caps"
    assert sec["no_internal_overlap"] is True
    assert len(sec["stick_positions_yz"]) == 6
    assert sec["stick_roles"].count("nucleo_central") == 4
