from __future__ import annotations

from src.services.rupture_estimator import estimate_rupture_load


def test_rupture_estimator_uses_fs_glue_shear_and_design_fs() -> None:
    cfg = {"analysis": {"acceptance_min_design_breaking_load_kgf": 80.0}}
    member_checks = [
        {
            "member_id": 1,
            "member_role": "primary",
            "group": "top_chord",
            "FS_min": 2.0,
            "FS_min_all_raw": 2.0,
            "FS_design": 2.0,
            "design_relevant": True,
            "governing_mode": "compression_direct",
        },
        {
            "member_id": 2,
            "member_role": "stabilizer",
            "group": "top_bracing",
            "FS_min": 0.6,
            "FS_min_all_raw": 0.6,
            "FS_design": None,
            "design_relevant": False,
            "governing_mode": "buckling_y",
        },
    ]
    support_checks = [{"node_id": 10, "FS_support_reaction": 1.8}]
    detailed = {
        "weakest_glue_joints": [
            {"joint_id": "J1", "member_group": "diagonal", "FS_glue_shear": 1.4}
        ]
    }

    out = estimate_rupture_load(cfg, member_checks, support_checks, detailed, load_kgf=80.0)
    assert out["predicted_breaking_load_primary_kgf"] == 160.0
    assert out["predicted_breaking_load_all_kgf"] == 48.0
    assert out["predicted_breaking_load_design_kgf"] == 112.0
    assert out["predicted_breaking_load_kgf"] == 112.0
    assert out["governing_limit_state"] == "glue"
    assert out["governing_joint_id"] == "J1"


def test_rupture_estimator_governing_mode_matches_smallest_candidate() -> None:
    cfg = {}
    member_checks = [
        {
            "member_id": 9,
            "member_role": "primary",
            "group": "bottom_chord",
            "FS_min": 1.1,
            "FS_min_all_raw": 1.1,
            "FS_design": 1.1,
            "design_relevant": True,
            "governing_mode": "tension_capacity",
        }
    ]
    support_checks = [{"node_id": 2, "FS_support_reaction": 0.95}]
    detailed = {"glue_joints": []}
    out = estimate_rupture_load(cfg, member_checks, support_checks, detailed, load_kgf=80.0)
    assert out["governing_limit_state"] == "support"
    assert out["governing_support_node_id"] == 2
    assert out["predicted_breaking_load_design_kgf"] == 76.0


def test_rupture_estimator_reports_glue_governing_without_fs_contradiction() -> None:
    cfg = {}
    member_checks = [
        {
            "member_id": 1,
            "member_role": "primary",
            "group": "bottom_chord",
            "FS_min": 1.7,
            "FS_min_all_raw": 1.7,
            "FS_design": 1.7,
            "design_relevant": True,
            "governing_mode": "tension_capacity",
        }
    ]
    support_checks = [{"node_id": 1, "FS_support_reaction": 1.4}]
    detailed = {"glue_joints": [{"joint_id": "Jb", "member_group": "bottom_chord", "FS_glue_shear": 0.9}]}

    out = estimate_rupture_load(cfg, member_checks, support_checks, detailed, load_kgf=80.0)

    assert out["min_fs_member_design"] == 1.7
    assert out["min_fs_glue"] == 0.9
    assert out["governing_limit_state"] == "glue"
    assert out["predicted_breaking_load_by_members_kgf"] == 136.0
    assert out["predicted_breaking_load_by_glue_kgf"] == 72.0
    assert out["predicted_breaking_load_design_kgf"] == 72.0
