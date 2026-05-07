from __future__ import annotations

from src.domain.models import Member, Node
from src.services.assembly_tutorial_service import AssemblyTutorialService


def test_assembly_tutorial_generates_steps_and_exports(tmp_path, base_cfg: dict) -> None:
    cfg = base_cfg
    nodes = [
        Node(1, 0.0, -50.0, 0.0, "bottom", "L", 0.0),
        Node(2, 100.0, -50.0, 0.0, "bottom", "L", 100.0),
        Node(3, 0.0, 50.0, 0.0, "bottom", "R", 0.0),
        Node(4, 100.0, 50.0, 0.0, "bottom", "R", 100.0),
    ]
    members = [
        Member(1, 1, 2, "bottom_chord", 2, 10.0, 10.0, 10.0, 20.0, 20.0, 1.0, 6000.0, 500.0, 1.0, 1.0, 100.0),
        Member(2, 3, 4, "bottom_chord", 2, 10.0, 10.0, 10.0, 20.0, 20.0, 1.0, 6000.0, 500.0, 1.0, 1.0, 100.0),
    ]
    detailed = {
        "stick_pieces": [
            {"member_id": 1, "member_group": "bottom_chord", "cut_length_mm": 90.0, "mass_g": 0.9, "x0_mm": 0, "y0_mm": -50, "z0_mm": 0, "x1_mm": 90, "y1_mm": -50, "z1_mm": 0},
            {"member_id": 2, "member_group": "bottom_chord", "cut_length_mm": 88.0, "mass_g": 0.88, "x0_mm": 0, "y0_mm": 50, "z0_mm": 0, "x1_mm": 88, "y1_mm": 50, "z1_mm": 0},
        ],
        "glue_joints": [
            {"joint_id": "J1", "member_id": 1, "joint_model": "single_lap", "overlap_length_mm": 30.0, "splice_center_mm": 45.0}
        ],
        "cutting_list": [
            {"cut_length_mm": 90.0, "quantity": 1, "total_length_mm": 90.0}
        ],
    }

    svc = AssemblyTutorialService()
    result = svc.build(cfg, nodes, members, detailed, tmp_path)

    assert result["assembly_steps"]
    assert result["cut_list"]
    assert result["joint_list"]
    assert result["stick_count_by_step"]
    assert result["stick_count_by_member"]
    assert result["symmetry_repetition_map"]

    assert (tmp_path / "assembly_tutorial.json").exists()
    assert (tmp_path / "assembly_tutorial.md").exists()
    assert (tmp_path / "assembly_steps.csv").exists()
