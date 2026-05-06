from __future__ import annotations

from src.domain.models import Member, Node
from src.services.connection_planner import ConnectionPlanner
from src.services.section_service import SectionService


def _member(mid: int, i: int, j: int, group: str = "diagonal", n_sticks: int = 2) -> Member:
    sec = SectionService().composite_section(
        n_sticks,
        {
            "stick_width_mm": 7.0,
            "stick_thickness_mm": 1.5,
            "compression_capacity_one_stick_N": 1.0,
            "compression_capacity_two_sticks_N": 2.0,
            "tension_capacity_per_stick_N": 3.0,
        },
        {"layout": "stacked"},
    )
    return Member(mid, i, j, group, n_sticks, sec["A"], sec["A"], sec["A"], sec["Iy"], sec["Iz"], sec["J"], 6000.0, 500.0, 1.0, 1.0, 100.0)


def test_connection_planner_force_ratio_levels(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["analysis"]["enforce_symmetry"] = False
    planner = ConnectionPlanner()
    nodes = [
        Node(1, 0.0, -50.0, 0.0, "bottom", "L", 0.0),
        Node(2, 100.0, -50.0, 0.0, "bottom", "L", 100.0),
        Node(3, 0.0, 50.0, 0.0, "bottom", "R", 0.0),
        Node(4, 100.0, 50.0, 0.0, "bottom", "R", 100.0),
        Node(5, 200.0, -50.0, 0.0, "bottom", "L", 200.0),
    ]
    members = [
        _member(1, 1, 2, "diagonal"),
        _member(2, 3, 4, "diagonal"),
        _member(3, 2, 5, "top_chord"),
        _member(4, 4, 5, "top_chord"),
    ]
    member_results = [
        {"member_id": 1, "N_N": 100.0},   # ratio 0.10 -> light
        {"member_id": 2, "N_N": 250.0},   # ratio 0.25 -> moderate
        {"member_id": 3, "N_N": 1000.0},  # ratio 1.00 -> reinforced
        {"member_id": 4, "N_N": -900.0},  # compressão crítica
    ]
    member_checks = [
        {"member_id": 1, "FS_min": 4.0, "utilization": 0.2},
        {"member_id": 2, "FS_min": 2.5, "utilization": 0.5},
        {"member_id": 3, "FS_min": 1.1, "utilization": 0.9},
        {"member_id": 4, "FS_min": 0.9, "utilization": 0.95},
    ]

    plan = planner.assign_member_joint_plan(cfg, nodes, members, member_results, member_checks)

    assert plan[1]["recommended_joint_model"] in {"single_lap", "single_lap_tala"}
    assert plan[2]["recommended_joint_model"] == "double_lap"
    assert plan[3]["recommended_joint_model"] == "double_lap_reinforced"
    assert plan[4]["force_state"] == "compression"
    assert plan[4]["recommended_joint_model"] != "butt_plain"
    assert plan[4]["recommended_joint_model"] in {"double_lap", "double_lap_reinforced"}
