from __future__ import annotations

from src.domain.models import Member, Node
from src.services.active_design_planner import ActiveDesignPlanner
from src.services.section_service import SectionService


def _member(mid: int, i: int, j: int, group: str, n_sticks: int, L: float = 100.0) -> Member:
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
    return Member(mid, i, j, group, n_sticks, sec["A"], sec["A"], sec["A"], sec["Iy"], sec["Iz"], sec["J"], 6000.0, 500.0, 1.0, 1.0, L)


def test_high_force_tension_with_high_fs_is_not_forced_to_reinforce(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["analysis"]["enforce_symmetry"] = False
    cfg["analysis"]["target_min_fs"] = 2.0
    cfg["planner"]["local_sizing"]["structural_floor_ratio_primary"] = 0.0
    cfg["member_sticks_by_group"]["vertical"] = 2

    nodes = [
        Node(1, 0.0, 0.0, 0.0, "bottom", "R", 0.0),
        Node(2, 100.0, 0.0, 0.0, "bottom", "R", 100.0),
        Node(3, 0.0, 0.0, 80.0, "top", "R", 0.0),
        Node(4, 100.0, 0.0, 80.0, "top", "R", 100.0),
    ]
    members = [
        _member(1, 1, 2, "top_chord", 2),
        _member(2, 1, 3, "vertical", 2),
    ]
    member_results = [
        {"member_id": 1, "N_N": 1500.0},
        {"member_id": 2, "N_N": -900.0},
    ]
    member_checks = [
        {"member_id": 1, "FS_min": 4.2, "utilization": 0.22, "governing_mode": "tension_capacity"},
        {"member_id": 2, "FS_min": 0.8, "utilization": 1.15, "governing_mode": "compression_direct"},
    ]

    plan = ActiveDesignPlanner().build_member_sizing_plan(cfg, nodes, members, member_results, member_checks)
    assert plan[1].action != "reinforce"
    assert plan[2].action == "reinforce"


def test_near_zero_vertical_can_reduce_to_one_stick(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["analysis"]["enforce_symmetry"] = False
    cfg["planner"]["local_sizing"]["structural_floor_ratio_primary"] = 0.0
    cfg["member_sticks_by_group"]["vertical"] = 1

    nodes = [
        Node(1, 0.0, 0.0, 0.0, "bottom", "R", 0.0),
        Node(2, 0.0, 0.0, 80.0, "top", "R", 0.0),
    ]
    members = [_member(1, 1, 2, "vertical", 3)]
    member_results = [{"member_id": 1, "N_N": 0.0}]
    member_checks = [{"member_id": 1, "FS_min": 8.0, "utilization": 0.01, "governing_mode": "tension_capacity"}]

    plan = ActiveDesignPlanner().build_member_sizing_plan(cfg, nodes, members, member_results, member_checks)
    assert plan[1].n_sticks_recommended == 1
    assert plan[1].action in {"lighten", "simplify_joint"}


def test_donor_pass_reallocates_mass_before_reinforcing(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["analysis"]["enforce_symmetry"] = False
    cfg["analysis"]["target_min_fs"] = 2.0
    cfg["planner"]["local_sizing"]["structural_floor_ratio_primary"] = 0.0
    cfg["planner"]["local_sizing"]["donor_fs_threshold"] = 3.0
    cfg["member_sticks_by_group"]["top_chord"] = 2
    cfg["member_sticks_by_group"]["diagonal"] = 1

    nodes = [
        Node(1, 0.0, 0.0, 0.0, "bottom", "R", 0.0),
        Node(2, 100.0, 0.0, 0.0, "bottom", "R", 100.0),
        Node(3, 50.0, 0.0, 80.0, "top", "R", 50.0),
    ]
    members = [
        _member(1, 1, 2, "top_chord", 2),
        _member(2, 1, 3, "diagonal", 3),
    ]
    member_results = [
        {"member_id": 1, "N_N": 1200.0},
        {"member_id": 2, "N_N": 30.0},
    ]
    member_checks = [
        {"member_id": 1, "FS_min": 3.2, "utilization": 0.8, "governing_mode": "tension_capacity"},
        {"member_id": 2, "FS_min": 5.0, "utilization": 0.1, "governing_mode": "tension_capacity"},
    ]

    plan = ActiveDesignPlanner().build_member_sizing_plan(cfg, nodes, members, member_results, member_checks)
    assert plan[1].n_sticks_recommended >= plan[1].n_sticks_current
    assert plan[2].n_sticks_recommended <= plan[2].n_sticks_current
