from __future__ import annotations

from src.domain.models import Member, Node
from src.services.active_design_planner import ActiveDesignPlanner
from src.services.section_service import SectionService


def _member(mid: int, i: int, j: int, group: str, n_sticks: int) -> Member:
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


def test_member_sizing_plan_reinforce_lighten_and_symmetry(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["analysis"]["enforce_symmetry"] = True
    cfg["analysis"]["target_min_fs"] = 2.0

    nodes = [
        Node(1, 100.0, -50.0, 0.0, "bottom", "L", 100.0),
        Node(2, 200.0, -50.0, 0.0, "bottom", "L", 200.0),
        Node(3, 100.0, 50.0, 0.0, "bottom", "R", 100.0),
        Node(4, 200.0, 50.0, 0.0, "bottom", "R", 200.0),
        Node(5, 150.0, 0.0, 0.0, "bottom", "R", 150.0),
    ]
    members = [
        _member(1, 1, 2, "top_chord", 2),
        _member(2, 3, 4, "top_chord", 2),
        _member(3, 2, 5, "top_bracing", 3),
    ]
    member_results = [
        {"member_id": 1, "N_N": 900.0},
        {"member_id": 2, "N_N": 900.0},
        {"member_id": 3, "N_N": 40.0},
    ]
    member_checks = [
        {"member_id": 1, "FS_min": 1.0, "governing_mode": "compression_buckling"},
        {"member_id": 2, "FS_min": 1.0, "governing_mode": "compression_buckling"},
        {"member_id": 3, "FS_min": 5.0, "governing_mode": "tension_capacity"},
    ]

    planner = ActiveDesignPlanner()
    plan = planner.build_member_sizing_plan(cfg, nodes, members, member_results, member_checks)

    assert plan[1].action == "reinforce"
    assert plan[1].n_sticks_recommended >= plan[1].n_sticks_current
    assert plan[3].action == "lighten"
    assert plan[3].n_sticks_recommended < plan[3].n_sticks_current
    assert 2 in plan[1].applied_to_member_ids
    assert plan[1].n_sticks_recommended == plan[2].n_sticks_recommended

    cfg2 = planner.apply_member_sizing_plan(cfg, plan)
    sid_1 = int(cfg2["member_sticks_by_id"]["1"])
    sid_2 = int(cfg2["member_sticks_by_id"]["2"])
    assert sid_1 == sid_2


def test_local_sizing_zero_force_member_does_not_inherit_group_reinforcement(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["analysis"]["enforce_symmetry"] = False
    cfg["analysis"]["target_min_fs"] = 2.0
    cfg["member_sticks_by_group"]["vertical"] = 4
    cfg.setdefault("planner", {}).setdefault("local_sizing", {})
    cfg["planner"]["local_sizing"]["allow_optional_member_removal"] = False

    nodes = [
        Node(1, 0.0, 0.0, 0.0, "bottom", "R", 0.0),
        Node(2, 100.0, 0.0, 0.0, "bottom", "R", 100.0),
        Node(3, 0.0, 0.0, 80.0, "top", "R", 0.0),
        Node(4, 100.0, 0.0, 80.0, "top", "R", 100.0),
    ]
    members = [
        _member(1, 1, 3, "vertical", 4),
        _member(2, 2, 4, "vertical", 4),
    ]
    member_results = [
        {"member_id": 1, "N_N": -1200.0},
        {"member_id": 2, "N_N": 0.0},
    ]
    member_checks = [
        {"member_id": 1, "FS_min": 0.9, "governing_mode": "compression_buckling", "utilization": 0.95},
        {"member_id": 2, "FS_min": 8.0, "governing_mode": "tension_capacity", "utilization": 0.01},
    ]

    planner = ActiveDesignPlanner()
    plan = planner.build_member_sizing_plan(cfg, nodes, members, member_results, member_checks)
    cfg2 = planner.apply_member_sizing_plan(cfg, plan)

    assert plan[1].action == "reinforce"
    assert plan[2].action in {"lighten", "simplify_joint"}
    assert int(cfg2["member_sticks_by_id"]["1"]) >= 4
    assert int(cfg2["member_sticks_by_id"]["2"]) >= 2
    assert int(cfg2["member_sticks_by_group"]["vertical"]) == 4


def test_cache_key_changes_when_member_sticks_by_id_changes(base_cfg: dict) -> None:
    planner = ActiveDesignPlanner()
    cfg_a = base_cfg
    cfg_b = {**base_cfg, "member_sticks_by_id": {"10": 2}}

    key_a = planner._cfg_cache_key(cfg_a)
    key_b = planner._cfg_cache_key(cfg_b)
    assert key_a != key_b


def test_score_prefers_stronger_solution_within_mass(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["analysis"]["planner_objective_profile"] = "balanced"
    cfg["analysis"]["target_min_fs"] = 2.0
    cfg.setdefault("planner", {})["max_bridge_mass_g"] = 1000.0
    cfg["bridge"]["load_total_N"] = float(cfg["bridge"]["load_total_kgf"]) * 9.80665

    planner = ActiveDesignPlanner()
    weak = {
        "min_fs_primary": 1.1,
        "estimated_breaking_load_kgf": 100.0,
        "mass_g": 650.0,
        "solver_status": "regular",
        "min_support_fs": 1.2,
        "equilibrium_error_N": 0.0,
        "inactive_support_count": 0,
        "weak_glue_joints": 0,
        "detailed": {"splice_stagger_report": {"critical_clusters": 0}},
    }
    strong = {
        "min_fs_primary": 2.3,
        "estimated_breaking_load_kgf": 170.0,
        "mass_g": 780.0,
        "solver_status": "regular",
        "min_support_fs": 1.2,
        "equilibrium_error_N": 0.0,
        "inactive_support_count": 0,
        "weak_glue_joints": 0,
        "detailed": {"splice_stagger_report": {"critical_clusters": 0}},
    }
    weak_score = planner._score_candidate(cfg, weak)
    strong_score = planner._score_candidate(cfg, strong)
    assert strong_score > weak_score
