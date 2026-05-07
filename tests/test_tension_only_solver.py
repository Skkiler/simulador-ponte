from __future__ import annotations

from src.domain.models import Load, Member, Node, Support
from src.services.postprocessor import PostProcessor
from src.services.section_service import SectionService
from src.solvers.linear_truss_solver import LinearTrussSolver


def _member(mid: int, i: int, j: int, group: str, nodes: dict[int, Node]) -> Member:
    mat = {
        "stick_width_mm": 7.0,
        "stick_thickness_mm": 1.5,
    }
    sec = SectionService().composite_section(1, mat, {"layout": "stacked"})
    L = SectionService.member_length_mm(nodes[i], nodes[j])
    return Member(
        mid,
        i,
        j,
        group,
        1,
        sec["A"],
        sec["A"],
        sec["A"],
        sec["Iy"],
        sec["Iz"],
        sec["J"],
        6000.0,
        500.0,
        1.0,
        1.0,
        L,
    )


def _base_cfg() -> dict:
    return {
        "bridge": {
            "unilateral_supports": True,
            "tension_only_bracing_interpretation": True,
            "tension_only_bracing_solver_enabled": True,
        },
        "analysis": {
            "primary_groups": ["bottom_chord"],
            "stabilizer_groups": ["top_bracing"],
            "tension_only_groups": ["top_bracing"],
        },
        "material": {
            "stick_length_mm": 115.0,
            "stick_width_mm": 7.0,
            "stick_thickness_mm": 1.5,
            "E_MPa": 6000.0,
            "compression_capacity_one_stick_N": 4.0 * 9.80665,
            "compression_capacity_two_sticks_N": 11.0 * 9.80665,
            "tension_capacity_per_stick_N": 72.0 * 9.80665,
        },
        "detail_model": {"overlap_length_mm": 30.0},
    }


def test_tension_only_member_is_released_and_solver_converges() -> None:
    nodes = {
        1: Node(1, 0.0, 0.0, 0.0, "bottom", "R", 0.0),
        2: Node(2, 100.0, 0.0, 0.0, "bottom", "R", 100.0),
    }
    members = [
        _member(1, 1, 2, "bottom_chord", nodes),
        _member(2, 1, 2, "top_bracing", nodes),
    ]
    supports = [
        Support(1, 1, 1, 1, 0, 0, 0, "left", True),
        Support(2, 0, 1, 1, 0, 0, 0, "right", True),
    ]
    loads = [Load("LC1", 2, -120.0, 0.0, 0.0)]

    solver = LinearTrussSolver()
    result = solver.solve(
        list(nodes.values()),
        members,
        supports,
        loads,
        unilateral_supports=True,
        tension_only_solver_enabled=True,
        tension_only_groups=["top_bracing"],
    )
    assert 2 in result.inactive_tension_only_member_ids
    assert result.tension_only_iterations >= 1
    assert "tension_only_converged" in result.status
    checks = PostProcessor().check_members(_base_cfg(), result.member_results)
    released = [r for r in checks if bool(r.get("tension_only_released"))]
    assert released
    assert all(r.get("design_relevant") is False for r in released)


def test_tension_only_removal_can_report_instability() -> None:
    cfg = _base_cfg()
    nodes = {
        1: Node(1, 0.0, 0.0, 0.0, "bottom", "R", 0.0),
        2: Node(2, 100.0, 0.0, 0.0, "bottom", "R", 100.0),
    }
    members = [_member(2, 1, 2, "top_bracing", nodes)]
    supports = [
        Support(1, 1, 1, 1, 0, 0, 0, "left", True),
        Support(2, 0, 1, 1, 0, 0, 0, "right", True),
    ]
    loads = [Load("LC1", 2, -120.0, 0.0, 0.0)]

    solver = LinearTrussSolver()
    result = solver.solve(
        list(nodes.values()),
        members,
        supports,
        loads,
        unilateral_supports=True,
        tension_only_solver_enabled=True,
        tension_only_groups=["top_bracing"],
    )
    assert "tension_only_singular" in result.status or result.instability_due_to_tension_only_bracing

    # Em fallback singular o solver pode reativar barras e ainda assim reportar
    # instabilidade; o requisito é reportar o estado, não mascarar como regular.
    assert result.instability_due_to_tension_only_bracing or "tension_only_singular" in result.status
