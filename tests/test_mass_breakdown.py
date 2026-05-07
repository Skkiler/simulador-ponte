from __future__ import annotations

from src.services.config_service import ConfigService
from src.services.geometry_service import GeometryService
from src.services.mass_guard import assert_mass_compliant
from src.services.postprocessor import PostProcessor
from src.services.stick_detail_service import StickDetailService
from src.solvers.linear_truss_solver import LinearTrussSolver


def _run_detail(base_cfg: dict, tmp_path) -> dict:
    cfg = ConfigService().normalize(base_cfg)
    nodes, members, supports, loads = GeometryService().generate(cfg)
    solver = LinearTrussSolver()
    result = solver.solve(
        nodes,
        members,
        supports,
        loads,
        unilateral_supports=bool(cfg["bridge"].get("unilateral_supports", True)),
        tension_only_solver_enabled=bool(cfg["bridge"].get("tension_only_bracing_solver_enabled", False)),
        tension_only_groups=cfg.get("analysis", {}).get("tension_only_groups", []),
    )
    checks = PostProcessor().check_members(cfg, result.member_results)
    detail = StickDetailService().analyze(
        cfg,
        nodes,
        members,
        result.member_results,
        checks,
        tmp_path / "details",
    )
    return detail


def test_competition_mass_excludes_cutting_scrap(base_cfg: dict, tmp_path) -> None:
    detail = _run_detail(base_cfg, tmp_path)
    s = detail["summary"]
    comp = float(s["competition_mass_g"])
    installed = float(s["installed_stick_mass_g"])
    cured = float(s["cured_glue_mass_g"])
    scrap = float(s["cutting_scrap_mass_g"])
    procurement = float(s["assembly_procurement_mass_g"])

    assert abs(comp - (installed + cured)) < 1.0e-6
    assert scrap >= 0.0
    assert procurement >= comp


def test_cured_glue_mass_uses_solids_fraction(base_cfg: dict, tmp_path) -> None:
    base_cfg["detail_model"]["glue_cure_solids_fraction"] = 0.50
    detail = _run_detail(base_cfg, tmp_path)
    s = detail["summary"]
    wet = float(s["wet_glue_mass_g"])
    cured = float(s["cured_glue_mass_g"])
    assert abs(cured - wet * 0.50) < 1.0e-6


def test_mass_guard_uses_competition_mass_when_available(base_cfg: dict) -> None:
    row = {
        "competition_mass_g": 980.0,
        "estimated_total_mass_g": 1200.0,
        "installed_stick_mass_g": 880.0,
        "wet_glue_mass_g": 90.0,
        "assembly_procurement_mass_g": 1290.0,
    }
    out = assert_mass_compliant(row, base_cfg, source="unit_test_mass_breakdown")
    assert out["mass_compliant"] is True
    assert out["competition_mass_compliant"] is True
    assert out["procurement_mass_warning_only"] is True


def test_installed_stick_budget_can_pass_with_high_procurement_mass(base_cfg: dict) -> None:
    row = {
        "competition_mass_g": 960.0,
        "installed_stick_mass_g": 890.0,
        "wet_glue_mass_g": 95.0,
        "purchased_stick_mass_g": 1300.0,
        "assembly_procurement_mass_g": 1395.0,
    }
    out = assert_mass_compliant(row, base_cfg, source="unit_test_mass_budget")
    assert out["mass_compliant"] is True
    assert out["stick_budget_compliant"] is True
