from __future__ import annotations

from src.services.mass_guard import assert_mass_compliant, effective_mass_limit_g, resolve_mass_limits


def test_resolve_mass_limits_and_effective_limit(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["planner"]["max_bridge_mass_g"] = 930.0
    cfg["material"]["mass_limit_g"] = 1000.0

    limits = resolve_mass_limits(cfg)
    assert limits["nominal_limit_g"] == 1000.0
    assert limits["planner_limit_g"] == 930.0
    assert limits["material_limit_g"] == 1000.0
    assert limits["effective_limit_g"] == 930.0
    assert effective_mass_limit_g(cfg) == 930.0


def test_assert_mass_compliant_annotates_limit_fields(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["planner"]["max_bridge_mass_g"] = 900.0
    row = {"mass_g": 901.0}
    out = assert_mass_compliant(row, cfg, source="unit_test")
    assert out["mass_compliant"] is False
    assert out["mass_limit_effective_g"] == 900.0
    assert out["mass_limit_nominal_g"] == 1000.0
