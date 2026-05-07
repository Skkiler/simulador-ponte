"""Mass constraints and compliance utilities.

Key semantics:
- ``competition_mass_g`` is the final bridge mass criterion (installed sticks + cured glue).
- Procurement/cutting masses are informational and must not drive rejection when
  competition mass is available.
"""

from __future__ import annotations

from typing import Any, Dict

from src.core.numeric import safe_float


def resolve_mass_limits(cfg: Dict[str, Any], *, nominal_limit_g: float = 1000.0) -> Dict[str, float | str | None]:
    """Resolve nominal/material/planner/effective competition mass limits."""
    planner = cfg.get("planner", {}) or {}
    material = cfg.get("material", {}) or {}

    planner_limit = safe_float(planner.get("max_bridge_mass_g"), None)
    material_limit = safe_float(material.get("mass_limit_g"), None)
    nominal_cfg = safe_float(material.get("nominal_competition_limit_g"), None)
    nominal_limit = max(1.0, nominal_cfg if nominal_cfg is not None else float(nominal_limit_g))

    candidates = [x for x in (planner_limit, material_limit) if x is not None]
    if candidates:
        effective = min(candidates)
        if planner_limit is not None and material_limit is not None:
            source = "min(planner,material)"
        elif planner_limit is not None:
            source = "planner"
        else:
            source = "material"
    else:
        effective = nominal_limit
        source = "nominal"

    return {
        "nominal_limit_g": nominal_limit,
        "planner_limit_g": planner_limit,
        "material_limit_g": material_limit,
        "effective_limit_g": float(effective),
        "effective_source": source,
    }


def effective_mass_limit_g(cfg: Dict[str, Any]) -> float:
    """Return effective competition mass limit in grams."""
    return float(resolve_mass_limits(cfg)["effective_limit_g"])


def _resolve_budget_defaults(cfg: Dict[str, Any]) -> Dict[str, float]:
    mat = cfg.get("material", {}) or {}
    planner = cfg.get("planner", {}) or {}

    stick_budget = safe_float(mat.get("stick_budget_g"), None)
    if stick_budget is None:
        stick_budget = safe_float(planner.get("target_installed_stick_mass_g"), 900.0)

    wet_glue_budget = safe_float(mat.get("wet_glue_budget_g"), None)
    if wet_glue_budget is None:
        wet_glue_budget = safe_float(planner.get("target_wet_glue_mass_g"), 100.0)

    return {
        "stick_budget_g": max(1.0, float(stick_budget or 900.0)),
        "wet_glue_budget_g": max(1.0, float(wet_glue_budget or 100.0)),
    }


def _extract_competition_mass(row_or_metrics: Dict[str, Any]) -> tuple[float | None, str]:
    comp = safe_float(row_or_metrics.get("competition_mass_g"), None)
    if comp is not None:
        return comp, "competition_mass_g"

    for key in ("mass_g", "estimated_total_mass_g", "mass"):
        val = safe_float(row_or_metrics.get(key), None)
        if val is not None:
            return val, key
    return None, ""


def assert_mass_compliant(
    row_or_metrics: Dict[str, Any],
    cfg: Dict[str, Any],
    source: str = "",
) -> Dict[str, Any]:
    """Annotate competition/stick/glue mass compliance flags."""
    comp_mass, comp_source = _extract_competition_mass(row_or_metrics)
    if comp_mass is None:
        return row_or_metrics

    limits = resolve_mass_limits(cfg)
    limit = float(limits["effective_limit_g"])
    tolerance = safe_float(cfg.get("analysis", {}).get("mass_tolerance_g"), 0.0) or 0.0
    budgets = _resolve_budget_defaults(cfg)

    installed_mass = safe_float(row_or_metrics.get("installed_stick_mass_g"), None)
    wet_glue_mass = safe_float(row_or_metrics.get("wet_glue_mass_g"), None)
    procurement_mass = safe_float(row_or_metrics.get("assembly_procurement_mass_g"), None)

    competition_mass_compliant = comp_mass <= limit + tolerance
    stick_budget_compliant = (
        True
        if installed_mass is None
        else installed_mass <= budgets["stick_budget_g"] + tolerance
    )
    wet_glue_budget_compliant = (
        True
        if wet_glue_mass is None
        else wet_glue_mass <= budgets["wet_glue_budget_g"] + tolerance
    )

    row_or_metrics["mass_reference_g"] = comp_mass
    row_or_metrics["mass_reference_source"] = comp_source
    row_or_metrics["mass_compliant"] = competition_mass_compliant
    row_or_metrics["competition_mass_compliant"] = competition_mass_compliant
    row_or_metrics["stick_budget_compliant"] = stick_budget_compliant
    row_or_metrics["wet_glue_budget_compliant"] = wet_glue_budget_compliant
    row_or_metrics["procurement_mass_warning_only"] = (
        procurement_mass is not None and procurement_mass > limit + tolerance
    )
    row_or_metrics["stick_budget_g"] = budgets["stick_budget_g"]
    row_or_metrics["wet_glue_budget_g"] = budgets["wet_glue_budget_g"]
    row_or_metrics["mass_limit_effective_g"] = limit
    row_or_metrics["mass_limit_nominal_g"] = float(limits["nominal_limit_g"])
    row_or_metrics["mass_limit_planner_g"] = limits["planner_limit_g"]
    row_or_metrics["mass_limit_material_g"] = limits["material_limit_g"]
    row_or_metrics["mass_margin_g"] = limit - comp_mass
    if installed_mass is not None:
        row_or_metrics["stick_budget_margin_g"] = budgets["stick_budget_g"] - installed_mass
    if wet_glue_mass is not None:
        row_or_metrics["wet_glue_budget_margin_g"] = budgets["wet_glue_budget_g"] - wet_glue_mass
    return row_or_metrics
