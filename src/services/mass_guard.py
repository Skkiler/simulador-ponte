"""
Utility functions for enforcing a global effective mass limit across the system.

The simulator permits multiple configuration sections to specify a mass limit.  In
practice this led to subtle divergence between the planner, material model,
report generator and other modules.  To avoid the situation where a design
passes a limit in one place but is rejected in another, this module provides
two small helpers:

* ``effective_mass_limit_g(cfg)`` resolves a single mass limit for a given
  configuration.  It inspects both ``cfg['planner']['max_bridge_mass_g']``
  and ``cfg['material']['mass_limit_g']`` and returns whichever is more
  restrictive (i.e. the lower value).  When only one limit is present the
  defined value is returned.  If neither is specified a sensible default of
  1000 grams is used.

* ``assert_mass_compliant(row_or_metrics, cfg, source)`` inspects a metrics
  dictionary or row dictionary and annotates it with a boolean flag
  ``mass_compliant``.  If a numeric ``mass_g`` or ``estimated_total_mass_g``
  field exceeds the effective mass limit plus an optional tolerance, the flag
  is set to ``False``.  Otherwise it is set to ``True``.  An optional tolerance
  may be provided via ``cfg['analysis']['mass_tolerance_g']``; when not
  defined it defaults to zero.  Consumers can use this flag to prevent
  overweight designs from being promoted to recommended configurations.

These functions are deliberately free of side effects.  They do not raise
exceptions, log or emit progress, but simply compute and annotate data.  This
allows them to be used in tight loops (e.g. during candidate evaluation) with
minimal overhead.
"""

from __future__ import annotations

from typing import Any, Dict

from src.core.numeric import safe_float


def resolve_mass_limits(cfg: Dict[str, Any], *, nominal_limit_g: float = 1000.0) -> Dict[str, float | str | None]:
    """Resolve nominal/material/planner/effective mass limits for one config."""
    planner = cfg.get("planner", {}) or {}
    material = cfg.get("material", {}) or {}

    planner_limit = safe_float(planner.get("max_bridge_mass_g"), None)
    material_limit = safe_float(material.get("mass_limit_g"), None)
    nominal_limit = max(1.0, float(nominal_limit_g))

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
    """Determine the global mass limit in grams for a configuration.

    The planner section (``cfg['planner']['max_bridge_mass_g']``) and the
    material section (``cfg['material']['mass_limit_g']``) may both define
    mass limits.  If both are present the smaller of the two is returned
    because exceeding either should render a design non‑compliant.  If only
    one is present it is returned.  When neither is specified a default of
    1000 grams (the typical competition limit) is assumed.
    """
    return float(resolve_mass_limits(cfg)["effective_limit_g"])


def assert_mass_compliant(row_or_metrics: Dict[str, Any], cfg: Dict[str, Any], source: str = "") -> Dict[str, Any]:
    """Annotate a metrics or row dict with a ``mass_compliant`` flag.

    The function reads ``mass_g`` or ``estimated_total_mass_g`` from the
    provided dictionary.  If neither field is present or cannot be converted
    to a float the check is skipped and the dictionary is returned unchanged.
    Otherwise the value is compared against the effective mass limit (plus an
    optional tolerance).  The tolerance may be supplied in
    ``cfg['analysis']['mass_tolerance_g']`` and defaults to zero.  A new
    boolean key ``mass_compliant`` is inserted into the dictionary
    indicating whether the mass is within the permitted limit.  Callers
    should use this flag to reject overweight designs during candidate
    selection.

    ``source`` is an optional string used for debugging; it is ignored by
    this function but may be logged by callers if desired.
    """
    # Extract a candidate mass from known fields
    mass = None
    for key in ("mass_g", "estimated_total_mass_g", "mass"):
        val = row_or_metrics.get(key)
        mass = safe_float(val, None)
        if mass is not None:
            break
    if mass is None:
        # cannot determine mass; leave untouched
        return row_or_metrics
    limits = resolve_mass_limits(cfg)
    limit = float(limits["effective_limit_g"])
    tolerance = safe_float(cfg.get("analysis", {}).get("mass_tolerance_g"), 0.0) or 0.0
    row_or_metrics["mass_compliant"] = (mass <= limit + tolerance)
    row_or_metrics["mass_limit_effective_g"] = limit
    row_or_metrics["mass_limit_nominal_g"] = float(limits["nominal_limit_g"])
    row_or_metrics["mass_limit_planner_g"] = limits["planner_limit_g"]
    row_or_metrics["mass_limit_material_g"] = limits["material_limit_g"]
    return row_or_metrics
