from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

from src.core.numeric import safe_float
from src.domain.models import Load
from src.services.load_distribution_service import LoadDistributionService
from src.services.geometry_service import GeometryService
from src.services.mass_guard import assert_mass_compliant, effective_mass_limit_g
from src.services.rupture_estimator import estimate_rupture_load


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))

INVALID_OBJECTIVE = -1.0e12
_BAD_SOLVER_STATUS_TOKENS = (
    "singular",
    "mechanism",
    "instability",
    "rank_",
    "rank deficient",
    "tension_only_singular",
)

class StagedFidelityFunnelPlanner:
    """Executa o funil S0..S8 com fidelidade crescente e cortes de custo."""

    def __init__(self, planner: Any) -> None:
        self.planner = planner
        self._case_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._stage_solves: Dict[str, int] = {}
        self._detailed_mass_cache: Dict[str, float | None] = {}

    @staticmethod
    def _hash_payload(payload: Any) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()

    def _signature_hashes(self, cfg: Dict[str, Any], load_case_name: str) -> Dict[str, str]:
        bridge = cfg.get("bridge", {}) or {}
        geometry_payload = {
            "span_mm": bridge.get("span_mm"),
            "width_mm": bridge.get("width_mm"),
            "center_height_mm": bridge.get("center_height_mm"),
            "end_height_mm": bridge.get("end_height_mm"),
            "panel_mm": bridge.get("panel_mm"),
            "top_profile": bridge.get("top_profile"),
            "plateau_start_mm": bridge.get("plateau_start_mm"),
            "plateau_end_mm": bridge.get("plateau_end_mm"),
            "support_contact_x_left_mm": bridge.get("support_contact_x_left_mm"),
            "support_contact_x_right_mm": bridge.get("support_contact_x_right_mm"),
            "support_contact_y_mm": bridge.get("support_contact_y_mm"),
            "load_distribution_x_mm": bridge.get("load_distribution_x_mm"),
            "load_application_level": bridge.get("load_application_level"),
        }
        topology_payload = {
            "side_truss_type": bridge.get("side_truss_type"),
            "internal_truss_type": bridge.get("internal_truss_type"),
            "top_chord_truss_type": bridge.get("top_chord_truss_type"),
            "bottom_chord_truss_type": bridge.get("bottom_chord_truss_type"),
            "chord_truss_type": bridge.get("chord_truss_type"),
            "include_top_x_bracing": bridge.get("include_top_x_bracing", True),
            "include_bottom_x_bracing": bridge.get("include_bottom_x_bracing", True),
            "include_cross_frame_bracing": bridge.get("include_cross_frame_bracing", True),
            "include_support_pad_members": bridge.get("include_support_pad_members", True),
            "panel_side_truss_pattern": bridge.get("panel_side_truss_pattern", {}) or {},
            "panel_top_chord_pattern": bridge.get("panel_top_chord_pattern", {}) or {},
            "panel_bottom_chord_pattern": bridge.get("panel_bottom_chord_pattern", {}) or {},
            "member_active_by_id": cfg.get("member_active_by_id", {}) or {},
            "disabled_member_ids": cfg.get("disabled_member_ids", []) or [],
        }
        sizing_payload = {
            "member_sticks_by_group": cfg.get("member_sticks_by_group", {}) or {},
            "member_sticks_by_id": cfg.get("member_sticks_by_id", {}) or {},
            "section_layout_by_group": cfg.get("section_layout_by_group", {}) or {},
            "effective_length_factor_by_group": cfg.get("effective_length_factor_by_group", {}) or {},
        }
        load_case_payload = {
            "case": str(load_case_name),
            "load_total_N": bridge.get("load_total_N"),
            "load_total_kgf": bridge.get("load_total_kgf"),
            "lateral_factor": cfg.get("multi_loadcase_screening", {}).get("lateral_imperfection_factor", 0.02),
            "self_weight_factor": cfg.get("multi_loadcase_screening", {}).get("self_weight_factor", 1.0),
        }
        return {
            "geometry_hash": self._hash_payload(geometry_payload),
            "topology_hash": self._hash_payload(topology_payload),
            "sizing_hash": self._hash_payload(sizing_payload),
            "load_case_hash": self._hash_payload(load_case_payload),
        }

    def _mark_solve(self, stage_name: str) -> None:
        self._stage_solves[stage_name] = int(self._stage_solves.get(stage_name, 0)) + 1

    def _cache_key(self, cfg: Dict[str, Any], load_case_name: str, tension_only: bool) -> str:
        sig = self._signature_hashes(cfg, load_case_name)
        payload = {
            **sig,
            "tension_only": bool(tension_only),
        }
        return self._hash_payload(payload)

    @staticmethod
    def _solver_regular(status: Any) -> bool:
        status_text = str(status or "").lower()
        if any(tok in status_text for tok in _BAD_SOLVER_STATUS_TOKENS):
            return False
        return status_text.split("|", 1)[0] == "regular"

    @staticmethod
    def _is_selectable_case(case_row: Dict[str, Any]) -> bool:
        status_text = str(case_row.get("solver_status", "")).lower()
        if any(tok in status_text for tok in _BAD_SOLVER_STATUS_TOKENS):
            return False
        if not bool(case_row.get("solver_regular")):
            return False
        if not bool(case_row.get("equilibrium_ok")):
            return False
        if (safe_float(case_row.get("topology_stability_proxy"), 0.0) or 0.0) <= 0.0:
            return False
        return True

    def _is_selectable_summary(self, summary: Dict[str, Any]) -> bool:
        cases = summary.get("cases") or []
        if not cases:
            return False
        if not bool(summary.get("solver_regular")):
            return False
        if not bool(summary.get("equilibrium_ok")):
            return False
        if (safe_float(summary.get("topology_stability_proxy"), 0.0) or 0.0) <= 0.0:
            return False
        return all(self._is_selectable_case(c) for c in cases)

    @staticmethod
    def _summary_valid_flag(summary: Dict[str, Any]) -> bool:
        if "valid_for_selection" in summary:
            return bool(summary.get("valid_for_selection"))
        return (
            bool(summary.get("solver_regular"))
            and bool(summary.get("equilibrium_ok"))
            and (safe_float(summary.get("topology_stability_proxy"), 0.0) or 0.0) > 0.0
        )

    def _detailed_competition_mass_estimate(
        self,
        cfg: Dict[str, Any],
        *,
        case_name: str = "center",
        stage_name: str = "S6_DETAILED_MASS",
        tension_only: bool = False,
    ) -> float | None:
        """Estimate final competition mass using the detailed cut/glue model.

        The fast mass proxy deliberately overestimates some long spliced members
        because it cannot reuse real cut pieces.  That is useful in early stages,
        but near the final limit it can reject high-yield reinforcements even when
        the detailed competition mass is still below 1 kg.  This helper keeps the
        proxy conservative while allowing late-stage decisions to verify the real
        detailed mass before rejecting a trial.
        """
        cache_key = self._hash_payload({
            "mass_sig": self._signature_hashes(cfg, str(case_name)),
            "case": str(case_name),
            "tension_only": bool(tension_only),
        })
        if cache_key in self._detailed_mass_cache:
            return self._detailed_mass_cache[cache_key]

        try:
            case = self._evaluate_case_cached(
                cfg,
                str(case_name),
                stage_name=stage_name,
                tension_only=tension_only,
            )
            with tempfile.TemporaryDirectory(prefix="bridge_mass_check_") as tmp:
                detail = self.planner.detail.analyze(
                    cfg,
                    case.get("nodes") or [],
                    case.get("members") or [],
                    case.get("member_results") or [],
                    case.get("member_checks") or [],
                    tmp,
                )
            summary = detail.get("summary", {}) if isinstance(detail, dict) else {}
            value = safe_float(
                summary.get("competition_mass_g"),
                safe_float(summary.get("estimated_total_mass_g"), None),
            )
            result = float(value) if value is not None else None
            self._detailed_mass_cache[cache_key] = result
            return result
        except Exception:
            self._detailed_mass_cache[cache_key] = None
            return None

    def _late_stage_mass_ok(
        self,
        cfg: Dict[str, Any],
        *,
        proxy_mass_g: float,
        proxy_limit_g: float,
        hard_limit_g: float,
        reserve_g: float,
        stage_name: str,
        tension_only: bool,
    ) -> tuple[bool, float | None, str]:
        """Check late-stage mass using proxy first and detailed mass as fallback."""
        proxy_mass_g = float(proxy_mass_g)
        proxy_limit_g = float(proxy_limit_g)
        hard_limit_g = float(hard_limit_g)
        reserve_g = max(0.0, float(reserve_g))
        if proxy_mass_g <= proxy_limit_g + 1.0e-9:
            return True, None, "proxy"

        ms = cfg.get("member_sizing", {}) or {}
        max_proxy_overrun = float(ms.get("late_stage_detailed_mass_max_proxy_overrun_g", 8.0))
        if proxy_mass_g - proxy_limit_g > max_proxy_overrun + 1.0e-9:
            return False, None, "proxy_overrun_too_large"

        detailed_mass = self._detailed_competition_mass_estimate(
            cfg,
            stage_name=stage_name,
            tension_only=tension_only,
        )
        if detailed_mass is None:
            return False, None, "proxy_over_and_detailed_unavailable"
        if detailed_mass <= hard_limit_g - reserve_g + 1.0e-9:
            return True, detailed_mass, "detailed"
        return False, detailed_mass, "detailed_over_limit"

    @staticmethod
    def _use_tension_only_for_stage(cfg: Dict[str, Any], stage_name: str) -> bool:
        analysis = cfg.get("analysis", {}) or {}
        bridge = cfg.get("bridge", {}) or {}

        groups = [
            str(g)
            for g in (analysis.get("tension_only_groups") or [])
            if str(g).strip()
        ]

        if not groups:
            return False

        # Para ponte de palito, tension-only não deve ser default.
        # Só ativa se o usuário declarar explicitamente que quer esse modelo.
        enabled = bool(
            analysis.get(
                "enable_tension_only_solver_in_funnel",
                bridge.get("tension_only_bracing_solver_enabled", False),
            )
        )

        if not enabled:
            return False

        stage_flags = analysis.get("tension_only_stages", {}) or {}

        if stage_flags:
            return bool(stage_flags.get(str(stage_name), False))

        return False

    @staticmethod
    def _min_positive(values: Iterable[Any], default: float = 0.0) -> float:
        cleaned = []
        for value in values:
            v = safe_float(value, None)
            if v is not None and math.isfinite(float(v)) and float(v) > 1.0e-12:
                cleaned.append(float(v))
        return min(cleaned) if cleaned else float(default)

    @staticmethod
    def _validate_dimension(
        violations: List[str],
        label: str,
        value: float,
        *,
        required: float | None,
        tolerance: float,
        unit: str = "mm",
    ) -> None:
        if value <= 0.0:
            violations.append(f"{label} inválido: {value:.2f} {unit}; deve ser > 0 {unit}.")
            return
        if required is not None and abs(value - required) > tolerance:
            violations.append(
                f"{label} inválido: {value:.2f} {unit}; edital exige "
                f"{required:.2f} ± {tolerance:.2f} {unit}."
            )

    @staticmethod
    def _span_xs_from_bridge(bridge: Dict[str, Any]) -> List[float]:
        span = float(bridge.get("span_mm", 1200.0))
        panel = max(1.0, float(bridge.get("panel_mm", 100.0)))
        left_overhang = float(bridge.get("left_support_overhang_mm", 100.0))
        right_overhang = float(bridge.get("right_support_overhang_mm", 100.0))
        xs = [-left_overhang]
        x = 0.0
        while x <= span + 1.0e-9:
            xs.append(round(x, 6))
            x += panel
        if abs(xs[-1] - span) > 1.0e-6:
            xs.append(round(span, 6))
        xs.append(round(span + right_overhang, 6))
        return sorted(set(xs))

    @classmethod
    def _span_panel_indices(cls, bridge: Dict[str, Any]) -> List[Tuple[int, float, float]]:
        span = float(bridge.get("span_mm", 1200.0))
        xs = cls._span_xs_from_bridge(bridge)
        return [
            (idx, float(x0), float(x1))
            for idx, (x0, x1) in enumerate(zip(xs[:-1], xs[1:]))
            if x0 >= -1.0e-6 and x1 <= span + 1.0e-6
        ]

    @classmethod
    def _make_symmetric_span_pattern(cls, bridge: Dict[str, Any]) -> Dict[str, str]:
        span_panels = cls._span_panel_indices(bridge)
        n = len(span_panels)
        pattern: Dict[str, str] = {}
        for local_i in range((n + 1) // 2):
            mirror_i = n - 1 - local_i
            mode = "Pratt_symmetric" if local_i % 2 == 0 else "Warren_symmetric"
            if local_i == n // 2 and n % 2 == 1:
                mode = "Warren_mid_braced"
            pattern[str(span_panels[local_i][0])] = mode
            pattern[str(span_panels[mirror_i][0])] = mode
        return pattern

    @staticmethod
    def _max_displacement(node_results: List[Dict[str, Any]]) -> float:
        max_u = 0.0
        for r in node_results or []:
            ux = safe_float(r.get("Ux_mm"), 0.0) or 0.0
            uy = safe_float(r.get("Uy_mm"), 0.0) or 0.0
            uz = safe_float(r.get("Uz_mm"), 0.0) or 0.0
            u = math.sqrt(ux * ux + uy * uy + uz * uz)
            if u > max_u:
                max_u = u
        return max_u

    @staticmethod
    def _support_reaction_balance(support_checks: List[Dict[str, Any]]) -> float:
        left = sum(
            float(safe_float(r.get("reaction_Z_N"), 0.0) or 0.0)
            for r in support_checks
            if str(r.get("support_group", "")).lower() == "left"
            and bool(r.get("support_active_vertical", True))
        )
        right = sum(
            float(safe_float(r.get("reaction_Z_N"), 0.0) or 0.0)
            for r in support_checks
            if str(r.get("support_group", "")).lower() == "right"
            and bool(r.get("support_active_vertical", True))
        )
        den = max(1.0e-9, abs(left) + abs(right))
        return float(_clamp(1.0 - abs(left - right) / den, 0.0, 1.0))

    @staticmethod
    def _load_path_score(member_results: List[Dict[str, Any]], tol_N: float = 2.0) -> float:
        if not member_results:
            return 0.0
        active = sum(1 for r in member_results if abs(safe_float(r.get("N_N"), 0.0) or 0.0) > tol_N)
        return float(active / max(1, len(member_results)))

    @staticmethod
    def _topology_complexity_penalty(member_count: int) -> float:
        # Penaliza crescimento acima de uma faixa prática para montagem manual.
        return max(0.0, (float(member_count) - 420.0) / 280.0)

    @staticmethod
    def _sync_bridge_contacts(bridge: Dict[str, Any]) -> None:
        span = float(bridge.get("span_mm", 1200.0))
        width = float(bridge.get("width_mm", 160.0))
        panel = max(1.0, float(bridge.get("panel_mm", 100.0)))
        left_overhang = float(bridge.get("left_support_overhang_mm", 100.0))
        right_overhang = float(bridge.get("right_support_overhang_mm", 100.0))

        half_w = width * 0.5
        bridge["support_contact_y_mm"] = [-half_w, half_w]
        bridge["support_contact_x_left_mm"] = [-left_overhang, 0.0]
        bridge["support_contact_x_right_mm"] = [span, span + right_overhang]

        if not bridge.get("load_distribution_x_mm"):
            p0 = max(0.0, min(float(bridge.get("plateau_start_mm", span / 3.0)), span))
            p1 = max(p0, min(float(bridge.get("plateau_end_mm", 2.0 * span / 3.0)), span))
            xs: List[float] = []
            x = p0
            while x <= p1 + 1.0e-9:
                xs.append(round(x, 6))
                x += panel
            bridge["load_distribution_x_mm"] = xs or [round(span * 0.5, 6)]

    def _build_loads_for_case(
        self,
        cfg: Dict[str, Any],
        nodes: List[Any],
        load_case_name: str,
        quick_mass_g: float,
    ) -> List[Load]:
        bridge = cfg.get("bridge", {}) or {}
        span = float(bridge.get("span_mm", 1200.0))
        total_N = abs(float(bridge.get("load_total_N", 0.0)))

        offset_fraction = float(
            (cfg.get("multi_loadcase_screening", {}) or {}).get(
                "offset_fraction_of_span",
                0.05,
            )
        )
        offset_delta = offset_fraction * span
        case = str(load_case_name)
        loads: List[Load] = []

        if case == "center":
            return LoadDistributionService.build_nodal_loads(
                cfg,
                nodes,
                loadcase=case,
                total_N=total_N,
            )

        if case in {"single_plate_center", "physical_plate_center"}:
            # One real plate/platen footprint centered on the configured centroid.
            # This is the physically correct counterpart to the legacy multi-patch
            # station list.  It exposes whether an apparent high score depended on
            # unrealistically spreading the load over several independent patches.
            plate_cfg = copy.deepcopy(cfg)
            plate_bridge = plate_cfg.setdefault("bridge", {})
            plate_bridge["load_distribution_model"] = "plate_surface_uniform"
            plate_bridge["load_footprint_interpretation"] = "centroid"
            return LoadDistributionService.build_nodal_loads(
                plate_cfg,
                nodes,
                loadcase=case,
                total_N=total_N,
            )

        if case in {"crown_contact", "loose_weight_crown_contact"}:
            return LoadDistributionService.build_crown_contact_loads(
                cfg,
                nodes,
                loadcase=case,
                total_N=total_N,
            )

        if case == "left_offset":
            return LoadDistributionService.build_nodal_loads(
                cfg,
                nodes,
                loadcase=case,
                total_N=total_N,
                x_targets=LoadDistributionService.shifted_targets(cfg, -offset_delta),
            )

        if case == "right_offset":
            return LoadDistributionService.build_nodal_loads(
                cfg,
                nodes,
                loadcase=case,
                total_N=total_N,
                x_targets=LoadDistributionService.shifted_targets(cfg, offset_delta),
            )

        torsion_match = re.fullmatch(r"torsion_(\d{1,3})_(\d{1,3})", str(case))
        if torsion_match:
            left_raw = float(torsion_match.group(1))
            right_raw = float(torsion_match.group(2))
            total_bias = max(1.0e-9, left_raw + right_raw)
            return LoadDistributionService.build_nodal_loads(
                cfg,
                nodes,
                loadcase=case,
                total_N=total_N,
                side_bias={"left": left_raw / total_bias, "right": right_raw / total_bias},
            )

        if case == "lateral_imperfection":
            lateral_factor = float(
                (cfg.get("multi_loadcase_screening", {}) or {}).get(
                    "lateral_imperfection_factor",
                    0.02,
                )
            )
            return LoadDistributionService.build_nodal_loads(
                cfg,
                nodes,
                loadcase=case,
                total_N=total_N,
                lateral_factor=lateral_factor,
            )

        if case == "self_weight":
            gN = max(0.0, float(quick_mass_g) / 1000.0 * 9.80665)

            if gN <= 0.0:
                gN = 0.08 * total_N

            all_struct_nodes = [
                n
                for n in nodes
                if getattr(n, "level", "") in {"top", "bottom"}
            ]

            fz_each = -gN / max(1, len(all_struct_nodes))

            for n in all_struct_nodes:
                loads.append(Load(case, int(n.id), 0.0, 0.0, fz_each))

            return loads

        return LoadDistributionService.build_nodal_loads(
            cfg,
            nodes,
            loadcase=case,
            total_N=total_N,
        )
    
    def _evaluate_case_cached(
        self,
        cfg: Dict[str, Any],
        load_case_name: str,
        *,
        stage_name: str,
        tension_only: bool,
    ) -> Dict[str, Any]:
        key = self._cache_key(cfg, load_case_name, tension_only)
        cached = self._case_cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return copy.deepcopy(cached)

        self._cache_misses += 1
        self._mark_solve(stage_name)

        nodes, members, supports, _ = self.planner.geometry.generate(cfg)
        quick_mass_g, _ = self.planner._quick_mass_estimate(cfg, members)
        loads = self._build_loads_for_case(cfg, nodes, load_case_name, quick_mass_g)

        kwargs = dict(self.planner._solver_kwargs_from_cfg(cfg))
        kwargs["tension_only_solver_enabled"] = bool(tension_only)

        result = self.planner.solver.solve(nodes, members, supports, loads, **kwargs)

        active_supports = [
            type(s)(
                s.node_id,
                s.UX,
                s.UY,
                s.UZ if s.node_id in result.active_support_node_ids else 0,
                s.RX,
                s.RY,
                s.RZ,
                s.support_group,
                s.node_id in result.active_support_node_ids,
            )
            for s in supports
        ]

        member_checks = self.planner.post.check_members(cfg, result.member_results)
        support_checks = self.planner.post.check_supports(cfg, nodes, active_supports, result.node_results)

        solver_regular = self._solver_regular(result.status)
        total_load_N = abs(sum(float(l.Fz) for l in loads))
        eq_tol_N = max(1.0e-6, 0.005 * max(total_load_N, 1.0))
        equilibrium_ok = abs(float(result.equilibrium_error_N)) <= eq_tol_N

        min_fs_primary = self._min_positive(
            (r.get("FS_min") for r in member_checks if r.get("member_role") == "primary"),
            default=0.0,
        )
        min_fs_design = self._min_positive(
            (r.get("FS_design") for r in member_checks if r.get("design_relevant", True)),
            default=min_fs_primary,
        )

        rupture = estimate_rupture_load(
            cfg,
            member_checks,
            support_checks,
            None,
            max(0.0, total_load_N / 9.80665),
        )
        predicted_break = safe_float(
            rupture.get("predicted_breaking_load_design_kgf"),
            safe_float(rupture.get("predicted_breaking_load_kgf"), 0.0),
        ) or 0.0

        max_tension = max((safe_float(r.get("N_N"), 0.0) or 0.0 for r in result.member_results), default=0.0)
        max_compression = max((-(safe_float(r.get("N_N"), 0.0) or 0.0) for r in result.member_results), default=0.0)
        buckling_risk = max(
            [
                safe_float(r.get("buckling_util_y"), 0.0) or 0.0
                for r in member_checks
            ]
            + [
                safe_float(r.get("buckling_util_z"), 0.0) or 0.0
                for r in member_checks
            ]
            + [0.0]
        )
        near_zero_ids = [
            int(r.get("member_id"))
            for r in result.member_results
            if abs(safe_float(r.get("N_N"), 0.0) or 0.0) <= 2.0
        ]

        try:
            topo = self.planner.topology.validate(cfg, nodes, members, supports, loads)
            topology_ok = bool(topo.get("is_valid"))
        except (TypeError, ValueError, KeyError, RuntimeError):
            topology_ok = False

        row = {
            "case": str(load_case_name),
            "solver_status": result.status,
            "solver_regular": solver_regular,
            "equilibrium_error_N": float(result.equilibrium_error_N),
            "equilibrium_ok": equilibrium_ok,
            "equilibrium_tol_N": eq_tol_N,
            "min_fs_primary": min_fs_primary,
            "min_fs_design": min_fs_design,
            "inactive_tension_only_members": getattr(result, "inactive_tension_only_members", []),
            "inactive_supports_uplift": getattr(result, "inactive_supports_uplift", []),
            "active_support_node_ids": sorted(getattr(result, "active_support_node_ids", []) or []),
            "predicted_breaking_load_proxy_kgf": predicted_break,
            "max_displacement_proxy_mm": self._max_displacement(result.node_results),
            "max_compression_proxy_N": max_compression,
            "max_tension_proxy_N": max_tension,
            "buckling_risk_proxy": buckling_risk,
            "mass_proxy_g": quick_mass_g,
            "load_path_score": self._load_path_score(result.member_results, tol_N=2.0),
            "support_reaction_balance": self._support_reaction_balance(support_checks),
            "topology_stability_proxy": 1.0 if (solver_regular and topology_ok) else 0.0,
            "nodal_stability_proxy": 1.0 / max(1.0, self._max_displacement(result.node_results) / 5.0),
            "member_checks": member_checks,
            "support_checks": support_checks,
            "member_results": result.member_results,
            "node_results": result.node_results,
            "near_zero_member_ids": near_zero_ids,
            "quick_mass_g": quick_mass_g,
            "nodes": nodes,
            "members": members,
            "supports": supports,
            "loads": loads,
        }
        self._case_cache[key] = copy.deepcopy(row)
        return row

    def _objective_score(
        self,
        cfg: Dict[str, Any],
        *,
        predicted_breaking_load_kgf: float,
        min_fs_design: float,
        mass_g: float,
        max_displacement_mm: float,
        mechanism_penalty: float,
        topology_complexity_penalty: float,
        glue_overuse_penalty: float,
        constructability_score: float,
    ) -> float:
        analysis = cfg.get("analysis", {}) or {}
        target_break = max(1.0, float(analysis.get("acceptance_min_design_breaking_load_kgf", 80.0)))
        acceptance_min_fs = max(0.1, float(analysis.get("acceptance_min_primary_fs", 1.05)))
        mass_limit = max(1.0, float(effective_mass_limit_g(cfg)))

        strength_ratio = predicted_breaking_load_kgf / max(1.0, target_break)
        mass_ratio = mass_g / max(1.0, mass_limit)
        if mass_ratio > 1.20:
            return INVALID_OBJECTIVE

        overweight_penalty = max(0.0, mass_ratio - 1.0)

        strength_to_weight_score = predicted_breaking_load_kgf / max(1.0, mass_g)
        strength_to_weight_score = _clamp(
            strength_to_weight_score / max(0.01, target_break / mass_limit),
            0.0,
            2.0,
        )

        # Massa abaixo do limite só é bônus depois que a ponte está perto da meta.
        # Antes disso, massa sobrando é capacidade estrutural não usada.
        if strength_ratio >= 0.90:
            competition_mass_margin_score = _clamp((mass_limit - mass_g) / mass_limit, -1.0, 1.0)
            unused_mass_while_weak_penalty = 0.0
        else:
            competition_mass_margin_score = 0.0
            useful_mass_target_ratio = 0.92
            unused_mass_while_weak_penalty = max(
                0.0,
                useful_mass_target_ratio - mass_ratio,
            ) * (1.0 - _clamp(strength_ratio / 0.90, 0.0, 1.0))

        displacement_penalty = max(0.0, max_displacement_mm / 30.0 - 1.0)
        dead_weight_penalty = max(0.0, mass_g / mass_limit - 1.0)

        return (
            4.0 * _clamp(predicted_breaking_load_kgf / target_break, 0.0, 2.0)
            + 2.0 * _clamp(min_fs_design / acceptance_min_fs, 0.0, 2.0)
            + 1.5 * strength_to_weight_score
            + 1.0 * competition_mass_margin_score
            + 0.5 * constructability_score
            - 2.0 * unused_mass_while_weak_penalty
            - 2.0 * mechanism_penalty
            - 8.0 * overweight_penalty
            - 1.5 * displacement_penalty
            - 1.0 * dead_weight_penalty
            - 1.0 * glue_overuse_penalty
            - 1.0 * topology_complexity_penalty
        )

    def _multi_case_summary(
        self,
        cfg: Dict[str, Any],
        load_cases: Iterable[str],
        *,
        stage_name: str,
        tension_only: bool,
    ) -> Dict[str, Any]:
        cases = []

        for case_name in load_cases:
            cases.append(
                self._evaluate_case_cached(
                    cfg,
                    str(case_name),
                    stage_name=stage_name,
                    tension_only=tension_only,
                )
            )

        if not cases:
            return {
                "solver_regular": False,
                "equilibrium_ok": False,
                "objective": INVALID_OBJECTIVE,
                "valid_for_selection": False,
                "cases": [],
            }

        ml_cfg = cfg.get("multi_loadcase_screening", {}) or {}

        default_strength_cases = ["center", "torsion_60_40", "lateral_imperfection"]
        if bool(ml_cfg.get("include_longitudinal_offsets_as_strength_cases", False)):
            default_strength_cases += ["left_offset", "right_offset"]
        strength_governing_names = {
            str(v)
            for v in (
                ml_cfg.get("strength_governing_cases")
                or default_strength_cases
            )
        }

        robustness_names = {
            str(v)
            for v in (
                ml_cfg.get("robustness_cases")
                or ["left_offset", "right_offset"]
            )
        }

        service_names = {
            str(v)
            for v in (
                ml_cfg.get("service_cases")
                or ["self_weight"]
            )
        }

        strength_cases = [
            c for c in cases
            if str(c.get("case")) in strength_governing_names
        ] or [
            c for c in cases
            if str(c.get("case")) not in service_names
        ] or cases

        robustness_cases = [
            c for c in cases
            if str(c.get("case")) in robustness_names
        ]

        min_fs_pre = self._min_positive(
            (c.get("min_fs_primary") for c in strength_cases),
            default=0.0,
        )

        min_fs_design = self._min_positive(
            (c.get("min_fs_design") for c in strength_cases),
            default=min_fs_pre,
        )

        predicted_break = min(
            (
                safe_float(c.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
                for c in strength_cases
            ),
            default=0.0,
        )

        robustness_min_fs = self._min_positive(
            (c.get("min_fs_design") for c in robustness_cases),
            default=min_fs_design,
        )

        robustness_min_break = min(
            (
                safe_float(c.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
                for c in robustness_cases
            ),
            default=predicted_break,
        )

        max_disp = max(
            (
                safe_float(c.get("max_displacement_proxy_mm"), 0.0) or 0.0
                for c in cases
            ),
            default=0.0,
        )

        mean_mass = sum(
            safe_float(c.get("mass_proxy_g"), 0.0) or 0.0
            for c in cases
        ) / max(1, len(cases))

        all_regular = all(bool(c.get("solver_regular")) for c in cases)
        all_eq = all(bool(c.get("equilibrium_ok")) for c in cases)

        near_zero_sets = [
            set(c.get("near_zero_member_ids") or [])
            for c in cases
        ]

        zero_force_intersection = (
            set.intersection(*near_zero_sets)
            if near_zero_sets
            else set()
        )

        lateral_case = next(
            (c for c in cases if c.get("case") == "lateral_imperfection"),
            None,
        )

        lateral_stability = 1.0 / max(
            1.0,
            (
                safe_float(
                    (lateral_case or {}).get("max_displacement_proxy_mm"),
                    0.0,
                )
                or 0.0
            )
            / 6.0,
        )

        nodal_stability_proxy = min(
            (
                safe_float(c.get("nodal_stability_proxy"), 0.0) or 0.0
                for c in cases
            ),
            default=0.0,
        )

        topology_stability_proxy = min(
            (
                safe_float(c.get("topology_stability_proxy"), 0.0) or 0.0
                for c in cases
            ),
            default=0.0,
        )

        support_balance = min(
            (
                safe_float(c.get("support_reaction_balance"), 0.0) or 0.0
                for c in cases
            ),
            default=0.0,
        )

        load_path = min(
            (
                safe_float(c.get("load_path_score"), 0.0) or 0.0
                for c in cases
            ),
            default=0.0,
        )

        buckling_risk = max(
            (
                safe_float(c.get("buckling_risk_proxy"), 0.0) or 0.0
                for c in cases
            ),
            default=0.0,
        )

        mass_limit = max(1.0, float(effective_mass_limit_g(cfg)))

        complexity_penalty = self._topology_complexity_penalty(
            int(len((cases[0] or {}).get("member_results") or []))
        )

        valid_for_selection = (
            all_regular
            and all_eq
            and topology_stability_proxy > 0.0
            and all(self._is_selectable_case(c) for c in cases)
        )

        mechanism_penalty = 0.0 if valid_for_selection else 1.0

        constructability = _clamp(
            0.5 * support_balance + 0.5 * load_path,
            0.0,
            1.0,
        )

        target_break = float(
            cfg.get("analysis", {}).get(
                "acceptance_min_design_breaking_load_kgf",
                80.0,
            )
        )

        # Robustez deslocada deve afetar o score, mas não deve derrubar a ruptura
        # nominal primária para 1 kgf se o caso central está regular.
        robustness_penalty = 0.0
        if robustness_cases:
            robustness_penalty = max(
                0.0,
                (0.35 * target_break - robustness_min_break)
                / max(1.0, 0.35 * target_break),
            )

        glue_overuse_penalty = robustness_penalty

        if valid_for_selection:
            objective = self._objective_score(
                cfg,
                predicted_breaking_load_kgf=predicted_break,
                min_fs_design=min_fs_design,
                mass_g=mean_mass,
                max_displacement_mm=max_disp,
                mechanism_penalty=mechanism_penalty,
                topology_complexity_penalty=complexity_penalty,
                glue_overuse_penalty=glue_overuse_penalty,
                constructability_score=constructability,
            )
        else:
            objective = INVALID_OBJECTIVE

        return {
            "cases": cases,
            "solver_regular": all_regular,
            "equilibrium_ok": all_eq,
            "min_fs_preliminary": min_fs_pre,
            "min_fs_design_proxy": min_fs_design,
            "predicted_breaking_load_proxy_kgf": predicted_break,
            "dead_weight_proxy_g": mean_mass,
            "max_displacement_proxy_mm": max_disp,
            "buckling_risk_proxy": buckling_risk,
            "robustness_min_fs_design_proxy": robustness_min_fs,
            "robustness_min_breaking_load_proxy_kgf": robustness_min_break,
            "robustness_penalty": robustness_penalty,
            "multi_case_zero_force_members": len(zero_force_intersection),
            "zero_force_member_ids": sorted(zero_force_intersection),
            "nodal_stability_proxy": nodal_stability_proxy,
            "lateral_stability_proxy": lateral_stability,
            "support_reaction_balance": support_balance,
            "load_path_score": load_path,
            "topology_stability_proxy": topology_stability_proxy,
            "objective": objective,
            "valid_for_selection": valid_for_selection,
            "mass_limit_g": mass_limit,
            "geometry_hash": self._signature_hashes(cfg, "center")["geometry_hash"],
            "topology_hash": self._signature_hashes(cfg, "center")["topology_hash"],
            "sizing_hash": self._signature_hashes(cfg, "center")["sizing_hash"],
            "load_case_hash": self._signature_hashes(cfg, "center")["load_case_hash"],
        }

    def _stage0_precheck_domain(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        bridge = cfg.get("bridge", {}) or {}
        mat = cfg.get("material", {}) or {}
        analysis = cfg.get("analysis", {}) or {}

        rows = {
            "span_mm": float(bridge.get("span_mm", 0.0)),
            "width_mm": float(bridge.get("width_mm", 0.0)),
            "center_height_mm": float(bridge.get("center_height_mm", 0.0)),
            "left_support_overhang_mm": abs(float(bridge.get("left_support_overhang_mm", 0.0))),
            "right_support_overhang_mm": abs(float(bridge.get("right_support_overhang_mm", 0.0))),
            "load_total_kgf": float(bridge.get("load_total_kgf", 0.0)),
            "mass_limit_g": float(effective_mass_limit_g(cfg)),
            "stick_length_mm": float(mat.get("stick_length_mm", 0.0)),
            "stick_width_mm": float(mat.get("stick_width_mm", 0.0)),
            "stick_thickness_mm": float(mat.get("stick_thickness_mm", 0.0)),
            "acceptance_min_design_breaking_load_kgf": float(analysis.get("acceptance_min_design_breaking_load_kgf", 80.0)),
        }

        violations: List[str] = []
        if abs(rows["span_mm"] - 1200.0) > 1.0e-6:
            violations.append(f"Vão inválido: {rows['span_mm']:.1f} mm (edital exige 1200 mm).")
        if rows["left_support_overhang_mm"] > 100.0 + 1.0e-6 or rows["right_support_overhang_mm"] > 100.0 + 1.0e-6:
            violations.append(
                "Apoios inválidos: sobreposição de apoio excede 100 mm por lado."
            )
        if not (100.0 - 1.0e-6 <= rows["width_mm"] <= 200.0 + 1.0e-6):
            violations.append(f"Largura inválida: {rows['width_mm']:.1f} mm (faixa 100..200 mm).")
        if rows["center_height_mm"] < 50.0 - 1.0e-6:
            violations.append(f"Altura central inválida: {rows['center_height_mm']:.1f} mm (mínimo 50 mm).")
        rules = cfg.get("competition_rules", {}) or {}
        enforce_stick_dims = bool(rules.get("enforce_nominal_stick_dimensions", False))
        self._validate_dimension(
            violations,
            "Palito - comprimento",
            rows["stick_length_mm"],
            required=(float(rules.get("required_stick_length_mm")) if enforce_stick_dims and rules.get("required_stick_length_mm") is not None else None),
            tolerance=float(rules.get("stick_length_tolerance_mm", 0.5)),
        )
        self._validate_dimension(
            violations,
            "Palito - largura",
            rows["stick_width_mm"],
            required=(float(rules.get("required_stick_width_mm")) if enforce_stick_dims and rules.get("required_stick_width_mm") is not None else None),
            tolerance=float(rules.get("stick_width_tolerance_mm", 0.2)),
        )
        self._validate_dimension(
            violations,
            "Palito - espessura",
            rows["stick_thickness_mm"],
            required=(float(rules.get("required_stick_thickness_mm")) if enforce_stick_dims and rules.get("required_stick_thickness_mm") is not None else None),
            tolerance=float(rules.get("stick_thickness_tolerance_mm", 0.2)),
        )
        if rows["load_total_kgf"] <= 0.0:
            violations.append("Carga de projeto inválida: load_total_kgf deve ser > 0.")

        rows["violations"] = violations
        rows["ok"] = len(violations) == 0
        return rows

    def _build_macro_archetypes(self, cfg: Dict[str, Any], macro_count: int) -> List[Dict[str, Any]]:
        b = cfg.get("bridge", {}) or {}
        span = float(b.get("span_mm", 1200.0))
        width = float(b.get("width_mm", 160.0))
        height = float(b.get("center_height_mm", 300.0))
        panel = float(b.get("panel_mm", 100.0))
        overlap = float(cfg.get("detail_model", {}).get("overlap_length_mm", 30.0))

        base = {
            "span_mm": span,
            "width_mm": width,
            "center_height_mm": height,
            "panel_mm": panel,
            "top_profile": "parker_plateau",
            "internal_truss_type": "X",
            "top_chord_truss_type": "X",
            "bottom_chord_truss_type": "X",
            "chord_truss_type": "none",
            "tension_joint_model": "double_lap_reinforced",
            "compression_joint_model": "double_lap_reinforced",
            "splice_mode": "overlap",
            "overlap_length_mm": overlap,
        }

        macros: List[Dict[str, Any]] = [
            {
                **base,
                "macro_name": "pratt",
                "global_pattern": "pratt",
                "side_truss_type": "Pratt_symmetric",
                "reinforcement_profile": "balanced",
            },
            {
                **base,
                "macro_name": "howe",
                "global_pattern": "howe",
                "side_truss_type": "Howe_inverted",
                "reinforcement_profile": "strong_top",
            },
            {
                **base,
                "macro_name": "warren",
                "global_pattern": "warren",
                "side_truss_type": "Warren_symmetric",
                "reinforcement_profile": "balanced",
            },
            {
                **base,
                "macro_name": "warren_with_verticals",
                "global_pattern": "warren_with_verticals",
                "side_truss_type": "Warren_mid_braced",
                "reinforcement_profile": "balanced",
            },
            {
                **base,
                "macro_name": "x_bracing_light",
                "global_pattern": "x_light",
                "side_truss_type": "Pratt_symmetric",
                "internal_truss_type": "X",
                "top_chord_truss_type": "X",
                "bottom_chord_truss_type": "X",
                "reinforcement_profile": "light",
                "force_tension_only_bracing": False,
            },
            {
                **base,
                "macro_name": "bowstring",
                "global_pattern": "bowstring",
                "side_truss_type": "Pratt_symmetric",
                "top_profile": "shallow_arch",
                "reinforcement_profile": "strong_top",
            },
            {
                **base,
                "macro_name": "support_dense_panels",
                "global_pattern": "support_dense",
                "side_truss_type": "Pratt_symmetric",

                # Antes: max(70.0, panel * 0.80)
                # Era agressivo demais: aumentava muito o número de membros.
                # Novo: densifica moderadamente sem explodir massa.
                "panel_mm": max(90.0, panel * 0.90),

                "reinforcement_profile": "balanced",
            },
            {
                **base,
                "macro_name": "high_variant",
                "global_pattern": "high",
                "side_truss_type": "Pratt_symmetric",

                # Antes: min(700.0, height * 1.20)
                # Altura alta ajuda reduzir esforços de banzos, mas demais aumenta diagonais,
                # verticais e massa. Mantém ganho moderado.
                "center_height_mm": min(460.0, max(height * 1.12, height + 35.0)),

                "reinforcement_profile": "strong_top",
            },
            {
                **base,
                "macro_name": "wide_torsional_mixed",
                "global_pattern": "wide_mixed",
                "side_truss_type": "Pratt_symmetric",

                # Antes: width * 1.20 até 200 mm.
                # Agora abre largura só o suficiente para rigidez torcional.
                "width_mm": min(180.0, max(100.0, width * 1.08)),

                "panel_side_truss_pattern": "__symmetric_span_mixed__",
                "reinforcement_profile": "balanced",
            },

            # Novos macros "box" corrigidos:
            # Objetivo: entrar no S2/S3 dentro da massa, não nascer superdimensionado.
            {
                **base,
                "macro_name": "high_short_panel_pratt_box",
                "global_pattern": "high_short_panel_pratt_box",
                "side_truss_type": "Pratt_symmetric",

                # Antes: max(90.0, panel * 0.70)
                # Isso aumentava demais o número de painéis/membros.
                "panel_mm": max(100.0, panel),

                # Antes: até 200 mm / mínimo 170 mm.
                # Agora largura moderada para não aumentar transversais e bracings demais.
                "width_mm": min(180.0, max(width * 1.05, 165.0)),

                # Antes: mínimo 420 mm e até 600 mm.
                # Agora altura alta, mas ainda compatível com massa.
                "center_height_mm": min(340.0, max(height * 1.05, 300.0)),
                "end_height_mm": max(80.0, height * 0.30),

                "reinforcement_profile": "compression_box_light",
            },
            {
                **base,
                "macro_name": "high_short_panel_howe_box",
                "global_pattern": "high_short_panel_howe_box",
                "side_truss_type": "Howe_inverted",

                "panel_mm": max(100.0, panel),
                "width_mm": min(180.0, max(width * 1.05, 165.0)),
                "center_height_mm": min(340.0, max(height * 1.05, 300.0)),
                "end_height_mm": max(80.0, height * 0.30),

                "reinforcement_profile": "compression_box_light",
            },
            {
                **base,
                "macro_name": "warren_no_verticals_high_box",
                "global_pattern": "warren_no_verticals_high_box",
                "side_truss_type": "Warren_symmetric",

                # Warren sem verticais tende a reduzir alguns membros secundários,
                # então pode manter painel semelhante aos outros box, sem exagerar.
                "panel_mm": max(100.0, panel),
                "width_mm": min(180.0, max(width * 1.05, 165.0)),
                "center_height_mm": min(440.0, max(height * 1.15, 360.0)),

                "reinforcement_profile": "compression_box_light",
            },

            # Variante adicional útil: alta moderada, sem painel curto.
            # Ajuda o funil a testar ganho de altura sem pagar o custo de aumentar painéis.
            {
                **base,
                "macro_name": "moderate_high_pratt_box",
                "global_pattern": "moderate_high_pratt_box",
                "side_truss_type": "Pratt_symmetric",
                "panel_mm": max(100.0, panel),
                "width_mm": min(175.0, max(width * 1.04, 160.0)),
                "center_height_mm": min(430.0, max(height * 1.12, 340.0)),
                "end_height_mm": max(80.0, height * 0.30),
                "reinforcement_profile": "compression_box_light",
            },
        ]

        macro_count = max(8, min(16, int(macro_count)))
        return macros[:macro_count]

    def _macro_to_config(self, base_cfg: Dict[str, Any], macro: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.planner._apply_candidate_geometry(base_cfg, macro)
        self.planner._apply_reinforcement_profile(cfg, str(macro.get("reinforcement_profile", "balanced")))
        bridge = cfg.setdefault("bridge", {})
        for k in ("panel_side_truss_pattern", "panel_top_chord_pattern", "panel_bottom_chord_pattern"):
            if k in macro:
                bridge[k] = copy.deepcopy(macro[k])
        self._sync_bridge_contacts(bridge)
        if bridge.get("panel_side_truss_pattern") == "__symmetric_span_mixed__":
            bridge["panel_side_truss_pattern"] = self._make_symmetric_span_pattern(bridge)
        if bool(macro.get("force_tension_only_bracing", False)):
            cfg.setdefault("analysis", {})["enable_tension_only_solver_in_funnel"] = True
            bridge["tension_only_bracing_solver_enabled"] = True
            bridge["tension_only_bracing_interpretation"] = True
        return self.planner.config.normalize(cfg)

    @staticmethod
    def _row_score(row: Dict[str, Any]) -> float:
        return (
            safe_float(
                row.get("quick_score", row.get("objective", INVALID_OBJECTIVE)),
                INVALID_OBJECTIVE,
            )
            or INVALID_OBJECTIVE
        )

    @classmethod
    def _pick_with_diversity(
        cls,
        rows: List[Dict[str, Any]],
        top_k: int,
        key_field: str = "global_pattern",
    ) -> List[Dict[str, Any]]:
        selectable = [
            r
            for r in rows
            if bool(
                r.get(
                    "valid_for_selection",
                    cls._row_score(r) > INVALID_OBJECTIVE / 10.0,
                )
            )
            and cls._row_score(r) > INVALID_OBJECTIVE / 10.0
        ]

        if not selectable:
            return []

        if len(selectable) <= top_k:
            return list(selectable)

        ordered = sorted(
            selectable,
            key=cls._row_score,
            reverse=True,
        )

        out: List[Dict[str, Any]] = []
        used = set()

        for r in ordered:
            fam = str(r.get(key_field, ""))

            if fam in used:
                continue

            out.append(r)
            used.add(fam)

            if len(out) >= top_k:
                return out

        for r in ordered:
            if r in out:
                continue

            out.append(r)

            if len(out) >= top_k:
                break

        return out[:top_k]

    def _quick_score_from_case(self, cfg: Dict[str, Any], case_row: Dict[str, Any]) -> float:
        if not self._is_selectable_case(case_row):
            return INVALID_OBJECTIVE

        pp = cfg.get("planner_pipeline", {}) or {}
        analysis = cfg.get("analysis", {}) or {}

        mass = safe_float(case_row.get("mass_proxy_g"), 0.0) or 0.0
        mass_limit = max(1.0, float(effective_mass_limit_g(cfg)))
        mass_ratio = mass / mass_limit

        predicted_break = safe_float(
            case_row.get("predicted_breaking_load_proxy_kgf"),
            0.0,
        ) or 0.0

        target_break = max(
            1.0,
            float(analysis.get("acceptance_min_design_breaking_load_kgf", 80.0)),
        )

        break_ratio = predicted_break / target_break

        min_fs = safe_float(case_row.get("min_fs_design"), 0.0) or 0.0
        max_disp = safe_float(case_row.get("max_displacement_proxy_mm"), 0.0) or 0.0

        # Massa em S2:
        #
        # 1. mass_ratio <= preferred_mass_ratio:
        #    candidato saudável para seguir.
        #
        # 2. preferred_mass_ratio < mass_ratio <= soft_mass_factor:
        #    candidato pode seguir sem penalização eliminatória; S5/S6 ainda podem redistribuir.
        #
        # 3. soft_mass_factor < mass_ratio <= hard_mass_factor:
        #    candidato só segue se já for estruturalmente promissor.
        #
        # 4. mass_ratio > hard_mass_factor:
        #    rejeição dura.
        #
        # Defaults recomendados:
        # preferred = 0.92
        # soft      = 1.00
        # hard      = 1.03
        #
        # Assim, um candidato de 1004 g pode passar se for realmente bom,
        # mas um candidato de 1400–1900 g morre em S2.
        preferred_mass_ratio = float(pp.get("s2_preferred_mass_ratio", 0.95))
        soft_mass_factor = float(pp.get("s2_soft_mass_factor", 1.00))
        hard_mass_factor = float(pp.get("s2_hard_mass_reject_factor", 1.40))

        # Mantém ordem defensiva caso o config venha incoerente.
        preferred_mass_ratio = max(0.50, min(preferred_mass_ratio, 1.00))
        soft_mass_factor = max(preferred_mass_ratio, min(soft_mass_factor, hard_mass_factor))
        hard_mass_factor = max(soft_mass_factor, hard_mass_factor)

        # Exigência mínima para permitir candidato acima do limite proxy.
        # Use 0.45 por padrão: só aceita passar de 100% da massa se já estiver
        # acima de ~36 kgf quando a meta é 80 kgf.
        overweight_min_break_ratio = float(
            pp.get("s2_overweight_min_break_ratio", 0.45)
        )

        # Rejeição dura: massa muito acima do limite.
        if mass_ratio > hard_mass_factor:
            return INVALID_OBJECTIVE

        # Rejeição condicional: passou do limite, mas ainda é fraco.
        if mass_ratio > soft_mass_factor and break_ratio < overweight_min_break_ratio:
            return INVALID_OBJECTIVE

        mechanism_penalty = 0.0 if bool(case_row.get("solver_regular")) else 1.0
        complexity = self._topology_complexity_penalty(
            int(len(case_row.get("member_results") or []))
        )

        constructability = (
            0.5 * (safe_float(case_row.get("load_path_score"), 0.0) or 0.0)
            + 0.5 * (safe_float(case_row.get("support_reaction_balance"), 0.0) or 0.0)
        )

        score = self._objective_score(
            cfg,
            predicted_breaking_load_kgf=predicted_break,
            min_fs_design=min_fs,
            mass_g=mass,
            max_displacement_mm=max_disp,
            mechanism_penalty=mechanism_penalty,
            topology_complexity_penalty=complexity,
            glue_overuse_penalty=0.0,
            constructability_score=constructability,
        )

        # Penalidade suave para massa acima do alvo preferencial, sem matar
        # candidatos próximos do limite que sejam estruturalmente bons.
        #
        # Exemplo:
        # mass_ratio = 0.95 penaliza pouco;
        # mass_ratio = 1.02 penaliza mais;
        # mass_ratio > hard_mass_factor já foi eliminado acima.
        if mass_ratio > preferred_mass_ratio:
            score -= 3.0 * (mass_ratio - preferred_mass_ratio)

        # Bônus pequeno para candidatos que ainda têm massa utilizável
        # e já mostram caminho de carga razoável.
        # Não deve dominar resistência, apenas desempatar.
        if mass_ratio < preferred_mass_ratio and break_ratio >= 0.25:
            score += 0.25 * (preferred_mass_ratio - mass_ratio)

        return score

    def _trust_region_refine(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
    ) -> Dict[str, Any]:
        settings = cfg.get("local_geometry_refinement", {}) or {}
        max_iterations = max(1, int(settings.get("max_iterations", 30)))
        patience = max(1, int(settings.get("patience", 6)))
        max_candidates = max(4, int(settings.get("max_candidates_per_iteration", 12)))
        shrink = float(settings.get("shrink_factor", 0.5))
        expand = float(settings.get("expand_factor", 1.25))
        min_delta = float(settings.get("min_delta_mm", 2.0))

        dh = float(settings.get("initial_delta_height_mm", 30.0))
        dp = float(settings.get("initial_delta_panel_x_mm", 15.0))
        dw = float(settings.get("initial_delta_width_mm", 20.0))

        cur_cfg = self.planner.config.normalize(cfg)
        cur_summary = self._multi_case_summary(cur_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
        best_cfg = cur_cfg
        best_summary = cur_summary
        if not self._summary_valid_flag(cur_summary):
            return {
                "best_cfg": best_cfg,
                "best_summary": best_summary,
                "trace_rows": [],
                "before": {
                    "center_height_mm": float(cfg.get("bridge", {}).get("center_height_mm", 0.0)),
                    "panel_mm": float(cfg.get("bridge", {}).get("panel_mm", 0.0)),
                    "width_mm": float(cfg.get("bridge", {}).get("width_mm", 0.0)),
                },
                "after": {
                    "center_height_mm": float(best_cfg.get("bridge", {}).get("center_height_mm", 0.0)),
                    "panel_mm": float(best_cfg.get("bridge", {}).get("panel_mm", 0.0)),
                    "width_mm": float(best_cfg.get("bridge", {}).get("width_mm", 0.0)),
                },
            }
        no_improve = 0
        trace_rows: List[Dict[str, Any]] = []
        rng = random.Random(42)

        for it in range(1, max_iterations + 1):
            b = cur_cfg.get("bridge", {}) or {}
            candidates: List[Tuple[str, Dict[str, Any], float, float, float]] = []

            base_moves = [
                ("height_plus", dh, 0.0, 0.0),
                ("height_minus", -dh, 0.0, 0.0),
                ("panel_plus", 0.0, dp, 0.0),
                ("panel_minus", 0.0, -dp, 0.0),
                ("width_plus", 0.0, 0.0, dw),
                ("width_minus", 0.0, 0.0, -dw),
            ]
            while len(base_moves) < max_candidates:
                base_moves.append(
                    (
                        f"rand_{len(base_moves)}",
                        rng.uniform(-dh, dh),
                        rng.uniform(-dp, dp),
                        rng.uniform(-dw, dw),
                    )
                )

            for label, mv_h, mv_p, mv_w in base_moves[:max_candidates]:
                cand = copy.deepcopy(cur_cfg)
                cb = cand.setdefault("bridge", {})
                cb["center_height_mm"] = max(50.0, float(b.get("center_height_mm", 300.0)) + mv_h)
                cb["panel_mm"] = max(40.0, float(b.get("panel_mm", 100.0)) + mv_p)
                cb["width_mm"] = _clamp(float(b.get("width_mm", 160.0)) + mv_w, 100.0, 200.0)
                cb["end_height_mm"] = max(50.0, min(float(cb["center_height_mm"]), float(cb.get("end_height_mm", 100.0)) + 0.25 * mv_h))
                span = float(cb.get("span_mm", 1200.0))
                cb["plateau_start_mm"] = _clamp(float(cb.get("plateau_start_mm", span / 3.0)) - 0.25 * mv_p, 0.0, span)
                cb["plateau_end_mm"] = _clamp(float(cb.get("plateau_end_mm", 2.0 * span / 3.0)) + 0.25 * mv_p, 0.0, span)
                self._sync_bridge_contacts(cb)
                cand = self.planner.config.normalize(cand)
                candidates.append((label, cand, mv_h, mv_p, mv_w))

            iter_best = None
            for label, cand_cfg, mv_h, mv_p, mv_w in candidates:
                try:
                    summary = self._multi_case_summary(cand_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
                except (TypeError, ValueError, KeyError, RuntimeError) as exc:
                    trace_rows.append(
                        {
                            "iteration": it,
                            "mutation": label,
                            "delta_height_mm": mv_h,
                            "delta_panel_mm": mv_p,
                            "delta_width_mm": mv_w,
                            "radius_height_mm": dh,
                            "radius_panel_mm": dp,
                            "radius_width_mm": dw,
                            "objective": -1.0e9,
                            "predicted_breaking_load_proxy_kgf": 0.0,
                            "min_fs_design_proxy": 0.0,
                            "dead_weight_proxy_g": None,
                            "solver_regular": False,
                            "equilibrium_ok": False,
                            "accepted": False,
                            "error": repr(exc),
                        }
                    )
                    continue
                row = {
                    "iteration": it,
                    "mutation": label,
                    "delta_height_mm": mv_h,
                    "delta_panel_mm": mv_p,
                    "delta_width_mm": mv_w,
                    "radius_height_mm": dh,
                    "radius_panel_mm": dp,
                    "radius_width_mm": dw,
                    "objective": summary.get("objective"),
                    "valid_for_selection": summary.get("valid_for_selection"),
                    "predicted_breaking_load_proxy_kgf": summary.get("predicted_breaking_load_proxy_kgf"),
                    "min_fs_design_proxy": summary.get("min_fs_design_proxy"),
                    "dead_weight_proxy_g": summary.get("dead_weight_proxy_g"),
                    "solver_regular": summary.get("solver_regular"),
                    "equilibrium_ok": summary.get("equilibrium_ok"),
                    "accepted": False,
                }
                trace_rows.append(row)

                if not self._summary_valid_flag(summary):
                    continue

                if iter_best is None or (safe_float(summary.get("objective"), -1.0e99) or -1.0e99) > (safe_float(iter_best[0].get("objective"), -1.0e99) or -1.0e99):
                    iter_best = (summary, cand_cfg, row)

            if iter_best is None:
                break

            chosen_summary, chosen_cfg, chosen_row = iter_best
            cur_obj = safe_float(cur_summary.get("objective"), -1.0e99) or -1.0e99
            new_obj = safe_float(chosen_summary.get("objective"), -1.0e99) or -1.0e99
            improved = new_obj > cur_obj + 1.0e-9

            if improved:
                chosen_row["accepted"] = True
                cur_cfg = chosen_cfg
                cur_summary = chosen_summary
                if new_obj > (safe_float(best_summary.get("objective"), -1.0e99) or -1.0e99):
                    best_cfg = chosen_cfg
                    best_summary = chosen_summary
                gain = new_obj - cur_obj
                if gain > 0.30:
                    dh *= expand
                    dp *= expand
                    dw *= expand
                no_improve = 0
            else:
                dh *= shrink
                dp *= shrink
                dw *= shrink
                no_improve += 1

            if no_improve >= patience:
                break
            if max(abs(dh), abs(dp), abs(dw)) < min_delta:
                break

        return {
            "best_cfg": best_cfg,
            "best_summary": best_summary,
            "trace_rows": trace_rows,
            "before": {
                "center_height_mm": float(cfg.get("bridge", {}).get("center_height_mm", 0.0)),
                "panel_mm": float(cfg.get("bridge", {}).get("panel_mm", 0.0)),
                "width_mm": float(cfg.get("bridge", {}).get("width_mm", 0.0)),
            },
            "after": {
                "center_height_mm": float(best_cfg.get("bridge", {}).get("center_height_mm", 0.0)),
                "panel_mm": float(best_cfg.get("bridge", {}).get("panel_mm", 0.0)),
                "width_mm": float(best_cfg.get("bridge", {}).get("width_mm", 0.0)),
            },
        }

    def _build_member_sizing_envelope(
        self,
        sizing_cases: List[Dict[str, Any]],
        *,
        zero_force_threshold_N: float = 2.0,
    ) -> Tuple[List[Any], List[Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        usable_cases = [
            c for c in sizing_cases
            if self._is_selectable_case(c)
        ] or list(sizing_cases or [])

        if not usable_cases:
            return [], [], [], []

        reference = usable_cases[0]
        nodes = reference.get("nodes") or []
        members = reference.get("members") or []

        result_maps: Dict[str, Dict[int, Dict[str, Any]]] = {}
        check_maps: Dict[str, Dict[int, Dict[str, Any]]] = {}

        for case in usable_cases:
            case_name = str(case.get("case", "unknown"))

            result_maps[case_name] = {
                int(r.get("member_id")): r
                for r in (case.get("member_results") or [])
                if r.get("member_id") is not None
            }

            check_maps[case_name] = {
                int(r.get("member_id")): r
                for r in (case.get("member_checks") or [])
                if r.get("member_id") is not None
            }

        all_member_ids = sorted(
            {
                int(getattr(m, "id"))
                for m in members
                if getattr(m, "id", None) is not None
            }
        )

        envelope_results: List[Dict[str, Any]] = []
        envelope_checks: List[Dict[str, Any]] = []

        for member_id in all_member_ids:
            force_samples: List[Tuple[str, float, Dict[str, Any]]] = []
            check_samples: List[Tuple[str, float, float, Dict[str, Any]]] = []

            for case_name, rmap in result_maps.items():
                result_row = rmap.get(member_id)
                if result_row is None:
                    continue

                n_val = safe_float(result_row.get("N_N"), 0.0) or 0.0
                force_samples.append((case_name, float(n_val), result_row))

                check_row = check_maps.get(case_name, {}).get(member_id, {})
                fs_design = safe_float(check_row.get("FS_design"), None)
                fs_min = safe_float(check_row.get("FS_min"), None)

                fs_for_rank = fs_design
                if fs_for_rank is None:
                    fs_for_rank = fs_min
                if fs_for_rank is None:
                    fs_for_rank = float("inf")

                util = safe_float(check_row.get("utilization"), 0.0) or 0.0
                check_samples.append((case_name, float(fs_for_rank), float(util), check_row))

            if not force_samples:
                continue

            force_case, signed_force_at_max_abs, force_template = max(
                force_samples,
                key=lambda item: abs(item[1]),
            )

            max_tension_N = max([max(0.0, n) for _, n, _ in force_samples] + [0.0])
            max_compression_N = max([max(0.0, -n) for _, n, _ in force_samples] + [0.0])
            max_abs_N = max(abs(n) for _, n, _ in force_samples)
            low_force_all_cases = all(abs(n) <= zero_force_threshold_N for _, n, _ in force_samples)

            if check_samples:
                worst_case, _, _, worst_check = min(
                    check_samples,
                    key=lambda item: item[1],
                )
                max_util = max(util for _, _, util, _ in check_samples)
            else:
                worst_case = force_case
                worst_check = {}
                max_util = 0.0

            fs_min_values = [
                safe_float(row.get("FS_min"), None)
                for _, _, _, row in check_samples
            ]
            fs_min_clean = [
                float(v) for v in fs_min_values
                if v is not None and math.isfinite(float(v)) and float(v) > 1.0e-12
            ]

            fs_design_values = [
                safe_float(row.get("FS_design"), None)
                for _, _, _, row in check_samples
            ]
            fs_design_clean = [
                float(v) for v in fs_design_values
                if v is not None and math.isfinite(float(v)) and float(v) > 1.0e-12
            ]

            env_result = copy.deepcopy(force_template)
            env_result["member_id"] = member_id
            env_result["N_N"] = float(signed_force_at_max_abs)
            env_result["envelope_force_case"] = force_case
            env_result["max_tension_N"] = float(max_tension_N)
            env_result["max_compression_N"] = float(max_compression_N)
            env_result["max_abs_N"] = float(max_abs_N)
            env_result["low_force_all_cases"] = bool(low_force_all_cases)

            env_check = copy.deepcopy(worst_check)
            env_check["member_id"] = member_id

            env_check["FS_min"] = (
                min(fs_min_clean)
                if fs_min_clean
                else safe_float(worst_check.get("FS_min"), 999.0)
            )

            # Regra crucial:
            # Se nenhum caso trouxe FS_design válido, este membro NÃO é relevante
            # para sizing global. Não converter FS_min local em FS_design.
            if fs_design_clean:
                env_check["FS_design"] = min(fs_design_clean)
                env_check["design_relevant"] = True
                env_check["utilization_design"] = float(max_util)
            else:
                env_check["FS_design"] = None
                env_check["design_relevant"] = False
                env_check["utilization_design"] = None

            env_check["utilization"] = float(max_util)
            env_check["worst_case"] = worst_case
            env_check["envelope_force_case"] = force_case
            env_check["governing_mode"] = str(
                worst_check.get("governing_mode")
                or worst_check.get("failure_mode")
                or "envelope"
            )
            env_check["max_tension_N"] = float(max_tension_N)
            env_check["max_compression_N"] = float(max_compression_N)
            env_check["max_abs_N"] = float(max_abs_N)
            env_check["low_force_all_cases"] = bool(low_force_all_cases)

            envelope_results.append(env_result)
            envelope_checks.append(env_check)

        return nodes, members, envelope_results, envelope_checks

    def _budget_member_sizing_plan(
        self,
        cfg: Dict[str, Any],
        plan: Dict[int, Any],
        *,
        current_mass_g: float,
        mass_limit_g: float,
        reserve_g: float = 10.0,
        max_reinforcements: int | None = None,
        current_breaking_load_kgf: float = 0.0,
        target_breaking_load_kgf: float = 80.0,
    ) -> Dict[int, Any]:
        competitive_ratio = float(
            (cfg.get("member_sizing", {}) or {}).get("competitive_mass_target_ratio", 0.98)
        )
        effective_budget_limit_g = min(float(mass_limit_g), float(mass_limit_g) * competitive_ratio)

        budget_g = max(
            0.0,
            effective_budget_limit_g - float(current_mass_g) - float(reserve_g),
        )

        keep: Dict[int, Any] = {}

        analysis = cfg.get("analysis", {}) or {}

        global_failure_groups = set(
            analysis.get(
                "global_failure_groups",
                ["bottom_chord", "top_chord", "vertical", "diagonal", "support_pad"],
            )
        )

        weak_and_under_mass = (
            float(current_breaking_load_kgf) < 0.75 * float(target_breaking_load_kgf)
            and float(current_mass_g) < float(mass_limit_g)
        )

        # 1. Alívios/reduções entram primeiro, mas com uma restrição:
        # se a ponte ainda está muito fraca e dentro da massa, NÃO aliviar
        # membros globais principais. O resultado atual mostra massa sobrando,
        # então aliviar top_chord/vertical/diagonal cedo demais é contraproducente.
        for mid, decision in plan.items():
            mid = int(mid)

            delta = safe_float(getattr(decision, "delta_mass_g", 0.0), 0.0) or 0.0
            action = str(getattr(decision, "action", "")).strip().lower()
            group = str(getattr(decision, "original_group", ""))

            is_reduction = (
                delta <= 0.0
                or action in {"lighten", "reduce", "donate", "simplify_joint"}
            )

            if not is_reduction:
                continue

            if weak_and_under_mass and group in global_failure_groups:
                # Não tirar massa dos membros que podem virar gargalo depois.
                continue

            keep[mid] = decision
            budget_g += abs(float(delta))

        # 2. Reforços entram por prioridade estrutural por grama.
        reinforcements: List[Tuple[float, float, float, str, int, Any]] = []

        for mid, decision in plan.items():
            mid = int(mid)

            if mid in keep:
                continue

            action = str(getattr(decision, "action", "")).strip().lower()

            if action != "reinforce":
                continue

            delta = safe_float(getattr(decision, "delta_mass_g", 0.0), 0.0) or 0.0

            if delta <= 0.0:
                continue

            fs = safe_float(getattr(decision, "FS_min", None), None)
            util = safe_float(getattr(decision, "utilization", 0.0), 0.0) or 0.0
            group = str(getattr(decision, "original_group", ""))

            group_priority = {
                "vertical": 8.0,
                "top_chord": 7.0,
                "diagonal": 4.5,
                "bottom_chord": 3.0,
                "support_pad": 2.0,
            }.get(group, 0.25)

            fs_value = float(fs if fs is not None else 1.0)
            fs_term = 1.0 / max(0.05, fs_value)

            # Não deixe o critério "ganho por grama" ignorar o pior membro global.
            # Primeiro entram membros globais com FS muito baixo; depois o ranking volta
            # a ser eficiência por massa. Isso evita deixar montantes centrais longos
            # com 2 palitos enquanto reforços baratos consomem o orçamento.
            score = (group_priority * fs_term + float(util)) / max(0.5, float(delta))

            reinforcements.append((score, float(delta), fs_value, group, mid, decision))

        critical_budget_first_fs = float(
            (cfg.get("member_sizing", {}) or {}).get("critical_budget_first_fs", 0.85)
        )

        critical_first = [
            item for item in reinforcements
            if item[3] in {"vertical", "top_chord", "diagonal"} and item[2] < critical_budget_first_fs
        ]
        regular_reinforcements = [item for item in reinforcements if item not in critical_first]

        critical_first.sort(key=lambda item: (item[2], -item[0]))
        regular_reinforcements.sort(key=lambda item: item[0], reverse=True)

        applied_reinforcements = 0

        for _, delta, _, _, mid, decision in critical_first + regular_reinforcements:
            if max_reinforcements is not None and applied_reinforcements >= int(max_reinforcements):
                break

            if delta <= budget_g + 1.0e-9:
                keep[int(mid)] = decision
                budget_g -= delta
                applied_reinforcements += 1

        # 3. Decisões neutras sem aumento de massa podem entrar.
        for mid, decision in plan.items():
            mid = int(mid)

            if mid in keep:
                continue

            delta = safe_float(getattr(decision, "delta_mass_g", 0.0), 0.0) or 0.0
            action = str(getattr(decision, "action", "")).strip().lower()

            if action == "keep" and abs(float(delta)) <= 1.0e-9:
                keep[mid] = decision

        return keep
    def _member_sizing_pass(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
    ) -> Dict[str, Any]:
        sizing_settings = cfg.get("member_sizing", {}) or {}
        local_sizing_settings = (
            (cfg.get("planner", {}) or {}).get("local_sizing", {}) or {}
        )
        ml_cfg = cfg.get("multi_loadcase_screening", {}) or {}
        sizing_load_cases = [
            str(v)
            for v in (
                sizing_settings.get("sizing_load_cases")
                or ml_cfg.get("strength_governing_cases")
                or ["center", "torsion_60_40", "lateral_imperfection"]
            )
        ]
        # S5 é iterativo e caro.  Usar os casos de resistência para as iterações
        # de dimensionamento e deixar a validação completa para S7/S8 evita que
        # o pipeline pare por timeout logo depois de S3, antes de gerar qualquer
        # relatório útil.
        s5_iteration_cases = sizing_load_cases if bool(sizing_settings.get("s5_fast_strength_cases_only", True)) else load_cases

        before_summary_probe = self._multi_case_summary(
            cfg,
            s5_iteration_cases,
            stage_name=stage_name,
            tension_only=tension_only,
        )

        if not self._summary_valid_flag(before_summary_probe):
            return {
                "best_cfg": cfg,
                "summary": before_summary_probe,
                "trace_rows": [],
                "donors": [],
                "critical": [],
                "before_after": [],
            }

        max_sizing_rounds = max(
            1,
            int(
                sizing_settings.get(
                    "max_sizing_rounds",
                    local_sizing_settings.get("max_sizing_rounds", 4),
                )
            ),
        )

        min_strength_gain_ratio = float(
            sizing_settings.get(
                "min_strength_gain_ratio",
                local_sizing_settings.get("min_strength_gain_ratio", 1.02),
            )
        )

        max_mass_overrun_ratio = float(
            sizing_settings.get(
                "max_mass_overrun_ratio",
                local_sizing_settings.get("max_mass_overrun_ratio", 1.02),
            )
        )

        allow_flat_pre_target_rounds = int(
            sizing_settings.get(
                "allow_flat_pre_target_rounds",
                local_sizing_settings.get("allow_flat_pre_target_rounds", 2),
            )
        )

        flat_rounds_used = 0

        target_break = float(
            cfg.get("analysis", {}).get(
                "acceptance_min_design_breaking_load_kgf",
                80.0,
            )
        )

        cur_cfg = self.planner.config.normalize(cfg)
        cur_summary = before_summary_probe

        best_cfg = cur_cfg
        best_summary = cur_summary

        all_trace_rows: List[Dict[str, Any]] = []
        all_donors: List[Dict[str, Any]] = []
        all_critical: List[Dict[str, Any]] = []
        all_before_after: List[Dict[str, Any]] = []

        best_break = safe_float(
            best_summary.get("predicted_breaking_load_proxy_kgf"),
            0.0,
        ) or 0.0

        best_fs = safe_float(
            best_summary.get("min_fs_design_proxy"),
            0.0,
        ) or 0.0

        best_obj = safe_float(
            best_summary.get("objective"),
            INVALID_OBJECTIVE,
        ) or INVALID_OBJECTIVE

        for sizing_round in range(1, max_sizing_rounds + 1):
            sizing_cases = [
                self._evaluate_case_cached(
                    cur_cfg,
                    case_name,
                    stage_name=stage_name,
                    tension_only=tension_only,
                )
                for case_name in sizing_load_cases
            ]

            zero_force_threshold_N = float(
                (cur_cfg.get("topology_cleanup", {}) or {}).get(
                    "near_zero_force_threshold_N",
                    2.0,
                )
            )

            envelope_nodes, envelope_members, envelope_results, envelope_checks = (
                self._build_member_sizing_envelope(
                    sizing_cases,
                    zero_force_threshold_N=zero_force_threshold_N,
                )
            )

            if not envelope_nodes or not envelope_members:
                break

            raw_plan = self.planner.build_member_sizing_plan(
                cur_cfg,
                envelope_nodes,
                envelope_members,
                envelope_results,
                envelope_checks,
            )

            if not raw_plan:
                break

            cur_mass = safe_float(
                cur_summary.get("dead_weight_proxy_g"),
                0.0,
            ) or 0.0

            mass_limit = float(effective_mass_limit_g(cur_cfg))

            member_sizing_cfg = cur_cfg.get("member_sizing", {}) or {}

            reserve_g = float(
                member_sizing_cfg.get(
                    "mass_reserve_for_fabrication_g",
                    25.0,
                )
            )

            max_reinforcements = member_sizing_cfg.get(
                "max_budgeted_reinforcements_per_round",
                None,
            )

            if max_reinforcements is not None:
                max_reinforcements = int(max_reinforcements)

            cur_break_for_budget = safe_float(
                cur_summary.get("predicted_breaking_load_proxy_kgf"),
                0.0,
            ) or 0.0

            plan = self._budget_member_sizing_plan(
                cur_cfg,
                raw_plan,
                current_mass_g=cur_mass,
                mass_limit_g=mass_limit,
                reserve_g=reserve_g,
                max_reinforcements=max_reinforcements,
                current_breaking_load_kgf=cur_break_for_budget,
                target_breaking_load_kgf=target_break,
            )

            if not plan:
                break

            envelope_check_by_id = {
                int(r.get("member_id")): r
                for r in envelope_checks
                if r.get("member_id") is not None
            }

            envelope_result_by_id = {
                int(r.get("member_id")): r
                for r in envelope_results
                if r.get("member_id") is not None
            }

            trace_rows: List[Dict[str, Any]] = []
            donors: List[Dict[str, Any]] = []
            critical: List[Dict[str, Any]] = []
            before_after: List[Dict[str, Any]] = []

            changed_members = 0
            reinforce_count = 0
            lighten_count = 0
            net_delta_mass_g = 0.0

            for decision in plan.values():
                env_chk = envelope_check_by_id.get(int(decision.member_id), {})
                env_res = envelope_result_by_id.get(int(decision.member_id), {})

                delta_mass = safe_float(decision.delta_mass_g, 0.0) or 0.0
                net_delta_mass_g += float(delta_mass)

                if decision.n_sticks_current != decision.n_sticks_recommended:
                    changed_members += 1

                if str(decision.action) == "reinforce":
                    reinforce_count += 1

                if str(decision.action) in {"lighten", "reduce", "donate"}:
                    lighten_count += 1

                row = {
                    "sizing_round": sizing_round,
                    "member_id": decision.member_id,
                    "group": decision.original_group,
                    "N_N": decision.force_N,
                    "max_tension_N": env_res.get("max_tension_N"),
                    "max_compression_N": env_res.get("max_compression_N"),
                    "max_abs_N": env_res.get("max_abs_N"),
                    "worst_case": env_chk.get("worst_case"),
                    "envelope_force_case": env_chk.get("envelope_force_case"),
                    "low_force_all_cases": env_chk.get("low_force_all_cases"),
                    "design_relevant": env_chk.get("design_relevant"),
                    "compression_direct_util": env_chk.get("compression_direct_util"),
                    "tension_util": env_chk.get("tension_util"),
                    "buckling_util_y": env_chk.get("buckling_util_y"),
                    "buckling_util_z": env_chk.get("buckling_util_z"),
                    "beam_column_util": env_chk.get("beam_column_util"),
                    "governing_mode": decision.governing_mode,
                    "FS_min": decision.FS_min,
                    "utilization": 1.0 / max(1.0e-9, decision.FS_min),
                    "action": decision.action,
                    "n_sticks_current": decision.n_sticks_current,
                    "n_sticks_recommended": decision.n_sticks_recommended,
                    "delta_mass_g": decision.delta_mass_g,
                    "reason": decision.reason,
                    "can_be_mass_donor": decision.can_be_mass_donor,
                    "round_changed_members": changed_members,
                    "round_reinforce_count": reinforce_count,
                    "round_lighten_count": lighten_count,
                    "round_net_delta_mass_g": net_delta_mass_g,
                    "budgeted_plan": True,
                    "raw_plan_size": len(raw_plan),
                    "budgeted_plan_size": len(plan),
                    "sizing_load_cases": ";".join(sizing_load_cases),
                }

                trace_rows.append(row)

                if bool(decision.can_be_mass_donor):
                    donors.append(row)

                if str(decision.action) == "reinforce":
                    critical.append(row)

                if decision.n_sticks_current != decision.n_sticks_recommended:
                    before_after.append(
                        {
                            "sizing_round": sizing_round,
                            "member_id": decision.member_id,
                            "group": decision.original_group,
                            "before_n_sticks": decision.n_sticks_current,
                            "after_n_sticks": decision.n_sticks_recommended,
                            "delta_mass_g": decision.delta_mass_g,
                            "action": decision.action,
                            "worst_case": env_chk.get("worst_case"),
                            "max_abs_N": env_res.get("max_abs_N"),
                            "FS_min": decision.FS_min,
                            "reason": decision.reason,
                        }
                    )

            all_trace_rows.extend(trace_rows)
            all_donors.extend(donors)
            all_critical.extend(critical)
            all_before_after.extend(before_after)

            if changed_members <= 0:
                break

            new_cfg = self.planner.apply_member_sizing_plan(cur_cfg, plan)
            new_cfg = self.planner.config.normalize(new_cfg)

            # Reavaliação completa com todos os load cases obrigatórios.
            # O sizing usa casos nominais, mas a aceitação continua sendo global.
            new_summary = self._multi_case_summary(
                new_cfg,
                s5_iteration_cases,
                stage_name=stage_name,
                tension_only=tension_only,
            )

            if not self._summary_valid_flag(new_summary):
                break

            before_break = safe_float(
                cur_summary.get("predicted_breaking_load_proxy_kgf"),
                0.0,
            ) or 0.0

            after_break = safe_float(
                new_summary.get("predicted_breaking_load_proxy_kgf"),
                0.0,
            ) or 0.0

            before_fs = safe_float(
                cur_summary.get("min_fs_design_proxy"),
                0.0,
            ) or 0.0

            after_fs = safe_float(
                new_summary.get("min_fs_design_proxy"),
                0.0,
            ) or 0.0

            before_obj = safe_float(
                cur_summary.get("objective"),
                INVALID_OBJECTIVE,
            ) or INVALID_OBJECTIVE

            after_obj = safe_float(
                new_summary.get("objective"),
                INVALID_OBJECTIVE,
            ) or INVALID_OBJECTIVE

            after_mass = safe_float(
                new_summary.get("dead_weight_proxy_g"),
                1.0e99,
            ) or 1.0e99

            mass_limit = float(effective_mass_limit_g(new_cfg))

            strength_improved = (
                after_break >= before_break * min_strength_gain_ratio
                or after_fs >= before_fs * min_strength_gain_ratio
            )

            near_flat_but_not_worse = (
                after_break >= before_break * 0.995
                and after_fs >= before_fs * 0.995
            )

            objective_improved = after_obj >= before_obj + 1.0e-9
            mass_acceptable = after_mass <= mass_limit * max_mass_overrun_ratio

            useful_pre_target_strength_gain = (
                before_break < target_break
                and strength_improved
                and mass_acceptable
            )

            flat_pre_target_round = (
                before_break < target_break
                and mass_acceptable
                and near_flat_but_not_worse
                and flat_rounds_used < allow_flat_pre_target_rounds
            )

            if objective_improved or useful_pre_target_strength_gain or flat_pre_target_round:
                if flat_pre_target_round and not (objective_improved or useful_pre_target_strength_gain):
                    flat_rounds_used += 1

                cur_cfg = new_cfg
                cur_summary = new_summary

                cur_break = safe_float(
                    cur_summary.get("predicted_breaking_load_proxy_kgf"),
                    0.0,
                ) or 0.0

                cur_fs = safe_float(
                    cur_summary.get("min_fs_design_proxy"),
                    0.0,
                ) or 0.0

                cur_obj = safe_float(
                    cur_summary.get("objective"),
                    INVALID_OBJECTIVE,
                ) or INVALID_OBJECTIVE

                if (
                    cur_obj > best_obj + 1.0e-9
                    or (
                        best_break < target_break
                        and cur_break >= best_break * 0.995
                        and cur_fs >= best_fs * 0.995
                    )
                ):
                    best_cfg = cur_cfg
                    best_summary = cur_summary
                    best_break = cur_break
                    best_fs = cur_fs
                    best_obj = cur_obj

                if cur_break >= target_break:
                    break

                continue

            break

        if s5_iteration_cases != load_cases:
            full_summary = self._multi_case_summary(
                best_cfg,
                load_cases,
                stage_name=f"{stage_name}_FULL_VALIDATE",
                tension_only=tension_only,
            )
            if self._summary_valid_flag(full_summary):
                best_summary = full_summary

        return {
            "best_cfg": best_cfg,
            "summary": best_summary,
            "trace_rows": all_trace_rows,
            "donors": all_donors,
            "critical": all_critical,
            "before_after": all_before_after,
        }
    
    def _reinvest_mass_into_critical_members(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
    ) -> Dict[str, Any]:
        """Reinveste massa recuperada pelo S6 em gargalos primários simétricos.

        O S5 dimensiona antes do resgate de massa; quando o S6 remove massa local,
        a versão anterior seguia direto para fabricação e deixava gargalos primários
        com FS baixo. Este passo usa a folga recuperada para acrescentar 1 palito
        por órbita simétrica crítica, sem remover membros e sem mudar topologia.
        """
        settings = cfg.get("member_sizing", {}) or {}
        if not bool(settings.get("enable_post_topology_reinvestment", True)):
            summary = self._multi_case_summary(
                cfg,
                load_cases,
                stage_name=stage_name,
                tension_only=tension_only,
            )
            return {"best_cfg": cfg, "summary": summary, "trace_rows": []}

        cur_cfg = self.planner.config.normalize(cfg)
        cur_summary = self._multi_case_summary(
            cur_cfg,
            load_cases,
            stage_name=stage_name,
            tension_only=tension_only,
        )

        if not self._summary_valid_flag(cur_summary):
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        analysis = cur_cfg.get("analysis", {}) or {}
        material = cur_cfg.get("material", {}) or {}
        bridge = cur_cfg.get("bridge", {}) or {}
        member_sizing = cur_cfg.get("member_sizing", {}) or {}

        mass_limit = float(effective_mass_limit_g(cur_cfg))
        current_mass = safe_float(cur_summary.get("dead_weight_proxy_g"), 0.0) or 0.0
        competitive_ratio = float(member_sizing.get("competitive_mass_target_ratio", 0.98))
        target_proxy_mass = min(
            mass_limit * competitive_ratio,
            mass_limit * float(member_sizing.get("reinvest_target_proxy_mass_ratio", 0.975)),
        )
        reserve_g = float(member_sizing.get("reinvest_final_mass_reserve_g", 16.0))
        available_budget = max(0.0, target_proxy_mass - current_mass - reserve_g)

        if available_budget <= 0.25:
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        if bool(member_sizing.get("reinvest_strength_cases_only", True)):
            ml_cfg = cur_cfg.get("multi_loadcase_screening", {}) or {}
            reinvest_case_names = [
                str(v)
                for v in (
                    member_sizing.get("sizing_load_cases")
                    or ml_cfg.get("strength_governing_cases")
                    or ["center", "torsion_60_40", "lateral_imperfection"]
                )
            ]
            cases = [
                self._evaluate_case_cached(
                    cur_cfg,
                    case_name,
                    stage_name=stage_name,
                    tension_only=tension_only,
                )
                for case_name in reinvest_case_names
            ]
        else:
            cases = cur_summary.get("cases") or []

        if not cases:
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        ref_case = cases[0]
        nodes = ref_case.get("nodes") or []
        members = ref_case.get("members") or []
        node_by_id = {int(getattr(n, "id")): n for n in nodes}
        member_by_id = {int(getattr(m, "id")): m for m in members}

        try:
            partners = self.planner.map_member_to_symmetry_partners(cur_cfg, nodes, members)
        except Exception:
            partners = {}

        global_groups = set(
            analysis.get(
                "global_failure_groups",
                ["bottom_chord", "top_chord", "vertical", "diagonal", "support_pad"],
            )
        )
        priority_group = {
            "vertical": 7.0,
            "top_chord": 6.5,
            "diagonal": 4.0,
            "bottom_chord": 2.0,
            "support_pad": 1.5,
        }
        fs_threshold = float(member_sizing.get("reinvest_fs_threshold", analysis.get("acceptance_min_primary_fs", 1.05)))
        max_orbits = int(member_sizing.get("reinvest_max_members", 12))
        max_sticks_increment = max(1, int(member_sizing.get("reinvest_max_sticks_per_member", 1)))
        min_abs_force_N = float(member_sizing.get("reinvest_min_abs_force_N", 25.0))
        stick_mass_g = float(material.get("stick_mass_g", 1.4))
        stick_len_mm = max(1.0, float(material.get("stick_length_mm", 115.0)))
        span = float(bridge.get("span_mm", 1200.0))
        max_default = int(analysis.get("planner_max_sticks_per_group", 12))
        max_by_group = analysis.get("planner_max_sticks_per_group_by_group", {}) or {}

        def max_for_group(group: str) -> int:
            raw = safe_float(max_by_group.get(group), None)
            return int(raw) if raw is not None else max_default

        worst_by_mid: Dict[int, Dict[str, Any]] = {}
        for case in cases:
            case_name = str(case.get("case", "unknown"))
            checks = case.get("member_checks") or []
            results = {
                int(r.get("member_id")): r
                for r in (case.get("member_results") or [])
                if r.get("member_id") is not None
            }
            for chk in checks:
                mid_raw = chk.get("member_id")
                if mid_raw is None:
                    continue
                mid = int(mid_raw)
                m = member_by_id.get(mid)
                if m is None:
                    continue
                group = str(getattr(m, "group", chk.get("group", "")))
                if group not in global_groups:
                    continue
                if chk.get("design_relevant") is False:
                    continue
                fs = safe_float(chk.get("FS_design"), None)
                if fs is None:
                    fs = safe_float(chk.get("FS_min"), None)
                if fs is None or fs >= fs_threshold:
                    continue
                res = results.get(mid, {})
                n_val = safe_float(res.get("N_N"), 0.0) or 0.0
                if abs(float(n_val)) < min_abs_force_N:
                    continue
                current = worst_by_mid.get(mid)
                if current is None or float(fs) < float(current.get("FS", 1.0e99)):
                    worst_by_mid[mid] = {
                        "FS": float(fs),
                        "case": case_name,
                        "group": group,
                        "N_N": n_val,
                        "mode": chk.get("governing_mode") or chk.get("failure_mode") or "",
                    }

        seen_orbits: set[Tuple[int, ...]] = set()
        candidates: List[Tuple[float, float, Tuple[int, ...], Dict[str, Any]]] = []

        for mid, meta in worst_by_mid.items():
            m = member_by_id.get(mid)
            if m is None:
                continue
            orbit = sorted(set([mid] + [int(v) for v in partners.get(mid, []) if int(v) in member_by_id]))
            orbit_key = tuple(orbit)
            if orbit_key in seen_orbits:
                continue
            seen_orbits.add(orbit_key)

            group = str(meta.get("group"))
            max_allowed = max_for_group(group)
            increments: Dict[int, int] = {}
            delta_mass = 0.0
            center_bonus = 0.0
            min_fs = float(meta.get("FS", 1.0e99))

            for oid in orbit:
                om = member_by_id.get(int(oid))
                if om is None:
                    continue
                old_n = max(1, int(getattr(om, "n_sticks", 1)))
                if old_n >= max_allowed:
                    continue
                inc = min(max_sticks_increment, max_allowed - old_n)
                if inc <= 0:
                    continue
                increments[int(oid)] = inc
                delta_mass += inc * float(getattr(om, "L", 0.0) or 0.0) / stick_len_mm * stick_mass_g
                ni = node_by_id.get(int(getattr(om, "i")))
                nj = node_by_id.get(int(getattr(om, "j")))
                if ni is not None and nj is not None and span > 1.0e-9:
                    mx = 0.5 * (float(getattr(ni, "x")) + float(getattr(nj, "x")))
                    center_bonus = max(center_bonus, 1.0 - abs(mx - 0.5 * span) / max(0.5 * span, 1.0))

            if not increments or delta_mass <= 0.0:
                continue

            score = (
                priority_group.get(group, 1.0) * (1.0 / max(0.05, min_fs))
                + 0.75 * center_bonus
            ) / max(0.5, delta_mass)
            candidates.append((score, delta_mass, orbit_key, {**meta, "increments": increments}))

        candidates.sort(key=lambda item: item[0], reverse=True)

        new_cfg = copy.deepcopy(cur_cfg)
        by_id = new_cfg.setdefault("member_sticks_by_id", {})
        trace_rows: List[Dict[str, Any]] = []
        used_budget = 0.0
        applied_orbits = 0

        for score, delta_mass, orbit, meta in candidates:
            if applied_orbits >= max_orbits:
                break
            if used_budget + delta_mass > available_budget + 1.0e-9:
                continue
            for oid, inc in meta["increments"].items():
                om = member_by_id.get(int(oid))
                if om is None:
                    continue
                old_n = max(1, int(getattr(om, "n_sticks", 1)))
                by_id[str(int(oid))] = old_n + int(inc)
                trace_rows.append(
                    {
                        "member_id": int(oid),
                        "group": str(meta.get("group")),
                        "old_n_sticks": old_n,
                        "new_n_sticks": old_n + int(inc),
                        "delta_mass_g_orbit": delta_mass,
                        "worst_case": meta.get("case"),
                        "FS_before": meta.get("FS"),
                        "N_N": meta.get("N_N"),
                        "score": score,
                        "reason": "post_topology_reinvest_critical_primary_orbit",
                    }
                )
            used_budget += delta_mass
            applied_orbits += 1

        if not trace_rows:
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        new_cfg = self.planner.config.normalize(new_cfg)
        new_summary = self._multi_case_summary(
            new_cfg,
            load_cases,
            stage_name=stage_name,
            tension_only=tension_only,
        )

        old_break = safe_float(cur_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        new_break = safe_float(new_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        old_fs = safe_float(cur_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        new_fs = safe_float(new_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        new_mass = safe_float(new_summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99

        if (
            self._summary_valid_flag(new_summary)
            and new_mass <= target_proxy_mass + 1.0e-9
            and new_break >= old_break * 0.995
            and new_fs >= old_fs * 0.995
        ):
            return {"best_cfg": new_cfg, "summary": new_summary, "trace_rows": trace_rows}

        return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}




    def _remap_member_stick_overrides_by_geometry(
        self,
        old_cfg: Dict[str, Any],
        new_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Reaplica `member_sticks_by_id` quando uma mutação muda a numeração.

        A numeração de membros depende da topologia. Se um plano X vira Warren,
        IDs posteriores podem ser deslocados e overrides antigos passam a cair em
        membros errados. O remapeamento usa grupo + coordenadas dos nós de ponta,
        preservando seções de membros que ainda existem fisicamente e deixando os
        membros novos herdarem o valor do grupo.
        """
        try:
            old_nodes, old_members, _, _ = self.planner.geometry.generate(old_cfg)
            new_nodes, new_members, _, _ = self.planner.geometry.generate(new_cfg)
        except Exception:
            return new_cfg

        def node_map(nodes: List[Any]) -> Dict[int, Any]:
            return {int(getattr(n, "id")): n for n in nodes}

        old_node_by_id = node_map(old_nodes)
        new_node_by_id = node_map(new_nodes)

        def point_key(n: Any) -> Tuple[float, float, float]:
            return (
                round(float(getattr(n, "x", 0.0)), 3),
                round(float(getattr(n, "y", 0.0)), 3),
                round(float(getattr(n, "z", 0.0)), 3),
            )

        def member_key(m: Any, nodes_by_id: Dict[int, Any]) -> Tuple[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] | None:
            ni = nodes_by_id.get(int(getattr(m, "i")))
            nj = nodes_by_id.get(int(getattr(m, "j")))
            if ni is None or nj is None:
                return None
            pts = tuple(sorted([point_key(ni), point_key(nj)]))
            return (str(getattr(m, "group", "")), pts)  # type: ignore[return-value]

        old_n_by_key: Dict[Any, int] = {}
        for m in old_members:
            k = member_key(m, old_node_by_id)
            if k is None:
                continue
            old_n_by_key[k] = max(1, int(getattr(m, "n_sticks", 1)))

        remapped: Dict[str, int] = {}
        for m in new_members:
            k = member_key(m, new_node_by_id)
            if k is None or k not in old_n_by_key:
                continue
            remapped[str(int(getattr(m, "id")))] = int(old_n_by_key[k])

        out = copy.deepcopy(new_cfg)
        if remapped:
            out["member_sticks_by_id"] = remapped
        else:
            out.pop("member_sticks_by_id", None)
        # IDs de ativação/desativação também são topologia-dependentes; após uma
        # mutação de plano, evitar herdar flags por número antigo.
        out.pop("member_active_by_id", None)
        out.pop("disabled_member_ids", None)
        return out


    def _plateau_width_efficiency_mutation(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
    ) -> Dict[str, Any]:
        """Busca a menor largura útil do platô sem perder resistência.

        Para carregamento por platô/deck, reduzir a largura dentro do intervalo do
        edital diminui travessas, contraventamentos e braço torsor do caso 60/40.
        A mutação é tardia: usa o candidato já dimensionado e só aceita largura
        que preserve ruptura/FS nos casos de projeto.
        """
        settings = cfg.get("member_sizing", {}) or {}
        if not bool(settings.get("enable_plateau_width_efficiency_mutation", True)):
            summary = self._multi_case_summary(cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
            return {"best_cfg": cfg, "summary": summary, "trace_rows": []}

        cur_cfg = self.planner.config.normalize(cfg)
        bridge = cur_cfg.get("bridge", {}) or {}
        top_profile = str(bridge.get("top_profile", "")).lower()
        load_model = str(bridge.get("load_distribution_model", "")).lower()
        if "plateau" not in top_profile and "plate" not in load_model:
            summary = self._multi_case_summary(cur_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
            return {"best_cfg": cur_cfg, "summary": summary, "trace_rows": []}

        cur_summary = self._multi_case_summary(cur_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
        if not self._summary_valid_flag(cur_summary):
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        cur_width = float(bridge.get("width_mm", 150.0))
        min_width = max(100.0, float(settings.get("plateau_width_efficiency_min_width_mm", 100.0)))
        raw_candidates = settings.get("plateau_width_efficiency_candidates_mm") or [100.0, 105.0, 110.0, 120.0, 130.0]
        candidates: List[float] = []
        for raw in raw_candidates:
            try:
                w = float(raw)
            except (TypeError, ValueError):
                continue
            if min_width - 1.0e-9 <= w <= 200.0 + 1.0e-9:
                candidates.append(round(w, 6))
        candidates.append(round(cur_width, 6))
        candidates = sorted({w for w in candidates if w <= cur_width + 1.0e-9})

        cur_break = safe_float(cur_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        cur_fs = safe_float(cur_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        cur_mass = safe_float(cur_summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
        min_break_ret = float(settings.get("plateau_width_efficiency_min_break_retention", 0.995))
        min_fs_ret = float(settings.get("plateau_width_efficiency_min_fs_retention", 0.995))
        min_saving = float(settings.get("plateau_width_efficiency_min_mass_saving_g", 8.0))
        update_footprint = bool(settings.get("plateau_width_efficiency_update_load_footprint", True))

        best_cfg = cur_cfg
        best_summary = cur_summary
        best_score = -1.0e99
        trace_rows: List[Dict[str, Any]] = []

        for width in candidates:
            if abs(width - cur_width) <= 1.0e-9:
                continue
            trial = copy.deepcopy(cur_cfg)
            tb = trial.setdefault("bridge", {})
            tb["width_mm"] = float(width)
            tb["support_contact_y_mm"] = [-0.5 * float(width), 0.5 * float(width)]
            if update_footprint:
                old_fp = safe_float(tb.get("load_footprint_width_mm"), cur_width)
                tb["load_footprint_width_mm"] = min(float(width), float(old_fp or width))
            trial = self.planner.config.normalize(trial)
            summary = self._multi_case_summary(trial, load_cases, stage_name=stage_name, tension_only=tension_only)
            if not self._summary_valid_flag(summary):
                trace_rows.append(
                    {
                        "old_width_mm": cur_width,
                        "new_width_mm": width,
                        "accepted": False,
                        "reason": "invalid_summary",
                    }
                )
                continue

            nb = safe_float(summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
            nf = safe_float(summary.get("min_fs_design_proxy"), 0.0) or 0.0
            nm = safe_float(summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
            saved = cur_mass - nm
            retained = nb >= cur_break * min_break_ret and nf >= cur_fs * min_fs_ret
            useful = saved >= min_saving or nb > cur_break + 1.0e-6 or nf > cur_fs + 1.0e-6
            accepted = bool(retained and useful)
            # Score privilegia resistência; a massa economizada só desempata.
            score = (nb * 10.0) + (nf * 30.0) + max(0.0, saved) * 0.05
            trace_rows.append(
                {
                    "old_width_mm": cur_width,
                    "new_width_mm": width,
                    "old_break_proxy_kgf": cur_break,
                    "new_break_proxy_kgf": nb,
                    "old_min_fs_design_proxy": cur_fs,
                    "new_min_fs_design_proxy": nf,
                    "old_mass_proxy_g": cur_mass,
                    "new_mass_proxy_g": nm,
                    "saved_mass_proxy_g": saved,
                    "load_footprint_width_mm": trial.get("bridge", {}).get("load_footprint_width_mm"),
                    "accepted": accepted,
                    "reason": "plateau_width_efficiency_mutation" if accepted else "not_retained",
                }
            )
            if accepted and score > best_score:
                best_score = score
                best_cfg = trial
                best_summary = summary

        return {"best_cfg": best_cfg, "summary": best_summary, "trace_rows": trace_rows}


    def _late_height_strength_mutation(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
    ) -> Dict[str, Any]:
        """Reavalia altura do platô depois das mutações de largura/seção.

        A altura maior reduz esforço axial nos banzos, mas aumenta comprimento de
        montantes, massa e sensibilidade à flambagem.  Nas etapas iniciais o
        efeito fica mascarado por outros gargalos; por isso esta mutação é tardia
        e só aceita altura que melhore ruptura/FS sem violar massa.
        """
        settings = cfg.get("member_sizing", {}) or {}
        if not bool(settings.get("enable_late_height_strength_mutation", True)):
            summary = self._multi_case_summary(cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
            return {"best_cfg": cfg, "summary": summary, "trace_rows": []}

        cur_cfg = self.planner.config.normalize(cfg)
        cur_summary = self._multi_case_summary(cur_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
        if not self._summary_valid_flag(cur_summary):
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        bridge = cur_cfg.get("bridge", {}) or {}
        planner_cfg = cur_cfg.get("planner", {}) or {}
        ms = cur_cfg.get("member_sizing", {}) or {}
        cur_h = float(bridge.get("center_height_mm", 325.0))
        h_min = safe_float(planner_cfg.get("height_min_mm"), 50.0) or 50.0
        h_max = safe_float(planner_cfg.get("height_max_mm"), 450.0) or 450.0
        h_max = min(float(h_max), float(settings.get("late_height_strength_max_mm", h_max)))
        h_min = max(float(h_min), float(settings.get("late_height_strength_min_mm", h_min)))

        raw_candidates = settings.get("late_height_strength_candidates_mm")
        if not raw_candidates:
            step = float(settings.get("late_height_strength_step_mm", 5.0))
            spread = float(settings.get("late_height_strength_spread_mm", 30.0))
            lo = max(h_min, cur_h - spread)
            hi = min(h_max, cur_h + spread)
            n = int(round((hi - lo) / max(step, 1.0)))
            raw_candidates = [lo + k * step for k in range(max(0, n) + 1)] + [cur_h]

        candidates: List[float] = []
        for raw in raw_candidates:
            try:
                h = float(raw)
            except (TypeError, ValueError):
                continue
            if h_min - 1.0e-9 <= h <= h_max + 1.0e-9:
                candidates.append(round(h, 6))
        candidates = sorted(set(candidates))

        cur_break = safe_float(cur_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        cur_fs = safe_float(cur_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        cur_mass = safe_float(cur_summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
        mass_limit = float(effective_mass_limit_g(cur_cfg))
        reserve_g = float(ms.get("late_stage_detailed_mass_reserve_g", 3.0))
        min_break_gain = float(settings.get("late_height_strength_min_break_gain_kgf", 0.30))
        min_fs_ret = float(settings.get("late_height_strength_min_fs_retention", 0.995))

        best_cfg = cur_cfg
        best_summary = cur_summary
        best_score = (cur_break * 10.0) + (cur_fs * 30.0) - max(0.0, cur_mass - 950.0) * 0.02
        trace_rows: List[Dict[str, Any]] = []

        for h in candidates:
            if abs(h - cur_h) <= 1.0e-9:
                continue
            trial = copy.deepcopy(cur_cfg)
            tb = trial.setdefault("bridge", {})
            tb["center_height_mm"] = float(h)
            # Preserve a feasible Parker/platô profile.  If the old end height is
            # higher than the new crown, normalization will clip it, but doing it
            # explicitly keeps traceability in config_used.json.
            tb["end_height_mm"] = min(float(tb.get("end_height_mm", h)), float(h))
            trial = self.planner.config.normalize(trial)
            summary = self._multi_case_summary(trial, load_cases, stage_name=stage_name, tension_only=tension_only)
            if not self._summary_valid_flag(summary):
                trace_rows.append({"old_height_mm": cur_h, "new_height_mm": h, "accepted": False, "reason": "invalid_summary"})
                continue
            nb = safe_float(summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
            nf = safe_float(summary.get("min_fs_design_proxy"), 0.0) or 0.0
            nm = safe_float(summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
            mass_ok, detailed_mass, mass_basis = self._late_stage_mass_ok(
                trial,
                proxy_mass_g=nm,
                proxy_limit_g=mass_limit,
                hard_limit_g=mass_limit,
                reserve_g=reserve_g,
                stage_name=f"{stage_name}_DETAILED_MASS",
                tension_only=tension_only,
            )
            accepted = bool(mass_ok and nb >= cur_break + min_break_gain and nf >= cur_fs * min_fs_ret)
            score = (nb * 10.0) + (nf * 30.0) - max(0.0, nm - cur_mass) * 0.02
            trace_rows.append(
                {
                    "old_height_mm": cur_h,
                    "new_height_mm": h,
                    "old_break_proxy_kgf": cur_break,
                    "new_break_proxy_kgf": nb,
                    "old_min_fs_design_proxy": cur_fs,
                    "new_min_fs_design_proxy": nf,
                    "old_mass_proxy_g": cur_mass,
                    "new_mass_proxy_g": nm,
                    "new_detailed_competition_mass_g": detailed_mass,
                    "mass_acceptance_basis": mass_basis,
                    "accepted": accepted,
                    "reason": "late_height_strength_mutation" if accepted else "not_improved_or_mass_limited",
                }
            )
            if accepted and score > best_score:
                best_score = score
                best_cfg = trial
                best_summary = summary

        return {"best_cfg": best_cfg, "summary": best_summary, "trace_rows": trace_rows}


    def _plane_bracing_efficiency_mutation(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
    ) -> Dict[str, Any]:
        """Troca o padrão de treliçamento dos planos superior/inferior para economizar massa.

        O output atual mostrou uma ponte governada por banzo superior, montantes e
        sapatas, enquanto parte relevante da massa ficava em treliçamento de plano
        inferior. Esta mutação testa alternativas simétricas de bracing e só aceita
        a troca se a ruptura e o FS forem preservados dentro de uma margem pequena.
        A massa liberada fica disponível para o empurrão final de resistência.
        """
        settings = cfg.get("member_sizing", {}) or {}
        if not bool(settings.get("enable_plane_bracing_efficiency_mutation", True)):
            summary = self._multi_case_summary(cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
            return {"best_cfg": cfg, "summary": summary, "trace_rows": []}

        cur_cfg = self.planner.config.normalize(cfg)
        cur_summary = self._multi_case_summary(cur_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
        if not self._summary_valid_flag(cur_summary):
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        cur_break = safe_float(cur_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        cur_fs = safe_float(cur_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        cur_mass = safe_float(cur_summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
        if cur_break <= 0.0 or cur_mass <= 0.0:
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        min_break_ret = float(settings.get("plane_bracing_efficiency_min_break_retention", 0.995))
        min_fs_ret = float(settings.get("plane_bracing_efficiency_min_fs_retention", 0.940))
        min_mass_saving_g = float(settings.get("plane_bracing_efficiency_min_mass_saving_g", 8.0))
        target_break = float(
            (cur_cfg.get("analysis", {}) or {}).get(
                "acceptance_min_design_breaking_load_kgf",
                settings.get("ultimate_strength_target_kgf", 120.0),
            )
        )
        # Abaixo da meta, economizar bracing só é aceitável se a segurança global
        # não cair. Caso contrário o funil produz exatamente o que apareceu no
        # output v32: mais leve, porém mais fraco.
        allow_strength_loss_below_target = bool(
            settings.get("plane_bracing_efficiency_allow_strength_loss_below_target", False)
        )
        below_strength_target = bool(cur_break < target_break)
        trials_raw = settings.get("plane_bracing_efficiency_trials") or [
            {"top": "X", "bottom": "Warren_symmetric", "cross_frame": True},
            {"top": "Warren_symmetric", "bottom": "X", "cross_frame": True},
            {"top": "Pratt_symmetric", "bottom": "X", "cross_frame": True},
            {"top": "Warren_symmetric", "bottom": "Warren_symmetric", "cross_frame": True},
            {"top": "X", "bottom": "Pratt_symmetric", "cross_frame": True},
        ]

        best_cfg = cur_cfg
        best_summary = cur_summary
        best_break = cur_break
        best_fs = cur_fs
        best_mass = cur_mass
        best_score = 0.0
        trace_rows: List[Dict[str, Any]] = []

        base_bridge = cur_cfg.get("bridge", {}) or {}
        seen: set[Tuple[str, str, bool]] = set()
        for idx, trial_def in enumerate(trials_raw, 1):
            top_mode = str(trial_def.get("top", base_bridge.get("top_chord_truss_type", "X")))
            bottom_mode = str(trial_def.get("bottom", base_bridge.get("bottom_chord_truss_type", "X")))
            cross_frame = bool(trial_def.get("cross_frame", base_bridge.get("include_cross_frame_bracing", True)))
            key = (top_mode, bottom_mode, cross_frame)
            if key in seen:
                continue
            seen.add(key)
            if (
                top_mode == str(base_bridge.get("top_chord_truss_type"))
                and bottom_mode == str(base_bridge.get("bottom_chord_truss_type"))
                and cross_frame == bool(base_bridge.get("include_cross_frame_bracing", True))
            ):
                continue

            trial = copy.deepcopy(cur_cfg)
            bridge = dict(trial.get("bridge", {}) or {})
            bridge["top_chord_truss_type"] = top_mode
            bridge["bottom_chord_truss_type"] = bottom_mode
            bridge["include_top_x_bracing"] = top_mode.lower() != "none"
            bridge["include_bottom_x_bracing"] = bottom_mode.lower() != "none"
            bridge["include_cross_frame_bracing"] = cross_frame
            trial["bridge"] = bridge
            trial = self.planner.config.normalize(trial)
            trial = self._remap_member_stick_overrides_by_geometry(cur_cfg, trial)
            trial = self.planner.config.normalize(trial)
            summary = self._multi_case_summary(trial, load_cases, stage_name=stage_name, tension_only=tension_only)

            nb = safe_float(summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
            nf = safe_float(summary.get("min_fs_design_proxy"), 0.0) or 0.0
            nm = safe_float(summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
            mass_saved = cur_mass - nm
            valid = self._summary_valid_flag(summary)
            strength_loss_ok = (
                (not below_strength_target)
                or allow_strength_loss_below_target
                or (nb >= cur_break * 0.9999 and nf >= cur_fs * 0.9999)
            )
            acceptable = (
                valid
                and mass_saved >= min_mass_saving_g
                and nb >= cur_break * min_break_ret
                and nf >= cur_fs * min_fs_ret
                and strength_loss_ok
            )
            # Peso alto para massa economizada, mas nunca aceitando perda relevante
            # de segurança antes da meta de 120 kgf.
            score = (mass_saved / max(1.0, cur_mass)) + 0.25 * (nb / max(1.0, cur_break) - 1.0)
            accepted = acceptable and score > best_score + 1.0e-12
            trace_rows.append(
                {
                    "trial_index": idx,
                    "top_chord_truss_type": top_mode,
                    "bottom_chord_truss_type": bottom_mode,
                    "include_cross_frame_bracing": cross_frame,
                    "old_break_proxy_kgf": cur_break,
                    "new_break_proxy_kgf": nb,
                    "old_min_fs_design_proxy": cur_fs,
                    "new_min_fs_design_proxy": nf,
                    "old_mass_proxy_g": cur_mass,
                    "new_mass_proxy_g": nm,
                    "mass_saved_g": mass_saved,
                    "min_break_retention": min_break_ret,
                    "min_fs_retention": min_fs_ret,
                    "strength_loss_ok": bool(strength_loss_ok),
                    "below_strength_target": bool(below_strength_target),
                    "accepted": bool(accepted),
                    "reason": "plane_bracing_efficiency_mutation" if accepted else ("not_retained" if valid else "invalid_summary"),
                }
            )
            if accepted:
                best_cfg = trial
                best_summary = summary
                best_break = nb
                best_fs = nf
                best_mass = nm
                best_score = score

        if best_cfg is not cur_cfg:
            # Marca somente a melhor tentativa como aceita no CSV final.
            best_top = str((best_cfg.get("bridge", {}) or {}).get("top_chord_truss_type"))
            best_bottom = str((best_cfg.get("bridge", {}) or {}).get("bottom_chord_truss_type"))
            best_cross = bool((best_cfg.get("bridge", {}) or {}).get("include_cross_frame_bracing", True))
            for row in trace_rows:
                row["accepted"] = bool(
                    row.get("top_chord_truss_type") == best_top
                    and row.get("bottom_chord_truss_type") == best_bottom
                    and bool(row.get("include_cross_frame_bracing")) == best_cross
                )
                if bool(row["accepted"]):
                    row["reason"] = "plane_bracing_efficiency_mutation"
        return {"best_cfg": best_cfg, "summary": best_summary, "trace_rows": trace_rows}



    def _bottom_chord_tension_donor_trim(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
    ) -> Dict[str, Any]:
        """Recupera massa do banzo inferior quando ele trabalha folgado à tração.

        Em pontes treliçadas simplesmente apoiadas com carga no tabuleiro superior,
        o banzo superior tende a governar por compressão/flambagem e o banzo inferior
        tende a trabalhar à tração.  Quando o banzo inferior mantém FS alto em todos
        os casos de carga, é mais eficiente retirar palitos dele e reinvestir essa
        massa no caminho comprimido.  A mutação é aceita apenas se a validação
        multi-loadcase preservar ruptura e FS, evitando assumir que todo banzo
        inferior será sempre tracionado.
        """
        settings = cfg.get("member_sizing", {}) or {}
        if not bool(settings.get("enable_bottom_chord_tension_donor_trim", True)):
            summary = self._multi_case_summary(cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
            return {"best_cfg": cfg, "summary": summary, "trace_rows": []}

        cur_cfg = self.planner.config.normalize(cfg)
        cur_summary = self._multi_case_summary(cur_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
        if not self._summary_valid_flag(cur_summary):
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        target_break = float((cur_cfg.get("analysis", {}) or {}).get("acceptance_min_design_breaking_load_kgf", 120.0))
        if (safe_float(cur_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0) >= target_break:
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        min_sticks = max(1, int(settings.get("bottom_chord_tension_donor_min_sticks", 1)))
        min_fs_floor = float(settings.get("bottom_chord_tension_donor_min_fs_after", 2.0))
        min_mass_saving_g = float(settings.get("bottom_chord_tension_donor_min_mass_saving_g", 12.0))
        max_compression_N = float(settings.get("bottom_chord_tension_donor_max_compression_N", 8.0))
        min_break_ret = float(settings.get("bottom_chord_tension_donor_min_break_retention", 0.995))
        min_fs_ret = float(settings.get("bottom_chord_tension_donor_min_fs_retention", 0.985))

        ref_case = self._evaluate_case_cached(cur_cfg, load_cases[0] if load_cases else "center", stage_name=stage_name, tension_only=tension_only)
        members = ref_case.get("members") or []
        bottom_ids = [int(getattr(m, "id")) for m in members if str(getattr(m, "group", "")) == "bottom_chord"]
        if not bottom_ids:
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        # Verifica se os banzos inferiores são doadores reais: pouca compressão e FS alto.
        min_bottom_fs = float("inf")
        max_bottom_compression = 0.0
        for case in (cur_summary.get("cases") or []):
            result_by_id = {
                int(r.get("member_id")): r
                for r in (case.get("member_results") or [])
                if r.get("member_id") is not None
            }
            for chk in (case.get("member_checks") or []):
                mid_raw = chk.get("member_id")
                if mid_raw is None:
                    continue
                mid = int(mid_raw)
                if mid not in bottom_ids:
                    continue
                n_val = safe_float((result_by_id.get(mid, {}) or {}).get("N_N"), 0.0) or 0.0
                if n_val < 0.0:
                    max_bottom_compression = max(max_bottom_compression, abs(float(n_val)))
                fs = safe_float(chk.get("FS_design"), None)
                if fs is None:
                    fs = safe_float(chk.get("FS_min"), None)
                if fs is not None:
                    min_bottom_fs = min(min_bottom_fs, float(fs))

        if max_bottom_compression > max_compression_N:
            return {
                "best_cfg": cur_cfg,
                "summary": cur_summary,
                "trace_rows": [{
                    "accepted": False,
                    "reason": "bottom_chord_not_tension_dominated",
                    "max_bottom_compression_N": max_bottom_compression,
                    "min_bottom_fs_before": None if min_bottom_fs == float("inf") else min_bottom_fs,
                }],
            }

        trial = copy.deepcopy(cur_cfg)
        group_map = dict(trial.get("member_sticks_by_group", {}) or {})
        old_group_n = int(safe_float(group_map.get("bottom_chord"), min_sticks) or min_sticks)
        if old_group_n <= min_sticks:
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}
        group_map["bottom_chord"] = min_sticks
        trial["member_sticks_by_group"] = group_map
        by_id = dict(trial.get("member_sticks_by_id", {}) or {})
        for mid in bottom_ids:
            by_id[str(mid)] = min_sticks
        trial["member_sticks_by_id"] = by_id
        trial = self.planner.config.normalize(trial)

        new_summary = self._multi_case_summary(trial, load_cases, stage_name=stage_name, tension_only=tension_only)
        old_break = safe_float(cur_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        old_fs = safe_float(cur_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        old_mass = safe_float(cur_summary.get("dead_weight_proxy_g"), 0.0) or 0.0
        new_break = safe_float(new_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        new_fs = safe_float(new_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        new_mass = safe_float(new_summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
        mass_saved = old_mass - new_mass

        # Recalcula FS mínimo do banzo inferior depois da mutação.
        min_bottom_fs_after = float("inf")
        for case in (new_summary.get("cases") or []):
            for chk in (case.get("member_checks") or []):
                mid_raw = chk.get("member_id")
                if mid_raw is None or int(mid_raw) not in bottom_ids:
                    continue
                fs = safe_float(chk.get("FS_design"), None)
                if fs is None:
                    fs = safe_float(chk.get("FS_min"), None)
                if fs is not None:
                    min_bottom_fs_after = min(min_bottom_fs_after, float(fs))

        accepted = (
            self._summary_valid_flag(new_summary)
            and mass_saved >= min_mass_saving_g
            and new_break >= old_break * min_break_ret
            and new_fs >= old_fs * min_fs_ret
            and (min_bottom_fs_after == float("inf") or min_bottom_fs_after >= min_fs_floor)
        )
        row = {
            "accepted": bool(accepted),
            "reason": "bottom_chord_tension_donor_trim" if accepted else "not_retained",
            "old_bottom_chord_sticks": old_group_n,
            "new_bottom_chord_sticks": min_sticks,
            "old_break_proxy_kgf": old_break,
            "new_break_proxy_kgf": new_break,
            "old_min_fs_design_proxy": old_fs,
            "new_min_fs_design_proxy": new_fs,
            "min_bottom_fs_before": None if min_bottom_fs == float("inf") else min_bottom_fs,
            "min_bottom_fs_after": None if min_bottom_fs_after == float("inf") else min_bottom_fs_after,
            "old_mass_proxy_g": old_mass,
            "new_mass_proxy_g": new_mass,
            "mass_saved_g": mass_saved,
            "max_bottom_compression_N": max_bottom_compression,
        }
        if accepted:
            return {"best_cfg": trial, "summary": new_summary, "trace_rows": [row]}
        return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": [row]}

    def _final_strength_reserve_push_dynamic(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
    ) -> Dict[str, Any]:
        """Reforço final com recálculo depois de cada órbita aceita.

        A versão estática montava uma lista de candidatos uma única vez.  Em
        pontes com vários gargalos próximos, isso consumia massa em banzos que
        eram críticos no primeiro passo, mesmo depois de um montante ou diagonal
        passar a governar.  Este método recalcula o envelope multi-loadcase a
        cada órbita aceita e volta a ordenar por FS atual, força e massa marginal.
        """
        settings = cfg.get("member_sizing", {}) or {}
        cur_cfg = self.planner.config.normalize(cfg)
        cur_summary = self._multi_case_summary(cur_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
        if not self._summary_valid_flag(cur_summary):
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        analysis = cur_cfg.get("analysis", {}) or {}
        material = cur_cfg.get("material", {}) or {}
        ms = cur_cfg.get("member_sizing", {}) or {}
        target_break = float(analysis.get("acceptance_min_design_breaking_load_kgf", 80.0))
        acceptance_fs = float(analysis.get("acceptance_min_primary_fs", 1.05))
        target_fs = float(analysis.get("target_min_fs", max(acceptance_fs, 1.5)))
        threshold_fs = float(ms.get("final_strength_push_fs_threshold", max(target_fs, acceptance_fs)))
        max_orbits = max(0, int(ms.get("final_strength_push_max_orbits", 8)))
        dyn_max_orbits_raw = ms.get("final_strength_push_dynamic_max_orbits")
        if dyn_max_orbits_raw is not None:
            max_orbits = min(max_orbits, max(1, int(dyn_max_orbits_raw)))
        max_trials = max(max_orbits, int(ms.get("final_strength_push_max_trials", 28)))
        dyn_max_trials_raw = ms.get("final_strength_push_dynamic_max_trials")
        if dyn_max_trials_raw is not None:
            max_trials = min(max_trials, max(max_orbits, int(dyn_max_trials_raw)))
        max_inc = max(1, int(ms.get("final_strength_push_max_increment_per_orbit", 1)))
        min_abs_force = float(ms.get("final_strength_push_min_abs_force_N", 30.0))
        mass_limit = float(effective_mass_limit_g(cur_cfg))
        default_mass_ratio = 1.0 if (safe_float(cur_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0) < target_break else 0.995
        target_proxy_mass = mass_limit * float(ms.get("final_strength_push_max_proxy_mass_ratio", default_mass_ratio))
        detailed_mass_reserve_g = float(ms.get("late_stage_detailed_mass_reserve_g", 3.0))
        break_ret = float(ms.get("final_strength_push_min_accept_break_retention", 0.9995))
        fs_ret = float(ms.get("final_strength_push_min_accept_fs_retention", 0.9980))
        groups = set(str(g) for g in (ms.get("final_strength_push_groups") or ["top_chord", "vertical", "diagonal"]))
        stick_mass_g = float(material.get("stick_mass_g", 1.4))
        stick_len_mm = max(1.0, float(material.get("stick_length_mm", 120.0)))
        max_default = int(analysis.get("planner_max_sticks_per_group", 12))
        max_by_group = analysis.get("planner_max_sticks_per_group_by_group", {}) or {}

        def max_for_group(group: str) -> int:
            raw = safe_float(max_by_group.get(group), None)
            return int(raw) if raw is not None else max_default

        ml_cfg = cur_cfg.get("multi_loadcase_screening", {}) or {}
        strength_case_names = [
            str(v)
            for v in (
                ms.get("sizing_load_cases")
                or ml_cfg.get("strength_governing_cases")
                or ["center", "torsion_60_40", "lateral_imperfection"]
            )
        ]
        summary_cases_for_trials = strength_case_names if bool(ms.get("final_strength_push_fast_strength_cases_only", True)) else load_cases

        best_cfg = cur_cfg
        best_summary = cur_summary
        best_break = safe_float(best_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        best_fs = safe_float(best_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        trace_rows: List[Dict[str, Any]] = []
        group_priority = {"vertical": 9.0, "top_chord": 8.0, "diagonal": 5.0, "bottom_chord": 2.0}

        for iteration in range(1, max_orbits + 1):
            if best_break >= target_break and best_fs >= target_fs:
                break

            cases = [
                self._evaluate_case_cached(best_cfg, c, stage_name=stage_name, tension_only=tension_only)
                for c in strength_case_names
            ]
            cases = [c for c in cases if self._is_selectable_case(c)]
            if not cases:
                break

            ref = cases[0]
            nodes = ref.get("nodes") or []
            members = ref.get("members") or []
            member_by_id = {int(getattr(m, "id")): m for m in members}
            try:
                partners = self.planner.map_member_to_symmetry_partners(best_cfg, nodes, members)
            except Exception:
                partners = {}

            worst_by_mid: Dict[int, Dict[str, Any]] = {}
            for case in cases:
                case_name = str(case.get("case", "unknown"))
                result_by_id = {
                    int(r.get("member_id")): r
                    for r in (case.get("member_results") or [])
                    if r.get("member_id") is not None
                }
                for chk in (case.get("member_checks") or []):
                    mid_raw = chk.get("member_id")
                    if mid_raw is None:
                        continue
                    mid = int(mid_raw)
                    m = member_by_id.get(mid)
                    if m is None:
                        continue
                    group = str(getattr(m, "group", chk.get("group", "")))
                    if group not in groups or chk.get("design_relevant") is False:
                        continue
                    fs = safe_float(chk.get("FS_design"), None)
                    if fs is None:
                        fs = safe_float(chk.get("FS_min"), None)
                    if fs is None:
                        continue
                    n_val = safe_float((result_by_id.get(mid, {}) or {}).get("N_N"), chk.get("N_N"))
                    if abs(float(n_val or 0.0)) < min_abs_force:
                        continue
                    cur = worst_by_mid.get(mid)
                    if cur is None or float(fs) < float(cur.get("FS", 1.0e99)):
                        worst_by_mid[mid] = {"FS": float(fs), "case": case_name, "group": group, "N_N": float(n_val or 0.0)}

            candidates: List[Tuple[float, Tuple[int, ...], Dict[str, Any]]] = []
            seen: set[Tuple[int, ...]] = set()
            for mid, meta in worst_by_mid.items():
                group = str(meta.get("group"))
                orbit = tuple(sorted(set([int(mid)] + [int(v) for v in partners.get(int(mid), []) if int(v) in member_by_id])))
                if orbit in seen:
                    continue
                seen.add(orbit)
                ns = [int(getattr(member_by_id[i], "n_sticks", 1)) for i in orbit]
                if not ns or max(ns) >= max_for_group(group):
                    continue
                fs_vals = [worst_by_mid.get(i, {}).get("FS") for i in orbit if worst_by_mid.get(i, {}).get("FS") is not None]
                if not fs_vals:
                    continue
                fs_min = min(float(v) for v in fs_vals)
                if fs_min > threshold_fs:
                    continue
                n_abs = max(abs(float(worst_by_mid.get(i, {}).get("N_N", 0.0))) for i in orbit if i in worst_by_mid)
                length_total = sum(float(getattr(member_by_id[i], "L", 0.0) or 0.0) for i in orbit)
                delta_mass = length_total / stick_len_mm * stick_mass_g * max_inc
                severity = max(0.01, threshold_fs - fs_min)
                # FS atual domina.  A força e o grupo só desempatarão candidatos
                # de severidade similar; isso evita gastar toda a margem em banzos
                # depois que montantes passam a governar o caso torsional.
                score = (severity ** 2) * group_priority.get(group, 1.0) * max(1.0, n_abs / 100.0) / max(0.5, delta_mass)
                candidates.append((score, orbit, {"group": group, "fs_min": fs_min, "delta_mass_g": delta_mass, "N_abs": n_abs, "case": meta.get("case")}))

            candidates.sort(key=lambda item: item[0], reverse=True)
            if not candidates:
                break

            accepted = False
            for _, orbit, meta in candidates[:max_trials]:
                trial = copy.deepcopy(best_cfg)
                by_id = trial.setdefault("member_sticks_by_id", {})
                old_ns: List[int] = []
                new_ns: List[int] = []
                for mid in orbit:
                    m = member_by_id[int(mid)]
                    old_n = max(1, int(getattr(m, "n_sticks", 1)))
                    old_ns.append(old_n)
                    new_ns.append(old_n + max_inc)
                    by_id[str(int(mid))] = old_n + max_inc
                trial = self.planner.config.normalize(trial)
                summary = self._multi_case_summary(trial, summary_cases_for_trials, stage_name=stage_name, tension_only=tension_only)
                if not self._summary_valid_flag(summary):
                    continue
                nb = safe_float(summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
                nf = safe_float(summary.get("min_fs_design_proxy"), 0.0) or 0.0
                nm = safe_float(summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
                mass_ok, detailed_mass, mass_basis = self._late_stage_mass_ok(
                    trial,
                    proxy_mass_g=nm,
                    proxy_limit_g=target_proxy_mass,
                    hard_limit_g=mass_limit,
                    reserve_g=detailed_mass_reserve_g,
                    stage_name=f"{stage_name}_DETAILED_MASS",
                    tension_only=tension_only,
                )
                if not mass_ok:
                    continue
                acceptable = nb >= best_break * break_ret and nf >= best_fs * fs_ret
                if not acceptable:
                    continue
                trace_rows.append(
                    {
                        "iteration": iteration,
                        "orbit_member_ids": ";".join(str(i) for i in orbit),
                        "group": meta.get("group"),
                        "old_n_sticks": ";".join(str(v) for v in old_ns),
                        "new_n_sticks": ";".join(str(v) for v in new_ns),
                        "FS_before": meta.get("fs_min"),
                        "N_abs_N": meta.get("N_abs"),
                        "worst_case": meta.get("case"),
                        "delta_mass_g_est": meta.get("delta_mass_g"),
                        "new_break_proxy_kgf": nb,
                        "new_min_fs_design_proxy": nf,
                        "new_mass_proxy_g": nm,
                        "new_detailed_competition_mass_g": detailed_mass,
                        "mass_acceptance_basis": mass_basis,
                        "reason": "final_strength_reserve_push_dynamic_recompute",
                    }
                )
                best_cfg = trial
                best_summary = summary
                best_break = nb
                best_fs = nf
                accepted = True
                break
            if not accepted:
                break

        # A seleção rápida usa só casos de resistência para custo.  Antes de
        # devolver ao funil, revalida no conjunto completo, incluindo auditorias
        # de contato e offsets configurados.
        full_summary = self._multi_case_summary(best_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
        return {"best_cfg": best_cfg, "summary": full_summary, "trace_rows": trace_rows}

    def _final_strength_reserve_push(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
    ) -> Dict[str, Any]:
        """Usa a margem final de massa para reforçar órbitas primárias críticas.

        Este passo é propositalmente conservador: preserva simetria, não remove nada
        e adiciona no máximo poucos palitos em órbitas que ainda governam por FS.
        Ele resolve o caso típico do output atual: banzos superiores críticos com
        4 palitos enquanto há 15-20 g de margem competitiva.
        """
        settings = cfg.get("member_sizing", {}) or {}
        if not bool(settings.get("enable_final_strength_reserve_push", True)):
            summary = self._multi_case_summary(cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
            return {"best_cfg": cfg, "summary": summary, "trace_rows": []}

        if bool(settings.get("final_strength_push_dynamic_recompute", True)):
            return self._final_strength_reserve_push_dynamic(
                cfg,
                load_cases,
                stage_name=stage_name,
                tension_only=tension_only,
            )

        cur_cfg = self.planner.config.normalize(cfg)
        cur_summary = self._multi_case_summary(cur_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
        if not self._summary_valid_flag(cur_summary):
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        analysis = cur_cfg.get("analysis", {}) or {}
        material = cur_cfg.get("material", {}) or {}
        ms = cur_cfg.get("member_sizing", {}) or {}
        target_break = float(analysis.get("acceptance_min_design_breaking_load_kgf", 80.0))
        acceptance_fs = float(analysis.get("acceptance_min_primary_fs", 1.05))
        target_fs = float(analysis.get("target_min_fs", max(acceptance_fs, 1.5)))
        # Quando a meta final é 80 kgf com FS=1,5, a triagem precisa mirar FS 1,5,
        # não apenas o FS mínimo de seleção. Caso contrário o push para antes de
        # liberar a resistência necessária para 120 kgf.
        threshold_fs = float(ms.get("final_strength_push_fs_threshold", max(target_fs, acceptance_fs)))
        max_orbits = max(0, int(ms.get("final_strength_push_max_orbits", 6)))
        max_trials = max(max_orbits, int(ms.get("final_strength_push_max_trials", 24)))
        max_inc = max(1, int(ms.get("final_strength_push_max_increment_per_orbit", 1)))
        min_abs_force = float(ms.get("final_strength_push_min_abs_force_N", 30.0))
        cur_break0 = safe_float(cur_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        min_break_gain = float(ms.get("final_strength_push_min_break_gain", 1.000))
        min_fs_gain = float(ms.get("final_strength_push_min_fs_gain", 1.000))
        min_actual_break_gain_kgf = float(
            ms.get("final_strength_push_min_actual_break_gain_kgf", 0.20 if cur_break0 < target_break else 0.0)
        )
        if cur_break0 < target_break:
            # Abaixo da meta, vários membros simétricos co-governam. Exigir ganho
            # imediato de ruptura em cada órbita bloqueia reforços cumulativos; uma
            # órbita melhora, mas outra continua governando. Aceitar ganho zero,
            # desde que não piore FS nem ruptura, permite avançar até o próximo
            # gargalo real.
            min_actual_break_gain_kgf = min(min_actual_break_gain_kgf, 0.0)
        allow_below_target = bool(ms.get("final_strength_push_allow_if_below_target", True))
        mass_limit = float(effective_mass_limit_g(cur_cfg))
        default_mass_ratio = 1.0 if cur_break0 < target_break else 0.995
        target_proxy_mass = mass_limit * float(ms.get("final_strength_push_max_proxy_mass_ratio", default_mass_ratio))
        detailed_mass_reserve_g = float(ms.get("late_stage_detailed_mass_reserve_g", 3.0))
        groups = set(str(g) for g in (ms.get("final_strength_push_groups") or ["top_chord", "vertical", "diagonal"]))
        stick_mass_g = float(material.get("stick_mass_g", 1.4))
        stick_len_mm = max(1.0, float(material.get("stick_length_mm", 120.0)))
        max_default = int(analysis.get("planner_max_sticks_per_group", 12))
        max_by_group = analysis.get("planner_max_sticks_per_group_by_group", {}) or {}

        def max_for_group(group: str) -> int:
            raw = safe_float(max_by_group.get(group), None)
            return int(raw) if raw is not None else max_default

        # Usa apenas casos governantes de resistência para identificar o gargalo.
        ml_cfg = cur_cfg.get("multi_loadcase_screening", {}) or {}
        strength_case_names = [
            str(v)
            for v in (
                ms.get("sizing_load_cases")
                or ml_cfg.get("strength_governing_cases")
                or ["center", "torsion_60_40", "lateral_imperfection"]
            )
        ]
        cases = [
            self._evaluate_case_cached(cur_cfg, c, stage_name=stage_name, tension_only=tension_only)
            for c in strength_case_names
        ]
        if not cases:
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        ref = cases[0]
        nodes = ref.get("nodes") or []
        members = ref.get("members") or []
        member_by_id = {int(getattr(m, "id")): m for m in members}
        try:
            partners = self.planner.map_member_to_symmetry_partners(cur_cfg, nodes, members)
        except Exception:
            partners = {}

        # Além da simetria entre as duas laterais (y), os trechos planos do
        # banzo superior devem preservar simetria longitudinal em torno do
        # centro do patamar superior. Isso evita resultados como um trecho
        # central com 6 palitos e seu par adjacente com 5.
        node_by_id = {int(getattr(n, "id")): n for n in nodes}
        max_top_z = max((float(getattr(n, "z", 0.0)) for n in nodes), default=0.0)
        flat_tol = float(ms.get("longitudinal_symmetry_flat_top_tol_mm", 3.0))
        enable_long_sym = bool(ms.get("longitudinal_symmetry_for_flat_top_chord", True))
        flat_top_ids: set[int] = set()
        flat_node_xs: List[float] = []

        def _rcoord(value: Any) -> float:
            return round(float(value), 3)

        for m in members:
            if str(getattr(m, "group", "")) != "top_chord":
                continue
            ni = node_by_id.get(int(getattr(m, "i")))
            nj = node_by_id.get(int(getattr(m, "j")))
            if ni is None or nj is None:
                continue
            if float(getattr(ni, "z", 0.0)) >= max_top_z - flat_tol and float(getattr(nj, "z", 0.0)) >= max_top_z - flat_tol:
                flat_top_ids.add(int(getattr(m, "id")))
                flat_node_xs.extend([float(getattr(ni, "x")), float(getattr(nj, "x"))])

        x_sym_axis = (min(flat_node_xs) + max(flat_node_xs)) * 0.5 if flat_node_xs else float(cur_cfg.get("bridge", {}).get("span_mm", 1200.0)) * 0.5
        flat_key_to_ids: Dict[Tuple[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]], List[int]] = {}

        def _point_key(n: Any) -> Tuple[float, float, float]:
            return (_rcoord(getattr(n, "x")), _rcoord(getattr(n, "y")), _rcoord(getattr(n, "z")))

        for mid in flat_top_ids:
            m = member_by_id.get(int(mid))
            if m is None:
                continue
            ni = node_by_id.get(int(getattr(m, "i")))
            nj = node_by_id.get(int(getattr(m, "j")))
            if ni is None or nj is None:
                continue
            pts = tuple(sorted([_point_key(ni), _point_key(nj)]))
            flat_key_to_ids.setdefault(("top_chord", pts), []).append(int(mid))

        def _extend_flat_top_longitudinal_orbit(orbit: Tuple[int, ...], group: str) -> Tuple[int, ...]:
            if not enable_long_sym or group != "top_chord" or not orbit:
                return orbit
            if not any(int(mid) in flat_top_ids for mid in orbit):
                return orbit
            out: set[int] = set(int(v) for v in orbit)
            transforms = [(False, False), (True, False), (False, True), (True, True)]
            for mid in list(out):
                m = member_by_id.get(int(mid))
                if m is None or int(mid) not in flat_top_ids:
                    continue
                ni = node_by_id.get(int(getattr(m, "i")))
                nj = node_by_id.get(int(getattr(m, "j")))
                if ni is None or nj is None:
                    continue
                base_pts = [_point_key(ni), _point_key(nj)]

                def _tx(pt: Tuple[float, float, float], mirror_x: bool, mirror_y: bool) -> Tuple[float, float, float]:
                    x, y, z = pt
                    if mirror_x:
                        x = _rcoord(2.0 * x_sym_axis - x)
                    if mirror_y:
                        y = _rcoord(-y)
                    return (_rcoord(x), _rcoord(y), _rcoord(z))

                for mx, my in transforms:
                    pts = tuple(sorted([_tx(base_pts[0], mx, my), _tx(base_pts[1], mx, my)]))
                    out.update(flat_key_to_ids.get(("top_chord", pts), []))
            return tuple(sorted(out))

        worst_by_mid: Dict[int, Dict[str, Any]] = {}
        for case in cases:
            case_name = str(case.get("case", "unknown"))
            result_by_id = {
                int(r.get("member_id")): r
                for r in (case.get("member_results") or [])
                if r.get("member_id") is not None
            }
            for chk in (case.get("member_checks") or []):
                mid_raw = chk.get("member_id")
                if mid_raw is None:
                    continue
                mid = int(mid_raw)
                m = member_by_id.get(mid)
                if m is None:
                    continue
                group = str(getattr(m, "group", chk.get("group", "")))
                if group not in groups or chk.get("design_relevant") is False:
                    continue
                fs = safe_float(chk.get("FS_design"), None)
                if fs is None:
                    fs = safe_float(chk.get("FS_min"), None)
                if fs is None:
                    continue
                n_val = safe_float((result_by_id.get(mid, {}) or {}).get("N_N"), chk.get("N_N"))
                n_abs = abs(float(n_val or 0.0))
                if n_abs < min_abs_force:
                    continue
                cur = worst_by_mid.get(mid)
                if cur is None or float(fs) < float(cur.get("FS", 1.0e99)):
                    worst_by_mid[mid] = {"FS": float(fs), "case": case_name, "group": group, "N_N": float(n_val or 0.0)}

        seen: set[Tuple[int, ...]] = set()
        candidates: List[Tuple[float, Tuple[int, ...], Dict[str, Any]]] = []
        group_priority = {"top_chord": 8.0, "vertical": 6.0, "diagonal": 4.0, "bottom_chord": 2.0}
        for mid, meta in worst_by_mid.items():
            orbit = tuple(sorted(set([mid] + [int(v) for v in partners.get(mid, []) if int(v) in member_by_id])))
            orbit = _extend_flat_top_longitudinal_orbit(orbit, str(meta.get("group")))
            if orbit in seen:
                continue
            seen.add(orbit)
            group = str(meta.get("group"))
            ns = [int(getattr(member_by_id[i], "n_sticks", 1)) for i in orbit]
            if max(ns) >= max_for_group(group):
                continue
            fs_vals = [worst_by_mid.get(i, {}).get("FS") for i in orbit if worst_by_mid.get(i, {}).get("FS") is not None]
            if not fs_vals:
                continue
            fs_min = min(float(v) for v in fs_vals)
            if fs_min > threshold_fs:
                continue
            length_total = sum(float(getattr(member_by_id[i], "L", 0.0) or 0.0) for i in orbit)
            delta_mass = length_total / stick_len_mm * stick_mass_g * max_inc
            n_abs = max(abs(float(worst_by_mid.get(i, {}).get("N_N", 0.0))) for i in orbit if i in worst_by_mid)
            # Abaixo da meta de 120 kgf, a ordem correta é severidade primeiro,
            # eficiência em massa depois. O critério antigo favorecia trechos muito
            # curtos com FS menos crítico e desperdiçava a última margem de massa.
            severity = max(0.01, threshold_fs - fs_min)
            score = (group_priority.get(group, 1.0) * severity * max(1.0, n_abs / 100.0)) / max(0.5, delta_mass)
            candidates.append((score, orbit, {"group": group, "fs_min": fs_min, "delta_mass_g": delta_mass, "N_abs": n_abs, "case": meta.get("case")}))

        candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = candidates[:max_trials]
        if not candidates or max_orbits <= 0:
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        best_cfg = cur_cfg
        best_summary = cur_summary
        best_break = safe_float(cur_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        best_fs = safe_float(cur_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        best_mass = safe_float(cur_summary.get("dead_weight_proxy_g"), 0.0) or 0.0
        trace_rows: List[Dict[str, Any]] = []
        used_orbits: set[Tuple[int, ...]] = set()

        for _, orbit, meta in candidates:
            if len(used_orbits) >= max_orbits:
                break
            if orbit in used_orbits:
                continue
            trial = copy.deepcopy(best_cfg)
            by_id = trial.setdefault("member_sticks_by_id", {})
            for mid in orbit:
                m = member_by_id[int(mid)]
                old_n = max(1, int(getattr(m, "n_sticks", 1)))
                by_id[str(int(mid))] = old_n + max_inc
            trial = self.planner.config.normalize(trial)
            summary = self._multi_case_summary(trial, load_cases, stage_name=stage_name, tension_only=tension_only)
            if not self._summary_valid_flag(summary):
                continue
            nb = safe_float(summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
            nf = safe_float(summary.get("min_fs_design_proxy"), 0.0) or 0.0
            nm = safe_float(summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
            mass_ok, detailed_mass, mass_basis = self._late_stage_mass_ok(
                trial,
                proxy_mass_g=nm,
                proxy_limit_g=target_proxy_mass,
                hard_limit_g=mass_limit,
                reserve_g=detailed_mass_reserve_g,
                stage_name=f"{stage_name}_DETAILED_MASS",
                tension_only=tension_only,
            )
            if not mass_ok:
                continue
            actual_break_gain = nb - best_break
            improved = (
                actual_break_gain >= min_actual_break_gain_kgf
                and (nb >= best_break * min_break_gain)
                and nf >= best_fs * 0.997
            ) or (
                min_actual_break_gain_kgf <= 0.0
                and nf >= best_fs * min_fs_gain
                and nb >= best_break * 0.9999
            )
            near_target_push = (
                allow_below_target
                and best_break < target_break
                and actual_break_gain >= min_actual_break_gain_kgf
                and nb >= best_break * 0.9999
                and nf >= best_fs * 0.997
            )
            if improved or near_target_push:
                old_ns = []
                new_ns = []
                for mid in orbit:
                    old = max(1, int(getattr(member_by_id[int(mid)], "n_sticks", 1)))
                    old_ns.append(old)
                    new_ns.append(old + max_inc)
                trace_rows.append(
                    {
                        "orbit_member_ids": ";".join(str(i) for i in orbit),
                        "group": meta.get("group"),
                        "old_n_sticks": ";".join(str(v) for v in old_ns),
                        "new_n_sticks": ";".join(str(v) for v in new_ns),
                        "FS_before": meta.get("fs_min"),
                        "N_abs_N": meta.get("N_abs"),
                        "worst_case": meta.get("case"),
                        "delta_mass_g_est": meta.get("delta_mass_g"),
                        "new_break_proxy_kgf": nb,
                        "new_min_fs_design_proxy": nf,
                        "new_mass_proxy_g": nm,
                        "new_detailed_competition_mass_g": detailed_mass,
                        "mass_acceptance_basis": mass_basis,
                        "reason": "final_strength_reserve_push_symmetric_primary_orbit",
                    }
                )
                best_cfg = trial
                best_summary = summary
                best_break = nb
                best_fs = nf
                best_mass = nm
                used_orbits.add(orbit)
                # Reconstroi membros para próximos incrementos com as novas quantidades.
                ref2 = self._evaluate_case_cached(best_cfg, strength_case_names[0], stage_name=stage_name, tension_only=tension_only)
                members2 = ref2.get("members") or []
                member_by_id = {int(getattr(m, "id")): m for m in members2}
                if best_break >= target_break and best_fs >= target_fs:
                    break

        return {"best_cfg": best_cfg, "summary": best_summary, "trace_rows": trace_rows}

    def _late_cross_group_strength_swap(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
    ) -> Dict[str, Any]:
        """Reinveste massa de membros superseguros em gargalos reais.

        O empurrão final tradicional só adiciona palitos enquanto houver massa.
        No v51, o melhor candidato ficou no limite de 1 kg; os gargalos eram
        montantes e banzos comprimidos, mas ainda havia palitos em banzos
        inferiores/sapatas/peças de baixo esforço com FS enorme.  Este passo é
        uma troca explícita: adiciona uma órbita crítica, retira tantas órbitas
        doadoras quantas forem necessárias para manter a massa, e só aceita se a
        ruptura multi-loadcase melhorar.  É deliberadamente tardio para não
        mascarar problemas de geometria nas etapas anteriores.
        """
        settings = cfg.get("member_sizing", {}) or {}
        if not bool(settings.get("enable_late_cross_group_strength_swap", True)):
            summary = self._multi_case_summary(cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
            return {"best_cfg": cfg, "summary": summary, "trace_rows": []}

        best_cfg = self.planner.config.normalize(cfg)
        ml_cfg = best_cfg.get("multi_loadcase_screening", {}) or {}
        ms_cfg = best_cfg.get("member_sizing", {}) or {}
        search_cases = [
            str(v)
            for v in (
                ms_cfg.get("late_multicase_reinvest_cases")
                or ml_cfg.get("strength_governing_cases")
                or ["center", "torsion_60_40", "torsion_70_30", "lateral_imperfection"]
            )
        ]
        best_summary = self._multi_case_summary(best_cfg, search_cases, stage_name=stage_name, tension_only=tension_only)
        if not self._summary_valid_flag(best_summary):
            return {"best_cfg": best_cfg, "summary": best_summary, "trace_rows": []}

        material = best_cfg.get("material", {}) or {}
        analysis = best_cfg.get("analysis", {}) or {}
        ms = best_cfg.get("member_sizing", {}) or {}
        ml_cfg = best_cfg.get("multi_loadcase_screening", {}) or {}
        stick_mass_g = float(material.get("stick_mass_g", 1.4))
        stick_len_mm = max(1.0, float(material.get("stick_length_mm", 120.0)))
        mass_limit = float(effective_mass_limit_g(best_cfg))
        max_mass_ratio = float(ms.get("late_cross_swap_max_proxy_mass_ratio", 1.0))
        max_proxy_mass = mass_limit * max_mass_ratio
        max_overrun_g = float(ms.get("late_cross_swap_proxy_mass_margin_g", 0.0))
        max_iterations = max(0, int(ms.get("late_cross_swap_max_iterations", 6)))
        max_critical_trials = max(1, int(ms.get("late_cross_swap_max_critical_trials", 8)))
        max_donor_orbits = max(1, int(ms.get("late_cross_swap_max_donor_orbits", 16)))
        min_break_gain = float(ms.get("late_cross_swap_min_break_gain_kgf", 0.20))
        min_fs_retention = float(ms.get("late_cross_swap_min_fs_retention", 0.985))
        min_abs_force = float(ms.get("late_cross_swap_min_abs_force_N", 25.0))
        critical_threshold = float(ms.get("late_cross_swap_critical_fs_threshold", analysis.get("target_min_fs", 1.5)))
        donor_threshold = float(ms.get("late_cross_swap_donor_fs_threshold", 8.0))
        min_by_group = (analysis.get("planner_min_sticks_per_group_by_group") or {})
        default_min = int(analysis.get("planner_min_sticks_per_group", 1))
        max_by_group = (analysis.get("planner_max_sticks_per_group_by_group") or {})
        default_max = int(analysis.get("planner_max_sticks_per_group", 12))
        critical_groups = set(ms.get("late_cross_swap_critical_groups") or ["top_chord", "vertical", "diagonal"])
        donor_groups = set(ms.get("late_cross_swap_donor_groups") or ["bottom_chord", "support_pad", "top_chord", "vertical", "diagonal"])
        case_names = [
            str(v)
            for v in (
                ms.get("late_cross_swap_cases")
                or ml_cfg.get("strength_governing_cases")
                or ["center", "torsion_60_40", "torsion_70_30", "torsion_80_20", "lateral_imperfection"]
            )
        ]

        def min_for_group(group: str) -> int:
            raw = safe_float(min_by_group.get(group), None)
            return int(raw) if raw is not None else default_min

        def max_for_group(group: str) -> int:
            raw = safe_float(max_by_group.get(group), None)
            return int(raw) if raw is not None else default_max

        def orbit_mass_delta(members_by_id: Dict[int, Any], orbit: Tuple[int, ...], delta_n: int) -> float:
            total_len = sum(float(getattr(members_by_id[int(mid)], "L", 0.0) or 0.0) for mid in orbit if int(mid) in members_by_id)
            return total_len / stick_len_mm * stick_mass_g * float(delta_n)

        trace_rows: List[Dict[str, Any]] = []
        best_break = safe_float(best_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        best_fs = safe_float(best_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        best_mass = safe_float(best_summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99

        for iteration in range(1, max_iterations + 1):
            cases = [self._evaluate_case_cached(best_cfg, c, stage_name=stage_name, tension_only=tension_only) for c in case_names]
            cases = [c for c in cases if self._is_selectable_case(c)]
            if not cases:
                break
            ref = cases[0]
            nodes = ref.get("nodes") or []
            members = ref.get("members") or []
            members_by_id = {int(getattr(m, "id")): m for m in members}
            try:
                partners = self.planner.map_member_to_symmetry_partners(best_cfg, nodes, members)
            except Exception:
                partners = {}

            worst: Dict[int, Dict[str, Any]] = {}
            for case in cases:
                cname = str(case.get("case", "unknown"))
                result_by_id = {
                    int(r.get("member_id")): r
                    for r in (case.get("member_results") or [])
                    if r.get("member_id") is not None
                }
                for chk in case.get("member_checks") or []:
                    mid_raw = chk.get("member_id")
                    if mid_raw is None:
                        continue
                    mid = int(mid_raw)
                    fs = safe_float(chk.get("FS_design"), safe_float(chk.get("FS_min"), None))
                    if fs is None:
                        continue
                    m = members_by_id.get(mid)
                    if m is None:
                        continue
                    nres = result_by_id.get(mid, {})
                    N = safe_float(nres.get("N_N"), chk.get("N_N")) or 0.0
                    cur = worst.get(mid)
                    if cur is None or float(fs) < float(cur.get("fs", 1.0e99)):
                        worst[mid] = {"fs": float(fs), "case": cname, "N_N": float(N), "group": str(getattr(m, "group", ""))}

            orbits: Dict[Tuple[int, ...], Dict[str, Any]] = {}
            for mid, meta in worst.items():
                orbit = tuple(sorted(set([mid] + [int(v) for v in partners.get(mid, []) if int(v) in members_by_id])))
                if orbit in orbits:
                    continue
                groups = [str(getattr(members_by_id[i], "group", "")) for i in orbit if i in members_by_id]
                group = groups[0] if groups else str(meta.get("group", ""))
                fs_vals = [worst.get(i, {}).get("fs") for i in orbit if worst.get(i, {}).get("fs") is not None]
                N_vals = [abs(float(worst.get(i, {}).get("N_N", 0.0))) for i in orbit]
                n_vals = [int(getattr(members_by_id[i], "n_sticks", 1)) for i in orbit if i in members_by_id]
                if not fs_vals or not n_vals:
                    continue
                orbits[orbit] = {
                    "orbit": orbit,
                    "group": group,
                    "fs_min": min(float(v) for v in fs_vals),
                    "N_abs_max": max(N_vals) if N_vals else 0.0,
                    "n_min": min(n_vals),
                    "n_max": max(n_vals),
                    "length_total": sum(float(getattr(members_by_id[i], "L", 0.0) or 0.0) for i in orbit if i in members_by_id),
                    "case": next((worst.get(i, {}).get("case") for i in orbit if i in worst), "unknown"),
                }

            critical = [
                o for o in orbits.values()
                if str(o["group"]) in critical_groups
                and float(o["fs_min"]) < critical_threshold
                and float(o["N_abs_max"]) >= min_abs_force
                and int(o["n_max"]) < max_for_group(str(o["group"]))
            ]
            donors = [
                o for o in orbits.values()
                if str(o["group"]) in donor_groups
                and float(o["fs_min"]) > donor_threshold
                and int(o["n_min"]) > min_for_group(str(o["group"]))
            ]
            critical.sort(key=lambda o: (float(o["fs_min"]), -float(o["N_abs_max"])))
            donors.sort(key=lambda o: (-float(o["fs_min"]), float(o["N_abs_max"]), -float(o["length_total"])))
            if not critical or not donors:
                break

            accepted = False
            for crit in critical[:max_critical_trials]:
                c_orbit = tuple(crit["orbit"])
                trial = copy.deepcopy(best_cfg)
                by_id = trial.setdefault("member_sticks_by_id", {})
                for mid in c_orbit:
                    by_id[str(mid)] = int(by_id.get(str(mid), getattr(members_by_id[mid], "n_sticks", 1))) + 1
                est_mass = best_mass + orbit_mass_delta(members_by_id, c_orbit, +1)
                used_donors: List[Dict[str, Any]] = []
                donor_ids_used: set[int] = set(c_orbit)
                for donor in donors[:max_donor_orbits]:
                    d_orbit = tuple(donor["orbit"])
                    if any(int(mid) in donor_ids_used for mid in d_orbit):
                        continue
                    # Do not pull material from a same-group orbit that is only marginally safe.
                    if donor["group"] == crit["group"] and float(donor["fs_min"]) < donor_threshold * 1.5:
                        continue
                    for mid in d_orbit:
                        old_n = int(by_id.get(str(mid), getattr(members_by_id[mid], "n_sticks", 1)))
                        if old_n <= min_for_group(str(donor["group"])):
                            continue
                        by_id[str(mid)] = old_n - 1
                    est_mass -= orbit_mass_delta(members_by_id, d_orbit, +1)
                    donor_ids_used.update(int(mid) for mid in d_orbit)
                    used_donors.append(donor)
                    if est_mass <= max_proxy_mass + max_overrun_g:
                        break
                if est_mass > max_proxy_mass + max_overrun_g:
                    continue
                trial = self.planner.config.normalize(trial)
                summary = self._multi_case_summary(trial, load_cases, stage_name=stage_name, tension_only=tension_only)
                if not self._summary_valid_flag(summary):
                    continue
                nb = safe_float(summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
                nf = safe_float(summary.get("min_fs_design_proxy"), 0.0) or 0.0
                nm = safe_float(summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
                if nm > max_proxy_mass + max_overrun_g:
                    continue
                if nb < best_break + min_break_gain and nf < best_fs * 1.01:
                    continue
                if nf < best_fs * min_fs_retention:
                    continue
                trace_rows.append(
                    {
                        "iteration": iteration,
                        "critical_orbit": ";".join(str(i) for i in c_orbit),
                        "critical_group": crit["group"],
                        "critical_fs_before": crit["fs_min"],
                        "critical_case_before": crit["case"],
                        "donor_orbits": "|".join(";".join(str(i) for i in d["orbit"]) for d in used_donors),
                        "donor_groups": "|".join(str(d["group"]) for d in used_donors),
                        "donor_fs_min": min((float(d["fs_min"]) for d in used_donors), default=None),
                        "old_break_proxy_kgf": best_break,
                        "new_break_proxy_kgf": nb,
                        "old_min_fs_design_proxy": best_fs,
                        "new_min_fs_design_proxy": nf,
                        "old_mass_proxy_g": best_mass,
                        "new_mass_proxy_g": nm,
                        "reason": "late_cross_group_strength_swap",
                    }
                )
                best_cfg = trial
                best_summary = summary
                best_break = nb
                best_fs = nf
                best_mass = nm
                accepted = True
                break
            if not accepted:
                break

        return {"best_cfg": best_cfg, "summary": best_summary, "trace_rows": trace_rows}


    def _late_nominal_strength_topoff(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
    ) -> Dict[str, Any]:
        """Use a pequena sobra de massa para atingir ruptura nominal.

        O ``late_cross_group_strength_swap`` é ótimo para redistribuir massa e
        corrigir o multi-loadcase, mas no v52 ele terminou conservador demais:
        deixou cerca de 70 g reais de margem e o caso nominal ficou em 107 kgf.
        Este passo é deliberadamente simples: depois que o projeto já está
        abaixo do limite de massa, reforça apenas órbitas primárias que governam
        o caso nominal configurado (por padrão, ``center``), sem retirar massa de
        outros membros.  A aceitação ainda reavalia o envelope multi-loadcase para
        impedir que a melhoria nominal piore o resultado global de forma oculta.
        """
        settings = cfg.get("member_sizing", {}) or {}
        if not bool(settings.get("enable_late_nominal_strength_topoff", True)):
            summary = self._multi_case_summary(cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
            return {"best_cfg": cfg, "summary": summary, "trace_rows": []}

        best_cfg = self.planner.config.normalize(cfg)
        best_summary = self._multi_case_summary(best_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
        if not self._summary_valid_flag(best_summary):
            return {"best_cfg": best_cfg, "summary": best_summary, "trace_rows": []}

        ms = best_cfg.get("member_sizing", {}) or {}
        analysis = best_cfg.get("analysis", {}) or {}
        material = best_cfg.get("material", {}) or {}
        target_kgf = float(ms.get("late_nominal_topoff_target_kgf", analysis.get("acceptance_min_design_breaking_load_kgf", 120.0)))
        target_case = str(ms.get("late_nominal_topoff_case", "center"))
        groups = set(str(g) for g in (ms.get("late_nominal_topoff_groups") or ["top_chord", "vertical"]))
        max_iterations = max(0, int(ms.get("late_nominal_topoff_max_iterations", 8)))
        min_abs_force = float(ms.get("late_nominal_topoff_min_abs_force_N", 40.0))
        min_nominal_gain = float(ms.get("late_nominal_topoff_min_nominal_gain_kgf", 0.20))
        min_multi_retention = float(ms.get("late_nominal_topoff_min_multi_retention", 0.995))
        max_mass = float(effective_mass_limit_g(best_cfg)) * float(ms.get("late_nominal_topoff_max_proxy_mass_ratio", 0.965))
        max_mass += float(ms.get("late_nominal_topoff_proxy_mass_margin_g", 0.0))
        max_default = int(analysis.get("planner_max_sticks_per_group", 12))
        max_by_group = analysis.get("planner_max_sticks_per_group_by_group", {}) or {}

        def max_for_group(group: str) -> int:
            raw = safe_float(max_by_group.get(group), None)
            return int(raw) if raw is not None else max_default

        best_multi_break = safe_float(best_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        best_multi_fs = safe_float(best_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        best_mass = safe_float(best_summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
        nominal_summary = self._multi_case_summary(best_cfg, [target_case], stage_name=stage_name, tension_only=tension_only)
        best_nominal_break = safe_float(nominal_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        best_nominal_fs = safe_float(nominal_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        trace_rows: List[Dict[str, Any]] = []

        for iteration in range(1, max_iterations + 1):
            if best_nominal_break >= target_kgf or best_mass >= max_mass:
                break
            case = self._evaluate_case_cached(best_cfg, target_case, stage_name=stage_name, tension_only=tension_only)
            if not self._is_selectable_case(case):
                break
            nodes = case.get("nodes") or []
            members = case.get("members") or []
            members_by_id = {int(getattr(m, "id")): m for m in members}
            result_by_id = {
                int(r.get("member_id")): r
                for r in (case.get("member_results") or [])
                if r.get("member_id") is not None
            }
            try:
                partners = self.planner.map_member_to_symmetry_partners(best_cfg, nodes, members)
            except Exception:
                partners = {}

            candidates: List[Dict[str, Any]] = []
            seen: set[Tuple[int, ...]] = set()
            for chk in case.get("member_checks") or []:
                mid_raw = chk.get("member_id")
                if mid_raw is None:
                    continue
                mid = int(mid_raw)
                m = members_by_id.get(mid)
                if m is None:
                    continue
                group = str(getattr(m, "group", chk.get("group", "")))
                if group not in groups or chk.get("design_relevant") is False:
                    continue
                fs = safe_float(chk.get("FS_design"), safe_float(chk.get("FS_min"), None))
                if fs is None:
                    continue
                N = safe_float((result_by_id.get(mid, {}) or {}).get("N_N"), chk.get("N_N")) or 0.0
                if abs(float(N)) < min_abs_force:
                    continue
                orbit = tuple(sorted(set([mid] + [int(v) for v in partners.get(mid, []) if int(v) in members_by_id])))
                if orbit in seen:
                    continue
                seen.add(orbit)
                n_vals = [int(getattr(members_by_id[i], "n_sticks", 1)) for i in orbit if i in members_by_id]
                if not n_vals or max(n_vals) >= max_for_group(group):
                    continue
                length_total = sum(float(getattr(members_by_id[i], "L", 0.0) or 0.0) for i in orbit if i in members_by_id)
                mass_delta = length_total / max(1.0, float(material.get("stick_length_mm", 120.0))) * float(material.get("stick_mass_g", 1.4))
                score = (target_kgf - best_nominal_break + 1.0) * max(0.01, float(fs) ** -2.0) * max(1.0, abs(float(N)) / 100.0) / max(0.5, mass_delta)
                candidates.append({"orbit": orbit, "group": group, "fs": float(fs), "N_N": float(N), "mass_delta_g": mass_delta, "score": score})
            candidates.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
            if not candidates:
                break

            accepted = False
            for cand in candidates[: max(1, int(ms.get("late_nominal_topoff_max_trials", 12)))]:
                trial = copy.deepcopy(best_cfg)
                by_id = trial.setdefault("member_sticks_by_id", {})
                old_ns: List[int] = []
                new_ns: List[int] = []
                for mid in cand["orbit"]:
                    old_n = int(by_id.get(str(mid), getattr(members_by_id[int(mid)], "n_sticks", 1)))
                    by_id[str(mid)] = old_n + 1
                    old_ns.append(old_n)
                    new_ns.append(old_n + 1)
                trial = self.planner.config.normalize(trial)
                nominal = self._multi_case_summary(trial, [target_case], stage_name=stage_name, tension_only=tension_only)
                full = self._multi_case_summary(trial, load_cases, stage_name=stage_name, tension_only=tension_only)
                if not self._summary_valid_flag(full):
                    continue
                nb_nom = safe_float(nominal.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
                nf_nom = safe_float(nominal.get("min_fs_design_proxy"), 0.0) or 0.0
                nb_full = safe_float(full.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
                nf_full = safe_float(full.get("min_fs_design_proxy"), 0.0) or 0.0
                nm = safe_float(full.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
                if nm > max_mass + 1.0e-9:
                    continue
                if nb_nom < best_nominal_break + min_nominal_gain and nf_nom < best_nominal_fs * 1.01:
                    continue
                if nb_full < best_multi_break * min_multi_retention or nf_full < best_multi_fs * min_multi_retention:
                    continue
                trace_rows.append(
                    {
                        "iteration": iteration,
                        "target_case": target_case,
                        "orbit_member_ids": ";".join(str(v) for v in cand["orbit"]),
                        "group": cand["group"],
                        "old_n_sticks": ";".join(str(v) for v in old_ns),
                        "new_n_sticks": ";".join(str(v) for v in new_ns),
                        "old_nominal_break_kgf": best_nominal_break,
                        "new_nominal_break_kgf": nb_nom,
                        "old_multi_break_kgf": best_multi_break,
                        "new_multi_break_kgf": nb_full,
                        "old_mass_proxy_g": best_mass,
                        "new_mass_proxy_g": nm,
                        "critical_fs_before": cand["fs"],
                        "critical_N_N": cand["N_N"],
                        "reason": "late_nominal_strength_topoff",
                    }
                )
                best_cfg = trial
                best_summary = full
                best_nominal_break = nb_nom
                best_nominal_fs = nf_nom
                best_multi_break = nb_full
                best_multi_fs = nf_full
                best_mass = nm
                accepted = True
                break
            if not accepted:
                break

        return {"best_cfg": best_cfg, "summary": best_summary, "trace_rows": trace_rows}


    def _late_multicase_strength_reinvestment(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
    ) -> Dict[str, Any]:
        """Reinveste massa em órbitas que governam o multi-loadcase.

        O topoff nominal melhora o caso central, mas tende a ignorar torsão
        60/40 e 70/30.  Esta etapa faz uma busca curta e explícita: adiciona um
        palito a uma órbita crítica e testa prefixos de órbitas doadoras reais,
        sempre reavaliando o envelope multi-loadcase.  Diferente do swap antigo,
        ela não remove massa até um alvo rígido antes de medir o efeito; cada
        prefixo doador é avaliado e só é aceito se a ruptura/FS global melhora.
        """
        settings = cfg.get("member_sizing", {}) or {}
        if not bool(settings.get("enable_late_multicase_strength_reinvestment", True)):
            summary = self._multi_case_summary(cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
            return {"best_cfg": cfg, "summary": summary, "trace_rows": []}

        best_cfg = self.planner.config.normalize(cfg)
        ml_cfg = best_cfg.get("multi_loadcase_screening", {}) or {}
        ms_cfg = best_cfg.get("member_sizing", {}) or {}
        search_cases = [
            str(v)
            for v in (
                ms_cfg.get("late_multicase_reinvest_cases")
                or ml_cfg.get("strength_governing_cases")
                or ["center", "torsion_60_40", "torsion_70_30", "lateral_imperfection"]
            )
        ]
        best_summary = self._multi_case_summary(best_cfg, search_cases, stage_name=stage_name, tension_only=tension_only)
        if not self._summary_valid_flag(best_summary):
            return {"best_cfg": best_cfg, "summary": best_summary, "trace_rows": []}

        ms = best_cfg.get("member_sizing", {}) or {}
        analysis = best_cfg.get("analysis", {}) or {}
        material = best_cfg.get("material", {}) or {}
        max_iterations = max(0, int(ms.get("late_multicase_reinvest_max_iterations", 4)))
        max_critical_trials = max(1, int(ms.get("late_multicase_reinvest_max_critical_trials", 8)))
        max_donor_prefixes = max(0, int(ms.get("late_multicase_reinvest_max_donor_prefixes", 8)))
        critical_threshold = float(ms.get("late_multicase_reinvest_critical_fs_threshold", 1.35))
        donor_threshold = float(ms.get("late_multicase_reinvest_donor_fs_threshold", 5.0))
        min_abs_force = float(ms.get("late_multicase_reinvest_min_abs_force_N", 25.0))
        min_break_gain = float(ms.get("late_multicase_reinvest_min_break_gain_kgf", 0.15))
        min_fs_gain_ratio = float(ms.get("late_multicase_reinvest_min_fs_gain_ratio", 1.002))
        max_mass = float(effective_mass_limit_g(best_cfg)) * float(ms.get("late_multicase_reinvest_max_proxy_mass_ratio", 0.995))
        max_mass += float(ms.get("late_multicase_reinvest_proxy_mass_margin_g", 0.0))
        critical_groups = set(str(g) for g in (ms.get("late_multicase_reinvest_critical_groups") or ["vertical", "top_chord"]))
        donor_groups = set(str(g) for g in (ms.get("late_multicase_reinvest_donor_groups") or ["bottom_chord", "support_pad", "diagonal", "top_chord"]))
        min_by_group = dict(analysis.get("planner_min_sticks_per_group_by_group") or {})
        default_min = int(analysis.get("planner_min_sticks_per_group", 1))
        max_by_group = dict(analysis.get("planner_max_sticks_per_group_by_group") or {})
        default_max = int(analysis.get("planner_max_sticks_per_group", 12))
        stick_len_mm = max(1.0, float(material.get("stick_length_mm", 120.0)))
        stick_mass_g = float(material.get("stick_mass_g", 1.4))

        def min_for_group(group: str) -> int:
            raw = safe_float(min_by_group.get(group), None)
            return int(raw) if raw is not None else default_min

        def max_for_group(group: str) -> int:
            raw = safe_float(max_by_group.get(group), None)
            return int(raw) if raw is not None else default_max

        def orbit_mass(members_by_id: Dict[int, Any], orbit: Tuple[int, ...]) -> float:
            return sum(float(getattr(members_by_id[int(mid)], "L", 0.0) or 0.0) for mid in orbit if int(mid) in members_by_id) / stick_len_mm * stick_mass_g

        trace_rows: List[Dict[str, Any]] = []
        best_break = safe_float(best_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        best_fs = safe_float(best_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        best_mass = safe_float(best_summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99

        for iteration in range(1, max_iterations + 1):
            cases = [self._evaluate_case_cached(best_cfg, c, stage_name=stage_name, tension_only=tension_only) for c in search_cases]
            cases = [c for c in cases if self._is_selectable_case(c)]
            if not cases:
                break
            ref = cases[0]
            nodes = ref.get("nodes") or []
            members = ref.get("members") or []
            members_by_id = {int(getattr(m, "id")): m for m in members}
            try:
                partners = self.planner.map_member_to_symmetry_partners(best_cfg, nodes, members)
            except Exception:
                partners = {}

            worst_by_mid: Dict[int, Dict[str, Any]] = {}
            for case in cases:
                cname = str(case.get("case", "unknown"))
                result_by_id = {
                    int(r.get("member_id")): r
                    for r in (case.get("member_results") or [])
                    if r.get("member_id") is not None
                }
                for chk in case.get("member_checks") or []:
                    mid_raw = chk.get("member_id")
                    if mid_raw is None or chk.get("design_relevant") is False:
                        continue
                    mid = int(mid_raw)
                    fs = safe_float(chk.get("FS_design"), safe_float(chk.get("FS_min"), None))
                    if fs is None or mid not in members_by_id:
                        continue
                    N = safe_float((result_by_id.get(mid, {}) or {}).get("N_N"), chk.get("N_N")) or 0.0
                    old = worst_by_mid.get(mid)
                    if old is None or float(fs) < float(old.get("fs", 1.0e99)):
                        worst_by_mid[mid] = {"fs": float(fs), "case": cname, "N_N": float(N), "group": str(getattr(members_by_id[mid], "group", ""))}

            orbit_data: Dict[Tuple[int, ...], Dict[str, Any]] = {}
            for mid, meta in worst_by_mid.items():
                orbit = tuple(sorted(set([int(mid)] + [int(v) for v in partners.get(int(mid), []) if int(v) in members_by_id])))
                if orbit in orbit_data:
                    continue
                fs_vals = [worst_by_mid.get(i, {}).get("fs") for i in orbit if worst_by_mid.get(i, {}).get("fs") is not None]
                n_vals = [int(getattr(members_by_id[i], "n_sticks", 1)) for i in orbit if i in members_by_id]
                N_vals = [abs(float(worst_by_mid.get(i, {}).get("N_N", 0.0))) for i in orbit if i in worst_by_mid]
                if not fs_vals or not n_vals:
                    continue
                group = str(getattr(members_by_id[orbit[0]], "group", meta.get("group", "")))
                orbit_data[orbit] = {
                    "orbit": orbit,
                    "group": group,
                    "fs_min": min(float(v) for v in fs_vals),
                    "N_abs_max": max(N_vals) if N_vals else 0.0,
                    "n_min": min(n_vals),
                    "n_max": max(n_vals),
                    "mass_delta_g": orbit_mass(members_by_id, orbit),
                    "case": next((worst_by_mid.get(i, {}).get("case") for i in orbit if i in worst_by_mid), "unknown"),
                }

            critical = [
                o for o in orbit_data.values()
                if str(o["group"]) in critical_groups
                and float(o["fs_min"]) < critical_threshold
                and float(o["N_abs_max"]) >= min_abs_force
                and int(o["n_max"]) < max_for_group(str(o["group"]))
            ]
            donors = [
                o for o in orbit_data.values()
                if str(o["group"]) in donor_groups
                and float(o["fs_min"]) > donor_threshold
                and int(o["n_min"]) > min_for_group(str(o["group"]))
            ]
            critical.sort(key=lambda o: (float(o["fs_min"]), -float(o["N_abs_max"])))
            donors.sort(key=lambda o: (-float(o["fs_min"]), float(o["N_abs_max"]), -float(o["mass_delta_g"])))
            if not critical:
                break

            accepted = False
            best_trial: Dict[str, Any] | None = None
            for crit in critical[:max_critical_trials]:
                c_orbit = tuple(crit["orbit"])
                # Avalia prefixos progressivos de doadores; não força remoção
                # excessiva antes de medir o efeito estrutural real.
                donor_prefixes: List[List[Dict[str, Any]]] = [[]]
                prefix: List[Dict[str, Any]] = []
                for donor in donors[:max_donor_prefixes]:
                    d_orbit = tuple(donor["orbit"])
                    if any(int(mid) in c_orbit for mid in d_orbit):
                        continue
                    if donor["group"] == crit["group"] and float(donor["fs_min"]) < donor_threshold * 1.5:
                        continue
                    prefix = prefix + [donor]
                    donor_prefixes.append(prefix)

                for used_donors in donor_prefixes:
                    trial = copy.deepcopy(best_cfg)
                    by_id = trial.setdefault("member_sticks_by_id", {})
                    for mid in c_orbit:
                        by_id[str(mid)] = int(by_id.get(str(mid), getattr(members_by_id[int(mid)], "n_sticks", 1))) + 1
                    for donor in used_donors:
                        for mid in tuple(donor["orbit"]):
                            if int(mid) in c_orbit:
                                continue
                            old_n = int(by_id.get(str(mid), getattr(members_by_id[int(mid)], "n_sticks", 1)))
                            if old_n > min_for_group(str(donor["group"])):
                                by_id[str(mid)] = old_n - 1
                    trial = self.planner.config.normalize(trial)
                    summary = self._multi_case_summary(trial, search_cases, stage_name=stage_name, tension_only=tension_only)
                    if not self._summary_valid_flag(summary):
                        continue
                    nb = safe_float(summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
                    nf = safe_float(summary.get("min_fs_design_proxy"), 0.0) or 0.0
                    nm = safe_float(summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
                    if nm > max_mass + 1.0e-9:
                        continue
                    if nb < best_break + min_break_gain and nf < best_fs * min_fs_gain_ratio:
                        continue
                    score = (nb - best_break) * 10.0 + (nf - best_fs) * 100.0 - max(0.0, nm - best_mass) * 0.02
                    candidate = {
                        "score": score,
                        "summary": summary,
                        "cfg": trial,
                        "crit": crit,
                        "donors": used_donors,
                        "nb": nb,
                        "nf": nf,
                        "nm": nm,
                    }
                    if best_trial is None or float(candidate["score"]) > float(best_trial["score"]):
                        best_trial = candidate

            if best_trial is None:
                break

            crit = best_trial["crit"]
            used_donors = best_trial["donors"]
            trace_rows.append(
                {
                    "iteration": iteration,
                    "critical_orbit": ";".join(str(i) for i in crit["orbit"]),
                    "critical_group": crit["group"],
                    "critical_fs_before": crit["fs_min"],
                    "critical_case_before": crit["case"],
                    "donor_orbits": "|".join(";".join(str(i) for i in d["orbit"]) for d in used_donors),
                    "donor_groups": "|".join(str(d["group"]) for d in used_donors),
                    "old_break_proxy_kgf": best_break,
                    "new_break_proxy_kgf": best_trial["nb"],
                    "old_min_fs_design_proxy": best_fs,
                    "new_min_fs_design_proxy": best_trial["nf"],
                    "old_mass_proxy_g": best_mass,
                    "new_mass_proxy_g": best_trial["nm"],
                    "reason": "late_multicase_strength_reinvestment",
                }
            )
            best_cfg = best_trial["cfg"]
            best_summary = best_trial["summary"]
            best_break = best_trial["nb"]
            best_fs = best_trial["nf"]
            best_mass = best_trial["nm"]
            accepted = True
            if not accepted:
                break

        if bool(ms.get("late_multicase_reinvest_full_revalidate", False)):
            best_summary = self._multi_case_summary(best_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
        return {"best_cfg": best_cfg, "summary": best_summary, "trace_rows": trace_rows}


    def _late_basic_7030_target_recovery(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
    ) -> Dict[str, Any]:
        """Recupera metas mínimas de forma curta e rastreável.

        O v55 terminou com massa em torno de 945 g, nominal em 87 kgf e 70/30
        em 75 kgf.  Havia margem real, mas as etapas anteriores rodavam em ordem
        desfavorável: o topoff nominal vinha antes do reinvestimento multi-case.
        Este passo roda no fim do funil e faz uma busca curta:

        * primeiro eleva o caso nominal até a meta mínima configurada;
        * depois reforça poucas órbitas críticas do caso 70/30;
        * por fim remove doadores um por um até voltar à massa-alvo.

        Diferente da busca combinatória anterior, este método avalia poucos
        candidatos por iteração, para não travar o run_cli.
        """
        settings = cfg.get("member_sizing", {}) or {}
        if not bool(settings.get("enable_late_basic_7030_target_recovery", True)):
            summary = self._multi_case_summary(cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
            return {"best_cfg": cfg, "summary": summary, "trace_rows": []}

        base_cfg = self.planner.config.normalize(cfg)
        ms0 = base_cfg.get("member_sizing", {}) or {}
        target_nominal = float(ms0.get("late_basic_target_nominal_kgf", 100.0))
        target_multi = float(ms0.get("late_basic_target_7030_kgf", 80.0))
        target_case = str(ms0.get("late_basic_target_case", "center"))
        search_cases = [str(v) for v in (ms0.get("late_basic_target_cases") or ["center", "torsion_60_40", "torsion_70_30"])]
        max_mass = float(effective_mass_limit_g(base_cfg)) * float(ms0.get("late_basic_target_max_proxy_mass_ratio", 0.995))
        max_mass += float(ms0.get("late_basic_target_proxy_mass_margin_g", 0.0))
        hard_mass = float(effective_mass_limit_g(base_cfg))
        max_add_orbits = max(0, int(ms0.get("late_basic_fast_max_critical_adds", 3)))
        donor_threshold = float(ms0.get("late_basic_donor_fs_threshold", 5.0))
        min_abs_force = float(ms0.get("late_basic_min_abs_force_N", 25.0))
        critical_groups = set(str(v) for v in (ms0.get("late_basic_multicase_critical_groups") or ["vertical", "top_chord"]))
        donor_groups = set(str(v) for v in (ms0.get("late_basic_multicase_donor_groups") or ["support_pad", "bottom_chord", "diagonal", "top_chord"]))
        analysis = base_cfg.get("analysis", {}) or {}
        min_by_group = dict(analysis.get("planner_min_sticks_per_group_by_group") or {})
        default_min = int(analysis.get("planner_min_sticks_per_group", 1))
        max_by_group = dict(analysis.get("planner_max_sticks_per_group_by_group") or {})
        default_max = int(analysis.get("planner_max_sticks_per_group", 12))

        def min_for_group(group: str) -> int:
            raw = safe_float(min_by_group.get(group), None)
            return int(raw) if raw is not None else default_min

        def max_for_group(group: str) -> int:
            raw = safe_float(max_by_group.get(group), None)
            return int(raw) if raw is not None else default_max

        def summarize(c: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            full = self._multi_case_summary(c, search_cases, stage_name=stage_name, tension_only=tension_only)
            nominal = self._multi_case_summary(c, [target_case], stage_name=stage_name, tension_only=tension_only)
            return full, nominal

        def collect_orbits(c: Dict[str, Any], cases: List[str]) -> Tuple[List[Dict[str, Any]], Dict[int, Any]]:
            evals = [self._evaluate_case_cached(c, name, stage_name=stage_name, tension_only=tension_only) for name in cases]
            evals = [ev for ev in evals if self._is_selectable_case(ev)]
            if not evals:
                return [], {}
            nodes = evals[0].get("nodes") or []
            members = evals[0].get("members") or []
            members_by_id = {int(getattr(m, "id")): m for m in members}
            try:
                partners = self.planner.map_member_to_symmetry_partners(c, nodes, members)
            except Exception:
                partners = {}
            worst_by_mid: Dict[int, Dict[str, Any]] = {}
            for ev in evals:
                cname = str(ev.get("case", "unknown"))
                result_by_id = {
                    int(r.get("member_id")): r
                    for r in (ev.get("member_results") or [])
                    if r.get("member_id") is not None
                }
                for chk in ev.get("member_checks") or []:
                    mid_raw = chk.get("member_id")
                    if mid_raw is None or chk.get("design_relevant") is False:
                        continue
                    mid = int(mid_raw)
                    if mid not in members_by_id:
                        continue
                    fs = safe_float(chk.get("FS_design"), safe_float(chk.get("FS_min"), None))
                    if fs is None:
                        continue
                    N = safe_float((result_by_id.get(mid, {}) or {}).get("N_N"), chk.get("N_N")) or 0.0
                    old = worst_by_mid.get(mid)
                    if old is None or float(fs) < float(old.get("fs", 1.0e99)):
                        worst_by_mid[mid] = {"fs": float(fs), "N_N": float(N), "case": cname}
            rows: List[Dict[str, Any]] = []
            seen: set[Tuple[int, ...]] = set()
            for mid, meta in worst_by_mid.items():
                orbit = tuple(sorted(set([int(mid)] + [int(v) for v in partners.get(int(mid), []) if int(v) in members_by_id])))
                if orbit in seen:
                    continue
                seen.add(orbit)
                group = str(getattr(members_by_id[orbit[0]], "group", ""))
                n_vals = [int(getattr(members_by_id[i], "n_sticks", 1)) for i in orbit if i in members_by_id]
                fs_vals = [worst_by_mid.get(i, {}).get("fs") for i in orbit if worst_by_mid.get(i, {}).get("fs") is not None]
                N_vals = [abs(float(worst_by_mid.get(i, {}).get("N_N", 0.0))) for i in orbit if i in worst_by_mid]
                if not n_vals or not fs_vals:
                    continue
                rows.append({
                    "orbit": orbit,
                    "group": group,
                    "fs_min": min(float(v) for v in fs_vals),
                    "N_abs_max": max(N_vals) if N_vals else 0.0,
                    "n_min": min(n_vals),
                    "n_max": max(n_vals),
                    "case": next((worst_by_mid.get(i, {}).get("case") for i in orbit if i in worst_by_mid), "unknown"),
                })
            return rows, members_by_id

        trace_rows: List[Dict[str, Any]] = []
        best_cfg = copy.deepcopy(base_cfg)

        # 1) Topoff nominal curto: reaproveita o algoritmo existente, mas com
        # meta 100 kgf, não 120 kgf.  Isso evita gastar massa em reforços que não
        # ajudam o requisito pedido pelo usuário.
        top_cfg = copy.deepcopy(best_cfg)
        top_cfg.setdefault("member_sizing", {})["late_nominal_topoff_target_kgf"] = target_nominal
        top_cfg.setdefault("member_sizing", {})["late_nominal_topoff_max_proxy_mass_ratio"] = min(0.995, float(ms0.get("late_basic_target_max_proxy_mass_ratio", 0.995)))
        topoff = self._late_nominal_strength_topoff(top_cfg, search_cases, stage_name=f"{stage_name}_NOMINAL", tension_only=tension_only)
        if topoff.get("trace_rows"):
            for r in topoff.get("trace_rows") or []:
                trace_rows.append({"phase": "nominal_topoff_to_minimum", **r})
            best_cfg = topoff["best_cfg"]

        full, nominal = summarize(best_cfg)
        if not self._summary_valid_flag(full):
            return {"best_cfg": best_cfg, "summary": full, "trace_rows": trace_rows}
        best_break = safe_float(full.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        best_fs = safe_float(full.get("min_fs_design_proxy"), 0.0) or 0.0
        best_mass = safe_float(full.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
        best_nominal = safe_float(nominal.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0

        # 2) Adiciona poucas órbitas críticas do envelope 70/30.  Permite uma
        # pequena excursão acima do alvo de massa durante o reforço, porque a
        # etapa seguinte remove doadores e revalida o resultado.
        for iteration in range(1, max_add_orbits + 1):
            if best_break >= target_multi:
                break
            orbits, members_by_id = collect_orbits(best_cfg, search_cases)
            critical = [
                o for o in orbits
                if str(o["group"]) in critical_groups
                and float(o["N_abs_max"]) >= min_abs_force
                and int(o["n_max"]) < max_for_group(str(o["group"]))
            ]
            critical.sort(key=lambda o: (float(o["fs_min"]), -float(o["N_abs_max"])))
            accepted = False
            for cand in critical[:max(1, int(ms0.get("late_basic_max_trials", 16)))]:
                trial = copy.deepcopy(best_cfg)
                by_id = trial.setdefault("member_sticks_by_id", {})
                for mid in tuple(cand["orbit"]):
                    by_id[str(mid)] = int(by_id.get(str(mid), getattr(members_by_id[int(mid)], "n_sticks", 1))) + 1
                trial = self.planner.config.normalize(trial)
                full_trial, nominal_trial = summarize(trial)
                if not self._summary_valid_flag(full_trial):
                    continue
                nb = safe_float(full_trial.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
                nf = safe_float(full_trial.get("min_fs_design_proxy"), 0.0) or 0.0
                nm = safe_float(full_trial.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
                nn = safe_float(nominal_trial.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
                if nm > hard_mass + 25.0:
                    continue
                # O v57 deixava esta etapa praticamente inoperante porque
                # exigia que UM único reforço já atingisse a meta nominal final
                # (115/120 kgf).  Isso bloqueava reforços progressivos: por
                # exemplo, uma órbita vertical podia elevar o caso center de
                # ~93 para ~100 kgf e o 70/30 de ~67 para ~76 kgf, mas era
                # descartada por ainda não chegar a 115 kgf.  A lógica correta
                # para uma etapa tardia é aceitar avanço mensurável, depois
                # aparar massa e continuar buscando a próxima órbita.
                min_nominal_gain = float(ms0.get("late_basic_min_nominal_gain_kgf", 0.10))
                min_multi_gain = float(ms0.get("late_basic_min_multi_gain_kgf", 0.03))
                nominal_improved = nn >= best_nominal + min_nominal_gain
                multi_improved = nb >= best_break + min_multi_gain or nf >= best_fs * 1.001
                if not (nominal_improved or multi_improved):
                    continue
                trace_rows.append({
                    "phase": "multicase_70_30_add",
                    "iteration": iteration,
                    "critical_orbit": ";".join(str(i) for i in cand["orbit"]),
                    "critical_group": cand["group"],
                    "critical_fs_before": cand["fs_min"],
                    "critical_case_before": cand["case"],
                    "donor_orbits": "",
                    "old_nominal_break_kgf": best_nominal,
                    "new_nominal_break_kgf": nn,
                    "old_break_proxy_kgf": best_break,
                    "new_break_proxy_kgf": nb,
                    "old_min_fs_design_proxy": best_fs,
                    "new_min_fs_design_proxy": nf,
                    "old_mass_proxy_g": best_mass,
                    "new_mass_proxy_g": nm,
                    "reason": "late_basic_7030_target_recovery",
                })
                best_cfg, best_break, best_fs, best_mass, best_nominal = trial, nb, nf, nm, nn
                accepted = True
                break
            if not accepted:
                break

        # 3) Se a massa passou do alvo, remove um doador por vez.  Cada remoção
        # é reavaliada; rejeita qualquer uma que derrube o nominal abaixo de 100
        # ou o multi abaixo de 80 depois que a meta já foi atingida.
        trim_iter = 0
        while best_mass > max_mass + 1.0e-9 and trim_iter < max(1, int(ms0.get("late_basic_max_donor_prefixes", 8))):
            trim_iter += 1
            orbits, members_by_id = collect_orbits(best_cfg, search_cases)
            donors = [
                o for o in orbits
                if str(o["group"]) in donor_groups
                and float(o["fs_min"]) > donor_threshold
                and int(o["n_min"]) > min_for_group(str(o["group"]))
            ]
            donors.sort(key=lambda o: (-float(o["fs_min"]), float(o["N_abs_max"])))
            best_trim: Dict[str, Any] | None = None
            for donor in donors[:max(1, int(ms0.get("late_basic_max_trials", 16)))]:
                trial = copy.deepcopy(best_cfg)
                by_id = trial.setdefault("member_sticks_by_id", {})
                for mid in tuple(donor["orbit"]):
                    old_n = int(by_id.get(str(mid), getattr(members_by_id[int(mid)], "n_sticks", 1)))
                    if old_n > min_for_group(str(donor["group"])):
                        by_id[str(mid)] = old_n - 1
                trial = self.planner.config.normalize(trial)
                full_trial, nominal_trial = summarize(trial)
                if not self._summary_valid_flag(full_trial):
                    continue
                nb = safe_float(full_trial.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
                nf = safe_float(full_trial.get("min_fs_design_proxy"), 0.0) or 0.0
                nm = safe_float(full_trial.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
                nn = safe_float(nominal_trial.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
                if nm >= best_mass - 1.0e-9:
                    continue
                # Se o reforço progressivo ainda não atingiu a meta nominal,
                # não faz sentido proibir qualquer trim.  O trim só precisa
                # preservar quase todo o ganho recém-obtido.  Quando a meta já
                # foi atingida, aí sim ela vira piso rígido.
                if best_nominal >= target_nominal:
                    if nn < target_nominal:
                        continue
                else:
                    nominal_retention = float(ms0.get("late_basic_trim_min_nominal_retention", 0.985))
                    if nn < best_nominal * nominal_retention:
                        continue
                if best_break >= target_multi and nb < target_multi:
                    continue
                if nb < best_break * 0.98:
                    continue
                score = (max_mass - nm) * 0.02 + nb * 10.0 + nf * 60.0
                cand = {"score": score, "cfg": trial, "nb": nb, "nf": nf, "nm": nm, "nn": nn, "donor": donor}
                if best_trim is None or float(cand["score"]) > float(best_trim["score"]):
                    best_trim = cand
            if best_trim is None:
                break
            donor = best_trim["donor"]
            trace_rows.append({
                "phase": "mass_backtrim_after_multicase_add",
                "iteration": trim_iter,
                "critical_orbit": "",
                "critical_group": "",
                "critical_fs_before": "",
                "critical_case_before": "",
                "donor_orbits": ";".join(str(i) for i in donor["orbit"]),
                "donor_groups": donor["group"],
                "old_nominal_break_kgf": best_nominal,
                "new_nominal_break_kgf": best_trim["nn"],
                "old_break_proxy_kgf": best_break,
                "new_break_proxy_kgf": best_trim["nb"],
                "old_min_fs_design_proxy": best_fs,
                "new_min_fs_design_proxy": best_trim["nf"],
                "old_mass_proxy_g": best_mass,
                "new_mass_proxy_g": best_trim["nm"],
                "reason": "late_basic_7030_target_recovery",
            })
            best_cfg = best_trim["cfg"]
            best_break = best_trim["nb"]
            best_fs = best_trim["nf"]
            best_mass = best_trim["nm"]
            best_nominal = best_trim["nn"]

        # 4) Segunda passada curta: depois que a massa voltou para baixo do
        # limite, ainda pode existir uma órbita crítica que cabe junto com um
        # doador seguro.  O v58 fica exatamente nesse caso: reforçar a órbita
        # crítica do banzo superior no 70/30 e remover uma órbita de sapata ou
        # de banzo de ponta melhora o envelope sem violar 1 kg.  A busca abaixo
        # testa pares (add crítico + remove doador) em um único passo, evitando
        # a excursão de massa que bloqueava a iteração anterior.
        pair_rounds = max(0, int(ms0.get("late_basic_post_trim_pair_rounds", 1)))
        for pair_it in range(1, pair_rounds + 1):
            orbits, members_by_id = collect_orbits(best_cfg, search_cases)
            critical = [
                o for o in orbits
                if str(o["group"]) in critical_groups
                and float(o["N_abs_max"]) >= min_abs_force
                and int(o["n_max"]) < max_for_group(str(o["group"]))
            ]
            critical.sort(key=lambda o: (float(o["fs_min"]), -float(o["N_abs_max"])))
            donors = [
                o for o in orbits
                if str(o["group"]) in donor_groups
                and float(o["fs_min"]) > donor_threshold
                and int(o["n_min"]) > min_for_group(str(o["group"]))
            ]
            donors.sort(key=lambda o: (-float(o["fs_min"]), float(o["N_abs_max"])))
            best_pair: Dict[str, Any] | None = None
            for crit in critical[:max(1, int(ms0.get("late_basic_max_trials", 16)))]:
                for donor in donors[:max(1, int(ms0.get("late_basic_max_donor_prefixes", 8)))]:
                    # Não remover do mesmo grupo/orbita que acabou de receber reforço.
                    if set(crit["orbit"]) & set(donor["orbit"]):
                        continue
                    trial = copy.deepcopy(best_cfg)
                    by_id = trial.setdefault("member_sticks_by_id", {})
                    for mid in tuple(crit["orbit"]):
                        by_id[str(mid)] = int(by_id.get(str(mid), getattr(members_by_id[int(mid)], "n_sticks", 1))) + 1
                    for mid in tuple(donor["orbit"]):
                        old_n = int(by_id.get(str(mid), getattr(members_by_id[int(mid)], "n_sticks", 1)))
                        if old_n > min_for_group(str(donor["group"])):
                            by_id[str(mid)] = old_n - 1
                    trial = self.planner.config.normalize(trial)
                    full_trial, nominal_trial = summarize(trial)
                    if not self._summary_valid_flag(full_trial):
                        continue
                    nb = safe_float(full_trial.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
                    nf = safe_float(full_trial.get("min_fs_design_proxy"), 0.0) or 0.0
                    nm = safe_float(full_trial.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
                    nn = safe_float(nominal_trial.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
                    if nm > hard_mass + 1.0e-9:
                        continue
                    min_nominal_gain = float(ms0.get("late_basic_min_nominal_gain_kgf", 0.10))
                    min_multi_gain = float(ms0.get("late_basic_min_multi_gain_kgf", 0.03))
                    if nn < best_nominal * 0.995:
                        continue
                    if nb < best_break * 0.995 or nf < best_fs * 0.995:
                        continue
                    improved = nn >= best_nominal + min_nominal_gain or nb >= best_break + min_multi_gain or nf >= best_fs * 1.005
                    if not improved:
                        continue
                    score = nb * 20.0 + nf * 80.0 + nn * 0.35 - max(0.0, nm - max_mass) * 0.10
                    cand = {"score": score, "cfg": trial, "nb": nb, "nf": nf, "nm": nm, "nn": nn, "crit": crit, "donor": donor}
                    if best_pair is None or float(cand["score"]) > float(best_pair["score"]):
                        best_pair = cand
            if best_pair is None:
                break
            crit = best_pair["crit"]
            donor = best_pair["donor"]
            trace_rows.append({
                "phase": "post_trim_pair_add_and_remove",
                "iteration": pair_it,
                "critical_orbit": ";".join(str(i) for i in crit["orbit"]),
                "critical_group": crit["group"],
                "critical_fs_before": crit["fs_min"],
                "critical_case_before": crit["case"],
                "donor_orbits": ";".join(str(i) for i in donor["orbit"]),
                "donor_groups": donor["group"],
                "old_nominal_break_kgf": best_nominal,
                "new_nominal_break_kgf": best_pair["nn"],
                "old_break_proxy_kgf": best_break,
                "new_break_proxy_kgf": best_pair["nb"],
                "old_min_fs_design_proxy": best_fs,
                "new_min_fs_design_proxy": best_pair["nf"],
                "old_mass_proxy_g": best_mass,
                "new_mass_proxy_g": best_pair["nm"],
                "reason": "late_basic_7030_target_recovery_post_trim_pair",
            })
            best_cfg = best_pair["cfg"]
            best_break = best_pair["nb"]
            best_fs = best_pair["nf"]
            best_mass = best_pair["nm"]
            best_nominal = best_pair["nn"]

        final_summary = self._multi_case_summary(best_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
        return {"best_cfg": best_cfg, "summary": final_summary, "trace_rows": trace_rows}


    def _force_support_pad_member_overrides(
        self,
        cfg: Dict[str, Any],
        n_sticks: int,
    ) -> Dict[str, Any]:
        """Garante que sapatas sigam o grupo mesmo quando há overrides por ID."""
        out = copy.deepcopy(cfg)
        try:
            _, members, _, _ = self.planner.geometry.generate(out)
        except Exception:
            return out
        by_id = dict(out.get("member_sticks_by_id", {}) or {})
        changed = False
        for m in members:
            if str(getattr(m, "group", "")) != "support_pad":
                continue
            by_id[str(int(getattr(m, "id")))] = int(n_sticks)
            changed = True
        if changed:
            out["member_sticks_by_id"] = by_id
        return out


    def _support_pad_capacity_push(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
    ) -> Dict[str, Any]:
        """Reforça sapatas de apoio quando a ruptura é limitada por reação.

        O relatório nominal atual mostra que os apoios têm pouca margem. Para
        buscar 100 kgf, não basta reforçar banzos: os quatro nós ativos de apoio
        precisam aceitar reações acima de ~25 kgf. Esta etapa aumenta
        `support_pad` em grupo, preservando simetria, e usa o PostProcessor
        adaptativo para converter sapata extra em linhas efetivas de contato.
        """
        settings = cfg.get("member_sizing", {}) or {}
        if not bool(settings.get("enable_support_pad_capacity_push", True)):
            summary = self._multi_case_summary(cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
            return {"best_cfg": cfg, "summary": summary, "trace_rows": []}

        cur_cfg = self.planner.config.normalize(cfg)
        cur_summary = self._multi_case_summary(cur_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
        if not self._summary_valid_flag(cur_summary):
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        ms = cur_cfg.get("member_sizing", {}) or {}
        analysis = cur_cfg.get("analysis", {}) or {}
        target_kgf = float(
            ms.get(
                "support_pad_push_target_kgf",
                ms.get(
                    "ultimate_strength_target_kgf",
                    analysis.get("acceptance_min_design_breaking_load_kgf", 120.0),
                ),
            )
        )
        max_group_sticks = int(ms.get("support_pad_push_max_group_sticks", 6))
        mass_limit = float(effective_mass_limit_g(cur_cfg))
        current_break0 = safe_float(cur_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        default_ratio = 1.0 if current_break0 < target_kgf else 0.995
        target_proxy_mass = mass_limit * float(ms.get("support_pad_push_max_proxy_mass_ratio", default_ratio))
        default_margin = 0.0 if current_break0 < target_kgf else 5.0
        proxy_margin = float(ms.get("support_pad_push_proxy_mass_margin_g", default_margin))
        target_proxy_mass = min(target_proxy_mass, mass_limit - proxy_margin)
        detailed_mass_reserve_g = float(ms.get("late_stage_detailed_mass_reserve_g", 3.0))
        min_break_ret = float(ms.get("support_pad_push_min_break_retention", 0.995))
        min_fs_ret = float(ms.get("support_pad_push_min_fs_retention", 0.995))
        require_gain_if_not_support_limited = bool(ms.get("support_pad_push_require_gain_if_not_support_limited", True))
        min_actual_break_gain_kgf = float(ms.get("support_pad_push_min_actual_break_gain_kgf", 0.10))

        best_cfg = cur_cfg
        best_summary = cur_summary
        best_break = safe_float(cur_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        best_fs = safe_float(cur_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        best_mass = safe_float(cur_summary.get("dead_weight_proxy_g"), 0.0) or 0.0
        trace_rows: List[Dict[str, Any]] = []
        if best_mass > target_proxy_mass + 1.0e-9:
            return {
                "best_cfg": cur_cfg,
                "summary": cur_summary,
                "trace_rows": [
                    {
                        "old_support_pad_sticks": None,
                        "new_support_pad_sticks": None,
                        "accepted": False,
                        "reason": "skipped_no_proxy_mass_margin",
                        "old_break_proxy_kgf": best_break,
                        "new_break_proxy_kgf": best_break,
                        "new_min_fs_design_proxy": best_fs,
                        "new_mass_proxy_g": best_mass,
                        "target_proxy_mass_g": target_proxy_mass,
                        "nominal_support_break_kgf_before": None,
                    }
                ],
            }

        # Usa o caso central para ler as margens de apoio, pois ele representa o
        # ensaio nominal. Os load cases completos continuam na validação abaixo.
        center_case = self._evaluate_case_cached(best_cfg, "center", stage_name=stage_name, tension_only=tension_only)
        support_fs_values = [
            safe_float(r.get("FS_support_reaction"), None)
            for r in (center_case.get("support_checks") or [])
            if bool(r.get("support_active_vertical", False))
        ]
        support_fs_values = [float(v) for v in support_fs_values if v is not None]
        min_support_fs = min(support_fs_values) if support_fs_values else None
        support_break_nominal = (80.0 * min_support_fs) if min_support_fs is not None else float("inf")

        group_map = dict(best_cfg.get("member_sticks_by_group", {}) or {})
        current_n = int(safe_float(group_map.get("support_pad"), 3) or 3)
        initial_support_pad_n = current_n
        # O caso torsional pode ser governado por apoio mesmo quando o caso central
        # já parece suficiente. Permite uma tentativa extra; depois disso, só
        # continua se a margem de apoio nominal ainda estiver abaixo da meta.
        force_one_attempt = bool(best_break < target_kgf and support_break_nominal >= target_kgf)

        while current_n < max_group_sticks and (support_break_nominal < target_kgf or (force_one_attempt and current_n == initial_support_pad_n)):
            trial = copy.deepcopy(best_cfg)
            trial_group = dict(trial.get("member_sticks_by_group", {}) or {})
            trial_group["support_pad"] = current_n + 1
            trial["member_sticks_by_group"] = trial_group
            trial = self.planner.config.normalize(trial)
            trial = self._force_support_pad_member_overrides(trial, current_n + 1)
            trial = self.planner.config.normalize(trial)

            summary = self._multi_case_summary(trial, load_cases, stage_name=stage_name, tension_only=tension_only)
            if not self._summary_valid_flag(summary):
                break

            nb = safe_float(summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
            nf = safe_float(summary.get("min_fs_design_proxy"), 0.0) or 0.0
            nm = safe_float(summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
            mass_ok, detailed_mass, mass_basis = self._late_stage_mass_ok(
                trial,
                proxy_mass_g=nm,
                proxy_limit_g=target_proxy_mass,
                hard_limit_g=mass_limit,
                reserve_g=detailed_mass_reserve_g,
                stage_name=f"{stage_name}_DETAILED_MASS",
                tension_only=tension_only,
            )
            if not mass_ok:
                trace_rows.append(
                    {
                        "old_support_pad_sticks": current_n,
                        "new_support_pad_sticks": current_n + 1,
                        "accepted": False,
                        "reason": "above_mass_target",
                        "old_break_proxy_kgf": best_break,
                        "new_break_proxy_kgf": nb,
                        "new_min_fs_design_proxy": nf,
                        "new_mass_proxy_g": nm,
                        "new_detailed_competition_mass_g": detailed_mass,
                        "mass_acceptance_basis": mass_basis,
                        "target_proxy_mass_g": target_proxy_mass,
                        "nominal_support_break_kgf_before": support_break_nominal,
                    }
                )
                break

            support_limited_before = bool(support_break_nominal < target_kgf)
            actual_gain = nb - best_break
            acceptable = nb >= best_break * min_break_ret and nf >= best_fs * min_fs_ret
            if acceptable and require_gain_if_not_support_limited and not support_limited_before:
                acceptable = actual_gain >= min_actual_break_gain_kgf
            trace_rows.append(
                {
                    "old_support_pad_sticks": current_n,
                    "new_support_pad_sticks": current_n + 1,
                    "accepted": bool(acceptable),
                    "reason": "support_pad_capacity_push" if acceptable else "not_retained",
                    "old_break_proxy_kgf": best_break,
                    "new_break_proxy_kgf": nb,
                    "new_min_fs_design_proxy": nf,
                    "new_mass_proxy_g": nm,
                    "new_detailed_competition_mass_g": detailed_mass,
                    "mass_acceptance_basis": mass_basis,
                    "nominal_support_break_kgf_before": support_break_nominal,
                    "actual_break_gain_kgf": actual_gain,
                    "support_limited_before": support_limited_before,
                }
            )
            if not acceptable:
                break

            best_cfg = trial
            best_summary = summary
            best_break = nb
            best_fs = nf
            best_mass = nm
            current_n += 1

            center_case = self._evaluate_case_cached(best_cfg, "center", stage_name=stage_name, tension_only=tension_only)
            support_fs_values = [
                safe_float(r.get("FS_support_reaction"), None)
                for r in (center_case.get("support_checks") or [])
                if bool(r.get("support_active_vertical", False))
            ]
            support_fs_values = [float(v) for v in support_fs_values if v is not None]
            min_support_fs = min(support_fs_values) if support_fs_values else None
            support_break_nominal = (80.0 * min_support_fs) if min_support_fs is not None else float("inf")

        return {"best_cfg": best_cfg, "summary": best_summary, "trace_rows": trace_rows}

    def _final_mass_symmetry_trim(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
    ) -> Dict[str, Any]:
        """Remove 1 palito de órbitas simétricas folgadas quando a massa passou do alvo.

        Diferente do cleanup topológico, este passo nunca desativa membro; ele só
        reduz n_sticks em órbitas primárias com FS folgado, preservando simetria
        entre as duas laterais. Serve para corrigir o caso típico: a ponte fica
        forte, mas 5-10 g acima do limite por reforços finais.
        """
        settings = cfg.get("member_sizing", {}) or {}
        if not bool(settings.get("enable_final_mass_symmetry_trim", True)):
            summary = self._multi_case_summary(cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
            return {"best_cfg": cfg, "summary": summary, "trace_rows": []}

        cur_cfg = self.planner.config.normalize(cfg)
        cur_summary = self._multi_case_summary(cur_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
        if not self._summary_valid_flag(cur_summary):
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        analysis = cur_cfg.get("analysis", {}) or {}
        material = cur_cfg.get("material", {}) or {}
        mass_limit = float(effective_mass_limit_g(cur_cfg))
        target_proxy_mass = mass_limit * float(settings.get("final_mass_trim_target_proxy_mass_ratio", 0.990))
        cur_mass = safe_float(cur_summary.get("dead_weight_proxy_g"), 0.0) or 0.0
        target_break = float(analysis.get("acceptance_min_design_breaking_load_kgf", 80.0))
        cur_break = safe_float(cur_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        require_strength_first = bool(settings.get("final_mass_trim_only_after_strength_target", True))
        strength_margin = float(settings.get("final_mass_trim_strength_target_ratio", 0.995))
        if require_strength_first and cur_break < target_break * strength_margin and cur_mass <= mass_limit + 1.0e-9:
            return {
                "best_cfg": cur_cfg,
                "summary": cur_summary,
                "trace_rows": [
                    {
                        "accepted": False,
                        "reason": "skipped_under_strength_target",
                        "current_break_proxy_kgf": cur_break,
                        "target_break_proxy_kgf": target_break,
                        "current_mass_proxy_g": cur_mass,
                        "mass_limit_g": mass_limit,
                    }
                ],
            }
        if cur_mass <= target_proxy_mass + 1.0e-9:
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        fs_threshold = float(settings.get("final_mass_trim_fs_threshold", 1.22))
        min_break_ret = float(settings.get("final_mass_trim_min_break_retention", 0.985))
        min_fs_ret = float(settings.get("final_mass_trim_min_fs_retention", 0.985))
        max_trials = max(1, int(settings.get("final_mass_trim_max_trials", 12)))
        groups = set(str(g) for g in (settings.get("final_mass_trim_groups") or ["top_chord", "vertical", "diagonal"]))
        min_by_group = cur_cfg.get("minimum_sticks_by_group", {}) or {}
        stick_mass_g = float(material.get("stick_mass_g", 1.4))
        stick_len_mm = max(1.0, float(material.get("stick_length_mm", 120.0)))

        def min_for_group(group: str) -> int:
            raw = safe_float(min_by_group.get(group), None)
            if raw is not None:
                return int(raw)
            return {"top_chord": 4, "bottom_chord": 2, "vertical": 2, "diagonal": 2, "support_pad": 2}.get(group, 1)

        ml_cfg = cur_cfg.get("multi_loadcase_screening", {}) or {}
        case_names = [
            str(v)
            for v in (
                settings.get("sizing_load_cases")
                or ml_cfg.get("strength_governing_cases")
                or ["center", "torsion_60_40", "lateral_imperfection"]
            )
        ]
        cases = [
            self._evaluate_case_cached(cur_cfg, c, stage_name=stage_name, tension_only=tension_only)
            for c in case_names
        ]
        if not cases:
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}
        ref = cases[0]
        nodes = ref.get("nodes") or []
        members = ref.get("members") or []
        member_by_id = {int(getattr(m, "id")): m for m in members}
        try:
            partners = self.planner.map_member_to_symmetry_partners(cur_cfg, nodes, members)
        except Exception:
            partners = {}

        worst_by_mid: Dict[int, Dict[str, Any]] = {}
        for case in cases:
            cname = str(case.get("case", "unknown"))
            res_by_id = {int(r.get("member_id")): r for r in (case.get("member_results") or []) if r.get("member_id") is not None}
            for chk in (case.get("member_checks") or []):
                mid_raw = chk.get("member_id")
                if mid_raw is None:
                    continue
                mid = int(mid_raw)
                m = member_by_id.get(mid)
                if m is None:
                    continue
                group = str(getattr(m, "group", chk.get("group", "")))
                if group not in groups or chk.get("design_relevant") is False:
                    continue
                fs = safe_float(chk.get("FS_design"), None)
                if fs is None:
                    fs = safe_float(chk.get("FS_min"), None)
                if fs is None:
                    continue
                n_val = safe_float((res_by_id.get(mid, {}) or {}).get("N_N"), chk.get("N_N"))
                cur = worst_by_mid.get(mid)
                if cur is None or float(fs) < float(cur.get("FS", 1.0e99)):
                    worst_by_mid[mid] = {"FS": float(fs), "case": cname, "group": group, "N_N": float(n_val or 0.0)}

        seen: set[Tuple[int, ...]] = set()
        candidates: List[Tuple[float, Tuple[int, ...], Dict[str, Any]]] = []
        for mid, meta in worst_by_mid.items():
            group = str(meta.get("group"))
            orbit = tuple(sorted(set([int(mid)] + [int(v) for v in partners.get(int(mid), []) if int(v) in member_by_id])))
            if orbit in seen:
                continue
            seen.add(orbit)
            ns = [int(getattr(member_by_id[i], "n_sticks", 1)) for i in orbit]
            if not ns or min(ns) <= min_for_group(group):
                continue
            fs_vals = [worst_by_mid.get(i, {}).get("FS") for i in orbit if worst_by_mid.get(i, {}).get("FS") is not None]
            if not fs_vals:
                continue
            fs_min = min(float(v) for v in fs_vals)
            if fs_min < fs_threshold:
                continue
            length_total = sum(float(getattr(member_by_id[i], "L", 0.0) or 0.0) for i in orbit)
            delta_mass = length_total / stick_len_mm * stick_mass_g
            n_abs = max(abs(float(worst_by_mid.get(i, {}).get("N_N", 0.0))) for i in orbit if i in worst_by_mid)
            # Prioriza órbitas longas, folgadas e pouco solicitadas.
            score = (fs_min * max(0.2, 1.0 / max(1.0, n_abs / 50.0)) * max(0.5, delta_mass))
            candidates.append((score, orbit, {"group": group, "fs_min": fs_min, "delta_mass_g_est": delta_mass, "N_abs_N": n_abs, "case": meta.get("case")}))

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_cfg = cur_cfg
        best_summary = cur_summary
        best_break = safe_float(cur_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        best_fs = safe_float(cur_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        best_mass = cur_mass
        trace_rows: List[Dict[str, Any]] = []

        for _, orbit, meta in candidates[:max_trials]:
            if best_mass <= target_proxy_mass + 1.0e-9:
                break
            trial = copy.deepcopy(best_cfg)
            by_id = trial.setdefault("member_sticks_by_id", {})
            old_ns: List[int] = []
            new_ns: List[int] = []
            group = str(meta.get("group"))
            ok = True
            for mid in orbit:
                m = member_by_id[int(mid)]
                old_n = max(1, int(getattr(m, "n_sticks", 1)))
                if old_n <= min_for_group(group):
                    ok = False
                    break
                old_ns.append(old_n)
                new_ns.append(old_n - 1)
                by_id[str(int(mid))] = old_n - 1
            if not ok:
                continue
            trial = self.planner.config.normalize(trial)
            summary = self._multi_case_summary(trial, load_cases, stage_name=stage_name, tension_only=tension_only)
            if not self._summary_valid_flag(summary):
                continue
            nb = safe_float(summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
            nf = safe_float(summary.get("min_fs_design_proxy"), 0.0) or 0.0
            nm = safe_float(summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
            acceptable = nm < best_mass - 1.0e-9 and nb >= best_break * min_break_ret and nf >= best_fs * min_fs_ret
            trace_rows.append(
                {
                    "orbit_member_ids": ";".join(str(i) for i in orbit),
                    "group": group,
                    "old_n_sticks": ";".join(str(v) for v in old_ns),
                    "new_n_sticks": ";".join(str(v) for v in new_ns),
                    "FS_before": meta.get("fs_min"),
                    "N_abs_N": meta.get("N_abs_N"),
                    "worst_case": meta.get("case"),
                    "delta_mass_g_est": -float(meta.get("delta_mass_g_est", 0.0)),
                    "new_break_proxy_kgf": nb,
                    "new_min_fs_design_proxy": nf,
                    "new_mass_proxy_g": nm,
                    "target_proxy_mass_g": target_proxy_mass,
                    "accepted": bool(acceptable),
                    "reason": "final_mass_symmetry_trim" if acceptable else "trim_not_retained",
                }
            )
            if not acceptable:
                continue
            best_cfg = trial
            best_summary = summary
            best_break = nb
            best_fs = nf
            best_mass = nm
            ref2 = self._evaluate_case_cached(best_cfg, case_names[0], stage_name=stage_name, tension_only=tension_only)
            members2 = ref2.get("members") or []
            member_by_id = {int(getattr(m, "id")): m for m in members2}

        return {"best_cfg": best_cfg, "summary": best_summary, "trace_rows": trace_rows}

    def _primary_symmetry_orbit_audit(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Audita órbitas simétricas para pegar diferenças de n_sticks/FS.

        O relatório visual pode parecer assimétrico quando o caso de carga é assimétrico,
        mas diferenças reais de n_sticks em uma órbita primária são erro de projeto.
        Este helper não altera a ponte; ele só escreve diagnóstico.
        """
        summary = self._multi_case_summary(
            cfg,
            load_cases,
            stage_name=stage_name,
            tension_only=tension_only,
        )
        cases = summary.get("cases") or []
        if not cases:
            return []

        ref = cases[0]
        nodes = ref.get("nodes") or []
        members = ref.get("members") or []
        member_by_id = {int(getattr(m, "id")): m for m in members}
        analysis = cfg.get("analysis", {}) or {}
        primary_groups = set(
            analysis.get(
                "global_failure_groups",
                ["bottom_chord", "top_chord", "vertical", "diagonal", "support_pad"],
            )
        )

        try:
            partners = self.planner.map_member_to_symmetry_partners(cfg, nodes, members)
        except Exception:
            partners = {}

        member_sizing = cfg.get("member_sizing", {}) or {}
        bridge = cfg.get("bridge", {}) or {}
        span = float(bridge.get("span_mm", 1200.0))
        node_by_id = {int(getattr(n, "id")): n for n in nodes}
        flat_tol = float(member_sizing.get("longitudinal_symmetry_flat_top_tol_mm", 3.0))
        geom_tol = max(1.0, flat_tol)

        def _member_points(mid: int):
            m = member_by_id.get(int(mid))
            if m is None:
                return None
            ni = node_by_id.get(int(getattr(m, "i")))
            nj = node_by_id.get(int(getattr(m, "j")))
            if ni is None or nj is None:
                return None
            return ni, nj

        def _is_flat_top_chord(mid: int) -> bool:
            m = member_by_id.get(int(mid))
            pts = _member_points(int(mid))
            if m is None or pts is None:
                return False
            if str(getattr(m, "group", "")) != "top_chord":
                return False
            ni, nj = pts
            if getattr(ni, "level", "top") != "top" or getattr(nj, "level", "top") != "top":
                return False
            return abs(float(getattr(ni, "z")) - float(getattr(nj, "z"))) <= flat_tol

        def _points_match(target, cand) -> bool:
            tx, ty, tz = target
            return (
                abs(float(getattr(cand, "x")) - tx) <= geom_tol
                and abs(float(getattr(cand, "y")) - ty) <= geom_tol
                and abs(float(getattr(cand, "z")) - tz) <= geom_tol
            )

        def _is_longitudinal_mirror(src_mid: int, cand_mid: int) -> bool:
            if not _is_flat_top_chord(src_mid) or not _is_flat_top_chord(cand_mid):
                return False
            src = _member_points(src_mid)
            cand = _member_points(cand_mid)
            if src is None or cand is None:
                return False
            a, b = src
            c, d = cand
            target_a = (span - float(getattr(a, "x")), float(getattr(a, "y")), float(getattr(a, "z")))
            target_b = (span - float(getattr(b, "x")), float(getattr(b, "y")), float(getattr(b, "z")))
            return (
                (_points_match(target_a, c) and _points_match(target_b, d))
                or (_points_match(target_a, d) and _points_match(target_b, c))
            )

        def _extend_flat_top_longitudinal_orbit(orbit: tuple[int, ...], group: str) -> tuple[int, ...]:
            # Corrige a assimetria longitudinal do banzo superior no platô central.
            # A órbita lateral pura pode gerar 10/51 com n diferente de 12/53.
            # Aqui a órbita é ampliada também pelo espelho longitudinal x -> span - x,
            # mas apenas para trechos realmente planos do top_chord, evitando alterar arcos/rampas.
            if str(group) != "top_chord":
                return tuple(sorted(set(int(v) for v in orbit)))

            out = set(int(v) for v in orbit)
            changed = True
            while changed:
                changed = False
                for src_mid in list(out):
                    if not _is_flat_top_chord(src_mid):
                        continue
                    for cand_mid, cand_member in member_by_id.items():
                        cand_mid = int(cand_mid)
                        if cand_mid in out:
                            continue
                        if str(getattr(cand_member, "group", "")) != "top_chord":
                            continue
                        if _is_longitudinal_mirror(src_mid, cand_mid):
                            out.add(cand_mid)
                            changed = True

            return tuple(sorted(out))

        # Pega o pior FS por membro entre casos de projeto. Isso evita comparar
        # uma cor visual isolada com outra carga diferente.
        fs_by_mid: Dict[int, float] = {}
        case_by_mid: Dict[int, str] = {}
        for case in cases:
            case_name = str(case.get("case", "unknown"))
            for chk in (case.get("member_checks") or []):
                mid_raw = chk.get("member_id")
                if mid_raw is None:
                    continue
                mid = int(mid_raw)
                fs = safe_float(chk.get("FS_design"), None)
                if fs is None:
                    fs = safe_float(chk.get("FS_min"), None)
                if fs is None:
                    continue
                old = fs_by_mid.get(mid)
                if old is None or float(fs) < old:
                    fs_by_mid[mid] = float(fs)
                    case_by_mid[mid] = case_name

        seen: set[Tuple[int, ...]] = set()
        rows: List[Dict[str, Any]] = []
        for mid, m in sorted(member_by_id.items()):
            group = str(getattr(m, "group", ""))
            if group not in primary_groups:
                continue
            orbit = tuple(sorted(set([mid] + [int(v) for v in partners.get(mid, []) if int(v) in member_by_id])))
            orbit = _extend_flat_top_longitudinal_orbit(orbit, group)
            if orbit in seen:
                continue
            seen.add(orbit)
            ns = [int(getattr(member_by_id[i], "n_sticks", 1)) for i in orbit]
            fs_vals = [fs_by_mid.get(i) for i in orbit if fs_by_mid.get(i) is not None]
            fs_min = min(fs_vals) if fs_vals else None
            fs_max = max(fs_vals) if fs_vals else None
            rows.append(
                {
                    "orbit_member_ids": ";".join(str(i) for i in orbit),
                    "group": group,
                    "n_sticks_min": min(ns) if ns else None,
                    "n_sticks_max": max(ns) if ns else None,
                    "FS_min": fs_min,
                    "FS_max": fs_max,
                    "FS_spread": (fs_max - fs_min) if fs_min is not None and fs_max is not None else None,
                    "worst_case_members": ";".join(f"{i}:{case_by_mid.get(i, '')}" for i in orbit),
                    "n_sticks_asymmetry_flag": bool(ns and min(ns) != max(ns)),
                    "fs_spread_diagnostic_flag": bool(fs_min is not None and fs_max is not None and (fs_max - fs_min) > 0.10),
                    "asymmetry_flag": bool(ns and min(ns) != max(ns)),
                }
            )
        return rows

    def _rebalance_primary_sticks_by_symmetry(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
    ) -> Dict[str, Any]:
        """Troca palitos de órbitas primárias folgadas para órbitas críticas.

        Diferente de reinvestir massa nova, este passo tenta ser quase neutro em massa:
        remove 1 palito de uma órbita do mesmo grupo com FS folgado e adiciona 1 palito
        em uma órbita crítica simétrica. Isso corrige o caso típico visto no output:
        alguns banzos superiores com 5 palitos e FS > 1.15, enquanto outro par simétrico
        de banzo ainda está com 4 palitos e FS < 1.0.
        """
        settings = cfg.get("member_sizing", {}) or {}
        if not bool(settings.get("enable_post_reinvest_rebalance", True)):
            summary = self._multi_case_summary(cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
            return {"best_cfg": cfg, "summary": summary, "trace_rows": []}

        cur_cfg = self.planner.config.normalize(cfg)
        cur_summary = self._multi_case_summary(cur_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
        if not self._summary_valid_flag(cur_summary):
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        analysis = cur_cfg.get("analysis", {}) or {}
        material = cur_cfg.get("material", {}) or {}
        member_sizing = cur_cfg.get("member_sizing", {}) or {}

        # Alias usado pelos helpers de simetria longitudinal.
        # Corrige NameError: ms is not defined.
        ms = member_sizing
        mass_limit = float(effective_mass_limit_g(cur_cfg))
        competitive_ratio = float(member_sizing.get("competitive_mass_target_ratio", 0.98))
        target_proxy_mass = mass_limit * float(member_sizing.get("rebalance_target_proxy_mass_ratio", competitive_ratio))
        current_mass = safe_float(cur_summary.get("dead_weight_proxy_g"), 0.0) or 0.0

        max_net_mass = float(member_sizing.get("rebalance_max_net_mass_g", 3.0))
        max_swaps = max(1, int(member_sizing.get("rebalance_max_swaps", 4)))
        fs_threshold = float(member_sizing.get("rebalance_fs_threshold", analysis.get("acceptance_min_primary_fs", 1.05)))
        donor_threshold = float(member_sizing.get("rebalance_donor_fs_threshold", 1.16))
        groups = set(member_sizing.get("rebalance_groups", ["top_chord", "vertical", "diagonal"]) or [])
        stick_mass_g = float(material.get("stick_mass_g", 1.4))
        stick_len_mm = max(1.0, float(material.get("stick_length_mm", 115.0)))
        max_default = int(analysis.get("planner_max_sticks_per_group", 12))
        max_by_group = analysis.get("planner_max_sticks_per_group_by_group", {}) or {}
        min_by_group = (cur_cfg.get("minimum_sticks_by_group", {}) or {})

        def max_for_group(group: str) -> int:
            raw = safe_float(max_by_group.get(group), None)
            return int(raw) if raw is not None else max_default

        def min_for_group(group: str) -> int:
            raw = safe_float(min_by_group.get(group), None)
            if raw is not None:
                return int(raw)
            return {"top_chord": 4, "bottom_chord": 2, "vertical": 2, "diagonal": 2, "support_pad": 2}.get(group, 1)

        ml_cfg = cur_cfg.get("multi_loadcase_screening", {}) or {}
        case_names = [
            str(v)
            for v in (
                member_sizing.get("sizing_load_cases")
                or ml_cfg.get("strength_governing_cases")
                or ["center", "torsion_60_40", "lateral_imperfection"]
            )
        ]
        cases = [
            self._evaluate_case_cached(cur_cfg, c, stage_name=stage_name, tension_only=tension_only)
            for c in case_names
        ]
        if not cases:
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}
        ref = cases[0]
        nodes = ref.get("nodes") or []
        members = ref.get("members") or []
        member_by_id = {int(getattr(m, "id")): m for m in members}
        try:
            partners = self.planner.map_member_to_symmetry_partners(cur_cfg, nodes, members)
        except Exception:
            partners = {}

        # Além da simetria entre as duas laterais (y), os trechos planos do
        # banzo superior devem preservar simetria longitudinal em torno do
        # centro do patamar superior. Isso evita resultados como um trecho
        # central com 6 palitos e seu par adjacente com 5.
        node_by_id = {int(getattr(n, "id")): n for n in nodes}
        max_top_z = max((float(getattr(n, "z", 0.0)) for n in nodes), default=0.0)
        flat_tol = float(ms.get("longitudinal_symmetry_flat_top_tol_mm", 3.0))
        enable_long_sym = bool(ms.get("longitudinal_symmetry_for_flat_top_chord", True))
        flat_top_ids: set[int] = set()
        flat_node_xs: List[float] = []

        def _rcoord(value: Any) -> float:
            return round(float(value), 3)

        for m in members:
            if str(getattr(m, "group", "")) != "top_chord":
                continue
            ni = node_by_id.get(int(getattr(m, "i")))
            nj = node_by_id.get(int(getattr(m, "j")))
            if ni is None or nj is None:
                continue
            if float(getattr(ni, "z", 0.0)) >= max_top_z - flat_tol and float(getattr(nj, "z", 0.0)) >= max_top_z - flat_tol:
                flat_top_ids.add(int(getattr(m, "id")))
                flat_node_xs.extend([float(getattr(ni, "x")), float(getattr(nj, "x"))])

        x_sym_axis = (min(flat_node_xs) + max(flat_node_xs)) * 0.5 if flat_node_xs else float(cur_cfg.get("bridge", {}).get("span_mm", 1200.0)) * 0.5
        flat_key_to_ids: Dict[Tuple[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]], List[int]] = {}

        def _point_key(n: Any) -> Tuple[float, float, float]:
            return (_rcoord(getattr(n, "x")), _rcoord(getattr(n, "y")), _rcoord(getattr(n, "z")))

        for mid in flat_top_ids:
            m = member_by_id.get(int(mid))
            if m is None:
                continue
            ni = node_by_id.get(int(getattr(m, "i")))
            nj = node_by_id.get(int(getattr(m, "j")))
            if ni is None or nj is None:
                continue
            pts = tuple(sorted([_point_key(ni), _point_key(nj)]))
            flat_key_to_ids.setdefault(("top_chord", pts), []).append(int(mid))

        def _extend_flat_top_longitudinal_orbit(orbit: Tuple[int, ...], group: str) -> Tuple[int, ...]:
            if not enable_long_sym or group != "top_chord" or not orbit:
                return orbit
            if not any(int(mid) in flat_top_ids for mid in orbit):
                return orbit
            out: set[int] = set(int(v) for v in orbit)
            transforms = [(False, False), (True, False), (False, True), (True, True)]
            for mid in list(out):
                m = member_by_id.get(int(mid))
                if m is None or int(mid) not in flat_top_ids:
                    continue
                ni = node_by_id.get(int(getattr(m, "i")))
                nj = node_by_id.get(int(getattr(m, "j")))
                if ni is None or nj is None:
                    continue
                base_pts = [_point_key(ni), _point_key(nj)]

                def _tx(pt: Tuple[float, float, float], mirror_x: bool, mirror_y: bool) -> Tuple[float, float, float]:
                    x, y, z = pt
                    if mirror_x:
                        x = _rcoord(2.0 * x_sym_axis - x)
                    if mirror_y:
                        y = _rcoord(-y)
                    return (_rcoord(x), _rcoord(y), _rcoord(z))

                for mx, my in transforms:
                    pts = tuple(sorted([_tx(base_pts[0], mx, my), _tx(base_pts[1], mx, my)]))
                    out.update(flat_key_to_ids.get(("top_chord", pts), []))
            return tuple(sorted(out))

        worst_by_mid: Dict[int, Dict[str, Any]] = {}
        for case in cases:
            case_name = str(case.get("case", "unknown"))
            for chk in (case.get("member_checks") or []):
                mid_raw = chk.get("member_id")
                if mid_raw is None:
                    continue
                mid = int(mid_raw)
                m = member_by_id.get(mid)
                if m is None:
                    continue
                group = str(getattr(m, "group", chk.get("group", "")))
                if group not in groups or chk.get("design_relevant") is False:
                    continue
                fs = safe_float(chk.get("FS_design"), None)
                if fs is None:
                    fs = safe_float(chk.get("FS_min"), None)
                if fs is None:
                    continue
                cur = worst_by_mid.get(mid)
                if cur is None or float(fs) < float(cur.get("FS", 1.0e99)):
                    worst_by_mid[mid] = {"FS": float(fs), "case": case_name, "group": group}

        seen: set[Tuple[int, ...]] = set()
        orbits: List[Dict[str, Any]] = []
        for mid, meta in worst_by_mid.items():
            orbit = tuple(sorted(set([mid] + [int(v) for v in partners.get(mid, []) if int(v) in member_by_id])))
            orbit = _extend_flat_top_longitudinal_orbit(orbit, str(meta.get("group")))
            if orbit in seen:
                continue
            seen.add(orbit)
            group = str(meta.get("group"))
            fs_vals = [worst_by_mid.get(i, {}).get("FS") for i in orbit if worst_by_mid.get(i, {}).get("FS") is not None]
            if not fs_vals:
                continue
            ns = [int(getattr(member_by_id[i], "n_sticks", 1)) for i in orbit]
            lengths = [float(getattr(member_by_id[i], "L", 0.0) or 0.0) for i in orbit]
            orbits.append(
                {
                    "orbit": orbit,
                    "group": group,
                    "fs_min": min(float(v) for v in fs_vals),
                    "fs_max": max(float(v) for v in fs_vals),
                    "n_min": min(ns),
                    "n_max": max(ns),
                    "length_total": sum(lengths),
                    "case": worst_by_mid.get(mid, {}).get("case"),
                }
            )

        critical = [o for o in orbits if o["fs_min"] < fs_threshold and o["n_max"] < max_for_group(str(o["group"]))]
        donors = [o for o in orbits if o["fs_min"] > donor_threshold and o["n_min"] > min_for_group(str(o["group"]))]
        critical.sort(key=lambda o: (o["fs_min"], -o["length_total"]))
        donors.sort(key=lambda o: (-o["fs_min"], o["length_total"]))

        if not critical or not donors:
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        best_cfg = cur_cfg
        best_summary = cur_summary
        best_break = safe_float(cur_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        best_fs = safe_float(cur_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        trace_rows: List[Dict[str, Any]] = []
        used_pairs: set[Tuple[Tuple[int, ...], Tuple[int, ...]]] = set()

        for _ in range(max_swaps):
            accepted = False
            for crit in critical:
                c_orbit = tuple(crit["orbit"])
                for donor in donors:
                    d_orbit = tuple(donor["orbit"])
                    if crit["group"] != donor["group"] or c_orbit == d_orbit:
                        continue
                    if (c_orbit, d_orbit) in used_pairs:
                        continue
                    used_pairs.add((c_orbit, d_orbit))
                    add_mass = sum(float(getattr(member_by_id[i], "L", 0.0) or 0.0) / stick_len_mm * stick_mass_g for i in c_orbit)
                    rem_mass = sum(float(getattr(member_by_id[i], "L", 0.0) or 0.0) / stick_len_mm * stick_mass_g for i in d_orbit)
                    net_mass = add_mass - rem_mass
                    if current_mass + net_mass > target_proxy_mass + max_net_mass + 1.0e-9:
                        continue
                    trial = copy.deepcopy(best_cfg)
                    by_id = trial.setdefault("member_sticks_by_id", {})
                    valid = True
                    for i in c_orbit:
                        old = max(1, int(getattr(member_by_id[i], "n_sticks", 1)))
                        if old >= max_for_group(str(crit["group"])):
                            valid = False
                            break
                        by_id[str(int(i))] = old + 1
                    for i in d_orbit:
                        old = max(1, int(getattr(member_by_id[i], "n_sticks", 1)))
                        if old <= min_for_group(str(donor["group"])):
                            valid = False
                            break
                        by_id[str(int(i))] = old - 1
                    if not valid:
                        continue
                    trial = self.planner.config.normalize(trial)
                    summary = self._multi_case_summary(trial, load_cases, stage_name=stage_name, tension_only=tension_only)
                    if not self._summary_valid_flag(summary):
                        continue
                    nb = safe_float(summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
                    nf = safe_float(summary.get("min_fs_design_proxy"), 0.0) or 0.0
                    nm = safe_float(summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
                    min_break_ret = float(member_sizing.get("rebalance_min_break_retention", 0.995))
                    min_fs_ret = float(member_sizing.get("rebalance_min_fs_retention", 0.995))
                    if nm <= target_proxy_mass + max_net_mass + 1.0e-9 and nb >= best_break * min_break_ret and nf >= best_fs * min_fs_ret and (nf > best_fs + 1.0e-6 or nb > best_break + 1.0e-6):
                        best_cfg = trial
                        best_summary = summary
                        best_break = nb
                        best_fs = nf
                        trace_rows.append(
                            {
                                "critical_orbit": ";".join(str(i) for i in c_orbit),
                                "donor_orbit": ";".join(str(i) for i in d_orbit),
                                "group": crit["group"],
                                "critical_fs_before": crit["fs_min"],
                                "donor_fs_before": donor["fs_min"],
                                "net_mass_g_est": net_mass,
                                "new_break_proxy_kgf": nb,
                                "new_min_fs_design_proxy": nf,
                                "new_mass_proxy_g": nm,
                                "reason": "symmetry_preserving_primary_rebalance",
                            }
                        )
                        accepted = True
                        break
                if accepted:
                    break
            if not accepted:
                break

        return {"best_cfg": best_cfg, "summary": best_summary, "trace_rows": trace_rows}


    def _section_efficiency_mutation(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
    ) -> Dict[str, Any]:
        """Testa melhorias de seção sem adicionar palitos.

        O objetivo é atacar flambagem/interação axial-flexão aumentando inércia
        geométrica por layout/orientação, antes de simplesmente adicionar massa.
        Mantém o modelo geral: se o usuário mudar dimensões do palito, a mutação
        ainda só ajusta espaçamentos relativos/absolutos configuráveis.
        """
        settings = cfg.get("member_sizing", {}) or {}
        if not bool(settings.get("enable_section_efficiency_mutation", True)):
            summary = self._multi_case_summary(cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
            return {"best_cfg": cfg, "summary": summary, "trace_rows": []}

        cur_cfg = self.planner.config.normalize(cfg)
        cur_summary = self._multi_case_summary(cur_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
        if not self._summary_valid_flag(cur_summary):
            return {"best_cfg": cur_cfg, "summary": cur_summary, "trace_rows": []}

        mass_limit = float(effective_mass_limit_g(cur_cfg))
        target_proxy_mass = mass_limit * float(settings.get("section_efficiency_max_proxy_mass_ratio", 0.985))
        min_break_gain = float(settings.get("section_efficiency_min_break_gain", 1.003))
        min_fs_gain = float(settings.get("section_efficiency_min_fs_gain", 1.003))
        min_robustness_gain = float(settings.get("section_efficiency_min_robustness_gain", 1.01))

        best_cfg = cur_cfg
        best_summary = cur_summary
        best_break = safe_float(cur_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        best_fs = safe_float(cur_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        best_mass = safe_float(cur_summary.get("dead_weight_proxy_g"), 0.0) or 0.0
        trace_rows: List[Dict[str, Any]] = []

        groups = {str(g) for g in (settings.get("section_efficiency_groups") or ["top_chord", "vertical", "diagonal"])}
        candidates: List[Tuple[str, str, float]] = []

        layout_base = (cur_cfg.get("section_layout_by_group", {}) or {})

        def _has_spatial_bracing(cfg_i: Dict[str, Any]) -> bool:
            b = cfg_i.get("bridge", {}) or {}
            return bool(
                b.get("include_top_x_bracing", True)
                and b.get("include_bottom_x_bracing", True)
                and b.get("include_cross_frame_bracing", True)
            )

        def _candidate_with_spacing(group: str, spacing: float) -> Dict[str, Any]:
            cand = copy.deepcopy(best_cfg)
            layout = cand.setdefault("section_layout_by_group", {})
            gcfg = dict(layout.get(group, {}) or {})
            gcfg["layout"] = "box"
            if group in {"top_chord", "bottom_chord", "vertical", "diagonal"}:
                gcfg["stick_orientation"] = "edge"
            gcfg["spacing_y_mm"] = max(float(gcfg.get("spacing_y_mm", 0.0) or 0.0), float(spacing))
            gcfg["spacing_z_mm"] = max(float(gcfg.get("spacing_z_mm", 0.0) or 0.0), float(spacing))
            layout[group] = gcfg
            return cand

        def _candidate_with_K(group: str, k_value: float) -> Dict[str, Any]:
            cand = copy.deepcopy(best_cfg)
            k_by_group = cand.setdefault("effective_length_factor_by_group", {})
            entry = dict(k_by_group.get(group, {}) or {})
            current_ky = safe_float(entry.get("Ky"), 1.0) or 1.0
            current_kz = safe_float(entry.get("Kz"), 1.0) or 1.0
            entry["Ky"] = min(float(current_ky), float(k_value))
            entry["Kz"] = min(float(current_kz), float(k_value))
            k_by_group[group] = entry
            return cand

        if "top_chord" in groups:
            for sp in settings.get("section_efficiency_top_chord_spacing_candidates_mm", [16.0, 18.0, 20.0, 22.0]):
                cur = layout_base.get("top_chord", {}) or {}
                if float(sp) > max(float(cur.get("spacing_y_mm", 0.0) or 0.0), float(cur.get("spacing_z_mm", 0.0) or 0.0)) + 1.0e-9:
                    candidates.append(("top_chord", "box_spacing", float(sp)))

        if "vertical" in groups:
            for sp in settings.get("section_efficiency_vertical_spacing_candidates_mm", [11.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]):
                cur = layout_base.get("vertical", {}) or {}
                if float(sp) > max(float(cur.get("spacing_y_mm", 0.0) or 0.0), float(cur.get("spacing_z_mm", 0.0) or 0.0)) + 1.0e-9:
                    candidates.append(("vertical", "box_spacing", float(sp)))

        if "diagonal" in groups:
            for sp in settings.get("section_efficiency_diagonal_spacing_candidates_mm", [10.0, 12.0, 14.0, 16.0, 18.0]):
                cur = layout_base.get("diagonal", {}) or {}
                cur_layout = str(cur.get("layout", "")).lower()
                cur_spacing = max(float(cur.get("spacing_y_mm", 0.0) or 0.0), float(cur.get("spacing_z_mm", 0.0) or 0.0))
                # A diagonal em double_stack tem eixo fraco muito baixo.  Testar uma
                # seção-caixa com o mesmo número de palitos melhora robustez a
                # carregamento deslocado sem adicionar massa no proxy.
                if cur_layout != "box" or float(sp) > cur_spacing + 1.0e-9:
                    candidates.append(("diagonal", "box_spacing", float(sp)))

        allow_k_mutation = (not bool(settings.get("section_efficiency_require_bracing_for_K", True))) or _has_spatial_bracing(best_cfg)
        if allow_k_mutation:
            current_k = (best_cfg.get("effective_length_factor_by_group", {}) or {})
            if "top_chord" in groups:
                cur_top = current_k.get("top_chord", {}) or {}
                cur_val = min(float(safe_float(cur_top.get("Ky"), 1.0) or 1.0), float(safe_float(cur_top.get("Kz"), 1.0) or 1.0))
                for kv in settings.get("section_efficiency_top_chord_K_candidates", [0.62, 0.58]):
                    if float(kv) < cur_val - 1.0e-9:
                        candidates.append(("top_chord", "effective_length_K", float(kv)))
            if "vertical" in groups:
                cur_v = current_k.get("vertical", {}) or {}
                cur_val = min(float(safe_float(cur_v.get("Ky"), 1.0) or 1.0), float(safe_float(cur_v.get("Kz"), 1.0) or 1.0))
                for kv in settings.get("section_efficiency_vertical_K_candidates", [0.76, 0.72]):
                    if float(kv) < cur_val - 1.0e-9:
                        candidates.append(("vertical", "effective_length_K", float(kv)))
            if "diagonal" in groups:
                cur_d = current_k.get("diagonal", {}) or {}
                cur_val = min(float(safe_float(cur_d.get("Ky"), 1.0) or 1.0), float(safe_float(cur_d.get("Kz"), 1.0) or 1.0))
                for kv in settings.get("section_efficiency_diagonal_K_candidates", [0.82]):
                    if float(kv) < cur_val - 1.0e-9:
                        candidates.append(("diagonal", "effective_length_K", float(kv)))

        for group, mutation, value in candidates:
            if mutation == "box_spacing":
                cand_cfg = _candidate_with_spacing(group, float(value))
            elif mutation == "effective_length_K":
                cand_cfg = _candidate_with_K(group, float(value))
            else:
                continue
            cand_cfg = self.planner.config.normalize(cand_cfg)
            summary = self._multi_case_summary(cand_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
            new_break = safe_float(summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
            new_fs = safe_float(summary.get("min_fs_design_proxy"), 0.0) or 0.0
            new_mass = safe_float(summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
            old_robustness = safe_float(best_summary.get("robustness_min_breaking_load_proxy_kgf"), best_break) or best_break
            new_robustness = safe_float(summary.get("robustness_min_breaking_load_proxy_kgf"), new_break) or new_break
            accepted = False
            reason = "not_improved"

            if not self._summary_valid_flag(summary):
                reason = "invalid_summary"
            elif new_mass > target_proxy_mass + 1.0e-9 and best_mass <= target_proxy_mass + 1.0e-9:
                reason = "above_proxy_mass_target"
            elif new_break >= best_break * min_break_gain or new_fs >= best_fs * min_fs_gain:
                accepted = True
                reason = "section_efficiency_improved"
            elif (
                new_break >= best_break * 0.999
                and new_fs >= best_fs * 0.999
                and new_robustness >= old_robustness * min_robustness_gain
            ):
                accepted = True
                reason = "section_efficiency_robustness_improved"

            if accepted:
                best_cfg = cand_cfg
                best_summary = summary
                best_break = new_break
                best_fs = new_fs
                best_mass = new_mass
                # Atualiza base para permitir mutações cumulativas coerentes.
                layout_base = best_cfg.get("section_layout_by_group", {}) or {}

            trace_rows.append(
                {
                    "group": group,
                    "mutation": mutation,
                    "value_mm": value,
                    "accepted": accepted,
                    "reason": reason,
                    "old_break_proxy_kgf": best_break if accepted else safe_float(cur_summary.get("predicted_breaking_load_proxy_kgf"), 0.0),
                    "new_break_proxy_kgf": new_break,
                    "new_min_fs_design_proxy": new_fs,
                    "new_robustness_break_proxy_kgf": safe_float(summary.get("robustness_min_breaking_load_proxy_kgf"), new_break),
                    "new_mass_proxy_g": new_mass,
                }
            )

        return {"best_cfg": best_cfg, "summary": best_summary, "trace_rows": trace_rows}

    def _topology_cleanup(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
        tension_only: bool = False,
        mass_rescue_only: bool = False,
    ) -> Dict[str, Any]:
        top_cfg = cfg.get("topology_cleanup", {}) or {}
        enabled = bool(top_cfg.get("enabled", True))
        if not enabled:
            base_summary = self._multi_case_summary(cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
            return {
                "best_cfg": cfg,
                "summary": base_summary,
                "trace_rows": [],
                "removed_members": [],
                "mixed_patterns": [],
                "zero_force_diag": [],
                "mass_realloc": [],
            }

        max_iters = max(1, int(top_cfg.get("max_topology_iterations", 80)))
        patience = max(1, int(top_cfg.get("patience", 10)))
        near_zero_N = float(top_cfg.get("near_zero_force_threshold_N", 2.0))
        near_zero_rel = float(top_cfg.get("near_zero_force_relative_threshold", 0.01))
        mass_rescue_target_ratio = float(top_cfg.get("mass_rescue_target_ratio", 0.985))
        mass_rescue_min_break_retention = float(top_cfg.get("mass_rescue_min_break_retention", 0.97))
        mass_rescue_min_fs_retention = float(top_cfg.get("mass_rescue_min_fs_retention", 0.97))

        cur_cfg = self.planner.config.normalize(cfg)
        cur_summary = self._multi_case_summary(cur_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
        if not self._summary_valid_flag(cur_summary):
            return {
                "best_cfg": cur_cfg,
                "summary": cur_summary,
                "trace_rows": [],
                "removed_members": [],
                "mixed_patterns": [],
                "zero_force_diag": [],
                "mass_realloc": [],
            }
        trace_rows: List[Dict[str, Any]] = []
        removed_members: List[Dict[str, Any]] = []
        mixed_patterns: List[Dict[str, Any]] = []
        zero_force_diag: List[Dict[str, Any]] = []
        no_improve = 0

        preserve_groups = {
            str(g)
            for g in (
                top_cfg.get("preserve_member_groups")
                or ["bottom_chord", "top_chord", "vertical", "diagonal", "support_pad"]
            )
        }
        removable_groups = {
            str(g)
            for g in (
                top_cfg.get("removable_member_groups")
                or [
                    "top_bracing",
                    "bottom_bracing",
                    "cross_frame_bracing",
                    "chord_lacing",
                    "top_transverse",
                    "bottom_transverse",
                ]
            )
        }
        preserve_symmetry = bool(top_cfg.get("preserve_symmetry_on_removal", True))

        def _active_case() -> Dict[str, Any]:
            cases = cur_summary.get("cases") or []
            return cases[0] if cases else {}

        def _round_coord(v: Any) -> float:
            return round(float(v), 3)

        def _member_symmetry_helpers() -> Tuple[Dict[int, Any], Dict[int, Any], Dict[Tuple[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]], List[int]]]:
            case = _active_case()
            members = case.get("members") or []
            nodes = case.get("nodes") or []
            node_by_id = {int(getattr(n, "id")): n for n in nodes}
            member_by_id = {int(getattr(m, "id")): m for m in members}

            key_to_ids: Dict[Tuple[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]], List[int]] = {}

            def point_key(n: Any) -> Tuple[float, float, float]:
                return (_round_coord(getattr(n, "x")), _round_coord(getattr(n, "y")), _round_coord(getattr(n, "z")))

            def member_key(m: Any) -> Tuple[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] | None:
                ni = node_by_id.get(int(getattr(m, "i")))
                nj = node_by_id.get(int(getattr(m, "j")))
                if ni is None or nj is None:
                    return None
                pts = tuple(sorted([point_key(ni), point_key(nj)]))
                return (str(getattr(m, "group", "")), pts)

            for m in members:
                k = member_key(m)
                if k is not None:
                    key_to_ids.setdefault(k, []).append(int(getattr(m, "id")))

            return member_by_id, node_by_id, key_to_ids

        def _symmetry_orbit_for_member(
            mid: int,
            member_by_id: Dict[int, Any],
            node_by_id: Dict[int, Any],
            key_to_ids: Dict[Tuple[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]], List[int]],
        ) -> List[int]:
            m = member_by_id.get(int(mid))
            if m is None:
                return []

            ni = node_by_id.get(int(getattr(m, "i")))
            nj = node_by_id.get(int(getattr(m, "j")))
            if ni is None or nj is None:
                return [int(mid)]

            span = float(cur_cfg.get("bridge", {}).get("span_mm", 1200.0))
            group = str(getattr(m, "group", ""))

            def p(n: Any) -> Tuple[float, float, float]:
                return (_round_coord(getattr(n, "x")), _round_coord(getattr(n, "y")), _round_coord(getattr(n, "z")))

            base_pts = [p(ni), p(nj)]

            def tx(pt: Tuple[float, float, float], mirror_x: bool, mirror_y: bool) -> Tuple[float, float, float]:
                x, y, z = pt
                if mirror_x:
                    x = _round_coord(span - x)
                if mirror_y:
                    y = _round_coord(-y)
                return (_round_coord(x), _round_coord(y), _round_coord(z))

            orbit: set[int] = set()
            transforms = [(False, False), (True, False), (False, True), (True, True)] if preserve_symmetry else [(False, False)]

            for mx, my in transforms:
                pts = tuple(sorted([tx(base_pts[0], mx, my), tx(base_pts[1], mx, my)]))
                orbit.update(key_to_ids.get((group, pts), []))

            return sorted(orbit or {int(mid)})

        def _is_removal_orbit_allowed(orbit_ids: List[int], member_by_id: Dict[int, Any]) -> bool:
            if not orbit_ids:
                return False
            for oid in orbit_ids:
                m = member_by_id.get(int(oid))
                if m is None:
                    return False
                group = str(getattr(m, "group", ""))
                if group in preserve_groups:
                    return False
                if group not in removable_groups:
                    return False
            return True

        for it in range(1, max_iters + 1):
            cur_summary = self._multi_case_summary(cur_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
            zero_ids = [int(v) for v in (cur_summary.get("zero_force_member_ids") or [])]
            case_member_results = [r for c in (cur_summary.get("cases") or []) for r in (c.get("member_results") or [])]
            max_abs = max((abs(safe_float(r.get("N_N"), 0.0) or 0.0) for r in case_member_results), default=1.0)
            zero_threshold = max(near_zero_N, near_zero_rel * max_abs)
            for mid in zero_ids:
                zero_force_diag.append(
                    {
                        "iteration": it,
                        "member_id": mid,
                        "threshold_N": zero_threshold,
                    }
                )

            candidates: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []

            # 1) Remoção conservadora de membros quase nulos.
            # Nunca remover membros que definem a forma global da ponte
            # (banzos, montantes, diagonais principais e apoios).
            # A remoção é aplicada por órbita de simetria para evitar ponte torta.
            max_remove = int(top_cfg.get("max_remove_candidates_per_iteration", 4))
            member_by_id, node_by_id, key_to_ids = _member_symmetry_helpers()
            seen_orbits: set[Tuple[int, ...]] = set()
            remove_candidates_added = 0

            for mid in zero_ids:
                if remove_candidates_added >= max_remove:
                    break

                orbit_ids = _symmetry_orbit_for_member(
                    int(mid),
                    member_by_id,
                    node_by_id,
                    key_to_ids,
                )
                orbit_key = tuple(sorted(int(v) for v in orbit_ids))

                if orbit_key in seen_orbits:
                    continue

                seen_orbits.add(orbit_key)

                if not _is_removal_orbit_allowed(list(orbit_key), member_by_id):
                    continue

                c = copy.deepcopy(cur_cfg)
                active_map = c.setdefault("member_active_by_id", {})
                disabled = {
                    int(v)
                    for v in (c.get("disabled_member_ids", []) or [])
                    if str(v).strip()
                }

                for oid in orbit_key:
                    active_map[str(int(oid))] = False
                    disabled.add(int(oid))

                c["disabled_member_ids"] = sorted(disabled)
                c = self.planner.config.normalize(c)
                candidates.append(
                    (
                        "remove_member",
                        c,
                        {
                            "member_id": int(mid),
                            "member_ids": ";".join(str(v) for v in orbit_key),
                            "removed_count": len(orbit_key),
                        },
                    )
                )
                remove_candidates_added += 1

            if not mass_rescue_only:
                # 2) Mutações globais e mistas. Em modo mass rescue,
                # não alterar padrão global; só remover peso local seguro.
                for side_mode, op in [
                    ("Pratt_symmetric", "convert_panel_to_pratt"),
                    ("Howe_inverted", "convert_panel_to_howe"),
                    ("Warren_symmetric", "convert_panel_to_warren"),
                ]:
                    if str(cur_cfg.get("bridge", {}).get("side_truss_type")) == side_mode:
                        continue
                    c = copy.deepcopy(cur_cfg)
                    c.setdefault("bridge", {})["side_truss_type"] = side_mode
                    c = self.planner.config.normalize(c)
                    candidates.append((op, c, {"side_truss_type": side_mode}))

                c_mixed = copy.deepcopy(cur_cfg)
                pattern = self._make_symmetric_span_pattern(c_mixed.setdefault("bridge", {}))
                c_mixed.setdefault("bridge", {})["panel_side_truss_pattern"] = pattern
                c_mixed = self.planner.config.normalize(c_mixed)
                candidates.append(("create_mixed_panel_pattern", c_mixed, {"pattern_len": len(pattern)}))

                # 3) Conversões de contraventamento X para diagonal única e vice-versa.
                c_single = copy.deepcopy(cur_cfg)
                c_single.setdefault("bridge", {})["top_chord_truss_type"] = "Pratt_symmetric"
                c_single.setdefault("bridge", {})["bottom_chord_truss_type"] = "Pratt_symmetric"
                c_single = self.planner.config.normalize(c_single)
                candidates.append(("convert_x_bracing_to_single_tension_diagonal", c_single, {}))

                c_double = copy.deepcopy(cur_cfg)
                c_double.setdefault("bridge", {})["top_chord_truss_type"] = "X"
                c_double.setdefault("bridge", {})["bottom_chord_truss_type"] = "X"
                c_double = self.planner.config.normalize(c_double)
                candidates.append(("convert_single_diagonal_to_x_tension_bracing", c_double, {}))

            if not candidates:
                break

            best_iter = None
            for op_name, cand_cfg, op_meta in candidates:
                try:
                    summary = self._multi_case_summary(cand_cfg, load_cases, stage_name=stage_name, tension_only=tension_only)
                except (TypeError, ValueError, KeyError, RuntimeError) as exc:
                    row = {
                        "iteration": it,
                        "operation": op_name,
                        "objective": -1.0e9,
                        "predicted_breaking_load_proxy_kgf": 0.0,
                        "min_fs_design_proxy": 0.0,
                        "dead_weight_proxy_g": None,
                        "solver_regular": False,
                        "equilibrium_ok": False,
                        "error": repr(exc),
                    }
                    row.update(op_meta)
                    trace_rows.append(row)
                    continue
                row = {
                    "iteration": it,
                    "operation": op_name,
                    "objective": summary.get("objective"),
                    "valid_for_selection": summary.get("valid_for_selection"),
                    "predicted_breaking_load_proxy_kgf": summary.get("predicted_breaking_load_proxy_kgf"),
                    "min_fs_design_proxy": summary.get("min_fs_design_proxy"),
                    "dead_weight_proxy_g": summary.get("dead_weight_proxy_g"),
                    "solver_regular": summary.get("solver_regular"),
                    "equilibrium_ok": summary.get("equilibrium_ok"),
                }
                row.update(op_meta)
                trace_rows.append(row)

                if not self._summary_valid_flag(summary):
                    continue

                if mass_rescue_only:
                    mass_limit_eff = float(effective_mass_limit_g(cur_cfg))
                    mass_val = safe_float(summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
                    break_val = safe_float(summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
                    fs_val = safe_float(summary.get("min_fs_design_proxy"), 0.0) or 0.0
                    objective_val = safe_float(summary.get("objective"), INVALID_OBJECTIVE) or INVALID_OBJECTIVE
                    rank = (
                        -max(0.0, mass_val - mass_rescue_target_ratio * mass_limit_eff),
                        -mass_val,
                        break_val,
                        fs_val,
                        objective_val,
                    )
                    if best_iter is None or rank > best_iter[3]:
                        best_iter = (summary, cand_cfg, row, rank)
                else:
                    if best_iter is None or (safe_float(summary.get("objective"), -1.0e99) or -1.0e99) > (safe_float(best_iter[0].get("objective"), -1.0e99) or -1.0e99):
                        best_iter = (summary, cand_cfg, row, None)

            if best_iter is None:
                break

            best_summary, best_cfg, best_row = best_iter[:3]
            cur_obj = safe_float(cur_summary.get("objective"), -1.0e99) or -1.0e99
            new_obj = safe_float(best_summary.get("objective"), -1.0e99) or -1.0e99

            cur_mass = safe_float(cur_summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
            new_mass = safe_float(best_summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
            cur_break = safe_float(cur_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
            new_break = safe_float(best_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
            cur_fs = safe_float(cur_summary.get("min_fs_design_proxy"), 0.0) or 0.0
            new_fs = safe_float(best_summary.get("min_fs_design_proxy"), 0.0) or 0.0
            mass_limit_eff = float(effective_mass_limit_g(cur_cfg))

            mass_rescue_accept = (
                mass_rescue_only
                and new_mass < cur_mass - 1.0e-6
                and new_break >= cur_break * mass_rescue_min_break_retention
                and new_fs >= cur_fs * mass_rescue_min_fs_retention
            )

            if new_obj > cur_obj + 1.0e-9 or mass_rescue_accept:
                cur_cfg = best_cfg
                cur_summary = best_summary
                no_improve = 0
                if best_row.get("operation") == "remove_member" and best_row.get("member_id") is not None:
                    removed_members.append(
                        {
                            "iteration": it,
                            "member_id": best_row.get("member_id"),
                            "member_ids": best_row.get("member_ids", str(best_row.get("member_id"))),
                            "removed_count": best_row.get("removed_count", 1),
                            "reason": "low_force_all_cases_local_member_symmetric_orbit",
                        }
                    )
                if best_row.get("operation") == "create_mixed_panel_pattern":
                    mixed_patterns.append(
                        {
                            "iteration": it,
                            "panel_side_truss_pattern": json.dumps(
                                cur_cfg.get("bridge", {}).get("panel_side_truss_pattern", {}),
                                ensure_ascii=False,
                            ),
                        }
                    )

                if mass_rescue_only:
                    current_mass_after_accept = safe_float(cur_summary.get("dead_weight_proxy_g"), 1.0e99) or 1.0e99
                    if current_mass_after_accept <= mass_rescue_target_ratio * mass_limit_eff:
                        break
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        # Realoção simples de massa liberada por topologia.
        before_mass = safe_float(
            self._multi_case_summary(cfg, ["center"], stage_name=stage_name, tension_only=tension_only).get("dead_weight_proxy_g"),
            0.0,
        ) or 0.0
        after_mass = safe_float(cur_summary.get("dead_weight_proxy_g"), before_mass) or before_mass
        freed = max(0.0, before_mass - after_mass)
        mass_realloc_rows = [
            {
                "topology_freed_mass_pool_g": freed,
                "before_mass_proxy_g": before_mass,
                "after_mass_proxy_g": after_mass,
            }
        ]

        return {
            "best_cfg": cur_cfg,
            "summary": cur_summary,
            "trace_rows": trace_rows,
            "removed_members": removed_members,
            "mixed_patterns": mixed_patterns,
            "zero_force_diag": zero_force_diag,
            "mass_realloc": mass_realloc_rows,
        }

    @staticmethod
    def _write_geometry_refinement_plot(trace_rows: List[Dict[str, Any]], out_path: Path) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except Exception:
            out_path.write_bytes(b"")
            return

        if not trace_rows:
            out_path.write_bytes(b"")
            return

        xs = [int(r.get("iteration", 0)) for r in trace_rows]
        ys = [safe_float(r.get("objective"), None) for r in trace_rows]
        ys = [y if y is not None else float("nan") for y in ys]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, marker="o", linewidth=1.2)
        plt.xlabel("Iteração")
        plt.ylabel("Objetivo")
        plt.title("Refinamento geométrico (trust-region)")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_path, dpi=140)
        plt.close()

    def run(
        self,
        cfg: Dict[str, Any],
        out_dir: str | Path,
        *,
        progress_callback: Callable[[float, str], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        debug_logger: Any | None = None,
    ) -> Dict[str, Any]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        base = self.planner.config.normalize(cfg)
        pp = base.get("planner_pipeline", {}) or {}
        macro_n = max(8, min(16, int(pp.get("macro_candidates_count", 12))))
        top_k_fast = max(1, int(pp.get("fast_screening_keep_top_k", 3)))
        top_k_multi = max(1, int(pp.get("multi_loadcase_keep_top_k", 2)))
        top_k_s4 = max(1, int(pp.get("geometry_refinement_keep_top_k", 1)))

        ml_cfg = base.get("multi_loadcase_screening", {}) or {}
        load_cases = [str(v) for v in (ml_cfg.get("load_cases") or ["center", "left_offset", "right_offset", "torsion_60_40", "lateral_imperfection", "self_weight"])]
        if bool(ml_cfg.get("include_load_contact_audit_cases", False)):
            for audit_case in ("single_plate_center", "crown_contact"):
                if audit_case not in load_cases:
                    insert_at = load_cases.index("self_weight") if "self_weight" in load_cases else len(load_cases)
                    load_cases.insert(insert_at, audit_case)
        tension_only_s3 = self._use_tension_only_for_stage(base, "S3")
        tension_only_s4 = self._use_tension_only_for_stage(base, "S4")
        tension_only_s5 = self._use_tension_only_for_stage(base, "S5")
        tension_only_s6 = self._use_tension_only_for_stage(base, "S6")
        tension_only_s8 = self._use_tension_only_for_stage(base, "S8")

        stage_times: Dict[str, float] = {}
        logs: List[str] = []
        discarded_rows: List[Dict[str, Any]] = []

        def emit_progress(v: float, t: str) -> None:
            if callable(progress_callback):
                progress_callback(max(0.0, min(1.0, float(v))), str(t))

        def emit_log(msg: str) -> None:
            m = str(msg)
            logs.append(m)
            if callable(log_callback):
                log_callback(m)

        def dbg(event_type: str, **kwargs: Any) -> None:
            if debug_logger is None:
                return
            try:
                debug_logger.event(event_type, **kwargs)
            except Exception:
                return

        emit_progress(0.0, "S0: validação de domínio")
        t0 = time.perf_counter()
        s0 = self._stage0_precheck_domain(base)
        (out / "design_domain.json").write_text(json.dumps(s0, indent=2, ensure_ascii=False), encoding="utf-8")
        if not bool(s0.get("ok")):
            msg = " | ".join(s0.get("violations") or ["Configuração inválida para o edital."])
            raise ValueError(msg)
        stage_times["S0"] = time.perf_counter() - t0

        emit_progress(0.08, "S1: geração de macroprojetos")
        t1 = time.perf_counter()
        macros = self._build_macro_archetypes(base, macro_n)
        s1_rows: List[Dict[str, Any]] = []
        for idx, macro in enumerate(macros, 1):
            cfg_i = self._macro_to_config(base, macro)
            row = {
                "stage": "S1",
                "candidate_id": f"S1-{idx:04d}",
                "macro_name": macro.get("macro_name"),
                "global_pattern": macro.get("global_pattern"),
                "side_truss_type": cfg_i.get("bridge", {}).get("side_truss_type"),
                "internal_truss_type": cfg_i.get("bridge", {}).get("internal_truss_type"),
                "top_chord_truss_type": cfg_i.get("bridge", {}).get("top_chord_truss_type"),
                "bottom_chord_truss_type": cfg_i.get("bridge", {}).get("bottom_chord_truss_type"),
                "top_profile": cfg_i.get("bridge", {}).get("top_profile"),
                "span_mm": cfg_i.get("bridge", {}).get("span_mm"),
                "width_mm": cfg_i.get("bridge", {}).get("width_mm"),
                "center_height_mm": cfg_i.get("bridge", {}).get("center_height_mm"),
                "panel_mm": cfg_i.get("bridge", {}).get("panel_mm"),
                "config": cfg_i,
            }
            s1_rows.append(row)
            dbg("s1_macro_generated", stage="s1", candidate_id=row["candidate_id"], metrics={"macro_name": row["macro_name"]})
        stage_times["S1"] = time.perf_counter() - t1

        emit_progress(0.18, "S2: triagem rápida")
        t2 = time.perf_counter()
        s2_rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(s1_rows, 1):
            cfg_i = row["config"]
            center_case = self._evaluate_case_cached(cfg_i, "center", stage_name="S2", tension_only=False)
            quick_score = self._quick_score_from_case(cfg_i, center_case)
            s2_rows.append(
                {
                    **{k: v for k, v in row.items() if k != "config"},
                    "stage": "S2",
                    "candidate_id": f"S2-{idx:04d}",
                    "solver_regular": center_case.get("solver_regular"),
                    "equilibrium_ok": center_case.get("equilibrium_ok"),
                    "compliance_proxy": center_case.get("max_displacement_proxy_mm"),
                    "max_displacement_proxy": center_case.get("max_displacement_proxy_mm"),
                    "max_compression_proxy": center_case.get("max_compression_proxy_N"),
                    "max_tension_proxy": center_case.get("max_tension_proxy_N"),
                    "buckling_risk_proxy": center_case.get("buckling_risk_proxy"),
                    "mass_proxy": center_case.get("mass_proxy_g"),
                    "load_path_score": center_case.get("load_path_score"),
                    "support_reaction_balance": center_case.get("support_reaction_balance"),
                    "topology_stability_proxy": center_case.get("topology_stability_proxy"),
                    "quick_score": quick_score,
                    "valid_for_selection": (
                        self._is_selectable_case(center_case)
                        and (
                            (
                                (safe_float(center_case.get("mass_proxy_g"), 0.0) or 0.0)
                                / max(1.0, float(effective_mass_limit_g(cfg_i)))
                            )
                            <= float((cfg_i.get("planner_pipeline", {}) or {}).get("s2_hard_mass_reject_factor", 1.40))
                        )
                        and not (
                            (
                                (safe_float(center_case.get("mass_proxy_g"), 0.0) or 0.0)
                                / max(1.0, float(effective_mass_limit_g(cfg_i)))
                            )
                            > float((cfg_i.get("planner_pipeline", {}) or {}).get("s2_soft_mass_factor", 1.00))
                            and (
                                (
                                    safe_float(center_case.get("predicted_breaking_load_proxy_kgf"), 0.0)
                                    or 0.0
                                )
                                / max(
                                    1.0,
                                    float(
                                        (cfg_i.get("analysis", {}) or {}).get(
                                            "acceptance_min_design_breaking_load_kgf",
                                            80.0,
                                        )
                                    ),
                                )
                            )
                            < float((cfg_i.get("planner_pipeline", {}) or {}).get("s2_overweight_min_break_ratio", 0.45))
                        )
                    ),
                    "mass_limit_g": float(effective_mass_limit_g(cfg_i)),
                    "mass_ratio": (
                        (safe_float(center_case.get("mass_proxy_g"), 0.0) or 0.0)
                        / max(1.0, float(effective_mass_limit_g(cfg_i)))
                    ),
                    "reject_reason": (
                        "mass_proxy_above_s2_limit"
                        if (safe_float(center_case.get("mass_proxy_g"), 0.0) or 0.0)
                        > float(effective_mass_limit_g(cfg_i))
                        * float((cfg_i.get("planner_pipeline", {}) or {}).get("s2_hard_mass_reject_factor", 1.40))
                        else ""
                    ),
                    "geometry_hash": self._signature_hashes(cfg_i, "center")["geometry_hash"],
                    "topology_hash": self._signature_hashes(cfg_i, "center")["topology_hash"],
                    "sizing_hash": self._signature_hashes(cfg_i, "center")["sizing_hash"],
                    "load_case_hash": self._signature_hashes(cfg_i, "center")["load_case_hash"],
                    "config": cfg_i,
                }
            )
        s2_rows = sorted(s2_rows, key=lambda r: safe_float(r.get("quick_score"), -1.0e99) or -1.0e99, reverse=True)
        keep_s2 = self._pick_with_diversity(
            s2_rows,
            top_k_fast,
            key_field="global_pattern",
        )

        GeometryService.write_csv(out / "fast_screening_results.csv", [{k: v for k, v in r.items() if k != "config"} for r in s2_rows])
        (out / "selected_top3.json").write_text(
            json.dumps(
                [{k: v for k, v in r.items() if k != "config"} for r in keep_s2],
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        gallery_lines = ["# Macro Candidate Gallery", "", "| id | família | lateral | topo | score rápido |", "| --- | --- | --- | --- | ---: |"]
        for r in s2_rows:
            gallery_lines.append(
                f"| {r.get('candidate_id')} | {r.get('global_pattern')} | {r.get('side_truss_type')} | {r.get('top_profile')} | {(safe_float(r.get('quick_score'), 0.0) or 0.0):.3f} |"
            )
        (out / "macro_candidate_gallery.md").write_text("\n".join(gallery_lines) + "\n", encoding="utf-8")
        stage_times["S2"] = time.perf_counter() - t2
        if not keep_s2:
            raise RuntimeError(
                "S2 não encontrou macroprojeto estruturalmente selecionável; "
                "revise estabilidade, apoios, contraventamento e grupos tension-only."
            )

        emit_progress(0.33, "S3: triagem multi-loadcase")
        t3 = time.perf_counter()
        s3_rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(keep_s2, 1):
            cfg_i = row["config"]
            summary = self._multi_case_summary(cfg_i, load_cases, stage_name="S3", tension_only=tension_only_s3)
            s3_rows.append(
                {
                    **{k: v for k, v in row.items() if k != "config"},
                    "stage": "S3",
                    "candidate_id": f"S3-{idx:04d}",
                    "objective": summary.get("objective"),
                    "valid_for_selection": summary.get("valid_for_selection"),
                    "solver_regular": summary.get("solver_regular"),
                    "equilibrium_ok": summary.get("equilibrium_ok"),
                    "min_fs_preliminary": summary.get("min_fs_preliminary"),
                    "min_fs_design_proxy": summary.get("min_fs_design_proxy"),
                    "predicted_breaking_load_proxy_kgf": summary.get("predicted_breaking_load_proxy_kgf"),
                    "multi_case_zero_force_members": summary.get("multi_case_zero_force_members"),
                    "dead_weight_proxy_g": summary.get("dead_weight_proxy_g"),
                    "nodal_stability_proxy": summary.get("nodal_stability_proxy"),
                    "lateral_stability_proxy": summary.get("lateral_stability_proxy"),
                    "support_reaction_balance": summary.get("support_reaction_balance"),
                    "load_path_score": summary.get("load_path_score"),
                    "topology_stability_proxy": summary.get("topology_stability_proxy"),
                    "geometry_hash": summary.get("geometry_hash"),
                    "topology_hash": summary.get("topology_hash"),
                    "sizing_hash": summary.get("sizing_hash"),
                    "load_case_hash": summary.get("load_case_hash"),
                    "config": cfg_i,
                    "case_metrics": summary.get("cases"),
                    "zero_force_member_ids": summary.get("zero_force_member_ids"),
                }
            )
        s3_rows = sorted(s3_rows, key=lambda r: safe_float(r.get("objective"), -1.0e99) or -1.0e99, reverse=True)
        keep_s3 = self._pick_with_diversity(
            s3_rows,
            top_k_multi,
            key_field="global_pattern",
        )

        GeometryService.write_csv(out / "multi_loadcase_screening.csv", [{k: v for k, v in r.items() if k not in {"config", "case_metrics", "zero_force_member_ids"}} for r in s3_rows])
        s3_case_diag: List[Dict[str, Any]] = []

        for r in s3_rows:
            for c in (r.get("case_metrics") or []):
                s3_case_diag.append(
                    {
                        "candidate_id": r.get("candidate_id"),
                        "global_pattern": r.get("global_pattern"),
                        "case": c.get("case"),
                        "solver_status": c.get("solver_status"),
                        "solver_regular": c.get("solver_regular"),
                        "equilibrium_ok": c.get("equilibrium_ok"),
                        "equilibrium_error_N": c.get("equilibrium_error_N"),
                        "equilibrium_tol_N": c.get("equilibrium_tol_N"),
                        "topology_stability_proxy": c.get("topology_stability_proxy"),
                        "support_reaction_balance": c.get("support_reaction_balance"),
                        "load_path_score": c.get("load_path_score"),
                        "max_displacement_proxy_mm": c.get("max_displacement_proxy_mm"),
                        "min_fs_primary": c.get("min_fs_primary"),
                        "min_fs_design": c.get("min_fs_design"),
                        "predicted_breaking_load_proxy_kgf": c.get("predicted_breaking_load_proxy_kgf"),
                        "inactive_tension_only_members": ";".join(
                            str(v)
                            for v in (c.get("inactive_tension_only_members") or [])
                        ),
                        "inactive_supports_uplift": ";".join(
                            str(v)
                            for v in (c.get("inactive_supports_uplift") or [])
                        ),
                    }
                )

        GeometryService.write_csv(out / "s3_case_diagnostics.csv", s3_case_diag)

        (out / "selected_top2.json").write_text(
            json.dumps(
                [
                    {
                        k: v
                        for k, v in r.items()
                        if k not in {"config", "case_metrics"}
                    }
                    for r in keep_s3
                ],
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        rejected = [r for r in s3_rows if r not in keep_s3]
        rej_md = ["# Rejected Macro Candidates", ""]
        for r in rejected:
            rej_md.append(
                f"- {r.get('candidate_id')}: objective={(safe_float(r.get('objective'), 0.0) or 0.0):.3f}, "
                f"solver_regular={r.get('solver_regular')}, min_fs_preliminary={(safe_float(r.get('min_fs_preliminary'), 0.0) or 0.0):.3f}"
            )
            discarded_rows.append({"stage": "S3", "candidate_id": r.get("candidate_id"), "reason": "below_top_k"})
        (out / "rejected_macro_candidates.md").write_text("\n".join(rej_md) + "\n", encoding="utf-8")
        stage_times["S3"] = time.perf_counter() - t3
        if not keep_s3:
            raise RuntimeError(
                "S3 eliminou todos os macroprojetos; "
                "nenhum candidato permaneceu regular em todos os load cases."
            )

        target_break = float(
            base.get("analysis", {}).get(
                "acceptance_min_design_breaking_load_kgf",
                80.0,
            )
        )

        min_promising_ratio = float(
            pp.get("s3_min_promising_break_ratio", 0.35)
        )

        best_s3_break = max(
            (
                safe_float(r.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
                for r in keep_s3
            ),
            default=0.0,
        )

        s3_low_before_sizing = best_s3_break < min_promising_ratio * target_break
        if s3_low_before_sizing:
            emit_log(
                "[S3:LOW_PROMISE] Candidatos regulares, mas ainda fracos antes do sizing: "
                f"melhor ruptura proxy {best_s3_break:.2f} kgf < "
                f"{min_promising_ratio:.2f} × meta {target_break:.2f} kgf. "
                "Prosseguindo para S5 porque o reforço discreto ainda não foi aplicado."
            )

        best_s3_nominal = max(
            (
                safe_float(r.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
                for r in keep_s3
            ),
            default=0.0,
        )

        target_break = float(
            base.get("analysis", {}).get(
                "acceptance_min_design_breaking_load_kgf",
                80.0,
            )
        )

        s3_abort_ratio = float(pp.get("s3_hard_abort_break_ratio", 0.0))
        s3_abort_abs = float(pp.get("s3_hard_abort_abs_break_kgf", 0.10))
        if best_s3_nominal < max(s3_abort_abs, s3_abort_ratio * target_break):
            emit_log(
                "[S3:VERY_LOW_NOMINAL] Melhor ruptura proxy antes do sizing = "
                f"{best_s3_nominal:.2f} kgf. Valor abaixo do limiar de diagnóstico "
                f"({max(s3_abort_abs, s3_abort_ratio * target_break):.2f} kgf), "
                "mas o funil continuará para S4/S5 porque as seções ainda não foram "
                "redimensionadas. Use os logs S5/S7/S8 para decidir se há erro real."
            )

        emit_progress(0.50, "S4: refinamento geométrico local")
        t4 = time.perf_counter()
        s4_rows: List[Dict[str, Any]] = []

        s4_settings = base.get("local_geometry_refinement", {}) or {}
        s4_refine_cases = [
            str(v)
            for v in (
                s4_settings.get("load_cases")
                or ["center", "left_offset", "right_offset"]
            )
        ]
        s4_trace_rows: List[Dict[str, Any]] = []
        before_after_rows: List[Dict[str, Any]] = []
        s4_input_rows = keep_s3[:top_k_s4]

        skip_s4_low = bool(pp.get("skip_geometry_refinement_when_low_pre_sizing", True)) and bool(s3_low_before_sizing)
        if skip_s4_low:
            low_cap = max(1, int(pp.get("low_pre_sizing_s5_seed_cap", 1)))
            s4_input_rows = keep_s3[:low_cap]

        if skip_s4_low:
            emit_log(
                "[S4:SKIPPED_LOW_PRE_SIZING] Refinamento geométrico local ignorado "
                "porque S3 está muito abaixo da meta antes do dimensionamento. "
                "Isto evita gastar minutos refinando uma geometria ainda subdimensionada; "
                "S5 aplicará o reforço discreto primeiro."
            )
            for idx, row in enumerate(s4_input_rows, 1):
                s4_rows.append(
                    {
                        **{k: v for k, v in row.items() if k != "config"},
                        "stage": "S4_SKIPPED",
                        "candidate_id": f"S4-{idx:04d}",
                        "objective": row.get("objective"),
                        "valid_for_selection": row.get("valid_for_selection"),
                        "predicted_breaking_load_proxy_kgf": row.get("predicted_breaking_load_proxy_kgf"),
                        "min_fs_design_proxy": row.get("min_fs_design_proxy"),
                        "dead_weight_proxy_g": row.get("dead_weight_proxy_g"),
                        "config": row["config"],
                    }
                )
                before_after_rows.append(
                    {
                        "candidate_id": f"S4-{idx:04d}",
                        "before": "S4 skipped: low pre-sizing strength",
                        "after": "unchanged",
                    }
                )
        else:
            for idx, row in enumerate(s4_input_rows, 1):
                refined = self._trust_region_refine(
                    row["config"],
                    s4_refine_cases,
                    stage_name="S4",
                    tension_only=tension_only_s4,
                )

                # Validação completa pós-refinamento. A busca local é barata, mas a seleção
                # continua usando todos os load cases obrigatórios.
                summary = self._multi_case_summary(
                    refined["best_cfg"],
                    load_cases,
                    stage_name="S4_VALIDATE",
                    tension_only=tension_only_s4,
                )
                s4_rows.append(
                    {
                        **{k: v for k, v in row.items() if k != "config"},
                        "stage": "S4",
                        "candidate_id": f"S4-{idx:04d}",
                        "objective": summary.get("objective"),
                        "valid_for_selection": summary.get("valid_for_selection"),
                        "predicted_breaking_load_proxy_kgf": summary.get("predicted_breaking_load_proxy_kgf"),
                        "min_fs_design_proxy": summary.get("min_fs_design_proxy"),
                        "dead_weight_proxy_g": summary.get("dead_weight_proxy_g"),
                        "config": refined["best_cfg"],
                    }
                )
                for tr in refined["trace_rows"]:
                    s4_trace_rows.append(
                        {
                            "candidate_id": f"S4-{idx:04d}",
                            **tr,
                        }
                    )
                before_after_rows.append(
                    {
                        "candidate_id": f"S4-{idx:04d}",
                        "before": json.dumps(refined.get("before", {}), ensure_ascii=False),
                        "after": json.dumps(refined.get("after", {}), ensure_ascii=False),
                    }
                )

        s4_rows = sorted(s4_rows, key=lambda r: safe_float(r.get("objective"), -1.0e99) or -1.0e99, reverse=True)
        keep_s4 = [
            r for r in s4_rows
            if bool(r.get("valid_for_selection", False))
        ][:top_k_s4]
        GeometryService.write_csv(out / "geometry_refinement_trace.csv", s4_trace_rows)
        (out / "geometry_before_after.json").write_text(
            json.dumps(before_after_rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._write_geometry_refinement_plot(s4_trace_rows, out / "plot_geometry_refinement.png")
        stage_times["S4"] = time.perf_counter() - t4
        if not keep_s4:
            raise RuntimeError(
                "S4 não produziu candidato selecionável; "
                "refinamento geométrico não corrigiu estabilidade/FS."
            )

        emit_progress(0.64, "S5: dimensionamento de membros")
        t5 = time.perf_counter()
        s5_rows: List[Dict[str, Any]] = []
        s5_trace: List[Dict[str, Any]] = []
        donors_all: List[Dict[str, Any]] = []
        critical_all: List[Dict[str, Any]] = []
        before_after_all: List[Dict[str, Any]] = []
        for idx, row in enumerate(keep_s4, 1):
            sized = self._member_sizing_pass(
                row["config"],
                load_cases,
                stage_name="S5",
                tension_only=tension_only_s5,
            )
            s = sized["summary"]
            s5_rows.append(
                {
                    **{k: v for k, v in row.items() if k != "config"},
                    "stage": "S5",
                    "candidate_id": f"S5-{idx:04d}",
                    "objective": s.get("objective"),
                    "valid_for_selection": s.get("valid_for_selection"),
                    "predicted_breaking_load_proxy_kgf": s.get("predicted_breaking_load_proxy_kgf"),
                    "min_fs_design_proxy": s.get("min_fs_design_proxy"),
                    "dead_weight_proxy_g": s.get("dead_weight_proxy_g"),
                    "config": sized["best_cfg"],
                }
            )
            for tr in sized["trace_rows"]:
                s5_trace.append({"candidate_id": f"S5-{idx:04d}", **tr})
            donors_all.extend([{"candidate_id": f"S5-{idx:04d}", **d} for d in sized["donors"]])
            critical_all.extend([{"candidate_id": f"S5-{idx:04d}", **c} for c in sized["critical"]])
            before_after_all.extend([{"candidate_id": f"S5-{idx:04d}", **b} for b in sized["before_after"]])

        s5_rows = sorted(s5_rows, key=lambda r: safe_float(r.get("objective"), -1.0e99) or -1.0e99, reverse=True)
        keep_s5_candidates = [
            r for r in s5_rows
            if bool(r.get("valid_for_selection", False))
        ]

        keep_s5 = keep_s5_candidates[
            : max(
                1,
                min(2, len(keep_s5_candidates))
                if bool(pp.get("allow_top2_full_detailing", False))
                else 1,
            )
        ]
        GeometryService.write_csv(out / "member_sizing_trace.csv", s5_trace)
        GeometryService.write_csv(out / "mass_donor_members.csv", donors_all)
        GeometryService.write_csv(out / "critical_reinforcements.csv", critical_all)
        GeometryService.write_csv(out / "sizing_before_after.csv", before_after_all)
        stage_times["S5"] = time.perf_counter() - t5
        if not keep_s5:
            raise RuntimeError(
                "S5 não manteve candidato selecionável após dimensionamento."
            )

        target_break = float(
            base.get("analysis", {}).get(
                "acceptance_min_design_breaking_load_kgf",
                80.0,
            )
        )

        min_post_sizing_ratio = float(
            pp.get("s5_min_promising_break_ratio", 0.35)
        )

        best_s5_break = max(
            (
                safe_float(r.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
                for r in keep_s5
            ),
            default=0.0,
        )

        if best_s5_break < min_post_sizing_ratio * target_break:
            emit_log(
                "[S5:LOW_PROMISE] Mesmo após sizing, o melhor candidato ainda está fraco: "
                f"{best_s5_break:.2f} kgf < "
                f"{min_post_sizing_ratio:.2f} × meta {target_break:.2f} kgf. "
                "O pipeline seguirá para S6/S8 para diagnóstico, mas a solução provavelmente reprovará."
            )

        best_s5_break = safe_float(
            keep_s5[0].get("predicted_breaking_load_proxy_kgf"),
            0.0,
        ) or 0.0

        best_s5_mass = safe_float(
            keep_s5[0].get("dead_weight_proxy_g"),
            0.0,
        ) or 0.0

        target_break = float(
            base.get("analysis", {}).get(
                "acceptance_min_design_breaking_load_kgf",
                80.0,
            )
        )

        mass_limit = float(effective_mass_limit_g(keep_s5[0]["config"]))

        # S6 não deve gastar solve limpando topologia quando a ponte ainda está
        # muito abaixo da meta e já está dentro da massa. Nessa situação, o
        # gargalo é reforço/capacidade, não remoção de membros.
        s6_skip_break_ratio = float(
            (base.get("topology_cleanup", {}) or {}).get(
                "skip_if_break_below_ratio",
                0.65,
            )
        )

        mass_rescue_target_ratio = float(
            (base.get("topology_cleanup", {}) or {}).get(
                "mass_rescue_target_ratio",
                0.955,
            )
        )
        mass_rescue_target_g = mass_rescue_target_ratio * mass_limit

        skip_topology_low_strength_regardless_mass = bool(
            (base.get("topology_cleanup", {}) or {}).get(
                "skip_if_very_low_strength_before_mass_rescue",
                True,
            )
        )
        skip_topology_when_weak_and_within_mass = (
            best_s5_break < s6_skip_break_ratio * target_break
            and (best_s5_mass <= mass_rescue_target_g or skip_topology_low_strength_regardless_mass)
        )

        # Mass rescue não deve ocorrer apenas quando passa de 1000 g.
        # Se a meta prática é ficar abaixo de ~980 g, S6 precisa liberar massa
        # local enquanto preservar quase toda a capacidade. Essa folga é então
        # reinvestida em membros primários críticos antes do S7.
        mass_rescue_only = (
            best_s5_mass > mass_rescue_target_g
            and best_s5_break < target_break
        )

        emit_progress(0.76, "S6: mutação topológica final")
        t6 = time.perf_counter()
        s6_rows: List[Dict[str, Any]] = []
        s6_trace: List[Dict[str, Any]] = []
        removed_members: List[Dict[str, Any]] = []
        mixed_patterns: List[Dict[str, Any]] = []
        zero_force_diag: List[Dict[str, Any]] = []
        mass_realloc_rows: List[Dict[str, Any]] = []

        if skip_topology_when_weak_and_within_mass:
            emit_log(
                "[S6:SKIPPED] Topologia fina ignorada porque o candidato ainda está "
                "muito abaixo da meta; S6 foi ignorado para evitar timeout em resgate de massa antes de haver capacidade estrutural. "
                f"ruptura={best_s5_break:.2f} kgf, "
                f"massa={best_s5_mass:.1f} g, "
                f"limite={mass_limit:.1f} g. "
                "A prioridade correta é reforço/distribuição de massa, não remoção de membros."
            )

            keep_s6 = []
            stage_times["S6"] = time.perf_counter() - t6
        else:
            if mass_rescue_only:
                emit_log(
                    "[S6:MASS_RESCUE] Candidato ainda abaixo da meta e acima da massa prática "
                    f"({best_s5_mass:.1f} g > alvo {mass_rescue_target_g:.1f} g). "
                    "S6 rodará como resgate de massa local para posterior reinvestimento crítico."
                )
            for idx, row in enumerate(keep_s5[:1], 1):
                try:
                    topo = self._topology_cleanup(
                        row["config"],
                        load_cases,
                        stage_name="S6",
                        tension_only=tension_only_s6,
                        mass_rescue_only=mass_rescue_only,
                    )
                except TypeError as exc:
                    # Compatibility with monkeypatched/older cleanup callables in tests
                    # and external scripts that do not yet accept mass_rescue_only.
                    if "mass_rescue_only" not in str(exc):
                        raise
                    topo = self._topology_cleanup(
                        row["config"],
                        load_cases,
                        stage_name="S6",
                        tension_only=tension_only_s6,
                    )
                s = topo["summary"]
                s6_rows.append(
                    {
                        **{k: v for k, v in row.items() if k != "config"},
                        "stage": "S6",
                        "candidate_id": f"S6-{idx:04d}",
                        "objective": s.get("objective"),
                        "valid_for_selection": s.get("valid_for_selection"),
                        "predicted_breaking_load_proxy_kgf": s.get("predicted_breaking_load_proxy_kgf"),
                        "min_fs_design_proxy": s.get("min_fs_design_proxy"),
                        "dead_weight_proxy_g": s.get("dead_weight_proxy_g"),
                        "config": topo["best_cfg"],
                    }
                )
                s6_trace.extend([{"candidate_id": f"S6-{idx:04d}", **r} for r in topo["trace_rows"]])
                removed_members.extend([{"candidate_id": f"S6-{idx:04d}", **r} for r in topo["removed_members"]])
                mixed_patterns.extend([{"candidate_id": f"S6-{idx:04d}", **r} for r in topo["mixed_patterns"]])
                zero_force_diag.extend([{"candidate_id": f"S6-{idx:04d}", **r} for r in topo["zero_force_diag"]])
                mass_realloc_rows.extend([{"candidate_id": f"S6-{idx:04d}", **r} for r in topo["mass_realloc"]])

        s6_rows = sorted(s6_rows, key=lambda r: safe_float(r.get("objective"), -1.0e99) or -1.0e99, reverse=True)
        keep_s6 = [
            r for r in s6_rows
            if bool(r.get("valid_for_selection", False))
        ][:1]
        GeometryService.write_csv(out / "topology_mutation_trace.csv", s6_trace)
        GeometryService.write_csv(out / "removed_members.csv", removed_members)
        GeometryService.write_csv(out / "mixed_panel_patterns.csv", mixed_patterns)
        GeometryService.write_csv(out / "zero_force_diagnostics.csv", zero_force_diag)
        GeometryService.write_csv(out / "mass_reallocation_after_topology.csv", mass_realloc_rows)
        stage_times["S6"] = time.perf_counter() - t6

        # S6 pode liberar massa local. Antes o funil seguia direto para S7;
        # agora a folga é reinvestida em gargalos primários simétricos, sem
        # mudar topologia e sem remover membros.
        reinvest_trace_rows: List[Dict[str, Any]] = []
        if bool((base.get("member_sizing", {}) or {}).get("enable_post_topology_reinvestment", True)):
            reinvest_input_cfg = (keep_s6[0]["config"] if keep_s6 else keep_s5[0]["config"])
            reinvested = self._reinvest_mass_into_critical_members(
                reinvest_input_cfg,
                load_cases,
                stage_name="S6_REINVEST",
                tension_only=tension_only_s6,
            )
            reinvest_trace_rows = [
                {"candidate_id": "S6R-0001", **r}
                for r in (reinvested.get("trace_rows") or [])
            ]
            if reinvest_trace_rows:
                s = reinvested["summary"]
                s6_rows = [
                    {
                        "stage": "S6_REINVEST",
                        "candidate_id": "S6R-0001",
                        "objective": s.get("objective"),
                        "valid_for_selection": s.get("valid_for_selection"),
                        "predicted_breaking_load_proxy_kgf": s.get("predicted_breaking_load_proxy_kgf"),
                        "min_fs_design_proxy": s.get("min_fs_design_proxy"),
                        "dead_weight_proxy_g": s.get("dead_weight_proxy_g"),
                        "config": reinvested["best_cfg"],
                    }
                ]
                keep_s6 = s6_rows[:1]
        GeometryService.write_csv(out / "post_topology_reinvestment.csv", reinvest_trace_rows)

        # Rebalanceamento quase neutro em massa: move 1 palito de órbitas primárias
        # folgadas para órbitas simétricas críticas. Isso corrige casos em que o
        # reinvestimento adiciona massa em alguns painéis, mas deixa outro par
        # simétrico de banzo/montante como gargalo.
        rebalance_trace_rows: List[Dict[str, Any]] = []
        if bool((base.get("member_sizing", {}) or {}).get("enable_post_reinvest_rebalance", True)):
            rebalance_input_cfg = (keep_s6[0]["config"] if keep_s6 else keep_s5[0]["config"])
            rebalanced = self._rebalance_primary_sticks_by_symmetry(
                rebalance_input_cfg,
                load_cases,
                stage_name="S6_REBALANCE",
                tension_only=tension_only_s6,
            )
            rebalance_trace_rows = [
                {"candidate_id": "S6B-0001", **r}
                for r in (rebalanced.get("trace_rows") or [])
            ]
            if rebalance_trace_rows:
                s = rebalanced["summary"]
                s6_rows = [
                    {
                        "stage": "S6_REBALANCE",
                        "candidate_id": "S6B-0001",
                        "objective": s.get("objective"),
                        "valid_for_selection": s.get("valid_for_selection"),
                        "predicted_breaking_load_proxy_kgf": s.get("predicted_breaking_load_proxy_kgf"),
                        "min_fs_design_proxy": s.get("min_fs_design_proxy"),
                        "dead_weight_proxy_g": s.get("dead_weight_proxy_g"),
                        "config": rebalanced["best_cfg"],
                    }
                ]
                keep_s6 = s6_rows[:1]
        GeometryService.write_csv(out / "post_reinvest_rebalance.csv", rebalance_trace_rows)

        # Mutação sem massa: melhora inércia efetiva de seções comprimidas
        # alterando orientação/espaçamento de banzos e montantes dentro de limites
        # construtivos. Isto afeta flambagem sem adicionar palitos.
        section_eff_trace_rows: List[Dict[str, Any]] = []
        if bool((base.get("member_sizing", {}) or {}).get("enable_section_efficiency_mutation", True)):
            section_input_cfg = (keep_s6[0]["config"] if keep_s6 else keep_s5[0]["config"])
            section_eff = self._section_efficiency_mutation(
                section_input_cfg,
                load_cases,
                stage_name="S6_SECTION_EFF",
                tension_only=tension_only_s6,
            )
            section_eff_trace_rows = [
                {"candidate_id": "S6E-0001", **r}
                for r in (section_eff.get("trace_rows") or [])
            ]
            if any(bool(r.get("accepted")) for r in section_eff_trace_rows):
                s = section_eff["summary"]
                s6_rows = [
                    {
                        "stage": "S6_SECTION_EFF",
                        "candidate_id": "S6E-0001",
                        "objective": s.get("objective"),
                        "valid_for_selection": s.get("valid_for_selection"),
                        "predicted_breaking_load_proxy_kgf": s.get("predicted_breaking_load_proxy_kgf"),
                        "min_fs_design_proxy": s.get("min_fs_design_proxy"),
                        "dead_weight_proxy_g": s.get("dead_weight_proxy_g"),
                        "config": section_eff["best_cfg"],
                    }
                ]
                keep_s6 = s6_rows[:1]
        GeometryService.write_csv(out / "section_efficiency_mutation.csv", section_eff_trace_rows)

        # Mutação de eficiência dos planos superior/inferior: tenta reduzir bracing
        # não governante mantendo ruptura e FS. A massa economizada pode ser
        # reinvestida no empurrão de resistência e nas sapatas de apoio.
        plane_bracing_eff_rows: List[Dict[str, Any]] = []
        if bool((base.get("member_sizing", {}) or {}).get("enable_plane_bracing_efficiency_mutation", True)):
            plane_input_cfg = (keep_s6[0]["config"] if keep_s6 else keep_s5[0]["config"])
            plane_eff = self._plane_bracing_efficiency_mutation(
                plane_input_cfg,
                load_cases,
                stage_name="S6_PLANE_BRACING_EFF",
                tension_only=tension_only_s6,
            )
            plane_bracing_eff_rows = [
                {"candidate_id": "S6PB-0001", **r}
                for r in (plane_eff.get("trace_rows") or [])
            ]
            if any(bool(r.get("accepted")) for r in plane_bracing_eff_rows):
                s = plane_eff["summary"]
                s6_rows = [
                    {
                        "stage": "S6_PLANE_BRACING_EFF",
                        "candidate_id": "S6PB-0001",
                        "objective": s.get("objective"),
                        "valid_for_selection": s.get("valid_for_selection"),
                        "predicted_breaking_load_proxy_kgf": s.get("predicted_breaking_load_proxy_kgf"),
                        "min_fs_design_proxy": s.get("min_fs_design_proxy"),
                        "dead_weight_proxy_g": s.get("dead_weight_proxy_g"),
                        "config": plane_eff["best_cfg"],
                    }
                ]
                keep_s6 = s6_rows[:1]
        GeometryService.write_csv(out / "plane_bracing_efficiency_mutation.csv", plane_bracing_eff_rows)

        # Eficiência do platô: depois de escolher o treliçamento de plano, testar
        # larguras menores ainda dentro do edital.  Com carga por platô, larguras
        # menores reduzem travessas e o braço torsor do load case 60/40; depois a
        # seção é reavaliada porque os membros governantes podem mudar.
        plateau_width_rows: List[Dict[str, Any]] = []
        if bool((base.get("member_sizing", {}) or {}).get("enable_plateau_width_efficiency_mutation", True)):
            width_input_cfg = (keep_s6[0]["config"] if keep_s6 else keep_s5[0]["config"])
            width_eff = self._plateau_width_efficiency_mutation(
                width_input_cfg,
                load_cases,
                stage_name="S6_PLATEAU_WIDTH_EFF",
                tension_only=tension_only_s6,
            )
            plateau_width_rows = [
                {"candidate_id": "S6W-0001", **r}
                for r in (width_eff.get("trace_rows") or [])
            ]
            if any(bool(r.get("accepted")) for r in plateau_width_rows):
                s = width_eff["summary"]
                s6_rows = [
                    {
                        "stage": "S6_PLATEAU_WIDTH_EFF",
                        "candidate_id": "S6W-0001",
                        "objective": s.get("objective"),
                        "valid_for_selection": s.get("valid_for_selection"),
                        "predicted_breaking_load_proxy_kgf": s.get("predicted_breaking_load_proxy_kgf"),
                        "min_fs_design_proxy": s.get("min_fs_design_proxy"),
                        "dead_weight_proxy_g": s.get("dead_weight_proxy_g"),
                        "config": width_eff["best_cfg"],
                    }
                ]
                keep_s6 = s6_rows[:1]
        GeometryService.write_csv(out / "plateau_width_efficiency_mutation.csv", plateau_width_rows)

        late_section_eff_rows: List[Dict[str, Any]] = []
        if bool((base.get("member_sizing", {}) or {}).get("enable_late_section_efficiency_after_width", True)):
            late_section_input_cfg = (keep_s6[0]["config"] if keep_s6 else keep_s5[0]["config"])
            late_section_eff = self._section_efficiency_mutation(
                late_section_input_cfg,
                load_cases,
                stage_name="S6_LATE_SECTION_EFF",
                tension_only=tension_only_s6,
            )
            late_section_eff_rows = [
                {"candidate_id": "S6LE-0001", **r}
                for r in (late_section_eff.get("trace_rows") or [])
            ]
            if any(bool(r.get("accepted")) for r in late_section_eff_rows):
                s = late_section_eff["summary"]
                s6_rows = [
                    {
                        "stage": "S6_LATE_SECTION_EFF",
                        "candidate_id": "S6LE-0001",
                        "objective": s.get("objective"),
                        "valid_for_selection": s.get("valid_for_selection"),
                        "predicted_breaking_load_proxy_kgf": s.get("predicted_breaking_load_proxy_kgf"),
                        "min_fs_design_proxy": s.get("min_fs_design_proxy"),
                        "dead_weight_proxy_g": s.get("dead_weight_proxy_g"),
                        "config": late_section_eff["best_cfg"],
                    }
                ]
                keep_s6 = s6_rows[:1]
        GeometryService.write_csv(out / "late_section_efficiency_mutation.csv", late_section_eff_rows)

        late_height_rows: List[Dict[str, Any]] = []
        if bool((base.get("member_sizing", {}) or {}).get("enable_late_height_strength_mutation", True)):
            height_input_cfg = (keep_s6[0]["config"] if keep_s6 else keep_s5[0]["config"])
            height_eff = self._late_height_strength_mutation(
                height_input_cfg,
                load_cases,
                stage_name="S6_LATE_HEIGHT_STRENGTH",
                tension_only=tension_only_s6,
            )
            late_height_rows = [
                {"candidate_id": "S6LH-0001", **r}
                for r in (height_eff.get("trace_rows") or [])
            ]
            if any(bool(r.get("accepted")) for r in late_height_rows):
                s = height_eff["summary"]
                s6_rows = [
                    {
                        "stage": "S6_LATE_HEIGHT_STRENGTH",
                        "candidate_id": "S6LH-0001",
                        "objective": s.get("objective"),
                        "valid_for_selection": s.get("valid_for_selection"),
                        "predicted_breaking_load_proxy_kgf": s.get("predicted_breaking_load_proxy_kgf"),
                        "min_fs_design_proxy": s.get("min_fs_design_proxy"),
                        "dead_weight_proxy_g": s.get("dead_weight_proxy_g"),
                        "config": height_eff["best_cfg"],
                    }
                ]
                keep_s6 = s6_rows[:1]
        GeometryService.write_csv(out / "late_height_strength_mutation.csv", late_height_rows)

        # Sapatas primeiro: no output v33 o reforço de apoio tinha o maior ganho
        # de ruptura por grama, mas era tentado depois do push genérico e acabava
        # bloqueado por uma margem proxy quase nula.  Priorizar o apoio evita gastar
        # massa em banzos antes de resolver o gargalo de reação/localização.
        support_pad_push_rows: List[Dict[str, Any]] = []
        if bool((base.get("member_sizing", {}) or {}).get("enable_support_pad_capacity_push", True)):
            support_input_cfg = (keep_s6[0]["config"] if keep_s6 else keep_s5[0]["config"])
            support_pushed = self._support_pad_capacity_push(
                support_input_cfg,
                load_cases,
                stage_name="S6_SUPPORT_PAD_PUSH",
                tension_only=tension_only_s6,
            )
            support_pad_push_rows = [
                {"candidate_id": "S6SP-0001", **r}
                for r in (support_pushed.get("trace_rows") or [])
            ]
            if any(bool(r.get("accepted")) for r in support_pad_push_rows):
                s = support_pushed["summary"]
                s6_rows = [
                    {
                        "stage": "S6_SUPPORT_PAD_PUSH",
                        "candidate_id": "S6SP-0001",
                        "objective": s.get("objective"),
                        "valid_for_selection": s.get("valid_for_selection"),
                        "predicted_breaking_load_proxy_kgf": s.get("predicted_breaking_load_proxy_kgf"),
                        "min_fs_design_proxy": s.get("min_fs_design_proxy"),
                        "dead_weight_proxy_g": s.get("dead_weight_proxy_g"),
                        "config": support_pushed["best_cfg"],
                    }
                ]
                keep_s6 = s6_rows[:1]
        GeometryService.write_csv(out / "support_pad_capacity_push.csv", support_pad_push_rows)

        # Doador de tração: se o banzo inferior está muito folgado e tracionado,
        # recupera massa antes do push final.  Essa massa é mais útil no banzo
        # superior comprimido e em montantes críticos do que em tensão redundante.
        bottom_chord_donor_rows: List[Dict[str, Any]] = []
        if bool((base.get("member_sizing", {}) or {}).get("enable_bottom_chord_tension_donor_trim", True)):
            donor_input_cfg = (keep_s6[0]["config"] if keep_s6 else keep_s5[0]["config"])
            donor_trim = self._bottom_chord_tension_donor_trim(
                donor_input_cfg,
                load_cases,
                stage_name="S6_BOTTOM_CHORD_DONOR",
                tension_only=tension_only_s6,
            )
            bottom_chord_donor_rows = [
                {"candidate_id": "S6D-0001", **r}
                for r in (donor_trim.get("trace_rows") or [])
            ]
            if any(bool(r.get("accepted")) for r in bottom_chord_donor_rows):
                s = donor_trim["summary"]
                s6_rows = [
                    {
                        "stage": "S6_BOTTOM_CHORD_DONOR",
                        "candidate_id": "S6D-0001",
                        "objective": s.get("objective"),
                        "valid_for_selection": s.get("valid_for_selection"),
                        "predicted_breaking_load_proxy_kgf": s.get("predicted_breaking_load_proxy_kgf"),
                        "min_fs_design_proxy": s.get("min_fs_design_proxy"),
                        "dead_weight_proxy_g": s.get("dead_weight_proxy_g"),
                        "config": donor_trim["best_cfg"],
                    }
                ]
                keep_s6 = s6_rows[:1]
        GeometryService.write_csv(out / "bottom_chord_tension_donor_trim.csv", bottom_chord_donor_rows)

        # Empurrão final: depois das sapatas e do banzo inferior doador, usa a
        # massa restante em órbitas primárias comprimidas.  A ordem é importante:
        # apoio é reforço barato; banzos/montantes são reforços distribuídos.
        final_strength_push_rows: List[Dict[str, Any]] = []
        if bool((base.get("member_sizing", {}) or {}).get("enable_final_strength_reserve_push", True)):
            ms_base = base.get("member_sizing", {}) or {}
            repeat_passes = max(1, int(ms_base.get("final_strength_push_repeat_passes", 2)))
            if bool(ms_base.get("final_strength_push_dynamic_recompute", True)):
                repeat_passes = 1
            push_input_cfg = (keep_s6[0]["config"] if keep_s6 else keep_s5[0]["config"])
            last_pushed: Dict[str, Any] | None = None
            for pass_idx in range(repeat_passes):
                stage_label = "S6_FINAL_STRENGTH_PUSH" if pass_idx == 0 else f"S6_FINAL_STRENGTH_PUSH_R{pass_idx + 1}"
                candidate_label = "S6P-0001" if pass_idx == 0 else f"S6P{pass_idx + 1}-0001"
                pushed = self._final_strength_reserve_push(
                    push_input_cfg,
                    load_cases,
                    stage_name=stage_label,
                    tension_only=tension_only_s6,
                )
                push_rows = [
                    {"candidate_id": candidate_label, **r}
                    for r in (pushed.get("trace_rows") or [])
                ]
                if not push_rows:
                    break
                final_strength_push_rows.extend(push_rows)
                last_pushed = pushed
                push_input_cfg = pushed["best_cfg"]

            if last_pushed is not None:
                s = last_pushed["summary"]
                s6_rows = [
                    {
                        "stage": "S6_FINAL_STRENGTH_PUSH",
                        "candidate_id": "S6P-0001",
                        "objective": s.get("objective"),
                        "valid_for_selection": s.get("valid_for_selection"),
                        "predicted_breaking_load_proxy_kgf": s.get("predicted_breaking_load_proxy_kgf"),
                        "min_fs_design_proxy": s.get("min_fs_design_proxy"),
                        "dead_weight_proxy_g": s.get("dead_weight_proxy_g"),
                        "config": last_pushed["best_cfg"],
                    }
                ]
                keep_s6 = s6_rows[:1]
        GeometryService.write_csv(out / "final_strength_reserve_push.csv", final_strength_push_rows)

        late_cross_swap_rows: List[Dict[str, Any]] = []
        if bool((base.get("member_sizing", {}) or {}).get("enable_late_cross_group_strength_swap", True)):
            swap_input_cfg = (keep_s6[0]["config"] if keep_s6 else keep_s5[0]["config"])
            swapped = self._late_cross_group_strength_swap(
                swap_input_cfg,
                load_cases,
                stage_name="S6_LATE_CROSS_GROUP_SWAP",
                tension_only=tension_only_s6,
            )
            late_cross_swap_rows = [
                {"candidate_id": "S6X-0001", **r}
                for r in (swapped.get("trace_rows") or [])
            ]
            if late_cross_swap_rows:
                s = swapped["summary"]
                s6_rows = [
                    {
                        "stage": "S6_LATE_CROSS_GROUP_SWAP",
                        "candidate_id": "S6X-0001",
                        "objective": s.get("objective"),
                        "valid_for_selection": s.get("valid_for_selection"),
                        "predicted_breaking_load_proxy_kgf": s.get("predicted_breaking_load_proxy_kgf"),
                        "min_fs_design_proxy": s.get("min_fs_design_proxy"),
                        "dead_weight_proxy_g": s.get("dead_weight_proxy_g"),
                        "config": swapped["best_cfg"],
                    }
                ]
                keep_s6 = s6_rows[:1]
        GeometryService.write_csv(out / "late_cross_group_strength_swap.csv", late_cross_swap_rows)

        late_nominal_topoff_rows: List[Dict[str, Any]] = []
        if bool((base.get("member_sizing", {}) or {}).get("enable_late_nominal_strength_topoff", True)):
            topoff_input_cfg = (keep_s6[0]["config"] if keep_s6 else keep_s5[0]["config"])
            topoff = self._late_nominal_strength_topoff(
                topoff_input_cfg,
                load_cases,
                stage_name="S6_LATE_NOMINAL_TOPOFF",
                tension_only=tension_only_s6,
            )
            late_nominal_topoff_rows = [
                {"candidate_id": "S6N-0001", **r}
                for r in (topoff.get("trace_rows") or [])
            ]
            if late_nominal_topoff_rows:
                s = topoff["summary"]
                s6_rows = [
                    {
                        "stage": "S6_LATE_NOMINAL_TOPOFF",
                        "candidate_id": "S6N-0001",
                        "objective": s.get("objective"),
                        "valid_for_selection": s.get("valid_for_selection"),
                        "predicted_breaking_load_proxy_kgf": s.get("predicted_breaking_load_proxy_kgf"),
                        "min_fs_design_proxy": s.get("min_fs_design_proxy"),
                        "dead_weight_proxy_g": s.get("dead_weight_proxy_g"),
                        "config": topoff["best_cfg"],
                    }
                ]
                keep_s6 = s6_rows[:1]
        GeometryService.write_csv(out / "late_nominal_strength_topoff.csv", late_nominal_topoff_rows)

        late_multicase_reinvest_rows: List[Dict[str, Any]] = []
        if bool((base.get("member_sizing", {}) or {}).get("enable_late_multicase_strength_reinvestment", True)):
            reinvest_input_cfg = (keep_s6[0]["config"] if keep_s6 else keep_s5[0]["config"])
            reinvest = self._late_multicase_strength_reinvestment(
                reinvest_input_cfg,
                load_cases,
                stage_name="S6_LATE_MULTICASE_REINVEST",
                tension_only=tension_only_s6,
            )
            late_multicase_reinvest_rows = [
                {"candidate_id": "S6M-0001", **r}
                for r in (reinvest.get("trace_rows") or [])
            ]
            if late_multicase_reinvest_rows:
                s = reinvest["summary"]
                s6_rows = [
                    {
                        "stage": "S6_LATE_MULTICASE_REINVEST",
                        "candidate_id": "S6M-0001",
                        "objective": s.get("objective"),
                        "valid_for_selection": s.get("valid_for_selection"),
                        "predicted_breaking_load_proxy_kgf": s.get("predicted_breaking_load_proxy_kgf"),
                        "min_fs_design_proxy": s.get("min_fs_design_proxy"),
                        "dead_weight_proxy_g": s.get("dead_weight_proxy_g"),
                        "config": reinvest["best_cfg"],
                    }
                ]
                keep_s6 = s6_rows[:1]
        GeometryService.write_csv(out / "late_multicase_strength_reinvestment.csv", late_multicase_reinvest_rows)

        late_basic_7030_rows: List[Dict[str, Any]] = []
        if bool((base.get("member_sizing", {}) or {}).get("enable_late_basic_7030_target_recovery", True)):
            basic_input_cfg = (keep_s6[0]["config"] if keep_s6 else keep_s5[0]["config"])
            basic_recovery = self._late_basic_7030_target_recovery(
                basic_input_cfg,
                load_cases,
                stage_name="S6_LATE_BASIC_7030_TARGET_RECOVERY",
                tension_only=tension_only_s6,
            )
            late_basic_7030_rows = [
                {"candidate_id": "S6B-0001", **r}
                for r in (basic_recovery.get("trace_rows") or [])
            ]
            if late_basic_7030_rows:
                s = basic_recovery["summary"]
                s6_rows = [
                    {
                        "stage": "S6_LATE_BASIC_7030_TARGET_RECOVERY",
                        "candidate_id": "S6B-0001",
                        "objective": s.get("objective"),
                        "valid_for_selection": s.get("valid_for_selection"),
                        "predicted_breaking_load_proxy_kgf": s.get("predicted_breaking_load_proxy_kgf"),
                        "min_fs_design_proxy": s.get("min_fs_design_proxy"),
                        "dead_weight_proxy_g": s.get("dead_weight_proxy_g"),
                        "config": basic_recovery["best_cfg"],
                    }
                ]
                keep_s6 = s6_rows[:1]
        GeometryService.write_csv(out / "late_basic_7030_target_recovery.csv", late_basic_7030_rows)

        final_mass_trim_rows: List[Dict[str, Any]] = []
        if bool((base.get("member_sizing", {}) or {}).get("enable_final_mass_symmetry_trim", True)):
            trim_input_cfg = (keep_s6[0]["config"] if keep_s6 else keep_s5[0]["config"])
            trimmed = self._final_mass_symmetry_trim(
                trim_input_cfg,
                load_cases,
                stage_name="S6_FINAL_MASS_TRIM",
                tension_only=tension_only_s6,
            )
            final_mass_trim_rows = [
                {"candidate_id": "S6T-0001", **r}
                for r in (trimmed.get("trace_rows") or [])
            ]
            if any(bool(r.get("accepted")) for r in final_mass_trim_rows):
                s = trimmed["summary"]
                s6_rows = [
                    {
                        "stage": "S6_FINAL_MASS_TRIM",
                        "candidate_id": "S6T-0001",
                        "objective": s.get("objective"),
                        "valid_for_selection": s.get("valid_for_selection"),
                        "predicted_breaking_load_proxy_kgf": s.get("predicted_breaking_load_proxy_kgf"),
                        "min_fs_design_proxy": s.get("min_fs_design_proxy"),
                        "dead_weight_proxy_g": s.get("dead_weight_proxy_g"),
                        "config": trimmed["best_cfg"],
                    }
                ]
                keep_s6 = s6_rows[:1]
        GeometryService.write_csv(out / "final_mass_symmetry_trim.csv", final_mass_trim_rows)

        symmetry_audit_rows = self._primary_symmetry_orbit_audit(
            (keep_s6[0]["config"] if keep_s6 else keep_s5[0]["config"]),
            load_cases,
            stage_name="S6_SYMMETRY_AUDIT",
            tension_only=tension_only_s6,
        )
        GeometryService.write_csv(out / "symmetry_audit.csv", symmetry_audit_rows)

        emit_progress(0.86, "S7: detalhamento de fabricação")
        t7 = time.perf_counter()
        s7_rows: List[Dict[str, Any]] = []
        best_cfg_s7 = keep_s6[0]["config"] if keep_s6 else keep_s5[0]["config"]
        detail_dir = out / "s7_fabrication"
        metrics_s7 = self.planner._evaluate_config(best_cfg_s7, include_detail=True, detail_dir=detail_dir)
        detailed_s7 = metrics_s7.get("detailed") or {}
        summary_s7 = detailed_s7.get("summary", {}) or {}
        GeometryService.write_csv(
            out / "fabrication_summary.csv",
            [
                {
                    "installed_stick_mass_g": summary_s7.get("installed_stick_mass_g"),
                    "purchased_stick_mass_g": summary_s7.get("purchased_stick_mass_g"),
                    "cutting_scrap_mass_g": summary_s7.get("cutting_scrap_mass_g"),
                    "wet_glue_mass_g": summary_s7.get("wet_glue_mass_g"),
                    "cured_glue_mass_g": summary_s7.get("cured_glue_mass_g"),
                    "competition_mass_g": summary_s7.get("competition_mass_g"),
                }
            ],
        )
        GeometryService.write_csv(out / "stick_cut_list.csv", detailed_s7.get("cutting_list") or [])
        GeometryService.write_csv(out / "glue_joints.csv", detailed_s7.get("glue_joints") or [])
        GeometryService.write_csv(
            out / "mass_breakdown.csv",
            [
                {"item": "installed_stick_mass_g", "value_g": summary_s7.get("installed_stick_mass_g")},
                {"item": "purchased_stick_mass_g", "value_g": summary_s7.get("purchased_stick_mass_g")},
                {"item": "cutting_scrap_mass_g", "value_g": summary_s7.get("cutting_scrap_mass_g")},
                {"item": "wet_glue_mass_g", "value_g": summary_s7.get("wet_glue_mass_g")},
                {"item": "cured_glue_mass_g", "value_g": summary_s7.get("cured_glue_mass_g")},
                {"item": "competition_mass_g", "value_g": summary_s7.get("competition_mass_g")},
            ],
        )
        s7_rows.append(
            {
                "stage": "S7",
                "candidate_id": "S7-0001",
                "installed_stick_mass_g": summary_s7.get("installed_stick_mass_g"),
                "purchased_stick_mass_g": summary_s7.get("purchased_stick_mass_g"),
                "cutting_scrap_mass_g": summary_s7.get("cutting_scrap_mass_g"),
                "wet_glue_mass_g": summary_s7.get("wet_glue_mass_g"),
                "cured_glue_mass_g": summary_s7.get("cured_glue_mass_g"),
                "competition_mass_g": summary_s7.get("competition_mass_g"),
                "config": best_cfg_s7,
            }
        )
        stage_times["S7"] = time.perf_counter() - t7

        emit_progress(0.93, "S8: validação final")
        t8 = time.perf_counter()
        final_summary = self._multi_case_summary(
            best_cfg_s7,
            load_cases,
            stage_name="S8",
            tension_only=tension_only_s8,
        )
        final_case_diag_rows = []
        for c in final_summary.get("cases") or []:
            final_case_diag_rows.append(
                {
                    "case": c.get("case"),
                    "solver_status": c.get("solver_status"),
                    "solver_regular": c.get("solver_regular"),
                    "equilibrium_ok": c.get("equilibrium_ok"),
                    "equilibrium_error_N": c.get("equilibrium_error_N"),
                    "min_fs_primary": c.get("min_fs_primary"),
                    "min_fs_design": c.get("min_fs_design"),
                    "predicted_breaking_load_proxy_kgf": c.get("predicted_breaking_load_proxy_kgf"),
                    "max_displacement_proxy_mm": c.get("max_displacement_proxy_mm"),
                    "support_reaction_balance": c.get("support_reaction_balance"),
                    "load_path_score": c.get("load_path_score"),
                    "mass_proxy_g": c.get("mass_proxy_g"),
                }
            )
        GeometryService.write_csv(out / "s8_case_diagnostics.csv", final_case_diag_rows)
        predicted_break = safe_float(final_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        min_fs = safe_float(final_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        mass_comp = safe_float(summary_s7.get("competition_mass_g"), safe_float(final_summary.get("dead_weight_proxy_g"), 0.0)) or 0.0
        target_break = float(base.get("analysis", {}).get("acceptance_min_design_breaking_load_kgf", 80.0))
        min_primary_req = float(base.get("analysis", {}).get("acceptance_min_primary_fs", 1.05))
        mass_limit = float(effective_mass_limit_g(best_cfg_s7))

        verdict = "APROVADA"
        failed_restriction = ""
        if not bool(final_summary.get("solver_regular")):
            verdict = "REPROVADA"
            failed_restriction = "solver_irregular"
        elif not bool(final_summary.get("equilibrium_ok")):
            verdict = "REPROVADA"
            failed_restriction = "equilibrium"
        elif mass_comp > mass_limit + 1.0e-6:
            verdict = "REPROVADA"
            failed_restriction = f"mass_above_limit:{mass_comp:.2f}>{mass_limit:.2f}"
        elif predicted_break < target_break:
            verdict = "REPROVADA"
            failed_restriction = f"rupture_below_target:{predicted_break:.2f}<{target_break:.2f}"
        elif min_fs < min_primary_req:
            verdict = "REPROVADA"
            failed_restriction = f"fs_below_acceptance:{min_fs:.3f}<{min_primary_req:.3f}"

        final_row = {
            "stage": "S8",
            "candidate_id": "S8-0001",
            "objective": final_summary.get("objective"),
            "solver_regular": final_summary.get("solver_regular"),
            "equilibrium_ok": final_summary.get("equilibrium_ok"),
            "min_fs_design": min_fs,
            "predicted_breaking_load_kgf": predicted_break,
            "competition_mass_g": mass_comp,
            "verdict": verdict,
            "failed_restriction": failed_restriction,
            "target_breaking_load_kgf": target_break,
            "mass_limit_g": mass_limit,
            "case_metrics": final_case_diag_rows,
            "config": best_cfg_s7,
        }
        assert_mass_compliant(final_row, best_cfg_s7, source="funnel_final")

        final_validation = {
            "verdict": verdict,
            "failed_restriction": failed_restriction,
            "predicted_breaking_load_kgf": predicted_break,
            "target_breaking_load_kgf": target_break,
            "competition_mass_g": mass_comp,
            "min_fs_design": min_fs,
            "solver_regular": bool(final_summary.get("solver_regular")),
            "equilibrium_ok": bool(final_summary.get("equilibrium_ok")),
            "hits_target_80kgf": bool(predicted_break >= 80.0),
        }
        (out / "final_validation_summary.json").write_text(
            json.dumps(final_validation, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        stage_times["S8"] = time.perf_counter() - t8

        pipeline_trace = {
            "stage_candidate_counts": {
                "S1": len(s1_rows),
                "S2": len(s2_rows),
                "S2_top_k": len(keep_s2),
                "S3": len(s3_rows),
                "S3_top_k": len(keep_s3),
                "S4": len(s4_rows),
                "S4_top_k": len(keep_s4),
                "S5": len(s5_rows),
                "S5_kept": len(keep_s5),
                "S6": len(s6_rows),
                "S7": len(s7_rows),
                "S8": 1,
            },
            "stage_time_seconds": stage_times,
            "stage_solves": self._stage_solves,
            "total_solves": int(sum(self._stage_solves.values())),
            "cache": {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "entries": len(self._case_cache),
            },
            "best_candidates": {
                "S2": (keep_s2[0].get("candidate_id") if keep_s2 else None),
                "S3": (keep_s3[0].get("candidate_id") if keep_s3 else None),
                "S4": (keep_s4[0].get("candidate_id") if keep_s4 else None),
                "S5": (keep_s5[0].get("candidate_id") if keep_s5 else None),
                "S6": (keep_s6[0].get("candidate_id") if keep_s6 else None),
                "S8": final_row.get("candidate_id"),
            },
            "topology_before_after": {
                "before": {
                    "objective": keep_s5[0].get("objective") if keep_s5 else None,
                    "predicted_breaking_load_proxy_kgf": keep_s5[0].get("predicted_breaking_load_proxy_kgf") if keep_s5 else None,
                    "dead_weight_proxy_g": keep_s5[0].get("dead_weight_proxy_g") if keep_s5 else None,
                },
                "after": {
                    "objective": keep_s6[0].get("objective") if keep_s6 else None,
                    "predicted_breaking_load_proxy_kgf": keep_s6[0].get("predicted_breaking_load_proxy_kgf") if keep_s6 else None,
                    "dead_weight_proxy_g": keep_s6[0].get("dead_weight_proxy_g") if keep_s6 else None,
                },
            },
        }
        (out / "pipeline_trace.json").write_text(
            json.dumps(pipeline_trace, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        emit_progress(1.0, "Funil S0..S8 concluído")
        emit_log(
            "Resumo do funil: "
            f"S1={len(s1_rows)} | S2={len(s2_rows)}→{len(keep_s2)} | "
            f"S3={len(s3_rows)}→{len(keep_s3)} | S4={len(s4_rows)}→{len(keep_s4)} | "
            f"S5={len(s5_rows)}→{len(keep_s5)} | S6={len(s6_rows)} | "
            f"solves={pipeline_trace['total_solves']}"
        )

        # Compatibilidade com consumidores legados (stage1..stage4).
        stage1_compat = [{k: v for k, v in r.items() if k != "config"} | {"config": r["config"]} for r in s2_rows]
        stage2_compat = [{k: v for k, v in r.items() if k != "config"} | {"config": r["config"]} for r in s3_rows]
        stage3_compat = [{k: v for k, v in r.items() if k != "config"} | {"config": r["config"]} for r in s5_rows]
        stage4_compat = [{k: v for k, v in r.items() if k != "config"} | {"config": r["config"]} for r in s6_rows]

        best_row = {
            **final_row,
            "feasible": verdict == "APROVADA",
            "score": final_row.get("objective"),
            "min_fs_primary": final_row.get("min_fs_design"),
            "min_fs_design": final_row.get("min_fs_design"),
            "mass_g": final_row.get("competition_mass_g"),
            "solver_status": "regular" if bool(final_row.get("solver_regular")) else "singular",
            "equilibrium_error_N": 0.0 if bool(final_row.get("equilibrium_ok")) else 1.0,
            "predicted_breaking_load_kgf": final_row.get("predicted_breaking_load_kgf"),
            "config": best_cfg_s7,
        }

        return {
            "s0_precheck": [s0],
            "s1_macro": s1_rows,
            "s2_fast_screening": s2_rows,
            "s3_multi_loadcase": s3_rows,
            "s4_geometry_refinement": s4_rows,
            "s4_geometry_trace": s4_trace_rows,
            "s5_member_sizing": s5_rows,
            "s6_topology": s6_rows,
            "removed_members": removed_members,
            "mixed_panel_patterns": mixed_patterns,
            "mass_reallocation_after_topology": mass_realloc_rows,
            "post_topology_reinvestment": reinvest_trace_rows,
            "post_reinvest_rebalance": rebalance_trace_rows,
            "section_efficiency_mutation": section_eff_trace_rows,
            "plane_bracing_efficiency_mutation": plane_bracing_eff_rows,
            "final_strength_reserve_push": final_strength_push_rows,
            "late_cross_group_strength_swap": late_cross_swap_rows,
            "late_nominal_strength_topoff": late_nominal_topoff_rows,
            "late_height_strength_mutation": late_height_rows,
            "support_pad_capacity_push": support_pad_push_rows,
            "symmetry_audit": symmetry_audit_rows,
            "s7_fabrication": s7_rows,
            "s8_final_validation": [final_row],
            "discarded": discarded_rows,
            "logs": logs,
            "best": best_row,
            "best_is_feasible": bool(best_row.get("feasible")),
            "stage1": stage1_compat,
            "stage2": stage2_compat,
            "stage3": stage3_compat,
            "stage4_trace": s4_trace_rows,
            "stage4": stage4_compat,
            "final_variants": {},
            "stage_counts": {
                "stage0_generated": len(s1_rows),
                "stage0_prefilter_passed": len(s1_rows),
                "stage0_prefilter_discarded": 0,
                "stage1": len(stage1_compat),
                "stage2": len(stage2_compat),
                "stage3": len(stage3_compat),
                "stage4": len(stage4_compat),
                "S1_macro_candidates": len(s1_rows),
                "S2_fast_screening_candidates": len(s2_rows),
                "S2_fast_screening_top_k": len(keep_s2),
                "S3_multi_loadcase_candidates": len(s3_rows),
                "S3_multi_loadcase_top_k": len(keep_s3),
                "S4_geometry_refinement_candidates": len(s4_rows),
                "S4_geometry_refinement_top_k": len(keep_s4),
                "S5_member_sizing_candidates": len(s5_rows),
                "S6_topology_candidates": len(s6_rows),
                "S7_fabrication_candidates": len(s7_rows),
                "S8_final_validation_candidates": 1,
                "solves_total": int(sum(self._stage_solves.values())),
                "solves_by_stage": dict(self._stage_solves),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
            },
            "pipeline_trace_path": str(out / "pipeline_trace.json"),
            "final_validation_path": str(out / "final_validation_summary.json"),
        }
