from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

from src.core.numeric import safe_float
from src.domain.models import Load
from src.services.geometry_service import GeometryService
from src.services.mass_guard import assert_mass_compliant, effective_mass_limit_g
from src.services.rupture_estimator import estimate_rupture_load


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


class StagedFidelityFunnelPlanner:
    """Executa o funil S0..S8 com fidelidade crescente e cortes de custo."""

    def __init__(self, planner: Any) -> None:
        self.planner = planner
        self._case_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._stage_solves: Dict[str, int] = {}

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
        return str(status or "").split("|", 1)[0] == "regular"

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
        top_nodes = [n for n in nodes if getattr(n, "level", "") == "top" and -1.0e-6 <= float(n.x) <= span + 1.0e-6]
        if not top_nodes:
            top_nodes = [n for n in nodes if getattr(n, "level", "") == "top"]
        xs = sorted({round(float(n.x), 6) for n in top_nodes})
        if not xs:
            return []

        def closest_x(target: float) -> float:
            return min(xs, key=lambda xv: abs(float(xv) - float(target)))

        def node_ids_at_x(x_target: float) -> List[int]:
            x_use = closest_x(x_target)
            ids = [int(n.id) for n in top_nodes if abs(float(n.x) - float(x_use)) <= 1.0e-6]
            if ids:
                return sorted(ids)
            nearest = min(top_nodes, key=lambda n: abs(float(n.x) - float(x_target)))
            return [int(nearest.id)]

        loads: List[Load] = []
        case = str(load_case_name)

        if case == "center":
            ids = node_ids_at_x(0.5 * span)
            fz_each = -total_N / max(1, len(ids))
            for nid in ids:
                loads.append(Load(case, nid, 0.0, 0.0, fz_each))
            return loads

        if case == "left_offset":
            ids = node_ids_at_x(0.35 * span)
            fz_each = -total_N / max(1, len(ids))
            for nid in ids:
                loads.append(Load(case, nid, 0.0, 0.0, fz_each))
            return loads

        if case == "right_offset":
            ids = node_ids_at_x(0.65 * span)
            fz_each = -total_N / max(1, len(ids))
            for nid in ids:
                loads.append(Load(case, nid, 0.0, 0.0, fz_each))
            return loads

        if case == "torsion_60_40":
            ids = node_ids_at_x(0.5 * span)
            left_ids = [nid for nid in ids if any(int(n.id) == nid and float(n.y) < 0 for n in top_nodes)]
            right_ids = [nid for nid in ids if any(int(n.id) == nid and float(n.y) >= 0 for n in top_nodes)]
            if not left_ids:
                left_ids = ids[: max(1, len(ids) // 2)]
            if not right_ids:
                right_ids = ids[max(1, len(ids) // 2):] or ids
            f_left = -0.60 * total_N / max(1, len(left_ids))
            f_right = -0.40 * total_N / max(1, len(right_ids))
            for nid in left_ids:
                loads.append(Load(case, nid, 0.0, 0.0, f_left))
            for nid in right_ids:
                loads.append(Load(case, nid, 0.0, 0.0, f_right))
            return loads

        if case == "lateral_imperfection":
            ids = node_ids_at_x(0.5 * span)
            fz_each = -total_N / max(1, len(ids))
            lateral_factor = float((cfg.get("multi_loadcase_screening", {}) or {}).get("lateral_imperfection_factor", 0.02))
            fy_mag = lateral_factor * total_N / max(1, len(ids))
            for nid in ids:
                sign = 1.0
                for n in top_nodes:
                    if int(n.id) == int(nid):
                        sign = -1.0 if float(n.y) < 0 else 1.0
                        break
                loads.append(Load(case, nid, 0.0, sign * fy_mag, fz_each))
            return loads

        if case == "self_weight":
            # Peso próprio aproximado distribuído em todos os nós superiores e inferiores.
            gN = max(0.0, float(quick_mass_g) / 1000.0 * 9.80665)
            if gN <= 0.0:
                gN = 0.08 * total_N
            all_struct_nodes = [n for n in nodes if getattr(n, "level", "") in {"top", "bottom"}]
            fz_each = -gN / max(1, len(all_struct_nodes))
            for n in all_struct_nodes:
                loads.append(Load(case, int(n.id), 0.0, 0.0, fz_each))
            return loads

        # fallback: carga central
        ids = node_ids_at_x(0.5 * span)
        fz_each = -total_N / max(1, len(ids))
        for nid in ids:
            loads.append(Load(case, nid, 0.0, 0.0, fz_each))
        return loads

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
        if tension_only:
            kwargs["tension_only_solver_enabled"] = True

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

        min_fs_primary = min(
            [safe_float(r.get("FS_min"), 0.0) or 0.0 for r in member_checks if r.get("member_role") == "primary"],
            default=0.0,
        )
        min_fs_design = min(
            [safe_float(r.get("FS_design"), safe_float(r.get("FS_min"), 0.0)) or 0.0 for r in member_checks if r.get("design_relevant", True)],
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

        strength_to_weight_score = predicted_breaking_load_kgf / max(1.0, mass_g)
        strength_to_weight_score = _clamp(strength_to_weight_score / max(0.01, target_break / mass_limit), 0.0, 2.0)

        competition_mass_margin_score = _clamp((mass_limit - mass_g) / mass_limit, -1.0, 1.0)
        displacement_penalty = max(0.0, max_displacement_mm / 30.0 - 1.0)
        dead_weight_penalty = max(0.0, mass_g / mass_limit - 1.0)

        return (
            4.0 * _clamp(predicted_breaking_load_kgf / target_break, 0.0, 2.0)
            + 2.0 * _clamp(min_fs_design / acceptance_min_fs, 0.0, 2.0)
            + 1.5 * strength_to_weight_score
            + 1.0 * competition_mass_margin_score
            + 0.5 * constructability_score
            - 2.0 * mechanism_penalty
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
                "objective": -1.0e9,
                "cases": [],
            }

        min_fs_pre = min((safe_float(c.get("min_fs_primary"), 0.0) or 0.0 for c in cases), default=0.0)
        min_fs_design = min((safe_float(c.get("min_fs_design"), min_fs_pre) or min_fs_pre for c in cases), default=min_fs_pre)
        predicted_break = min((safe_float(c.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0 for c in cases), default=0.0)
        max_disp = max((safe_float(c.get("max_displacement_proxy_mm"), 0.0) or 0.0 for c in cases), default=0.0)
        mean_mass = sum((safe_float(c.get("mass_proxy_g"), 0.0) or 0.0 for c in cases)) / max(1, len(cases))

        all_regular = all(bool(c.get("solver_regular")) for c in cases)
        all_eq = all(bool(c.get("equilibrium_ok")) for c in cases)

        near_zero_sets = [set(c.get("near_zero_member_ids") or []) for c in cases]
        zero_force_intersection = set.intersection(*near_zero_sets) if near_zero_sets else set()

        lateral_case = next((c for c in cases if c.get("case") == "lateral_imperfection"), None)
        lateral_stability = 1.0 / max(1.0, (safe_float((lateral_case or {}).get("max_displacement_proxy_mm"), 0.0) or 0.0) / 6.0)

        nodal_stability_proxy = min((safe_float(c.get("nodal_stability_proxy"), 0.0) or 0.0 for c in cases), default=0.0)
        topology_stability_proxy = min((safe_float(c.get("topology_stability_proxy"), 0.0) or 0.0 for c in cases), default=0.0)
        support_balance = min((safe_float(c.get("support_reaction_balance"), 0.0) or 0.0 for c in cases), default=0.0)
        load_path = min((safe_float(c.get("load_path_score"), 0.0) or 0.0 for c in cases), default=0.0)
        buckling_risk = max((safe_float(c.get("buckling_risk_proxy"), 0.0) or 0.0 for c in cases), default=0.0)

        mass_limit = max(1.0, float(effective_mass_limit_g(cfg)))
        complexity_penalty = self._topology_complexity_penalty(
            int(len((cases[0] or {}).get("member_results") or []))
        )
        mechanism_penalty = 0.0 if (all_regular and topology_stability_proxy > 0.0) else 1.0
        constructability = _clamp(0.5 * support_balance + 0.5 * load_path, 0.0, 1.0)
        glue_overuse_penalty = 0.0

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
            "multi_case_zero_force_members": len(zero_force_intersection),
            "zero_force_member_ids": sorted(zero_force_intersection),
            "nodal_stability_proxy": nodal_stability_proxy,
            "lateral_stability_proxy": lateral_stability,
            "support_reaction_balance": support_balance,
            "load_path_score": load_path,
            "topology_stability_proxy": topology_stability_proxy,
            "objective": objective,
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
        if abs(rows["stick_length_mm"] - 115.0) > 1.0e-6:
            violations.append(f"Palito inválido: comprimento {rows['stick_length_mm']:.1f} mm (referência 115 mm).")
        if abs(rows["stick_width_mm"] - 7.0) > 1.0e-6:
            violations.append(f"Palito inválido: largura {rows['stick_width_mm']:.2f} mm (referência 7.0 mm).")
        if abs(rows["stick_thickness_mm"] - 1.5) > 1.0e-6:
            violations.append(f"Palito inválido: espessura {rows['stick_thickness_mm']:.2f} mm (referência 1.5 mm).")
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
                "macro_name": "k_truss",
                "global_pattern": "k",
                "side_truss_type": "K_symmetric",
                "reinforcement_profile": "strong",
            },
            {
                **base,
                "macro_name": "x_bracing_tension_only",
                "global_pattern": "x_tension",
                "side_truss_type": "Pratt_symmetric",
                "internal_truss_type": "X",
                "top_chord_truss_type": "X",
                "bottom_chord_truss_type": "X",
                "reinforcement_profile": "light",
                "force_tension_only_bracing": True,
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
                "panel_mm": max(70.0, panel * 0.80),
                "reinforcement_profile": "strong",
            },
            {
                **base,
                "macro_name": "high_variant",
                "global_pattern": "high",
                "side_truss_type": "Pratt_symmetric",
                "center_height_mm": min(700.0, height * 1.20),
                "reinforcement_profile": "strong_top",
            },
            {
                **base,
                "macro_name": "wide_torsional_mixed",
                "global_pattern": "wide_mixed",
                "side_truss_type": "Pratt_symmetric",
                "width_mm": min(200.0, max(100.0, width * 1.20)),
                "panel_side_truss_pattern": {
                    "0": "Pratt_symmetric",
                    "1": "Warren_symmetric",
                    "2": "Pratt_symmetric",
                    "3": "Warren_mid_braced",
                    "4": "Pratt_symmetric",
                    "5": "Warren_symmetric",
                },
                "reinforcement_profile": "strong",
            },
        ]

        macro_count = max(8, min(12, int(macro_count)))
        return macros[:macro_count]

    def _macro_to_config(self, base_cfg: Dict[str, Any], macro: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.planner._apply_candidate_geometry(base_cfg, macro)
        self.planner._apply_reinforcement_profile(cfg, str(macro.get("reinforcement_profile", "balanced")))
        bridge = cfg.setdefault("bridge", {})
        for k in ("panel_side_truss_pattern", "panel_top_chord_pattern", "panel_bottom_chord_pattern"):
            if k in macro:
                bridge[k] = copy.deepcopy(macro[k])
        self._sync_bridge_contacts(bridge)
        if bool(macro.get("force_tension_only_bracing", False)):
            bridge["tension_only_bracing_solver_enabled"] = True
            bridge["tension_only_bracing_interpretation"] = True
        return self.planner.config.normalize(cfg)

    @staticmethod
    def _pick_with_diversity(rows: List[Dict[str, Any]], top_k: int, key_field: str = "global_pattern") -> List[Dict[str, Any]]:
        if len(rows) <= top_k:
            return list(rows)
        ordered = sorted(rows, key=lambda r: safe_float(r.get("quick_score", r.get("objective", -1.0e99)), -1.0e99) or -1.0e99, reverse=True)
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
        predicted_break = safe_float(case_row.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        min_fs = safe_float(case_row.get("min_fs_design"), 0.0) or 0.0
        mass = safe_float(case_row.get("mass_proxy_g"), 0.0) or 0.0
        max_disp = safe_float(case_row.get("max_displacement_proxy_mm"), 0.0) or 0.0
        mechanism_penalty = 0.0 if bool(case_row.get("solver_regular")) else 1.0
        complexity = self._topology_complexity_penalty(int(len(case_row.get("member_results") or [])))
        constructability = 0.5 * (safe_float(case_row.get("load_path_score"), 0.0) or 0.0) + 0.5 * (safe_float(case_row.get("support_reaction_balance"), 0.0) or 0.0)
        return self._objective_score(
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

    def _trust_region_refine(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
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
        cur_summary = self._multi_case_summary(cur_cfg, load_cases, stage_name=stage_name, tension_only=True)
        best_cfg = cur_cfg
        best_summary = cur_summary
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
                    summary = self._multi_case_summary(cand_cfg, load_cases, stage_name=stage_name, tension_only=True)
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
                    "predicted_breaking_load_proxy_kgf": summary.get("predicted_breaking_load_proxy_kgf"),
                    "min_fs_design_proxy": summary.get("min_fs_design_proxy"),
                    "dead_weight_proxy_g": summary.get("dead_weight_proxy_g"),
                    "solver_regular": summary.get("solver_regular"),
                    "equilibrium_ok": summary.get("equilibrium_ok"),
                    "accepted": False,
                }
                trace_rows.append(row)
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

    def _member_sizing_pass(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
    ) -> Dict[str, Any]:
        center_case = self._evaluate_case_cached(cfg, "center", stage_name=stage_name, tension_only=True)
        plan = self.planner.build_member_sizing_plan(
            cfg,
            center_case.get("nodes") or [],
            center_case.get("members") or [],
            center_case.get("member_results") or [],
            center_case.get("member_checks") or [],
        )

        trace_rows: List[Dict[str, Any]] = []
        donors: List[Dict[str, Any]] = []
        critical: List[Dict[str, Any]] = []
        before_after: List[Dict[str, Any]] = []

        for decision in plan.values():
            row = {
                "member_id": decision.member_id,
                "group": decision.original_group,
                "N_N": decision.force_N,
                "compression_direct_util": None,
                "tension_util": None,
                "buckling_util_y": None,
                "buckling_util_z": None,
                "beam_column_util": None,
                "governing_mode": decision.governing_mode,
                "FS_min": decision.FS_min,
                "utilization": 1.0 / max(1.0e-9, decision.FS_min),
                "action": decision.action,
                "n_sticks_current": decision.n_sticks_current,
                "n_sticks_recommended": decision.n_sticks_recommended,
                "delta_mass_g": decision.delta_mass_g,
                "reason": decision.reason,
                "can_be_mass_donor": decision.can_be_mass_donor,
            }
            trace_rows.append(row)
            if bool(decision.can_be_mass_donor):
                donors.append(row)
            if str(decision.action) == "reinforce":
                critical.append(row)
            if decision.n_sticks_current != decision.n_sticks_recommended:
                before_after.append(
                    {
                        "member_id": decision.member_id,
                        "group": decision.original_group,
                        "before_n_sticks": decision.n_sticks_current,
                        "after_n_sticks": decision.n_sticks_recommended,
                        "delta_mass_g": decision.delta_mass_g,
                        "action": decision.action,
                    }
                )

        new_cfg = self.planner.apply_member_sizing_plan(cfg, plan)
        before_summary = self._multi_case_summary(cfg, load_cases, stage_name=stage_name, tension_only=True)
        after_summary = self._multi_case_summary(new_cfg, load_cases, stage_name=stage_name, tension_only=True)

        if (safe_float(after_summary.get("objective"), -1.0e99) or -1.0e99) < (safe_float(before_summary.get("objective"), -1.0e99) or -1.0e99):
            # Mantém a versão original se o sizing piorar o objetivo global.
            return {
                "best_cfg": cfg,
                "summary": before_summary,
                "trace_rows": trace_rows,
                "donors": donors,
                "critical": critical,
                "before_after": before_after,
            }
        return {
            "best_cfg": new_cfg,
            "summary": after_summary,
            "trace_rows": trace_rows,
            "donors": donors,
            "critical": critical,
            "before_after": before_after,
        }

    def _topology_cleanup(
        self,
        cfg: Dict[str, Any],
        load_cases: List[str],
        *,
        stage_name: str,
    ) -> Dict[str, Any]:
        top_cfg = cfg.get("topology_cleanup", {}) or {}
        enabled = bool(top_cfg.get("enabled", True))
        if not enabled:
            base_summary = self._multi_case_summary(cfg, load_cases, stage_name=stage_name, tension_only=True)
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

        cur_cfg = self.planner.config.normalize(cfg)
        cur_summary = self._multi_case_summary(cur_cfg, load_cases, stage_name=stage_name, tension_only=True)
        trace_rows: List[Dict[str, Any]] = []
        removed_members: List[Dict[str, Any]] = []
        mixed_patterns: List[Dict[str, Any]] = []
        zero_force_diag: List[Dict[str, Any]] = []
        no_improve = 0

        for it in range(1, max_iters + 1):
            center_case = self._evaluate_case_cached(cur_cfg, "center", stage_name=stage_name, tension_only=True)
            member_results = center_case.get("member_results") or []
            max_abs = max((abs(safe_float(r.get("N_N"), 0.0) or 0.0) for r in member_results), default=1.0)
            zero_threshold = max(near_zero_N, near_zero_rel * max_abs)
            zero_ids = [
                int(r.get("member_id"))
                for r in member_results
                if abs(safe_float(r.get("N_N"), 0.0) or 0.0) <= zero_threshold
            ]
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
            for mid in zero_ids[:8]:
                c = copy.deepcopy(cur_cfg)
                c.setdefault("member_active_by_id", {})[str(int(mid))] = False
                disabled = {
                    int(v)
                    for v in (c.get("disabled_member_ids", []) or [])
                    if str(v).strip()
                }
                disabled.add(int(mid))
                c["disabled_member_ids"] = sorted(disabled)
                c = self.planner.config.normalize(c)
                candidates.append(("remove_member", c, {"member_id": int(mid)}))

            # 2) Mutações globais e mistas.
            for side_mode, op in [
                ("Pratt_symmetric", "convert_panel_to_pratt"),
                ("Howe_inverted", "convert_panel_to_howe"),
                ("Warren_symmetric", "convert_panel_to_warren"),
                ("K_symmetric", "convert_panel_to_k_truss"),
            ]:
                if str(cur_cfg.get("bridge", {}).get("side_truss_type")) == side_mode:
                    continue
                c = copy.deepcopy(cur_cfg)
                c.setdefault("bridge", {})["side_truss_type"] = side_mode
                c = self.planner.config.normalize(c)
                candidates.append((op, c, {"side_truss_type": side_mode}))

            c_mixed = copy.deepcopy(cur_cfg)
            span = float(c_mixed.get("bridge", {}).get("span_mm", 1200.0))
            panel = max(40.0, float(c_mixed.get("bridge", {}).get("panel_mm", 100.0)))
            n_panels = max(2, int(round(span / panel)))
            pattern = {}
            for i in range(n_panels):
                pattern[str(i)] = "Pratt_symmetric" if i % 2 == 0 else "Warren_symmetric"
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

            best_iter = None
            for op_name, cand_cfg, op_meta in candidates:
                try:
                    summary = self._multi_case_summary(cand_cfg, load_cases, stage_name=stage_name, tension_only=True)
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
                    "predicted_breaking_load_proxy_kgf": summary.get("predicted_breaking_load_proxy_kgf"),
                    "min_fs_design_proxy": summary.get("min_fs_design_proxy"),
                    "dead_weight_proxy_g": summary.get("dead_weight_proxy_g"),
                    "solver_regular": summary.get("solver_regular"),
                    "equilibrium_ok": summary.get("equilibrium_ok"),
                }
                row.update(op_meta)
                trace_rows.append(row)
                if best_iter is None or (safe_float(summary.get("objective"), -1.0e99) or -1.0e99) > (safe_float(best_iter[0].get("objective"), -1.0e99) or -1.0e99):
                    best_iter = (summary, cand_cfg, row)

            if best_iter is None:
                break

            best_summary, best_cfg, best_row = best_iter
            cur_obj = safe_float(cur_summary.get("objective"), -1.0e99) or -1.0e99
            new_obj = safe_float(best_summary.get("objective"), -1.0e99) or -1.0e99
            if new_obj > cur_obj + 1.0e-9:
                cur_cfg = best_cfg
                cur_summary = best_summary
                no_improve = 0
                if best_row.get("operation") == "remove_member" and best_row.get("member_id") is not None:
                    removed_members.append(
                        {
                            "iteration": it,
                            "member_id": best_row.get("member_id"),
                            "reason": "low_force_all_cases",
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
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        # Realoção simples de massa liberada por topologia.
        before_mass = safe_float(
            self._multi_case_summary(cfg, ["center"], stage_name=stage_name, tension_only=True).get("dead_weight_proxy_g"),
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
        macro_n = max(8, min(12, int(pp.get("macro_candidates_count", 10))))
        top_k_fast = max(1, int(pp.get("fast_screening_keep_top_k", 3)))
        top_k_multi = max(1, int(pp.get("multi_loadcase_keep_top_k", 2)))
        top_k_s4 = max(1, int(pp.get("geometry_refinement_keep_top_k", 1)))

        ml_cfg = base.get("multi_loadcase_screening", {}) or {}
        load_cases = [str(v) for v in (ml_cfg.get("load_cases") or ["center", "left_offset", "right_offset", "torsion_60_40", "lateral_imperfection", "self_weight"])]

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
                    "geometry_hash": self._signature_hashes(cfg_i, "center")["geometry_hash"],
                    "topology_hash": self._signature_hashes(cfg_i, "center")["topology_hash"],
                    "sizing_hash": self._signature_hashes(cfg_i, "center")["sizing_hash"],
                    "load_case_hash": self._signature_hashes(cfg_i, "center")["load_case_hash"],
                    "config": cfg_i,
                }
            )
        s2_rows = sorted(s2_rows, key=lambda r: safe_float(r.get("quick_score"), -1.0e99) or -1.0e99, reverse=True)
        keep_s2 = self._pick_with_diversity(s2_rows, top_k_fast, key_field="global_pattern") if bool(pp.get("preserve_diversity_in_fast_screening", True)) else s2_rows[:top_k_fast]

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

        emit_progress(0.33, "S3: triagem multi-loadcase")
        t3 = time.perf_counter()
        s3_rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(keep_s2, 1):
            cfg_i = row["config"]
            summary = self._multi_case_summary(cfg_i, load_cases, stage_name="S3", tension_only=True)
            s3_rows.append(
                {
                    **{k: v for k, v in row.items() if k != "config"},
                    "stage": "S3",
                    "candidate_id": f"S3-{idx:04d}",
                    "objective": summary.get("objective"),
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
        keep_s3 = self._pick_with_diversity(s3_rows, top_k_multi, key_field="global_pattern")

        GeometryService.write_csv(out / "multi_loadcase_screening.csv", [{k: v for k, v in r.items() if k not in {"config", "case_metrics", "zero_force_member_ids"}} for r in s3_rows])
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

        emit_progress(0.50, "S4: refinamento geométrico local")
        t4 = time.perf_counter()
        s4_rows: List[Dict[str, Any]] = []
        s4_trace_rows: List[Dict[str, Any]] = []
        before_after_rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(keep_s3, 1):
            refined = self._trust_region_refine(row["config"], load_cases, stage_name="S4")
            summary = refined["best_summary"]
            s4_rows.append(
                {
                    **{k: v for k, v in row.items() if k != "config"},
                    "stage": "S4",
                    "candidate_id": f"S4-{idx:04d}",
                    "objective": summary.get("objective"),
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
        keep_s4 = s4_rows[:top_k_s4]
        GeometryService.write_csv(out / "geometry_refinement_trace.csv", s4_trace_rows)
        (out / "geometry_before_after.json").write_text(
            json.dumps(before_after_rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._write_geometry_refinement_plot(s4_trace_rows, out / "plot_geometry_refinement.png")
        stage_times["S4"] = time.perf_counter() - t4

        emit_progress(0.64, "S5: dimensionamento de membros")
        t5 = time.perf_counter()
        s5_rows: List[Dict[str, Any]] = []
        s5_trace: List[Dict[str, Any]] = []
        donors_all: List[Dict[str, Any]] = []
        critical_all: List[Dict[str, Any]] = []
        before_after_all: List[Dict[str, Any]] = []
        for idx, row in enumerate(keep_s4, 1):
            sized = self._member_sizing_pass(row["config"], load_cases, stage_name="S5")
            s = sized["summary"]
            s5_rows.append(
                {
                    **{k: v for k, v in row.items() if k != "config"},
                    "stage": "S5",
                    "candidate_id": f"S5-{idx:04d}",
                    "objective": s.get("objective"),
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
        keep_s5 = s5_rows[: max(1, min(2, len(s5_rows)) if bool(pp.get("allow_top2_full_detailing", False)) else 1)]
        GeometryService.write_csv(out / "member_sizing_trace.csv", s5_trace)
        GeometryService.write_csv(out / "mass_donor_members.csv", donors_all)
        GeometryService.write_csv(out / "critical_reinforcements.csv", critical_all)
        GeometryService.write_csv(out / "sizing_before_after.csv", before_after_all)
        stage_times["S5"] = time.perf_counter() - t5

        emit_progress(0.76, "S6: mutação topológica final")
        t6 = time.perf_counter()
        s6_rows: List[Dict[str, Any]] = []
        s6_trace: List[Dict[str, Any]] = []
        removed_members: List[Dict[str, Any]] = []
        mixed_patterns: List[Dict[str, Any]] = []
        zero_force_diag: List[Dict[str, Any]] = []
        mass_realloc_rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(keep_s5[:1], 1):
            topo = self._topology_cleanup(row["config"], load_cases, stage_name="S6")
            s = topo["summary"]
            s6_rows.append(
                {
                    **{k: v for k, v in row.items() if k != "config"},
                    "stage": "S6",
                    "candidate_id": f"S6-{idx:04d}",
                    "objective": s.get("objective"),
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
        keep_s6 = s6_rows[:1]
        GeometryService.write_csv(out / "topology_mutation_trace.csv", s6_trace)
        GeometryService.write_csv(out / "removed_members.csv", removed_members)
        GeometryService.write_csv(out / "mixed_panel_patterns.csv", mixed_patterns)
        GeometryService.write_csv(out / "zero_force_diagnostics.csv", zero_force_diag)
        GeometryService.write_csv(out / "mass_reallocation_after_topology.csv", mass_realloc_rows)
        stage_times["S6"] = time.perf_counter() - t6

        emit_progress(0.86, "S7: detalhamento de fabricação")
        t7 = time.perf_counter()
        s7_rows: List[Dict[str, Any]] = []
        best_cfg_s7 = keep_s6[0]["config"] if keep_s6 else (keep_s5[0]["config"] if keep_s5 else (keep_s4[0]["config"] if keep_s4 else base))
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
        final_summary = self._multi_case_summary(best_cfg_s7, load_cases, stage_name="S8", tension_only=True)
        predicted_break = safe_float(final_summary.get("predicted_breaking_load_proxy_kgf"), 0.0) or 0.0
        min_fs = safe_float(final_summary.get("min_fs_design_proxy"), 0.0) or 0.0
        mass_comp = safe_float(summary_s7.get("competition_mass_g"), safe_float(final_summary.get("dead_weight_proxy_g"), 0.0)) or 0.0
        target_break = float(base.get("analysis", {}).get("acceptance_min_design_breaking_load_kgf", 80.0))
        min_primary_req = float(base.get("analysis", {}).get("acceptance_min_primary_fs", 1.05))

        verdict = "APROVADA"
        failed_restriction = ""
        if not bool(final_summary.get("solver_regular")):
            verdict = "REPROVADA"
            failed_restriction = "solver_irregular"
        elif not bool(final_summary.get("equilibrium_ok")):
            verdict = "REPROVADA"
            failed_restriction = "equilibrium"
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
