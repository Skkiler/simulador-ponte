from __future__ import annotations

import copy
import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from src.core.numeric import safe_float
from src.domain.models import Member, Node
from src.services.config_service import ConfigService
from src.services.geometry_service import GeometryService
from src.services.postprocessor import PostProcessor
from src.services.stick_detail_service import StickDetailService
# Mass guard utilities ensure a single mass limit across planner, material and report.
from src.services.mass_guard import effective_mass_limit_g, assert_mass_compliant
# Rupture estimator computes estimated breaking load and diagnostic info
from src.services.rupture_estimator import estimate_rupture_load
# Connection planner chooses joint models per member automatically.
from src.services.connection_planner import ConnectionPlanner
from src.solvers.linear_truss_solver import LinearTrussSolver


@dataclass
class MemberSizingDecision:
    member_id: int
    original_group: str
    effective_group: str
    n_sticks_current: int
    n_sticks_recommended: int
    axial_force_N: float
    axial_force_ratio: float
    FS_min: float
    governing_mode: str
    action: str
    reason: str
    symmetry_partner_ids: List[int]
    applied_to_member_ids: List[int]


def min_numeric(values: Iterable[Any], default: float | None = None) -> float | None:
    clean = [v for v in (safe_float(x, None) for x in values) if v is not None]
    return min(clean) if clean else default


class ActiveDesignPlanner:
    """
    Planejador ativo em múltiplas etapas.

    Etapas:
    - stage 1: varredura ampla de tipologia e envelope geométrico.
    - stage 2: refino local de número de palitos por grupo.
    - stage 3: validação com detalhamento peça-a-peça e cola.
    """

    def __init__(self) -> None:
        self.config = ConfigService()
        self.geometry = GeometryService()
        self.solver = LinearTrussSolver()
        self.post = PostProcessor()
        self.detail = StickDetailService()
        self.connection = ConnectionPlanner()
        self._base_eval_cache: Dict[str, Dict] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_lock = threading.Lock()

    @staticmethod
    def _linspace_values(
        min_value: float,
        max_value: float,
        count: int,
        round_to: float = 1.0,
    ) -> List[float]:
        a = float(min_value)
        b = float(max_value)

        if b < a:
            a, b = b, a

        if abs(b - a) < 1e-9:
            return [round(a, 6)]

        if count <= 1:
            return [round((a + b) * 0.5, 6)]

        vals = [
            a + (b - a) * i / (count - 1)
            for i in range(count)
        ]

        if round_to > 0:
            vals = [round(v / round_to) * round_to for v in vals]

        return sorted(set(round(v, 6) for v in vals))

    @staticmethod
    def _quick_mass_estimate(cfg: Dict, members: List) -> tuple[float, int]:
        mat = cfg["material"]
        stick_len = max(1.0, float(mat["stick_length_mm"]))
        stick_mass = float(mat["stick_mass_g"])
        detail = cfg.get("detail_model", {})
        waste = float(detail.get("construction_waste_factor", 0.08))
        overlap = max(0.0, min(float(detail.get("overlap_length_mm", 30.0)), 0.85 * stick_len))
        kerf = max(0.0, float(detail.get("saw_kerf_mm", 1.0)))
        glue_spread = float(detail.get("glue_spread_g_per_m2", 160.0))
        glue_eff = max(1.0e-6, float(detail.get("glue_mass_efficiency", 0.65)))
        # Width of a single stick.  Use the configured value or the new default (7.0 mm).
        stick_w = float(mat.get("stick_width_mm", 7.0))

        # Calibração para aproximar o detalhamento completo sem rodar corte/cola detalhado.
        fast_mass_scale = max(0.75, min(1.05, float(detail.get("fast_mass_scale", 0.90))))

        total_cut_len = 0.0
        total_splices = 0

        for m in members:
            L = max(0.0, float(m.L))
            lanes = max(1, int(m.n_sticks))
            step = max(1.0e-6, stick_len - overlap)

            if L <= stick_len:
                pieces = 1
            else:
                pieces = int(math.ceil((L - stick_len) / step)) + 1

            splices = max(0, pieces - 1)
            lane_cut_len = L + splices * overlap + max(0, pieces - 1) * kerf
            total_cut_len += lanes * lane_cut_len
            total_splices += lanes * splices

        raw_sticks = int(math.ceil(total_cut_len / stick_len))
        sticks_with_waste = int(math.ceil(raw_sticks * (1.0 + waste)))
        glue_area_mm2 = total_splices * overlap * stick_w * 1.6
        approx_glue_mass = (glue_area_mm2 / 1_000_000.0) * glue_spread / glue_eff
        mass = (sticks_with_waste * stick_mass + approx_glue_mass) * fast_mass_scale
        return mass, sticks_with_waste

    @staticmethod
    def _reinforcement_profiles() -> Dict[str, Dict[str, int]]:
        return {
            "light": {
                "top_chord": 3,
                "bottom_chord": 3,
                "diagonal": 2,
                "vertical": 2,
                "top_transverse": 1,
                "bottom_transverse": 1,
                "support_pad": 3,
                "chord_lacing": 1,
            },
            "balanced": {
                "top_chord": 4,
                "bottom_chord": 3,
                "diagonal": 2,
                "vertical": 2,
                "top_transverse": 1,
                "bottom_transverse": 1,
                "support_pad": 4,
                "chord_lacing": 1,
            },
            "strong_top": {
                "top_chord": 5,
                "bottom_chord": 3,
                "diagonal": 3,
                "vertical": 2,
                "top_transverse": 2,
                "bottom_transverse": 1,
                "support_pad": 4,
                "chord_lacing": 1,
            },
            "strong": {
                "top_chord": 5,
                "bottom_chord": 4,
                "diagonal": 3,
                "vertical": 3,
                "top_transverse": 2,
                "bottom_transverse": 2,
                "support_pad": 4,
                "chord_lacing": 1,
            },
            "ultra_compression": {
                "top_chord": 7,
                "bottom_chord": 5,
                "diagonal": 4,
                "vertical": 4,
                "top_transverse": 3,
                "bottom_transverse": 3,
                "support_pad": 5,
                "chord_lacing": 2,
            },
            "ultra_pratt": {
                "top_chord": 8,
                "bottom_chord": 5,
                "diagonal": 4,
                "vertical": 3,
                "top_transverse": 3,
                "bottom_transverse": 2,
                "support_pad": 5,
                "chord_lacing": 2,
            },
        }

    def _apply_reinforcement_profile(
        self,
        cfg: Dict,
        profile_name: str,
    ) -> None:
        profile = self._reinforcement_profiles().get(profile_name, self._reinforcement_profiles()["balanced"])
        sticks = cfg.setdefault("member_sticks_by_group", {})
        sticks.update(profile)

        for g in ("top_bracing", "bottom_bracing", "cross_frame_bracing"):
            sticks.setdefault(g, 1)

    def _apply_candidate_geometry(self, cfg: Dict, candidate: Dict) -> Dict:
        v = copy.deepcopy(cfg)
        b = v["bridge"]
        p = v.setdefault("planner", {})

        span = float(candidate["span_mm"])
        width = float(candidate["width_mm"])
        height = float(candidate["center_height_mm"])
        panel = float(candidate["panel_mm"])
        top_profile = str(candidate["top_profile"])
        side_truss = str(candidate["side_truss_type"])
        internal = str(candidate["internal_truss_type"])
        top_chord = str(candidate.get("top_chord_truss_type", candidate.get("chord_truss_type", "X")))
        bottom_chord = str(candidate.get("bottom_chord_truss_type", candidate.get("chord_truss_type", "X")))
        chord_legacy = str(candidate.get("chord_truss_type", "none"))
        tension_joint_model = str(candidate.get("tension_joint_model", "double_lap_reinforced"))
        compression_joint_model = str(candidate.get("compression_joint_model", "double_lap_reinforced"))
        overlap_length_mm = safe_float(candidate.get("overlap_length_mm"), None)
        splice_mode = str(candidate.get("splice_mode", "overlap"))

        b.update(
            {
                "span_mm": span,
                "width_mm": width,
                "center_height_mm": height,
                "panel_mm": panel,
                "top_profile": top_profile,
                "truss_type": side_truss,
                "side_truss_type": side_truss,
                "internal_truss_type": internal,
                "cross_frame_truss_type": internal,
                "top_chord_truss_type": top_chord,
                "bottom_chord_truss_type": bottom_chord,
                "chord_truss_type": chord_legacy,
                "end_height_mm": height if top_profile == "flat" else max(50.0, height / 3.0),
                "plateau_start_mm": span / 3.0,
                "plateau_end_mm": 2.0 * span / 3.0,
                "support_contact_y_mm": [-width / 2.0, width / 2.0],
                "support_contact_x_left_mm": [-float(b.get("left_support_overhang_mm", 100.0)), 0.0],
                "support_contact_x_right_mm": [
                    span,
                    span + float(b.get("right_support_overhang_mm", 100.0)),
                ],
            }
        )

        load_xs = []
        x = b["plateau_start_mm"]
        while x <= b["plateau_end_mm"] + 1e-9:
            load_xs.append(round(x, 6))
            x += panel
        b["load_distribution_x_mm"] = load_xs or [span / 2.0]

        p["target_load_kgf"] = float(p.get("target_load_kgf", b["load_total_kgf"]))
        p["target_breaking_load_kgf"] = float(
            p.get("target_breaking_load_kgf", p.get("target_load_kgf", b["load_total_kgf"]))
        )

        detail = v.setdefault("detail_model", {})
        detail["tension_joint_model"] = tension_joint_model
        detail["compression_joint_model"] = compression_joint_model
        detail["splice_mode"] = splice_mode
        if overlap_length_mm is not None:
            detail["overlap_length_mm"] = float(overlap_length_mm)

        return self.config.normalize(v)

    def _score_candidate(self, cfg: Dict, metrics: Dict) -> float:
        analysis = cfg.get("analysis", {})
        planner = cfg.get("planner", {})
        bridge = cfg.get("bridge", {})
        mat = cfg.get("material", {})

        target_fs = max(0.5, float(analysis.get("target_min_fs", 2.0)))
        target_break = max(
            1.0,
            float(
                planner.get(
                    "target_breaking_load_kgf",
                    planner.get("target_load_kgf", bridge.get("load_total_kgf", 120.0)),
                )
            ),
        )
        max_mass = max(1.0, float(effective_mass_limit_g(cfg)))
        target_mass = max(1.0, float(planner.get("target_bridge_mass_g", max_mass * 0.85)))

        min_fs_primary = max(0.0, safe_float(metrics.get("min_fs_primary"), 0.0) or 0.0)
        mass_g = max(0.0, safe_float(metrics.get("mass_g"), 0.0) or 0.0)
        min_support_fs = safe_float(metrics.get("min_support_fs"), None)
        eq_error = abs(safe_float(metrics.get("equilibrium_error_N"), 0.0) or 0.0)
        load_total_N = abs(safe_float(bridge.get("load_total_N"), 1.0) or 1.0)

        predicted_break = max(
            0.0,
            safe_float(metrics.get("estimated_breaking_load_kgf"), 0.0) or 0.0,
        )
        target_load = max(1.0, float(planner.get("target_load_kgf", bridge.get("load_total_kgf", 120.0))))

        fs_score = min(2.5, min_fs_primary / target_fs)
        break_score = max(0.0, min(2.0, predicted_break / target_break))
        mass_target_score = max(0.0, 1.0 - abs(mass_g - target_mass) / target_mass)
        mass_limit_score = max(0.0, min(1.0, (max_mass - mass_g) / max_mass))

        profile = str(analysis.get("planner_objective_profile", "balanced")).lower()
        presets = {
            "balanced": (0.65, 0.25, 0.07, 0.03),
            "max_strength": (0.70, 0.23, 0.05, 0.02),
            "min_mass": (0.38, 0.22, 0.25, 0.15),
        }
        default_weights = presets.get(profile, presets["balanced"])

        w_fs = safe_float(analysis.get("planner_objective_weight_fs"), default_weights[0]) or default_weights[0]
        w_break = safe_float(analysis.get("planner_objective_weight_break"), default_weights[1]) or default_weights[1]
        w_mass_target = safe_float(analysis.get("planner_objective_weight_mass_target"), default_weights[2]) or default_weights[2]
        w_mass_limit = safe_float(analysis.get("planner_objective_weight_mass_limit"), default_weights[3]) or default_weights[3]

        w_sum = max(1.0e-9, w_fs + w_break + w_mass_target + w_mass_limit)
        w_fs /= w_sum
        w_break /= w_sum
        w_mass_target /= w_sum
        w_mass_limit /= w_sum

        # Massa é restrição dura; não pode vencer nem como fallback de score.
        if mass_g > max_mass:
            return -1.0e9 - 1000.0 * (mass_g - max_mass)

        score = 0.0
        score += 100.0 * (
            w_fs * min(1.2, fs_score)
            + w_break * min(1.2, break_score)
            + w_mass_target * mass_target_score
            + w_mass_limit * mass_limit_score
        )
        score += 55.0 * w_fs * max(0.0, fs_score - 1.0)
        score += 22.0 * w_break * max(0.0, break_score - 1.0)

        if fs_score < 1.0:
            score -= 170.0 * w_fs * (1.0 - fs_score)

        if min_fs_primary < 1.0:
            score -= 110.0 * w_fs * (1.0 - min_fs_primary)

        if metrics.get("solver_status") != "regular":
            score -= 80.0

        if min_support_fs is not None and min_support_fs < 1.0:
            score -= 70.0 * (1.0 - min_support_fs)

        if eq_error > 1e-3 * load_total_N:
            score -= 50.0

        score -= 2.5 * float(metrics.get("inactive_support_count", 0))

        # Penaliza fortemente soluções com ruptura muito abaixo da carga alvo.
        if predicted_break < target_load:
            score -= 90.0 * (1.0 - predicted_break / target_load)
        else:
            score += 8.0

        # Bônus/penalidades de simetria estrutural.
        enforce_symmetry = bool(analysis.get("enforce_symmetry", True))
        side = self._normalize_topology_name(bridge.get("side_truss_type", bridge.get("truss_type", "Parker")))
        top_ch = self._normalize_topology_name(bridge.get("top_chord_truss_type", "X"))
        bot_ch = self._normalize_topology_name(bridge.get("bottom_chord_truss_type", "X"))
        span = max(1.0, float(bridge.get("span_mm", 1200.0)))
        height = max(1.0, float(bridge.get("center_height_mm", 300.0)))
        panel = max(1.0, float(bridge.get("panel_mm", 100.0)))
        n_panels = span / panel
        if enforce_symmetry:
            if abs(n_panels - round(n_panels)) <= 1.0e-6 and int(round(n_panels)) % 2 == 0 and top_ch == bot_ch:
                score += 9.0
            else:
                score -= 120.0
        elif top_ch != bot_ch:
            score -= 8.0

        # Heurística de literatura: com material muito mais forte à tração que à compressão,
        # Pratt tende a ser mais eficiente sob carga vertical estática.
        t_cap = float(mat.get("tension_capacity_per_stick_kgf", 72.0))
        c_cap = max(1.0e-6, float(mat.get("compression_capacity_two_sticks_kgf", 11.0)) / 2.0)
        tc_ratio = t_cap / c_cap
        if tc_ratio >= 6.0:
            if side == "pratt_symmetric":
                score += 8.0
            elif side in {"howe", "howe_inverted"}:
                score -= 8.0
            elif side == "warren_symmetric":
                score -= 2.0

        # Geometria estruturalmente favorável para reduzir compressão/flambagem.
        h_over_l = height / span
        p_over_h = panel / height
        if 0.18 <= h_over_l <= 0.34:
            score += 4.0
        if p_over_h <= 0.55:
            score += 3.0

        # Robustez de emendas críticas e risco de alinhamento.
        detailed = metrics.get("detailed") or {}
        weak_glue = safe_float(metrics.get("weak_glue_joints"), 0.0) or 0.0
        if weak_glue <= 0:
            score += 4.0
        else:
            score -= min(20.0, 0.8 * weak_glue)
        splice_report = detailed.get("splice_stagger_report") or {}
        aligned_critical = int(
            splice_report.get(
                "critical_aligned_count",
                splice_report.get("critical_clusters", 0),
            )
            or 0
        )
        if aligned_critical == 0:
            score += 5.0
        else:
            score -= 10.0 + 2.0 * aligned_critical

        return score

    @staticmethod
    def _group_min_fs(member_checks: List[Dict]) -> Dict[str, float]:
        out: Dict[str, float] = {}

        for row in member_checks:
            group = str(row.get("group", ""))
            fs = safe_float(row.get("FS_min"), None)

            if fs is None:
                continue

            if group not in out or fs < out[group]:
                out[group] = fs

        return out

    def _cfg_cache_key(self, cfg: Dict) -> str:
        c = cfg
        payload = {
            "bridge": c.get("bridge", {}),
            "material": c.get("material", {}),
            "member_sticks_by_group": c.get("member_sticks_by_group", {}),
            "effective_length_factor_by_group": c.get("effective_length_factor_by_group", {}),
            "section_layout_by_group": c.get("section_layout_by_group", {}),
            "analysis": {
                "target_min_fs": c.get("analysis", {}).get("target_min_fs"),
                "primary_groups": c.get("analysis", {}).get("primary_groups"),
                "stabilizer_groups": c.get("analysis", {}).get("stabilizer_groups"),
            },
            "detail_model": c.get("detail_model", {}),
            "support_check": c.get("support_check", {}),
            "planner": {
                "max_bridge_mass_g": c.get("planner", {}).get("max_bridge_mass_g"),
                "target_bridge_mass_g": c.get("planner", {}).get("target_bridge_mass_g"),
                "target_breaking_load_kgf": c.get("planner", {}).get("target_breaking_load_kgf"),
            },
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    def _solve_and_check_base(self, cfg: Dict) -> Dict:
        key = self._cfg_cache_key(cfg)
        with self._cache_lock:
            cached = self._base_eval_cache.get(key)
        if cached is not None:
            with self._cache_lock:
                self._cache_hits += 1
            return cached

        with self._cache_lock:
            self._cache_misses += 1

        nodes, members, supports, loads = self.geometry.generate(cfg)
        sol = self.solver.solve(
            nodes,
            members,
            supports,
            loads,
            unilateral_supports=bool(cfg["bridge"].get("unilateral_supports", True)),
        )

        active_supports = [
            type(s)(
                s.node_id,
                s.UX,
                s.UY,
                s.UZ if s.node_id in sol.active_support_node_ids else 0,
                s.RX,
                s.RY,
                s.RZ,
                s.support_group,
                s.node_id in sol.active_support_node_ids,
            )
            for s in supports
        ]

        member_checks = self.post.check_members(cfg, sol.member_results)
        support_checks = self.post.check_supports(
            cfg,
            nodes,
            active_supports,
            sol.node_results,
        )
        quick_mass_g, quick_sticks = self._quick_mass_estimate(cfg, members)

        payload = {
            "nodes": nodes,
            "members": members,
            "supports": supports,
            "loads": loads,
            "solver": sol,
            "member_checks": member_checks,
            "support_checks": support_checks,
            "quick_mass_g": quick_mass_g,
            "quick_sticks": quick_sticks,
        }
        with self._cache_lock:
            self._base_eval_cache[key] = payload
        return payload

    def _adaptive_step(self, cfg: Dict, metrics: Dict) -> tuple[Dict, bool, List[str]]:
        c = copy.deepcopy(cfg)
        analysis = c.get("analysis", {})
        planner = c.get("planner", {})
        bridge = c.get("bridge", {})
        sticks = c.setdefault("member_sticks_by_group", {})
        actions: List[str] = []
        changed = False

        min_sticks = max(1, int(analysis.get("planner_min_sticks_per_group", 1)))
        max_sticks = max(min_sticks, int(analysis.get("planner_max_sticks_per_group", 12)))
        max_by_group_cfg = analysis.get("planner_max_sticks_per_group_by_group", {}) or {}

        def max_for(group: str) -> int:
            v = safe_float(max_by_group_cfg.get(group), None)
            if v is None:
                return max_sticks
            return max(min_sticks, int(v))
        target_fs = float(analysis.get("target_min_fs", 2.0))
        max_mass = float(effective_mass_limit_g(c))
        current_mass = float(metrics.get("mass_g", 0.0) or 0.0)

        # Refinamento local por membro com espelhamento de parceiros simétricos.
        member_decisions: Dict[int, MemberSizingDecision] = {}
        nodes = metrics.get("nodes") or []
        members = metrics.get("members") or []
        member_results = metrics.get("member_results") or []
        member_checks = metrics.get("member_checks") or []
        if nodes and members and member_results and member_checks:
            try:
                member_decisions = self.build_member_sizing_plan(
                    c,
                    nodes,
                    members,
                    member_results,
                    member_checks,
                )
            except (TypeError, ValueError, KeyError):
                member_decisions = {}

        if member_decisions:
            before = dict((c.get("member_sticks_by_id", {}) or {}))
            c = self.apply_member_sizing_plan(c, member_decisions)
            after = dict((c.get("member_sticks_by_id", {}) or {}))
            if before != after:
                changed = True
            top_rows = sorted(
                member_decisions.values(),
                key=lambda d: (d.action != "reinforce", -d.axial_force_ratio),
            )
            for d in top_rows[:18]:
                if d.action == "keep":
                    continue
                actions.append(
                    f"membro {d.member_id}: {d.action} {d.n_sticks_current}->{d.n_sticks_recommended} "
                    f"(ratio={d.axial_force_ratio:.2f}, FS={d.FS_min:.2f})"
                )

        primary = sorted(
            [
                r for r in metrics.get("member_checks", [])
                if r.get("member_role") == "primary"
            ],
            key=lambda r: safe_float(r.get("FS_min"), 1.0e99) or 1.0e99,
        )

        requested_inc: Dict[str, int] = {}
        requested_fs: Dict[str, float] = {}

        for row in primary[:10]:
            group = str(row.get("group", ""))
            fs = safe_float(row.get("FS_min"), None)
            gov = str(row.get("governing_mode", ""))

            if not group or fs is None:
                continue

            inc = 0
            if fs < 0.25:
                inc = 2
            elif fs < 0.85:
                inc = 1

            if "buckling" in gov and group in {
                "diagonal",
                "vertical",
                "top_chord",
                "bottom_chord",
                "top_transverse",
                "bottom_transverse",
            }:
                inc = max(inc, 1)

            if inc <= 0:
                continue

            prev = requested_inc.get(group, 0)
            requested_inc[group] = max(prev, inc)
            if group not in requested_fs or fs < requested_fs[group]:
                requested_fs[group] = fs

        for group, inc in requested_inc.items():
            old = int(sticks.get(group, min_sticks))
            new = min(max_for(group), old + inc)

            if new != old:
                sticks[group] = new
                changed = True
                fs_txt = requested_fs.get(group, 0.0)
                actions.append(f"{group}: {old}->{new} (FS={fs_txt:.2f})")

        min_support_fs = safe_float(metrics.get("min_support_fs"), None)
        if min_support_fs is not None and min_support_fs < 1.2:
            old = int(sticks.get("support_pad", min_sticks))
            new = min(max_for("support_pad"), old + 1)
            if new != old:
                sticks["support_pad"] = new
                changed = True
                actions.append(f"support_pad: {old}->{new} (FS_apoio={min_support_fs:.2f})")

        if metrics.get("solver_status") != "regular":
            for g in ("top_bracing", "bottom_bracing", "cross_frame_bracing"):
                old = int(sticks.get(g, min_sticks))
                new = min(max_for(g), old + 1)
                if new != old:
                    sticks[g] = new
                    changed = True
                    actions.append(f"{g}: {old}->{new} (solver irregular)")

        min_fs_primary = float(metrics.get("min_fs_primary", 0.0) or 0.0)
        if current_mass > max_mass:
            group_min_fs = self._group_min_fs(metrics.get("member_checks", []))
            for g in ("top_bracing", "bottom_bracing", "cross_frame_bracing", "chord_lacing"):
                old = int(sticks.get(g, min_sticks))
                fs_g = group_min_fs.get(g, 999.0)
                if old > min_sticks and fs_g >= 2.8:
                    sticks[g] = old - 1
                    changed = True
                    actions.append(f"{g}: {old}->{old-1} (alívio de massa)")
            if min_fs_primary < target_fs * 0.85:
                # Troca massa de grupos folgados para grupos críticos (quase neutra em massa).
                group_min_fs = self._group_min_fs(metrics.get("member_checks", []))
                critical_groups = [
                    g for g, fs in sorted(group_min_fs.items(), key=lambda kv: kv[1])
                    if g in {"top_chord", "vertical", "diagonal", "bottom_transverse"}
                ]
                donor_groups = [
                    g for g, fs in sorted(group_min_fs.items(), key=lambda kv: kv[1], reverse=True)
                    if g in {"top_bracing", "bottom_bracing", "cross_frame_bracing", "top_transverse"}
                    and (safe_float(fs, 0.0) or 0.0) >= 2.8
                ]
                if critical_groups and donor_groups:
                    cg = critical_groups[0]
                    dg = donor_groups[0]
                    old_c = int(sticks.get(cg, min_sticks))
                    old_d = int(sticks.get(dg, min_sticks))
                    new_c = min(max_for(cg), old_c + 1)
                    new_d = max(min_sticks, old_d - 1)
                    if new_c != old_c and new_d != old_d:
                        sticks[cg] = new_c
                        sticks[dg] = new_d
                        changed = True
                        actions.append(f"redistribuição: {cg} {old_c}->{new_c} | {dg} {old_d}->{new_d}")
        elif current_mass > 0.92 * max_mass or min_fs_primary > target_fs * 1.45:
            # Enxuga membros muito folgados para gerar variantes mais leves.
            # Se a massa estiver próxima do limite ou a estrutura estiver muito acima da meta de FS,
            # reduzir o número de palitos em grupos com alta folga estrutural.  O limiar de remoção
            # é obtido de allow_recommend_removal_if_fs_gt na configuração de detalhamento (padrão 8.0).
            group_min_fs = self._group_min_fs(metrics.get("member_checks", []))
            remove_if = float(cfg.get("detail_model", {}).get("allow_recommend_removal_if_fs_gt", 8.0))
            groups_to_check = (
                "top_transverse",
                "bottom_transverse",
                "diagonal",
                "vertical",
                "top_bracing",
                "bottom_bracing",
                "cross_frame_bracing",
                "chord_lacing",
                "top_chord",
                "bottom_chord",
            )
            for g in groups_to_check:
                old = int(sticks.get(g, min_sticks))
                fs_g = safe_float(group_min_fs.get(g), None)
                if old > min_sticks and fs_g is not None and fs_g >= remove_if:
                    sticks[g] = old - 1
                    changed = True
                    actions.append(f"{g}: {old}->{old-1} (redução por alta FS={fs_g:.2f})")

        if min_fs_primary < target_fs * 0.7:
            h = float(bridge["center_height_mm"])
            panel = float(bridge["panel_mm"])
            h_max = float(planner.get("height_max_mm", h))
            panel_min = float(planner.get("panel_min_mm", panel))
            panel_max = float(planner.get("panel_max_mm", panel))

            new_h = min(h_max, h * 1.16)
            if new_h > h + 1.0:
                bridge["center_height_mm"] = round(new_h, 6)
                bridge["end_height_mm"] = new_h if bridge.get("top_profile") == "flat" else max(50.0, new_h / 3.0)
                changed = True
                actions.append(f"altura: {h:.0f}->{new_h:.0f} mm")

            new_panel = max(panel_min, panel * 0.86)
            if new_panel < panel - 1.0:
                bridge["panel_mm"] = round(new_panel, 6)
                changed = True
                actions.append(f"painel: {panel:.0f}->{new_panel:.0f} mm")
            elif current_mass > 0.85 * max_mass:
                # Quando massa está no limite, aumentar painel pode reduzir nº de membros e liberar massa.
                up_panel = min(panel_max, panel * 1.14)
                if up_panel > panel + 1.0:
                    bridge["panel_mm"] = round(up_panel, 6)
                    changed = True
                    actions.append(f"painel: {panel:.0f}->{up_panel:.0f} mm (alívio de massa)")

            # Mudança topológica orientada por material com compressão limitada.
            t_cap = float(c.get("material", {}).get("tension_capacity_per_stick_kgf", 72.0))
            c_cap = max(1.0e-6, float(c.get("material", {}).get("compression_capacity_two_sticks_kgf", 11.0)) / 2.0)
            tc_ratio = t_cap / c_cap
            if tc_ratio >= 6.0:
                if str(bridge.get("side_truss_type", "")).lower() != "pratt":
                    bridge["side_truss_type"] = "Pratt"
                    bridge["truss_type"] = "Pratt"
                    changed = True
                    actions.append("topologia lateral: -> Pratt")
                if str(bridge.get("top_profile", "")).lower() in {"flat", "reto", "reta"}:
                    bridge["top_profile"] = "triangular_peak"
                    changed = True
                    actions.append("perfil topo: reto -> triangular")
                if str(bridge.get("top_chord_truss_type", "X")).lower() in {"none", "sem", "nenhuma"}:
                    bridge["top_chord_truss_type"] = "X"
                    changed = True
                    actions.append("banzo superior: none -> X")
                if str(bridge.get("bottom_chord_truss_type", "X")).lower() in {"none", "sem", "nenhuma"}:
                    bridge["bottom_chord_truss_type"] = "X"
                    changed = True
                    actions.append("banzo inferior: none -> X")

        if changed:
            c = self.config.normalize(c)

        return c, changed, actions

    def _evaluate_config(
        self,
        cfg: Dict,
        *,
        include_detail: bool = False,
        detail_dir: Path | None = None,
    ) -> Dict:
        base = self._solve_and_check_base(cfg)
        nodes = base["nodes"]
        members = base["members"]
        sol = base["solver"]
        member_checks = base["member_checks"]
        support_checks = base["support_checks"]

        primary_checks = [
            r for r in member_checks
            if r.get("member_role") == "primary"
        ]
        min_fs_primary = min_numeric((r.get("FS_min") for r in primary_checks), 0.0) or 0.0
        min_fs_all = min_numeric((r.get("FS_min") for r in member_checks), 0.0) or 0.0
        min_support_fs = min_numeric((r.get("FS_support_reaction") for r in support_checks), None)

        quick_mass_g = float(base["quick_mass_g"])
        quick_sticks = int(base["quick_sticks"])

        detailed = None
        mass_g = quick_mass_g
        glue_mass_g = None
        weak_glue = None
        eval_warnings: List[str] = []

        if include_detail:
            dd = detail_dir or Path("outputs/tmp_detail")
            # compute per-member joint plan automatically.  This prevents the
            # user from having to specify global tension/compression joints.
            try:
                plan = self.connection.assign_member_joint_plan(
                    cfg,
                    nodes,
                    members,
                    sol.member_results,
                    member_checks,
                )
            except (TypeError, ValueError, KeyError) as exc:
                plan = {}
                eval_warnings.append(f"Falha ao gerar connection_plan: {exc!r}")
            try:
                sizing_plan_obj = self.build_member_sizing_plan(
                    cfg,
                    nodes,
                    members,
                    sol.member_results,
                    member_checks,
                )
                sizing_rows = [asdict(v) for v in sizing_plan_obj.values()]
            except (TypeError, ValueError, KeyError) as exc:
                sizing_plan_obj = {}
                sizing_rows = []
                eval_warnings.append(f"Falha ao gerar member_sizing_plan: {exc!r}")
            try:
                dd.mkdir(parents=True, exist_ok=True)
                connection_rows = sorted(
                    (dict(v) for v in plan.values()),
                    key=lambda r: int(r.get("member_id", -1)),
                )
                GeometryService.write_csv(dd / "connection_plan.csv", connection_rows)
                (dd / "connection_plan.json").write_text(
                    json.dumps(connection_rows, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                GeometryService.write_csv(dd / "member_sizing_plan.csv", sizing_rows)
                (dd / "member_sizing_plan.json").write_text(
                    json.dumps(sizing_rows, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except (OSError, TypeError, ValueError) as exc:
                eval_warnings.append(f"Falha ao exportar planos de detalhe: {exc!r}")
            # work on a copy of cfg to avoid side effects
            cfg_with_plan = copy.deepcopy(cfg)
            cfg_with_plan["member_joint_plan"] = plan
            cfg_with_plan["member_sizing_plan_by_id"] = {
                str(k): asdict(v) for k, v in sizing_plan_obj.items()
            }
            detailed = self.detail.analyze(
                cfg_with_plan,
                nodes,
                members,
                sol.member_results,
                member_checks,
                dd,
            )
            dsum = detailed.get("summary", {})
            mass_g = safe_float(dsum.get("estimated_total_mass_g"), quick_mass_g) or quick_mass_g
            glue_mass_g = safe_float(dsum.get("estimated_glue_mass_g"), None)
            weak_glue = len(detailed.get("weakest_glue_joints", []) or [])
            detailed["connection_plan"] = list(plan.values())

        load_kgf = float(cfg["bridge"]["load_total_kgf"])
        # Estimate rupture load using multiple limit states.
        rupture = estimate_rupture_load(cfg, member_checks, support_checks, detailed, load_kgf)
        estimated_breaking_load_kgf = rupture.get("predicted_breaking_load_kgf")
        group_min_fs = self._group_min_fs(member_checks)

        critical_members = sorted(
            primary_checks,
            key=lambda r: safe_float(r.get("FS_min"), 1.0e99) or 1.0e99,
        )[:3]

        critical_summary = "; ".join(
            f"M{int(c.get('member_id', -1))}:{c.get('group', '—')}:{safe_float(c.get('FS_min'), 0.0) or 0.0:.2f}:{c.get('governing_mode', '—')}"
            for c in critical_members
        )

        metrics = {
            "n_nodes": len(nodes),
            "n_members": len(members),
            "solver_status": sol.status,
            "equilibrium_error_N": sol.equilibrium_error_N,
            "active_support_count": len(sol.active_support_node_ids),
            "inactive_support_count": len(sol.inactive_support_node_ids),
            "min_fs_primary": min_fs_primary,
            "min_fs_all": min_fs_all,
            "min_support_fs": min_support_fs,
            "mass_g": mass_g,
            "quick_mass_g": quick_mass_g,
            "estimated_sticks": quick_sticks,
            "estimated_breaking_load_kgf": estimated_breaking_load_kgf,
            "critical_members": critical_summary,
            "glue_mass_g": glue_mass_g,
            "weak_glue_joints": weak_glue,
            "member_checks": member_checks,
            "support_checks": support_checks,
            "member_results": sol.member_results,
            "members": members,
            "nodes": nodes,
            "detailed": detailed,
            "rupture_details": rupture,
        }

        try:
            sizing_plan = self.build_member_sizing_plan(
                cfg,
                nodes,
                members,
                sol.member_results,
                member_checks,
            )
        except (TypeError, ValueError, KeyError) as exc:
            sizing_plan = {}
            eval_warnings.append(f"Falha ao gerar sizing summary: {exc!r}")
        sizing_summary = self.summarize_sizing_actions(sizing_plan)
        metrics["member_sizing_plan"] = sizing_summary.get("rows", [])
        metrics["member_sizing_summary"] = {
            k: v
            for k, v in sizing_summary.items()
            if k != "rows"
        }
        if include_detail and detail_dir is not None:
            try:
                detail_dir.mkdir(parents=True, exist_ok=True)
                rows = sizing_summary.get("rows", [])
                GeometryService.write_csv(detail_dir / "member_sizing_plan.csv", rows)
                (detail_dir / "member_sizing_plan.json").write_text(
                    json.dumps(rows, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except (OSError, TypeError, ValueError) as exc:
                eval_warnings.append(f"Falha ao exportar member_sizing_plan final: {exc!r}")

        metrics["score"] = self._score_candidate(cfg, metrics)

        # target safety factor and effective mass limit.  Using the global
        # effective mass limit avoids divergence between planner and material configs.
        target_fs = float(cfg.get("analysis", {}).get("target_min_fs", 2.0))
        max_mass = effective_mass_limit_g(cfg)
        total_load_N = abs(float(cfg["bridge"].get("load_total_N", 0.0)))
        eq_tol_N = max(1.0e-6, 0.005 * max(total_load_N, 1.0))
        equilibrium_ok = abs(float(sol.equilibrium_error_N)) <= eq_tol_N

        # A design is considered feasible only if the solver ran regularly,
        # the minimum safety factor exceeds the target, the detailed mass
        # estimate is within the effective mass limit, the system is in
        # equilibrium, support reactions are acceptable, and predicted rupture
        # is not below the load target.
        target_load_kgf = float(cfg.get("planner", {}).get("target_load_kgf", cfg.get("bridge", {}).get("load_total_kgf", 120.0)))
        predicted_break = safe_float(metrics.get("estimated_breaking_load_kgf"), 0.0) or 0.0
        feasible = (
            metrics["solver_status"] == "regular"
            and metrics["min_fs_primary"] >= target_fs
            and metrics["mass_g"] <= max_mass
            and equilibrium_ok
            and predicted_break >= target_load_kgf
            and (
                metrics["min_support_fs"] is None
                or metrics["min_support_fs"] >= 1.0
            )
        )
        metrics["feasible"] = feasible
        metrics["equilibrium_ok"] = equilibrium_ok
        metrics["equilibrium_tol_N"] = eq_tol_N
        metrics["group_min_fs"] = group_min_fs
        metrics["evaluation_warnings"] = eval_warnings
        # Annotate with mass compliance flag for downstream consumers.
        assert_mass_compliant(metrics, cfg, source="evaluate_config")
        return metrics

    @staticmethod
    def _worker_count(analysis: Dict) -> int:
        requested = int(safe_float(analysis.get("planner_threads"), 0) or 0)
        if requested <= 0:
            return max(1, min(16, (os.cpu_count() or 4)))
        return max(1, min(32, requested))

    @staticmethod
    def _joint_rank(detail: Dict, mode: str, force_type: str) -> int:
        key = "joint_model_rank_tension" if force_type == "tension" else "joint_model_rank_compression"
        rank = list(detail.get(key, []))
        try:
            return rank.index(mode)
        except ValueError:
            return len(rank) + 1

    @staticmethod
    def _estimate_mass_by_group_units(
        cfg: Dict,
        group_piece_units: Dict[str, int],
    ) -> tuple[float, int]:
        mat = cfg.get("material", {})
        detail = cfg.get("detail_model", {})
        sticks = cfg.get("member_sticks_by_group", {})
        waste = float(detail.get("construction_waste_factor", 0.08))
        stick_mass = float(mat.get("stick_mass_g", 1.4))

        raw_sticks = 0
        for g, unit_count in group_piece_units.items():
            raw_sticks += int(unit_count) * int(sticks.get(g, 1))

        total_sticks = int(math.ceil(raw_sticks * (1.0 + waste)))
        approx_mass = total_sticks * stick_mass
        return approx_mass, total_sticks

    def _material_bias(self, cfg: Dict) -> Dict[str, float]:
        mat = cfg.get("material", {})
        detail = cfg.get("detail_model", {})
        t_cap = float(mat.get("tension_capacity_per_stick_kgf", 72.0))
        c_cap = max(1.0e-6, float(mat.get("compression_capacity_two_sticks_kgf", 11.0)) / 2.0)
        tc_ratio = t_cap / c_cap
        overlap = float(detail.get("overlap_length_mm", 30.0))
        stick_len = max(1.0, float(mat.get("stick_length_mm", 115.0)))
        overlap_ratio = overlap / stick_len
        return {
            "tc_ratio": tc_ratio,
            "overlap_ratio": overlap_ratio,
        }

    @staticmethod
    def _normalize_topology_name(name: Any) -> str:
        raw = str(name or "").strip().lower()
        alias = {
            "parker": "pratt_symmetric",
            "baltimore": "pratt_symmetric",
            "pratt": "pratt_symmetric",
            "pratt_symmetric": "pratt_symmetric",
            "pratt simétrica": "pratt_symmetric",
            "warren": "warren_symmetric",
            "warren_symmetric": "warren_symmetric",
            "warren simétrica": "warren_symmetric",
            "warren_mid_braced": "warren_mid_braced",
            "warren intermediária": "warren_mid_braced",
            "warren intermedia": "warren_mid_braced",
            "howe": "howe",
            "howe_inverted": "howe_inverted",
            "howe invertida": "howe_inverted",
            "k": "k_symmetric",
            "k_symmetric": "k_symmetric",
            "k simétrica": "k_symmetric",
            "x": "x",
            "n": "pratt_symmetric",
            "none": "none",
            "sem": "none",
            "nenhuma": "none",
        }
        return alias.get(raw, raw)

    def _is_symmetry_compliant_candidate(self, cfg: Dict, candidate: Dict) -> tuple[bool, str]:
        analysis = cfg.get("analysis", {})
        if not bool(analysis.get("enforce_symmetry", True)):
            return True, "OK"

        span = float(candidate["span_mm"])
        panel = max(1.0, float(candidate["panel_mm"]))
        n_panels = span / panel
        n_rounded = int(round(n_panels))
        if abs(n_panels - n_rounded) > 1.0e-6:
            return False, "SYM_panelizacao_nao_inteira"
        if n_rounded % 2 != 0:
            return False, "SYM_numero_paineis_impar"

        mid = span * 0.5
        station_idx = round(mid / panel)
        if abs(station_idx * panel - mid) > 1.0e-6:
            return False, "SYM_meio_vao_sem_estacao"

        top_profile = str(candidate.get("top_profile", "")).lower()
        if top_profile in {"half_parker", "half_arch", "inclinado"}:
            return False, "SYM_top_profile_assimetrico"

        side = self._normalize_topology_name(candidate.get("side_truss_type"))
        if side in {"none"}:
            return False, "SYM_side_truss_invalida"

        top_ch = self._normalize_topology_name(candidate.get("top_chord_truss_type"))
        bot_ch = self._normalize_topology_name(candidate.get("bottom_chord_truss_type"))
        if "none" in {top_ch, bot_ch}:
            return False, "SYM_banzo_sem_travamento_principal"
        if top_ch != bot_ch:
            return False, "SYM_banzos_incompativeis"

        return True, "OK"

    def map_member_to_symmetry_partners(
        self,
        cfg: Dict,
        nodes: List[Node],
        members: List[Member],
    ) -> Dict[int, List[int]]:
        span = float(cfg.get("bridge", {}).get("span_mm", 0.0))
        x_mid = 0.5 * span
        node_by_id = {n.id: n for n in nodes}

        def key_from_points(
            p1: tuple[float, float, float],
            p2: tuple[float, float, float],
            group: str,
        ) -> tuple:
            a = (round(p1[0], 6), round(p1[1], 6), round(p1[2], 6))
            b = (round(p2[0], 6), round(p2[1], 6), round(p2[2], 6))
            e0, e1 = (a, b) if a <= b else (b, a)
            return (e0, e1, str(group))

        key_to_id: Dict[tuple, int] = {}
        for m in members:
            ni = node_by_id[m.i]
            nj = node_by_id[m.j]
            key = key_from_points((ni.x, ni.y, ni.z), (nj.x, nj.y, nj.z), m.group)
            key_to_id[key] = int(m.id)

        partners: Dict[int, List[int]] = {}
        ops = [(False, False), (True, False), (False, True), (True, True)]
        for m in members:
            ni = node_by_id[m.i]
            nj = node_by_id[m.j]
            ids = set([int(m.id)])
            for mx, my in ops:
                p1x = (2.0 * x_mid - float(ni.x)) if mx else float(ni.x)
                p2x = (2.0 * x_mid - float(nj.x)) if mx else float(nj.x)
                p1y = (-float(ni.y)) if my else float(ni.y)
                p2y = (-float(nj.y)) if my else float(nj.y)
                key = key_from_points((p1x, p1y, float(ni.z)), (p2x, p2y, float(nj.z)), m.group)
                mid = key_to_id.get(key)
                if mid is not None:
                    ids.add(int(mid))
            partners[int(m.id)] = sorted(i for i in ids if i != int(m.id))
        return partners

    def build_member_sizing_plan(
        self,
        cfg: Dict,
        nodes: List[Node],
        members: List[Member],
        member_results: List[Dict],
        member_checks: List[Dict],
    ) -> Dict[int, MemberSizingDecision]:
        analysis = cfg.get("analysis", {})
        target_fs = float(analysis.get("target_min_fs", 2.0))
        min_sticks = max(1, int(analysis.get("planner_min_sticks_per_group", 1)))
        max_sticks = max(min_sticks, int(analysis.get("planner_max_sticks_per_group", 12)))
        max_by_group = analysis.get("planner_max_sticks_per_group_by_group", {}) or {}

        def max_for(group: str) -> int:
            v = safe_float(max_by_group.get(group), None)
            if v is None:
                return max_sticks
            return max(min_sticks, int(v))

        res_map = {int(r.get("member_id")): r for r in (member_results or []) if r.get("member_id") is not None}
        chk_map = {int(r.get("member_id")): r for r in (member_checks or []) if r.get("member_id") is not None}
        max_abs_n = max((abs(safe_float(r.get("N_N"), 0.0) or 0.0) for r in res_map.values()), default=0.0)
        max_abs_n = max(1.0e-9, max_abs_n)

        primary_groups = set(
            analysis.get(
                "primary_groups",
                [
                    "top_chord",
                    "bottom_chord",
                    "diagonal",
                    "vertical",
                    "top_transverse",
                    "bottom_transverse",
                    "support_pad",
                ],
            )
        )
        partners = self.map_member_to_symmetry_partners(cfg, nodes, members)

        decisions: Dict[int, MemberSizingDecision] = {}
        for m in members:
            mid = int(m.id)
            chk = chk_map.get(mid, {})
            res = res_map.get(mid, {})
            n_cur = max(1, int(getattr(m, "n_sticks", 1)))
            n_force = abs(safe_float(res.get("N_N"), 0.0) or 0.0)
            ratio = n_force / max_abs_n
            fs_min = safe_float(chk.get("FS_min"), target_fs) or target_fs
            group = str(getattr(m, "group", ""))
            role = "primary" if group in primary_groups else "secondary"
            mode = str(chk.get("governing_mode", "") or "")

            action = "keep"
            n_new = n_cur
            reason = "stable_load_band"
            if ratio >= 0.40 or fs_min < target_fs:
                action = "reinforce"
                inc = 2 if (fs_min < max(1.0, 0.8 * target_fs) or ratio >= 0.7) else 1
                n_new = min(max_for(group), n_cur + inc)
                reason = "high_axial_ratio_or_low_fs"
            elif ratio < 0.15 and fs_min >= target_fs * 1.6:
                action = "lighten"
                dec = 1 if role == "primary" else 2
                n_new = max(1, n_cur - dec)
                reason = "very_low_ratio_high_fs"

            if role == "primary" and fs_min < target_fs and n_new == n_cur:
                n_new = min(max_for(group), n_cur + 1)
                action = "reinforce"
                reason = "primary_member_guard"

            sym_ids = partners.get(mid, [])
            apply_ids = [mid] + sym_ids if bool(analysis.get("enforce_symmetry", True)) else [mid]
            decisions[mid] = MemberSizingDecision(
                member_id=mid,
                original_group=group,
                effective_group=group,
                n_sticks_current=n_cur,
                n_sticks_recommended=n_new,
                axial_force_N=n_force,
                axial_force_ratio=ratio,
                FS_min=fs_min,
                governing_mode=mode,
                action=action,
                reason=reason,
                symmetry_partner_ids=sym_ids,
                applied_to_member_ids=apply_ids,
            )
        return decisions

    def apply_member_sizing_plan(self, cfg: Dict, plan: Dict[int, MemberSizingDecision]) -> Dict:
        out = copy.deepcopy(cfg)
        by_id = out.setdefault("member_sticks_by_id", {})
        by_group = out.setdefault("member_sticks_by_group", {})
        for d in plan.values():
            target = max(1, int(d.n_sticks_recommended))
            for mid in d.applied_to_member_ids:
                by_id[str(int(mid))] = int(target)
            cur_group = int(by_group.get(d.effective_group, d.n_sticks_current))
            if d.action == "reinforce":
                by_group[d.effective_group] = max(cur_group, target)
            elif d.action == "lighten":
                by_group[d.effective_group] = max(1, min(cur_group, target))
        return self.config.normalize(out)

    @staticmethod
    def summarize_sizing_actions(plan: Dict[int, MemberSizingDecision]) -> Dict[str, Any]:
        rows = [asdict(v) for v in plan.values()]
        return {
            "total_members": len(rows),
            "reinforce": sum(1 for r in rows if r.get("action") == "reinforce"),
            "lighten": sum(1 for r in rows if r.get("action") == "lighten"),
            "keep": sum(1 for r in rows if r.get("action") == "keep"),
            "rows": rows,
        }

    def _build_stage1_candidates(self, cfg: Dict, n_target: int) -> List[Dict]:
        planner = cfg.get("planner", {})
        analysis = cfg.get("analysis", {})
        enforce_symmetry = bool(analysis.get("enforce_symmetry", True))
        rng = random.Random(42)

        span_vals = self._linspace_values(
            float(planner.get("span_min_mm", cfg["bridge"]["span_mm"])),
            float(planner.get("span_max_mm", cfg["bridge"]["span_mm"])),
            4,
            round_to=10.0,
        )
        width_vals = self._linspace_values(
            float(planner.get("width_min_mm", cfg["bridge"]["width_mm"])),
            float(planner.get("width_max_mm", cfg["bridge"]["width_mm"])),
            3,
            round_to=5.0,
        )
        height_vals = self._linspace_values(
            float(planner.get("height_min_mm", cfg["bridge"]["center_height_mm"])),
            float(planner.get("height_max_mm", cfg["bridge"]["center_height_mm"])),
            4,
            round_to=5.0,
        )
        panel_vals = self._linspace_values(
            float(planner.get("panel_min_mm", cfg["bridge"]["panel_mm"])),
            float(planner.get("panel_max_mm", cfg["bridge"]["panel_mm"])),
            4,
            round_to=5.0,
        )

        side_vals = list(
            planner.get(
                "consider_side_trusses",
                [
                    "Pratt_symmetric",
                    "Warren_symmetric",
                    "Warren_mid_braced",
                    "Howe_inverted",
                    "K_symmetric",
                    "X",
                ],
            )
        )
        top_vals = list(planner.get("consider_top_profiles", ["parker_plateau", "triangular_peak", "shallow_arch", "flat"]))
        internal_vals = list(
            planner.get(
                "consider_internal_trusses",
                ["X", "Warren_symmetric", "Pratt_symmetric", "Howe_inverted", "K_symmetric", "Warren_mid_braced"],
            )
        )
        top_chord_vals = list(
            planner.get(
                "consider_top_chord_trusses",
                ["X", "Warren_symmetric", "Pratt_symmetric", "Howe_inverted", "K_symmetric", "Warren_mid_braced"],
            )
        )
        bottom_chord_vals = list(
            planner.get(
                "consider_bottom_chord_trusses",
                ["X", "Warren_symmetric", "Pratt_symmetric", "Howe_inverted", "K_symmetric", "Warren_mid_braced"],
            )
        )
        if enforce_symmetry:
            side_vals = [
                v
                for v in side_vals
                if self._normalize_topology_name(v)
                in {"pratt_symmetric", "warren_symmetric", "warren_mid_braced", "howe_inverted", "k_symmetric", "x", "howe"}
            ] or ["Pratt_symmetric", "Warren_symmetric", "Warren_mid_braced"]
            internal_vals = [v for v in internal_vals if self._normalize_topology_name(v) != "none"] or ["X"]
            top_chord_vals = [v for v in top_chord_vals if self._normalize_topology_name(v) != "none"] or ["X"]
            bottom_chord_vals = [v for v in bottom_chord_vals if self._normalize_topology_name(v) != "none"] or ["X"]
        reinforce_vals = list(self._reinforcement_profiles().keys())
        detail = cfg.get("detail_model", {})
        tension_models = list(detail.get("joint_model_rank_tension", [])) or ["double_lap_reinforced", "double_lap", "single_lap_tala", "single_lap", "butt_plain"]
        compression_models = list(detail.get("joint_model_rank_compression", [])) or ["double_lap_reinforced", "double_lap", "single_lap_tala", "single_lap", "butt_plain"]
        overlap_base = float(detail.get("overlap_length_mm", 30.0))
        overlap_vals = sorted({
            max(10.0, overlap_base * 0.8),
            overlap_base,
            overlap_base * 1.2,
            overlap_base * 1.4,
        })
        splice_modes = ["overlap", "butt_with_splints"]

        # Preferências estruturais baseadas na relação tração/compressão do material.
        t_cap = float(cfg.get("material", {}).get("tension_capacity_per_stick_kgf", 72.0))
        c_cap = max(1.0e-6, float(cfg.get("material", {}).get("compression_capacity_two_sticks_kgf", 11.0)) / 2.0)
        tc_ratio = t_cap / c_cap

        def weighted_choice(options: List[str], weights_map: Dict[str, float], default_w: float = 1.0) -> str:
            ws = [max(0.01, float(weights_map.get(o, default_w))) for o in options]
            return rng.choices(options, weights=ws, k=1)[0]

        side_weights = {o: 1.0 for o in side_vals}
        if tc_ratio >= 6.0:
            side_weights.update({"Pratt": 3.2, "Parker": 2.2, "Baltimore": 2.2, "Warren": 1.2, "K": 1.4, "Howe": 0.55})

        top_weights = {o: 1.0 for o in top_vals}
        top_weights.update({"triangular_peak": 1.8, "parker_plateau": 1.6, "shallow_arch": 1.3, "flat": 0.7})

        internal_weights = {o: 1.0 for o in internal_vals}
        internal_weights.update({"X": 2.5, "Warren": 1.45, "Pratt": 1.25, "K": 1.25, "N": 1.1, "Howe": 0.9, "none": 0.35})

        chord_weights = {o: 1.0 for o in top_chord_vals}
        chord_weights.update({"X": 2.2, "Warren": 1.6, "Pratt": 1.25, "K": 1.25, "N": 1.1, "Howe": 0.9, "none": 0.4})

        reinforce_weights = {o: 1.0 for o in reinforce_vals}
        reinforce_weights.update({
            "balanced": 1.8,
            "strong_top": 2.1,
            "strong": 1.7,
            "ultra_compression": 1.5,
            "ultra_pratt": 2.2 if tc_ratio >= 6.0 else 1.2,
            "light": 0.4,
        })
        tension_weights = {m: max(0.2, 1.0 + 0.08 * (len(tension_models) - i)) for i, m in enumerate(tension_models)}
        compression_weights = {m: max(0.2, 1.0 + 0.12 * (len(compression_models) - i)) for i, m in enumerate(compression_models)}

        candidates: List[Dict] = []
        seen = set()

        def add_candidate(c: Dict) -> None:
            key = (
                c["span_mm"],
                c["width_mm"],
                c["center_height_mm"],
                c["panel_mm"],
                c["side_truss_type"],
                c["top_profile"],
                c["internal_truss_type"],
                c["top_chord_truss_type"],
                c["bottom_chord_truss_type"],
                c["reinforcement_profile"],
                c["tension_joint_model"],
                c["compression_joint_model"],
                c["splice_mode"],
                round(float(c["overlap_length_mm"]), 3),
            )

            if key in seen:
                return

            span = float(c["span_mm"])
            panel = float(c["panel_mm"])
            n_panels = span / max(panel, 1.0)

            if n_panels < 6 or n_panels > 24:
                return
            if enforce_symmetry:
                ok_sym, _ = self._is_symmetry_compliant_candidate(cfg, c)
                if not ok_sym:
                    return

            seen.add(key)
            candidates.append(c)

        # Candidatos âncora para garantir cobertura de extremos.
        for side in side_vals:
            for top in top_vals[:2]:
                for span in (span_vals[0], span_vals[-1]):
                    for height in (height_vals[0], height_vals[-1]):
                        add_candidate(
                            {
                                "span_mm": span,
                                "width_mm": width_vals[len(width_vals) // 2],
                                "center_height_mm": height,
                                "panel_mm": panel_vals[len(panel_vals) // 2],
                                "side_truss_type": side,
                                "top_profile": top,
                                "internal_truss_type": internal_vals[0],
                                "top_chord_truss_type": top_chord_vals[0],
                                "bottom_chord_truss_type": top_chord_vals[0] if enforce_symmetry else bottom_chord_vals[0],
                                "chord_truss_type": "none",
                                "reinforcement_profile": "balanced",
                                "tension_joint_model": tension_models[0],
                                "compression_joint_model": compression_models[0],
                                "splice_mode": splice_modes[0],
                                "overlap_length_mm": overlap_vals[min(1, len(overlap_vals) - 1)],
                            }
                        )

        attempts = max(200, n_target * 30)

        for _ in range(attempts):
            if len(candidates) >= n_target:
                break

            span_chosen = rng.choice(span_vals)
            panel_chosen = rng.choice(panel_vals)
            if enforce_symmetry:
                valid_panels = [
                    p
                    for p in panel_vals
                    if abs((span_chosen / max(1.0, p)) - round(span_chosen / max(1.0, p))) <= 1.0e-6
                    and int(round(span_chosen / max(1.0, p))) % 2 == 0
                ]
                if valid_panels:
                    panel_chosen = rng.choice(valid_panels)
                elif abs((span_chosen / max(1.0, panel_chosen)) - round(span_chosen / max(1.0, panel_chosen))) > 1.0e-6:
                    panel_chosen = span_chosen / max(6.0, 2.0 * round(span_chosen / (2.0 * panel_chosen)))

            top_ch = weighted_choice(top_chord_vals, chord_weights)
            bot_ch = weighted_choice(bottom_chord_vals, chord_weights)
            if enforce_symmetry:
                bot_ch = top_ch

            add_candidate(
                {
                    "span_mm": span_chosen,
                    "width_mm": rng.choice(width_vals),
                    "center_height_mm": rng.choice(height_vals),
                    "panel_mm": panel_chosen,
                    "side_truss_type": weighted_choice(side_vals, side_weights),
                    "top_profile": weighted_choice(top_vals, top_weights),
                    "internal_truss_type": weighted_choice(internal_vals, internal_weights),
                    "top_chord_truss_type": top_ch,
                    "bottom_chord_truss_type": bot_ch,
                    "chord_truss_type": "none",
                    "reinforcement_profile": weighted_choice(reinforce_vals, reinforce_weights),
                    "tension_joint_model": weighted_choice(tension_models, tension_weights),
                    "compression_joint_model": weighted_choice(compression_models, compression_weights),
                    "splice_mode": weighted_choice(splice_modes, {"overlap": 1.0, "butt_with_splints": 1.4}),
                    "overlap_length_mm": rng.choice(overlap_vals),
                }
            )

        return candidates[:n_target]

    @staticmethod
    def _stage_row(stage: str, idx: int, candidate: Dict, metrics: Dict, cfg: Dict) -> Dict:
        group_min_fs = metrics.get("group_min_fs") or {}
        critical_groups = ",".join(
            g
            for g, _ in sorted(group_min_fs.items(), key=lambda kv: kv[1])[:4]
        )
        row = {
            "stage": stage,
            "candidate_id": f"{stage.upper()}-{idx:04d}",
            "side_truss_type": candidate.get("side_truss_type", cfg["bridge"].get("side_truss_type")),
            "top_profile": candidate.get("top_profile", cfg["bridge"].get("top_profile")),
            "internal_truss_type": candidate.get("internal_truss_type", cfg["bridge"].get("internal_truss_type")),
            "top_chord_truss_type": candidate.get("top_chord_truss_type", cfg["bridge"].get("top_chord_truss_type", "X")),
            "bottom_chord_truss_type": candidate.get("bottom_chord_truss_type", cfg["bridge"].get("bottom_chord_truss_type", "X")),
            "chord_truss_type": candidate.get("chord_truss_type", cfg["bridge"].get("chord_truss_type", "none")),
            "tension_joint_model": cfg.get("detail_model", {}).get("tension_joint_model"),
            "compression_joint_model": cfg.get("detail_model", {}).get("compression_joint_model"),
            "splice_mode": cfg.get("detail_model", {}).get("splice_mode"),
            "overlap_length_mm": cfg.get("detail_model", {}).get("overlap_length_mm"),
            "reinforcement_profile": candidate.get("reinforcement_profile", "custom"),
            "span_mm": float(cfg["bridge"]["span_mm"]),
            "width_mm": float(cfg["bridge"]["width_mm"]),
            "center_height_mm": float(cfg["bridge"]["center_height_mm"]),
            "panel_mm": float(cfg["bridge"]["panel_mm"]),
            "score": metrics.get("score"),
            "feasible": metrics.get("feasible"),
            "solver_status": metrics.get("solver_status"),
            "equilibrium_error_N": metrics.get("equilibrium_error_N"),
            "min_fs_primary": metrics.get("min_fs_primary"),
            "min_fs_all": metrics.get("min_fs_all"),
            "min_support_fs": metrics.get("min_support_fs"),
            "mass_g": metrics.get("mass_g"),
            "quick_mass_g": metrics.get("quick_mass_g"),
            "estimated_sticks": metrics.get("estimated_sticks"),
            "predicted_breaking_load_kgf": metrics.get("estimated_breaking_load_kgf"),
            "inactive_support_count": metrics.get("inactive_support_count"),
            "critical_members": metrics.get("critical_members"),
            "critical_groups": critical_groups,
            "glue_mass_g": metrics.get("glue_mass_g"),
            "weak_glue_joints": metrics.get("weak_glue_joints"),
            "config": cfg,
        }
        return row

    @staticmethod
    def _sort_rows(rows: List[Dict]) -> List[Dict]:
        return sorted(
            rows,
            key=lambda r: safe_float(r.get("score"), -1.0e99) or -1.0e99,
            reverse=True,
        )

    @staticmethod
    def _mutate_sticks(
        base_cfg: Dict,
        *,
        group_min_fs: Dict[str, float] | None = None,
        min_support_fs: float | None = None,
    ) -> List[Dict]:
        groups = [
            "top_chord",
            "bottom_chord",
            "diagonal",
            "vertical",
            "top_transverse",
            "bottom_transverse",
            "support_pad",
            "top_bracing",
            "bottom_bracing",
            "cross_frame_bracing",
        ]

        group_min_fs = group_min_fs or {}
        analysis = base_cfg.get("analysis", {})
        min_default = max(1, int(analysis.get("planner_min_sticks_per_group", 1)))
        max_default = max(min_default, int(analysis.get("planner_max_sticks_per_group", 12)))
        max_by_group_cfg = analysis.get("planner_max_sticks_per_group_by_group", {}) or {}

        def lim_min(group: str) -> int:
            return min_default

        def lim_max(group: str) -> int:
            vg = safe_float(max_by_group_cfg.get(group), None)
            if vg is None:
                return max_default
            return max(min_default, int(vg))

        critical_order = [
            g
            for g, _ in sorted(group_min_fs.items(), key=lambda kv: kv[1])
            if g in groups
        ]
        critical_order = critical_order[:5]

        patterns = [
            {},
            {"top_chord": 1},
            {"bottom_chord": 1},
            {"diagonal": 1},
            {"vertical": 1},
            {"top_transverse": 1, "bottom_transverse": 1},
        ]

        # Reforça grupos realmente críticos no seed.
        for g in critical_order[:3]:
            patterns.append({g: 1})
            fs_g = safe_float(group_min_fs.get(g), 9.9) or 9.9
            if fs_g < 0.8:
                patterns.append({g: 2})

        if len(critical_order) >= 2:
            patterns.append({critical_order[0]: 1, critical_order[1]: 1})
        if len(critical_order) >= 3:
            patterns.append({critical_order[0]: 1, critical_order[2]: 1})
        if critical_order and critical_order[0] in {"top_chord", "vertical", "diagonal", "bottom_transverse"}:
            patterns.append({critical_order[0]: 2})

        # Caso apoio esteja fraco, força incremento no suporte.
        if min_support_fs is not None and min_support_fs < 1.2:
            patterns.append({"support_pad": 1})
            patterns.append({"support_pad": 2})

        # Alívio leve em grupos com sobra de FS.
        high_margin_groups = [
            g for g, fs in group_min_fs.items()
            if g in {"top_bracing", "bottom_bracing", "cross_frame_bracing", "top_transverse", "bottom_transverse"}
            and (safe_float(fs, 0.0) or 0.0) >= 3.0
        ]
        for g in high_margin_groups[:2]:
            patterns.append({g: -1})
            if critical_order:
                patterns.append({critical_order[0]: 1, g: -1})
        if len(critical_order) >= 2:
            for g in high_margin_groups[:2]:
                patterns.append({critical_order[0]: 1, critical_order[1]: 1, g: -1})

        # Dedup de padrões.
        unique_patterns = []
        seen = set()
        for p in patterns:
            key = tuple(sorted((k, int(v)) for k, v in p.items()))
            if key in seen:
                continue
            seen.add(key)
            unique_patterns.append(p)

        variants = []

        for p in unique_patterns:
            cfg = copy.deepcopy(base_cfg)
            sticks = cfg.setdefault("member_sticks_by_group", {})

            for g in groups:
                base_n = int(sticks.get(g, lim_min(g)))
                delta = int(p.get(g, 0))
                sticks[g] = max(lim_min(g), min(lim_max(g), base_n + delta))

            # Variações de montagem com impacto em resistência/massa.
            detail = cfg.setdefault("detail_model", {})
            if p.get("top_chord", 0) >= 1 or p.get("vertical", 0) >= 1:
                detail["compression_joint_model"] = "double_lap_reinforced"
                detail["tension_joint_model"] = "double_lap"
            if p.get("diagonal", 0) >= 1:
                detail["tension_joint_model"] = "double_lap_reinforced"

            variants.append(cfg)

        return variants

    @staticmethod
    def _for_csv(rows: List[Dict]) -> List[Dict]:
        clean = []
        for r in rows:
            c = {k: v for k, v in r.items() if k != "config"}
            clean.append(c)
        return clean

    def _prefilter_candidate(self, cfg: Dict, candidate: Dict) -> tuple[bool, str]:
        """
        Filtros rápidos anteriores ao solver para eliminar propostas inviáveis.
        Estratégias:
        - PF1: razão altura/vão.
        - PF2: razão largura/vão.
        - PF3: razão painel/altura.
        - PF4: estimativa de massa preliminar.
        - PF5: regras eliminatórias de apoio e vão do edital.
        - PF6: viabilidade mecânica mínima aproximada (compressão de banzo).
        """
        planner = cfg.get("planner", {})
        material = cfg.get("material", {})
        bridge = cfg.get("bridge", {})

        span = float(candidate["span_mm"])
        width = float(candidate["width_mm"])
        height = float(candidate["center_height_mm"])
        panel = max(1.0, float(candidate["panel_mm"]))
        n_panels = span / panel

        ok_sym, sym_reason = self._is_symmetry_compliant_candidate(cfg, candidate)
        if not ok_sym:
            return False, sym_reason

        if n_panels < 6.0 or n_panels > 24.0:
            return False, "PF0_n_paineis_fora_faixa"

        hs = height / max(1.0, span)
        if hs < 0.04 or hs > 0.60:
            return False, "PF1_relacao_altura_vao"

        ws = width / max(1.0, span)
        if ws < 0.06 or ws > 0.30:
            return False, "PF2_relacao_largura_vao"

        ph = panel / max(1.0, height)
        if ph > 2.40:
            return False, "PF3_relacao_painel_altura"

        # Filtra combinações sabidamente frágeis à flambagem para materiais com baixa compressão.
        t_cap = float(material.get("tension_capacity_per_stick_kgf", 72.0))
        c_cap = max(1.0e-6, float(material.get("compression_capacity_two_sticks_kgf", 11.0)) / 2.0)
        tc_ratio = t_cap / c_cap
        side = self._normalize_topology_name(candidate.get("side_truss_type"))
        top = str(candidate.get("top_profile", "")).lower()

        if tc_ratio >= 6.0:
            if side in {"howe", "howe_inverted"}:
                return False, "PF5_topologia_howe_desfavoravel_material"
            if hs < 0.16:
                return False, "PF6_altura_relativa_baixa_para_compressao"
            if top == "flat" and hs < 0.22:
                return False, "PF7_topo_reto_insuficiente"

        # Estimativa preliminar de massa (heurística leve).
        profile = str(candidate.get("reinforcement_profile", "balanced"))
        profile_factor = {
            "light": 0.90,
            "balanced": 1.00,
            "strong_top": 1.14,
            "strong": 1.24,
            "ultra_compression": 1.40,
            "ultra_pratt": 1.48,
        }.get(profile, 1.00)

        base_sticks = max(1.0, 12.0 * n_panels + 2.8 * n_panels * max(0.4, hs * 2.0))
        width_factor = max(0.75, min(1.35, width / 150.0))
        approx_sticks = base_sticks * width_factor * profile_factor
        stick_mass = float(material.get("stick_mass_g", 1.4))
        detail = cfg.get("detail_model", {})
        waste = float(detail.get("construction_waste_factor", 0.08))
        fast_mass_scale = max(0.75, min(1.05, float(detail.get("fast_mass_scale", 0.90))))
        # Aproximação rápida alinhada ao detalhamento: sem reserva fixa de cola.
        approx_mass = approx_sticks * stick_mass * (1.0 + waste) * fast_mass_scale
        mass_limit = float(effective_mass_limit_g(cfg))
        if approx_mass > mass_limit * 1.35:
            return False, "PF8_massa_preliminar_excessiva"

        # Filtro mecânico rápido para compressão no banzo superior:
        # F_chord ~ M/h, com M~(P/2)*L/4 por treliça lateral.
        load_total_kgf = float(planner.get("target_load_kgf", bridge.get("load_total_kgf", 120.0)))
        side_share_kgf = 0.5 * load_total_kgf
        mmax_kgf_mm = side_share_kgf * span / 4.0
        chord_force_kgf = mmax_kgf_mm / max(40.0, height)
        profile = str(candidate.get("reinforcement_profile", "balanced"))
        top_guess = int(self._reinforcement_profiles().get(profile, {}).get("top_chord", 4))
        comp_per_stick = max(0.1, float(material.get("compression_capacity_two_sticks_kgf", 11.0)) / 2.0)
        joint_rank = list(cfg.get("detail_model", {}).get("joint_model_rank_compression", []))
        comp_joint = str(candidate.get("compression_joint_model", cfg.get("detail_model", {}).get("compression_joint_model", "double_lap_reinforced")))
        if joint_rank:
            rank_pos = joint_rank.index(comp_joint) if comp_joint in joint_rank else len(joint_rank)
            joint_factor = max(0.6, 1.08 - 0.05 * rank_pos)
        else:
            joint_factor = 1.0
        approx_chord_capacity_kgf = top_guess * comp_per_stick * joint_factor
        if chord_force_kgf > 1.95 * approx_chord_capacity_kgf:
            return False, "PF9_capacidade_topo_aprox_insuficiente"

        # Se o material é muito mais forte em tração, evitar painéis excessivamente densos
        # (massa alta) sem ganho proporcional para compressão.
        if tc_ratio >= 6.0 and n_panels >= 14.0 and hs <= 0.20:
            return False, "PF10_densidade_paineis_alta_para_compressao_limitada"

        # Regras eliminatórias base do edital para envelope de apoio/vão.
        left_overhang = abs(float(bridge.get("left_support_overhang_mm", 100.0)))
        right_overhang = abs(float(bridge.get("right_support_overhang_mm", 100.0)))
        if left_overhang > 100.0 or right_overhang > 100.0:
            return False, "PF11_apoio_excede_100mm"

        return True, "OK"

    @staticmethod
    def _floor_to_step(value: float, step: float) -> float:
        step = max(1.0e-9, float(step))
        return math.floor(float(value) / step) * step

    @staticmethod
    def _ceil_to_step(value: float, step: float) -> float:
        step = max(1.0e-9, float(step))
        return math.ceil(float(value) / step) * step

    @classmethod
    def _round_abs_to_step(cls, value: float, step: float, mode: str) -> float:
        sign = -1.0 if float(value) < 0 else 1.0
        mag = abs(float(value))

        if mode == "min":
            m = cls._floor_to_step(mag, step)
        else:
            m = cls._ceil_to_step(mag, step)

        return sign * m

    def _build_conservative_cfg(self, cfg: Dict, mode: str) -> Dict:
        v = copy.deepcopy(cfg)
        a = v.get("analysis", {})
        b = v.get("bridge", {})
        m = v.get("material", {})
        d = v.get("detail_model", {})

        length_step = max(0.5, float(a.get("final_round_step_length_mm", 5.0)))
        section_step = max(0.01, float(a.get("final_round_step_section_mm", 0.1)))
        mass_step = max(0.01, float(a.get("final_round_step_mass_g", 0.1)))

        def round_pos(key: str, step: float = length_step, min_v: float | None = None) -> None:
            val = safe_float(b.get(key), None)
            if val is None:
                return
            rv = self._floor_to_step(val, step) if mode == "min" else self._ceil_to_step(val, step)
            if min_v is not None:
                rv = max(min_v, rv)
            b[key] = round(rv, 6)

        def round_pos_material(key: str, step: float, min_v: float | None = None) -> None:
            val = safe_float(m.get(key), None)
            if val is None:
                return
            rv = self._floor_to_step(val, step) if mode == "min" else self._ceil_to_step(val, step)
            if min_v is not None:
                rv = max(min_v, rv)
            m[key] = round(rv, 6)

        def round_pos_detail(key: str, step: float = length_step, min_v: float | None = None) -> None:
            val = safe_float(d.get(key), None)
            if val is None:
                return
            rv = self._floor_to_step(val, step) if mode == "min" else self._ceil_to_step(val, step)
            if min_v is not None:
                rv = max(min_v, rv)
            d[key] = round(rv, 6)

        def round_array_abs(key: str, step: float = length_step) -> None:
            arr = b.get(key)
            if not isinstance(arr, list):
                return
            out = []
            for x in arr:
                xv = safe_float(x, None)
                if xv is None:
                    continue
                out.append(round(self._round_abs_to_step(xv, step, mode), 6))
            b[key] = list(dict.fromkeys(out))

        # Bridge geometry and distribution
        round_pos("span_mm", min_v=100.0)
        round_pos("width_mm", min_v=20.0)
        round_pos("center_height_mm", min_v=20.0)
        round_pos("end_height_mm", min_v=20.0)
        round_pos("panel_mm", min_v=20.0)
        round_pos("left_support_overhang_mm", min_v=0.0)
        round_pos("right_support_overhang_mm", min_v=0.0)
        round_pos("plateau_start_mm", min_v=0.0)
        round_pos("plateau_end_mm", min_v=0.0)
        round_array_abs("load_distribution_x_mm", length_step)
        round_array_abs("support_contact_x_left_mm", length_step)
        round_array_abs("support_contact_x_right_mm", length_step)
        round_array_abs("support_contact_y_mm", length_step)

        # Material and detail (palito a palito + cola)
        round_pos_material("stick_length_mm", length_step, min_v=30.0)
        round_pos_material("stick_width_mm", section_step, min_v=0.2)
        round_pos_material("stick_thickness_mm", section_step, min_v=0.2)
        round_pos_material("stick_mass_g", mass_step, min_v=0.01)
        round_pos_material("mass_limit_g", mass_step, min_v=1.0)
        round_pos_material("glue_reserved_g", mass_step, min_v=0.0)
        round_pos_detail("overlap_length_mm", length_step, min_v=1.0)
        round_pos_detail("min_end_margin_mm", length_step, min_v=1.0)
        round_pos_detail("reinforcement_length_mm", length_step, min_v=1.0)

        # Coerência básica pós-arredondamento
        span = float(b.get("span_mm", 1200.0))
        center_h = float(b.get("center_height_mm", 300.0))
        b["end_height_mm"] = min(float(b.get("end_height_mm", center_h)), center_h)
        b["plateau_start_mm"] = max(0.0, min(float(b.get("plateau_start_mm", span / 3.0)), span))
        b["plateau_end_mm"] = max(float(b["plateau_start_mm"]), min(float(b.get("plateau_end_mm", 2.0 * span / 3.0)), span))
        b["panel_mm"] = max(20.0, float(b.get("panel_mm", 100.0)))

        planner = v.get("planner", {})
        h_min = safe_float(planner.get("height_min_mm"), None)
        h_max = safe_float(planner.get("height_max_mm"), None)
        p_min = safe_float(planner.get("panel_min_mm"), None)
        p_max = safe_float(planner.get("panel_max_mm"), None)
        w_min = safe_float(planner.get("width_min_mm"), None)
        w_max = safe_float(planner.get("width_max_mm"), None)
        s_min = safe_float(planner.get("span_min_mm"), None)
        s_max = safe_float(planner.get("span_max_mm"), None)

        if s_min is not None:
            b["span_mm"] = max(b["span_mm"], s_min)
        if s_max is not None:
            b["span_mm"] = min(b["span_mm"], s_max)
        if w_min is not None:
            b["width_mm"] = max(b["width_mm"], w_min)
        if w_max is not None:
            b["width_mm"] = min(b["width_mm"], w_max)
        if h_min is not None:
            b["center_height_mm"] = max(b["center_height_mm"], h_min)
        if h_max is not None:
            b["center_height_mm"] = min(b["center_height_mm"], h_max)
        if p_min is not None:
            b["panel_mm"] = max(b["panel_mm"], p_min)
        if p_max is not None:
            b["panel_mm"] = min(b["panel_mm"], p_max)

        if not b.get("load_distribution_x_mm"):
            b["load_distribution_x_mm"] = [b["span_mm"] / 2.0]

        return self.config.normalize(v)

    def run(
        self,
        cfg: Dict,
        out_dir: str | Path,
        *,
        progress_callback: Callable[[float, str], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        debug_logger: Any | None = None,
    ) -> Dict:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        base = self.config.normalize(cfg)
        analysis = base.get("analysis", {})
        self._base_eval_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        t0_global = time.perf_counter()

        logs: List[str] = []

        def dbg(event_type: str, **kwargs: Any) -> None:
            if debug_logger is None:
                return
            try:
                debug_logger.event(event_type, **kwargs)
            except (TypeError, ValueError, RuntimeError) as exc:
                logs.append(f"[WARN:DEBUG_LOGGER_FAILED] {exc!r}")

        def emit_log(msg: str) -> None:
            text = str(msg)
            logs.append(text)
            if callable(log_callback):
                try:
                    log_callback(text)
                except (TypeError, ValueError, RuntimeError) as exc:
                    logs.append(f"[WARN:LOG_CALLBACK_FAILED] {exc!r}")

        def emit_progress(value: float, text: str) -> None:
            if callable(progress_callback):
                try:
                    progress_callback(max(0.0, min(1.0, float(value))), str(text))
                except (TypeError, ValueError, RuntimeError) as exc:
                    logs.append(f"[WARN:PROGRESS_CALLBACK_FAILED] {exc!r}")

        stage1_n = int(analysis.get("planner_stage1_variants", 220))
        stage1_top_k = int(analysis.get("planner_stage1_top_k", 42))
        stage2_top_k = int(analysis.get("planner_stage2_top_k", 14))
        stage2a_top_k = int(analysis.get("planner_stage2a_top_k", 220))
        stage2b_top_k = int(analysis.get("planner_stage2b_top_k", 80))
        stage3_top_k = int(analysis.get("planner_stage3_top_k", 6))
        worker_count = self._worker_count(analysis)

        stage1_rows: List[Dict] = []
        stage2_rows: List[Dict] = []
        stage3_rows: List[Dict] = []
        stage4_trace_rows: List[Dict] = []
        stage4_rows: List[Dict] = []
        discarded_rows: List[Dict] = []
        prefilter_discarded_by_reason: Dict[str, int] = {}
        discarded_by_reason: Dict[str, int] = {}

        emit_progress(0.0, "Planejador ativo: inicializando")
        emit_log("Início do planejamento multiestágio.")
        dbg("config_loaded", stage="planner", metrics={"out_dir": str(out)})

        generated_candidates = self._build_stage1_candidates(base, stage1_n)
        emit_log(f"S0 geração: {len(generated_candidates)} propostas candidatas.")
        for i, cand in enumerate(generated_candidates, 1):
            dbg(
                "s0_candidate_generated",
                stage="s0",
                candidate_id=f"S0-{i:04d}",
                metrics={
                    "side_truss_type": cand.get("side_truss_type"),
                    "top_profile": cand.get("top_profile"),
                    "panel_mm": cand.get("panel_mm"),
                },
            )

        filtered_candidates: List[Dict] = []
        for cand in generated_candidates:
            ok, reason = self._prefilter_candidate(base, cand)
            if ok:
                filtered_candidates.append(cand)
                dbg(
                    "s0_candidate_symmetry_validated",
                    stage="s0",
                    candidate_id=f"S0-{len(filtered_candidates):04d}",
                    metrics={"enforce_symmetry": bool(base.get("analysis", {}).get("enforce_symmetry", True))},
                )
                continue

            prefilter_discarded_by_reason[reason] = prefilter_discarded_by_reason.get(reason, 0) + 1
            discarded_by_reason[reason] = discarded_by_reason.get(reason, 0) + 1
            dbg(
                "s0_candidate_rejected",
                stage="s0",
                reason=reason,
                level="warning",
                metrics={"candidate": cand},
            )
            discarded_rows.append(
                {
                    "stage": "prefilter",
                    "discard_reason": reason,
                    "span_mm": cand.get("span_mm"),
                    "width_mm": cand.get("width_mm"),
                    "center_height_mm": cand.get("center_height_mm"),
                    "panel_mm": cand.get("panel_mm"),
                    "side_truss_type": cand.get("side_truss_type"),
                    "top_profile": cand.get("top_profile"),
                    "internal_truss_type": cand.get("internal_truss_type"),
                    "top_chord_truss_type": cand.get("top_chord_truss_type"),
                    "bottom_chord_truss_type": cand.get("bottom_chord_truss_type"),
                    "reinforcement_profile": cand.get("reinforcement_profile"),
                }
            )

        emit_log(
            f"S0 filtros iniciais: aprovadas={len(filtered_candidates)} | "
            f"descartadas={len(discarded_rows)}."
        )
        prefilter_discard_count = len(discarded_rows)
        for reason, qty in sorted(prefilter_discarded_by_reason.items(), key=lambda kv: kv[1], reverse=True):
            emit_log(f"  - {reason}: {qty}")

        emit_progress(0.08, "Etapa S1: avaliação estrutural preliminar")
        t0_s1 = time.perf_counter()
        s1_discard_post_solver = 0
        s1_payload: List[tuple[int, Dict, Dict]] = []

        for idx, cand in enumerate(filtered_candidates, 1):
            v = self._apply_candidate_geometry(base, cand)
            self._apply_reinforcement_profile(v, cand["reinforcement_profile"])
            v = self.config.normalize(v)
            s1_payload.append((idx, cand, v))

        def handle_s1_result(idx: int, cand: Dict, vcfg: Dict, metrics: Dict) -> None:
            nonlocal s1_discard_post_solver
            dbg("s1_solver_finished", stage="s1", candidate_id=f"S1-{idx:04d}", metrics={"solver_status": metrics.get("solver_status"), "score": metrics.get("score")})
            row = self._stage_row("s1", idx, cand, metrics, vcfg)
            if (
                metrics.get("solver_status") != "regular"
                and (safe_float(metrics.get("min_fs_primary"), 0.0) or 0.0) < 0.20
            ):
                row["discard_reason"] = "PF12_solver_irregular_fs_extremamente_baixo"
                discarded_by_reason[row["discard_reason"]] = discarded_by_reason.get(row["discard_reason"], 0) + 1
                discarded_rows.append({k: v for k, v in row.items() if k != "config"})
                s1_discard_post_solver += 1
            else:
                stage1_rows.append(row)

        if worker_count > 1 and len(s1_payload) > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as ex:
                futures = {
                    ex.submit(self._evaluate_config, vcfg, include_detail=False): (idx, cand, vcfg)
                    for idx, cand, vcfg in s1_payload
                }
                done = 0
                for fut in as_completed(futures):
                    done += 1
                    idx, cand, vcfg = futures[fut]
                    dbg("s1_solver_started", stage="s1", candidate_id=f"S1-{idx:04d}")
                    try:
                        metrics = fut.result()
                    except (TypeError, ValueError, KeyError, RuntimeError) as exc:
                        discarded_by_reason["PF12A_s1_erro_avaliacao"] = discarded_by_reason.get("PF12A_s1_erro_avaliacao", 0) + 1
                        discarded_rows.append({"stage": "s1", "discard_reason": "PF12A_s1_erro_avaliacao", "error": repr(exc)})
                        continue
                    handle_s1_result(idx, cand, vcfg, metrics)
                    if done % max(1, len(s1_payload) // 12) == 0:
                        emit_progress(0.08 + 0.36 * (done / max(1, len(s1_payload))), f"S1: {done}/{len(s1_payload)}")
        else:
            for done, (idx, cand, vcfg) in enumerate(s1_payload, 1):
                dbg("s1_solver_started", stage="s1", candidate_id=f"S1-{idx:04d}")
                metrics = self._evaluate_config(vcfg, include_detail=False)
                handle_s1_result(idx, cand, vcfg, metrics)
                if done % max(1, len(s1_payload) // 12) == 0:
                    emit_progress(0.08 + 0.36 * (done / max(1, len(s1_payload))), f"S1: {done}/{len(s1_payload)}")

        stage1_rows = self._sort_rows(stage1_rows)
        stage1_top = stage1_rows[:max(1, stage1_top_k)]
        emit_log(
            f"S1 concluída: avaliadas={len(filtered_candidates)} | "
            f"descartadas_pós_solver={s1_discard_post_solver} | válidas={len(stage1_rows)} | "
            f"selecionadas para S2={len(stage1_top)}"
        )
        if s1_discard_post_solver:
            emit_log("  - PF12_solver_irregular_fs_extremamente_baixo: " + str(s1_discard_post_solver))
        emit_log(f"S1 tempo: {time.perf_counter() - t0_s1:.2f}s")
        emit_progress(0.46, "Etapa S2A: geração e triagem rápida")
        t0_s2 = time.perf_counter()

        s2_idx = 0
        seen_cfg_keys = set()
        s2_generated_variants = 0
        s2a_preselected: List[tuple[float, Dict]] = []
        s2a_discard_mass = 0
        s2a_discard_no_gain = 0

        for seed_row in stage1_top:
            seed_cfg = self.config.normalize(seed_row["config"])
            seed_metrics = self._evaluate_config(seed_cfg, include_detail=False)
            seed_base = self._solve_and_check_base(seed_cfg)
            group_units: Dict[str, int] = {}
            stick_len_seed = max(1.0, float(seed_cfg.get("material", {}).get("stick_length_mm", 115.0)))
            for m in seed_base.get("members", []):
                pieces_per_lane = max(1, int(math.ceil(float(m.L) / stick_len_seed)))
                group_units[m.group] = group_units.get(m.group, 0) + pieces_per_lane

            mut_list = self._mutate_sticks(
                seed_cfg,
                group_min_fs=seed_metrics.get("group_min_fs", {}),
                min_support_fs=seed_metrics.get("min_support_fs"),
            )
            seed_detail = seed_cfg.get("detail_model", {})
            comp_rank = list(seed_detail.get("joint_model_rank_compression", []))
            tension_rank = list(seed_detail.get("joint_model_rank_tension", []))
            if comp_rank:
                for model in comp_rank[:3]:
                    alt = copy.deepcopy(seed_cfg)
                    alt.setdefault("detail_model", {})["compression_joint_model"] = model
                    if tension_rank:
                        alt["detail_model"]["tension_joint_model"] = tension_rank[0]
                    alt["detail_model"]["overlap_length_mm"] = float(seed_detail.get("overlap_length_mm", 30.0)) * (1.0 + 0.08 * max(0, comp_rank.index(model)))
                    mut_list.append(alt)
            s2_generated_variants += len(mut_list)

            critical_groups = [
                g for g, fs in sorted((seed_metrics.get("group_min_fs") or {}).items(), key=lambda kv: kv[1])
                if (safe_float(fs, 9.9) or 9.9) < 1.1
            ][:3]

            for mut_cfg in mut_list:
                c = self.config.normalize(mut_cfg)
                sticks = c.get("member_sticks_by_group", {})
                key = (
                    c["bridge"]["side_truss_type"],
                    c["bridge"]["top_profile"],
                    c["bridge"]["internal_truss_type"],
                    c["bridge"].get("top_chord_truss_type", "X"),
                    c["bridge"].get("bottom_chord_truss_type", "X"),
                    c["bridge"]["chord_truss_type"],
                    c["bridge"]["span_mm"],
                    c["bridge"]["width_mm"],
                    c["bridge"]["center_height_mm"],
                    c["bridge"]["panel_mm"],
                    sticks.get("top_chord"),
                    sticks.get("bottom_chord"),
                    sticks.get("diagonal"),
                    sticks.get("vertical"),
                    sticks.get("top_transverse"),
                    sticks.get("bottom_transverse"),
                    sticks.get("support_pad"),
                    c.get("detail_model", {}).get("tension_joint_model"),
                    c.get("detail_model", {}).get("compression_joint_model"),
                    round(float(c.get("detail_model", {}).get("overlap_length_mm", 0.0)), 3),
                )
                if key in seen_cfg_keys:
                    continue
                seen_cfg_keys.add(key)
                s2_idx += 1

                max_mass = float(effective_mass_limit_g(c))
                approx_mass, _ = self._estimate_mass_by_group_units(c, group_units)
                if approx_mass > 1.25 * max_mass:
                    s2a_discard_mass += 1
                    discarded_by_reason["PF13A_s2a_massa_aprox_excessiva"] = discarded_by_reason.get("PF13A_s2a_massa_aprox_excessiva", 0) + 1
                    discarded_rows.append({"stage": "s2a", "discard_reason": "PF13A_s2a_massa_aprox_excessiva", "mass_g": approx_mass, "mass_limit_g": max_mass})
                    continue

                if critical_groups:
                    gain = any(int(sticks.get(g, 1)) > int(seed_cfg.get("member_sticks_by_group", {}).get(g, 1)) for g in critical_groups)
                    if not gain:
                        s2a_discard_no_gain += 1
                        discarded_by_reason["PF13B_s2a_sem_ganho_grupo_critico"] = discarded_by_reason.get("PF13B_s2a_sem_ganho_grupo_critico", 0) + 1
                        discarded_rows.append({"stage": "s2a", "discard_reason": "PF13B_s2a_sem_ganho_grupo_critico", "critical_groups": ",".join(critical_groups)})
                        continue

                rough_strength = (
                    4.0 * int(sticks.get("top_chord", 1))
                    + 3.0 * int(sticks.get("vertical", 1))
                    + 1.8 * int(sticks.get("diagonal", 1))
                    + 1.4 * int(sticks.get("top_transverse", 1))
                )
                rough_score = rough_strength - 0.008 * approx_mass
                s2a_preselected.append((rough_score, c))

        s2a_preselected = sorted(s2a_preselected, key=lambda t: t[0], reverse=True)
        if len(s2a_preselected) > stage2a_top_k:
            s2a_preselected = s2a_preselected[:stage2a_top_k]
        emit_log(
            f"S2A concluída: geradas={s2_generated_variants} | aprovadas_rapido={len(s2a_preselected)} | "
            f"desc_massa={s2a_discard_mass} | desc_sem_ganho={s2a_discard_no_gain}"
        )

        emit_progress(0.54, "Etapa S2B: avaliação paralela no solver")
        s2b_payload = [c for _, c in s2a_preselected]
        s2b_rows: List[Dict] = []
        s2b_evaluated = 0

        if worker_count > 1 and len(s2b_payload) > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as ex:
                futures = {ex.submit(self._evaluate_config, c, include_detail=False): c for c in s2b_payload}
                for fut in as_completed(futures):
                    c = futures[fut]
                    s2b_evaluated += 1
                    try:
                        metrics = fut.result()
                    except (TypeError, ValueError, KeyError, RuntimeError) as exc:
                        discarded_by_reason["PF14_s2b_erro_avaliacao"] = discarded_by_reason.get("PF14_s2b_erro_avaliacao", 0) + 1
                        discarded_rows.append({"stage": "s2b", "discard_reason": "PF14_s2b_erro_avaliacao", "error": repr(exc)})
                        continue
                    cand = {
                        "side_truss_type": c["bridge"]["side_truss_type"],
                        "top_profile": c["bridge"]["top_profile"],
                        "internal_truss_type": c["bridge"]["internal_truss_type"],
                        "top_chord_truss_type": c["bridge"].get("top_chord_truss_type", "X"),
                        "bottom_chord_truss_type": c["bridge"].get("bottom_chord_truss_type", "X"),
                        "chord_truss_type": c["bridge"]["chord_truss_type"],
                        "reinforcement_profile": "custom_refined",
                        "span_mm": c["bridge"]["span_mm"],
                        "width_mm": c["bridge"]["width_mm"],
                        "center_height_mm": c["bridge"]["center_height_mm"],
                        "panel_mm": c["bridge"]["panel_mm"],
                    }
                    s2b_rows.append(self._stage_row("s2", s2b_evaluated, cand, metrics, c))
                    if s2b_evaluated % max(1, len(s2b_payload) // 10) == 0:
                        emit_progress(0.54 + 0.08 * (s2b_evaluated / max(1, len(s2b_payload))), f"S2B: {s2b_evaluated}/{len(s2b_payload)}")
        else:
            for idx, c in enumerate(s2b_payload, 1):
                s2b_evaluated += 1
                metrics = self._evaluate_config(c, include_detail=False)
                cand = {
                    "side_truss_type": c["bridge"]["side_truss_type"],
                    "top_profile": c["bridge"]["top_profile"],
                    "internal_truss_type": c["bridge"]["internal_truss_type"],
                    "top_chord_truss_type": c["bridge"].get("top_chord_truss_type", "X"),
                    "bottom_chord_truss_type": c["bridge"].get("bottom_chord_truss_type", "X"),
                    "chord_truss_type": c["bridge"]["chord_truss_type"],
                    "reinforcement_profile": "custom_refined",
                    "span_mm": c["bridge"]["span_mm"],
                    "width_mm": c["bridge"]["width_mm"],
                    "center_height_mm": c["bridge"]["center_height_mm"],
                    "panel_mm": c["bridge"]["panel_mm"],
                }
                s2b_rows.append(self._stage_row("s2", idx, cand, metrics, c))
                if idx % max(1, len(s2b_payload) // 10) == 0:
                    emit_progress(0.54 + 0.08 * (idx / max(1, len(s2b_payload))), f"S2B: {idx}/{len(s2b_payload)}")

        emit_progress(0.62, "Etapa S2C: filtragem de massa e shortlist")
        stage2_rows = []
        for row in s2b_rows:
            c = row["config"]
            max_mass = float(effective_mass_limit_g(c))
            mass_val = safe_float(row.get("mass_g"), 1.0e99) or 1.0e99
            if mass_val > 1.10 * max_mass:
                discarded_by_reason["PF15_s2c_massa_excedente"] = discarded_by_reason.get("PF15_s2c_massa_excedente", 0) + 1
                discarded_rows.append({"stage": "s2c", "discard_reason": "PF15_s2c_massa_excedente", "mass_g": mass_val, "mass_limit_g": max_mass})
                continue
            stage2_rows.append(row)

        stage2_rows = self._sort_rows(stage2_rows)
        stage2_top = stage2_rows[:max(1, min(stage2_top_k, stage2b_top_k))]
        emit_log(
            f"S2 concluída: geradas={s2_generated_variants} | s2a={len(s2a_preselected)} | "
            f"s2b_avaliadas={s2b_evaluated} | únicas={len(stage2_rows)} | "
            f"selecionadas para S3={len(stage2_top)}"
        )
        emit_log(f"S2 tempo: {time.perf_counter() - t0_s2:.2f}s")
        emit_progress(0.64, "Etapa S3: validação detalhada")
        t0_s3 = time.perf_counter()

        for idx, row in enumerate(stage2_top, 1):
            c = self.config.normalize(row["config"])
            detail_dir = out / "stage3_details" / f"candidate_{idx:02d}"
            metrics = self._evaluate_config(
                c,
                include_detail=True,
                detail_dir=detail_dir,
            )
            cand = {
                "side_truss_type": c["bridge"]["side_truss_type"],
                "top_profile": c["bridge"]["top_profile"],
                "internal_truss_type": c["bridge"]["internal_truss_type"],
                "top_chord_truss_type": c["bridge"].get("top_chord_truss_type", "X"),
                "bottom_chord_truss_type": c["bridge"].get("bottom_chord_truss_type", "X"),
                "chord_truss_type": c["bridge"]["chord_truss_type"],
                "reinforcement_profile": "validated_detail",
                "span_mm": c["bridge"]["span_mm"],
                "width_mm": c["bridge"]["width_mm"],
                "center_height_mm": c["bridge"]["center_height_mm"],
                "panel_mm": c["bridge"]["panel_mm"],
            }
            stage3_rows.append(self._stage_row("s3", idx, cand, metrics, c))
            emit_progress(0.64 + 0.16 * (idx / max(1, len(stage2_top))), f"S3: {idx}/{len(stage2_top)}")

        stage3_rows = self._sort_rows(stage3_rows)[:max(1, stage3_top_k)]
        emit_log(
            f"S3 concluída: validadas={len(stage3_rows)} | "
            f"viáveis={sum(1 for r in stage3_rows if r.get('feasible'))}"
        )
        emit_log(f"S3 tempo: {time.perf_counter() - t0_s3:.2f}s")
        emit_progress(0.80, "Etapa S4: refinamento adaptativo")

        if bool(analysis.get("planner_adaptive_refinement", True)):
            t0_s4 = time.perf_counter()
            seed_top_k = max(1, int(analysis.get("planner_stage4_seed_top_k", 4)))
            max_iters = max(1, int(analysis.get("planner_stage4_iterations", 12)))
            seeds = (stage3_rows or stage2_top)[:seed_top_k]
            stage4_idx = 0

            for seed_i, seed_row in enumerate(seeds, 1):
                cur_cfg = self.config.normalize(seed_row["config"])
                for it in range(1, max_iters + 1):
                    quick_metrics = self._evaluate_config(cur_cfg, include_detail=False)
                    stage4_idx += 1
                    cand = {
                        "side_truss_type": cur_cfg["bridge"]["side_truss_type"],
                        "top_profile": cur_cfg["bridge"]["top_profile"],
                        "internal_truss_type": cur_cfg["bridge"]["internal_truss_type"],
                        "top_chord_truss_type": cur_cfg["bridge"].get("top_chord_truss_type", "X"),
                        "bottom_chord_truss_type": cur_cfg["bridge"].get("bottom_chord_truss_type", "X"),
                        "chord_truss_type": cur_cfg["bridge"]["chord_truss_type"],
                        "reinforcement_profile": f"adaptive_seed_{seed_i}",
                        "span_mm": cur_cfg["bridge"]["span_mm"],
                        "width_mm": cur_cfg["bridge"]["width_mm"],
                        "center_height_mm": cur_cfg["bridge"]["center_height_mm"],
                        "panel_mm": cur_cfg["bridge"]["panel_mm"],
                    }
                    row = self._stage_row("s4_trace", stage4_idx, cand, quick_metrics, cur_cfg)
                    row["adaptive_seed"] = seed_i
                    row["adaptive_iteration"] = it
                    stage4_trace_rows.append(row)

                    if quick_metrics.get("feasible"):
                        break

                    nxt_cfg, changed, actions = self._adaptive_step(cur_cfg, quick_metrics)
                    dbg("s4_iteration_started", stage="s4", candidate_id=f"S4-{stage4_idx:04d}", metrics={"iteration": it, "score": quick_metrics.get("score")})
                    if actions:
                        row["adaptive_actions"] = " | ".join(actions)
                        for act in actions:
                            if "->" in act:
                                if "redução" in act or "alívio" in act:
                                    dbg("s4_member_lightened", stage="s4", reason=act)
                                else:
                                    dbg("s4_member_reinforced", stage="s4", reason=act)
                    if not changed:
                        row["adaptive_actions"] = row.get("adaptive_actions", "") + " | sem mudanças adicionais"
                        break
                    cur_cfg = nxt_cfg

            stage4_trace_rows = self._sort_rows(stage4_trace_rows)
            stage4_candidates = []
            seen_s4_cfg = set()
            for row in stage4_trace_rows:
                key = self._cfg_cache_key(row["config"])
                if key in seen_s4_cfg:
                    continue
                seen_s4_cfg.add(key)
                stage4_candidates.append(row)
                if len(stage4_candidates) >= max(1, stage3_top_k):
                    break

            for idx, row in enumerate(stage4_candidates, 1):
                c = self.config.normalize(row["config"])
                detail_dir = out / "stage4_details" / f"candidate_{idx:02d}"
                metrics = self._evaluate_config(
                    c,
                    include_detail=True,
                    detail_dir=detail_dir,
                )
                cand = {
                    "side_truss_type": c["bridge"]["side_truss_type"],
                    "top_profile": c["bridge"]["top_profile"],
                    "internal_truss_type": c["bridge"]["internal_truss_type"],
                    "top_chord_truss_type": c["bridge"].get("top_chord_truss_type", "X"),
                    "bottom_chord_truss_type": c["bridge"].get("bottom_chord_truss_type", "X"),
                    "chord_truss_type": c["bridge"]["chord_truss_type"],
                    "reinforcement_profile": "adaptive_validated",
                    "span_mm": c["bridge"]["span_mm"],
                    "width_mm": c["bridge"]["width_mm"],
                    "center_height_mm": c["bridge"]["center_height_mm"],
                    "panel_mm": c["bridge"]["panel_mm"],
                }
                final_row = self._stage_row("s4", idx, cand, metrics, c)
                final_row["adaptive_seed"] = row.get("adaptive_seed")
                final_row["adaptive_iteration"] = row.get("adaptive_iteration")
                final_row["adaptive_actions"] = row.get("adaptive_actions")
                stage4_rows.append(final_row)
                dbg("s4_score_changed", stage="s4", candidate_id=f"S4-{idx:04d}", metrics={"new_score": metrics.get("score")})
                emit_progress(0.80 + 0.10 * (idx / max(1, len(stage4_candidates))), f"S4: {idx}/{len(stage4_candidates)}")

            stage4_rows = self._sort_rows(stage4_rows)
            emit_log(
                f"S4 concluída: rastros={len(stage4_trace_rows)} | "
                f"validadas={len(stage4_rows)}"
            )
            emit_log(f"S4 tempo: {time.perf_counter() - t0_s4:.2f}s")
        else:
            emit_log("S4 desativada pela configuração.")

        pick_pool = stage4_rows or stage3_rows or stage2_rows or stage1_rows
        feasible = [r for r in pick_pool if r.get("feasible")]
        planner_cfg = base.get("planner", {})
        strict_mass = bool(analysis.get("strict_mass_acceptance", True))
        # use global effective mass limit here instead of mixing planner and material limits
        max_mass = effective_mass_limit_g(base)
        mass_limited_pool = [
            r for r in pick_pool
            if (safe_float(r.get("mass_g"), 1.0e99) or 1.0e99) <= max_mass
        ]
        target_break = max(
            1.0,
            float(
                planner_cfg.get(
                    "target_breaking_load_kgf",
                    planner_cfg.get("target_load_kgf", base.get("bridge", {}).get("load_total_kgf", 120.0)),
                )
            ),
        )
        if feasible:
            best = sorted(
                feasible,
                key=lambda r: (
                    -(safe_float(r.get("predicted_breaking_load_kgf"), 0.0) or 0.0),
                    -(safe_float(r.get("min_fs_primary"), 0.0) or 0.0),
                    -(safe_float(r.get("score"), -1.0e99) or -1.0e99),
                    (safe_float(r.get("mass_g"), 1.0e99) or 1.0e99),
                ),
            )[0]
        else:
            if strict_mass and not mass_limited_pool:
                best = None
                emit_log("Nenhuma proposta ficou dentro do limite de massa. Resultado não aceito.")
            else:
                near_mass_pool = [
                    r for r in pick_pool
                    if (safe_float(r.get("mass_g"), 1.0e99) or 1.0e99) <= 1.15 * max_mass
                ]
                choice_pool = mass_limited_pool if strict_mass else (near_mass_pool if near_mass_pool else pick_pool)

                def fallback_key(r: Dict) -> tuple[float, float, float]:
                    pred = safe_float(r.get("predicted_breaking_load_kgf"), 0.0) or 0.0
                    fs = safe_float(r.get("min_fs_primary"), 0.0) or 0.0
                    mass = safe_float(r.get("mass_g"), max_mass * 10.0) or (max_mass * 10.0)
                    mass_pen = max(0.0, mass / max(1.0, max_mass) - 1.0)
                    return (pred - 35.0 * mass_pen, fs, safe_float(r.get("score"), -1.0e99) or -1.0e99)

                best = sorted(
                    choice_pool,
                    key=fallback_key,
                    reverse=True,
                )[0] if choice_pool else None

        final_variants: Dict[str, Dict] = {}
        emit_progress(0.92, "Exportando resultados intermediários")

        GeometryService.write_csv(out / "active_stage1.csv", self._for_csv(stage1_rows))
        GeometryService.write_csv(out / "active_stage2.csv", self._for_csv(stage2_rows))
        GeometryService.write_csv(out / "active_stage3.csv", self._for_csv(stage3_rows))
        GeometryService.write_csv(out / "active_stage4_trace.csv", self._for_csv(stage4_trace_rows))
        GeometryService.write_csv(out / "active_stage4.csv", self._for_csv(stage4_rows))
        GeometryService.write_csv(out / "active_prefilter_discarded.csv", discarded_rows)
        GeometryService.write_csv(
            out / "active_candidates_all.csv",
            self._for_csv(stage1_rows + stage2_rows + stage3_rows + stage4_trace_rows + stage4_rows),
        )

        recommended_path = out / "recommended_config.json"
        summary_path = out / "active_best_summary.json"
        final_variants_path = out / "active_final_variants.csv"
        final_family_summary_path = out / "recommended_final_variants_summary.json"
        recommended_ideal_path = out / "recommended_config_ideal.json"
        recommended_min_path = out / "recommended_config_min.json"
        recommended_max_path = out / "recommended_config_max.json"

        if best:
            ideal_cfg = self.config.normalize(best["config"])

            if bool(analysis.get("final_variants_enabled", True)):
                variant_map = {
                    "ideal": ideal_cfg,
                    "min": self._build_conservative_cfg(ideal_cfg, "min"),
                    "max": self._build_conservative_cfg(ideal_cfg, "max"),
                }

                final_rows = []

                for idx, (label, vc) in enumerate(variant_map.items(), 1):
                    metrics = self._evaluate_config(
                        vc,
                        include_detail=True,
                        detail_dir=out / "final_variants_details" / label,
                    )
                    cand = {
                        "side_truss_type": vc["bridge"]["side_truss_type"],
                        "top_profile": vc["bridge"]["top_profile"],
                        "internal_truss_type": vc["bridge"]["internal_truss_type"],
                        "top_chord_truss_type": vc["bridge"].get("top_chord_truss_type", "X"),
                        "bottom_chord_truss_type": vc["bridge"].get("bottom_chord_truss_type", "X"),
                        "chord_truss_type": vc["bridge"]["chord_truss_type"],
                        "reinforcement_profile": f"final_{label}",
                        "span_mm": vc["bridge"]["span_mm"],
                        "width_mm": vc["bridge"]["width_mm"],
                        "center_height_mm": vc["bridge"]["center_height_mm"],
                        "panel_mm": vc["bridge"]["panel_mm"],
                    }
                    row = self._stage_row("final_variant", idx, cand, metrics, vc)
                    row["variant_label"] = label
                    final_rows.append(row)
                    final_variants[label] = row

                GeometryService.write_csv(final_variants_path, self._for_csv(final_rows))
                final_family_summary_path.write_text(
                    json.dumps(
                        {k: {ik: iv for ik, iv in v.items() if ik != "config"} for k, v in final_variants.items()},
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                recommended_ideal_path.write_text(
                    json.dumps(final_variants["ideal"]["config"], indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                recommended_min_path.write_text(
                    json.dumps(final_variants["min"]["config"], indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                recommended_max_path.write_text(
                    json.dumps(final_variants["max"]["config"], indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                # evaluate each final variant against the global mass limit
                max_mass_limit = effective_mass_limit_g(base)
                if strict_mass:
                    accepted_labels = [
                        k for k, row in final_variants.items()
                        if (safe_float(row.get("mass_g"), 1.0e99) or 1.0e99) <= max_mass_limit
                    ]
                else:
                    accepted_labels = list(final_variants.keys())
                if accepted_labels:
                    accepted_rows = [final_variants[k] for k in accepted_labels]
                    best_final = sorted(
                        accepted_rows,
                        key=lambda r: (
                            -(safe_float(r.get("predicted_breaking_load_kgf"), 0.0) or 0.0),
                            -(safe_float(r.get("min_fs_primary"), 0.0) or 0.0),
                            -(safe_float(r.get("score"), -1.0e99) or -1.0e99),
                            (safe_float(r.get("mass_g"), 1.0e99) or 1.0e99),
                        ),
                    )[0]
                    recommended_path.write_text(
                        json.dumps(best_final["config"], indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    summary_path.write_text(
                        json.dumps(
                            {k: v for k, v in best_final.items() if k != "config"},
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    best = best_final
                    dbg("final_candidate_selected", stage="final", metrics={"variant_label": best_final.get("variant_label"), "score": best_final.get("score")})
                else:
                    best = None
                    emit_log("Nenhuma versão final (ideal/min/max) atende ao limite de massa.")
                    dbg("final_candidate_rejected", stage="final", level="warning", reason="all_variants_over_mass_limit")
            else:
                recommended_path.write_text(
                    json.dumps(ideal_cfg, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                summary_path.write_text(
                    json.dumps(
                        {k: v for k, v in best.items() if k != "config"},
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
        else:
            GeometryService.write_csv(final_variants_path, [])

        emit_log(
            f"Cache de avaliação: hits={self._cache_hits} | misses={self._cache_misses} | "
            f"entradas={len(self._base_eval_cache)}"
        )
        emit_log(f"Tempo total do planejador: {time.perf_counter() - t0_global:.2f}s")
        emit_progress(1.0, "Planejamento concluído")
        emit_log("Fim do planejamento multiestágio.")
        if best is None:
            dbg("no_feasible_candidate", stage="final", level="warning")

        return {
            "stage1": stage1_rows,
            "stage2": stage2_rows,
            "stage3": stage3_rows,
            "stage4_trace": stage4_trace_rows,
            "stage4": stage4_rows,
            "discarded": discarded_rows,
            "logs": logs,
            "best": best,
            "best_is_feasible": bool(best and best.get("feasible")),
            "recommended_config_path": str(recommended_path),
            "best_summary_path": str(summary_path),
            "final_variants": final_variants,
            "final_variants_path": str(final_variants_path),
            "recommended_config_ideal_path": str(recommended_ideal_path),
            "recommended_config_min_path": str(recommended_min_path),
            "recommended_config_max_path": str(recommended_max_path),
            "final_family_summary_path": str(final_family_summary_path),
            "stage_counts": {
                "symmetry_enforced": bool(base.get("analysis", {}).get("enforce_symmetry", True)),
                "symmetry_discarded": sum(
                    int(v)
                    for k, v in prefilter_discarded_by_reason.items()
                    if str(k).startswith("SYM_")
                ),
                "stage0_generated": len(generated_candidates),
                "stage0_prefilter_passed": len(filtered_candidates),
                "stage0_prefilter_discarded": prefilter_discard_count,
                "discarded_total": len(discarded_rows),
                "stage0_prefilter_discarded_by_reason": prefilter_discarded_by_reason,
                "discarded_by_reason": discarded_by_reason,
                "stage1_evaluated": len(filtered_candidates),
                "stage1_discarded_post_solver": s1_discard_post_solver,
                "stage1": len(stage1_rows),
                "stage2_generated": s2_generated_variants,
                "stage2a_selected": len(s2a_preselected),
                "stage2b_evaluated": s2b_evaluated,
                "stage2_unique": len(stage2_rows),
                "stage2": len(stage2_rows),
                "stage3": len(stage3_rows),
                "stage4_trace": len(stage4_trace_rows),
                "stage4": len(stage4_rows),
                "final_variants": len(final_variants),
            },
        }
