from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.services.config_service import ConfigService
from src.services.geometry_service import GeometryService
from src.services.postprocessor import PostProcessor
from src.services.stick_detail_service import StickDetailService
from src.solvers.linear_truss_solver import LinearTrussSolver


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default

        if isinstance(value, str) and value.strip() == "":
            return default

        v = float(value)

        if math.isnan(v) or math.isinf(v):
            return default

        return v
    except Exception:
        return default


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
        waste = float(cfg.get("detail_model", {}).get("construction_waste_factor", 0.08))
        glue_reserved = float(mat.get("glue_reserved_g", 0.0))

        raw_sticks = 0

        for m in members:
            pieces_per_lane = max(1, int(math.ceil(float(m.L) / stick_len)))
            raw_sticks += pieces_per_lane * int(m.n_sticks)

        sticks_with_waste = int(math.ceil(raw_sticks * (1.0 + waste)))
        mass = sticks_with_waste * stick_mass + glue_reserved
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
        chord = str(candidate["chord_truss_type"])

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
                "chord_truss_type": chord,
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
        max_mass = max(1.0, float(planner.get("max_bridge_mass_g", mat.get("mass_limit_g", 1000.0))))
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

        fs_score = min(2.0, min_fs_primary / target_fs)
        break_score = max(0.0, 1.0 - abs(predicted_break - target_break) / target_break)
        mass_target_score = max(0.0, 1.0 - abs(mass_g - target_mass) / target_mass)
        mass_limit_score = max(0.0, min(1.0, (max_mass - mass_g) / max_mass))

        profile = str(analysis.get("planner_objective_profile", "balanced")).lower()
        presets = {
            "balanced": (0.52, 0.28, 0.12, 0.08),
            "max_strength": (0.68, 0.24, 0.05, 0.03),
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

        score = 0.0
        score += 100.0 * (
            w_fs * min(1.0, fs_score)
            + w_break * break_score
            + w_mass_target * mass_target_score
            + w_mass_limit * mass_limit_score
        )
        score += 35.0 * w_fs * max(0.0, fs_score - 1.0)

        if fs_score < 1.0:
            score -= 170.0 * w_fs * (1.0 - fs_score)

        if min_fs_primary < 1.0:
            score -= 110.0 * w_fs * (1.0 - min_fs_primary)

        if mass_g > max_mass:
            score -= 40.0 + 75.0 * (mass_g - max_mass) / max_mass

        if metrics.get("solver_status") != "regular":
            score -= 18.0

        if min_support_fs is not None and min_support_fs < 1.0:
            score -= 35.0 * (1.0 - min_support_fs)

        if eq_error > 1e-3 * load_total_N:
            score -= 20.0

        score -= 2.5 * float(metrics.get("inactive_support_count", 0))

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
        target_fs = float(analysis.get("target_min_fs", 2.0))
        max_mass = float(planner.get("max_bridge_mass_g", c["material"].get("mass_limit_g", 1000.0)))
        current_mass = float(metrics.get("mass_g", 0.0) or 0.0)

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
            new = min(max_sticks, old + inc)

            if new != old:
                sticks[group] = new
                changed = True
                fs_txt = requested_fs.get(group, 0.0)
                actions.append(f"{group}: {old}->{new} (FS={fs_txt:.2f})")

        min_support_fs = safe_float(metrics.get("min_support_fs"), None)
        if min_support_fs is not None and min_support_fs < 1.2:
            old = int(sticks.get("support_pad", min_sticks))
            new = min(max_sticks, old + 1)
            if new != old:
                sticks["support_pad"] = new
                changed = True
                actions.append(f"support_pad: {old}->{new} (FS_apoio={min_support_fs:.2f})")

        if metrics.get("solver_status") != "regular":
            for g in ("top_bracing", "bottom_bracing", "cross_frame_bracing"):
                old = int(sticks.get(g, min_sticks))
                new = min(max_sticks, old + 1)
                if new != old:
                    sticks[g] = new
                    changed = True
                    actions.append(f"{g}: {old}->{new} (solver irregular)")

        if current_mass > max_mass:
            group_min_fs = self._group_min_fs(metrics.get("member_checks", []))
            for g in ("top_bracing", "bottom_bracing", "cross_frame_bracing", "chord_lacing"):
                old = int(sticks.get(g, min_sticks))
                fs_g = group_min_fs.get(g, 999.0)
                if old > min_sticks and fs_g >= 2.8:
                    sticks[g] = old - 1
                    changed = True
                    actions.append(f"{g}: {old}->{old-1} (alívio de massa)")

        min_fs_primary = float(metrics.get("min_fs_primary", 0.0) or 0.0)
        if min_fs_primary < target_fs * 0.7:
            h = float(bridge["center_height_mm"])
            panel = float(bridge["panel_mm"])
            h_max = float(planner.get("height_max_mm", h))
            panel_min = float(planner.get("panel_min_mm", panel))

            new_h = min(h_max, h * 1.08)
            if new_h > h + 1.0:
                bridge["center_height_mm"] = round(new_h, 6)
                bridge["end_height_mm"] = new_h if bridge.get("top_profile") == "flat" else max(50.0, new_h / 3.0)
                changed = True
                actions.append(f"altura: {h:.0f}->{new_h:.0f} mm")

            new_panel = max(panel_min, panel * 0.92)
            if new_panel < panel - 1.0:
                bridge["panel_mm"] = round(new_panel, 6)
                changed = True
                actions.append(f"painel: {panel:.0f}->{new_panel:.0f} mm")

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

        primary_checks = [
            r for r in member_checks
            if r.get("member_role") == "primary"
        ]
        min_fs_primary = min_numeric((r.get("FS_min") for r in primary_checks), 0.0) or 0.0
        min_fs_all = min_numeric((r.get("FS_min") for r in member_checks), 0.0) or 0.0
        min_support_fs = min_numeric((r.get("FS_support_reaction") for r in support_checks), None)

        quick_mass_g, quick_sticks = self._quick_mass_estimate(cfg, members)

        detailed = None
        mass_g = quick_mass_g
        glue_mass_g = None
        weak_glue = None

        if include_detail:
            dd = detail_dir or Path("outputs/tmp_detail")
            detailed = self.detail.analyze(
                cfg,
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

        load_kgf = float(cfg["bridge"]["load_total_kgf"])
        estimated_breaking_load_kgf = load_kgf * min_fs_primary

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
            "detailed": detailed,
        }

        metrics["score"] = self._score_candidate(cfg, metrics)

        target_fs = float(cfg.get("analysis", {}).get("target_min_fs", 2.0))
        max_mass = float(cfg.get("planner", {}).get("max_bridge_mass_g", cfg["material"].get("mass_limit_g", 1000.0)))

        feasible = (
            metrics["solver_status"] == "regular"
            and metrics["min_fs_primary"] >= target_fs
            and metrics["mass_g"] <= max_mass
            and (
                metrics["min_support_fs"] is None
                or metrics["min_support_fs"] >= 1.0
            )
        )
        metrics["feasible"] = feasible
        return metrics

    def _build_stage1_candidates(self, cfg: Dict, n_target: int) -> List[Dict]:
        planner = cfg.get("planner", {})
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

        side_vals = list(planner.get("consider_side_trusses", ["Parker", "Pratt", "Howe", "Warren"]))
        top_vals = list(planner.get("consider_top_profiles", ["parker_plateau", "triangular_peak", "shallow_arch", "flat"]))
        internal_vals = list(planner.get("consider_internal_trusses", ["X", "Warren", "Pratt", "Howe", "none"]))
        chord_vals = list(planner.get("consider_chord_trusses", ["none", "Warren", "X"]))
        reinforce_vals = list(self._reinforcement_profiles().keys())

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
                c["chord_truss_type"],
                c["reinforcement_profile"],
            )

            if key in seen:
                return

            span = float(c["span_mm"])
            panel = float(c["panel_mm"])
            n_panels = span / max(panel, 1.0)

            if n_panels < 6 or n_panels > 24:
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
                                "chord_truss_type": chord_vals[0],
                                "reinforcement_profile": "balanced",
                            }
                        )

        attempts = max(200, n_target * 30)

        for _ in range(attempts):
            if len(candidates) >= n_target:
                break

            add_candidate(
                {
                    "span_mm": rng.choice(span_vals),
                    "width_mm": rng.choice(width_vals),
                    "center_height_mm": rng.choice(height_vals),
                    "panel_mm": rng.choice(panel_vals),
                    "side_truss_type": rng.choice(side_vals),
                    "top_profile": rng.choice(top_vals),
                    "internal_truss_type": rng.choice(internal_vals),
                    "chord_truss_type": rng.choice(chord_vals),
                    "reinforcement_profile": rng.choice(reinforce_vals),
                }
            )

        return candidates[:n_target]

    @staticmethod
    def _stage_row(stage: str, idx: int, candidate: Dict, metrics: Dict, cfg: Dict) -> Dict:
        row = {
            "stage": stage,
            "candidate_id": f"{stage.upper()}-{idx:04d}",
            "side_truss_type": candidate.get("side_truss_type", cfg["bridge"].get("side_truss_type")),
            "top_profile": candidate.get("top_profile", cfg["bridge"].get("top_profile")),
            "internal_truss_type": candidate.get("internal_truss_type", cfg["bridge"].get("internal_truss_type")),
            "chord_truss_type": candidate.get("chord_truss_type", cfg["bridge"].get("chord_truss_type")),
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
    def _mutate_sticks(base_cfg: Dict) -> List[Dict]:
        groups = [
            "top_chord",
            "bottom_chord",
            "diagonal",
            "vertical",
            "top_transverse",
            "bottom_transverse",
            "support_pad",
        ]
        patterns = [
            {},
            {"top_chord": 1},
            {"top_chord": 1, "diagonal": 1},
            {"bottom_chord": 1},
            {"vertical": 1},
            {"diagonal": 1},
            {"top_transverse": 1, "bottom_transverse": 1},
            {"top_chord": -1, "bottom_chord": -1},
            {"diagonal": -1, "vertical": -1},
            {"support_pad": 1},
            {"top_chord": 2, "diagonal": 1},
            {"top_chord": -1, "diagonal": -1},
        ]

        variants = []

        for p in patterns:
            cfg = copy.deepcopy(base_cfg)
            sticks = cfg.setdefault("member_sticks_by_group", {})

            for g in groups:
                base_n = int(sticks.get(g, 1))
                delta = int(p.get(g, 0))
                sticks[g] = max(1, min(9, base_n + delta))

            variants.append(cfg)

        return variants

    @staticmethod
    def _for_csv(rows: List[Dict]) -> List[Dict]:
        clean = []
        for r in rows:
            c = {k: v for k, v in r.items() if k != "config"}
            clean.append(c)
        return clean

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

    def run(self, cfg: Dict, out_dir: str | Path) -> Dict:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        base = self.config.normalize(cfg)
        analysis = base.get("analysis", {})

        stage1_n = int(analysis.get("planner_stage1_variants", 220))
        stage1_top_k = int(analysis.get("planner_stage1_top_k", 42))
        stage2_top_k = int(analysis.get("planner_stage2_top_k", 14))
        stage3_top_k = int(analysis.get("planner_stage3_top_k", 6))

        stage1_rows: List[Dict] = []
        stage2_rows: List[Dict] = []
        stage3_rows: List[Dict] = []
        stage4_trace_rows: List[Dict] = []
        stage4_rows: List[Dict] = []

        for idx, cand in enumerate(self._build_stage1_candidates(base, stage1_n), 1):
            v = self._apply_candidate_geometry(base, cand)
            self._apply_reinforcement_profile(v, cand["reinforcement_profile"])
            v = self.config.normalize(v)
            metrics = self._evaluate_config(v, include_detail=False)
            stage1_rows.append(self._stage_row("s1", idx, cand, metrics, v))

        stage1_rows = self._sort_rows(stage1_rows)
        stage1_top = stage1_rows[:max(1, stage1_top_k)]

        s2_idx = 0
        seen_cfg_keys = set()

        for seed_row in stage1_top:
            for mut_cfg in self._mutate_sticks(seed_row["config"]):
                c = self.config.normalize(mut_cfg)
                sticks = c.get("member_sticks_by_group", {})
                key = (
                    c["bridge"]["side_truss_type"],
                    c["bridge"]["top_profile"],
                    c["bridge"]["internal_truss_type"],
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
                )
                if key in seen_cfg_keys:
                    continue
                seen_cfg_keys.add(key)
                s2_idx += 1
                metrics = self._evaluate_config(c, include_detail=False)
                cand = {
                    "side_truss_type": c["bridge"]["side_truss_type"],
                    "top_profile": c["bridge"]["top_profile"],
                    "internal_truss_type": c["bridge"]["internal_truss_type"],
                    "chord_truss_type": c["bridge"]["chord_truss_type"],
                    "reinforcement_profile": "custom_refined",
                    "span_mm": c["bridge"]["span_mm"],
                    "width_mm": c["bridge"]["width_mm"],
                    "center_height_mm": c["bridge"]["center_height_mm"],
                    "panel_mm": c["bridge"]["panel_mm"],
                }
                stage2_rows.append(self._stage_row("s2", s2_idx, cand, metrics, c))

        stage2_rows = self._sort_rows(stage2_rows)
        stage2_top = stage2_rows[:max(1, stage2_top_k)]

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
                "chord_truss_type": c["bridge"]["chord_truss_type"],
                "reinforcement_profile": "validated_detail",
                "span_mm": c["bridge"]["span_mm"],
                "width_mm": c["bridge"]["width_mm"],
                "center_height_mm": c["bridge"]["center_height_mm"],
                "panel_mm": c["bridge"]["panel_mm"],
            }
            stage3_rows.append(self._stage_row("s3", idx, cand, metrics, c))

        stage3_rows = self._sort_rows(stage3_rows)[:max(1, stage3_top_k)]

        if bool(analysis.get("planner_adaptive_refinement", True)):
            seed_top_k = max(1, int(analysis.get("planner_stage4_seed_top_k", 4)))
            max_iters = max(1, int(analysis.get("planner_stage4_iterations", 8)))
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
                    if actions:
                        row["adaptive_actions"] = " | ".join(actions)
                    if not changed:
                        row["adaptive_actions"] = row.get("adaptive_actions", "") + " | sem mudanças adicionais"
                        break
                    cur_cfg = nxt_cfg

            stage4_trace_rows = self._sort_rows(stage4_trace_rows)
            stage4_candidates = stage4_trace_rows[:max(1, stage3_top_k)]

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

            stage4_rows = self._sort_rows(stage4_rows)

        pick_pool = stage4_rows or stage3_rows or stage2_rows or stage1_rows
        feasible = [r for r in pick_pool if r.get("feasible")]
        best = feasible[0] if feasible else (pick_pool[0] if pick_pool else None)

        final_variants: Dict[str, Dict] = {}

        GeometryService.write_csv(out / "active_stage1.csv", self._for_csv(stage1_rows))
        GeometryService.write_csv(out / "active_stage2.csv", self._for_csv(stage2_rows))
        GeometryService.write_csv(out / "active_stage3.csv", self._for_csv(stage3_rows))
        GeometryService.write_csv(out / "active_stage4_trace.csv", self._for_csv(stage4_trace_rows))
        GeometryService.write_csv(out / "active_stage4.csv", self._for_csv(stage4_rows))
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

                # Mantém retrocompatibilidade: recomendado principal = ideal
                recommended_path.write_text(
                    json.dumps(final_variants["ideal"]["config"], indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                summary_path.write_text(
                    json.dumps(
                        {k: v for k, v in final_variants["ideal"].items() if k != "config"},
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                best = final_variants["ideal"]
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

        return {
            "stage1": stage1_rows,
            "stage2": stage2_rows,
            "stage3": stage3_rows,
            "stage4_trace": stage4_trace_rows,
            "stage4": stage4_rows,
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
                "stage1": len(stage1_rows),
                "stage2": len(stage2_rows),
                "stage3": len(stage3_rows),
                "stage4_trace": len(stage4_trace_rows),
                "stage4": len(stage4_rows),
                "final_variants": len(final_variants),
            },
        }
