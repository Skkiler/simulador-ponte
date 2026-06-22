from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.core.numeric import safe_float
from src.domain.models import Member, Node


@dataclass
class JointDecision:
    member_id: int
    member_group: str
    force_state: str
    abs_force_N: float
    axial_force_ratio: float
    FS_min: float
    utilization: float
    recommended_joint_model: str
    required_overlap_mm: float
    joint_area_factor: float
    secondary_bending_factor: float
    estimated_joint_fs: float
    geometry_fit_ok: bool
    angle_adjustment_required: bool
    reason: str
    compatibility_flags: List[str]
    symmetry_partner_ids: List[int]


class ConnectionPlanner:
    @staticmethod
    def _joint_properties(model: str) -> Tuple[float, float]:
        area = {
            "butt_plain": 0.35,
            "single_lap": 1.00,
            "single_lap_tala": 1.30,
            "double_lap": 1.75,
            "double_lap_reinforced": 2.10,
        }
        bend = {
            "butt_plain": 1.55,
            "single_lap": 1.25,
            "single_lap_tala": 1.12,
            "double_lap": 1.00,
            "double_lap_reinforced": 0.95,
        }
        return float(area.get(model, 1.0)), float(bend.get(model, 1.0))

    @staticmethod
    def _force_state(n_value: float, near_zero_tol: float) -> str:
        if abs(n_value) <= near_zero_tol:
            return "near_zero"
        return "tension" if n_value >= 0.0 else "compression"

    @staticmethod
    def _required_overlap_mm(
        base_overlap: float,
        level: str,
        state: str,
        ratio: float,
        *,
        force_per_lane_N: float = 0.0,
        stick_width_mm: float = 7.0,
        glue_shear_strength_MPa: float = 3.5,
        glue_safety_factor: float = 2.0,
        area_factor: float = 1.0,
        secondary_bending_factor: float = 1.0,
        stick_length_mm: float = 115.0,
    ) -> float:
        severity = {
            "light": 0.85,
            "moderate": 1.00,
            "reinforced": 1.20,
        }.get(level, 1.0)
        if state == "compression":
            severity *= 1.12
        if ratio >= 0.70:
            severity *= 1.10

        stick_w = max(1.0, float(stick_width_mm))
        tau_design = max(0.05, float(glue_shear_strength_MPa) / max(1.0e-6, float(glue_safety_factor)))
        area_fac = max(0.10, float(area_factor))
        bend_fac = max(0.50, float(secondary_bending_factor))
        demand = abs(float(force_per_lane_N)) * bend_fac
        overlap_by_shear = demand / max(1.0e-9, tau_design * stick_w * area_fac)

        constructive_min = max(10.0, 0.12 * float(stick_length_mm), 0.70 * float(base_overlap))
        required = max(constructive_min * severity, overlap_by_shear * 1.15)
        return max(8.0, min(0.85 * float(stick_length_mm), required))

    def _symmetry_partners(
        self,
        cfg: Dict,
        nodes: List[Node],
        members: List[Member],
    ) -> Dict[int, List[int]]:
        span = float(cfg.get("bridge", {}).get("span_mm", 0.0))
        x_mid = span * 0.5
        node_by_id = {n.id: n for n in nodes}

        def ekey(p1: tuple[float, float, float], p2: tuple[float, float, float], group: str) -> tuple:
            a = (round(p1[0], 6), round(p1[1], 6), round(p1[2], 6))
            b = (round(p2[0], 6), round(p2[1], 6), round(p2[2], 6))
            e0, e1 = (a, b) if a <= b else (b, a)
            return (e0, e1, str(group))

        key_to_id: Dict[tuple, int] = {}
        for m in members:
            ni = node_by_id[m.i]
            nj = node_by_id[m.j]
            key_to_id[ekey((ni.x, ni.y, ni.z), (nj.x, nj.y, nj.z), m.group)] = int(m.id)

        out: Dict[int, List[int]] = {}
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
                k = ekey((p1x, p1y, float(ni.z)), (p2x, p2y, float(nj.z)), m.group)
                pid = key_to_id.get(k)
                if pid is not None:
                    ids.add(int(pid))
            out[int(m.id)] = sorted(i for i in ids if i != int(m.id))
        return out

    def build_joint_catalog(self, cfg: Dict) -> Dict[str, str]:
        detail = cfg.get("detail_model", {}) or {}
        keys = set()
        for mp in (
            detail.get("joint_efficiency_tension_by_model", {}),
            detail.get("joint_efficiency_compression_by_model", {}),
        ):
            keys.update(mp.keys())
        if not keys:
            keys = {"single_lap", "single_lap_tala", "double_lap", "double_lap_reinforced"}
        if bool(detail.get("physical_connection_policy_enabled", False)):
            prohibited = {"butt_plain", "edge_bond_plain", "side_by_side_plain"}
            keys = {k for k in keys if str(k) not in prohibited}
        return {k: k for k in sorted(keys)}

    def assign_member_joint_plan(
        self,
        cfg: Dict,
        nodes: List[Node],
        members: List[Member],
        member_results: List[Dict],
        member_checks: List[Dict],
        member_sizing_plan: Dict[int, Dict[str, Any]] | None = None,
    ) -> Dict[int, Dict]:
        analysis = cfg.get("analysis", {}) or {}
        planner = cfg.get("planner", {}) or {}
        local = planner.get("local_sizing", {}) or {}
        detail = cfg.get("detail_model", {}) or {}
        target_fs = float(analysis.get("target_min_fs", 2.0))
        base_overlap = float(detail.get("overlap_length_mm", 30.0))
        min_end_margin = float(detail.get("min_end_margin_mm", 10.0))
        near_zero_tol = float(
            local.get(
                "zero_force_tolerance_N",
                detail.get("near_zero_force_tolerance_N", 8.0),
            )
        )
        stick_len = float(cfg.get("material", {}).get("stick_length_mm", 115.0))
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

        res_map = {int(r.get("member_id")): r for r in (member_results or []) if r.get("member_id") is not None}
        chk_map = {int(r.get("member_id")): r for r in (member_checks or []) if r.get("member_id") is not None}
        sizing_plan = member_sizing_plan or {}
        max_abs_n = max((abs(safe_float(r.get("N_N"), 0.0) or 0.0) for r in res_map.values()), default=0.0)
        max_abs_n = max(1.0e-9, max_abs_n)
        partners = self._symmetry_partners(cfg, nodes, members) if members and nodes else {}

        def base_level(ratio: float) -> str:
            if ratio < 0.15:
                return "light"
            if ratio < 0.40:
                return "moderate"
            return "reinforced"

        # Primeira passagem.
        raw: Dict[int, Dict[str, Any]] = {}
        for m in members:
            mid = int(m.id)
            res = res_map.get(mid, {})
            chk = chk_map.get(mid, {})
            n_val = safe_float(res.get("N_N"), 0.0) or 0.0
            abs_n = abs(n_val)
            ratio = abs_n / max_abs_n
            fs_min = safe_float(chk.get("FS_min"), target_fs) or target_fs
            util = safe_float(chk.get("utilization"), 0.0) or 0.0
            state = self._force_state(n_val, near_zero_tol)
            group = str(getattr(m, "group", ""))
            role = "primary" if group in primary_groups else "secondary"
            sizing_row = sizing_plan.get(mid) or sizing_plan.get(str(mid)) or {}
            force_band = str(sizing_row.get("force_band", "") or "")

            lvl = base_level(ratio)
            if fs_min < target_fs:
                lvl = {"light": "moderate", "moderate": "reinforced", "reinforced": "reinforced"}[lvl]
            if state == "compression" and lvl == "light" and state != "near_zero":
                lvl = "moderate"
            if role == "primary" and ratio >= 0.40:
                lvl = "reinforced"
            if force_band == "near_zero" or state == "near_zero":
                lvl = "light"

            if force_band == "near_zero" or state == "near_zero":
                model = "single_lap_tala" if n_val >= 0 else "single_lap"
            elif lvl == "light":
                model = "single_lap_tala" if state == "tension" else "single_lap"
            elif lvl == "moderate":
                model = "double_lap"
            else:
                model = "double_lap_reinforced"

            if state == "compression" and model == "butt_plain":
                model = "double_lap"
            if bool(detail.get("physical_connection_policy_enabled", False)) and model == "butt_plain":
                model = "double_lap_reinforced"

            area_factor, bend_factor = self._joint_properties(model)
            required_overlap = self._required_overlap_mm(
                base_overlap,
                lvl,
                state,
                ratio,
                force_per_lane_N=abs_n / max(1, int(getattr(m, "n_sticks", 1))),
                stick_width_mm=float(cfg.get("material", {}).get("stick_width_mm", 7.0)),
                glue_shear_strength_MPa=float(detail.get("glue_shear_strength_MPa", 3.5)),
                glue_safety_factor=float(detail.get("default_joint_safety_factor", 2.0)),
                area_factor=area_factor,
                secondary_bending_factor=bend_factor,
                stick_length_mm=stick_len,
            )
            eff_map_t = detail.get("joint_efficiency_tension_by_model", {}) or {}
            eff_map_c = detail.get("joint_efficiency_compression_by_model", {}) or {}
            eff = float(eff_map_t.get(model, 0.85)) if state != "compression" else float(eff_map_c.get(model, 0.80))
            estimated_joint_fs = max(0.0, fs_min * eff / max(1.0e-6, bend_factor))

            # Ajuste geométrico básico.
            geometry_fit_ok = True
            flags: List[str] = []
            primary_overlap_floor = max(0.0, float(detail.get("min_primary_overlap_mm", 25.0)))
            chord_overlap_floor = max(primary_overlap_floor, float(detail.get("min_chord_overlap_mm", 30.0)))
            overlap_floor = 0.0
            if group in {"top_chord", "bottom_chord"} and state in {"tension", "compression"}:
                overlap_floor = chord_overlap_floor
            elif role == "primary" and state in {"tension", "compression"}:
                overlap_floor = primary_overlap_floor
            if required_overlap < overlap_floor:
                required_overlap = overlap_floor
                flags.append("overlap_raised_to_constructive_min")
            if required_overlap > 0.85 * stick_len:
                geometry_fit_ok = False
                flags.append("overlap_exceeds_stick_fraction")
            if required_overlap + min_end_margin > max(1.0, float(m.L) * 0.9):
                geometry_fit_ok = False
                flags.append("overlap_not_compatible_with_member_length")
            if state == "compression":
                flags.append("avoid_eccentricity")
            if role == "primary" and ratio >= 0.40:
                flags.append("primary_heavy_loaded")

            reason = (
                f"ratio={ratio:.3f}, FS={fs_min:.2f}, state={state}, role={role}, "
                f"level={lvl}"
            )
            raw[mid] = {
                "member_id": mid,
                "member_group": group,
                "force_state": state,
                "abs_force_N": abs_n,
                "axial_force_ratio": ratio,
                "force_band": force_band or state,
                "FS_min": fs_min,
                "utilization": util,
                "recommended_joint_model": model,
                "required_overlap_mm": required_overlap,
                "joint_area_factor": area_factor,
                "secondary_bending_factor": bend_factor,
                "estimated_joint_fs": estimated_joint_fs,
                "geometry_fit_ok": geometry_fit_ok,
                "angle_adjustment_required": bool(abs_n > 0 and state == "compression" and ratio >= 0.40),
                "reason": reason,
                "compatibility_flags": sorted(set(flags)),
                "symmetry_partner_ids": partners.get(mid, []),
            }

        # Segunda passagem: coerência entre pares simétricos (mesmo nível de robustez).
        severity = {
            "single_lap": 1,
            "single_lap_tala": 1,
            "double_lap": 2,
            "double_lap_reinforced": 3,
            "butt_plain": 0,
        }
        enforce_symmetry = bool(analysis.get("enforce_symmetry", True))
        if enforce_symmetry:
            visited = set()
            for mid, row in raw.items():
                if mid in visited:
                    continue
                ids = set([mid] + list(row.get("symmetry_partner_ids", [])))
                visited.update(ids)
                max_sev = max(severity.get(raw.get(i, {}).get("recommended_joint_model", "single_lap"), 1) for i in ids if i in raw)
                for i in ids:
                    if i not in raw:
                        continue
                    cur = raw[i]["recommended_joint_model"]
                    cur_sev = severity.get(cur, 1)
                    if cur_sev < max_sev:
                        raw[i]["recommended_joint_model"] = "double_lap" if max_sev == 2 else "double_lap_reinforced"
                        raw[i]["reason"] += " | symmetry_upgraded"
                        af, bf = self._joint_properties(raw[i]["recommended_joint_model"])
                        raw[i]["joint_area_factor"] = af
                        raw[i]["secondary_bending_factor"] = bf

        return {int(k): v for k, v in raw.items()}

    def choose_joint_model(self, member, N: float, FS: float, group: str, cfg: Dict) -> JointDecision:
        fake_member = Member(
            int(member.id),
            int(getattr(member, "i", 0)),
            int(getattr(member, "j", 0)),
            str(group),
            int(getattr(member, "n_sticks", 1)),
            float(getattr(member, "A", 1.0)),
            float(getattr(member, "Asy", 1.0)),
            float(getattr(member, "Asz", 1.0)),
            float(getattr(member, "Iy", 1.0)),
            float(getattr(member, "Iz", 1.0)),
            float(getattr(member, "J", 1.0)),
            float(getattr(member, "E", 1.0)),
            float(getattr(member, "G", 1.0)),
            float(getattr(member, "Ky", 1.0)),
            float(getattr(member, "Kz", 1.0)),
            float(getattr(member, "L", 1.0)),
            str(getattr(member, "layout", "stacked")),
            str(getattr(member, "stick_orientation", "flat")),
        )
        plan = self.assign_member_joint_plan(
            cfg,
            [],
            [fake_member],
            [{"member_id": int(member.id), "N_N": float(N)}],
            [{"member_id": int(member.id), "FS_min": float(FS), "utilization": 0.0}],
        )
        d = plan.get(int(member.id), {})
        return JointDecision(
            member_id=int(member.id),
            member_group=str(group),
            force_state=str(d.get("force_state", "near_zero")),
            abs_force_N=float(d.get("abs_force_N", abs(float(N)))),
            axial_force_ratio=float(d.get("axial_force_ratio", 0.0)),
            FS_min=float(d.get("FS_min", FS)),
            utilization=float(d.get("utilization", 0.0)),
            recommended_joint_model=str(d.get("recommended_joint_model", "double_lap")),
            required_overlap_mm=float(d.get("required_overlap_mm", cfg.get("detail_model", {}).get("overlap_length_mm", 30.0))),
            joint_area_factor=float(d.get("joint_area_factor", 1.0)),
            secondary_bending_factor=float(d.get("secondary_bending_factor", 1.0)),
            estimated_joint_fs=float(d.get("estimated_joint_fs", FS)),
            geometry_fit_ok=bool(d.get("geometry_fit_ok", True)),
            angle_adjustment_required=bool(d.get("angle_adjustment_required", False)),
            reason=str(d.get("reason", "")),
            compatibility_flags=list(d.get("compatibility_flags", [])),
            symmetry_partner_ids=list(d.get("symmetry_partner_ids", [])),
        )
