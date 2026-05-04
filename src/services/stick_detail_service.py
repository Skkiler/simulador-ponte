from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.domain.models import Member, Node
from src.services.geometry_service import GeometryService
from src.services.section_service import SectionService


def safe_float(value: Any, default: float | None = None) -> float | None:
    """
    Converte para float sem quebrar quando o valor vier como None, string vazia,
    NaN, infinito ou texto.

    Use isto sempre que um campo puder vir de CSV, JSON ou pós-processamento.
    """
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


def safe_sort_key(value: Any, default: float = 1.0e99) -> float:
    """
    Chave segura para ordenação crescente.
    Valores ausentes ou inválidos vão para o final.
    """
    v = safe_float(value, None)
    return default if v is None else v


def safety_label(value: Any) -> str:
    """
    Representação humana para fator de segurança.
    """
    v = safe_float(value, None)

    if v is None:
        return "sem solicitação"

    return f"{v:.3f}"


def risk_from_fs(value: Any) -> str:
    """
    Classificação simples por fator de segurança.
    """
    fs = safe_float(value, None)

    if fs is None:
        return "OK"

    if fs < 1.0:
        return "CRITICAL"

    if fs < 2.0:
        return "LOW_MARGIN"

    return "OK"


class StickDetailService:
    """
    Modelo rápido peça-a-peça: palitos, sobreposições, cola, massa e recomendações.

    Este serviço não faz FEM. Ele expande cada membro estrutural equivalente em
    peças de palito, estima cortes, sobreposições, áreas coladas, tensões médias,
    massa e recomendações construtivas.
    """

    def __init__(self, section_service: SectionService | None = None) -> None:
        self.sections = section_service or SectionService()

    @staticmethod
    def _unit_vector(ni: Node, nj: Node) -> Tuple[float, float, float, float]:
        dx = nj.x - ni.x
        dy = nj.y - ni.y
        dz = nj.z - ni.z

        L = math.sqrt(dx * dx + dy * dy + dz * dz)

        if L <= 0:
            return 0.0, 0.0, 0.0, 0.0

        return dx / L, dy / L, dz / L, L

    @staticmethod
    def _piece_intervals(
        L: float,
        stick_len: float,
        overlap: float,
    ) -> List[Tuple[float, float, float]]:
        """
        Divide um membro de comprimento L em peças de palito.

        Retorna lista de:
            s0, s1, comprimento_de_corte

        A peça seguinte começa antes do fim da anterior quando há sobreposição.
        """
        if L <= 0:
            return []

        if stick_len <= 0:
            raise ValueError("stick_length_mm precisa ser maior que zero.")

        if L <= stick_len:
            return [(0.0, L, L)]

        overlap = max(0.0, min(overlap, stick_len * 0.75))
        step = max(1.0e-6, stick_len - overlap)

        out: List[Tuple[float, float, float]] = []
        s0 = 0.0

        while s0 < L - 1.0e-9:
            s1 = min(L, s0 + stick_len)
            out.append((s0, s1, s1 - s0))

            if s1 >= L - 1.0e-9:
                break

            s0 += step

        return out

    @staticmethod
    def _pack_cuts_best_fit(
        cuts: List[float],
        blank_length: float,
        kerf: float = 1.0,
    ) -> List[List[float]]:
        """
        Empacota cortes em palitos brutos usando heurística best-fit decrescente.
        Não é otimização exata, mas é rápida e suficiente para plano preliminar.
        """
        clean_cuts = [
            float(c)
            for c in cuts
            if safe_float(c, None) is not None and float(c) > 0
        ]

        clean_cuts = sorted(clean_cuts, reverse=True)

        bins: List[List[float]] = []
        remaining: List[float] = []

        for c in clean_cuts:
            best_i = None
            best_rem = None

            for i, rem in enumerate(remaining):
                need = c + (kerf if bins[i] else 0.0)

                if need <= rem + 1.0e-9:
                    nr = rem - need

                    if best_rem is None or nr < best_rem:
                        best_i = i
                        best_rem = nr

            if best_i is None:
                bins.append([c])
                remaining.append(max(0.0, blank_length - c))
            else:
                bins[best_i].append(c)
                remaining[best_i] = best_rem if best_rem is not None else remaining[best_i]

        return bins

    def analyze(
        self,
        cfg: Dict,
        nodes: List[Node],
        members: List[Member],
        member_results: List[Dict],
        member_checks: List[Dict],
        out_dir: str | Path,
    ) -> Dict:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        mat = cfg["material"]
        detail = cfg.get("detail_model", {})

        stick_len = float(mat.get("stick_length_mm", 120.0))
        stick_w = float(mat.get("stick_width_mm", 7.0))
        stick_t = float(mat.get("stick_thickness_mm", 1.5))
        stick_mass = float(mat.get("stick_mass_g", 1.4))

        overlap = float(detail.get("overlap_length_mm", 30.0))
        glue_tau = float(detail.get("glue_shear_strength_MPa", 3.5))
        glue_sf = float(detail.get("default_joint_safety_factor", 2.0))
        glue_spread = float(detail.get("glue_spread_g_per_m2", 160.0))
        glue_eff = float(detail.get("glue_mass_efficiency", 0.65))
        imperfection_e = float(detail.get("imperfection_eccentricity_mm", 2.0))
        waste = float(detail.get("construction_waste_factor", 0.08))
        kerf = float(detail.get("saw_kerf_mm", 1.0))
        reinforce_if = float(detail.get("reinforce_if_fs_lt", 2.0))
        remove_if = float(detail.get("allow_recommend_removal_if_fs_gt", 8.0))
        tension_only = bool(detail.get("tension_only_stabilizers", True))

        node_by_id = {n.id: n for n in nodes}
        res_by = {int(r["member_id"]): r for r in member_results}
        chk_by = {int(r["member_id"]): r for r in member_checks}

        stabilizers = set(cfg.get("analysis", {}).get("stabilizer_groups", []))

        stick_rows: List[Dict] = []
        joint_rows: List[Dict] = []
        member_rows: List[Dict] = []
        reinf_rows: List[Dict] = []

        cut_counter: Counter = Counter()
        cut_lengths: List[float] = []

        total_glue_area = 0.0
        total_pieces = 0
        total_cut = 0.0

        for m in members:
            ni = node_by_id[m.i]
            nj = node_by_id[m.j]

            ux, uy, uz, L = self._unit_vector(ni, nj)

            if L <= 0:
                continue

            res = res_by.get(m.id, {})
            chk = chk_by.get(m.id, {})

            N = safe_float(res.get("N_N"), 0.0) or 0.0
            n_lanes = max(1, int(m.n_sticks))

            layout_cfg = cfg.get("section_layout_by_group", {}).get(
                m.group,
                {"layout": "stacked"},
            )

            sec = self.sections.composite_section(n_lanes, mat, layout_cfg)

            per_lane = N / n_lanes
            piece_area = stick_w * stick_t
            per_sigma = per_lane / piece_area if piece_area else 0.0

            intervals = self._piece_intervals(L, stick_len, overlap)

            r_y = self.sections.radius_of_gyration(sec["Iy"], sec["A"])
            r_z = self.sections.radius_of_gyration(sec["Iz"], sec["A"])

            slender_y = m.Ky * L / r_y if r_y else None
            slender_z = m.Kz * L / r_z if r_z else None

            M_imp = abs(N) * imperfection_e if N < 0 else 0.0

            c_y = sec.get("width_mm", stick_w) / 2.0
            c_z = sec.get("thickness_mm", stick_t) / 2.0

            sig_by = M_imp * c_z / sec["Iy"] if sec["Iy"] else 0.0
            sig_bz = M_imp * c_y / sec["Iz"] if sec["Iz"] else 0.0

            sigma_axial_member = N / sec["A"] if sec["A"] else 0.0
            sig_comb = abs(sigma_axial_member) + abs(sig_by) + abs(sig_bz)

            member_glue = 0.0
            joint_fs_values: List[float] = []

            for lane in range(1, n_lanes + 1):
                prev_id = None
                prev_end = None

                for piece_index, (s0, s1, cut_len) in enumerate(intervals, 1):
                    sid = f"M{m.id:03d}-L{lane:02d}-P{piece_index:02d}"

                    x0 = ni.x + ux * s0
                    y0 = ni.y + uy * s0
                    z0 = ni.z + uz * s0

                    x1 = ni.x + ux * s1
                    y1 = ni.y + uy * s1
                    z1 = ni.z + uz * s1

                    total_pieces += 1
                    total_cut += cut_len

                    cut_lengths.append(cut_len)
                    cut_counter[round(cut_len, 1)] += 1

                    stick_rows.append(
                        {
                            "stick_id": sid,
                            "member_id": m.id,
                            "member_group": m.group,
                            "lane": lane,
                            "piece_index": piece_index,
                            "s0_mm": s0,
                            "s1_mm": s1,
                            "cut_length_mm": cut_len,
                            "x0_mm": x0,
                            "y0_mm": y0,
                            "z0_mm": z0,
                            "x1_mm": x1,
                            "y1_mm": y1,
                            "z1_mm": z1,
                            "N_piece_N": per_lane,
                            "sigma_axial_piece_MPa": per_sigma,
                            "member_state": "tension" if N >= 0 else "compression",
                            "mass_g": stick_mass * cut_len / stick_len,
                        }
                    )

                    if prev_id is not None and prev_end is not None:
                        overlap_actual = max(0.0, prev_end - s0)
                        glue_area = overlap_actual * stick_w

                        if glue_area > 0:
                            glue_shear = abs(per_lane) / glue_area
                        else:
                            glue_shear = None

                        glue_allow = glue_tau / glue_sf if glue_sf > 0 else None

                        if glue_shear is None or glue_shear <= 0 or glue_allow is None:
                            fs_glue = None
                        else:
                            fs_glue = glue_allow / glue_shear

                        fs_glue_clean = safe_float(fs_glue, None)

                        if fs_glue_clean is not None:
                            joint_fs_values.append(fs_glue_clean)

                        member_glue += glue_area
                        total_glue_area += glue_area

                        joint_rows.append(
                            {
                                "joint_id": f"J-M{m.id:03d}-L{lane:02d}-P{piece_index-1:02d}-{piece_index:02d}",
                                "member_id": m.id,
                                "member_group": m.group,
                                "lane": lane,
                                "piece_a": prev_id,
                                "piece_b": sid,
                                "joint_type": "lap_overlap",
                                "overlap_length_mm": overlap_actual,
                                "glue_area_mm2": glue_area,
                                "force_transfer_N": abs(per_lane),
                                "glue_shear_MPa": glue_shear,
                                "glue_allow_design_MPa": glue_allow,
                                "FS_glue_shear": fs_glue_clean,
                                "FS_glue_shear_label": safety_label(fs_glue_clean),
                                "risk_flag": risk_from_fs(fs_glue_clean),
                            }
                        )

                    prev_id = sid
                    prev_end = s1

            glue_mass = (member_glue / 1_000_000.0) * glue_spread / max(glue_eff, 1.0e-6)

            fs_min_global = safe_float(chk.get("FS_min"), None)
            fs_min_global_label = safety_label(fs_min_global)

            role = chk.get("member_role", "secondary")
            gov = chk.get("governing_mode", "")
            report_mode = chk.get("report_mode", gov)

            if role == "stabilizer" and tension_only:
                action = (
                    "manter como travamento/tension-only; "
                    "não dimensionar como coluna comprimida"
                )
                priority = "interpretation"
            elif fs_min_global is not None and fs_min_global < reinforce_if:
                if "buckling" in str(gov):
                    action = (
                        "reforçar: aumentar inércia, usar seção caixa/espaçada "
                        "ou reduzir comprimento livre"
                    )
                else:
                    action = (
                        "reforçar: adicionar palitos contínuos ou aumentar "
                        "sobreposição/talas"
                    )
                priority = "high"
            elif fs_min_global is not None and fs_min_global > remove_if and role != "primary":
                action = "avaliar remoção/redução: baixa solicitação relativa"
                priority = "low"
            else:
                action = "manter"
                priority = "normal"

            if priority != "normal":
                reinf_rows.append(
                    {
                        "member_id": m.id,
                        "group": m.group,
                        "role": role,
                        "N_N": N,
                        "FS_min": fs_min_global,
                        "FS_min_label": fs_min_global_label,
                        "governing_mode": gov,
                        "report_mode": report_mode,
                        "suggested_action": action,
                        "priority": priority,
                    }
                )

            fs_min_glue = min(joint_fs_values) if joint_fs_values else None

            member_rows.append(
                {
                    "member_id": m.id,
                    "group": m.group,
                    "role": role,
                    "n_lanes_sticks": n_lanes,
                    "pieces_per_lane": len(intervals),
                    "total_piece_count": len(intervals) * n_lanes,
                    "member_length_mm": L,
                    "layout": sec.get("layout"),
                    "section_A_mm2": sec["A"],
                    "section_Iy_mm4": sec["Iy"],
                    "section_Iz_mm4": sec["Iz"],
                    "section_J_mm4_est": sec["J"],
                    "radius_y_mm": r_y,
                    "radius_z_mm": r_z,
                    "slenderness_y": slender_y,
                    "slenderness_z": slender_z,
                    "N_member_N": N,
                    "N_per_lane_N": per_lane,
                    "sigma_axial_member_MPa": sigma_axial_member,
                    "sigma_axial_piece_MPa": per_sigma,
                    "M_imperfection_Nmm": M_imp,
                    "sigma_bending_est_MPa": max(abs(sig_by), abs(sig_bz)),
                    "sigma_combined_est_MPa": sig_comb,
                    "glue_area_total_mm2": member_glue,
                    "glue_mass_est_g": glue_mass,
                    "FS_min_global": fs_min_global,
                    "FS_min_global_label": fs_min_global_label,
                    "FS_min_glue": fs_min_glue,
                    "FS_min_glue_label": safety_label(fs_min_glue),
                    "governing_mode_global": gov,
                    "report_mode_global": report_mode,
                    "suggested_action": action,
                }
            )

        cutting_rows = [
            {
                "cut_length_mm": k,
                "quantity": v,
                "total_length_mm": k * v,
            }
            for k, v in sorted(cut_counter.items(), reverse=True)
        ]

        bins = self._pack_cuts_best_fit(cut_lengths, stick_len, kerf)

        blank_plan: List[Dict] = []

        for idx, cuts in enumerate(bins, 1):
            used = sum(cuts) + max(0, len(cuts) - 1) * kerf

            blank_plan.append(
                {
                    "blank_stick_index": idx,
                    "cuts_mm": ";".join(f"{c:.1f}" for c in cuts),
                    "n_cuts": len(cuts),
                    "used_length_mm_including_kerf": used,
                    "waste_length_mm": max(0.0, stick_len - used),
                }
            )

        blank = len(bins)
        extra = math.ceil(blank * waste)
        total = blank + extra

        piece_mass = sum(float(r["mass_g"]) for r in stick_rows)
        glue_mass = (total_glue_area / 1_000_000.0) * glue_spread / max(glue_eff, 1.0e-6)
        total_mass = total * stick_mass + glue_mass

        limit = float(mat.get("mass_limit_g", 1000.0))

        summary = {
            "total_members": len(member_rows),
            "total_piece_instances": total_pieces,
            "total_cut_length_mm": total_cut,
            "estimated_blank_sticks_needed": blank,
            "waste_factor": waste,
            "extra_sticks_for_waste": extra,
            "estimated_total_sticks_with_waste": total,
            "estimated_piece_mass_g_without_waste_scaling": piece_mass,
            "estimated_glue_area_mm2": total_glue_area,
            "estimated_glue_mass_g": glue_mass,
            "estimated_total_mass_g": total_mass,
            "mass_limit_g": limit,
            "mass_margin_g": limit - total_mass,
            "glue_shear_strength_MPa": glue_tau,
            "glue_safety_factor": glue_sf,
        }

        weakest = sorted(
            member_rows,
            key=lambda r: safe_sort_key(r.get("FS_min_global")),
        )[:30]

        glue_weak = sorted(
            joint_rows,
            key=lambda r: safe_sort_key(r.get("FS_glue_shear")),
        )[:30]

        exports = {
            "stick_pieces.csv": stick_rows,
            "glue_joints.csv": joint_rows,
            "member_detail_checks.csv": member_rows,
            "cutting_list.csv": cutting_rows,
            "blank_cut_plan.csv": blank_plan,
            "reinforcement_suggestions.csv": reinf_rows,
            "weakest_members.csv": weakest,
            "weakest_glue_joints.csv": glue_weak,
        }

        for filename, rows in exports.items():
            GeometryService.write_csv(out / filename, rows)

        (out / "detailed_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return {
            "stick_pieces": stick_rows,
            "glue_joints": joint_rows,
            "member_detail_checks": member_rows,
            "cutting_list": cutting_rows,
            "blank_cut_plan": blank_plan,
            "reinforcement_suggestions": reinf_rows,
            "weakest_members": weakest,
            "weakest_glue_joints": glue_weak,
            "summary": summary,
        }