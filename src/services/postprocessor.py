from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

from src.core.numeric import safe_float
from src.core.safety import risk_from_fs, safety_label
from src.services.geometry_service import GeometryService
from src.services.section_service import SectionService


class PostProcessor:
    """Convert solver forces into structural checks and ranked diagnostics."""

    def __init__(self, section_service: SectionService | None = None) -> None:
        self.sections = section_service or SectionService()

    def check_members(self, cfg: Dict, member_results: List[Dict]) -> List[Dict]:
        mat = cfg["material"]
        detail = cfg.get("detail_model", {})
        primary = set(cfg["analysis"].get("primary_groups", []))
        stabilizers = set(cfg["analysis"].get("stabilizer_groups", []))
        tension_only_bracing = bool(
            cfg.get("bridge", {}).get("tension_only_bracing_interpretation", True)
        )

        imperfection_e = float(detail.get("imperfection_eccentricity_mm", 2.0))
        stick_w = float(mat.get("stick_width_mm", 7.0))
        stick_t = float(mat.get("stick_thickness_mm", 1.5))
        stick_area = max(1.0e-9, stick_w * stick_t)
        sigma_c_default = float(mat.get("compression_capacity_one_stick_N", 0.0)) / stick_area
        sigma_c = max(1.0, float(mat.get("compression_strength_MPa", sigma_c_default)))
        fb = max(1.0, float(mat.get("bending_strength_MPa", 55.0)))

        rows: List[Dict] = []

        for r in member_results:
            n = max(1, int(float(r["n_sticks"])))
            N = float(r["N_N"])
            absN = abs(N)
            L = float(r["L_mm"])
            A = float(r["A_mm2"])
            Iy = float(r["Iy_mm4"])
            Iz = float(r["Iz_mm4"])
            Ky = float(r.get("Ky", 1.0))
            Kz = float(r.get("Kz", 1.0))
            layout = str(r.get("layout", "stacked"))

            eta_t = self.sections.splice_efficiency_factor(
                L_mm=L,
                stick_length_mm=float(mat.get("stick_length_mm", 115.0)),
                overlap_length_mm=float(detail.get("overlap_length_mm", 30.0)),
                model_efficiency=float(
                    (detail.get("joint_efficiency_tension_by_model", {}) or {}).get(
                        str(detail.get("tension_joint_model", "double_lap_reinforced")),
                        1.0,
                    )
                ),
                decay_per_splice=float(detail.get("joint_efficiency_decay_per_splice_tension", 0.03)),
            )
            eta_c = self.sections.splice_efficiency_factor(
                L_mm=L,
                stick_length_mm=float(mat.get("stick_length_mm", 115.0)),
                overlap_length_mm=float(detail.get("overlap_length_mm", 30.0)),
                model_efficiency=float(
                    (detail.get("joint_efficiency_compression_by_model", {}) or {}).get(
                        str(detail.get("compression_joint_model", "double_lap_reinforced")),
                        1.0,
                    )
                ),
                decay_per_splice=float(detail.get("joint_efficiency_decay_per_splice_compression", 0.04)),
            )

            col_y = self.sections.column_capacity_N(
                E_MPa=float(mat["E_MPa"]),
                A_mm2=A,
                I_mm4=Iy,
                K=Ky,
                L_mm=L,
                sigma_c_MPa=sigma_c,
                method="auto",
                eccentricity_mm=imperfection_e if N < 0 else 0.0,
            )
            col_z = self.sections.column_capacity_N(
                E_MPa=float(mat["E_MPa"]),
                A_mm2=A,
                I_mm4=Iz,
                K=Kz,
                L_mm=L,
                sigma_c_MPa=sigma_c,
                method="auto",
                eccentricity_mm=imperfection_e if N < 0 else 0.0,
            )
            cap_col_y = (safe_float(col_y.get("capacity_N"), 0.0) or 0.0) * max(0.55, min(1.2, eta_c))
            cap_col_z = (safe_float(col_z.get("capacity_N"), 0.0) or 0.0) * max(0.55, min(1.2, eta_c))

            cap_t = self.sections.tension_capacity_N(n, mat) * max(0.55, min(1.2, eta_t))
            cap_c = self.sections.compression_capacity_N(n, mat, layout=layout) * max(0.55, min(1.2, eta_c))

            fs_t = cap_t / N if N > 0 else None
            fs_c = cap_c / absN if N < 0 and absN > 1.0e-12 else None
            fs_by = cap_col_y / absN if N < 0 and absN > 1.0e-12 else None
            fs_bz = cap_col_z / absN if N < 0 and absN > 1.0e-12 else None

            r_y = self.sections.radius_of_gyration(Iy, A)
            r_z = self.sections.radius_of_gyration(Iz, A)
            c_z = max(stick_t * 0.5, r_y * math.sqrt(3.0))
            c_y = max(stick_w * 0.5, r_z * math.sqrt(3.0))
            M_imp = absN * imperfection_e if N < 0 else 0.0
            sigma_by = M_imp * c_z / Iy if Iy > 0 else 0.0
            sigma_bz = M_imp * c_y / Iz if Iz > 0 else 0.0

            T_allow = max(1.0e-9, cap_t)
            C_allow = max(1.0e-9, min(cap_c, cap_col_y if cap_col_y > 0 else cap_c, cap_col_z if cap_col_z > 0 else cap_c))
            pcr_y = max(1.0e-9, cap_col_y if cap_col_y > 0 else C_allow)
            pcr_z = max(1.0e-9, cap_col_z if cap_col_z > 0 else C_allow)
            ratio_y = min(0.99, absN / pcr_y)
            ratio_z = min(0.99, absN / pcr_z)
            B1y = min(8.0, 1.0 / max(0.15, 1.0 - ratio_y))
            B1z = min(8.0, 1.0 / max(0.15, 1.0 - ratio_z))

            if N >= 0:
                beam_col_util = (absN / T_allow) + (abs(sigma_by) / fb) + (abs(sigma_bz) / fb)
            else:
                beam_col_util = (absN / C_allow) + B1y * (abs(sigma_by) / fb) + B1z * (abs(sigma_bz) / fb)
            fs_beam_col = (1.0 / beam_col_util) if beam_col_util > 1.0e-12 else None

            fs_t_clean = safe_float(fs_t, None)
            fs_c_clean = safe_float(fs_c, None)
            fs_by_clean = safe_float(fs_by, None)
            fs_bz_clean = safe_float(fs_bz, None)
            fs_beam_col_clean = safe_float(fs_beam_col, None)

            if N >= 0:
                candidates = {
                    "tension_capacity": fs_t_clean,
                    "beam_column_interaction": fs_beam_col_clean,
                }
            else:
                candidates = {
                    "compression_direct": fs_c_clean,
                    "buckling_y": fs_by_clean,
                    "buckling_z": fs_bz_clean,
                    "beam_column_interaction": fs_beam_col_clean,
                }

            valid_candidates = {k: v for k, v in candidates.items() if v is not None}
            if valid_candidates:
                governing, fs_min = min(valid_candidates.items(), key=lambda kv: kv[1])
            else:
                governing, fs_min = "unchecked", None

            fs_min_clean = safe_float(fs_min, None)
            utilization = (1.0 / fs_min_clean) if (fs_min_clean is not None and fs_min_clean > 1.0e-12) else None

            group = r["group"]
            if group in primary:
                role = "primary"
            elif group in stabilizers:
                role = "stabilizer"
            else:
                role = "secondary"

            released_tension_only = bool(r.get("tension_only_released", False))
            tension_only_compressed = bool(role == "stabilizer" and N < 0 and tension_only_bracing)
            design_relevant = not (released_tension_only or tension_only_compressed)
            fs_design = fs_min_clean if design_relevant else None
            utilization_design = utilization if design_relevant else None

            if released_tension_only:
                risk = "TENSION_ONLY_RELEASED"
                report_mode = "tension_only_released"
            elif tension_only_compressed:
                risk = "STABILIZER_COMPRESSION"
                report_mode = "travamento: compressão deve ser interpretada com cautela"
            else:
                risk = risk_from_fs(fs_design)
                report_mode = governing

            rows.append(
                {
                    **r,
                    "tension_capacity_N": safe_float(cap_t, None),
                    "compression_capacity_N": safe_float(cap_c, None),
                    "compression_capacity_column_y_N": safe_float(cap_col_y, None),
                    "compression_capacity_column_z_N": safe_float(cap_col_z, None),
                    "FS_tension": fs_t_clean,
                    "FS_compression_direct": fs_c_clean,
                    "FS_buckling_y": fs_by_clean,
                    "FS_buckling_z": fs_bz_clean,
                    "FS_beam_column": fs_beam_col_clean,
                    "FS_min": fs_min_clean,
                    "FS_min_all_raw": fs_min_clean,
                    "FS_design": fs_design,
                    "FS_min_label": safety_label(fs_min_clean),
                    "FS_design_label": safety_label(fs_design),
                    "governing_mode": governing,
                    "report_mode": report_mode,
                    "member_role": role,
                    "risk_flag": risk,
                    "splice_eff_tension": eta_t,
                    "splice_eff_compression": eta_c,
                    "slenderness_y": col_y.get("slenderness"),
                    "slenderness_z": col_z.get("slenderness"),
                    "column_method_y": col_y.get("method"),
                    "column_method_z": col_z.get("method"),
                    "Pcr_y_N": safe_float(col_y.get("euler_N"), None),
                    "Pcr_z_N": safe_float(col_z.get("euler_N"), None),
                    "utilization": utilization,
                    "utilization_design": utilization_design,
                    "compression_direct_util": (absN / cap_c) if (N < 0 and cap_c > 1.0e-12) else None,
                    "buckling_util_y": (absN / cap_col_y) if (N < 0 and cap_col_y > 1.0e-12) else None,
                    "buckling_util_z": (absN / cap_col_z) if (N < 0 and cap_col_z > 1.0e-12) else None,
                    "tension_util": (absN / cap_t) if (N >= 0 and cap_t > 1.0e-12) else None,
                    "beam_column_util": beam_col_util,
                    "beam_column_B1y": B1y,
                    "beam_column_B1z": B1z,
                    "tension_only_released": released_tension_only,
                    "design_relevant": design_relevant,
                }
            )

        return sorted(
            rows,
            key=lambda x: safe_float(x.get("FS_min"), 1.0e99) or 1.0e99,
        )

    def check_supports(
        self,
        cfg: Dict,
        nodes: List,
        supports: List,
        node_results: List[Dict],
    ) -> List[Dict]:
        node_by_id = {n.id: n for n in nodes}
        support_by_node = {s.node_id: s for s in supports}

        max_reac = float(cfg["support_check"]["allowable_reaction_per_support_node_kgf"]) * 9.80665
        contact_len = float(cfg["support_check"]["contact_length_per_support_node_mm"])
        n_contact = int(cfg["support_check"]["n_contact_sticks_per_support_node"])
        w = float(cfg["material"]["stick_width_mm"])
        area = contact_len * w * n_contact

        rows: List[Dict] = []
        for r in node_results:
            nid = int(r["node_id"])
            if nid not in support_by_node:
                continue

            rz = float(r["Rz_N"])
            n = node_by_id[nid]
            active = bool(r.get("support_active_vertical", False))

            if not active:
                flag = "UPLIFT_NO_CONTACT"
                fs_clean = None
                pressure = 0.0
            else:
                fs_raw = max_reac / abs(rz) if abs(rz) > 1.0e-12 else None
                fs_clean = safe_float(fs_raw, None)
                pressure = abs(rz) / area if area else 0.0
                if fs_clean is None:
                    flag = "OK"
                elif fs_clean < 1.0:
                    flag = "CRITICAL"
                elif fs_clean < 2.0:
                    flag = "LOW_MARGIN"
                else:
                    flag = "OK"

            rows.append(
                {
                    "node_id": nid,
                    "support_group": support_by_node[nid].support_group,
                    "x_mm": n.x,
                    "y_mm": n.y,
                    "reaction_Z_N": rz,
                    "reaction_Z_kgf": rz / 9.80665,
                    "contact_area_mm2": area,
                    "bearing_pressure_MPa": pressure,
                    "allowable_reaction_node_N": max_reac,
                    "FS_support_reaction": fs_clean,
                    "FS_support_reaction_label": safety_label(fs_clean),
                    "support_active_vertical": active,
                    "risk_flag": flag,
                }
            )
        return rows

    @staticmethod
    def export(
        member_checks: List[Dict],
        support_checks: List[Dict],
        out_dir: str | Path,
    ) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        GeometryService.write_csv(out / "member_failure_checks.csv", member_checks)
        GeometryService.write_csv(out / "support_reaction_checks.csv", support_checks)
