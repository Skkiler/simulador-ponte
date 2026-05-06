from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.core.numeric import safe_float
from src.core.safety import risk_from_fs, safety_label
from src.services.geometry_service import GeometryService
from src.services.section_service import SectionService


class PostProcessor:
    """Converte esforços em verificações estruturais e rankings."""

    def __init__(self, section_service: SectionService | None = None) -> None:
        self.sections = section_service or SectionService()

    def check_members(self, cfg: Dict, member_results: List[Dict]) -> List[Dict]:
        mat = cfg["material"]
        detail = cfg.get("detail_model", {})
        primary = set(cfg["analysis"].get("primary_groups", []))
        stabilizers = set(cfg["analysis"].get("stabilizer_groups", []))

        rows: List[Dict] = []

        for r in member_results:
            n = int(float(r["n_sticks"]))
            N = float(r["N_N"])
            L = float(r["L_mm"])
            Iy = float(r["Iy_mm4"])
            Iz = float(r["Iz_mm4"])
            Ky = float(r.get("Ky", 1.0))
            Kz = float(r.get("Kz", 1.0))

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

            Pcr_y = self.sections.euler_buckling_N(mat["E_MPa"], Iy, Ky, L)
            Pcr_z = self.sections.euler_buckling_N(mat["E_MPa"], Iz, Kz, L)
            # Em compressão, emendas e excentricidades reduzem capacidade efetiva.
            Pcr_y *= max(0.55, min(1.2, eta_c))
            Pcr_z *= max(0.55, min(1.2, eta_c))

            Pcr_y_clean = safe_float(Pcr_y, None)
            Pcr_z_clean = safe_float(Pcr_z, None)

            if Pcr_y_clean is None and Pcr_z_clean is None:
                Pcr_min_clean = None
            elif Pcr_y_clean is None:
                Pcr_min_clean = Pcr_z_clean
            elif Pcr_z_clean is None:
                Pcr_min_clean = Pcr_y_clean
            else:
                Pcr_min_clean = min(Pcr_y_clean, Pcr_z_clean)

            cap_t = self.sections.tension_capacity_N(n, mat) * max(0.55, min(1.2, eta_t))
            cap_c = self.sections.compression_capacity_N(n, mat) * max(0.55, min(1.2, eta_c))

            fs_t = cap_t / N if N > 0 else None
            fs_c = cap_c / abs(N) if N < 0 else None
            fs_by = Pcr_y / abs(N) if N < 0 else None
            fs_bz = Pcr_z / abs(N) if N < 0 else None

            fs_t_clean = safe_float(fs_t, None)
            fs_c_clean = safe_float(fs_c, None)
            fs_by_clean = safe_float(fs_by, None)
            fs_bz_clean = safe_float(fs_bz, None)

            if N >= 0:
                governing = "tension_capacity"
                fs_min = fs_t_clean
            else:
                candidates = {
                    "compression_direct": fs_c_clean,
                    "buckling_y": fs_by_clean,
                    "buckling_z": fs_bz_clean,
                }

                valid_candidates = {
                    k: v
                    for k, v in candidates.items()
                    if v is not None
                }

                if valid_candidates:
                    governing, fs_min = min(valid_candidates.items(), key=lambda kv: kv[1])
                else:
                    governing, fs_min = "compression_unchecked", None

            fs_min_clean = safe_float(fs_min, None)

            group = r["group"]

            if group in primary:
                role = "primary"
            elif group in stabilizers:
                role = "stabilizer"
            else:
                role = "secondary"

            tension_only_bracing = bool(
                cfg["bridge"].get("tension_only_bracing_interpretation", True)
            )

            if role == "stabilizer" and N < 0 and tension_only_bracing:
                risk = "STABILIZER_COMPRESSION"
                report_mode = "travamento: compressão deve ser interpretada com cautela"
            else:
                risk = risk_from_fs(fs_min_clean)
                report_mode = governing

            rows.append(
                {
                    **r,
                    "tension_capacity_N": safe_float(cap_t, None),
                    "compression_capacity_N": safe_float(cap_c, None),
                    "Pcr_y_N": Pcr_y_clean,
                    "Pcr_z_N": Pcr_z_clean,
                    "Pcr_min_N": Pcr_min_clean,
                    "FS_tension": fs_t_clean,
                    "FS_compression_direct": fs_c_clean,
                    "FS_buckling_y": fs_by_clean,
                    "FS_buckling_z": fs_bz_clean,
                    "FS_min": fs_min_clean,
                    "FS_min_label": safety_label(fs_min_clean),
                    "governing_mode": governing,
                    "report_mode": report_mode,
                    "member_role": role,
                    "risk_flag": risk,
                    "splice_eff_tension": eta_t,
                    "splice_eff_compression": eta_c,
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
