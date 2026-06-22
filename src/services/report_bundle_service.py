from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.core.numeric import safe_float
from src.services.geometry_service import GeometryService
from src.services.section_service import SectionService


class ReportBundleService:
    """Generate consolidated final report bundle under outputs/final_report."""

    @staticmethod
    def _solver_regular(status: Any) -> bool:
        return str(status or "").split("|", 1)[0] == "regular"

    @staticmethod
    def _iter_stage_rows(optimization: Dict[str, Any] | None) -> List[Dict[str, Any]]:
        if not optimization:
            return []
        ordered_keys = [
            "s8_final_validation",
            "s7_fabrication",
            "s6_topology",
            "s5_member_sizing",
            "s4_geometry_refinement",
            "s3_multi_loadcase",
            "s2_fast_screening",
            "stage4",
            "stage3",
            "stage2",
            "stage1",
        ]
        rows: List[Dict[str, Any]] = []
        for key in ordered_keys:
            rows.extend(list((optimization or {}).get(key, []) or []))
        return rows

    def _verdict(self, cfg: Dict[str, Any], metrics: Dict[str, Any], optimization: Dict[str, Any] | None) -> tuple[str, str, List[str]]:
        analysis = cfg.get("analysis", {}) or {}
        pred = safe_float(metrics.get("predicted_breaking_load_kgf"), 0.0) or 0.0
        break_target = float(analysis.get("acceptance_min_design_breaking_load_kgf", 80.0))
        min_primary = safe_float(
            metrics.get("min_fs_member_design"),
            safe_float(metrics.get("min_fs_primary"), 0.0),
        ) or 0.0
        min_primary_target = float(analysis.get("acceptance_min_primary_fs", 1.05))
        min_support = safe_float(metrics.get("min_support_fs"), None)
        min_support_target = float(analysis.get("acceptance_min_support_fs", 1.0))
        min_glue = safe_float(
            metrics.get("min_fs_glue"),
            safe_float(metrics.get("min_glue_fs"), None),
        )
        min_glue_target = float(analysis.get("acceptance_min_glue_fs", 1.5))
        comp_ok = bool(metrics.get("competition_mass_compliant", metrics.get("mass_compliant", False)))
        solver_ok = self._solver_regular(metrics.get("solver_status"))
        eq_ok = bool(metrics.get("equilibrium_ok", True))

        failures: List[str] = []
        if not solver_ok:
            failures.append("solver irregular")
        if not eq_ok:
            failures.append("equilíbrio não atendido")
        if not comp_ok:
            failures.append("massa competitiva acima do limite")
        if pred < break_target:
            failures.append(f"ruptura prevista {pred:.2f} < {break_target:.2f} kgf")
        if min_primary < min_primary_target:
            failures.append(f"FS primário {min_primary:.3f} < {min_primary_target:.3f}")
        if min_support is not None and min_support < min_support_target:
            failures.append(f"FS apoio {min_support:.3f} < {min_support_target:.3f}")
        if min_glue is not None and min_glue < min_glue_target:
            failures.append(f"FS cola {min_glue:.3f} < {min_glue_target:.3f}")
        as_built_intersections = int(metrics.get("as_built_interpenetration_count", 0) or 0)
        if as_built_intersections > 0:
            failures.append(f"interpenetrações volumétricas as-built: {as_built_intersections}")

        # O relatório nominal e o funil S8 podem usar bases diferentes.
        # Para evitar contradição, o relatório final também considera o S8
        # multi-loadcase quando disponível.
        s8_rows = list((optimization or {}).get("s8_final_validation", []) or [])
        if s8_rows:
            s8 = s8_rows[0]
            s8_verdict = str(s8.get("verdict", "")).strip().upper()
            s8_break = safe_float(s8.get("predicted_breaking_load_kgf"), None)
            s8_failed = str(s8.get("failed_restriction", "") or "")
            s8_stage = str(s8.get("validation_stage", "pre_detail")).strip().lower()
            if s8_verdict == "REPROVADA" and not s8_stage.startswith("pre_detail"):
                if s8_break is not None:
                    failures.append(
                        f"ruptura design multi-loadcase {s8_break:.2f} < {break_target:.2f} kgf"
                    )
                elif s8_failed:
                    failures.append(f"S8 multi-loadcase: {s8_failed}")

        if not failures:
            advisories: List[str] = []
            target_fs = safe_float(analysis.get("target_min_fs"), None)
            if target_fs is not None and min_primary < target_fs:
                advisories.append(f"FS mínimo de membro {min_primary:.3f} abaixo do alvo recomendado {target_fs:.3f}")
            mass_margin = (safe_float(metrics.get("mass_limit_effective_g"), 1000.0) or 1000.0) - (safe_float(metrics.get("competition_mass_g"), 0.0) or 0.0)
            fabrication_reserve = safe_float((cfg.get("planner", {}).get("local_sizing", {}) or {}).get("mass_reserve_for_fabrication_g"), 25.0) or 25.0
            if mass_margin < fabrication_reserve:
                advisories.append(f"margem de massa {mass_margin:.2f} g abaixo da reserva prática de fabricação {fabrication_reserve:.2f} g")
            frame_status = str(metrics.get("frame3dd_status", "") or "")
            if frame_status and frame_status not in {"ok", "regular", "passed"}:
                advisories.append(f"checagem Frame3DD não validou equivalência do modelo ({frame_status})")
            if advisories:
                return "APROVADA COM RESSALVAS", "Atende aos critérios mínimos configurados, porém: " + "; ".join(advisories) + ".", failures
            return "APROVADA", "Atende aos critérios de ruptura, massa, S8 multi-loadcase e regularidade do solver.", failures

        has_feasible = any(bool(r.get("feasible")) for r in self._iter_stage_rows(optimization))

        if has_feasible:
            return "REPROVADA", "Melhor candidato final não cumpriu todos os critérios de aceitação.", failures
        return "NENHUMA SOLUÇÃO VIÁVEL", "Nenhum candidato cumpriu simultaneamente os critérios mínimos.", failures

    @staticmethod
    def _top_critical_members(member_checks: List[Dict[str, Any]], top_k: int = 15) -> List[Dict[str, Any]]:
        ordered = sorted(
            list(member_checks or []),
            key=lambda r: safe_float(r.get("FS_design", r.get("FS_min")), 1.0e99) or 1.0e99,
        )
        rows: List[Dict[str, Any]] = []
        for r in ordered[:top_k]:
            fs_val = safe_float(r.get("FS_design", r.get("FS_min")), None)
            util = safe_float(r.get("utilization_design", r.get("utilization")), None)
            rows.append(
                {
                    "member_id": r.get("member_id"),
                    "group": r.get("group"),
                    "role": r.get("member_role"),
                    "N_N": r.get("N_N"),
                    "state": r.get("state"),
                    "n_sticks": r.get("n_sticks"),
                    "layout": r.get("layout"),
                    "governing_mode": r.get("governing_mode"),
                    "FS_min": fs_val,
                    "utilization": util,
                    "recommended_action": r.get("risk_flag"),
                }
            )
        return rows

    @staticmethod
    def _candidate_ranking(optimization: Dict[str, Any] | None) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not optimization:
            return rows
        ordered_keys = [
            "s8_final_validation",
            "s7_fabrication",
            "s6_topology",
            "s5_member_sizing",
            "s4_geometry_refinement",
            "s3_multi_loadcase",
            "s2_fast_screening",
            "stage4",
            "stage3",
            "stage2",
            "stage1",
        ]
        for stage_name in ordered_keys:
            for r in optimization.get(stage_name, []) or []:
                rows.append(
                    {
                        "stage": stage_name,
                        "candidate_id": r.get("candidate_id"),
                        "feasible": r.get("feasible"),
                        "score": r.get("score", r.get("objective")),
                        "predicted_breaking_load_kgf": r.get(
                            "predicted_breaking_load_kgf",
                            r.get("predicted_breaking_load_proxy_kgf"),
                        ),
                        "competition_mass_g": r.get("competition_mass_g", r.get("mass_g", r.get("dead_weight_proxy_g"))),
                        "min_fs_primary": r.get("min_fs_primary", r.get("min_fs_preliminary")),
                        "min_fs_design": r.get("min_fs_design", r.get("min_fs_design_proxy", r.get("min_fs_all"))),
                        "solver_status": r.get("solver_status", "regular" if r.get("solver_regular") else "unknown"),
                    }
                )
        rows.sort(
            key=lambda r: (
                -(safe_float(r.get("predicted_breaking_load_kgf"), 0.0) or 0.0),
                -(safe_float(r.get("score"), -1.0e99) or -1.0e99),
            )
        )
        return rows

    @staticmethod
    def _load_contact_audit_rows(cfg: Dict[str, Any], optimization: Dict[str, Any] | None) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        s8_rows = list((optimization or {}).get("s8_final_validation", []) or [])
        if s8_rows:
            for c in (s8_rows[0].get("case_metrics") or []):
                case = str(c.get("case") or "")
                if case in {"center", "single_plate_center", "crown_contact", "torsion_60_40", "torsion_70_30", "torsion_80_20", "left_offset", "right_offset"}:
                    interp = {
                        "center": "carga distribuída pela plataforma/placa prevista no modelo",
                        "single_plate_center": "placa rígida centrada tocando múltiplas estações",
                        "crown_contact": "contato local no ponto mais alto; usar se não houver plataforma rígida",
                        "left_offset": "placa deslocada longitudinalmente para a esquerda",
                        "right_offset": "placa deslocada longitudinalmente para a direita",
                    }.get(case)
                    if interp is None and case.startswith("torsion_"):
                        interp = f"placa com assimetria lateral {case.replace('torsion_', '').replace('_', '/')}"
                    rows.append(
                        {
                            "case": case,
                            "predicted_breaking_load_kgf": c.get("predicted_breaking_load_proxy_kgf"),
                            "min_fs_design": c.get("min_fs_design"),
                            "max_displacement_mm": c.get("max_displacement_proxy_mm"),
                            "load_path_score": c.get("load_path_score"),
                            "governing_member_id": c.get("governing_member_id"),
                            "governing_member_group": c.get("governing_member_group"),
                            "governing_mode": c.get("governing_mode"),
                            "support_reactions": c.get("support_reactions"),
                            "interpretation": interp or "caso auxiliar",
                        }
                    )
        return rows


    @staticmethod
    def _section_layout_audit_rows(cfg: Dict[str, Any], member_checks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Summarize whether the chosen stick layouts are buildable as modeled.

        The structural solver sees only equivalent A/I/J values at the member
        centerline.  This audit exposes the hidden assumptions behind those
        equivalent sections: stick orientation, local y/z spacing, centroid
        offset and whether a section is tension-dominant or compression-critical.
        """
        mat = cfg.get("material", {}) or {}
        layouts = cfg.get("section_layout_by_group", {}) or {}
        detail = cfg.get("detail_model", {}) or {}
        sectioner = SectionService()

        grouped: Dict[tuple, Dict[str, Any]] = {}
        for chk in member_checks or []:
            group = str(chk.get("group", ""))
            n = int(safe_float(chk.get("n_sticks"), 1) or 1)
            layout_name = str(chk.get("layout", "")) or str((layouts.get(group, {}) or {}).get("layout", "stacked"))
            key = (group, n, layout_name)
            row = grouped.setdefault(
                key,
                {
                    "group": group,
                    "layout": layout_name,
                    "n_sticks": n,
                    "member_ids_example": [],
                    "compression_members": 0,
                    "tension_members": 0,
                    "max_abs_force_N": 0.0,
                    "min_FS_design": None,
                    "min_FS_global": None,
                },
            )
            if chk.get("member_id") is not None and len(row["member_ids_example"]) < 10:
                row["member_ids_example"].append(str(chk.get("member_id")))
            force = safe_float(chk.get("N_N"), 0.0) or 0.0
            if force < -1.0e-9:
                row["compression_members"] += 1
            elif force > 1.0e-9:
                row["tension_members"] += 1
            row["max_abs_force_N"] = max(float(row["max_abs_force_N"]), abs(float(force)))
            fs_d = safe_float(chk.get("FS_design"), None)
            fs_g = safe_float(chk.get("FS_min"), None)
            if fs_d is not None and (row["min_FS_design"] is None or fs_d < row["min_FS_design"]):
                row["min_FS_design"] = float(fs_d)
            if fs_g is not None and (row["min_FS_global"] is None or fs_g < row["min_FS_global"]):
                row["min_FS_global"] = float(fs_g)

        rows: List[Dict[str, Any]] = []
        for (group, n, layout_name), row in grouped.items():
            layout_cfg = dict((layouts.get(group, {}) or {}))
            layout_cfg.setdefault("layout", layout_name)
            layout_cfg.setdefault("composite_action", detail.get("composite_action", {}))
            sec = sectioner.composite_section(n, mat, layout_cfg)
            # Audit simple 2D cross-section intersections.  Palitos podem tocar,
            # mas não podem ocupar o mesmo volume no corte y/z.  Isso captura o
            # erro antigo em que contact_box/tee3 aumentavam I por interpenetração
            # de retângulos, o que não é montável com palitos íntegros.
            overlap_pairs = 0
            overlap_area_mm2 = 0.0
            positions = list(sec.get("stick_positions_yz", []) or [])
            y_dims = list(sec.get("stick_width_y_mm_by_lane", []) or [])
            z_dims = list(sec.get("stick_height_z_mm_by_lane", []) or [])
            rects = []
            for idx, yz in enumerate(positions):
                if not isinstance(yz, (list, tuple)) or len(yz) < 2:
                    continue
                yd = safe_float(y_dims[idx] if idx < len(y_dims) else sec.get("stick_width_y_mm"), 0.0) or 0.0
                zd = safe_float(z_dims[idx] if idx < len(z_dims) else sec.get("stick_height_z_mm"), 0.0) or 0.0
                y = safe_float(yz[0], 0.0) or 0.0
                z = safe_float(yz[1], 0.0) or 0.0
                rects.append((y - 0.5 * yd, y + 0.5 * yd, z - 0.5 * zd, z + 0.5 * zd))
            for i in range(len(rects)):
                for j in range(i + 1, len(rects)):
                    a = rects[i]
                    b = rects[j]
                    oy = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
                    oz = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
                    if oy > 1.0e-9 and oz > 1.0e-9:
                        overlap_pairs += 1
                        overlap_area_mm2 += oy * oz
            cy = safe_float(sec.get("centroid_y_mm"), 0.0) or 0.0
            cz = safe_float(sec.get("centroid_z_mm"), 0.0) or 0.0
            ry = SectionService.radius_of_gyration(float(sec.get("Iy", 0.0) or 0.0), float(sec.get("A", 0.0) or 0.0))
            rz = SectionService.radius_of_gyration(float(sec.get("Iz", 0.0) or 0.0), float(sec.get("A", 0.0) or 0.0))
            centroid_offset = (cy * cy + cz * cz) ** 0.5
            compression_dominant = int(row["compression_members"]) > 0 and int(row["compression_members"]) >= int(row["tension_members"])

            warnings: List[str] = []
            construction_note = "seguir orientação do gabarito; não afastar palitos sem peça de ligação modelada"
            if bool(sec.get("laced_box_demoted_to_contact")):
                warnings.append("laced_box solicitado foi rebaixado para contact_box: não há lacing/talas extras modelados")
                construction_note = "montar caixa apenas com palitos longitudinais em contato face/lado; não afastar lanes sem palitos de ligação contabilizados"
            if group == "bottom_chord" and n == 1:
                construction_note = "banzo inferior atua majoritariamente como tirante; manter continuidade e emendas bem taladas"
                if compression_dominant:
                    warnings.append("banzo inferior simples com compressão detectada; revisar caso torsional")
            if str(layout_name).lower() in {"tee3", "laminated2"}:
                warnings.append("pedido de box subpreenchido foi convertido para laminação conectada; não é caixa")
                construction_note = "montar face-a-face/tee conforme posições locais; não afastar como caixa"
            if str(layout_name).lower() in {"box", "contact_box"} and n >= 5 and n % 2 == 1:
                warnings.append("box com número ímpar usa reforço central balanceado; não mover o palito extra para um canto")
                construction_note = "montar no gabarito exatamente na posição y/z listada; mover o palito extra altera o momento de inércia"
            if overlap_pairs > 0:
                warnings.append(f"interpenetração física detectada: {overlap_pairs} pares, área {overlap_area_mm2:.2f} mm²")
            if centroid_offset > 0.5:
                warnings.append(f"centroide local deslocado {centroid_offset:.2f} mm; controlar excentricidade na colagem")
            if compression_dominant and min(ry, rz) < 1.0:
                warnings.append("raio de giração baixo para membro comprimido")

            out = dict(row)
            out.update(
                {
                    "stick_orientation": sec.get("stick_orientation"),
                    "section_A_mm2": sec.get("A"),
                    "section_Iy_mm4": sec.get("Iy"),
                    "section_Iz_mm4": sec.get("Iz"),
                    "section_Icrit_mm4": min(float(sec.get("Iy", 0.0) or 0.0), float(sec.get("Iz", 0.0) or 0.0)),
                    "radius_y_mm": ry,
                    "radius_z_mm": rz,
                    "centroid_y_mm": cy,
                    "centroid_z_mm": cz,
                    "centroid_offset_mm": centroid_offset,
                    "eta_I": sec.get("eta_I"),
                    "section_overlap_pair_count": overlap_pairs,
                    "section_overlap_area_mm2": overlap_area_mm2,
                    "local_stick_positions_yz": json.dumps(sec.get("stick_positions_yz", []), ensure_ascii=False),
                    "section_connection_model": sec.get("section_connection_model"),
                    "requested_box_extra_stick_strategy": sec.get("requested_box_extra_stick_strategy"),
                    "laced_box_demoted_to_contact": sec.get("laced_box_demoted_to_contact"),
                    "warning": "; ".join(warnings) if warnings else "OK",
                    "construction_note": construction_note,
                    "member_ids_example": ";".join(row["member_ids_example"]),
                }
            )
            rows.append(out)

        rows.sort(
            key=lambda r: (
                str(r.get("warning")) == "OK",
                safe_float(r.get("min_FS_design"), safe_float(r.get("min_FS_global"), 1.0e9)) or 1.0e9,
                str(r.get("group")),
            )
        )
        return rows

    @staticmethod
    def _write_section_layout_audit(out: Path, rows: List[Dict[str, Any]]) -> None:
        GeometryService.write_csv(out / "section_layout_audit.csv", rows)
        table = "\n".join(
            f"| {r.get('group')} | {r.get('n_sticks')} | {r.get('layout')} | {r.get('stick_orientation')} | "
            f"{safe_float(r.get('section_Iy_mm4'), None):.1f} | {safe_float(r.get('section_Iz_mm4'), None):.1f} | "
            f"{safe_float(r.get('centroid_offset_mm'), None):.2f} | {r.get('warning')} | {r.get('construction_note')} |"
            for r in rows[:30]
        ) or "| — | — | — | — | — | — | — | — | — |"
        (out / "08_auditoria_secao_e_realismo.md").write_text(
            f"""# Auditoria de seção, posição dos palitos e realismo construtivo

Este relatório existe porque a imagem 3D mostra apenas as linhas centrais dos membros. A resistência, porém, depende da seção composta: quantidade de palitos, orientação (`edge`/`flat`), espaçamento local, ação composta da cola e centroide real da seção.

## Leitura prática
- `Iy` e `Iz` são os momentos de inércia usados no cálculo de flambagem/beam-column. O menor deles costuma controlar a flambagem.
- `centroid_offset_mm` indica se a seção local ficou excêntrica. Se esse valor aparecer alto, a peça precisa ser montada exatamente como modelada ou recalculada.
- Banzos inferiores finos podem ser aceitáveis quando trabalham como tirantes; o risco construtivo passa a ser continuidade, emendas e desalinhamento, não flambagem.
- Em seção `box` com número ímpar de palitos, o modelo usa posição central/simétrica para evitar excentricidade; mover o palito extra para um canto altera centroide e inércia.
- `laced_box`/`spaced_box` sem palitos de ligação contabilizados é automaticamente rebaixado para `contact_box`; a ponte usa apenas palitos longitudinais e cola nessa seção.
- O relatório também verifica interpenetração de retângulos no corte da seção; contato é permitido, sobreposição de volumes não é.

## Tabela executiva
| grupo | n | layout | orientação | Iy [mm⁴] | Iz [mm⁴] | offset centroide [mm] | aviso | nota construtiva |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |
{table}

Arquivo completo: `section_layout_audit.csv`.
""",
            encoding="utf-8",
        )

    @staticmethod
    def _group_piece_plan(detailed: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = []
        grouped: Dict[tuple, Dict[str, Any]] = {}
        for r in (detailed or {}).get("stick_pieces", []) or []:
            key = (
                str(r.get("member_group", "")),
                str(r.get("stick_orientation", "flat")),
                float(safe_float(r.get("cut_length_mm"), 0.0) or 0.0),
            )
            item = grouped.setdefault(
                key,
                {
                    "member_group": key[0],
                    "stick_orientation": key[1],
                    "cut_length_mm": key[2],
                    "quantity": 0,
                    "total_length_mm": 0.0,
                    "example_member_ids": set(),
                },
            )
            item["quantity"] += 1
            item["total_length_mm"] += key[2]
            if r.get("member_id") is not None and len(item["example_member_ids"]) < 12:
                item["example_member_ids"].add(str(r.get("member_id")))
        for item in grouped.values():
            item["example_member_ids"] = ";".join(sorted(item["example_member_ids"], key=lambda v: int(v) if str(v).isdigit() else 999999))
            rows.append(item)
        rows.sort(key=lambda r: (str(r.get("member_group")), -float(r.get("cut_length_mm") or 0.0), str(r.get("stick_orientation"))))
        return rows

    @staticmethod
    def _joint_summary_plan(detailed: Dict[str, Any]) -> List[Dict[str, Any]]:
        grouped: Dict[tuple, Dict[str, Any]] = {}
        for r in (detailed or {}).get("glue_joints", []) or []:
            key = (
                str(r.get("member_group", "")),
                str(r.get("joint_model", r.get("joint_type", ""))),
                float(safe_float(r.get("overlap_length_mm"), 0.0) or 0.0),
                str(r.get("splice_pattern", "")),
            )
            item = grouped.setdefault(
                key,
                {
                    "member_group": key[0],
                    "joint_model": key[1],
                    "overlap_length_mm": key[2],
                    "splice_pattern": key[3],
                    "joint_count": 0,
                    "min_FS_glue_shear": None,
                    "example_joint_ids": [],
                },
            )
            item["joint_count"] += 1
            fs = safe_float(r.get("FS_glue_shear", r.get("FS_glue", r.get("FS"))), None)
            if fs is not None and (item["min_FS_glue_shear"] is None or fs < item["min_FS_glue_shear"]):
                item["min_FS_glue_shear"] = float(fs)
            if r.get("joint_id") is not None and len(item["example_joint_ids"]) < 8:
                item["example_joint_ids"].append(str(r.get("joint_id")))
        rows = []
        for item in grouped.values():
            item["example_joint_ids"] = ";".join(item["example_joint_ids"])
            rows.append(item)
        rows.sort(key=lambda r: (str(r.get("member_group")), str(r.get("joint_model")), float(r.get("overlap_length_mm") or 0.0)))
        return rows

    @staticmethod
    def _assembly_sequence_rows(detailed: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = list((detailed or {}).get("assembly_steps", []) or [])
        if rows:
            return rows
        # Fallback mínimo caso o detalhamento não tenha gerado assembly_steps.
        return [
            {"step_index": 1, "title": "Cortar e separar peças", "instruction": "Agrupar por comprimento, orientação e grupo."},
            {"step_index": 2, "title": "Montar banzos inferiores", "instruction": "Montar em gabarito plano e conferir simetria."},
            {"step_index": 3, "title": "Montar banzos superiores", "instruction": "Usar orientação edge/lateral para cima e seção box sem torção."},
            {"step_index": 4, "title": "Montar laterais", "instruction": "Adicionar montantes e diagonais em pares espelhados."},
            {"step_index": 5, "title": "Fechar estrutura 3D", "instruction": "Instalar transversais, cross-frames e bracing superior/inferior."},
            {"step_index": 6, "title": "Inspecionar e pesar", "instruction": "Conferir massa, simetria, cura e alinhamento antes do ensaio."},
        ]

    @staticmethod
    def _subassembly_rows(detailed: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Aggregate member detail rows into practical subassemblies.

        These rows are intentionally higher level than ``stick_pieces.csv`` and
        lower level than the executive summary: one line tells the builder how a
        family of members must be glued and oriented.
        """
        grouped: Dict[tuple, Dict[str, Any]] = {}
        for r in (detailed or {}).get("member_detail_checks", []) or []:
            key = (
                str(r.get("group", "")),
                str(r.get("layout", "")),
                int(safe_float(r.get("n_lanes_sticks", r.get("n_sticks_current", 1)), 1) or 1),
                str(r.get("joint_model", "")),
            )
            item = grouped.setdefault(
                key,
                {
                    "member_group": key[0],
                    "layout": key[1],
                    "n_sticks": key[2],
                    "joint_model": key[3],
                    "member_count": 0,
                    "member_ids_example": [],
                    "min_length_mm": None,
                    "max_length_mm": None,
                    "pieces_per_lane": None,
                    "total_piece_count": 0,
                    "min_FS_global": None,
                    "min_FS_glue": None,
                    "construction_note": "",
                },
            )
            item["member_count"] += 1
            if r.get("member_id") is not None and len(item["member_ids_example"]) < 16:
                item["member_ids_example"].append(str(r.get("member_id")))
            L = safe_float(r.get("member_length_mm"), None)
            if L is not None:
                item["min_length_mm"] = L if item["min_length_mm"] is None else min(float(item["min_length_mm"]), L)
                item["max_length_mm"] = L if item["max_length_mm"] is None else max(float(item["max_length_mm"]), L)
            ppl = safe_float(r.get("pieces_per_lane"), None)
            if ppl is not None:
                item["pieces_per_lane"] = int(ppl)
            item["total_piece_count"] += int(safe_float(r.get("total_piece_count"), 0) or 0)
            fs_g = safe_float(r.get("FS_min_global"), None)
            if fs_g is not None and (item["min_FS_global"] is None or fs_g < item["min_FS_global"]):
                item["min_FS_global"] = float(fs_g)
            fs_c = safe_float(r.get("FS_min_glue"), None)
            if fs_c is not None and (item["min_FS_glue"] is None or fs_c < item["min_FS_glue"]):
                item["min_FS_glue"] = float(fs_c)

        rows: List[Dict[str, Any]] = []
        for item in grouped.values():
            g = str(item.get("member_group"))
            layout = str(item.get("layout"))
            n = int(item.get("n_sticks") or 1)
            if g == "top_chord":
                note = "banzo superior comprimido: sanduíche fechado de 8 palitos (4 no núcleo, 2 capas internas e 2 capas externas contínuas); primeira lâmina do núcleo recebe os cortes-guia; cola longitudinal das faces contabilizada"
            elif g == "bottom_chord":
                note = "banzo inferior em T: alma edge com 14 palitos por lateral sobrepostos 20 mm face-a-face; mesa flat também sobreposta face-a-face e reforçada localmente quando o FS da cola exigir"
            elif g == "vertical":
                note = "montantes: nas estações x=0 e x=1300 usar sanduíche fechado de 6 palitos (4 no núcleo + 2 capas); demais montantes mantêm a seção exportada; controlar perpendicularidade e flambagem"
            elif g == "diagonal":
                note = "diagonais: montar em pares espelhados; não alinhar emendas em painéis vizinhos"
            elif "bracing" in g or "transverse" in g:
                note = "travamento/transversal: instalar antes de manusear a ponte fora do gabarito; controla torção e sidesway"
            elif g == "support_pad":
                note = "sapata/apoio: garantir contato plano e simétrico com a mesa"
            else:
                note = "seguir posição e orientação do stick_pieces.csv"
            if layout == "box" and n % 2 == 1:
                note += "; seção box ímpar usa palito central — não deslocar para canto"
            item["construction_note"] = note
            item["member_ids_example"] = ";".join(item.get("member_ids_example", []))
            rows.append(item)
        rows.sort(key=lambda r: (str(r.get("member_group")), int(r.get("n_sticks") or 0), str(r.get("layout"))))
        return rows

    def _write_detailed_fabrication_method(
        self,
        out: Path,
        cfg: Dict[str, Any],
        detailed: Dict[str, Any],
    ) -> Dict[str, str]:
        piece_plan = self._group_piece_plan(detailed)
        joint_plan = self._joint_summary_plan(detailed)
        subassembly_rows = self._subassembly_rows(detailed)
        assembly_rows = self._assembly_sequence_rows(detailed)
        GeometryService.write_csv(out / "04_plano_pecas_por_medida.csv", piece_plan)
        GeometryService.write_csv(out / "04_subconjuntos_montagem.csv", subassembly_rows)
        GeometryService.write_csv(out / "05_mapa_juntas_por_tipo.csv", joint_plan)
        GeometryService.write_csv(out / "06_sequencia_montagem.csv", assembly_rows)

        top_piece_rows = "\n".join(
            f"| {r.get('member_group')} | {r.get('stick_orientation')} | {float(r.get('cut_length_mm') or 0):.1f} | {r.get('quantity')} | {r.get('example_member_ids')} |"
            for r in piece_plan[:40]
        ) or "| — | — | — | — | — |"
        subassembly_md_rows = "\n".join(
            f"| {r.get('member_group')} | {r.get('layout')} | {r.get('n_sticks')} | "
            f"{safe_float(r.get('min_length_mm'), None) if r.get('min_length_mm') is not None else '—'}–"
            f"{safe_float(r.get('max_length_mm'), None) if r.get('max_length_mm') is not None else '—'} | "
            f"{r.get('member_count')} | {r.get('pieces_per_lane')} | {r.get('joint_model')} | {r.get('construction_note')} |"
            for r in subassembly_rows[:50]
        ) or "| — | — | — | — | — | — | — | — |"
        joint_rows = "\n".join(
            f"| {r.get('member_group')} | {r.get('joint_model')} | {float(r.get('overlap_length_mm') or 0):.1f} | {r.get('joint_count')} | {safe_float(r.get('min_FS_glue_shear'), None)} |"
            for r in joint_plan[:40]
        ) or "| — | — | — | — | — |"
        assembly_md = "\n".join(
            f"{int(safe_float(r.get('step_index'), i+1) or (i+1))}. **{r.get('title', 'Etapa')}** — {r.get('instruction', '')}"
            for i, r in enumerate(assembly_rows)
        )
        bridge = cfg.get("bridge", {}) or {}
        detail = cfg.get("detail_model", {}) or {}
        method = f"""# Plano detalhado de fabricação e montagem

Este documento consolida o que antes ficava espalhado em vários CSVs. Use os CSVs ao lado apenas como tabelas executivas; este arquivo descreve a ordem real de montagem.

## 1. Preparação
- Monte um gabarito em escala real com vão de {bridge.get('span_mm')} mm, largura de {bridge.get('width_mm')} mm e altura central de {bridge.get('center_height_mm')} mm.
- Separe palitos por massa/comprimento semelhante; rejeite peças empenadas para banzos e montantes centrais.
- Marque banzos superiores e inferiores com orientação `edge` quando indicada no arquivo `stick_pieces.csv`.
- Faça montagem a seco antes da cola.

## 2. Plano de cortes agrupado
| grupo | orientação | corte [mm] | quantidade | membros exemplo |
| --- | --- | ---: | ---: | --- |
{top_piece_rows}

Arquivo completo: `04_plano_pecas_por_medida.csv`.

## 3. Subconjuntos construtivos por grupo
| grupo | seção/layout | palitos por membro | faixa L [mm] | membros | peças/lane | junta | instrução crítica |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
{subassembly_md_rows}

Arquivo completo: `04_subconjuntos_montagem.csv`.

## 4. Ligações por tipo
| grupo | modelo de junta | overlap [mm] | quantidade | menor FS cola |
| --- | --- | ---: | ---: | ---: |
{joint_rows}

Arquivo completo: `05_mapa_juntas_por_tipo.csv`.

## 5. Visualizações de montagem
- Geometria 3D com cargas, apoios e FS/uso: `../plots/01_geometria_3d_fs_uso.html`.
- Geometria 3D com prismas reais completos: `../plots/02_montagem_3d_prismas_reais_interativa.html`.
- Vistas 2D/CAD peça-a-peça gerais: `../plots/16_vistas_cad_peca_a_peca.png`.
- Vistas 2D/CAD peça-a-peça por subconjunto: `../plots/cad_subconjuntos/`.
- HTMLs peça-a-peça por subconjunto: `../plots/subconjuntos_html/`.

## 6. Sequência lógica de montagem
{assembly_md}

## 7. Regras construtivas críticas
- **Banzos superiores:** montar primeiro como subpeças retas/segmentadas em `edge + box`; controlar torção durante a cura.
- **Banzos inferiores:** preservar continuidade; emendas sempre desencontradas entre lanes.
- **Montantes centrais:** colar em pares simétricos; manter perpendicularidade ao banzo inferior.
- **Diagonais:** montar sempre em pares espelhados para não introduzir torção global.
- **Cross-frames e bracing:** instalar antes de manusear a ponte fora do gabarito; eles são parte do travamento do banzo comprimido.
- **Sapata de apoio:** conferir se há contato pleno em todos os pontos ativos. Se houver folga, corrigir lixando/ajustando antes do ensaio.

## 8. Cola, cura e inspeção
- Overlap nominal: {detail.get('overlap_length_mm')} mm.
- Evite excesso de cola; excesso aumenta massa e raramente aumenta resistência proporcional.
- Prense as juntas até a pega inicial e deixe cura completa antes de fechar a estrutura 3D.
- Antes do ensaio, conferir `symmetry_audit.csv`, massa final, ausência de torção nos banzos e alinhamento dos apoios.
"""
        (out / "04_plano_montagem_detalhado.md").write_text(method, encoding="utf-8")
        (out / "04_subconjuntos_montagem.md").write_text(
            "# Subconjuntos de montagem\n\n"
            "| grupo | seção/layout | palitos por membro | faixa L [mm] | membros | peças/lane | junta | instrução crítica |\n"
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n"
            f"{subassembly_md_rows}\n",
            encoding="utf-8",
        )
        (out / "05_mapa_juntas_por_tipo.md").write_text(
            "# Mapa de juntas por tipo\n\n"
            "| grupo | modelo de junta | overlap [mm] | quantidade | menor FS cola |\n"
            "| --- | --- | ---: | ---: | ---: |\n"
            f"{joint_rows}\n",
            encoding="utf-8",
        )
        (out / "06_sequencia_montagem.md").write_text(
            "# Sequência lógica de montagem\n\n" + assembly_md + "\n",
            encoding="utf-8",
        )
        return {
            "detailed_fabrication_method_md": str(out / "04_plano_montagem_detalhado.md"),
            "subassembly_plan_md": str(out / "04_subconjuntos_montagem.md"),
            "joint_plan_md": str(out / "05_mapa_juntas_por_tipo.md"),
            "assembly_sequence_md": str(out / "06_sequencia_montagem.md"),
            "piece_plan_csv": str(out / "04_plano_pecas_por_medida.csv"),
            "joint_plan_csv": str(out / "05_mapa_juntas_por_tipo.csv"),
            "subassembly_plan_csv": str(out / "04_subconjuntos_montagem.csv"),
            "assembly_sequence_csv": str(out / "06_sequencia_montagem.csv"),
        }

    @staticmethod
    def _write_connectivity_cut_audit(out: Path, cfg: Dict[str, Any], detailed: Dict[str, Any]) -> str:
        rows = list((detailed or {}).get("stick_pieces", []) or [])
        mat = cfg.get("material", {}) or {}
        stick_len = safe_float(mat.get("stick_length_mm"), 120.0) or 120.0
        stick_w = safe_float(mat.get("stick_width_mm"), 7.0) or 7.0
        stick_t = safe_float(mat.get("stick_thickness_mm"), 1.5) or 1.5
        max_cut = safe_float((cfg.get("detail_model", {}) or {}).get("max_cut_length_mm"), stick_len) or stick_len
        cut_limit = min(float(stick_len), float(max_cut))

        total = len(rows)
        as_built = (detailed or {}).get("as_built_audit", {}) or {}
        as_built_intersections = int(as_built.get("interpenetration_count", 0) or 0)
        over_len = []
        short_constructive = []
        bad_width = []
        bad_thk = []
        by_model: Dict[str, int] = {}
        by_x_layer: Dict[str, int] = {}
        x_unresolved = []
        for r in rows:
            cut = safe_float(r.get("cut_length_mm"), None)
            if cut is not None and cut > cut_limit + 1.0e-9:
                over_len.append(r)
            if not bool(r.get("constructive_piece_length_ok", True)):
                short_constructive.append(r)
            vw = safe_float(r.get("visual_width_mm"), None)
            vt = safe_float(r.get("visual_thickness_mm"), None)
            if vw is not None and vt is not None:
                if max(vw, vt) > max(stick_w, stick_t) + 1.0e-9:
                    bad_width.append(r)
                if min(vw, vt) > min(stick_w, stick_t) + 1.0e-9:
                    bad_thk.append(r)
            model = str(r.get("section_connection_model", r.get("section_layout_effective", "unknown")))
            by_model[model] = by_model.get(model, 0) + 1
            x_handling = str(r.get("x_bracing_crossing_handling", "") or "")
            if x_handling:
                by_x_layer[x_handling] = by_x_layer.get(x_handling, 0) + 1
            ok_x_handling = {"single_diagonal_no_crossing", "not_x_bracing"}
            if str(r.get("member_group")) in {"bottom_bracing", "top_bracing", "cross_frame_bracing"} and x_handling not in ok_x_handling:
                x_unresolved.append(r)

        top_over = "\n".join(
            f"| {r.get('stick_id')} | {r.get('member_id')} | {r.get('member_group')} | {safe_float(r.get('cut_length_mm'), None)} | {safe_float(r.get('max_cut_length_mm'), cut_limit)} |"
            for r in over_len[:25]
        ) or "| — | — | — | — | — |"
        top_short = "\n".join(
            f"| {r.get('stick_id')} | {r.get('member_id')} | {r.get('member_group')} | {safe_float(r.get('geometric_piece_length_mm'), None)} | {safe_float(r.get('min_constructive_piece_length_mm'), None)} |"
            for r in short_constructive[:25]
        ) or "| — | — | — | — | — |"
        model_rows = "\n".join(
            f"| {model} | {count} |"
            for model, count in sorted(by_model.items(), key=lambda kv: (-kv[1], kv[0]))
        ) or "| — | — |"
        x_rows = "\n".join(
            f"| {model} | {count} |"
            for model, count in sorted(by_x_layer.items(), key=lambda kv: (-kv[1], kv[0]))
        ) or "| — | — |"
        top_x_bad = "\n".join(
            f"| {r.get('stick_id')} | {r.get('member_id')} | {r.get('member_group')} | {r.get('x_bracing_crossing_handling')} |"
            for r in x_unresolved[:25]
        ) or "| — | — | — |"
        verdict = "OK" if not over_len and not short_constructive and not bad_width and not bad_thk and not x_unresolved and as_built_intersections == 0 else "FALHA"
        text = f"""# Auditoria de conectividade e limites físicos dos palitos

Veredito: **{verdict}**.

## Limites usados
- Comprimento máximo de corte: **{cut_limit:.1f} mm**.
- Dimensão individual do palito: **{stick_len:.1f} × {stick_w:.1f} × {stick_t:.1f} mm**.
- Total de peças detalhadas: **{total}**.

## Contagens de falha
- Cortes acima do limite: **{len(over_len)}**.
- Peças abaixo do comprimento construtivo mínimo: **{len(short_constructive)}**.
- Prismas/peças com largura visual acima do palito: **{len(bad_width)}**.
- Prismas/peças com espessura visual acima do palito: **{len(bad_thk)}**.
- Contraventamentos em X ou bracings equivalentes sem solução física de cruzamento: **{len(x_unresolved)}**.
- Interpenetrações volumétricas na geometria `as_built`: **{as_built_intersections}**.

## Modelos de conexão usados
| modelo de conexão | peças |
| --- | ---: |
{model_rows}

## Tratamento de cruzamentos em X
| tratamento | peças |
| --- | ---: |
{x_rows}

O tratamento `single_diagonal_no_crossing` substitui X por diagonais alternadas do tipo Warren/Pratt-Howe, eliminando o cruzamento físico. `split_midpoint_lap_joint` e `alternate_front_back_layer_no_midspan_joint` ficam apenas como modos de estudo; no pacote final, eles são sinalizados como não resolvidos porque ainda exigem uma junta central/camada de extremidade que precisa ser modelada com mais detalhes.

### Cruzamentos em X não resolvidos
| stick | membro | grupo | tratamento |
| --- | ---: | --- | --- |
{top_x_bad}

## Retalhos construtivamente frágeis
| stick | membro | grupo | comprimento geométrico [mm] | mínimo [mm] |
| --- | ---: | --- | ---: | ---: |
{top_short}

## Cortes acima do limite
| stick | membro | grupo | corte [mm] | limite [mm] |
| --- | ---: | --- | ---: | ---: |
{top_over}

## Interpretação
- `face_to_face_lamination`: palitos colados por face, sem crédito para contato por aresta.
- `closed_face_sandwich_core_caps_external_covers`: sanduíche fechado com núcleo, capas internas e capas externas contínuas; cola longitudinal contabilizada e ação composta reduzida por `eta_I`.
- `mixed_T_contact_lamination`: seção de 3 palitos conectada; não é caixa.
- `four_side_contact_box_with_face_side_glue`: caixa de 4+ palitos por contato lateral/face, coerente com montagem física.
- Se qualquer contagem de falha for maior que zero, o relatório estrutural não deve ser usado como memorial final sem corrigir o detalhamento.
"""
        path = out / "09_auditoria_conectividade_e_cortes.md"
        path.write_text(text, encoding="utf-8")
        return str(path)


    @staticmethod
    def _md_number(value: Any, digits: int = 3) -> str:
        number = safe_float(value, None)
        return "—" if number is None else f"{number:.{digits}f}"

    @staticmethod
    def _md_text(value: Any) -> str:
        if value in (None, ""):
            return "—"
        return str(value).replace("|", "/").replace("\n", " ")

    def _write_complete_calculation_ledger(
        self,
        out: Path,
        cfg: Dict[str, Any],
        metrics: Dict[str, Any],
        detailed: Dict[str, Any],
    ) -> str:
        """Exporta o memorial rastreável membro -> peça -> junta colada."""
        member_rows = sorted(
            list((detailed or {}).get("member_detail_checks", []) or []),
            key=lambda r: int(r.get("member_id", 0) or 0),
        )
        piece_rows = sorted(
            list((detailed or {}).get("stick_pieces", []) or []),
            key=lambda r: (
                str(r.get("member_group", "")),
                int(r.get("member_id", 0) or 0),
                int(r.get("lane", 0) or 0),
                int(r.get("piece_index", 0) or 0),
            ),
        )
        joint_rows = sorted(
            list((detailed or {}).get("glue_joints", []) or []),
            key=lambda r: (
                str(r.get("member_group", "")),
                int(r.get("member_id", 0) or 0),
                str(r.get("joint_id", "")),
            ),
        )
        detail = cfg.get("detail_model", {}) or {}
        material = cfg.get("material", {}) or {}

        group_summary: Dict[str, Dict[str, Any]] = {}
        for r in member_rows:
            group = str(r.get("group", "—"))
            row = group_summary.setdefault(group, {"members": 0, "min_fs": None, "max_N": 0.0, "mode": "—"})
            row["members"] += 1
            fs = safe_float(r.get("FS_min_global"), None)
            if fs is not None and (row["min_fs"] is None or fs < row["min_fs"]):
                row["min_fs"] = fs
                row["mode"] = r.get("governing_mode_global") or "—"
            n = abs(safe_float(r.get("N_member_N"), 0.0) or 0.0)
            row["max_N"] = max(row["max_N"], n)

        grouped_lines = [
            "| grupo | membros | |N| máximo (N) | FS mínimo pós-detalhe | modo governante |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for group in sorted(group_summary):
            r = group_summary[group]
            grouped_lines.append(
                f"| {self._md_text(group)} | {r['members']} | {self._md_number(r['max_N'])} | {self._md_number(r['min_fs'])} | {self._md_text(r['mode'])} |"
            )

        member_lines = [
            "| id | grupo | função | L (mm) | seção / n | A (mm²) | Iy (mm⁴) | Iz (mm⁴) | ηI | N (N) | σ axial (MPa) | M imperf. (N·mm) | σ comb. (MPa) | FS global | modo |",
            "| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for r in member_rows:
            member_lines.append(
                "| {id} | {grp} | {role} | {L} | {layout} / {n} | {A} | {Iy} | {Iz} | {eta} | {N} | {sig} | {M} | {comb} | {FS} | {mode} |".format(
                    id=self._md_text(r.get("member_id")), grp=self._md_text(r.get("group")), role=self._md_text(r.get("role")),
                    L=self._md_number(r.get("member_length_mm")), layout=self._md_text(r.get("layout")), n=self._md_text(r.get("n_sticks_current")),
                    A=self._md_number(r.get("section_A_mm2")), Iy=self._md_number(r.get("section_Iy_mm4")), Iz=self._md_number(r.get("section_Iz_mm4")),
                    eta=self._md_number(r.get("section_eta_I")), N=self._md_number(r.get("N_member_N")), sig=self._md_number(r.get("sigma_axial_member_MPa")),
                    M=self._md_number(r.get("M_imperfection_Nmm")), comb=self._md_number(r.get("sigma_combined_est_MPa")), FS=self._md_number(r.get("FS_min_global")),
                    mode=self._md_text(r.get("governing_mode_global")),
                )
            )

        piece_lines = [
            "| palito | membro / grupo | faixa | papel | orientação | corte (mm) | massa (g) | N atribuído (N) | σ (MPa) | corte início / fim | hospedeiro início / fim |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for r in piece_rows:
            start = "sim" if bool(r.get("miter_cut_start_required", False)) else "não"
            end = "sim" if bool(r.get("miter_cut_end_required", False)) else "não"
            piece_lines.append(
                "| {sid} | {mid} / {grp} | L{lane}-P{piece} | {role} | {ori} | {cut} | {mass} | {N} | {sig} | {sta} / {end} | {hs} / {he} |".format(
                    sid=self._md_text(r.get("stick_id")), mid=self._md_text(r.get("member_id")), grp=self._md_text(r.get("member_group")),
                    lane=self._md_text(r.get("lane")), piece=self._md_text(r.get("piece_index")), role=self._md_text(r.get("structural_lane_role")),
                    ori=self._md_text(r.get("stick_orientation")), cut=self._md_number(r.get("shop_cut_length_mm", r.get("cut_length_mm"))),
                    mass=self._md_number(r.get("mass_g")), N=self._md_number(r.get("N_piece_N")), sig=self._md_number(r.get("sigma_axial_piece_MPa")),
                    sta=start, end=end, hs=self._md_text(r.get("miter_cut_start_host_group")), he=self._md_text(r.get("miter_cut_end_host_group")),
                )
            )

        joint_lines = [
            "| junta | membro / grupo | modelo | sobreposição (mm) | talas | área física cola (mm²) | força transferida (N) | τ cola (MPa) | FS cola | risco |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for r in joint_rows:
            joint_lines.append(
                "| {jid} | {mid} / {grp} | {model} | {ov} | {spl} | {area} | {force} | {tau} | {fs} | {risk} |".format(
                    jid=self._md_text(r.get("joint_id")), mid=self._md_text(r.get("member_id")), grp=self._md_text(r.get("member_group")), model=self._md_text(r.get("joint_model")),
                    ov=self._md_number(r.get("overlap_length_mm")), spl=self._md_text(r.get("splints_per_splice")), area=self._md_number(r.get("physical_glue_area_mm2")),
                    force=self._md_number(r.get("force_transfer_N")), tau=self._md_number(r.get("glue_shear_MPa")), fs=self._md_number(r.get("FS_glue_shear")), risk=self._md_text(r.get("risk_flag")),
                )
            )

        text = f"""# Memorial completo por membro, palito e junta

Este arquivo é a rastreabilidade pós-detalhamento da estrutura fabricável. Ele lista as propriedades de seção efetivamente usadas, esforços axiais, momento de imperfeição para peças comprimidas, tensões combinadas, fatores de segurança, cortes de cada palito e verificação de cada junta colada registrada.

## Hipóteses numéricas registradas

| parâmetro | valor |
| --- | ---: |
| carga nominal total (N) | {self._md_number((cfg.get('bridge', {}) or {}).get('load_total_N'))} |
| massa competitiva calculada (g) | {self._md_number(metrics.get('competition_mass_g', metrics.get('estimated_total_mass_g')))} |
| limite de massa (g) | {self._md_number(metrics.get('mass_limit_effective_g', material.get('mass_limit_g')))} |
| resistência de cisalhamento adotada para cola (MPa) | {self._md_number(detail.get('glue_shear_strength_MPa'))} |
| excentricidade base de imperfeição (mm) | {self._md_number(detail.get('imperfection_eccentricity_mm'))} |
| comprimento máximo de corte (mm) | {self._md_number(detail.get('max_cut_length_mm', material.get('stick_length_mm')))} |
| comprimento mínimo construtivo (mm) | {self._md_number(detail.get('min_constructive_piece_length_mm'))} |

## Síntese por grupo

{chr(10).join(grouped_lines)}

## Cálculo membro a membro

{chr(10).join(member_lines)}

## Rastreabilidade palito a palito

{chr(10).join(piece_lines)}

## Juntas e transferência por cola

{chr(10).join(joint_lines)}

## Leitura obrigatória

A tabela permite localizar o ponto calculado mais solicitado, mas a carga real de ruptura depende da dispersão da madeira e da execução da cola. A validação física requer corpos de prova do lote de palitos e da junta empregada, além de conferência dimensional do gabarito e da cura.
"""
        path = out / "12_memorial_completo_membro_palito_junta.md"
        path.write_text(text, encoding="utf-8")
        GeometryService.write_csv(out / "12_memorial_membros.csv", member_rows)
        GeometryService.write_csv(out / "12_memorial_palitos.csv", piece_rows)
        GeometryService.write_csv(out / "12_memorial_juntas.csv", joint_rows)
        return str(path)

    def _write_model_basis_and_feature_audit(
        self,
        out: Path,
        cfg: Dict[str, Any],
        metrics: Dict[str, Any],
        detailed: Dict[str, Any],
    ) -> str:
        pieces = list((detailed or {}).get("stick_pieces", []) or [])
        joints = list((detailed or {}).get("glue_joints", []) or [])
        summary = (detailed or {}).get("summary", {}) or {}
        by_group: Dict[str, List[Dict[str, Any]]] = {}
        for r in pieces:
            by_group.setdefault(str(r.get("member_group", "")), []).append(r)

        def endpoint_compliance(group: str) -> tuple[int, int]:
            rows = by_group.get(group, [])
            mids = sorted({int(r.get("member_id", 0) or 0) for r in rows})
            compliant = 0
            for mid in mids:
                pr = [r for r in rows if int(r.get("member_id", 0) or 0) == mid]
                has_start = any(bool(r.get("miter_cut_start_required", False)) for r in pr)
                has_end = any(bool(r.get("miter_cut_end_required", False)) for r in pr)
                if has_start and has_end:
                    compliant += 1
            return compliant, len(mids)

        d_ok, d_total = endpoint_compliance("diagonal")
        x_ok, x_total = endpoint_compliance("cross_frame_bracing")
        tee = by_group.get("bottom_chord", [])
        tee_web = [r for r in tee if str(r.get("structural_lane_role", "")) in {"tee_web", "alma"}]
        tee_flange = [r for r in tee if str(r.get("structural_lane_role", "")) in {"tee_flange", "mesa_superior"}]
        bottom_joints = [r for r in joints if str(r.get("member_group", "")) == "bottom_chord"]
        min_bottom_glue = min((safe_float(r.get("FS_glue_shear"), 1.0e99) or 1.0e99 for r in bottom_joints), default=None)
        max_cut = max((safe_float(r.get("shop_cut_length_mm", r.get("cut_length_mm")), 0.0) or 0.0 for r in pieces), default=0.0)
        min_cut = min((safe_float(r.get("shop_cut_length_mm", r.get("cut_length_mm")), 1.0e99) or 1.0e99 for r in pieces), default=0.0)
        bridge = cfg.get("bridge", {}) or {}
        detail = cfg.get("detail_model", {}) or {}
        analysis = cfg.get("analysis", {}) or {}
        top_bracing_count = len(by_group.get("top_bracing", []))
        bottom_bracing_count = len(by_group.get("bottom_bracing", []))
        internal_count = len(by_group.get("cross_frame_bracing", []))

        text = f"""# Validação da modelagem física e base técnica

## Alterações construtivas verificadas

| requisito de projeto | implementação auditada | situação |
| --- | --- | --- |
| Banzo inferior em T fabricável | alma vertical e mesa superior; fabricação contínua com juntas desencontradas | {'atendido' if tee_web and tee_flange else 'não atendido'} |
| Sem zig-zag no plano do banzo inferior | peças `bottom_bracing` geradas: {bottom_bracing_count} | {'atendido' if bottom_bracing_count == 0 else 'não atendido'} |
| Sem zig-zag longitudinal no banzo superior | peças `top_bracing` geradas: {top_bracing_count}; topo usa travessas transversais reforçadas nas estações carregadas | {'atendido' if top_bracing_count == 0 else 'não atendido'} |
| Zig-zag interno entre as laterais | peças `cross_frame_bracing` geradas: {internal_count}, incluindo diafragmas transversais internos e zig-zag longitudinal 3D; tipo-base fixado em `{self._md_text(bridge.get('cross_frame_truss_type'))}` | {'atendido' if internal_count > 0 else 'não atendido'} |
| Cortes nas duas pontas das diagonais laterais | membros conformes: {d_ok}/{d_total} | {'atendido' if d_total > 0 and d_ok == d_total else 'revisar'} |
| Cortes nas duas pontas das diagonais internas | membros conformes: {x_ok}/{x_total} | {'atendido' if x_total > 0 and x_ok == x_total else 'revisar'} |
| Limite construtivo de comprimento | faixa de corte: {min_cut:.3f} a {max_cut:.3f} mm | {'atendido' if min_cut >= float(detail.get('min_constructive_piece_length_mm', 20.0)) - 1e-9 and max_cut <= float(detail.get('max_cut_length_mm', 120.0)) + 1e-9 else 'não atendido'} |
| Colisão volumétrica as-built | interpenetrações registradas: {int(metrics.get('as_built_interpenetration_count', 0) or 0)} | {'atendido' if int(metrics.get('as_built_interpenetration_count', 0) or 0) == 0 else 'não atendido'} |

## Banzo inferior em T e colagem

A alma do T é tratada como caminho principal de tração; a mesa superior fornece assentamento para os nós e rigidez secundária. O detalhamento gerou **{len(tee_web)}** peças de alma e **{len(tee_flange)}** peças de mesa. A alma usa exatamente **{self._md_text(detail.get('bottom_chord_tee_web_piece_count_per_side'))} palitos por lateral**, com sobreposição face-a-face de **{self._md_number(detail.get('bottom_chord_tee_web_overlap_mm'))} mm**; a mesa usa sobreposição face-a-face desencontrada pela fase nominal de **{self._md_number(detail.get('bottom_chord_tee_flange_phase_mm'))} mm**. Talas locais somente são adicionadas onde o FS calculado da junta sobreposta não alcança o mínimo de cola configurado. O menor FS de cola registrado para o T é **{self._md_number(min_bottom_glue)}**.

## Modelo mecânico efetivamente considerado

O cálculo global continua baseado na malha estrutural tridimensional do simulador, com membros discretizados por eixo e esforços obtidos por análise elástica. O pós-processamento não assume apenas tração/compressão ideal: para membros comprimidos, registra flambagem por Euler/Johnson, imperfeição geométrica equivalente e interação flexo-compressiva; para seções coladas, reduz a inércia composta por um fator de ação parcial `eta_I`, em vez de presumir ligação monolítica perfeita.

| fator modelado | valor/configuração registrada |
| --- | --- |
| Frame3DD habilitado quando disponível | {self._md_text(analysis.get('run_frame3dd_if_available'))} |
| rigidez geométrica Frame3DD | {self._md_text(analysis.get('frame3dd_include_geometric_stiffness'))} |
| deformação por cisalhamento Frame3DD | {self._md_text(analysis.get('frame3dd_include_shear_deformation'))} |
| excentricidade nominal de imperfeição | {self._md_number(detail.get('imperfection_eccentricity_mm'))} mm |
| FS mínimo de membro calculado / alvo recomendado | {self._md_number(metrics.get('min_fs_member_design', metrics.get('min_fs_design')))} / {self._md_number(analysis.get('target_min_fs'))} |
| FS mínimo de cola aceitável | {self._md_number(analysis.get('acceptance_min_glue_fs'))} |
| solver retornado | {self._md_text(metrics.get('solver_status'))} |
| status da checagem Frame3DD | {self._md_text(metrics.get('frame3dd_status'))} |
| massa competitiva / limite | {self._md_number(metrics.get('competition_mass_g', metrics.get('estimated_total_mass_g')))} / {self._md_number(metrics.get('mass_limit_effective_g'))} g |

## Fundamentação e limite de validade

A geometria adotada segue a premissa de treliça: esforços devem entrar nos nós com excentricidade mínima, para reduzir flexões secundárias. A modelagem das peças coladas considera que a ligação transfere esforço pela área efetivamente colada e que a seção composta não é perfeitamente monolítica sem calibração. O emprego de análise de pórticos/treliças em 3D com rigidez geométrica e forças internas é compatível com a classe de problema resolvida pelo Frame3DD.

O veredito deve ser lido com ressalva sempre que o FS mínimo calculado ficar abaixo do alvo recomendado ou quando a margem de massa não absorver variação real de cola e palitos. A aprovação numérica não substitui validação experimental. Palitos de picolé apresentam dispersão de propriedades, e a capacidade real da cola depende do adesivo, preparo superficial, pressão, cura e falha da madeira. Antes da construção definitiva, devem ser ensaiados: palitos do lote em tração/compressão, emendas com as talas previstas, uma amostra do nó T–montante–diagonal e o contato real da carga e dos apoios.

### Referências técnicas utilizadas como base conceitual

- USDA Forest Products Laboratory. *Wood Handbook: Wood as an Engineering Material*, FPL-GTR-282, capítulos de propriedades mecânicas e adesivos para madeira, 2021.
- Gavin, H. P. *Frame3DD: Static and Dynamic Structural Analysis of 2D and 3D Frames*, documentação do solucionador empregado como verificação complementar.
"""
        path = out / "13_validacao_modelagem_e_base_tecnica.md"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def generate(
        self,
        cfg: Dict[str, Any],
        metrics: Dict[str, Any],
        member_checks: List[Dict[str, Any]],
        detailed: Dict[str, Any],
        optimization: Dict[str, Any] | None,
        warnings: List[Dict[str, str]] | None,
        out_dir: str | Path,
    ) -> Dict[str, str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        summary = (detailed or {}).get("summary", {}) or {}
        verdict, verdict_reason, failures = self._verdict(cfg, metrics, optimization)
        acceptance_break = float(cfg.get("analysis", {}).get("acceptance_min_design_breaking_load_kgf", 80.0))
        pred_break = safe_float(
            metrics.get("predicted_breaking_load_design_kgf"),
            safe_float(metrics.get("predicted_breaking_load_kgf"), 0.0),
        ) or 0.0
        s8_rows_for_report = list((optimization or {}).get("s8_final_validation", []) or [])
        s8_for_report = s8_rows_for_report[0] if s8_rows_for_report else {}
        design_break = safe_float(
            metrics.get("predicted_breaking_load_kgf"),
            safe_float(
                metrics.get("predicted_breaking_load_design_kgf"),
                safe_float(s8_for_report.get("predicted_breaking_load_kgf"), None),
            ),
        )
        design_verdict = verdict
        governing_limit_state = (
            s8_for_report.get("governing_limit_state")
            or metrics.get("governing_limit_state")
        )
        governing_fs = safe_float(
            s8_for_report.get("governing_fs"),
            safe_float(metrics.get("governing_fs"), None),
        )
        competition_mass = safe_float(metrics.get("competition_mass_g"), safe_float(metrics.get("estimated_total_mass_g"), 0.0)) or 0.0
        mass_limit = safe_float(metrics.get("mass_limit_effective_g"), 1000.0) or 1000.0
        mass_margin = mass_limit - competition_mass
        solver_regular = self._solver_regular(metrics.get("solver_status"))

        exec_summary = {
            "verdict": verdict,
            "reason": verdict_reason,
            "predicted_breaking_load_nominal_kgf": pred_break,
            "predicted_breaking_load_design_multicase_kgf": design_break,
            "design_multicase_verdict": design_verdict,
            "design_multicase_governing_case": s8_for_report.get("governing_case"),
            "design_multicase_governing_strength_case": s8_for_report.get("governing_strength_case"),
            "design_multicase_governing_service_case": s8_for_report.get("governing_service_case"),
            "design_multicase_governing_contact_case": s8_for_report.get("governing_contact_case"),
            "design_multicase_governing_member_id": s8_for_report.get("governing_member_id"),
            "design_multicase_governing_member_group": s8_for_report.get("governing_member_group"),
            "design_multicase_governing_mode": s8_for_report.get("governing_mode"),
            "governing_limit_state": governing_limit_state,
            "governing_fs": governing_fs,
            "predicted_breaking_load_kgf": pred_break,
            "predicted_breaking_load_by_members_kgf": metrics.get("predicted_breaking_load_by_members_kgf"),
            "predicted_breaking_load_by_supports_kgf": metrics.get("predicted_breaking_load_by_supports_kgf"),
            "predicted_breaking_load_by_glue_kgf": metrics.get("predicted_breaking_load_by_glue_kgf"),
            "min_fs_member_design": metrics.get("min_fs_member_design", metrics.get("min_fs_design")),
            "min_fs_support": metrics.get("min_fs_support", metrics.get("min_support_fs")),
            "min_fs_glue": metrics.get("min_fs_glue", metrics.get("min_glue_fs")),
            "target_breaking_load_kgf": acceptance_break,
            "competition_mass_g": competition_mass,
            "competition_mass_margin_g": mass_margin,
            "solver_status": metrics.get("solver_status"),
            "solver_regular": solver_regular,
            "constraints_failed": failures,
        }
        (out / "executive_summary.json").write_text(
            json.dumps(exec_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        critical_members = self._top_critical_members(member_checks, top_k=15)
        GeometryService.write_csv(out / "critical_members.csv", critical_members)

        sizing_rows = list((metrics.get("member_sizing_plan") or []))
        donor_rows = sorted(
            [
                {
                    "member_id": r.get("member_id"),
                    "group": r.get("original_group"),
                    "N_N": r.get("force_N"),
                    "FS_min": r.get("FS_min"),
                    "utilization": r.get("old_utilization", r.get("utilization")),
                    "old_n": r.get("n_sticks_current"),
                    "new_n": r.get("n_sticks_recommended"),
                    "delta_mass_g": r.get("delta_mass_g"),
                    "reason": r.get("reason"),
                }
                for r in sizing_rows
                if bool(r.get("can_be_mass_donor"))
            ],
            key=lambda r: -(safe_float(r.get("FS_min"), 0.0) or 0.0),
        )[:15]
        GeometryService.write_csv(out / "member_sizing_actions.csv", sizing_rows)
        GeometryService.write_csv(out / "candidate_ranking.csv", self._candidate_ranking(optimization))
        GeometryService.write_csv(out / "donor_members.csv", donor_rows)

        load_contact_rows = self._load_contact_audit_rows(cfg, optimization)
        GeometryService.write_csv(out / "load_contact_assessment.csv", load_contact_rows)
        bridge_cfg = cfg.get("bridge", {}) or {}
        arch_warning = ""
        if str(bridge_cfg.get("top_profile", "")).lower() in {"shallow_arch", "shallow_arch_faceted", "arco"}:
            arch_warning = (
                "\n> Atenção: banzo superior arqueado só é compatível com carga distribuída "
                "se existir uma sela/plataforma rígida de madeira transferindo a carga "
                "para vários nós. Sem essa sela, o caso `crown_contact` é mais representativo.\n"
            )
        load_contact_md_rows = "\n".join(
            f"| {r.get('case')} | {safe_float(r.get('predicted_breaking_load_kgf'), None) if r.get('predicted_breaking_load_kgf') is not None else '—'} | "
            f"{safe_float(r.get('min_fs_design'), None) if r.get('min_fs_design') is not None else '—'} | "
            f"{r.get('governing_member_id') or '—'} / {r.get('governing_member_group') or '—'} | "
            f"{r.get('interpretation')} |"
            for r in load_contact_rows
        ) or "| — | — | — | — | — |"
        governing_load_contact = min(
            load_contact_rows,
            key=lambda r: safe_float(r.get('predicted_breaking_load_kgf'), 1.0e99) or 1.0e99,
            default={},
        )
        (out / "07_avaliacao_contato_carga.md").write_text(
            f"""# Avaliação do contato da carga

O modelo principal aceita carga distribuída por superfície, mas isso é uma hipótese construtiva: a ponte precisa ter uma região de apoio da carga que realmente transfira o peso para os nós previstos. Em banzo superior arqueado, uma anilha ou prato solto tende a encostar primeiro no ponto mais alto, concentrando força no montante/região central.
{arch_warning}
## Casos auditados
| caso | ruptura estimada (kgf) | FS design | membro governante | interpretação |
| --- | ---: | ---: | --- | --- |
{load_contact_md_rows}

## Caso governante de contato
- Caso: **{governing_load_contact.get('case', '—')}**.
- Ruptura estimada: **{safe_float(governing_load_contact.get('predicted_breaking_load_kgf'), None) if governing_load_contact else '—'} kgf**.
- Membro governante: **{governing_load_contact.get('governing_member_id', '—')} / {governing_load_contact.get('governing_member_group', '—')}**.
- Modo governante: **{governing_load_contact.get('governing_mode', '—')}**.

## Reações de apoio
As reações completas por nó são gravadas em `load_contact_assessment.csv` na coluna `support_reactions`. No memorial textual, usar a tabela acima como envelope de projeto e conferir se nenhuma reação de apoio excede o FS mínimo configurado.

## Requisito construtivo recomendado
- Usar uma plataforma/sela rígida colada ao topo, com contato em múltiplos nós e nos dois lados da ponte.
- A plataforma deve encostar em pelo menos três estações longitudinais simétricas e nos dois planos laterais da ponte.
- Se a carga real for aplicada por gancho/anilha pequena sem plataforma, usar `crown_contact` como caso governante, não o caso distribuído.
""",
            encoding="utf-8",
        )

        section_layout_rows = self._section_layout_audit_rows(cfg, member_checks)
        self._write_section_layout_audit(out, section_layout_rows)

        # Relatório de debug humano: os CSVs continuam internos, mas o pacote
        # de entrega precisa permitir rastrear por texto quais comparações
        # governaram o resultado.
        # A tabela principal deve refletir o mesmo conjunto que governa o
        # veredito.  Bracings secundários/tension-only podem ter FS de compressão
        # baixíssimo no pós-processador, mas não participam do envelope de
        # ruptura quando ``design_relevant`` é falso.  Misturar esses itens com
        # membros primários gerava um debug visualmente alarmante e incompatível
        # com o ``executive_summary``.
        worst_checks = sorted(
            [
                r for r in (member_checks or [])
                if r.get("design_relevant") is not False
                and safe_float(r.get("FS_min"), None) is not None
            ],
            key=lambda r: safe_float(r.get("FS_min"), 1.0e99) or 1.0e99,
        )[:25]
        secondary_low_checks = sorted(
            [
                r for r in (member_checks or [])
                if r.get("design_relevant") is False
                and safe_float(r.get("FS_min"), None) is not None
            ],
            key=lambda r: safe_float(r.get("FS_min"), 1.0e99) or 1.0e99,
        )[:12]
        debug_rows = "\n".join(
            f"| {r.get('member_id')} | {r.get('group')} | {r.get('n_sticks')} | {r.get('layout')} | "
            f"{safe_float(r.get('N_N'), None) if r.get('N_N') is not None else '—'} | "
            f"{safe_float(r.get('FS_min'), None) if r.get('FS_min') is not None else '—'} | "
            f"{r.get('governing_mode')} | {safe_float(r.get('Pcr_y_N'), None) if r.get('Pcr_y_N') is not None else '—'} | "
            f"{safe_float(r.get('Pcr_z_N'), None) if r.get('Pcr_z_N') is not None else '—'} |"
            for r in worst_checks
        ) or "| — | — | — | — | — | — | — | — | — |"
        secondary_debug_rows = "\n".join(
            f"| {r.get('member_id')} | {r.get('group')} | {r.get('n_sticks')} | {r.get('layout')} | "
            f"{safe_float(r.get('N_N'), None) if r.get('N_N') is not None else '—'} | "
            f"{safe_float(r.get('FS_min'), None) if r.get('FS_min') is not None else '—'} | "
            f"{r.get('governing_mode')} |"
            for r in secondary_low_checks
        ) or "| — | — | — | — | — | — | — |"
        (out / "10_debug_calculos_criticos.md").write_text(
            f"""# Debug dos cálculos críticos

Este arquivo resume os membros primários que governam a ruptura e os valores usados na comparação. Ele foi criado para evitar que a depuração dependa de planilhas.

| membro | grupo | palitos | layout | N [N] | FS mínimo | modo governante | Pcr y [N] | Pcr z [N] |
| ---: | --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
{debug_rows}

## Itens secundários com FS baixo, mas fora do envelope de veredito
Esses elementos aparecem para auditoria construtiva. Eles não devem ser lidos como ruptura primária se `design_relevant = false`, mas ajudam a identificar travamentos que precisam ser montados como tension-only, em X, ou reforçados por gabarito.

| membro | grupo | palitos | layout | N [N] | FS mínimo | modo governante |
| ---: | --- | ---: | --- | ---: | ---: | --- |
{secondary_debug_rows}

## Leituras úteis
- Se `Pcr y/z` for muito menor que a compressão direta, o problema é flambagem/inércia, não resistência axial do palito.
- Se o modo governante for `beam_column_interaction`, a imperfeição/excentricidade está amplificando a compressão.
- Se uma seção aparecer como `box` com menos de 4 palitos, ela deve ser reinterpretada como laminação compacta ou seção triangular travada; não é uma caixa real.
""",
            encoding="utf-8",
        )

        mass_breakdown = [
            {"item": "installed_stick_mass_g", "value_g": summary.get("installed_stick_mass_g")},
            {"item": "wet_glue_mass_g", "value_g": summary.get("wet_glue_mass_g")},
            {"item": "cured_glue_mass_g", "value_g": summary.get("cured_glue_mass_g")},
            {"item": "evaporated_glue_water_g", "value_g": summary.get("evaporated_glue_water_g")},
            {"item": "competition_mass_g", "value_g": summary.get("competition_mass_g")},
            {"item": "competition_mass_margin_g", "value_g": summary.get("competition_mass_margin_g")},
            {"item": "purchased_stick_mass_g", "value_g": summary.get("purchased_stick_mass_g")},
            {"item": "cutting_scrap_mass_g", "value_g": summary.get("cutting_scrap_mass_g")},
            {"item": "assembly_procurement_mass_g", "value_g": summary.get("assembly_procurement_mass_g")},
        ]
        GeometryService.write_csv(out / "mass_breakdown.csv", mass_breakdown)

        fabrication_summary = [
            {
                "purchased_blank_sticks_needed": summary.get("purchased_blank_sticks_needed"),
                "extra_sticks_for_waste": summary.get("extra_sticks_for_waste"),
                "estimated_total_sticks_with_waste": summary.get("estimated_total_sticks_with_waste"),
                "installed_stick_mass_g": summary.get("installed_stick_mass_g"),
                "purchased_stick_mass_g": summary.get("purchased_stick_mass_g"),
                "cutting_scrap_mass_g": summary.get("cutting_scrap_mass_g"),
                "wet_glue_mass_g": summary.get("wet_glue_mass_g"),
                "cured_glue_mass_g": summary.get("cured_glue_mass_g"),
            }
        ]
        GeometryService.write_csv(out / "fabrication_summary.csv", fabrication_summary)

        assumptions_md = f"""# Hipóteses e avisos
- Modelo estrutural: treliça axial linear.
- Solver tension-only: {'ativo' if bool(cfg.get('bridge', {}).get('tension_only_bracing_solver_enabled', False)) else 'inativo'}.
- Colunas: Euler/Johnson com ajuste de excentricidade simplificado.
- Seção composta: ação parcial com `eta_I`. Seções `box` com 2 ou 3 palitos são automaticamente tratadas como laminação compacta; uma caixa real exige 4+ palitos em contato e não recebe inércia de lacing/talas não modelados.
- Interação axial-flexão: verificação simplificada beam-column.
- Limitação: modelo não substitui ensaio físico.
"""
        (out / "assumptions_and_warnings.md").write_text(assumptions_md, encoding="utf-8")

        weak_glue = int(safe_float(summary.get("n_weak_glue_joints"), 0.0) or 0)
        top5_changes: List[str] = []
        if pred_break < acceptance_break:
            top5_changes.append("Aumentar capacidade dos membros críticos em compressão/flambagem.")
        if not bool(metrics.get("competition_mass_compliant", metrics.get("mass_compliant", False))):
            top5_changes.append("Reduzir massa instalada e cola curada para atender ao limite competitivo.")
        if weak_glue > 0:
            top5_changes.append("Reforçar juntas coladas com FS_glue_shear abaixo do alvo.")
        if not solver_regular:
            top5_changes.append("Eliminar singularidades/instabilidades e revisar conectividade.")
        if (safe_float(metrics.get("min_fs_primary"), 0.0) or 0.0) < float(cfg.get("analysis", {}).get("acceptance_min_primary_fs", 1.05)):
            top5_changes.append("Reforçar membros primários com FS abaixo da aceitação.")
        while len(top5_changes) < 5:
            top5_changes.append("Refinar distribuição de massa entre membros donors e críticos.")

        critical_md = "\n".join(
            [
                f"| {r.get('member_id')} | {r.get('group')} | {r.get('role')} | {safe_float(r.get('FS_min'), None) if r.get('FS_min') is not None else '—'} | {r.get('governing_mode')} |"
                for r in critical_members
            ]
        ) or "| — | — | — | — | — |"
        donor_md = "\n".join(
            [
                f"| {r.get('member_id')} | {r.get('group')} | {safe_float(r.get('FS_min'), None) if r.get('FS_min') is not None else '—'} | {safe_float(r.get('delta_mass_g'), None) if r.get('delta_mass_g') is not None else '—'} | {r.get('reason')} |"
                for r in donor_rows
            ]
        ) or "| — | — | — | — | — |"
        failures_md = "\n".join(f"- {f}" for f in failures) if failures else "- Nenhuma restrição falhou."
        changes_md = "\n".join(f"{i}. {txt}" for i, txt in enumerate(top5_changes[:5], 1))

        pipeline_trace = {}
        trace_path = str((optimization or {}).get("pipeline_trace_path") or "").strip()
        if trace_path:
            p_trace = Path(trace_path)
            if p_trace.exists():
                try:
                    pipeline_trace = json.loads(p_trace.read_text(encoding="utf-8"))
                except (TypeError, ValueError, OSError, json.JSONDecodeError):
                    pipeline_trace = {}
        stage_counts = (pipeline_trace.get("stage_candidate_counts") or {})
        stage_times = (pipeline_trace.get("stage_time_seconds") or {})
        best_ids = (pipeline_trace.get("best_candidates") or {})
        topo_before_after = pipeline_trace.get("topology_before_after") or {}

        removed_members = list((optimization or {}).get("removed_members", []) or [])
        mixed_patterns = list((optimization or {}).get("mixed_panel_patterns", []) or [])
        mass_realloc = list((optimization or {}).get("mass_reallocation_after_topology", []) or [])

        GeometryService.write_csv(out / "removed_members.csv", removed_members)
        GeometryService.write_csv(out / "mixed_panel_patterns.csv", mixed_patterns)
        GeometryService.write_csv(out / "mass_reallocation_after_topology.csv", mass_realloc)

        stage_trace_rows = []
        for st, ct in stage_counts.items():
            stage_trace_rows.append(
                {
                    "stage": st,
                    "candidate_count": ct,
                    "time_seconds": stage_times.get(st),
                }
            )
        GeometryService.write_csv(out / "pipeline_stage_trace.csv", stage_trace_rows)

        stage_trace_md = "\n".join(
            f"| {st} | {stage_counts.get(st)} | {safe_float(stage_times.get(st), None) if stage_times.get(st) is not None else '—'} |"
            for st in sorted(stage_counts.keys())
        ) or "| — | — | — |"
        removed_md = "\n".join(
            f"| {r.get('member_id')} | {r.get('reason', '—')} |"
            for r in removed_members[:20]
        ) or "| — | — |"
        mixed_md = "\n".join(
            f"| {r.get('iteration', '—')} | {r.get('panel_side_truss_pattern', '—')} |"
            for r in mixed_patterns[:10]
        ) or "| — | — |"
        mass_realloc_md = "\n".join(
            f"| {r.get('topology_freed_mass_pool_g', '—')} | {r.get('before_mass_proxy_g', '—')} | {r.get('after_mass_proxy_g', '—')} |"
            for r in mass_realloc[:10]
        ) or "| — | — | — |"

        index_md = f"""# Relatório Final

## 1. Veredito
- **{verdict}**
- Motivo: {verdict_reason}
- Carga de ruptura estimada: {pred_break:.2f} kgf
- Massa competitiva final: {competition_mass:.2f} g
- Margem de massa: {mass_margin:.2f} g
- Solver regular: {'sim' if solver_regular else 'não'}

## 2. Resumo numérico
| métrica | valor |
| --- | --- |
| load_total_kgf | {cfg.get('bridge', {}).get('load_total_kgf')} |
| target_breaking_load_kgf | {acceptance_break} |
| predicted_breaking_load_kgf | {pred_break:.3f} |
| break_margin_kgf | {pred_break - acceptance_break:.3f} |
| min_fs_primary | {metrics.get('min_fs_primary')} |
| min_fs_design | {metrics.get('min_fs_design')} |
| min_support_fs | {metrics.get('min_support_fs')} |
| min_glue_fs | {metrics.get('min_glue_fs')} |
| competition_mass_g | {metrics.get('competition_mass_g')} |
| installed_stick_mass_g | {metrics.get('installed_stick_mass_g')} |
| wet_glue_mass_g | {metrics.get('wet_glue_mass_g')} |
| cured_glue_mass_g | {metrics.get('cured_glue_mass_g')} |
| stick_budget_margin_g | {metrics.get('stick_budget_margin_g')} |
| wet_glue_budget_margin_g | {metrics.get('wet_glue_budget_margin_g')} |
| removed_members_count | {len(removed_members)} |
| mixed_panel_patterns_count | {len(mixed_patterns)} |
| topology_mass_reallocation_rows | {len(mass_realloc)} |

## 3. Massa
| item | valor (g) |
| --- | --- |
| palito instalado | {summary.get('installed_stick_mass_g')} |
| cola úmida | {summary.get('wet_glue_mass_g')} |
| cola curada | {summary.get('cured_glue_mass_g')} |
| água evaporada | {summary.get('evaporated_glue_water_g')} |
| massa competitiva | {summary.get('competition_mass_g')} |
| descarte de corte | {summary.get('cutting_scrap_mass_g')} |
| palitos comprados | {summary.get('purchased_blank_sticks_needed')} |
| massa de compra/produção | {summary.get('assembly_procurement_mass_g')} |

## 4. Top 15 membros críticos
| member_id | group | role | FS_min | governing_mode |
| --- | --- | --- | --- | --- |
{critical_md}

## 5. Top 15 donors de massa
| member_id | group | FS_min | delta_mass_g | reason |
| --- | --- | --- | --- | --- |
{donor_md}

## 6. Traço do pipeline S0..S8
| stage | candidatos | tempo (s) |
| --- | ---: | ---: |
{stage_trace_md}

Melhores candidatos por estágio:
- S2: {best_ids.get('S2')}
- S3: {best_ids.get('S3')}
- S4: {best_ids.get('S4')}
- S5: {best_ids.get('S5')}
- S6: {best_ids.get('S6')}
- S8: {best_ids.get('S8')}

Comparação antes/depois da fase topológica:
- Antes: {json.dumps(topo_before_after.get('before', {}), ensure_ascii=False)}
- Depois: {json.dumps(topo_before_after.get('after', {}), ensure_ascii=False)}

## 7. Topologia mista e remoções
Membros removidos:
| member_id | reason |
| --- | --- |
{removed_md}

Padrões mistos finais:
| iteration | panel_side_truss_pattern |
| --- | --- |
{mixed_md}

Massa realocada após topologia:
| topology_freed_mass_pool_g | before_mass_proxy_g | after_mass_proxy_g |
| --- | --- | --- |
{mass_realloc_md}

## 8. Ações de reforço
- Ver arquivo `member_sizing_actions.csv` para lista completa de ações com ganho/custo estimado.

## 9. Juntas e cola
- Cola úmida estimada: {summary.get('wet_glue_mass_g')}
- Cola curada estimada: {summary.get('cured_glue_mass_g')}
- Juntas abaixo do FS alvo: {summary.get('n_weak_glue_joints')}

## 10. Contato real da carga
- Ver `07_avaliacao_contato_carga.md` e `load_contact_assessment.csv`.
- Se o topo for arqueado, a carga distribuída exige uma sela/plataforma de transferência. Sem ela, o contato físico tende a concentrar carga no ponto mais alto.

## 11. Hipóteses do modelo
- Ver `assumptions_and_warnings.md`.

## 12. Links para gráficos
- `outputs/plots/`
- `outputs/optimization/plot_geometry_refinement.png`

## 13. Reprovação honesta
{failures_md}

Top 5 mudanças necessárias:
{changes_md}
"""
        (out / "index.md").write_text(index_md, encoding="utf-8")
        escaped_index_md = index_md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        (out / "index.html").write_text(
            f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>Relatório final — ponte de palitos</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.45; color: #111827; }}
    h1, h2 {{ margin-bottom: 0.35rem; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
    iframe {{ width: 100%; height: 720px; border: 1px solid #d1d5db; border-radius: 8px; }}
    pre {{ white-space: pre-wrap; background: #f9fafb; padding: 16px; border-radius: 8px; }}
    .links a {{ display: inline-block; margin: 0 10px 8px 0; }}
  </style>
</head>
<body>
  <h1>Relatório final — ponte de palitos</h1>
  <p>Este HTML consolida o resumo, os links principais e as visualizações interativas úteis para montagem.</p>
  <div class="links">
    <a href="00_resumo_executivo.md">Resumo executivo</a>
    <a href="01_memorial_calculo.md">Memorial de cálculo</a>
    <a href="02_guia_fabricacao.md">Guia de fabricação</a>
    <a href="04_plano_montagem_detalhado.md">Plano detalhado de montagem</a>
    <a href="../plots/16_vistas_cad_peca_a_peca.png">Vistas CAD 2D peça-a-peça</a>
    <a href="../plots/cad_subconjuntos/">CAD 2D por subconjunto</a>
    <a href="../plots/subconjuntos_html/">3D por subconjunto</a>
  </div>
  <div class="grid">
    <section>
      <h2>3D com cargas, apoios e FS/uso</h2>
      <iframe src="../plots/01_geometria_3d_fs_uso.html"></iframe>
    </section>
    <section>
      <h2>3D com prismas reais — estrutura completa</h2>
      <iframe src="../plots/02_montagem_3d_prismas_reais_interativa.html"></iframe>
    </section>
  </div>
  <h2>Resumo markdown bruto</h2>
  <pre>{escaped_index_md}</pre>
</body>
</html>""",
            encoding="utf-8",
        )

        bridge = cfg.get("bridge", {}) or {}
        mat = cfg.get("material", {}) or {}
        detail = cfg.get("detail_model", {}) or {}
        layout = cfg.get("section_layout_by_group", {}) or {}
        k_by_group = cfg.get("effective_length_factor_by_group", {}) or {}
        ms = cfg.get("member_sizing", {}) or {}

        top_layout = layout.get("top_chord", {}) or {}
        bottom_layout = layout.get("bottom_chord", {}) or {}
        vertical_layout = layout.get("vertical", {}) or {}

        executive_readme = f"""# Pacote focado de análise e fabricação

Este pacote reduz a redundância dos outputs e organiza os arquivos por função: decisão, cálculo, fabricação e auditoria.

## Resultado
- Veredito: **{verdict}**
- Ruptura estimada nominal: **{pred_break:.2f} kgf**
- Ruptura design multi-loadcase: **{design_break if design_break is not None else 'n/d'} kgf**
- Veredito design multi-loadcase: **{design_verdict or 'n/d'}**
- Meta: **{acceptance_break:.2f} kgf**
- Massa competitiva: **{competition_mass:.2f} g**
- Margem de massa: **{mass_margin:.2f} g**
- Solver regular: **{'sim' if solver_regular else 'não'}**

## Arquivos principais
1. `00_resumo_executivo.md`: decisão e próximos ajustes.
2. `01_memorial_calculo.md`: hipóteses, load cases, seções e critérios de falha.
3. `02_guia_fabricacao.md`: método construtivo, orientação dos palitos, emendas e sequência.
4. `critical_members.csv`: membros que governam o projeto.
5. `mass_breakdown.csv`: massa competitiva e massa de fabricação.
6. `symmetry_audit.csv`: auditoria de simetria das órbitas primárias.
7. `stick_pieces.csv` e `cutting_list.csv`: fabricação.

## Falhas que ainda governam
{failures_md}
"""

        calculation_memorial = f"""# Memorial de cálculo — ponte de palitos

## Geometria e edital
| item | valor |
| --- | ---: |
| vão | {bridge.get('span_mm')} mm |
| largura | {bridge.get('width_mm')} mm |
| altura central | {bridge.get('center_height_mm')} mm |
| altura nas extremidades | {bridge.get('end_height_mm')} mm |
| painel | {bridge.get('panel_mm')} mm |
| carga de projeto | {bridge.get('load_total_kgf')} kgf |
| ruptura nominal estimada | {pred_break:.2f} kgf |
| ruptura design multi-loadcase | {design_break if design_break is not None else 'n/d'} kgf |
| massa limite | {mass_limit:.2f} g |
| alvo competitivo interno | {safe_float(ms.get('competitive_mass_target_ratio'), 0.98) or 0.98:.3f} × limite |

## Material e palito
| item | valor |
| --- | ---: |
| comprimento do palito | {mat.get('stick_length_mm')} mm |
| largura do palito | {mat.get('stick_width_mm')} mm |
| espessura do palito | {mat.get('stick_thickness_mm')} mm |
| massa por palito | {mat.get('stick_mass_g')} g |
| módulo E | {mat.get('E_MPa')} MPa |
| módulo G | {mat.get('G_MPa')} MPa |

## Seções principais
| grupo | layout | orientação | espaçamento y | espaçamento z | K |
| --- | --- | --- | ---: | ---: | --- |
| top_chord | {top_layout.get('layout')} | {top_layout.get('stick_orientation')} | {top_layout.get('spacing_y_mm')} | {top_layout.get('spacing_z_mm')} | {k_by_group.get('top_chord')} |
| bottom_chord | {bottom_layout.get('layout')} | {bottom_layout.get('stick_orientation')} | {bottom_layout.get('spacing_y_mm')} | {bottom_layout.get('spacing_z_mm')} | {k_by_group.get('bottom_chord')} |
| vertical | {vertical_layout.get('layout')} | {vertical_layout.get('stick_orientation')} | {vertical_layout.get('spacing_y_mm')} | {vertical_layout.get('spacing_z_mm')} | {k_by_group.get('vertical')} |

## Critérios usados
- Solver axial linear 3D com verificações pós-processadas de tração, compressão direta, flambagem Euler/Johnson e interação axial-flexão.
- Para membros comprimidos, o risco é dominado por `EI/(KL)^2`; portanto, orientação, laminação maciça face-a-face e conexões coladas alteram diretamente a capacidade. Seções ocas e talas não explicitamente detalhadas não recebem crédito resistente.
- `left_offset` e `right_offset` são tratados como robustez; a ruptura de projeto usa os casos governantes definidos em `multi_loadcase_screening.strength_governing_cases`.
- O banzo superior modelado usa sanduíche fechado de oito palitos: quatro lâminas centrais, duas capas de fechamento e duas capas externas contínuas. A massa da cola longitudinal é contabilizada, enquanto a ação composta permanece reduzida por `eta_I`.

## Membros críticos
| member_id | group | role | FS | modo |
| --- | --- | --- | ---: | --- |
{critical_md}

## Interpretação
- Se os críticos forem `top_chord`, a revisão deve atuar no sanduíche real e no travamento lateral; não aumentar rigidez por caixa vazia não construída.
- Se os críticos forem `vertical`, verificar quais estações possuem sanduíche real e reduzir comprimento efetivo somente por travamentos fisicamente presentes.
- Se os críticos forem `diagonal`, verificar se o caso governante é nominal ou robustez deslocada antes de reforçar.
"""

        fabrication_guide = f"""# Guia de fabricação — ponte de palitos

## Princípios construtivos
1. Monte as duas treliças laterais em gabaritos rígidos e espelhados.
2. Use a mesma sequência em ambos os lados para preservar simetria.
3. Não corte, alivie ou substitua banzos/montantes primários sem atualizar o modelo.
4. Cure as submontagens antes de fechar a ponte 3D.

## Orientação dos palitos
| grupo | orientação recomendada | motivo |
| --- | --- | --- |
| banzo superior | sanduíche fechado de 8 palitos | núcleo e capas por faces coladas; capas externas contínuas; massa de cola incluída |
| banzo inferior | seção T superior: alma `edge` + mesa `flat` | alma mantém o caminho axial; mesa dá área de colagem aos nós; emendas de alma e mesa devem ser desencontradas |
| montantes construídos em x=0 e x=1300 | sanduíche fechado de 6 palitos | quatro lâminas centrais e duas capas coladas por face |
| diagonais | manter simetria e evitar emendas coincidentes | caminho de carga e estabilidade lateral |

## Emendas e cola
- Escalone emendas: não alinhe emendas de palitos no mesmo nó ou no mesmo painel.
- Use sobreposição face-a-face de {detail.get('overlap_length_mm')} mm nas continuidades modeladas; ponta-a-ponta só é admissível quando o plano indicar talas face-a-face.
- Remova excesso de cola: massa curada conta na competição e excesso raramente aumenta FS proporcionalmente.
- Pressione as juntas durante a cura; junta empenada aumenta excentricidade e reduz capacidade de flambagem.

## Sequência sugerida
1. Cortar e separar peças por grupo (`cutting_list.csv`).
2. Montar banzos superior/inferior com orientação indicada em `stick_pieces.csv`.
3. Colar treliças laterais em gabarito plano.
4. Colar montantes e diagonais, sempre aos pares simétricos.
5. Unir as duas laterais com transversais e cross-frames.
6. Adicionar bracing superior/inferior, conferindo esquadro.
7. Pesagem intermediária antes da cura final.
8. Cura completa, inspeção de alinhamento e pesagem final.

## Inspeção antes do ensaio
- Conferir se `symmetry_audit.csv` não indica diferença de `n_sticks` em grupos primários.
- Conferir se os banzos superiores não estão torcidos.
- Conferir se os pontos de carga encostam em ambos os lados da ponte.
- Conferir se a massa final fica abaixo do alvo prático, não apenas abaixo de 1000 g.
"""

        construction_checklist = f"""# Checklist de construção e ensaio

## Antes de colar
- [ ] Palitos separados por massa/comprimento semelhante.
- [ ] Gabarito de lateral esquerda e direita conferido.
- [ ] Orientação dos banzos marcada: `edge/lateral para cima`.
- [ ] Emendas planejadas com stagger.

## Durante a montagem
- [ ] Treliça lateral esquerda e direita montadas em espelho.
- [ ] Montantes centrais alinhados sem inclinação parasita.
- [ ] Banzos superiores colados em box, sem torção.
- [ ] Cross-frames instalados antes de manuseio pesado.

## Antes do teste
- [ ] Massa competitiva medida e anotada.
- [ ] Carga aplicada no nível correto: {bridge.get('load_application_level')}.
- [ ] Apoios posicionados nos contatos do edital.
- [ ] Sem palito quebrado, junta branca/solta ou empenamento visível.
"""

        detailed_method_paths = self._write_detailed_fabrication_method(out, cfg, detailed or {})
        connectivity_audit_path = self._write_connectivity_cut_audit(out, cfg, detailed or {})
        complete_ledger_path = self._write_complete_calculation_ledger(out, cfg, metrics, detailed or {})
        model_basis_path = self._write_model_basis_and_feature_audit(out, cfg, metrics, detailed or {})

        focused_manifest = {
            "purpose": "pacote executivo/fabricacao com poucos arquivos e alta densidade de informacao",
            "primary_files": [
                "00_resumo_executivo.md",
                "01_memorial_calculo.md",
                "02_guia_fabricacao.md",
                "03_checklist_construcao.md",
                "04_plano_montagem_detalhado.md",
                "04_plano_montagem_detalhado.md",
                "04_subconjuntos_montagem.md",
                "05_mapa_juntas_por_tipo.md",
                "06_sequencia_montagem.md",
                "07_avaliacao_contato_carga.md",
                "08_auditoria_secao_e_realismo.md",
                "09_auditoria_conectividade_e_cortes.md",
                "10_debug_calculos_criticos.md",
                "12_memorial_completo_membro_palito_junta.md",
                "13_validacao_modelagem_e_base_tecnica.md",
            ],
            "verdict": verdict,
            "predicted_breaking_load_kgf": pred_break,
            "competition_mass_g": competition_mass,
            "failures": failures,
        }

        (out / "00_resumo_executivo.md").write_text(executive_readme, encoding="utf-8")
        (out / "01_memorial_calculo.md").write_text(calculation_memorial, encoding="utf-8")
        (out / "02_guia_fabricacao.md").write_text(fabrication_guide, encoding="utf-8")
        (out / "03_checklist_construcao.md").write_text(construction_checklist, encoding="utf-8")
        (out / "focused_outputs_manifest.json").write_text(
            json.dumps(focused_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return {
            "index_md": str(out / "index.md"),
            "index_html": str(out / "index.html"),
            "executive_readme_md": str(out / "00_resumo_executivo.md"),
            "calculation_memorial_md": str(out / "01_memorial_calculo.md"),
            "fabrication_guide_md": str(out / "02_guia_fabricacao.md"),
            "construction_checklist_md": str(out / "03_checklist_construcao.md"),
            **detailed_method_paths,
            "connectivity_cut_audit_md": connectivity_audit_path,
            "complete_member_piece_joint_memorial_md": complete_ledger_path,
            "model_basis_and_feature_audit_md": model_basis_path,
            "focused_outputs_manifest_json": str(out / "focused_outputs_manifest.json"),
            "executive_summary_json": str(out / "executive_summary.json"),
            "critical_members_csv": str(out / "critical_members.csv"),
            "mass_breakdown_csv": str(out / "mass_breakdown.csv"),
            "candidate_ranking_csv": str(out / "candidate_ranking.csv"),
            "member_sizing_actions_csv": str(out / "member_sizing_actions.csv"),
            "fabrication_summary_csv": str(out / "fabrication_summary.csv"),
            "assumptions_md": str(out / "assumptions_and_warnings.md"),
            "removed_members_csv": str(out / "removed_members.csv"),
            "mixed_panel_patterns_csv": str(out / "mixed_panel_patterns.csv"),
            "mass_reallocation_after_topology_csv": str(out / "mass_reallocation_after_topology.csv"),
            "pipeline_stage_trace_csv": str(out / "pipeline_stage_trace.csv"),
            "section_layout_audit_md": str(out / "08_auditoria_secao_e_realismo.md"),
            "section_layout_audit_csv": str(out / "section_layout_audit.csv"),
        }
