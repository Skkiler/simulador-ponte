from __future__ import annotations

from typing import Any, Dict, List

from src.core.numeric import safe_float
from src.services.mass_guard import resolve_mass_limits


def fmt_float(
    value: Any,
    decimals: int = 2,
    default: str = "—",
    suffix: str = "",
) -> str:
    """
    Formata número de forma segura para texto.
    """
    v = safe_float(value, None)

    if v is None:
        return default

    return f"{v:.{decimals}f}{suffix}"


def is_below(value: Any, limit: float) -> bool:
    """
    Retorna True apenas se value for numérico válido e menor que limit.
    """
    v = safe_float(value, None)
    return v is not None and v < limit


def is_above(value: Any, limit: float) -> bool:
    """
    Retorna True apenas se value for numérico válido e maior que limit.
    """
    v = safe_float(value, None)
    return v is not None and v > limit


class RecommendationService:
    """Gera diagnóstico textual e sugestões de projeto a partir dos resultados."""

    def build(
        self,
        cfg: Dict,
        member_checks: List[Dict],
        support_checks: List[Dict],
        solver_summary: Dict,
        detailed: Dict | None = None,
        optimization: Dict | None = None,
    ) -> Dict[str, List[str] | str]:
        def topo_pt(value: Any) -> str:
            raw = str(value or "").strip().lower()
            mp = {
                "parker_plateau": "platô",
                "triangular_peak": "pontiagudo/triangular",
                "shallow_arch": "arco",
                "shallow_arch_faceted": "arco",
                "flat": "reto",
            }
            return mp.get(raw, str(value or "—"))

        detailed = detailed or {}
        optimization = optimization or {}
        mat = cfg.get("material", {})
        detail_cfg = cfg.get("detail_model", {})
        t_cap = float(mat.get("tension_capacity_per_stick_kgf", 72.0))
        c_cap = max(1.0e-6, float(mat.get("compression_capacity_two_sticks_kgf", 11.0)) / 2.0)
        tc_ratio = t_cap / c_cap

        primary = [
            r
            for r in member_checks
            if r.get("member_role") == "primary"
        ]

        stabilizers = [
            r
            for r in member_checks
            if r.get("member_role") == "stabilizer"
        ]

        critical_primary = [
            r
            for r in primary
            if r.get("risk_flag") == "CRITICAL"
        ]

        low_primary = [
            r
            for r in primary
            if r.get("risk_flag") == "LOW_MARGIN"
        ]

        support_critical = [
            r
            for r in support_checks
            if r.get("risk_flag") == "CRITICAL"
        ]

        uplift = [
            r
            for r in support_checks
            if r.get("risk_flag") == "UPLIFT_NO_CONTACT"
        ]

        stab_compression = [
            r
            for r in stabilizers
            if r.get("risk_flag") == "STABILIZER_COMPRESSION"
        ]

        dsum = detailed.get("summary", {}) or {}
        limits = resolve_mass_limits(cfg)

        weak_glue = []

        for r in detailed.get("weakest_glue_joints", []) or []:
            if is_below(r.get("FS_glue_shear"), 2.0):
                weak_glue.append(r)
        weak_glue_count = int(
            safe_float(dsum.get("n_weak_glue_joints"), len(weak_glue)) or 0
        )

        reinf = detailed.get("reinforcement_suggestions", []) or []

        equilibrium_error = safe_float(
            solver_summary.get("equilibrium_error_N"),
            0.0,
        )

        summary = [
            (
                f"Solver axial: {solver_summary.get('status', 'desconhecido')}. "
                f"Erro de equilíbrio vertical: {equilibrium_error:.3e} N."
            )
        ]
        span_mm = float(cfg.get("bridge", {}).get("span_mm", 1200.0))
        height_mm = max(40.0, float(cfg.get("bridge", {}).get("center_height_mm", 300.0)))
        load_kgf = float(cfg.get("bridge", {}).get("load_total_kgf", 120.0))
        n_top = int(cfg.get("member_sticks_by_group", {}).get("top_chord", 1))
        c_per_stick = max(0.1, float(mat.get("compression_capacity_two_sticks_kgf", 11.0)) / 2.0)
        chord_force_aprox = (0.5 * load_kgf * span_mm / 4.0) / height_mm
        top_cap_aprox = n_top * c_per_stick
        summary.append(
            "Checagem simplificada M/h: "
            f"força de banzo por treliça ≈ {chord_force_aprox:.1f} kgf, "
            f"capacidade direta aproximada do banzo (n={n_top}) ≈ {top_cap_aprox:.1f} kgf."
        )
        if tc_ratio >= 6.0:
            summary.append(
                "Material com alta razão tração/compressão detectada. "
                "Sob carga vertical, topologias tipo Pratt tendem a ser mais eficientes "
                "porque deslocam diagonais longas para tração."
            )
        summary.append(
            "Modelos de emenda ativos: "
            f"tração={detail_cfg.get('tension_joint_model', '—')}, "
            f"compressão={detail_cfg.get('compression_joint_model', '—')}, "
            f"sobreposição={fmt_float(detail_cfg.get('overlap_length_mm'), 1, '—', ' mm')}."
        )

        if critical_primary:
            summary.append(
                f"Há {len(critical_primary)} membros principais com FS < 1,0. "
                "O projeto precisa de redimensionamento antes de considerar a carga segura."
            )
        elif low_primary:
            summary.append(
                f"Não há falha imediata nos membros principais, mas {len(low_primary)} "
                "membros têm margem baixa (1 <= FS < 2)."
            )
        else:
            summary.append(
                "Os membros principais passaram na checagem preliminar axial/flambagem "
                "para a carga informada."
            )

        if stab_compression:
            summary.append(
                f"{len(stab_compression)} estabilizadores aparecem comprimidos/esbeltos. "
                "Interprete-os como travamento/tension-only ou reforce-os se forem usados "
                "como barras comprimidas reais."
            )

        if uplift:
            summary.append(
                f"{len(uplift)} pontos de apoio perderam contato no modelo unilateral. "
                "Isso é coerente com apoio livre, mas concentra reação nos apoios internos."
            )

        if support_critical:
            summary.append(
                f"{len(support_critical)} apoios ativos excedem a capacidade simplificada "
                "adotada. Reforce a região de apoio e aumente área de contato."
            )

        if dsum:
            estimated_sticks = dsum.get("estimated_total_sticks_with_waste", "—")
            estimated_mass = fmt_float(
                dsum.get("estimated_total_mass_g"),
                decimals=1,
                default="—",
                suffix=" g",
            )

            summary.append(
                f"Modelo peça-a-peça: {estimated_sticks} palitos estimados com perdas "
                f"e massa total de {estimated_mass}."
            )
            summary.append(
                "Limites de massa: "
                f"nominal={fmt_float(limits.get('nominal_limit_g'), 0, '—', ' g')}, "
                f"planner={fmt_float(limits.get('planner_limit_g'), 0, '—', ' g')}, "
                f"material={fmt_float(limits.get('material_limit_g'), 0, '—', ' g')}, "
                f"efetivo={fmt_float(limits.get('effective_limit_g'), 0, '—', ' g')}."
            )

            if is_below(dsum.get("mass_margin_g"), 0.0):
                summary.append(
                    "A massa estimada excede o limite configurado; reduza reforços "
                    "ou revise a geometria."
                )

        if weak_glue_count > 0:
            summary.append(
                f"Há {weak_glue_count} juntas coladas com FS < 2,0 ao cisalhamento "
                "estimado. Aumente a sobreposição ou use tala dupla nesses pontos."
            )

        suggestions: List[str] = []
        proposal_description = ""
        comparison_notes: List[str] = []

        primary_sorted = sorted(
            primary,
            key=lambda r: safe_float(r.get("FS_min"), 1.0e99) or 1.0e99,
        )

        for r in primary_sorted[:8]:
            if r.get("risk_flag") not in {"CRITICAL", "LOW_MARGIN"}:
                continue

            member_id = r.get("member_id", "—")
            group = r.get("group", "—")
            gov = str(r.get("governing_mode", "") or "")
            fs_txt = fmt_float(r.get("FS_min"), decimals=2, default="—")

            if "buckling" in gov:
                suggestions.append(
                    f"Membro {member_id} ({group}, FS={fs_txt}): crítico por {gov}. "
                    "Aumente a inércia da seção, reduza o comprimento livre ou adicione "
                    "travamento intermediário."
                )
            elif gov == "tension_capacity":
                suggestions.append(
                    f"Membro {member_id} ({group}, FS={fs_txt}): crítico por tração. "
                    "Aumente o número de palitos contínuos ou melhore emendas/talas."
                )
            elif gov == "compression_direct":
                suggestions.append(
                    f"Membro {member_id} ({group}, FS={fs_txt}): crítico por compressão direta. "
                    "Aumente a área efetiva ou distribua melhor o esforço."
                )
            else:
                suggestions.append(
                    f"Membro {member_id} ({group}, FS={fs_txt}): margem baixa. "
                    "Reforce ou redistribua a carga."
                )

        if support_critical:
            suggestions.append(
                "Apoios: aumente o grupo support_pad, acrescente travessas inferiores "
                "em x=0 e x=span, ou distribua o contato em área maior."
            )

        if stab_compression:
            suggestions.append(
                "Contraventamentos: mantenha X duplo, mas trate o par como tension-only; "
                "se trabalhar comprimido de verdade, use dois palitos ou reduza o vão livre."
            )

        for r in reinf[:12]:
            member_id = r.get("member_id", "—")
            group = r.get("group", "—")
            action = r.get("suggested_action", "revisar membro")
            fs_txt = fmt_float(r.get("FS_min"), decimals=2, default="—")

            suggestions.append(
                f"Membro {member_id} ({group}, FS={fs_txt}): {action}"
            )

        mass_margin = safe_float(dsum.get("mass_margin_g"), None)

        if mass_margin is not None and mass_margin < 80.0:
            suggestions.append(
                "Massa: margem pequena. Procure reforços removíveis em estabilizadores "
                "e membros de baixa solicitação."
            )

        if weak_glue_count > 0:
            suggestions.append(
                "Colagens: aumente a sobreposição dos membros listados em "
                "weakest_glue_joints.csv ou adote talas simétricas."
            )

        # Recomendações de montagem e detalhamento baseadas no modo de falha dominante.
        compression_critical = [
            r for r in primary_sorted[:20]
            if str(r.get("governing_mode", "")).startswith("compression")
            or "buckling" in str(r.get("governing_mode", ""))
        ]
        overlap_mm = safe_float(cfg.get("detail_model", {}).get("overlap_length_mm"), 30.0) or 30.0
        comp_joint_model = str(cfg.get("detail_model", {}).get("compression_joint_model", "double_lap_reinforced"))

        if compression_critical:
            suggestions.append(
                "Montagem de membros comprimidos: priorize seção caixa/afastada em banzo superior e verticais, "
                "com talas simétricas (dupla sobreposição) para reduzir excentricidade local."
            )
            if overlap_mm < 25.0:
                suggestions.append(
                    "Sobreposição atual curta para emendas comprimidas. Considere 25 a 40 mm para melhorar transferência."
                )
            if comp_joint_model not in {"double_lap_reinforced", "double_lap"}:
                suggestions.append(
                    "Modelo de junta comprimida conservador. Considere `double_lap` ou `double_lap_reinforced`."
                )
            suggestions.append(
                "Se a massa estiver no limite, aumente o painel (menos barras totais) e redirecione palitos para grupos críticos."
            )

        best = optimization.get("best")
        stage_counts = optimization.get("stage_counts") or {}
        final_variants = optimization.get("final_variants") or {}

        if best:
            max_mass = float(limits["effective_limit_g"])
            best_mass = safe_float(best.get("mass_g"), None)
            b_height = fmt_float(
                best.get("center_height_mm"),
                decimals=0,
                default="—",
                suffix=" mm",
            )

            b_panel = fmt_float(
                best.get("panel_mm"),
                decimals=0,
                default="—",
                suffix=" mm",
            )
            b_width = fmt_float(
                best.get("width_mm"),
                decimals=0,
                default="—",
                suffix=" mm",
            )
            b_span = fmt_float(
                best.get("span_mm"),
                decimals=0,
                default="—",
                suffix=" mm",
            )
            b_side = best.get("side_truss_type", "—")
            b_int = best.get("internal_truss_type", "—")
            b_top_chord = best.get("top_chord_truss_type", "—")
            b_bottom_chord = best.get("bottom_chord_truss_type", "—")

            proposal_description = (
                f"Proposta selecionada automaticamente: treliça lateral {b_side}, "
                f"topo {topo_pt(best.get('top_profile', '—'))}, treliça interna {b_int}, "
                f"banzo superior {b_top_chord}, banzo inferior {b_bottom_chord}, "
                f"vão {b_span}, largura {b_width}, altura {b_height}, painel {b_panel}."
            )
            summary.insert(1, proposal_description)

            suggestions.append(
                "A proposta selecionada já foi calculada e detalhada. "
                "Use os arquivos de saída para construção e comparação entre versões final_ideal/min/max."
            )

            pred_break = fmt_float(
                best.get("predicted_breaking_load_kgf"),
                decimals=1,
                default="—",
                suffix=" kgf",
            )
            suggestions.append(
                f"Carga de ruptura estimada do melhor candidato: {pred_break}."
            )
            if best_mass is not None and best_mass > max_mass + 1e-6:
                suggestions.append(
                    f"Atenção: a proposta excede a massa máxima ({best_mass:.1f} g > {max_mass:.1f} g) "
                    "e não deve ser aceita como solução final."
                )

            if stage_counts:
                if "S1_macro_candidates" in stage_counts:
                    suggestions.append(
                        "Funil executado: "
                        f"S1={stage_counts.get('S1_macro_candidates', 0)}, "
                        f"S2={stage_counts.get('S2_fast_screening_candidates', 0)}→{stage_counts.get('S2_fast_screening_top_k', 0)}, "
                        f"S3={stage_counts.get('S3_multi_loadcase_candidates', 0)}→{stage_counts.get('S3_multi_loadcase_top_k', 0)}, "
                        f"S4={stage_counts.get('S4_geometry_refinement_candidates', 0)}→{stage_counts.get('S4_geometry_refinement_top_k', 0)}, "
                        f"S5={stage_counts.get('S5_member_sizing_candidates', 0)}, "
                        f"S6={stage_counts.get('S6_topology_candidates', 0)}, "
                        f"S7={stage_counts.get('S7_fabrication_candidates', 0)}, "
                        f"S8={stage_counts.get('S8_final_validation_candidates', 0)}, "
                        f"solves={stage_counts.get('solves_total', 0)}."
                    )
                else:
                    suggestions.append(
                        f"Busca executada: S0-gerados={stage_counts.get('stage0_generated', 0)}, "
                        f"S0-aprovados={stage_counts.get('stage0_prefilter_passed', 0)}, "
                        f"S1-válidos={stage_counts.get('stage1', 0)}, "
                        f"S2A={stage_counts.get('stage2a_selected', 0)}, "
                        f"S2B={stage_counts.get('stage2b_evaluated', 0)}, "
                        f"S2-únicos={stage_counts.get('stage2_unique', stage_counts.get('stage2', 0))}, "
                        f"S3={stage_counts.get('stage3', 0)}, "
                        f"S4={stage_counts.get('stage4', 0)}."
                    )

            if final_variants:
                vmin = final_variants.get("min") or {}
                vmax = final_variants.get("max") or {}
                videal = final_variants.get("ideal") or {}
                suggestions.append(
                    "Versões finais conservadoras geradas: "
                    f"MIN (FS={fmt_float(vmin.get('min_fs_primary'), 2)}, "
                    f"massa={fmt_float(vmin.get('mass_g'), 1, '—', ' g')}) e "
                    f"MAX (FS={fmt_float(vmax.get('min_fs_primary'), 2)}, "
                    f"massa={fmt_float(vmax.get('mass_g'), 1, '—', ' g')})."
                )
                comparison_notes.append(
                    "IDEAL: "
                    f"FS={fmt_float(videal.get('min_fs_primary'), 2)} | "
                    f"ruptura={fmt_float(videal.get('predicted_breaking_load_kgf'), 1, '—', ' kgf')} | "
                    f"massa={fmt_float(videal.get('mass_g'), 1, '—', ' g')}"
                )
                comparison_notes.append(
                    "MIN: "
                    f"FS={fmt_float(vmin.get('min_fs_primary'), 2)} | "
                    f"ruptura={fmt_float(vmin.get('predicted_breaking_load_kgf'), 1, '—', ' kgf')} | "
                    f"massa={fmt_float(vmin.get('mass_g'), 1, '—', ' g')}"
                )
                comparison_notes.append(
                    "MAX: "
                    f"FS={fmt_float(vmax.get('min_fs_primary'), 2)} | "
                    f"ruptura={fmt_float(vmax.get('predicted_breaking_load_kgf'), 1, '—', ' kgf')} | "
                    f"massa={fmt_float(vmax.get('mass_g'), 1, '—', ' g')}"
                )

        if not suggestions:
            suggestions.append(
                "Próximo refinamento: calibrar módulo E dos palitos por ensaio simples "
                "e comparar o modelo NumPy com o Frame3DD."
            )

        return {
            "summary": "\n".join(summary),
            "suggestions": suggestions,
            "proposal_description": proposal_description,
            "comparison_notes": comparison_notes,
        }
