from __future__ import annotations

import math
from typing import Any, Dict, List


def safe_float(value: Any, default: float | None = None) -> float | None:
    """
    Converte para float sem quebrar com None, string vazia, texto, NaN ou infinito.

    Use sempre que o valor vier de:
    - CSV;
    - JSON;
    - pós-processamento;
    - campos opcionais;
    - resultados de cálculo que podem não existir para certos membros.
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
        detailed = detailed or {}
        optimization = optimization or {}

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

        weak_glue = []

        for r in detailed.get("weakest_glue_joints", []) or []:
            if is_below(r.get("FS_glue_shear"), 2.0):
                weak_glue.append(r)

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

            if is_below(dsum.get("mass_margin_g"), 0.0):
                summary.append(
                    "A massa estimada excede o limite configurado; reduza reforços "
                    "ou revise a geometria."
                )

        if weak_glue:
            summary.append(
                f"Há {len(weak_glue)} juntas coladas com FS < 2,0 ao cisalhamento "
                "estimado. Aumente a sobreposição ou use tala dupla nesses pontos."
            )

        suggestions: List[str] = []

        primary_sorted = sorted(
            primary,
            key=lambda r: safe_float(r.get("FS_min"), 1.0e99) or 1.0e99,
        )

        for r in primary_sorted[:10]:
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

        if weak_glue:
            suggestions.append(
                "Colagens: aumente a sobreposição dos membros listados em "
                "weakest_glue_joints.csv ou adote talas simétricas."
            )

        best = optimization.get("best")
        stage_counts = optimization.get("stage_counts") or {}
        final_variants = optimization.get("final_variants") or {}

        if best:
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

            suggestions.append(
                f"Proposta automática: testar {best.get('truss_type', best.get('side_truss_type', '—'))} / "
                f"{best.get('top_profile', '—')} com altura {b_height} "
                f"e painel {b_panel}. Use outputs/optimization/recommended_config.json."
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

            if stage_counts:
                suggestions.append(
                    f"Busca executada: S1={stage_counts.get('stage1', 0)}, "
                    f"S2={stage_counts.get('stage2', 0)}, "
                    f"S3={stage_counts.get('stage3', 0)}, "
                    f"S4={stage_counts.get('stage4', 0)}."
                )

            if final_variants:
                vmin = final_variants.get("min") or {}
                vmax = final_variants.get("max") or {}
                suggestions.append(
                    "Versões finais conservadoras geradas: "
                    f"MIN (FS={fmt_float(vmin.get('min_fs_primary'), 2)}, "
                    f"massa={fmt_float(vmin.get('mass_g'), 1, '—', ' g')}) e "
                    f"MAX (FS={fmt_float(vmax.get('min_fs_primary'), 2)}, "
                    f"massa={fmt_float(vmax.get('mass_g'), 1, '—', ' g')})."
                )

        if not suggestions:
            suggestions.append(
                "Próximo refinamento: calibrar módulo E dos palitos por ensaio simples "
                "e comparar o modelo NumPy com o Frame3DD."
            )

        return {
            "summary": "\n".join(summary),
            "suggestions": suggestions,
        }
