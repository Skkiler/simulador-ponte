from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.core.numeric import safe_float
from src.services.mass_guard import resolve_mass_limits


class ReportService:
    @staticmethod
    def _topo_pt(value: Any) -> str:
        raw = str(value or "").strip().lower()
        mp = {
            "parker_plateau": "platô",
            "triangular_peak": "pontiagudo/triangular",
            "shallow_arch": "arco",
            "shallow_arch_faceted": "arco",
            "flat": "reto",
        }
        return mp.get(raw, str(value or "—"))

    def _fmt(self, value: Any, decimals: int = 2, suffix: str = "") -> str:
        v = safe_float(value, None)
        if v is None:
            return "—"
        return f"{v:.{decimals}f}{suffix}"
    
    @staticmethod
    def _stick_dimension_rule(cfg: Dict) -> str:
        rules = cfg.get("competition_rules", {}) or {}
        if not bool(rules.get("enforce_nominal_stick_dimensions", False)):
            return "configurável; dimensões devem ser > 0"

        parts = []
        for key, tol_key, label in (
            ("required_stick_length_mm", "stick_length_tolerance_mm", "compr."),
            ("required_stick_thickness_mm", "stick_thickness_tolerance_mm", "esp."),
            ("required_stick_width_mm", "stick_width_tolerance_mm", "larg."),
        ):
            required = rules.get(key)
            if required is None:
                parts.append(f"{label}=config.")
            else:
                parts.append(f"{label}={float(required):.2f}±{float(rules.get(tol_key, 0.2)):.2f} mm")
        return "; ".join(parts)

    @staticmethod
    def _stick_dimension_ok(cfg: Dict) -> bool:
        mat = cfg.get("material", {}) or {}
        rules = cfg.get("competition_rules", {}) or {}
        length = float(mat.get("stick_length_mm", 0.0))
        thickness = float(mat.get("stick_thickness_mm", 0.0))
        width = float(mat.get("stick_width_mm", 0.0))
        if length <= 0.0 or thickness <= 0.0 or width <= 0.0:
            return False
        if not bool(rules.get("enforce_nominal_stick_dimensions", False)):
            return True

        checks = (
            (length, rules.get("required_stick_length_mm"), float(rules.get("stick_length_tolerance_mm", 0.5))),
            (thickness, rules.get("required_stick_thickness_mm"), float(rules.get("stick_thickness_tolerance_mm", 0.2))),
            (width, rules.get("required_stick_width_mm"), float(rules.get("stick_width_tolerance_mm", 0.2))),
        )
        return all(required is None or abs(value - float(required)) <= tol for value, required, tol in checks)

    def _criterios_edital(self, cfg: Dict, metrics: Dict, detailed: Dict | None = None) -> List[Dict]:
        detailed = detailed or {}
        dsum = detailed.get("summary", {}) or {}
        b = cfg.get("bridge", {})
        m = cfg.get("material", {})

        span = float(b.get("span_mm", 0.0))
        left_support = abs(float(b.get("left_support_overhang_mm", 0.0)))
        right_support = abs(float(b.get("right_support_overhang_mm", 0.0)))
        width = float(b.get("width_mm", 0.0))
        height = float(b.get("center_height_mm", 0.0))
        mass_g = safe_float(
            dsum.get("competition_mass_g"),
            safe_float(
                dsum.get("estimated_total_mass_g"),
                safe_float(metrics.get("competition_mass_g"), safe_float(metrics.get("estimated_total_mass_g"), 0.0)),
            ),
        ) or 0.0
        limits = resolve_mass_limits(cfg)
        effective_limit = float(limits["effective_limit_g"])
        nominal_limit = float(limits["nominal_limit_g"])
        if abs(effective_limit - nominal_limit) <= 1e-6:
            mass_rule = f"máximo {nominal_limit:.0f} g"
        else:
            mass_rule = f"máximo {effective_limit:.0f} g (efetivo) / {nominal_limit:.0f} g (nominal)"

        rows = [
            {
                "critério": "Vão obrigatório",
                "valor": f"{span:.1f} mm",
                "regra": "1200 mm",
                "conforme": abs(span - 1200.0) <= 1e-6,
            },
            {
                "critério": "Apoio máximo por lado",
                "valor": f"E={left_support:.1f} mm | D={right_support:.1f} mm",
                "regra": "até 100 mm por lado",
                "conforme": left_support <= 100.0 + 1e-6 and right_support <= 100.0 + 1e-6,
            },
            {
                "critério": "Largura da ponte",
                "valor": f"{width:.1f} mm",
                "regra": "entre 100 e 200 mm",
                "conforme": 100.0 - 1e-6 <= width <= 200.0 + 1e-6,
            },
            {
                "critério": "Altura mínima",
                "valor": f"{height:.1f} mm",
                "regra": "mínimo 50 mm",
                "conforme": height >= 50.0 - 1e-6,
            },
            {
                "critério": "Peso máximo",
                "valor": f"{mass_g:.1f} g",
                "regra": mass_rule,
                "conforme": mass_g <= effective_limit + 1e-6,
            },
            {
                "critério": "Dimensão do palito",
                "valor": (
                    f"{float(m.get('stick_length_mm', 0.0)):.1f} x "
                    f"{float(m.get('stick_thickness_mm', 0.0)):.2f} x "
                    f"{float(m.get('stick_width_mm', 0.0)):.2f} mm"
                ),
                # Reference for stick dimensions uses 115 mm × 1.5 mm × 7.0 mm
                "regra": self._stick_dimension_rule(cfg),
                "conforme": self._stick_dimension_ok(cfg),
            },
            {
                "critério": "Compressão 1 palito",
                "valor": f"{float(m.get('compression_capacity_one_stick_kgf', 0.0)):.2f} kgf",
                "regra": "mínimo 4,0 kgf",
                "conforme": float(m.get("compression_capacity_one_stick_kgf", 0.0)) >= 4.0 - 1e-6,
            },
            {
                "critério": "Compressão 2 palitos colados",
                "valor": f"{float(m.get('compression_capacity_two_sticks_kgf', 0.0)):.2f} kgf",
                "regra": "mínimo 11,0 kgf",
                "conforme": float(m.get("compression_capacity_two_sticks_kgf", 0.0)) >= 11.0 - 1e-6,
            },
            {
                "critério": "Tração por palito",
                "valor": f"{float(m.get('tension_capacity_per_stick_kgf', 0.0)):.2f} kgf",
                "regra": "mínimo 72,0 kgf",
                "conforme": float(m.get("tension_capacity_per_stick_kgf", 0.0)) >= 72.0 - 1e-6,
            },
        ]
        return rows

    def write_markdown(
        self,
        cfg: Dict,
        metrics: Dict,
        recommendations: Dict,
        out_path: str | Path,
        detailed: Dict | None = None,
    ) -> Path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        detailed = detailed or {}
        dsum = detailed.get("summary", {}) or {}
        limits = resolve_mass_limits(cfg)
        weakest = detailed.get("weakest_members", []) or []
        suggestions = recommendations.get("suggestions", []) or []
        summary = recommendations.get("summary", "") or ""

        span = float(cfg["bridge"]["span_mm"])
        width = float(cfg["bridge"]["width_mm"])
        h = float(cfg["bridge"]["center_height_mm"])
        panel = float(cfg["bridge"]["panel_mm"])
        load_kgf = float(cfg["bridge"]["load_total_kgf"])

        min_fs = safe_float(
            metrics.get("min_fs_member_design"),
            safe_float(metrics.get("min_fs_primary"), None),
        )
        pred_break = safe_float(metrics.get("predicted_breaking_load_kgf"), None)
        if pred_break is None and min_fs is not None:
            pred_break = load_kgf * min_fs
        collapse_est = pred_break
        rupture = metrics.get("rupture_details", {}) or {}
        governing_ls = str(
            metrics.get("governing_limit_state", rupture.get("governing_limit_state", "—"))
            or "—"
        )
        fs_member = safe_float(metrics.get("min_fs_member_design"), safe_float(metrics.get("min_fs_design"), None))
        fs_support = safe_float(metrics.get("min_fs_support"), safe_float(metrics.get("min_support_fs"), None))
        fs_glue = safe_float(metrics.get("min_fs_glue"), safe_float(metrics.get("min_glue_fs"), None))
        break_by_members = safe_float(metrics.get("predicted_breaking_load_by_members_kgf"), None)
        break_by_supports = safe_float(metrics.get("predicted_breaking_load_by_supports_kgf"), None)
        break_by_glue = safe_float(metrics.get("predicted_breaking_load_by_glue_kgf"), None)

        criter = self._criterios_edital(cfg, metrics, detailed)
        criter_lines = "\n".join(
            f"| {r['critério']} | {r['valor']} | {r['regra']} | {'Sim' if r['conforme'] else 'Não'} |"
            for r in criter
        )

        weak_lines = []
        for row in weakest[:10]:
            fs_value = row.get("FS_min")
            if fs_value is None:
                fs_value = row.get("FS_min_global")
            mode = row.get("governing_mode")
            if not mode:
                mode = row.get("governing_mode_global", "—")
            weak_lines.append(
                "| {member} | {group} | {fs} | {mode} |".format(
                    member=row.get("member_id", "—"),
                    group=row.get("group", "—"),
                    fs=self._fmt(fs_value, 3, ""),
                    mode=mode,
                )
            )
        weak_table = "\n".join(weak_lines) if weak_lines else "| — | — | — | — |"

        suggestions_md = "\n".join(f"- {s}" for s in suggestions) if suggestions else "- Sem recomendações adicionais."

        quarter_used = bool(metrics.get("quarter_model_used"))
        quarter_fallback = str(metrics.get("quarter_model_fallback_reason") or "").strip()
        quarter_text = (
            "Projeto analisado por 1/4 e replicado por simetria."
            if quarter_used
            else "Análise executada no modelo completo."
        )
        if (not quarter_used) and quarter_fallback:
            quarter_text += f" Fallback: {quarter_fallback}."
        symmetry_text = (
            "Simetria estrutural imposta."
            if bool(cfg.get("analysis", {}).get("enforce_symmetry", True))
            else "Simetria estrutural não imposta (modo avançado)."
        )

        conn = detailed.get("connection_plan", []) or []
        conn_light = sum(1 for r in conn if str(r.get("recommended_joint_model", "")).startswith("single_lap"))
        conn_mod = sum(1 for r in conn if str(r.get("recommended_joint_model", "")) == "double_lap")
        conn_strong = sum(1 for r in conn if str(r.get("recommended_joint_model", "")) == "double_lap_reinforced")
        splice_report = detailed.get("splice_stagger_report", {}) or {}
        aligned_critical = int(splice_report.get("critical_clusters", splice_report.get("critical_aligned_count", 0)) or 0)
        member_detail = detailed.get("member_detail_checks", []) or []
        sizing_changed = sum(
            1
            for r in member_detail
            if int(r.get("n_sticks_recommended", r.get("n_sticks_current", 0)) or 0)
            != int(r.get("n_sticks_current", 0) or 0)
        )
        governing_note = ""
        if str(governing_ls).strip().lower() == "glue":
            governing_note = (
                f"- Ruptura governada por cola: FS_glue = {self._fmt(fs_glue, 3, '')}; "
                f"carga prevista = {self._fmt(break_by_glue, 2, ' kgf')}."
            )
        elif str(governing_ls).strip().lower() == "support":
            governing_note = (
                f"- Ruptura governada por apoio: FS_support = {self._fmt(fs_support, 3, '')}; "
                f"carga prevista = {self._fmt(break_by_supports, 2, ' kgf')}."
            )
        elif str(governing_ls).strip().lower().startswith("member"):
            governing_note = (
                f"- Ruptura governada por membros: FS_member = {self._fmt(fs_member, 3, '')}; "
                f"carga prevista = {self._fmt(break_by_members, 2, ' kgf')}."
            )

        md = f"""# Relatório técnico automático - ponte de palitos

## 1) Projeto detalhado selecionado
- Tipologia lateral: {cfg['bridge'].get('side_truss_type', '—')}
- Perfil de topo: {self._topo_pt(cfg['bridge'].get('top_profile', '—'))}
- Treliça interna: {cfg['bridge'].get('internal_truss_type', '—')}
- Treliça do banzo superior: {cfg['bridge'].get('top_chord_truss_type', '—')}
- Treliça do banzo inferior: {cfg['bridge'].get('bottom_chord_truss_type', '—')}
- Vão final: {span:.1f} mm
- Largura final: {width:.1f} mm
- Altura final: {h:.1f} mm
- Painel final: {panel:.1f} mm
- Carga de projeto: {load_kgf:.2f} kgf

## 2) Análise estrutural
- Status do solver: {metrics.get('solver_status', '—')}
- Erro de equilíbrio vertical: {self._fmt(metrics.get('equilibrium_error_N'), 4, ' N')}
- Menor FS (membros principais): {self._fmt(metrics.get('min_fs_primary'), 3, '')}
- Menor FS (todos os membros): {self._fmt(metrics.get('min_fs_all'), 3, '')}
- Apoios ativos: {metrics.get('n_active_supports', '—')}
- Apoios com perda de contato: {metrics.get('n_uplift_supports', '—')}

## 3) Simetria e quarter-model
- Simetria estrutural: {symmetry_text}
- Quarter-model: {quarter_text}
- Simetria construtiva: emendas podem ser desalinhadas intencionalmente entre quadrantes/lâminas.

## 4) Memorial de cálculo (síntese)
- Hipótese base: modelo linear axial 3D.
- Carga prevista de colapso: {self._fmt(collapse_est, 2, ' kgf')}
- Carga prevista de ruptura: {self._fmt(pred_break, 2, ' kgf')}
- FS membros (design): {self._fmt(fs_member, 3, '')}
- FS apoios: {self._fmt(fs_support, 3, '')}
- FS cola: {self._fmt(fs_glue, 3, '')}
- Ruptura governante: {governing_ls}
- Ruptura por membros: {self._fmt(break_by_members, 2, ' kgf')}
- Ruptura por apoios: {self._fmt(break_by_supports, 2, ' kgf')}
- Ruptura por cola: {self._fmt(break_by_glue, 2, ' kgf')}
- Observação do estado limite governante: {governing_note or '—'}
- Verificações consideradas: tração, compressão direta, flambagem por Euler e reações de apoio.

## 5) Peso, dimensões e consumo
- Massa de palitos instalados: {self._fmt(dsum.get('installed_stick_mass_g'), 1, ' g')}
- Massa de cola úmida: {self._fmt(dsum.get('wet_glue_mass_g'), 1, ' g')}
- Massa de cola curada: {self._fmt(dsum.get('cured_glue_mass_g'), 1, ' g')}
- Água evaporada estimada: {self._fmt(dsum.get('evaporated_glue_water_g'), 1, ' g')}
- Massa competitiva final: {self._fmt(dsum.get('competition_mass_g', dsum.get('estimated_total_mass_g')), 1, ' g')}
- Margem até o limite efetivo: {self._fmt(dsum.get('mass_margin_g'), 1, ' g')}
- Palitos brutos comprados (estimados): {dsum.get('purchased_blank_sticks_needed', dsum.get('estimated_total_sticks_with_waste', '—'))}
- Descarte de corte estimado: {self._fmt(dsum.get('cutting_scrap_mass_g'), 1, ' g')}
- Massa de compra/produção: {self._fmt(dsum.get('assembly_procurement_mass_g'), 1, ' g')}
- Observação: descarte de corte e sobra de compra não entram na massa final competitiva.

## 6) Dimensionamento por membro e emendas
- Membros com ajuste local de sticks: {sizing_changed}
- Emendas leves/moderadas/reforçadas: {conn_light}/{conn_mod}/{conn_strong}
- Clusters críticos de emenda alinhada: {aligned_critical}
- Seção real do palito usada em A, Iy, Iz e flambagem: sim.

## 7) Verificações contra critérios eliminatórios do edital
| Critério | Valor obtido | Regra | Conforme |
| --- | --- | --- | --- |
{criter_lines}

## 8) Tabelas resumidas
### 8.1 Membros críticos
| Membro | Grupo | FS mínimo | Modo governante |
| --- | --- | --- | --- |
{weak_table}

### 8.2 Recomendações objetivas
{suggestions_md}

## 9) Diagrama de esforços e gráficos de suporte
- Diagrama de esforços axiais: `outputs/plots/06_esforcos_axiais_todos.png` e `07_esforcos_axiais_principais.png`.
- Deformada: `outputs/plots/08_forma_deformada.png`.
- Ranking de falha: `outputs/plots/09_ranking_falha_principal.png`.
- Reações de apoio: `outputs/plots/10_reacoes_apoio.png`.
- Planos de treliça: `outputs/plots/02_trelica_lateral_esquerda.png`, `03_trelica_lateral_direita.png`, `04_plano_superior.png`, `05_plano_inferior.png`.

## 10) Comentários finais
{summary}

Limitação do modelo:
- A previsão acima é idealizada (modelo numérico). Ensaios físicos reais ainda são obrigatórios para validação experimental.

Arquivos detalhados de apoio:
- `outputs/details/stick_pieces.csv`
- `outputs/details/glue_joints.csv`
- `outputs/details/cutting_list.csv`
- `outputs/opensees/member_failure_checks.csv`
- `outputs/opensees/support_reaction_checks.csv`
"""
        p.write_text(md, encoding="utf-8")
        return p
