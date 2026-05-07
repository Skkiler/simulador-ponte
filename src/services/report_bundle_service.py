from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.core.numeric import safe_float
from src.services.geometry_service import GeometryService


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
        min_primary = safe_float(metrics.get("min_fs_primary"), 0.0) or 0.0
        min_primary_target = float(analysis.get("acceptance_min_primary_fs", 1.05))
        min_support = safe_float(metrics.get("min_support_fs"), None)
        min_support_target = float(analysis.get("acceptance_min_support_fs", 1.0))
        min_glue = safe_float(metrics.get("min_glue_fs"), None)
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

        if not failures:
            return "APROVADA", "Atende aos critérios de ruptura, massa e regularidade do solver.", failures

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
        pred_break = safe_float(metrics.get("predicted_breaking_load_kgf"), 0.0) or 0.0
        competition_mass = safe_float(metrics.get("competition_mass_g"), safe_float(metrics.get("estimated_total_mass_g"), 0.0)) or 0.0
        mass_limit = safe_float(metrics.get("mass_limit_effective_g"), 1000.0) or 1000.0
        mass_margin = mass_limit - competition_mass
        solver_regular = self._solver_regular(metrics.get("solver_status"))

        exec_summary = {
            "verdict": verdict,
            "reason": verdict_reason,
            "predicted_breaking_load_kgf": pred_break,
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
- Seção composta: ação parcial com `eta_I`.
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

## 10. Hipóteses do modelo
- Ver `assumptions_and_warnings.md`.

## 11. Links para gráficos
- `outputs/plots/`
- `outputs/optimization/plot_geometry_refinement.png`

## 12. Reprovação honesta
{failures_md}

Top 5 mudanças necessárias:
{changes_md}
"""
        (out / "index.md").write_text(index_md, encoding="utf-8")
        (out / "index.html").write_text(
            "<html><body><pre>" + index_md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + "</pre></body></html>",
            encoding="utf-8",
        )

        return {
            "index_md": str(out / "index.md"),
            "index_html": str(out / "index.html"),
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
        }
