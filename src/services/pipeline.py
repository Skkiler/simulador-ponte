from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from src.core.numeric import numeric_values, safe_float
from src.services.config_service import ConfigService
from src.services.geometry_service import GeometryService
from src.services.postprocessor import PostProcessor
from src.services.mass_guard import effective_mass_limit_g, resolve_mass_limits
from src.services.recommendation_service import RecommendationService
from src.services.report_service import ReportService
from src.services.visualization_service import VisualizationService
from src.services.stick_detail_service import StickDetailService
from src.services.detail_visualization_service import DetailVisualizationService
from src.services.active_design_planner import ActiveDesignPlanner
from src.services.planner_debug_logger import PlannerDebugLogger
from src.services.quarter_model_service import QuarterModelService
from src.services.connection_planner import ConnectionPlanner
from src.solvers.frame3dd_adapter import Frame3DDAdapter
from src.solvers.linear_truss_solver import LinearTrussSolver


class SimulationPipeline:
    """Orquestra as etapas. As dependências são explícitas e substituíveis."""

    def __init__(self, output_root: str | Path = "outputs") -> None:
        self.output_root = Path(output_root)
        self.config_service = ConfigService()
        self.geometry = GeometryService()
        self.solver = LinearTrussSolver()
        self.post = PostProcessor()
        self.viz = VisualizationService()
        self.detail = StickDetailService()
        self.detail_viz = DetailVisualizationService()
        self.recs = RecommendationService()
        self.reporter = ReportService()
        self.frame3dd = Frame3DDAdapter()
        self.optimizer = ActiveDesignPlanner()
        self.connection = ConnectionPlanner()

    @staticmethod
    def _evaluate_edital_criteria(cfg: Dict, metrics: Dict, detailed: Dict) -> List[Dict]:
        bridge = cfg.get("bridge", {})
        mat = cfg.get("material", {})
        dsum = (detailed or {}).get("summary", {}) or {}

        def row(name: str, measured: Any, rule: str, ok: bool) -> Dict:
            return {
                "criterio": name,
                "valor_obtido": measured,
                "regra": rule,
                "conforme": bool(ok),
            }

        span = float(bridge.get("span_mm", 0.0))
        left_support = abs(float(bridge.get("left_support_overhang_mm", 0.0)))
        right_support = abs(float(bridge.get("right_support_overhang_mm", 0.0)))
        width = float(bridge.get("width_mm", 0.0))
        height = float(bridge.get("center_height_mm", 0.0))
        mass_g = safe_float(dsum.get("estimated_total_mass_g"), None)
        if mass_g is None:
            mass_g = safe_float(metrics.get("estimated_total_mass_g"), 0.0) or 0.0

        stick_len = float(mat.get("stick_length_mm", 0.0))
        stick_thk = float(mat.get("stick_thickness_mm", 0.0))
        stick_w = float(mat.get("stick_width_mm", 0.0))
        c1 = float(mat.get("compression_capacity_one_stick_kgf", 0.0))
        c2 = float(mat.get("compression_capacity_two_sticks_kgf", 0.0))
        tr = float(mat.get("tension_capacity_per_stick_kgf", 0.0))

        mass_limits = resolve_mass_limits(cfg)
        eff_limit = float(mass_limits["effective_limit_g"])
        nominal_limit = float(mass_limits["nominal_limit_g"])
        if abs(eff_limit - nominal_limit) <= 1e-6:
            mass_rule = f"máximo {nominal_limit:.0f} g"
        else:
            mass_rule = f"máximo {eff_limit:.0f} g (config.) / {nominal_limit:.0f} g (edital)"

        # Build a list of edital checks.  The stick dimension reference values use
        # the updated specification (7.0 mm × 1.5 mm), replacing the legacy 8.2 mm × 2.0 mm
        checks = [
            row("Vão livre", f"{span:.1f} mm", "obrigatório 1200 mm", abs(span - 1200.0) <= 1e-6),
            row("Apoio esquerdo", f"{left_support:.1f} mm", "máximo 100 mm", left_support <= 100.0 + 1e-6),
            row("Apoio direito", f"{right_support:.1f} mm", "máximo 100 mm", right_support <= 100.0 + 1e-6),
            row("Largura", f"{width:.1f} mm", "entre 100 e 200 mm", 100.0 - 1e-6 <= width <= 200.0 + 1e-6),
            row("Altura central", f"{height:.1f} mm", "mínimo 50 mm", height >= 50.0 - 1e-6),
            row("Peso total estimado", f"{mass_g:.1f} g", mass_rule, mass_g <= eff_limit + 1e-6),
            row("Palito - comprimento", f"{stick_len:.1f} mm", "115 mm (referência)", abs(stick_len - 115.0) <= 1e-6),
            row(
                "Palito - espessura",
                f"{stick_thk:.2f} mm",
                "1,5 mm (referência)",
                abs(stick_thk - 1.5) <= 1e-6,
            ),
            row(
                "Palito - largura",
                f"{stick_w:.2f} mm",
                "7,0 mm (referência)",
                abs(stick_w - 7.0) <= 1e-6,
            ),
            row("Compressão 1 palito", f"{c1:.2f} kgf", "mínimo 4,0 kgf", c1 >= 4.0 - 1e-6),
            row("Compressão 2 palitos", f"{c2:.2f} kgf", "mínimo 11,0 kgf", c2 >= 11.0 - 1e-6),
            row("Tração por palito", f"{tr:.2f} kgf", "mínimo 72,0 kgf", tr >= 72.0 - 1e-6),
        ]
        return checks

    def run(
        self,
        cfg: Dict,
        *,
        progress_callback: Callable[[float, str], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> Dict:
        input_cfg = self.config_service.normalize(cfg)
        debug_logger = PlannerDebugLogger(
            self.output_root / "logs",
            enabled=bool(input_cfg.get("analysis", {}).get("planner_debug_enabled", True)),
        )
        debug_logger.event(
            "config_loaded",
            stage="pipeline",
            metrics={
                "enforce_symmetry": bool(input_cfg.get("analysis", {}).get("enforce_symmetry", True)),
                "use_quarter_model_requested": bool(input_cfg.get("analysis", {}).get("use_quarter_model", False)),
            },
        )
        debug_logger.event("user_inputs_normalized", stage="pipeline")

        execution_logs: List[str] = []
        warnings: List[Dict[str, str]] = []

        def emit_log(msg: str) -> None:
            text = str(msg)
            execution_logs.append(text)
            if callable(log_callback):
                try:
                    log_callback(text)
                except (TypeError, ValueError, RuntimeError) as exc:
                    warnings.append(
                        {
                            "code": "WARN_LOG_CALLBACK_FAILED",
                            "stage": "pipeline",
                            "message": f"Falha ao enviar log para callback: {exc!r}",
                        }
                    )

        def emit_progress(value: float, text: str) -> None:
            if callable(progress_callback):
                try:
                    progress_callback(max(0.0, min(1.0, float(value))), str(text))
                except (TypeError, ValueError, RuntimeError) as exc:
                    warnings.append(
                        {
                            "code": "WARN_PROGRESS_CALLBACK_FAILED",
                            "stage": "pipeline",
                            "message": f"Falha ao enviar progresso para callback: {exc!r}",
                        }
                    )

        def emit_warning(code: str, message: str, stage: str = "pipeline") -> None:
            warnings.append({"code": code, "stage": stage, "message": message})
            emit_log(f"[WARN:{code}] {message}")

        self.output_root.mkdir(parents=True, exist_ok=True)

        model_dir = self.output_root / "model"
        solver_dir = self.output_root / "opensees"
        plot_dir = self.output_root / "plots"
        detail_dir = self.output_root / "details"
        frame_dir = self.output_root / "frame3dd"
        report_dir = self.output_root / "reports"

        optimization = None
        cfg = input_cfg

        emit_progress(0.0, "Inicializando pipeline")
        emit_log("Pipeline iniciado.")
        for msg in input_cfg.get("compatibility_warnings", []) or []:
            emit_warning("WARN_CONFIG_COMPAT", str(msg), stage="config")

        planner_enabled = bool(
            input_cfg.get("analysis", {}).get("active_planner_enabled", True)
        )
        optimize_variants = bool(
            input_cfg.get("analysis", {}).get("optimize_variants", True)
        )

        if optimize_variants and planner_enabled:
            try:
                optimization = self.optimizer.run(
                    input_cfg,
                    self.output_root / "optimization",
                    progress_callback=lambda p, t: emit_progress(0.05 + 0.55 * p, f"Planejador: {t}"),
                    log_callback=lambda m: emit_log(f"[Planejador] {m}"),
                    debug_logger=debug_logger,
                )
                best = (optimization or {}).get("best")
                best_cfg = (best or {}).get("config")
                sc = (optimization or {}).get("stage_counts") or {}

                if sc:
                    emit_log(
                        "Resumo da busca: "
                        f"S0-geradas={sc.get('stage0_generated', 0)} | "
                        f"S0-aprovadas={sc.get('stage0_prefilter_passed', 0)} | "
                        f"S0-descartadas={sc.get('stage0_prefilter_discarded', 0)} | "
                        f"S1-avaliadas={sc.get('stage1_evaluated', 0)} | "
                        f"S1-descartadas_pós_solver={sc.get('stage1_discarded_post_solver', 0)} | "
                        f"S1-válidas={sc.get('stage1', 0)} | "
                        f"S2A={sc.get('stage2a_selected', 0)} | "
                        f"S2B={sc.get('stage2b_evaluated', 0)} | "
                        f"S2-únicas={sc.get('stage2_unique', sc.get('stage2', 0))} | "
                        f"S3={sc.get('stage3', 0)} | S4={sc.get('stage4', 0)}"
                    )
                    by_reason = sc.get("discarded_by_reason", {}) or {}
                    if by_reason:
                        top_reason, top_count = sorted(by_reason.items(), key=lambda kv: kv[1], reverse=True)[0]
                        emit_log(f"Principal motivo de descarte: {top_reason} ({top_count}).")

                if best_cfg:
                    cfg = self.config_service.normalize(best_cfg)
                    emit_log("Configuração recomendada pelo planejador aplicada ao pipeline final.")
                    target_break = float(
                        cfg.get("planner", {}).get(
                            "target_breaking_load_kgf",
                            cfg.get("planner", {}).get("target_load_kgf", cfg.get("bridge", {}).get("load_total_kgf", 120.0)),
                        )
                    )
                    pred_break = safe_float((best or {}).get("predicted_breaking_load_kgf"), 0.0) or 0.0
                    att = 100.0 * pred_break / max(1.0, target_break)
                    emit_log(
                        f"Atingimento da meta de ruptura (planejador): "
                        f"{pred_break:.1f}/{target_break:.1f} kgf ({att:.1f}%)."
                    )
                else:
                    emit_log(
                        "Planejador não retornou proposta aceitável para o limite de massa. "
                        "Pipeline seguirá com a configuração solicitada para diagnóstico."
                    )
            except (RuntimeError, ValueError, TypeError, KeyError) as exc:
                optimization = {
                    "error": repr(exc),
                    "stage1": [],
                    "stage2": [],
                    "stage3": [],
                    "best": None,
                }
                emit_log(f"Falha no planejador: {repr(exc)}")
                emit_warning(
                    "WARN_PLANNER_EXECUTION_FAILED",
                    f"Planejador falhou e o pipeline seguirá com a configuração solicitada: {exc!r}",
                    stage="planner",
                )

        emit_progress(0.62, "Gerando geometria estrutural")
        nodes, members, supports, loads = self.geometry.generate(cfg)
        self.geometry.export_csvs(cfg, model_dir)
        qsvc = QuarterModelService()
        quarter_summary: Dict[str, Any] = {
            "enabled_requested": qsvc.is_quarter_model_enabled(cfg),
            "enabled_used": False,
            "validation": {},
            "fallback_reason": None,
        }
        full_replicated_model_summary: Dict[str, Any] = {}

        # Se o modo de quarto-modelo estiver ativado e validado, resolve o quarto
        # e replica o resultado para o modelo completo.
        if qsvc.is_quarter_model_enabled(cfg):
            debug_logger.event("quarter_model_enabled", stage="pipeline")
            val = qsvc.validate_quarter_symmetry(cfg, nodes, members, supports, loads)
            quarter_summary["validation"] = val
            if bool(val.get("is_valid")):
                emit_progress(0.68, "Executando solver estrutural (1/4)")
                q_model = qsvc.build_quarter_model(cfg, nodes, members, supports, loads)
                debug_logger.event(
                    "quarter_model_built",
                    stage="pipeline",
                    metrics={
                        "quarter_nodes": len(q_model.nodes),
                        "quarter_members": len(q_model.members),
                        "quarter_supports": len(q_model.supports),
                        "quarter_loads": len(q_model.loads),
                    },
                )
                quarter_nodes_csv = [n.__dict__.copy() for n in q_model.nodes]
                quarter_members_csv = [m.__dict__.copy() for m in q_model.members]
                quarter_supports_csv = [s.__dict__.copy() for s in q_model.supports]
                quarter_loads_csv = [l.__dict__.copy() for l in q_model.loads]
                GeometryService.write_csv(model_dir / "quarter_model_nodes.csv", quarter_nodes_csv)
                GeometryService.write_csv(model_dir / "quarter_model_members.csv", quarter_members_csv)
                GeometryService.write_csv(model_dir / "quarter_model_supports.csv", quarter_supports_csv)
                GeometryService.write_csv(model_dir / "quarter_model_loads.csv", quarter_loads_csv)

                full = qsvc.solve_quarter_and_replicate(cfg, self.solver, q_model)
                debug_logger.event(
                    "quarter_model_solved",
                    stage="pipeline",
                    metrics={
                        "status": full.result.status,
                        "iterations": full.result.iterations,
                        "eq_error_N": full.result.equilibrium_error_N,
                    },
                )
                nodes, members, supports, loads, result = (
                    full.nodes,
                    full.members,
                    full.supports,
                    full.loads,
                    full.result,
                )
                cfg.setdefault("analysis", {})["quarter_member_count"] = int(full.quarter_member_count)
                cfg["analysis"]["use_quarter_model"] = True
                cfg["analysis"]["quarter_model_mode"] = str(cfg.get("analysis", {}).get("quarter_model_mode", "strict"))
                quarter_summary["enabled_used"] = True
                full_replicated_model_summary = {
                    "full_nodes": len(nodes),
                    "full_members": len(members),
                    "full_supports": len(supports),
                    "full_loads": len(loads),
                    "quarter_member_count": int(full.quarter_member_count),
                }
                (model_dir / "symmetry_maps.json").write_text(
                    json.dumps(full.mirror_maps, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                (model_dir / "quarter_model_results.json").write_text(
                    json.dumps(
                        {
                            "status": full.result.status,
                            "iterations": full.result.iterations,
                            "equilibrium_error_N": full.result.equilibrium_error_N,
                            "node_results_count": len(full.result.node_results),
                            "member_results_count": len(full.result.member_results),
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                debug_logger.event(
                    "quarter_model_replicated",
                    stage="pipeline",
                    metrics=full_replicated_model_summary,
                )
            else:
                quarter_summary["fallback_reason"] = "symmetry_validation_failed"
                cfg.setdefault("analysis", {})["use_quarter_model"] = False
                debug_logger.event(
                    "quarter_model_validation_failed",
                    stage="pipeline",
                    level="warning",
                    reason=";".join(val.get("reasons", []) or ["symmetry validation failed"]),
                )
                emit_log("Quarter-model desativado: validação de simetria falhou. Executando modelo completo.")
                emit_progress(0.68, "Executando solver estrutural")
                result = self.solver.solve(
                    nodes,
                    members,
                    supports,
                    loads,
                    unilateral_supports=bool(cfg["bridge"].get("unilateral_supports", True)),
                )
        else:
            emit_progress(0.68, "Executando solver estrutural")
            result = self.solver.solve(
                nodes,
                members,
                supports,
                loads,
                unilateral_supports=bool(cfg["bridge"].get("unilateral_supports", True)),
            )

        (model_dir / "quarter_model_summary.json").write_text(
            json.dumps(quarter_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (model_dir / "full_replicated_model_summary.json").write_text(
            json.dumps(full_replicated_model_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Atualiza apoios para Frame3DD e plots com os apoios ativos reais.
        active_supports = []

        for s in supports:
            active_supports.append(
                type(s)(
                    s.node_id,
                    s.UX,
                    s.UY,
                    s.UZ if s.node_id in result.active_support_node_ids else 0,
                    s.RX,
                    s.RY,
                    s.RZ,
                    s.support_group,
                    s.node_id in result.active_support_node_ids,
                )
            )

        self.solver.export(result, solver_dir)
        emit_log(f"Solver concluído com status: {result.status}")
        emit_progress(0.74, "Pós-processando verificações")

        member_checks = self.post.check_members(cfg, result.member_results)
        support_checks = self.post.check_supports(
            cfg,
            nodes,
            active_supports,
            result.node_results,
        )

        self.post.export(member_checks, support_checks, solver_dir)

        frame_input = self.frame3dd.write_input(
            cfg,
            nodes,
            members,
            active_supports,
            loads,
            frame_dir / "ponte_palitos.3dd",
        )

        frame_result = {"status": "skipped"}

        if cfg["analysis"].get("run_frame3dd_if_available", True):
            emit_progress(0.78, "Rodando validação Frame3DD")
            frame_result = self.frame3dd.run(
                cfg,
                frame_input,
                frame_dir / "ponte_palitos.out",
            )

            (frame_dir / "frame3dd_status.json").write_text(
                json.dumps(frame_result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            emit_log(f"Frame3DD: {frame_result.get('status')}")

        emit_progress(0.84, "Detalhando peças e juntas")
        connection_plan = self.connection.assign_member_joint_plan(
            cfg,
            nodes,
            members,
            result.member_results,
            member_checks,
        )
        member_sizing_plan = self.optimizer.build_member_sizing_plan(
            cfg,
            nodes,
            members,
            result.member_results,
            member_checks,
        )

        cfg_detail = copy.deepcopy(cfg)
        cfg_detail["member_joint_plan"] = connection_plan
        cfg_detail["member_sizing_plan_by_id"] = {
            str(k): v.__dict__.copy()
            for k, v in member_sizing_plan.items()
        }
        try:
            connection_rows = sorted(
                (dict(v) for v in connection_plan.values()),
                key=lambda r: int(r.get("member_id", -1)),
            )
            GeometryService.write_csv(detail_dir / "connection_plan.csv", connection_rows)
            (detail_dir / "connection_plan.json").write_text(
                json.dumps(connection_rows, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            sizing_rows = [v.__dict__.copy() for v in member_sizing_plan.values()]
            GeometryService.write_csv(detail_dir / "member_sizing_plan.csv", sizing_rows)
            (detail_dir / "member_sizing_plan.json").write_text(
                json.dumps(sizing_rows, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except (OSError, ValueError, TypeError) as exc:
            emit_warning(
                "WARN_DETAIL_PLAN_EXPORT_FAILED",
                f"Falha ao exportar planos de conexão/sizing: {exc!r}",
                stage="detail",
            )

        detailed = self.detail.analyze(
            cfg_detail,
            nodes,
            members,
            result.member_results,
            member_checks,
            detail_dir,
        )
        detailed["connection_plan"] = list(connection_plan.values())
        detailed["member_sizing_plan"] = [v.__dict__.copy() for v in member_sizing_plan.values()]
        debug_logger.event(
            "splice_stagger_applied",
            stage="detail",
            metrics={
                "enabled": bool(cfg.get("detail_model", {}).get("splice_stagger_enabled", True)),
                "joints": len(detailed.get("glue_joints", []) or []),
            },
        )
        aligned_critical = int(
            (detailed.get("splice_stagger_report", {}) or {}).get(
                "critical_clusters",
                (detailed.get("splice_stagger_report", {}) or {}).get("critical_aligned_count", 0),
            )
            or 0
        )
        if aligned_critical > 0:
            debug_logger.event(
                "aligned_splice_detected",
                stage="detail",
                level="warning",
                metrics={"critical_clusters": aligned_critical},
            )

        self.viz.save_all(
            nodes,
            members,
            active_supports,
            loads,
            result.node_results,
            result.member_results,
            member_checks,
            support_checks,
            plot_dir,
            deformed_scale=float(cfg["analysis"].get("deformed_scale", 30.0)),
        )

        # After generating the detailed piece breakdown, collapse pieces into
        # assembly groups.  This reduces thousands of entries into a handful of
        # buckets, each representing a typical subassembly.  These
        # aggregates are saved into the ``detailed`` dictionary and to JSON/CSV
        # files for later use.  Failures in grouping should not interrupt
        # the pipeline.
        try:
            from src.services.assembly_grouping_service import AssemblyGroupingService  # type: ignore

            assembler = AssemblyGroupingService()
            pieces = (detailed or {}).get("stick_pieces", []) or []
            if pieces:
                groups = assembler.group_stick_pieces(pieces)
                summary = assembler.summarize(pieces)
                detailed["assembly_groups"] = groups
                detailed["assembly_summary"] = summary
                # Persist to files for external inspection
                try:
                    (detail_dir / "assembly_groups.json").write_text(
                        json.dumps(groups, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    (detail_dir / "assembly_summary.json").write_text(
                        json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    # Write CSV using GeometryService
                    GeometryService.write_csv(detail_dir / "assembly_groups.csv", groups)
                except (OSError, ValueError, TypeError) as exc:
                    emit_warning(
                        "WARN_ASSEMBLY_EXPORT_FAILED",
                        f"Falha ao exportar agrupamentos de montagem: {exc!r}",
                        stage="detail",
                    )
        except (ImportError, AttributeError, ValueError, TypeError) as exc:
            emit_warning(
                "WARN_ASSEMBLY_GROUPING_FAILED",
                f"Falha no agrupamento de montagem: {exc!r}",
                stage="detail",
            )

        if cfg.get("detail_model", {}).get("generate_piece_views", True):
            self.detail_viz.save_all(detailed, plot_dir)
        emit_progress(0.90, "Gerando recomendações e relatório")

        primary = [
            r
            for r in member_checks
            if r.get("member_role") == "primary"
        ]

        fs_primary = numeric_values(r.get("FS_min") for r in primary)
        fs_all = numeric_values(r.get("FS_min") for r in member_checks)

        detailed_summary = detailed.get("summary", {})

        metrics = {
            "n_nodes": len(nodes),
            "n_members": len(members),
            "n_active_supports": len(result.active_support_node_ids),
            "n_uplift_supports": len(result.inactive_support_node_ids),
            "equilibrium_error_N": result.equilibrium_error_N,
            "min_fs_primary": min(fs_primary) if fs_primary else None,
            "min_fs_all": min(fs_all) if fs_all else None,
            "solver_status": result.status,
            "frame3dd_status": frame_result.get("status"),
            "estimated_sticks_total": detailed_summary.get("estimated_total_sticks_with_waste"),
            "estimated_total_mass_g": detailed_summary.get("estimated_total_mass_g"),
            "mass_margin_g": detailed_summary.get("mass_margin_g"),
            "estimated_glue_mass_g": detailed_summary.get("estimated_glue_mass_g"),
            "mass_limit_nominal_g": detailed_summary.get("mass_limit_nominal_g"),
            "mass_limit_material_g": detailed_summary.get("mass_limit_material_g"),
            "mass_limit_planner_g": detailed_summary.get("mass_limit_planner_g"),
            "mass_limit_effective_g": detailed_summary.get("mass_limit_effective_g"),
            "mass_limit_effective_source": detailed_summary.get("mass_limit_effective_source"),
            "quarter_model_requested": quarter_summary.get("enabled_requested"),
            "quarter_model_used": quarter_summary.get("enabled_used"),
            "quarter_model_fallback_reason": quarter_summary.get("fallback_reason"),
        }

        if optimization and optimization.get("best"):
            best = optimization["best"]
            metrics["planner_score"] = safe_float(best.get("score"), None)
            metrics["predicted_breaking_load_kgf"] = safe_float(
                best.get("predicted_breaking_load_kgf"),
                None,
            )
            metrics["planner_feasible"] = bool(best.get("feasible"))
            metrics["planner_objective_profile"] = cfg.get("analysis", {}).get(
                "planner_objective_profile",
                "balanced",
            )
            counts = optimization.get("stage_counts") or {}
            metrics["planner_stage1_count"] = counts.get("stage1")
            metrics["planner_stage2_count"] = counts.get("stage2")
            metrics["planner_stage3_count"] = counts.get("stage3")
            metrics["planner_stage4_count"] = counts.get("stage4")
            metrics["planner_final_variants_count"] = counts.get("final_variants")

            fv = optimization.get("final_variants") or {}
            fmin = fv.get("min") or {}
            fmax = fv.get("max") or {}
            metrics["final_min_fs_primary"] = safe_float(fmin.get("min_fs_primary"), None)
            metrics["final_max_fs_primary"] = safe_float(fmax.get("min_fs_primary"), None)
            metrics["final_min_mass_g"] = safe_float(fmin.get("mass_g"), None)
            metrics["final_max_mass_g"] = safe_float(fmax.get("mass_g"), None)

        recommendations = self.recs.build(
            cfg,
            member_checks,
            support_checks,
            {
                "status": result.status,
                "equilibrium_error_N": result.equilibrium_error_N,
            },
            detailed,
            optimization=optimization,
        )

        report_path = self.reporter.write_markdown(
            cfg,
            metrics,
            recommendations,
            report_dir / "relatorio_automatico.md",
            detailed=detailed,
        )

        edital_checks = self._evaluate_edital_criteria(cfg, metrics, detailed)
        (report_dir / "criterios_edital.json").write_text(
            json.dumps(edital_checks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        emit_log(
            "Critérios eliminatórios do edital: "
            f"{sum(1 for c in edital_checks if c.get('conforme'))}/{len(edital_checks)} conformes."
        )

        self.config_service.save(cfg, self.output_root / "config_used.json")
        self.config_service.save(input_cfg, self.output_root / "config_requested.json")
        debug_logger.event(
            "mass_limit_check",
            stage="pipeline",
            metrics={
                "mass_g": safe_float(detailed_summary.get("estimated_total_mass_g"), None),
                "mass_limit_g": effective_mass_limit_g(cfg),
            },
        )
        if optimization and optimization.get("best"):
            if bool((optimization.get("best") or {}).get("feasible")):
                debug_logger.event("final_candidate_selected", stage="pipeline")
            else:
                debug_logger.event("final_candidate_rejected", stage="pipeline", reason="best_not_feasible")
        else:
            debug_logger.event("no_feasible_candidate", stage="pipeline", level="warning")

        zip_path = self.output_root / "resultados_simulacao.zip"
        self.zip_outputs(zip_path)
        emit_progress(1.0, "Pipeline concluído")
        emit_log("Pipeline finalizado.")
        debug_logger.write_summary()

        return {
            "cfg": cfg,
            "input_cfg": input_cfg,
            "nodes": nodes,
            "members": members,
            "supports": active_supports,
            "loads": loads,
            "solver_result": result,
            "member_checks": member_checks,
            "support_checks": support_checks,
            "detailed": detailed,
            "metrics": metrics,
            "recommendations": recommendations,
            "report_path": report_path,
            "edital_checks": edital_checks,
            "frame3dd_result": frame_result,
            "zip_path": zip_path,
            "optimization": optimization,
            "execution_logs": execution_logs,
            "warnings": warnings,
            "planner_debug_jsonl": str(debug_logger.jsonl_path),
            "planner_debug_summary": str(debug_logger.summary_path),
        }

    def zip_outputs(self, zip_path: str | Path) -> Path:
        zp = Path(zip_path)

        if zp.exists():
            zp.unlink()

        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for p in self.output_root.rglob("*"):
                if p.is_file() and p != zp:
                    z.write(p, p.relative_to(self.output_root))

        return zp
