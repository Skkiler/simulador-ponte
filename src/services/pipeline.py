from __future__ import annotations

import copy
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from src.core.numeric import numeric_values, safe_float
from src.services.config_service import ConfigService
from src.services.geometry_service import GeometryService
from src.services.postprocessor import PostProcessor
from src.services.mass_guard import assert_mass_compliant, effective_mass_limit_g, resolve_mass_limits
from src.services.rupture_estimator import estimate_rupture_load
from src.services.recommendation_service import RecommendationService
from src.services.report_bundle_service import ReportBundleService
from src.services.report_service import ReportService
from src.services.visualization_service import VisualizationService
from src.services.stick_detail_service import StickDetailService
from src.services.detail_visualization_service import DetailVisualizationService
from src.services.active_design_planner import ActiveDesignPlanner
from src.services.planner_debug_logger import PlannerDebugLogger
from src.services.quarter_model_service import QuarterModelService
from src.services.connection_planner import ConnectionPlanner
from src.services.assembly_tutorial_service import AssemblyTutorialService
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
        self.report_bundle = ReportBundleService()
        self.frame3dd = Frame3DDAdapter()
        self.optimizer = ActiveDesignPlanner()
        self.connection = ConnectionPlanner()
        self.assembly_tutorial = AssemblyTutorialService()

    @staticmethod
    def _solver_is_regular(status: Any) -> bool:
        return str(status or "").split("|", 1)[0] == "regular"

    @staticmethod
    def _solver_kwargs_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        bridge = cfg.get("bridge", {}) or {}
        analysis = cfg.get("analysis", {}) or {}

        tension_only_groups = [
            str(g)
            for g in (analysis.get("tension_only_groups") or [])
            if str(g).strip()
        ]

        tension_only_enabled = (
            bool(bridge.get("tension_only_bracing_solver_enabled", False))
            and bool(tension_only_groups)
            and bool(analysis.get("enable_tension_only_solver_globally", False))
        )

        return {
            "unilateral_supports": bool(bridge.get("unilateral_supports", True)),
            "tension_only_solver_enabled": tension_only_enabled,
            "tension_only_groups": tension_only_groups,
            "tension_only_compression_tolerance_N": float(
                analysis.get("tension_only_compression_tolerance_N", 1.0e-6)
            ),
        }

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
        mass_g = safe_float(dsum.get("competition_mass_g"), None)
        if mass_g is None:
            mass_g = safe_float(dsum.get("estimated_total_mass_g"), None)
        if mass_g is None:
            mass_g = safe_float(
                metrics.get("competition_mass_g"),
                safe_float(metrics.get("estimated_total_mass_g"), 0.0),
            ) or 0.0

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

        rules = cfg.get("competition_rules", {}) or {}
        enforce_stick_dims = bool(rules.get("enforce_nominal_stick_dimensions", False))
        if enforce_stick_dims:
            req_len = rules.get("required_stick_length_mm")
            req_thk = rules.get("required_stick_thickness_mm")
            req_w = rules.get("required_stick_width_mm")
            tol_len = float(rules.get("stick_length_tolerance_mm", 0.5))
            tol_thk = float(rules.get("stick_thickness_tolerance_mm", 0.2))
            tol_w = float(rules.get("stick_width_tolerance_mm", 0.2))
            len_ok = req_len is None or abs(stick_len - float(req_len)) <= tol_len
            thk_ok = req_thk is None or abs(stick_thk - float(req_thk)) <= tol_thk
            width_ok = req_w is None or abs(stick_w - float(req_w)) <= tol_w
            len_rule = "configurável" if req_len is None else f"{float(req_len):.1f} ± {tol_len:.1f} mm"
            thk_rule = "configurável" if req_thk is None else f"{float(req_thk):.2f} ± {tol_thk:.2f} mm"
            width_rule = "configurável" if req_w is None else f"{float(req_w):.2f} ± {tol_w:.2f} mm"
        else:
            len_ok = stick_len > 0.0
            thk_ok = stick_thk > 0.0
            width_ok = stick_w > 0.0
            len_rule = "configurável; deve ser > 0 mm"
            thk_rule = "configurável; deve ser > 0 mm"
            width_rule = "configurável; deve ser > 0 mm"

        checks = [
            row("Vão livre", f"{span:.1f} mm", "obrigatório 1200 mm", abs(span - 1200.0) <= 1e-6),
            row("Apoio esquerdo", f"{left_support:.1f} mm", "máximo 100 mm", left_support <= 100.0 + 1e-6),
            row("Apoio direito", f"{right_support:.1f} mm", "máximo 100 mm", right_support <= 100.0 + 1e-6),
            row("Largura", f"{width:.1f} mm", "entre 100 e 200 mm", 100.0 - 1e-6 <= width <= 200.0 + 1e-6),
            row("Altura central", f"{height:.1f} mm", "mínimo 50 mm", height >= 50.0 - 1e-6),
            row("Peso total estimado", f"{mass_g:.1f} g", mass_rule, mass_g <= eff_limit + 1e-6),
            row("Palito - comprimento", f"{stick_len:.1f} mm", len_rule, len_ok),
            row(
                "Palito - espessura",
                f"{stick_thk:.2f} mm",
                thk_rule,
                thk_ok,
            ),
            row(
                "Palito - largura",
                f"{stick_w:.2f} mm",
                width_rule,
                width_ok,
            ),
            row("Compressão 1 palito", f"{c1:.2f} kgf", "mínimo 4,0 kgf", c1 >= 4.0 - 1e-6),
            row("Compressão 2 palitos", f"{c2:.2f} kgf", "mínimo 11,0 kgf", c2 >= 11.0 - 1e-6),
            row("Tração por palito", f"{tr:.2f} kgf", "mínimo 72,0 kgf", tr >= 72.0 - 1e-6),
        ]
        return checks

    @staticmethod
    def _select_mass_compliant_candidate(
        optimization: Dict[str, Any] | None,
        cfg: Dict[str, Any],
        *,
        return_all: bool = False,
    ) -> Dict[str, Any] | List[Dict[str, Any]] | None:
        if not optimization:
            return None

        eff_limit = float(effective_mass_limit_g(cfg))
        load_total_N = abs(float(cfg.get("bridge", {}).get("load_total_N", 0.0)))
        eq_tol_N = max(1.0e-6, 0.005 * max(load_total_N, 1.0))
        analysis = cfg.get("analysis", {}) or {}
        acceptance_fs = max(0.1, float(analysis.get("acceptance_min_primary_fs", 1.05)))
        min_fs_ratio = max(0.0, float(analysis.get("planner_fallback_min_fs_ratio", 0.10)))
        min_fs_floor = max(0.0, acceptance_fs * min_fs_ratio)
        target_load_kgf = max(
            0.1,
            float(
                cfg.get("analysis", {}).get(
                    "acceptance_min_design_breaking_load_kgf",
                    cfg.get("planner", {}).get(
                        "target_load_kgf",
                        cfg.get("bridge", {}).get("load_total_kgf", 120.0),
                    ),
                )
            ),
        )
        min_break_ratio = max(0.0, float(analysis.get("planner_fallback_min_break_ratio", 0.20)))
        min_break_floor_kgf = target_load_kgf * min_break_ratio
        # Para o modo return_all usado pela validação detalhada, aceite também
        # candidatos um pouco acima da massa: eles podem ser trazidos para baixo
        # por micro-trim de member_sticks_by_id.  Sem isso, o S5/S7 forte, mas
        # 3–6% acima do limite, era descartado antes de qualquer tentativa de
        # aliviar massa; o pipeline caía para S3 fraco e leve.
        overmass_trim_ratio = max(0.0, float(analysis.get("planner_fallback_trim_overmass_ratio", 0.08)))
        mass_screen_limit = eff_limit * (1.0 + overmass_trim_ratio if return_all else 1.0)

        def row_mass_g(row: Dict[str, Any]) -> float | None:
            for key in ("mass_g", "competition_mass_g", "dead_weight_proxy_g", "mass_proxy_g", "mass_proxy"):
                val = safe_float(row.get(key), None)
                if val is not None:
                    return float(val)
            return None

        def row_break_kgf(row: Dict[str, Any]) -> float | None:
            for key in ("predicted_breaking_load_kgf", "predicted_breaking_load_proxy_kgf", "estimated_breaking_load_kgf"):
                val = safe_float(row.get(key), None)
                if val is not None:
                    return float(val)
            return None

        def row_fs(row: Dict[str, Any]) -> float | None:
            for key in ("min_fs_primary", "min_fs_design", "min_fs_design_proxy", "min_fs_preliminary"):
                val = safe_float(row.get(key), None)
                if val is not None:
                    return float(val)
            return None

        def row_solver_status(row: Dict[str, Any]) -> str:
            raw = row.get("solver_status")
            if raw is None and bool(row.get("solver_regular", False)):
                return "regular"
            return str(raw or "regular")

        stage_priority = {
            "s8_final_validation": 0,
            "s7_fabrication": 1,
            "s6_topology": 2,
            "s5_member_sizing": 3,
            "s4_geometry_refinement": 4,
            "s3_multi_loadcase": 5,
            "s2_fast_screening": 6,
            "stage4": 7,
            "stage3": 8,
            "stage2": 9,
            "stage1": 10,
        }
        ranked_rows: List[tuple[int, Dict[str, Any]]] = []
        for stage_name, prio in stage_priority.items():
            for row in (optimization.get(stage_name) or []):
                ranked_rows.append((prio, row))

        valid_rows: List[tuple[int, Dict[str, Any]]] = []
        for prio, row in ranked_rows:
            mass_val = row_mass_g(row)
            if mass_val is None or mass_val > mass_screen_limit + 1.0e-6:
                continue
            if not SimulationPipeline._solver_is_regular(row_solver_status(row)):
                continue
            eq_err = abs(safe_float(row.get("equilibrium_error_N"), 0.0) or 0.0)
            if eq_err > eq_tol_N:
                continue
            fs_val = row_fs(row)
            if fs_val is None or fs_val < min_fs_floor:
                continue
            break_val = row_break_kgf(row)
            if break_val is None or break_val < min_break_floor_kgf:
                continue
            normalized_row = dict(row)
            normalized_row.setdefault("mass_g", mass_val)
            normalized_row.setdefault("predicted_breaking_load_kgf", break_val)
            normalized_row.setdefault("min_fs_primary", fs_val)
            normalized_row.setdefault("solver_status", row_solver_status(row))
            valid_rows.append((prio, normalized_row))

        if not valid_rows:
            return [] if return_all else None

        # Primeiro a fidelidade do estágio, depois desempenho.  Sem isto uma
        # linha antiga/grosseira de S3/S6 pode vencer uma validação S7/S8 por
        # ter ruptura proxy maior, levando ao relatório final uma configuração
        # pesada, fraca e diferente da que foi detalhada.
        valid_rows.sort(
            key=lambda item: (
                item[0],
                -(safe_float(item[1].get("predicted_breaking_load_kgf"), 0.0) or 0.0),
                -(safe_float(item[1].get("min_fs_primary"), 0.0) or 0.0),
                -(safe_float(item[1].get("score"), -1.0e99) or -1.0e99),
                (safe_float(item[1].get("mass_g"), 1.0e99) or 1.0e99),
            )
        )
        if return_all:
            return [row for _, row in valid_rows]
        return valid_rows[0][1]

    def _build_mass_trim_variants(self, cfg: Dict[str, Any]) -> List[tuple[str, Dict[str, Any]]]:
        """Generate conservative trim variants for a near-overweight fallback.

        Strong candidates often use ``member_sticks_by_id`` overrides, not just
        group counts.  The previous trimmer reduced only ``member_sticks_by_group``;
        therefore it could not actually lighten an S5/S7 candidate and the
        pipeline fell back to a much weaker S3.  This method first creates
        ID-level trims in non-primary groups, then falls back to group-level trims.
        """
        out: List[tuple[str, Dict[str, Any]]] = []
        priority = [
            "support_pad",
            "top_bracing",
            "bottom_bracing",
            "cross_frame_bracing",
            "top_transverse",
            "bottom_transverse",
            "bottom_chord",
            "top_chord",
        ]
        min_by_group = {
            "support_pad": 1,
            "top_bracing": 1,
            "bottom_bracing": 1,
            "cross_frame_bracing": 1,
            "top_transverse": 1,
            "bottom_transverse": 1,
            "bottom_chord": 1,
            "top_chord": 3,
            "vertical": 3,
            "diagonal": 2,
        }

        by_id = cfg.get("member_sticks_by_id", {}) or {}
        if isinstance(by_id, dict) and by_id:
            try:
                _, members, _, _ = self.geometry.generate(cfg)
                member_group = {str(int(m.id)): str(m.group) for m in members}
            except Exception:
                member_group = {}

            for group in priority:
                trim_ids: List[str] = []
                for mid, raw_n in by_id.items():
                    g = member_group.get(str(mid))
                    if g != group:
                        continue
                    cur_i = max(1, int(safe_float(raw_n, 1) or 1))
                    if cur_i > min_by_group.get(group, 1):
                        trim_ids.append(str(mid))
                if not trim_ids:
                    continue
                # Variações progressivas: alivia 25%, 50% e 100% dos IDs do grupo.
                for frac in (0.25, 0.50, 1.00):
                    n_take = max(1, int(round(len(trim_ids) * frac)))
                    vcfg = copy.deepcopy(cfg)
                    v_by_id = vcfg.setdefault("member_sticks_by_id", {})
                    for mid in trim_ids[:n_take]:
                        old_n = max(1, int(safe_float(v_by_id.get(mid), 1) or 1))
                        v_by_id[mid] = old_n - 1
                    out.append((f"id_trim_{group}_{n_take}members", vcfg))

        sticks = (cfg.get("member_sticks_by_group", {}) or {})
        if isinstance(sticks, dict):
            for group in priority:
                cur = safe_float(sticks.get(group), None)
                if cur is None:
                    continue
                cur_i = max(1, int(cur))
                if cur_i <= min_by_group.get(group, 1):
                    continue
                vcfg = copy.deepcopy(cfg)
                vcfg.setdefault("member_sticks_by_group", {})[group] = cur_i - 1
                out.append((f"group_trim_{group}:{cur_i}->{cur_i - 1}", vcfg))
        return out

    def run(
        self,
        cfg: Dict,
        *,
        progress_callback: Callable[[float, str], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> Dict:
        input_cfg = self.config_service.normalize(cfg)
        self.output_root.mkdir(parents=True, exist_ok=True)
        run_started_at = datetime.now(timezone.utc).isoformat()
        run_id = run_started_at.replace(":", "").replace("+", "Z")
        analysis0 = input_cfg.get("analysis", {}) or {}
        cleaned_dirs: List[str] = []
        if bool(analysis0.get("clean_output_dirs_before_run", True)):
            for rel in analysis0.get("output_dirs_to_clean_before_run", []) or []:
                # Defensive: clean only direct children of output_root.
                rel_str = str(rel).strip().replace("\\", "/").strip("/")
                if not rel_str or "/" in rel_str or rel_str in {".", ".."}:
                    continue
                path = self.output_root / rel_str
                if path.exists():
                    shutil.rmtree(path, ignore_errors=True)
                cleaned_dirs.append(rel_str)
        input_cfg.setdefault("runtime", {})["run_id"] = run_id
        input_cfg.setdefault("runtime", {})["started_at_utc"] = run_started_at
        input_cfg.setdefault("runtime", {})["cleaned_output_dirs"] = cleaned_dirs
        (self.output_root / "run_metadata.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "started_at_utc": run_started_at,
                    "cleaned_output_dirs": cleaned_dirs,
                    "clean_output_dirs_before_run": bool(analysis0.get("clean_output_dirs_before_run", True)),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        debug_logger = PlannerDebugLogger(
            self.output_root / "logs",
            enabled=bool(input_cfg.get("analysis", {}).get("planner_debug_enabled", True)),
        )
        debug_logger.event(
            "config_loaded",
            stage="pipeline",
            metrics={
                "run_id": run_id,
                "cleaned_output_dirs": cleaned_dirs,
                "enforce_symmetry": bool(input_cfg.get("analysis", {}).get("enforce_symmetry", True)),
                "use_quarter_model_requested": bool(input_cfg.get("analysis", {}).get("use_quarter_model", False)),
            },
        )
        debug_logger.event("user_inputs_normalized", stage="pipeline", metrics={"run_id": run_id})

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

        model_dir = self.output_root / "model"
        solver_dir = self.output_root / "opensees"
        plot_dir = self.output_root / "plots"
        detail_dir = self.output_root / "details"
        frame_dir = self.output_root / "frame3dd"
        report_dir = self.output_root / "reports"

        plot_dir.mkdir(parents=True, exist_ok=True)

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
        strict_mass_acceptance = bool(
            input_cfg.get("analysis", {}).get("strict_mass_acceptance", True)
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
                best_from_mass_fallback = False
                sc = (optimization or {}).get("stage_counts") or {}
                fallback_validation_cap = max(
                    1,
                    int(
                        input_cfg.get("analysis", {}).get(
                            "planner_fallback_validation_cap",
                            24,
                        )
                    ),
                )

                if sc:
                    if "S1_macro_candidates" in sc:
                        emit_log(
                            "Resumo do funil: "
                            f"S1={sc.get('S1_macro_candidates', 0)} | "
                            f"S2={sc.get('S2_fast_screening_candidates', 0)}→{sc.get('S2_fast_screening_top_k', 0)} | "
                            f"S3={sc.get('S3_multi_loadcase_candidates', 0)}→{sc.get('S3_multi_loadcase_top_k', 0)} | "
                            f"S4={sc.get('S4_geometry_refinement_candidates', 0)}→{sc.get('S4_geometry_refinement_top_k', 0)} | "
                            f"S5={sc.get('S5_member_sizing_candidates', 0)} | "
                            f"S6={sc.get('S6_topology_candidates', 0)} | "
                            f"S7={sc.get('S7_fabrication_candidates', 0)} | "
                            f"S8={sc.get('S8_final_validation_candidates', 0)} | "
                            f"solves={sc.get('solves_total', 0)}"
                        )
                    else:
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

                best_is_feasible = bool((best or {}).get("feasible", False))
                if strict_mass_acceptance and (not best_cfg or not best_is_feasible):
                    fallback_rows = self._select_mass_compliant_candidate(
                        optimization,
                        input_cfg,
                        return_all=True,
                    ) or []
                    emit_log(
                        f"Fallback por massa/estágio: {len(fallback_rows)} candidato(s) elegíveis; "
                        f"validando até {fallback_validation_cap}."
                    )
                    debug_logger.event(
                        "fallback_candidate_pool",
                        stage="planner",
                        metrics={
                            "eligible_count": len(fallback_rows),
                            "validation_cap": fallback_validation_cap,
                            "strict_mass_acceptance": strict_mass_acceptance,
                        },
                    )
                    for fallback_idx, fallback_row in enumerate(list(fallback_rows)[:fallback_validation_cap], 1):
                        if fallback_row.get("config") is None:
                            continue
                        fallback_cfg = self.config_service.normalize(fallback_row.get("config"))
                        emit_log(
                            "Validando fallback "
                            f"#{fallback_idx}: estágio={fallback_row.get('stage')} | "
                            f"ruptura={safe_float(fallback_row.get('predicted_breaking_load_kgf'), 0.0) or 0.0:.1f} kgf | "
                            f"massa={safe_float(fallback_row.get('mass_g'), 0.0) or 0.0:.1f} g | "
                            f"FS={safe_float(fallback_row.get('min_fs_primary'), 0.0) or 0.0:.3f}."
                        )
                        try:
                            val_dir = self.output_root / "optimization" / "fallback_validation"
                            fallback_metrics = self.optimizer._evaluate_config(
                                fallback_cfg,
                                include_detail=True,
                                detail_dir=val_dir,
                            )
                        except (TypeError, ValueError, KeyError, RuntimeError) as exc:
                            emit_log(f"Validação de fallback por massa falhou: {exc!r}")
                            continue

                        mass_ok = bool(fallback_metrics.get("mass_compliant", False))
                        solver_ok = self._solver_is_regular(fallback_metrics.get("solver_status", ""))
                        eq_ok = bool(fallback_metrics.get("equilibrium_ok", False))
                        debug_logger.event(
                            "fallback_candidate_validation",
                            stage="planner",
                            metrics={
                                "index": fallback_idx,
                                "mass_g": fallback_metrics.get("mass_g"),
                                "mass_limit_g": fallback_metrics.get("mass_limit_effective_g"),
                                "break_kgf": fallback_metrics.get("estimated_breaking_load_kgf"),
                                "min_fs_primary": fallback_metrics.get("min_fs_primary"),
                                "solver_status": fallback_metrics.get("solver_status"),
                                "equilibrium_error_N": fallback_metrics.get("equilibrium_error_N"),
                                "mass_ok": mass_ok,
                                "solver_ok": solver_ok,
                                "equilibrium_ok": eq_ok,
                            },
                        )
                        if not (mass_ok and solver_ok and eq_ok):
                            # Se o candidato está próximo do limite, tenta microajustes de massa
                            # antes de descartá-lo (ex.: aliviar support_pad em 1 palito).
                            mass_val = safe_float(fallback_metrics.get("mass_g"), None)
                            limit_val = safe_float(fallback_metrics.get("mass_limit_effective_g"), None)
                            trim_applied = False
                            if (
                                mass_val is not None
                                and limit_val is not None
                                and mass_val > limit_val
                                and mass_val <= (limit_val * 1.05)
                            ):
                                trim_variants = self._build_mass_trim_variants(fallback_cfg)
                                for trim_label, trim_cfg_raw in trim_variants[:16]:
                                    trim_cfg = self.config_service.normalize(trim_cfg_raw)
                                    try:
                                        trim_dir = self.output_root / "optimization" / "fallback_trim_validation"
                                        trim_metrics = self.optimizer._evaluate_config(
                                            trim_cfg,
                                            include_detail=True,
                                            detail_dir=trim_dir,
                                        )
                                    except (TypeError, ValueError, KeyError, RuntimeError):
                                        continue

                                    trim_mass_ok = bool(trim_metrics.get("mass_compliant", False))
                                    trim_solver_ok = self._solver_is_regular(trim_metrics.get("solver_status", ""))
                                    trim_eq_ok = bool(trim_metrics.get("equilibrium_ok", False))
                                    if not (trim_mass_ok and trim_solver_ok and trim_eq_ok):
                                        continue

                                    best = dict(fallback_row)
                                    best.update(
                                        {
                                            "mass_g": trim_metrics.get("mass_g"),
                                            "mass_limit_g": trim_metrics.get("mass_limit_effective_g"),
                                            "predicted_breaking_load_kgf": trim_metrics.get("estimated_breaking_load_kgf"),
                                            "min_fs_primary": trim_metrics.get("min_fs_primary"),
                                            "solver_status": trim_metrics.get("solver_status"),
                                            "equilibrium_error_N": trim_metrics.get("equilibrium_error_N"),
                                            "score": trim_metrics.get("score"),
                                            "feasible": trim_metrics.get("feasible"),
                                        }
                                    )
                                    best_cfg = trim_cfg
                                    best_from_mass_fallback = True
                                    optimization["best"] = best
                                    optimization["best_is_feasible"] = bool(best.get("feasible"))
                                    optimization["best_mass_compliant_fallback"] = True
                                    emit_log(
                                        "Fallback validado após microajuste de massa "
                                        f"({trim_label})."
                                    )
                                    trim_applied = True
                                    break

                            if trim_applied:
                                break
                            emit_log(
                                "Fallback por massa rejeitado após validação detalhada: "
                                f"solver={fallback_metrics.get('solver_status')} | "
                                f"massa={safe_float(fallback_metrics.get('mass_g'), 0.0) or 0.0:.1f} g."
                            )
                            continue

                        best = dict(fallback_row)
                        best.update(
                            {
                                "mass_g": fallback_metrics.get("mass_g"),
                                "mass_limit_g": fallback_metrics.get("mass_limit_effective_g"),
                                "predicted_breaking_load_kgf": fallback_metrics.get("estimated_breaking_load_kgf"),
                                "min_fs_primary": fallback_metrics.get("min_fs_primary"),
                                "solver_status": fallback_metrics.get("solver_status"),
                                "equilibrium_error_N": fallback_metrics.get("equilibrium_error_N"),
                                "score": fallback_metrics.get("score"),
                                "feasible": fallback_metrics.get("feasible"),
                            }
                        )
                        best_cfg = fallback_cfg
                        best_from_mass_fallback = True
                        optimization["best"] = best
                        optimization["best_is_feasible"] = bool(best.get("feasible"))
                        optimization["best_mass_compliant_fallback"] = True
                        emit_log(
                            "Planejador sem proposta totalmente viável; "
                            "aplicando fallback validado com massa conforme ao limite."
                        )
                        break
                    if (not best_cfg) and fallback_rows:
                        emit_warning(
                            "WARN_PLANNER_FALLBACK_NOT_VALIDATED",
                            "Candidatos de fallback por massa rápida existiam, mas nenhum passou "
                            "na validação detalhada. Fallback não validado foi descartado.",
                            stage="planner",
                        )

                if best_cfg and (bool((best or {}).get("feasible", False)) or best_from_mass_fallback):
                    cfg = self.config_service.normalize(best_cfg)
                    if best_from_mass_fallback and not bool((best or {}).get("feasible", False)):
                        emit_warning(
                            "WARN_PLANNER_NO_FULL_FEASIBLE",
                            "Nenhuma proposta atendeu todos os critérios. "
                            "Aplicado fallback estrito de massa para manter limite de peso.",
                            stage="planner",
                        )
                    else:
                        emit_log("Configuração viável recomendada pelo planejador aplicada ao pipeline final.")
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
                elif best_cfg:
                    best_mass = safe_float((best or {}).get("mass_g"), None)
                    best_solver_status = str((best or {}).get("solver_status", "regular"))
                    best_regular = self._solver_is_regular(best_solver_status)

                    diagnostic_mass_ok = True
                    if strict_mass_acceptance:
                        try:
                            diagnostic_mass = best_mass
                            if diagnostic_mass is None:
                                preview_nodes, preview_members, _, _ = self.geometry.generate(best_cfg)
                                diagnostic_mass, _ = self.optimizer._quick_mass_estimate(best_cfg, preview_members)
                            diagnostic_mass_ok = float(diagnostic_mass) <= float(effective_mass_limit_g(best_cfg)) + 1.0e-6
                        except (TypeError, ValueError, RuntimeError, KeyError):
                            diagnostic_mass_ok = False

                    if best_regular and diagnostic_mass_ok:
                        cfg = self.config_service.normalize(best_cfg)
                        emit_warning(
                            "WARN_PLANNER_BEST_NONFEASIBLE_APPLIED_FOR_DIAGNOSTIC",
                            "Planejador retornou proposta não viável, mas regular e dentro do limite de massa. "
                            "Ela será aplicada ao pipeline final como diagnóstico, "
                            "com veredito reprovado se não atingir ruptura/FS.",
                            stage="planner",
                        )
                    elif best_regular and strict_mass_acceptance:
                        emit_warning(
                            "WARN_PLANNER_BEST_NONFEASIBLE_NOT_APPLIED_OVER_MASS",
                            "Planejador retornou proposta regular, porém não viável e/ou acima do limite de massa. "
                            "Como a aceitação estrita de massa está ativa, ela não será aplicada ao pipeline final.",
                            stage="planner",
                        )
                    else:
                        emit_log(
                            "Planejador retornou proposta não viável e não regular; "
                            "pipeline manterá a configuração solicitada para segurança dos cálculos."
                        )
                else:
                    emit_log(
                        "Planejador não retornou proposta aceitável para o limite de massa. "
                        "Pipeline seguirá com a configuração solicitada para diagnóstico."
                    )
            except (RuntimeError, ValueError, TypeError, KeyError) as exc:
                mode = str((cfg.get("planner_pipeline", {}) or {}).get("mode", "")).strip().lower()

                if mode == "staged_fidelity_funnel":
                    optimization = {
                        "error": repr(exc),
                        "stage1": [],
                        "stage2": [],
                        "stage3": [],
                        "stage4": [],
                        "best": {
                            "feasible": False,
                            "verdict": "NENHUMA SOLUÇÃO VIÁVEL",
                            "failed_restriction": repr(exc),
                            "config": cfg,
                        },
                        "best_is_feasible": False,
                        "logs": [f"Funil falhou: {repr(exc)}"],
                    }

                    emit_log(f"Falha no funil de fidelidade crescente: {repr(exc)}")
                    emit_warning(
                        "WARN_PLANNER_FUNNEL_NO_VIABLE_SOLUTION",
                        f"O funil não encontrou solução viável: {exc!r}",
                        stage="planner",
                    )
                else:
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

        # Se o pipeline escolheu um fallback validado após o funil, a pasta
        # optimization ainda pode conter o final_validation_summary gerado pelo
        # candidato S8 original.  Reescrever o resumo evita mismatch entre
        # final_report e optimization/final_validation_summary.json.
        try:
            best_opt = (optimization or {}).get("best") or {}
            if best_opt and (optimization or {}).get("best_mass_compliant_fallback"):
                opt_dir = self.output_root / "optimization"
                opt_dir.mkdir(parents=True, exist_ok=True)
                fv = {
                    "verdict": "FALLBACK_MASS_COMPLIANT_REPROVADO",
                    "failed_restriction": "fallback_mass_compliant_but_strength_or_fs_below_target",
                    "predicted_breaking_load_kgf": best_opt.get("predicted_breaking_load_kgf"),
                    "target_breaking_load_kgf": cfg.get("analysis", {}).get("acceptance_min_design_breaking_load_kgf", 120.0),
                    "competition_mass_g": best_opt.get("mass_g"),
                    "min_fs_design": best_opt.get("min_fs_primary"),
                    "validation_stage": "pre_detail",
                    "solver_regular": self._solver_is_regular(str(best_opt.get("solver_status", "regular"))),
                    "equilibrium_ok": abs(safe_float(best_opt.get("equilibrium_error_N"), 0.0) or 0.0) <= 1.0e-6,
                    "source": "pipeline_best_mass_compliant_fallback",
                }
                (opt_dir / "final_validation_summary.json").write_text(
                    json.dumps(fv, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except (OSError, TypeError, ValueError, KeyError) as exc:
            emit_warning(
                "WARN_FINAL_VALIDATION_REWRITE_FAILED",
                f"Falha ao sincronizar final_validation_summary com fallback aplicado: {exc!r}",
                stage="planner",
            )

        # Garantia de integridade: mesmo quando o funil é consumido em memória
        # pela UI, o pacote final precisa conter a validação S8 atual.  Runs
        # anteriores podiam terminar com `optimization/final_validation_summary`
        # ausente, fazendo o relatório de integridade acusar WARN e deixando
        # `ruptura design multi-loadcase` como n/d.
        try:
            opt_dir = self.output_root / "optimization"
            opt_dir.mkdir(parents=True, exist_ok=True)
            fv_path = opt_dir / "final_validation_summary.json"
            s8_rows = list((optimization or {}).get("s8_final_validation", []) or [])
            if s8_rows and not fv_path.exists():
                s8 = dict(s8_rows[0])
                fv = {
                    "verdict": s8.get("verdict"),
                    "failed_restriction": s8.get("failed_restriction"),
                    "predicted_breaking_load_kgf": s8.get("predicted_breaking_load_kgf"),
                    "predicted_breaking_load_proxy_kgf": s8.get("predicted_breaking_load_proxy_kgf"),
                    "predicted_breaking_load_by_members_kgf": s8.get("predicted_breaking_load_by_members_kgf"),
                    "predicted_breaking_load_by_supports_kgf": s8.get("predicted_breaking_load_by_supports_kgf"),
                    "predicted_breaking_load_by_glue_kgf": s8.get("predicted_breaking_load_by_glue_kgf"),
                    "target_breaking_load_kgf": s8.get("target_breaking_load_kgf"),
                    "competition_mass_g": s8.get("competition_mass_g"),
                    "min_fs_design": s8.get("min_fs_design"),
                    "min_fs_member_design": s8.get("min_fs_member_design", s8.get("min_fs_design")),
                    "min_fs_support": s8.get("min_fs_support"),
                    "min_fs_glue": s8.get("min_fs_glue"),
                    "solver_regular": s8.get("solver_regular"),
                    "equilibrium_ok": s8.get("equilibrium_ok"),
                    "validation_stage": s8.get("validation_stage", "pre_detail"),
                    "governing_limit_state": s8.get("governing_limit_state"),
                    "governing_fs": s8.get("governing_fs"),
                    "governing_case": s8.get("governing_case"),
                    "governing_strength_case": s8.get("governing_strength_case", s8.get("governing_case")),
                    "governing_service_case": s8.get("governing_service_case"),
                    "governing_contact_case": s8.get("governing_contact_case"),
                    "governing_member_id": s8.get("governing_member_id"),
                    "governing_member_group": s8.get("governing_member_group"),
                    "case_metrics": s8.get("case_metrics") or [],
                    "source": "pipeline_s8_memory_sync",
                }
                fv_path.write_text(json.dumps(fv, indent=2, ensure_ascii=False), encoding="utf-8")
        except (OSError, TypeError, ValueError, KeyError) as exc:
            emit_warning(
                "WARN_FINAL_VALIDATION_SYNC_FAILED",
                f"Falha ao sincronizar final_validation_summary a partir do S8: {exc!r}",
                stage="planner",
            )

        emit_progress(0.62, "Gerando geometria estrutural")
        nodes, members, supports, loads = self.geometry.generate(cfg)
        self.geometry.export_csvs(cfg, model_dir)
        full_nodes_base = list(nodes)
        full_members_base = list(members)
        full_supports_base = list(supports)
        full_loads_base = list(loads)

        def _solve_full_model() -> Any:
            return self.solver.solve(
                full_nodes_base,
                full_members_base,
                full_supports_base,
                full_loads_base,
                **self._solver_kwargs_from_cfg(cfg),
            )

        def _solver_is_acceptable(result_obj: Any, load_rows: List[Any]) -> tuple[bool, float]:
            total_load_N = abs(sum(float(getattr(ld, "Fz", 0.0)) for ld in (load_rows or [])))
            eq_tol_N = max(1.0e-6, 0.005 * max(total_load_N, 1.0))
            status_ok = self._solver_is_regular(getattr(result_obj, "status", ""))
            eq_ok = abs(float(getattr(result_obj, "equilibrium_error_N", 0.0) or 0.0)) <= eq_tol_N
            return (status_ok and eq_ok), eq_tol_N

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
                try:
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
                    quarter_summary["enabled_used"] = True
                    full_replicated_model_summary = {
                        "full_nodes": len(full.nodes),
                        "full_members": len(full.members),
                        "full_supports": len(full.supports),
                        "full_loads": len(full.loads),
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

                    quarter_ok, quarter_tol = _solver_is_acceptable(
                        full.result,
                        full.loads,
                    )
                    if not quarter_ok:
                        quarter_summary["fallback_reason"] = (
                            f"quarter_solution_invalid:status={full.result.status},"
                            f"eq_error={full.result.equilibrium_error_N:.6f},eq_tol={quarter_tol:.6f}"
                        )
                        emit_log(
                            "Quarter-model retornou solução não regular; "
                            "reexecutando com modelo completo para garantir consistência."
                        )
                        nodes, members, supports, loads = (
                            full_nodes_base,
                            full_members_base,
                            full_supports_base,
                            full_loads_base,
                        )
                        result = _solve_full_model()
                        cfg.setdefault("analysis", {})["use_quarter_model"] = False
                    else:
                        finalize_with_full = bool(
                            cfg.get("analysis", {}).get("quarter_model_finalize_with_full", True)
                        )
                        if finalize_with_full:
                            emit_log(
                                "Quarter-model válido. Reexecutando modelo completo para "
                                "resultados finais e visualização sem segmentação."
                            )
                            nodes, members, supports, loads = (
                                full_nodes_base,
                                full_members_base,
                                full_supports_base,
                                full_loads_base,
                            )
                            result = _solve_full_model()
                            quarter_summary["finalized_with_full_model"] = True
                        else:
                            nodes, members, supports, loads, result = (
                                full.nodes,
                                full.members,
                                full.supports,
                                full.loads,
                                full.result,
                            )
                            cfg.setdefault("analysis", {})["quarter_member_count"] = int(full.quarter_member_count)
                            cfg["analysis"]["use_quarter_model"] = True
                            cfg["analysis"]["quarter_model_mode"] = str(
                                cfg.get("analysis", {}).get("quarter_model_mode", "strict")
                            )
                except (TypeError, ValueError, KeyError, RuntimeError) as exc:
                    quarter_summary["fallback_reason"] = f"quarter_model_execution_failed:{exc!r}"
                    emit_log(
                        "Falha no quarter-model; executando modelo completo para manter "
                        f"segurança dos cálculos ({exc!r})."
                    )
                    nodes, members, supports, loads = (
                        full_nodes_base,
                        full_members_base,
                        full_supports_base,
                        full_loads_base,
                    )
                    result = _solve_full_model()
                    cfg.setdefault("analysis", {})["use_quarter_model"] = False
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
                nodes, members, supports, loads = (
                    full_nodes_base,
                    full_members_base,
                    full_supports_base,
                    full_loads_base,
                )
                result = _solve_full_model()
        else:
            emit_progress(0.68, "Executando solver estrutural")
            nodes, members, supports, loads = (
                full_nodes_base,
                full_members_base,
                full_supports_base,
                full_loads_base,
            )
            result = _solve_full_model()

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

        emit_log(f"Solver concluído com status: {result.status}")
        emit_progress(0.74, "Pós-processando verificações")

        member_checks = self.post.check_members(cfg, result.member_results)
        support_checks = self.post.check_supports(
            cfg,
            nodes,
            active_supports,
            result.node_results,
        )

        has_funnel_member_sizing = (
            str((cfg.get("planner_pipeline", {}) or {}).get("mode", "")).strip().lower()
            == "staged_fidelity_funnel"
            and bool(cfg.get("member_sticks_by_id"))
        )

        if has_funnel_member_sizing:
            emit_log(
                "Sizing local final ignorado: a proposta do funil já contém "
                "dimensionamento por membro. Evitando sobrescrever por um plano "
                "all-or-nothing."
            )
            member_sizing_plan = {}
            cfg_with_sizing = copy.deepcopy(cfg)
            sizing_changed = False
        else:
            member_sizing_plan = self.optimizer.build_member_sizing_plan(
                cfg,
                nodes,
                members,
                result.member_results,
                member_checks,
            )

            # Para o pipeline final aplicamos apenas redimensionamento por membro.
            # Não desabilitamos barras automaticamente aqui para evitar singularidade.
            cfg_with_sizing = copy.deepcopy(cfg)
            sized_by_id: Dict[str, int] = dict(cfg.get("member_sticks_by_id", {}) or {})
            sized_active: Dict[str, bool] = dict(cfg.get("member_active_by_id", {}) or {})

            for mid, decision in member_sizing_plan.items():
                sized_by_id[str(int(mid))] = max(1, int(decision.n_sticks_recommended))
                sized_active[str(int(mid))] = True

            cfg_with_sizing["member_sticks_by_id"] = sized_by_id
            cfg_with_sizing["member_active_by_id"] = sized_active
            # Preserve a topologia existente; nunca limpe disabled_member_ids aqui.
            cfg_with_sizing["disabled_member_ids"] = list(cfg.get("disabled_member_ids", []) or [])
            cfg_with_sizing = self.config_service.normalize(cfg_with_sizing)
            sizing_changed = any(
                [
                    (cfg_with_sizing.get("member_sticks_by_id", {}) or {}) != (cfg.get("member_sticks_by_id", {}) or {}),
                    (cfg_with_sizing.get("member_active_by_id", {}) or {}) != (cfg.get("member_active_by_id", {}) or {}),
                    (cfg_with_sizing.get("disabled_member_ids", []) or []) != (cfg.get("disabled_member_ids", []) or []),
                ]
            )

        if sizing_changed:
            emit_log(
                "Aplicando dimensionamento local por membro e reexecutando o solver "
                "para consolidar N, distribuição e consumo de palitos."
            )
            load_kgf_for_sizing = float(cfg.get("bridge", {}).get("load_total_kgf", 0.0))
            pre_primary_fs_vals = numeric_values(
                r.get("FS_min")
                for r in member_checks
                if r.get("member_role") == "primary"
            )
            pre_min_fs_primary = min(pre_primary_fs_vals) if pre_primary_fs_vals else None
            pre_break = safe_float(
                estimate_rupture_load(
                    cfg,
                    member_checks,
                    support_checks,
                    None,
                    load_kgf_for_sizing,
                ).get("predicted_breaking_load_kgf"),
                None,
            )
            cfg_prev = cfg
            nodes_prev, members_prev, supports_prev, loads_prev = nodes, members, supports, loads
            result_prev = result
            active_supports_prev = list(active_supports)
            member_checks_prev = list(member_checks)
            support_checks_prev = list(support_checks)
            preview_nodes, preview_members, preview_supports, preview_loads = self.geometry.generate(cfg_with_sizing)
            can_apply_sizing = True
            if strict_mass_acceptance:
                try:
                    preview_mass_g, _ = self.optimizer._quick_mass_estimate(cfg_with_sizing, preview_members)
                    preview_limit_g = float(effective_mass_limit_g(cfg_with_sizing))
                    current_mass_g, _ = self.optimizer._quick_mass_estimate(cfg, members)
                    if preview_mass_g > preview_limit_g + 1.0e-6:
                        # Se o sizing ainda excede o limite mas reduz massa significativamente
                        # em relação à configuração atual, mantemos a tentativa.
                        if preview_mass_g >= (current_mass_g - 1.0e-6):
                            can_apply_sizing = False
                            emit_warning(
                                "WARN_MEMBER_SIZING_SKIPPED_OVER_MASS",
                                "Sizing local por membro foi descartado porque a prévia de massa "
                                f"({preview_mass_g:.1f} g) excede o limite efetivo ({preview_limit_g:.1f} g) "
                                f"e não reduz a massa atual ({current_mass_g:.1f} g).",
                                stage="detail",
                            )
                except (TypeError, ValueError, RuntimeError, KeyError):
                    pass

            if can_apply_sizing:
                cfg = cfg_with_sizing
                nodes, members, supports, loads = (
                    preview_nodes,
                    preview_members,
                    preview_supports,
                    preview_loads,
                )
                self.geometry.export_csvs(cfg, model_dir)
                result = self.solver.solve(
                    nodes,
                    members,
                    supports,
                    loads,
                    **self._solver_kwargs_from_cfg(cfg),
                )
                active_supports = [
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
                    for s in supports
                ]
                member_checks = self.post.check_members(cfg, result.member_results)
                support_checks = self.post.check_supports(
                    cfg,
                    nodes,
                    active_supports,
                    result.node_results,
                )
                member_sizing_plan = self.optimizer.build_member_sizing_plan(
                    cfg,
                    nodes,
                    members,
                    result.member_results,
                    member_checks,
                )
                if not self._solver_is_regular(result.status):
                    emit_warning(
                        "WARN_MEMBER_SIZING_REEVAL_SINGULAR",
                        (
                            "Reavaliação após sizing por membro gerou solver não regular; "
                            "mantendo solução anterior para preservar consistência estrutural."
                        ),
                        stage="solver",
                    )
                    cfg = cfg_prev
                    nodes, members, supports, loads = (
                        nodes_prev,
                        members_prev,
                        supports_prev,
                        loads_prev,
                    )
                    self.geometry.export_csvs(cfg, model_dir)
                    result = result_prev
                    active_supports = active_supports_prev
                    member_checks = member_checks_prev
                    support_checks = support_checks_prev
                    member_sizing_plan = self.optimizer.build_member_sizing_plan(
                        cfg,
                        nodes,
                        members,
                        result.member_results,
                        member_checks,
                    )
                else:
                    post_primary_fs_vals = numeric_values(
                        r.get("FS_min")
                        for r in member_checks
                        if r.get("member_role") == "primary"
                    )
                    post_min_fs_primary = min(post_primary_fs_vals) if post_primary_fs_vals else None
                    post_break = safe_float(
                        estimate_rupture_load(
                            cfg,
                            member_checks,
                            support_checks,
                            None,
                            load_kgf_for_sizing,
                        ).get("predicted_breaking_load_kgf"),
                        None,
                    )

                    fs_drop_reject = False
                    if pre_min_fs_primary is not None and post_min_fs_primary is not None:
                        min_ratio = float(
                            cfg.get("analysis", {}).get(
                                "post_sizing_min_fs_ratio_min",
                                0.92,
                            )
                        )
                        fs_drop_reject = post_min_fs_primary < (pre_min_fs_primary * min_ratio)

                    break_drop_reject = False
                    if pre_break is not None and post_break is not None:
                        min_ratio_break = float(
                            cfg.get("analysis", {}).get(
                                "post_sizing_break_ratio_min",
                                0.95,
                            )
                        )
                        break_drop_reject = post_break < (pre_break * min_ratio_break)

                    if fs_drop_reject or break_drop_reject:
                        emit_warning(
                            "WARN_MEMBER_SIZING_REJECTED_WEAKENING",
                            (
                                "Reavaliação pós-sizing reduziu desempenho estrutural "
                                f"(FS: {pre_min_fs_primary} -> {post_min_fs_primary}, "
                                f"ruptura: {pre_break} -> {post_break}); "
                                "mantendo solução anterior."
                            ),
                            stage="solver",
                        )
                        cfg = cfg_prev
                        nodes, members, supports, loads = (
                            nodes_prev,
                            members_prev,
                            supports_prev,
                            loads_prev,
                        )
                        self.geometry.export_csvs(cfg, model_dir)
                        result = result_prev
                        active_supports = active_supports_prev
                        member_checks = member_checks_prev
                        support_checks = support_checks_prev
                        member_sizing_plan = self.optimizer.build_member_sizing_plan(
                            cfg,
                            nodes,
                            members,
                            result.member_results,
                            member_checks,
                        )
                    else:
                        emit_log(f"Solver pós-sizing concluído com status: {result.status}")

        self.solver.export(result, solver_dir)
        self.post.export(member_checks, support_checks, solver_dir)

        frame_input = self.frame3dd.write_input(
            cfg,
            nodes,
            members,
            active_supports,
            loads,
            frame_dir / "ponte_palitos.3dd",
        )

        frame_result = {"status": "not_run", "classification": "not_run"}

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
            member_sizing_plan={
                int(k): v.__dict__.copy()
                for k, v in member_sizing_plan.items()
            },
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

        try:
            tutorial = self.assembly_tutorial.build(
                cfg_detail,
                nodes,
                members,
                detailed,
                detail_dir,
            )
            detailed["assembly_tutorial"] = tutorial
        except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
            emit_warning(
                "WARN_ASSEMBLY_TUTORIAL_FAILED",
                f"Falha ao gerar tutorial de montagem: {exc!r}",
                stage="detail",
            )

        if cfg.get("detail_model", {}).get("generate_piece_views", True):
            emit_progress(0.875, "Gerando vistas 2D peça-a-peça")
            self.detail_viz.save_all(detailed, plot_dir)
            try:
                pieces = (detailed or {}).get("stick_pieces", []) or []
                detail_cfg = cfg.get("detail_model", {}) or {}
                max_piece_html = max(50, int(detail_cfg.get("max_piece_prism_html_pieces", 600)))
                generate_group_html = bool(detail_cfg.get("generate_group_piece_html", False))
                if pieces:
                    prism_html = plot_dir / "02_geometria_3d_prismas_reais_completo.html"
                    legacy_prism_html = plot_dir / "17_modelo_peca_a_peca_prismas_reais.html"
                    exploded_prism_html = plot_dir / "18_modelo_peca_a_peca_explodido.html"
                    emit_progress(0.882, f"Pré-calculando malhas 3D peça-a-peça ({min(len(pieces), max_piece_html)}/{len(pieces)} peças)")
                    color_by = str(detail_cfg.get("piece_view_color_by", "assembly_unit"))
                    mounted_scale = float(detail_cfg.get("piece_view_mounted_connection_offset_scale", 0.0))
                    exploded_scale = float(detail_cfg.get("piece_view_exploded_connection_offset_scale", 0.60))
                    prism_batches = self.viz.prepare_stick_piece_mesh_batches(
                        pieces,
                        max_pieces=max_piece_html,
                        color_by=color_by,
                        connection_offset_scale=mounted_scale,
                    )
                    as_built_audit = {
                        "mode": "as_built" if abs(float(mounted_scale)) < 1.0e-9 else "mounted_with_offset",
                        "connection_offset_scale": float(mounted_scale),
                        "interpenetration_count": int(prism_batches.get("as_built_interpenetration_count", 0) or 0),
                        "node_connection_gap_piece_count": int(prism_batches.get("as_built_gap_piece_count", 0) or 0),
                        "interpenetration_samples": list(prism_batches.get("as_built_interpenetration_samples") or []),
                    }
                    detailed["as_built_audit"] = as_built_audit
                    try:
                        (detail_dir / "as_built_geometry_audit.json").write_text(
                            json.dumps(as_built_audit, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        GeometryService.write_csv(
                            detail_dir / "as_built_interpenetrations.csv",
                            as_built_audit.get("interpenetration_samples") or [],
                        )
                        audit_md = detail_dir / "09_auditoria_conectividade_e_cortes.md"
                        as_built_verdict = "FALHA" if int(as_built_audit.get("interpenetration_count", 0) or 0) > 0 else "OK"
                        with audit_md.open("a", encoding="utf-8") as fh:
                            fh.write(
                                "\n## Auditoria as-built pós-render\n"
                                f"- Veredito as-built: **{as_built_verdict}**\n"
                                f"- Interpenetrações físicas: **{int(as_built_audit.get('interpenetration_count', 0) or 0)}**\n"
                                f"- Peças com lacuna de conexão: **{int(as_built_audit.get('node_connection_gap_piece_count', 0) or 0)}**\n"
                                "- Regra: qualquer interpenetração em vista montada real reprova a geometria, mesmo se a conectividade topológica estiver fechada.\n"
                            )
                    except (OSError, TypeError, ValueError):
                        pass
                    emit_progress(0.886, "Renderizando HTML 3D peça-a-peça em posição de encaixe")
                    fig_prisms = self.viz.plotly_stick_pieces(
                        pieces,
                        max_pieces=max_piece_html,
                        render_mode="prismas reais",
                        precomputed_mesh_batches=prism_batches,
                        color_by=color_by,
                        connection_offset_scale=mounted_scale,
                    )
                    fig_prisms.write_html(prism_html)
                    # Alias de compatibilidade para relatórios antigos.
                    fig_prisms.write_html(legacy_prism_html)
                    emit_progress(0.888, "Renderizando HTML 3D peça-a-peça em vista explodida")
                    exploded_batches = self.viz.prepare_stick_piece_mesh_batches(
                        pieces,
                        max_pieces=max_piece_html,
                        color_by=color_by,
                        connection_offset_scale=exploded_scale,
                    )
                    self.viz.plotly_stick_pieces(
                        pieces,
                        max_pieces=max_piece_html,
                        render_mode="prismas reais",
                        precomputed_mesh_batches=exploded_batches,
                        color_by=color_by,
                        connection_offset_scale=exploded_scale,
                    ).write_html(exploded_prism_html)
                    group_files = []
                    if generate_group_html:
                        group_dir = plot_dir / "subconjuntos_html"
                        group_dir.mkdir(parents=True, exist_ok=True)
                        preferred_groups = [
                            "top_chord",
                            "bottom_chord",
                            "vertical",
                            "diagonal",
                            "top_transverse",
                            "bottom_transverse",
                            "top_bracing",
                            "bottom_bracing",
                            "cross_frame_bracing",
                            "support_pad",
                        ]
                        for group_name in preferred_groups:
                            subset = [r for r in pieces if str(r.get("member_group")) == group_name]
                            if not subset:
                                continue
                            safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in group_name)
                            path = group_dir / f"{safe_name}_peca_a_peca.html"
                            subset_max = min(max_piece_html, 600)
                            subset_batches = self.viz.prepare_stick_piece_mesh_batches(
                                subset,
                                max_pieces=subset_max,
                                color_by=color_by,
                                connection_offset_scale=mounted_scale,
                            )
                            self.viz.plotly_stick_pieces(
                                subset,
                                max_pieces=subset_max,
                                render_mode="prismas reais",
                                precomputed_mesh_batches=subset_batches,
                                color_by=color_by,
                                connection_offset_scale=mounted_scale,
                            ).write_html(path)
                            group_files.append(str(path))
                    detailed["piece_view_files"] = {
                        "real_prisms_all_html": str(prism_html),
                        "legacy_real_prisms_all_html": str(legacy_prism_html),
                        "exploded_real_prisms_all_html": str(exploded_prism_html),
                        "group_html": group_files,
                        "orthographic_overview_png": str(plot_dir / "16_vistas_cad_peca_a_peca.png"),
                        "piece_html_sampled_count": min(len(pieces), max_piece_html),
                        "piece_html_total_count": len(pieces),
                        "as_built_interpenetration_count": as_built_audit.get("interpenetration_count"),
                        "as_built_gap_piece_count": as_built_audit.get("node_connection_gap_piece_count"),
                    }
            except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
                emit_warning(
                    "WARN_PIECE_VIEW_EXPORT_FAILED",
                    f"Falha ao exportar visualizações peça-a-peça: {exc!r}",
                    stage="detail",
                )
        emit_progress(0.90, "Gerando recomendações e relatório")

        primary = [
            r
            for r in member_checks
            if r.get("member_role") == "primary"
        ]

        fs_primary = numeric_values(r.get("FS_min") for r in primary)
        fs_all_raw = numeric_values(r.get("FS_min_all_raw", r.get("FS_min")) for r in member_checks)
        fs_design = numeric_values(
            r.get("FS_design")
            for r in member_checks
            if r.get("design_relevant", True)
        )
        if not fs_design:
            fs_design = list(fs_primary)

        detailed_summary = detailed.get("summary", {})
        load_kgf = float(cfg.get("bridge", {}).get("load_total_kgf", 0.0))
        rupture = estimate_rupture_load(
            cfg,
            member_checks,
            support_checks,
            detailed,
            load_kgf,
        )
        predicted_breaking_load_kgf = safe_float(
            rupture.get("predicted_breaking_load_design_kgf"),
            None,
        )
        if predicted_breaking_load_kgf is None:
            fallback_fs = min(fs_design) if fs_design else (min(fs_primary) if fs_primary else None)
            if fallback_fs is not None:
                predicted_breaking_load_kgf = load_kgf * float(fallback_fs)

        min_glue_fs = min(
            (
                safe_float(r.get("FS_glue_shear"), 1.0e99) or 1.0e99
                for r in (detailed.get("glue_joints") or [])
                if safe_float(r.get("FS_glue_shear"), None) is not None
            ),
            default=None,
        )
        if min_glue_fs == 1.0e99:
            min_glue_fs = None

        competition_mass_g = safe_float(
            detailed_summary.get("competition_mass_g"),
            safe_float(detailed_summary.get("estimated_total_mass_g"), None),
        )
        installed_stick_mass_g = safe_float(
            detailed_summary.get("installed_stick_mass_g"),
            safe_float(detailed_summary.get("estimated_piece_mass_g_without_waste_scaling"), None),
        )
        wet_glue_mass_g = safe_float(
            detailed_summary.get("wet_glue_mass_g"),
            safe_float(detailed_summary.get("estimated_glue_mass_g"), None),
        )
        cured_glue_mass_g = safe_float(
            detailed_summary.get("cured_glue_mass_g"),
            wet_glue_mass_g,
        )
        assembly_procurement_mass_g = safe_float(
            detailed_summary.get("assembly_procurement_mass_g"),
            None,
        )

        metrics = {
            "n_nodes": len(nodes),
            "n_members": len(members),
            "n_active_supports": len(result.active_support_node_ids),
            "n_uplift_supports": len(result.inactive_support_node_ids),
            "n_inactive_tension_only_members": len(getattr(result, "inactive_tension_only_member_ids", set())),
            "tension_only_iterations": getattr(result, "tension_only_iterations", 0),
            "tension_only_converged": getattr(result, "tension_only_converged", True),
            "tension_only_compression_released_N_total": getattr(result, "tension_only_compression_released_N_total", 0.0),
            "equilibrium_error_N": result.equilibrium_error_N,
            "min_fs_primary": min(fs_primary) if fs_primary else None,
            "min_fs_all": min(fs_design) if fs_design else None,
            "min_fs_design": min(fs_design) if fs_design else None,
            "min_fs_all_raw": min(fs_all_raw) if fs_all_raw else None,
            "solver_status": result.status,
            "frame3dd_status": frame_result.get("status"),
            "estimated_sticks_total": detailed_summary.get("estimated_total_sticks_with_waste"),
            "competition_mass_g": competition_mass_g,
            "installed_stick_mass_g": installed_stick_mass_g,
            "wet_glue_mass_g": wet_glue_mass_g,
            "cured_glue_mass_g": cured_glue_mass_g,
            "assembly_procurement_mass_g": assembly_procurement_mass_g,
            "estimated_total_mass_g": detailed_summary.get("estimated_total_mass_g"),
            "mass_margin_g": detailed_summary.get("mass_margin_g"),
            "estimated_glue_mass_g": detailed_summary.get("estimated_glue_mass_g"),
            "min_glue_fs": safe_float(rupture.get("min_fs_glue"), min_glue_fs),
            "min_fs_glue": safe_float(rupture.get("min_fs_glue"), min_glue_fs),
            "min_support_fs": safe_float(rupture.get("min_fs_support"), None),
            "min_fs_support": safe_float(rupture.get("min_fs_support"), None),
            "min_fs_member_design": safe_float(rupture.get("min_fs_member_design"), min(fs_design) if fs_design else None),
            "predicted_breaking_load_kgf": predicted_breaking_load_kgf,
            "predicted_breaking_load_primary_kgf": rupture.get("predicted_breaking_load_primary_kgf"),
            "predicted_breaking_load_all_kgf": rupture.get("predicted_breaking_load_all_kgf"),
            "predicted_breaking_load_design_kgf": rupture.get("predicted_breaking_load_design_kgf"),
            "predicted_breaking_load_by_members_kgf": rupture.get("predicted_breaking_load_by_members_kgf"),
            "predicted_breaking_load_by_supports_kgf": rupture.get("predicted_breaking_load_by_supports_kgf"),
            "predicted_breaking_load_by_glue_kgf": rupture.get("predicted_breaking_load_by_glue_kgf"),
            "governing_limit_state": rupture.get("governing_limit_state"),
            "governing_fs": rupture.get("governing_fs"),
            "governing_mode": rupture.get("governing_mode"),
            "governing_member_id": rupture.get("governing_member_id"),
            "governing_group": rupture.get("governing_group"),
            "governing_joint_id": rupture.get("governing_joint_id"),
            "governing_support_node_id": rupture.get("governing_support_node_id"),
            "rupture_details": rupture,
            "mass_limit_nominal_g": detailed_summary.get("mass_limit_nominal_g"),
            "mass_limit_material_g": detailed_summary.get("mass_limit_material_g"),
            "mass_limit_planner_g": detailed_summary.get("mass_limit_planner_g"),
            "mass_limit_effective_g": detailed_summary.get("mass_limit_effective_g"),
            "mass_limit_effective_source": detailed_summary.get("mass_limit_effective_source"),
            "quarter_model_requested": quarter_summary.get("enabled_requested"),
            "quarter_model_used": quarter_summary.get("enabled_used"),
            "quarter_model_fallback_reason": quarter_summary.get("fallback_reason"),
            "mass_limit_g": detailed_summary.get("mass_limit_effective_g"),
            "n_weak_glue_joints": detailed_summary.get("n_weak_glue_joints"),
            "as_built_interpenetration_count": int((detailed.get("as_built_audit", {}) or {}).get("interpenetration_count", 0) or 0),
            "as_built_gap_piece_count": int((detailed.get("as_built_audit", {}) or {}).get("node_connection_gap_piece_count", 0) or 0),
        }
        assert_mass_compliant(metrics, cfg, source="pipeline_final")
        metrics["metric_strength_to_weight"] = (
            (predicted_breaking_load_kgf / competition_mass_g)
            if (predicted_breaking_load_kgf is not None and competition_mass_g and competition_mass_g > 1.0e-9)
            else None
        )
        metrics["metric_break_margin_kgf"] = (
            (predicted_breaking_load_kgf - float(cfg.get("analysis", {}).get("acceptance_min_design_breaking_load_kgf", 80.0)))
            if predicted_breaking_load_kgf is not None
            else None
        )
        try:
            mass_val = safe_float(
                detailed_summary.get("competition_mass_g"),
                safe_float(detailed_summary.get("estimated_total_mass_g"), None),
            )
            limit_val = safe_float(detailed_summary.get("mass_limit_effective_g"), None)
            if mass_val is not None and limit_val is not None:
                mass_passed = bool(mass_val <= limit_val + 1.0e-6)
                min_fs_primary_val = safe_float(metrics.get("min_fs_primary"), None)
                min_fs_design_val = safe_float(metrics.get("min_fs_design"), min_fs_primary_val)
                min_support_val = safe_float(metrics.get("min_support_fs"), None)
                min_glue_val = safe_float(metrics.get("min_glue_fs"), None)
                pred_break_val = safe_float(metrics.get("predicted_breaking_load_kgf"), None)
                load_target_kgf = max(
                    0.1,
                    float(
                        cfg.get("analysis", {}).get(
                            "acceptance_min_design_breaking_load_kgf",
                            cfg.get("planner", {}).get(
                                "target_breaking_load_kgf",
                                cfg.get("bridge", {}).get("load_total_kgf", 120.0),
                            ),
                        )
                    ),
                )
                acceptance_min_primary = float(cfg.get("analysis", {}).get("acceptance_min_primary_fs", 1.05))
                acceptance_min_support = float(cfg.get("analysis", {}).get("acceptance_min_support_fs", 1.0))
                acceptance_min_glue = float(cfg.get("analysis", {}).get("acceptance_min_glue_fs", 1.5))
                use_target_hard = bool(cfg.get("analysis", {}).get("use_target_min_fs_as_hard_acceptance", False))
                target_min_fs = float(cfg.get("analysis", {}).get("target_min_fs", 2.0))
                solver_regular = self._solver_is_regular(metrics.get("solver_status", ""))
                as_built_intersections = int(metrics.get("as_built_interpenetration_count", 0) or 0)
                as_built_required = bool(cfg.get("detail_model", {}).get("as_built_interpenetration_is_failure", True))
                structural_passed = bool(
                    solver_regular
                    and (not as_built_required or as_built_intersections == 0)
                    and (min_fs_primary_val is not None and min_fs_primary_val >= acceptance_min_primary)
                    and (min_support_val is None or min_support_val >= acceptance_min_support)
                    and (min_glue_val is None or min_glue_val >= acceptance_min_glue)
                    and ((not use_target_hard) or (min_fs_design_val is not None and min_fs_design_val >= target_min_fs))
                    and (pred_break_val is not None and pred_break_val >= load_target_kgf)
                )
                passed = bool(mass_passed and structural_passed)
                metrics["mass_constraint_passed"] = mass_passed
                metrics["mass_margin_g"] = limit_val - mass_val
                metrics["mass_violation_reason"] = (
                    ""
                    if mass_passed
                    else f"mass_above_effective_limit:{mass_val:.3f}>{limit_val:.3f}"
                )
                metrics["solution_structural_passed"] = structural_passed
                if strict_mass_acceptance:
                    metrics["solution_accepted"] = passed
                    if not passed:
                        if not mass_passed:
                            metrics["solution_block_reason"] = "mass_limit_exceeded"
                            emit_warning(
                                "WARN_STRICT_MASS_REJECTED_FINAL",
                                "Resultado final excede o limite efetivo de massa e não pode ser aceito como proposta final.",
                                stage="pipeline",
                            )
                        elif not solver_regular:
                            metrics["solution_block_reason"] = "solver_not_regular"
                            emit_warning(
                                "WARN_STRUCTURAL_REJECTED_FINAL",
                                "Resultado final foi rejeitado: solver estrutural não regular.",
                                stage="pipeline",
                            )
                        elif as_built_required and as_built_intersections > 0:
                            metrics["solution_block_reason"] = "as_built_interpenetration"
                            emit_warning(
                                "WARN_STRUCTURAL_REJECTED_FINAL",
                                f"Resultado final foi rejeitado: {as_built_intersections} interpenetrações na geometria as-built.",
                                stage="pipeline",
                            )
                        elif min_fs_primary_val is None or min_fs_primary_val < acceptance_min_primary:
                            metrics["solution_block_reason"] = "fs_primary_below_1"
                            emit_warning(
                                "WARN_STRUCTURAL_REJECTED_FINAL",
                                "Resultado final foi rejeitado: FS mínimo dos membros principais abaixo do limite de aceitação.",
                                stage="pipeline",
                            )
                        elif pred_break_val is None or pred_break_val < load_target_kgf:
                            metrics["solution_block_reason"] = "predicted_break_below_load"
                            emit_warning(
                                "WARN_STRUCTURAL_REJECTED_FINAL",
                                "Resultado final foi rejeitado: ruptura prevista abaixo da carga de projeto.",
                                stage="pipeline",
                            )
        except (TypeError, ValueError):
            pass

        if optimization and optimization.get("best"):
            best = optimization["best"]
            metrics["planner_score"] = safe_float(best.get("score"), None)
            metrics["planner_predicted_breaking_load_kgf"] = safe_float(
                best.get("predicted_breaking_load_kgf"),
                None,
            )
            if metrics.get("predicted_breaking_load_kgf") is None:
                metrics["predicted_breaking_load_kgf"] = metrics["planner_predicted_breaking_load_kgf"]
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
            metrics["planner_s1_macro_count"] = counts.get("S1_macro_candidates")
            metrics["planner_s2_fast_count"] = counts.get("S2_fast_screening_candidates")
            metrics["planner_s3_multicase_count"] = counts.get("S3_multi_loadcase_candidates")
            metrics["planner_s4_refine_count"] = counts.get("S4_geometry_refinement_candidates")
            metrics["planner_s5_sizing_count"] = counts.get("S5_member_sizing_candidates")
            metrics["planner_s6_topology_count"] = counts.get("S6_topology_candidates")
            metrics["planner_s7_fabrication_count"] = counts.get("S7_fabrication_candidates")
            metrics["planner_s8_final_count"] = counts.get("S8_final_validation_candidates")
            metrics["planner_solves_total"] = counts.get("solves_total")

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
        final_report_paths = self.report_bundle.generate(
            cfg,
            metrics,
            member_checks,
            detailed,
            optimization,
            warnings,
            self.output_root / "final_report",
        )

        # Sincroniza a validação final de otimização com a mesma base pós-detalhe
        # usada no relatório final (incluindo cola). Isso elimina divergência
        # entre S8 e relatório quando a governança muda após stick_pieces/glue_joints.
        try:
            opt_dir = self.output_root / "optimization"
            opt_dir.mkdir(parents=True, exist_ok=True)
            fv_path = opt_dir / "final_validation_summary.json"
            prior_fv: Dict[str, Any] = {}
            if fv_path.exists():
                try:
                    prior_fv = json.loads(fv_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, TypeError):
                    prior_fv = {}
            s8_rows = list((optimization or {}).get("s8_final_validation", []) or [])
            s8_ref = dict(s8_rows[0]) if s8_rows else {}
            target_break_sync = float(cfg.get("analysis", {}).get("acceptance_min_design_breaking_load_kgf", 100.0))
            torsion_80_20_target_sync = float(cfg.get("analysis", {}).get("acceptance_min_torsion_80_20_kgf", 80.0))
            mass_limit_sync = float(effective_mass_limit_g(cfg))
            final_break_sync = safe_float(metrics.get("predicted_breaking_load_kgf"), None)
            final_mass_sync = safe_float(
                detailed_summary.get("competition_mass_g"),
                safe_float(detailed_summary.get("estimated_total_mass_g"), None),
            )
            min_fs_member_sync = safe_float(
                metrics.get("min_fs_member_design"),
                safe_float(metrics.get("min_fs_design"), None),
            )
            min_fs_support_sync = safe_float(
                metrics.get("min_fs_support"),
                safe_float(metrics.get("min_support_fs"), None),
            )
            min_fs_glue_sync = safe_float(
                metrics.get("min_fs_glue"),
                safe_float(metrics.get("min_glue_fs"), None),
            )
            accept_min_primary_sync = float(cfg.get("analysis", {}).get("acceptance_min_primary_fs", 1.05))
            accept_min_glue_sync = float(cfg.get("analysis", {}).get("acceptance_min_glue_fs", 1.50))
            verdict_sync = "APROVADA"
            failed_sync = ""
            if not self._solver_is_regular(metrics.get("solver_status", "")):
                verdict_sync = "REPROVADA"
                failed_sync = "solver_irregular"
            elif not bool(metrics.get("equilibrium_ok", True)):
                verdict_sync = "REPROVADA"
                failed_sync = "equilibrium"
            elif final_mass_sync is not None and final_mass_sync > mass_limit_sync + 1.0e-6:
                verdict_sync = "REPROVADA"
                failed_sync = f"mass_above_limit:{final_mass_sync:.2f}>{mass_limit_sync:.2f}"
            elif bool(cfg.get("detail_model", {}).get("as_built_interpenetration_is_failure", True)) and int(metrics.get("as_built_interpenetration_count", 0) or 0) > 0:
                verdict_sync = "REPROVADA"
                failed_sync = f"as_built_interpenetration:{int(metrics.get('as_built_interpenetration_count', 0) or 0)}"
            elif final_break_sync is not None and final_break_sync < target_break_sync:
                verdict_sync = "REPROVADA"
                failed_sync = f"nominal_rupture_below_target:{final_break_sync:.2f}<{target_break_sync:.2f}"
            elif (
                str(s8_ref.get("governing_strength_case", "")) == "torsion_80_20"
                and safe_float(s8_ref.get("governing_strength_case_breaking_load_kgf"), None) is not None
                and float(s8_ref.get("governing_strength_case_breaking_load_kgf")) < torsion_80_20_target_sync
            ):
                verdict_sync = "REPROVADA"
                failed_sync = f"torsion_80_20_below_target:{float(s8_ref.get('governing_strength_case_breaking_load_kgf')):.2f}<{torsion_80_20_target_sync:.2f}"
            elif min_fs_member_sync is not None and min_fs_member_sync < accept_min_primary_sync:
                verdict_sync = "REPROVADA"
                failed_sync = f"fs_member_below_acceptance:{min_fs_member_sync:.3f}<{accept_min_primary_sync:.3f}"
            elif min_fs_glue_sync is not None and min_fs_glue_sync < accept_min_glue_sync:
                verdict_sync = "REPROVADA"
                failed_sync = f"fs_glue_below_acceptance:{min_fs_glue_sync:.3f}<{accept_min_glue_sync:.3f}"

            final_validation_sync = {
                "verdict": verdict_sync,
                "failed_restriction": failed_sync,
                "validation_stage": "post_detail",
                "predicted_breaking_load_kgf": final_break_sync,
                "predicted_breaking_load_system_kgf": final_break_sync,
                "predicted_breaking_load_proxy_kgf": s8_ref.get("predicted_breaking_load_proxy_kgf", prior_fv.get("predicted_breaking_load_proxy_kgf")),
                "predicted_breaking_load_by_members_kgf": metrics.get("predicted_breaking_load_by_members_kgf"),
                "predicted_breaking_load_by_supports_kgf": metrics.get("predicted_breaking_load_by_supports_kgf"),
                "predicted_breaking_load_by_glue_kgf": metrics.get("predicted_breaking_load_by_glue_kgf"),
                "target_breaking_load_kgf": target_break_sync,
                "nominal_required_breaking_load_kgf": target_break_sync,
                "torsion_80_20_required_breaking_load_kgf": torsion_80_20_target_sync,
                "competition_mass_g": final_mass_sync,
                "min_fs_design": min_fs_member_sync,
                "min_fs_member_design": min_fs_member_sync,
                "min_fs_support": min_fs_support_sync,
                "min_fs_glue": min_fs_glue_sync,
                "governing_limit_state": metrics.get("governing_limit_state"),
                "governing_fs": metrics.get("governing_fs"),
                "solver_regular": self._solver_is_regular(metrics.get("solver_status", "")),
                "equilibrium_ok": bool(metrics.get("equilibrium_ok", True)),
                # Separação explícita: o alvo nominal usa a ruptura de sistema;
                # o requisito 80/20 usa o caso de robustez correspondente, para
                # não mascarar a diferença entre membro, apoio e cola.
                "hits_min_required_80kgf": bool(
                    (safe_float(s8_ref.get("governing_strength_case_breaking_load_kgf"), None) or 0.0) >= torsion_80_20_target_sync
                ),
                "hits_torsion_80_20_80kgf": bool(
                    (safe_float(s8_ref.get("governing_strength_case_breaking_load_kgf"), None) or 0.0) >= torsion_80_20_target_sync
                ),
                "hits_target_80kgf": bool((final_break_sync or 0.0) >= 80.0),
                "hits_nominal_100kgf": bool((final_break_sync or 0.0) >= target_break_sync),
                "governing_case": s8_ref.get("governing_case", prior_fv.get("governing_case")),
                "governing_strength_case": s8_ref.get("governing_strength_case", prior_fv.get("governing_strength_case")),
                "governing_service_case": s8_ref.get("governing_service_case", prior_fv.get("governing_service_case")),
                "governing_contact_case": s8_ref.get("governing_contact_case", prior_fv.get("governing_contact_case")),
                "governing_case_breaking_load_kgf": s8_ref.get("governing_case_breaking_load_kgf", prior_fv.get("governing_case_breaking_load_kgf")),
                "governing_strength_case_breaking_load_kgf": s8_ref.get(
                    "governing_strength_case_breaking_load_kgf",
                    prior_fv.get("governing_strength_case_breaking_load_kgf"),
                ),
                "governing_service_case_breaking_load_kgf": s8_ref.get(
                    "governing_service_case_breaking_load_kgf",
                    prior_fv.get("governing_service_case_breaking_load_kgf"),
                ),
                "governing_contact_case_breaking_load_kgf": s8_ref.get(
                    "governing_contact_case_breaking_load_kgf",
                    prior_fv.get("governing_contact_case_breaking_load_kgf"),
                ),
                "governing_member_id": s8_ref.get("governing_member_id", prior_fv.get("governing_member_id")),
                "governing_member_group": s8_ref.get("governing_member_group", prior_fv.get("governing_member_group")),
                "governing_mode": s8_ref.get("governing_mode", prior_fv.get("governing_mode")),
                "as_built_interpenetration_count": int(metrics.get("as_built_interpenetration_count", 0) or 0),
                "as_built_gap_piece_count": int(metrics.get("as_built_gap_piece_count", 0) or 0),
                "case_metrics": s8_ref.get("case_metrics", prior_fv.get("case_metrics", [])),
                "source": "pipeline_post_detail_sync",
            }
            fv_path.write_text(
                json.dumps(final_validation_sync, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError, KeyError) as exc:
            emit_warning(
                "WARN_FINAL_VALIDATION_POST_DETAIL_SYNC_FAILED",
                f"Falha ao sincronizar validação final pós-detalhe: {exc!r}",
                stage="pipeline",
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

        final_mass = safe_float(
            detailed_summary.get("competition_mass_g"),
            safe_float(detailed_summary.get("estimated_total_mass_g"), None),
        )
        final_break = safe_float(metrics.get("predicted_breaking_load_kgf"), None)
        integrity_rows: List[Dict[str, Any]] = []
        opt_fv_path = self.output_root / "optimization" / "final_validation_summary.json"
        opt_final: Dict[str, Any] = {}
        if opt_fv_path.exists():
            try:
                opt_final = json.loads(opt_fv_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                opt_final = {"read_error": True}
        opt_break = safe_float(opt_final.get("predicted_breaking_load_kgf"), None)
        opt_mass = safe_float(opt_final.get("competition_mass_g"), None)
        mass_delta = None if (final_mass is None or opt_mass is None) else final_mass - opt_mass
        break_delta = None if (final_break is None or opt_break is None) else final_break - opt_break
        integrity_status = "OK"
        integrity_reasons: List[str] = []
        opt_validation_stage = str(opt_final.get("validation_stage", "") or "").strip().lower()
        opt_is_pre_detail = opt_validation_stage.startswith("pre_detail")
        if not opt_fv_path.exists():
            integrity_status = "WARN"
            integrity_reasons.append("optimization/final_validation_summary.json ausente no run atual")
        if mass_delta is not None and abs(mass_delta) > 2.0:
            if opt_is_pre_detail:
                if integrity_status == "OK":
                    integrity_status = "WARN"
                integrity_reasons.append(
                    f"massa final difere da validação pré-detalhamento em {mass_delta:+.2f} g"
                )
            else:
                integrity_status = "MISMATCH"
                integrity_reasons.append(f"massa final difere da validação S8 em {mass_delta:+.2f} g")
        if break_delta is not None and abs(break_delta) > 2.0:
            if opt_is_pre_detail:
                if integrity_status == "OK":
                    integrity_status = "WARN"
                integrity_reasons.append(
                    f"ruptura final difere da validação pré-detalhamento em {break_delta:+.2f} kgf"
                )
            else:
                integrity_status = "MISMATCH"
                integrity_reasons.append(f"ruptura final difere da validação S8 em {break_delta:+.2f} kgf")
        if final_mass is not None and final_mass > effective_mass_limit_g(cfg) + 1.0e-6:
            if integrity_status == "OK":
                integrity_status = "CONSISTENTE_REPROVADO"
            integrity_reasons.append("relatório final acima do limite de massa")
        final_verdict = str(opt_final.get("verdict", "") or "").strip().upper()
        failed_restriction = str(opt_final.get("failed_restriction", "") or "").strip()
        if final_verdict == "REPROVADA":
            if integrity_status == "OK":
                integrity_status = "CONSISTENTE_REPROVADO"
            elif integrity_status == "WARN":
                integrity_status = "WARN_REPROVADO"
            integrity_reasons.append(
                "validação final pós-detalhe reprovada"
                + (f": {failed_restriction}" if failed_restriction else "")
            )
        elif final_verdict and final_verdict != "APROVADA" and integrity_status == "OK":
            integrity_status = "WARN"
            integrity_reasons.append(f"veredito final não reconhecido: {final_verdict}")
        integrity = {
            "run_id": run_id,
            "status": integrity_status,
            "reasons": integrity_reasons,
            "final_report": {
                "predicted_breaking_load_kgf": final_break,
                "competition_mass_g": final_mass,
                "mass_limit_g": effective_mass_limit_g(cfg),
            },
            "optimization_final_validation": opt_final,
            "optimization_validation_stage": opt_validation_stage or None,
            "final_verdict": final_verdict or None,
            "failed_restriction": failed_restriction or None,
            "deltas": {
                "break_final_minus_s8_kgf": break_delta,
                "mass_final_minus_s8_g": mass_delta,
            },
            "cleaned_output_dirs": cleaned_dirs,
        }
        (self.output_root / "output_integrity_report.json").write_text(
            json.dumps(integrity, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (self.output_root / "final_report" / "11_integridade_execucao.md").write_text(
            "# Integridade da execução\n\n"
            f"- Run ID: `{run_id}`\n"
            f"- Status: **{integrity_status}**\n"
            f"- Diretórios limpos no início: `{', '.join(cleaned_dirs) if cleaned_dirs else 'nenhum'}`\n"
            f"- Massa final: **{final_mass if final_mass is not None else '—'} g**\n"
            f"- Massa S8: **{opt_mass if opt_mass is not None else '—'} g**\n"
            f"- Ruptura final: **{final_break if final_break is not None else '—'} kgf**\n"
            f"- Ruptura S8: **{opt_break if opt_break is not None else '—'} kgf**\n\n"
            "## Motivos / alertas\n"
            + ("\n".join(f"- {r}" for r in integrity_reasons) if integrity_reasons else "- Nenhum alerta de divergência entre S8 e relatório final; veredito final aprovado.")
            + "\n\nArquivo bruto: `../output_integrity_report.json`.\n",
            encoding="utf-8",
        )
        if integrity_status == "MISMATCH":
            emit_warning(
                "WARN_OUTPUT_INTEGRITY_MISMATCH",
                "O relatório final divergiu da validação S8. Consulte final_report/11_integridade_execucao.md.",
                stage="pipeline",
            )
        debug_logger.event(
            "mass_limit_check",
            stage="pipeline",
            metrics={
                "mass_g": final_mass,
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
        focused_zip_path = self.output_root / "pacote_focado_fabricacao_e_calculo.zip"
        self.zip_focused_outputs(focused_zip_path)
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
            "final_report_paths": final_report_paths,
            "edital_checks": edital_checks,
            "frame3dd_result": frame_result,
            "zip_path": zip_path,
            "focused_zip_path": focused_zip_path,
            "optimization": optimization,
            "execution_logs": execution_logs,
            "warnings": warnings,
            "planner_debug_jsonl": str(debug_logger.jsonl_path),
            "planner_debug_summary": str(debug_logger.summary_path),
        }


    def zip_focused_outputs(self, zip_path: str | Path) -> Path:
        """Create a smaller, analysis-oriented package.

        The full output zip remains available for debugging. This package is for
        fabrication/review: fewer files, clearer responsibilities, no repeated
        planner scratch outputs unless they directly support a decision.
        """
        zp = Path(zip_path)
        if zp.exists():
            zp.unlink()

        include_paths = [
            "config_used.json",
            "config_requested.json",
            "final_report/00_resumo_executivo.md",
            "final_report/01_memorial_calculo.md",
            "final_report/02_guia_fabricacao.md",
            "final_report/03_checklist_construcao.md",
            "final_report/04_plano_montagem_detalhado.md",
            "final_report/04_subconjuntos_montagem.md",
            "final_report/05_mapa_juntas_por_tipo.md",
            "final_report/06_sequencia_montagem.md",
            "final_report/07_avaliacao_contato_carga.md",
            "final_report/08_auditoria_secao_e_realismo.md",
            "final_report/09_auditoria_conectividade_e_cortes.md",
            "final_report/10_debug_calculos_criticos.md",
            "final_report/11_integridade_execucao.md",
            "final_report/focused_outputs_manifest.json",
            "output_integrity_report.json",
            "run_metadata.json",
            "final_report/executive_summary.json",
            "optimization/final_validation_summary.json",
            "optimization/pipeline_trace.json",
            "details/cutting_list.csv",
            "details/stick_pieces.csv",
            "details/glue_joints.csv",
            "details/assembly_tutorial.md",
            "details/assembly_steps.csv",
            "final_report/04_plano_pecas_por_medida.csv",
            "final_report/04_subconjuntos_montagem.csv",
            "final_report/05_mapa_juntas_por_tipo.csv",
            "final_report/06_sequencia_montagem.csv",
            "final_report/load_contact_assessment.csv",
            "plots/01_geometria_3d_fs_uso.html",
            "plots/geometria_3d_interativa.html",
            "plots/02_geometria_3d_prismas_reais_completo.html",
            "plots/16_vistas_cad_peca_a_peca.png",
            "plots/17_modelo_peca_a_peca_prismas_reais.html",
        ]

        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for rel in include_paths:
                p = self.output_root / rel
                if p.exists() and p.is_file():
                    z.write(p, rel)
        return zp

    def zip_outputs(self, zip_path: str | Path) -> Path:
        zp = Path(zip_path)

        if zp.exists():
            zp.unlink()

        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for p in self.output_root.rglob("*"):
                if p.is_file() and p != zp:
                    z.write(p, p.relative_to(self.output_root))

        return zp
