from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable

from src.services.config_service import ConfigService
from src.services.geometry_service import GeometryService
from src.services.postprocessor import PostProcessor
from src.services.recommendation_service import RecommendationService
from src.services.report_service import ReportService
from src.services.visualization_service import VisualizationService
from src.services.stick_detail_service import StickDetailService
from src.services.detail_visualization_service import DetailVisualizationService
from src.services.design_optimizer import DesignOptimizer
from src.solvers.frame3dd_adapter import Frame3DDAdapter
from src.solvers.linear_truss_solver import LinearTrussSolver


def safe_float(value: Any, default: float | None = None) -> float | None:
    """Converte para float sem quebrar com None, string vazia, NaN, infinito ou texto."""
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


def numeric_values(values: Iterable[Any]) -> list[float]:
    """Filtra apenas valores numéricos válidos."""
    out: list[float] = []

    for value in values:
        v = safe_float(value, None)

        if v is not None:
            out.append(v)

    return out


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
        self.optimizer = DesignOptimizer()

    def run(self, cfg: Dict) -> Dict:
        cfg = self.config_service.normalize(cfg)

        self.output_root.mkdir(parents=True, exist_ok=True)

        model_dir = self.output_root / "model"
        solver_dir = self.output_root / "opensees"
        plot_dir = self.output_root / "plots"
        detail_dir = self.output_root / "details"
        frame_dir = self.output_root / "frame3dd"
        report_dir = self.output_root / "reports"

        nodes, members, supports, loads = self.geometry.generate(cfg)
        self.geometry.export_csvs(cfg, model_dir)

        result = self.solver.solve(
            nodes,
            members,
            supports,
            loads,
            unilateral_supports=bool(cfg["bridge"].get("unilateral_supports", True)),
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
            frame_result = self.frame3dd.run(
                cfg,
                frame_input,
                frame_dir / "ponte_palitos.out",
            )

            (frame_dir / "frame3dd_status.json").write_text(
                json.dumps(frame_result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        detailed = self.detail.analyze(
            cfg,
            nodes,
            members,
            result.member_results,
            member_checks,
            detail_dir,
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

        if cfg.get("detail_model", {}).get("generate_piece_views", True):
            self.detail_viz.save_all(detailed, plot_dir)

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
        }

        optimization = None

        if cfg.get("analysis", {}).get("optimize_variants", True):
            try:
                optimization = self.optimizer.run(
                    cfg,
                    self.output_root / "optimization",
                )
            except Exception as exc:
                optimization = {
                    "error": repr(exc),
                    "variants": [],
                    "best": None,
                }

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

        self.config_service.save(cfg, self.output_root / "config_used.json")

        zip_path = self.output_root / "resultados_simulacao.zip"
        self.zip_outputs(zip_path)

        return {
            "cfg": cfg,
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
            "frame3dd_result": frame_result,
            "zip_path": zip_path,
            "optimization": optimization,
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