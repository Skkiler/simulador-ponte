from __future__ import annotations

import json

from src.services.report_bundle_service import ReportBundleService
from src.services.report_service import ReportService


def test_final_report_bundle_is_created(base_cfg: dict, tmp_path) -> None:
    cfg = base_cfg
    cfg["analysis"]["acceptance_min_design_breaking_load_kgf"] = 80.0
    metrics = {
        "predicted_breaking_load_kgf": 70.0,
        "competition_mass_g": 980.0,
        "mass_limit_effective_g": 1000.0,
        "mass_compliant": True,
        "competition_mass_compliant": True,
        "solver_status": "singular_lstsq_rank_4_of_6",
        "equilibrium_ok": False,
        "min_fs_primary": 1.02,
        "min_fs_design": 1.02,
        "min_support_fs": 1.10,
        "min_glue_fs": 1.60,
        "member_sizing_plan": [],
    }
    detailed = {
        "summary": {
            "installed_stick_mass_g": 860.0,
            "wet_glue_mass_g": 95.0,
            "cured_glue_mass_g": 47.5,
            "evaporated_glue_water_g": 47.5,
            "competition_mass_g": 907.5,
            "competition_mass_margin_g": 92.5,
            "purchased_blank_sticks_needed": 760,
            "purchased_stick_mass_g": 1064.0,
            "cutting_scrap_mass_g": 204.0,
            "assembly_procurement_mass_g": 1159.0,
            "n_weak_glue_joints": 3,
            "extra_sticks_for_waste": 30,
            "estimated_total_sticks_with_waste": 760,
        }
    }
    out = ReportBundleService().generate(
        cfg,
        metrics,
        member_checks=[],
        detailed=detailed,
        optimization={"stage1": [{"feasible": True, "candidate_id": "S1-0001"}]},
        warnings=[],
        out_dir=tmp_path / "final_report",
    )
    assert (tmp_path / "final_report" / "index.md").exists()
    assert (tmp_path / "final_report" / "executive_summary.json").exists()
    summary = json.loads((tmp_path / "final_report" / "executive_summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "REPROVADA"
    assert "predicted_breaking_load_kgf" in summary
    assert "competition_mass_g" in summary
    assert out["index_md"].endswith("index.md")


def test_markdown_report_uses_full_model_when_quarter_fallback(base_cfg: dict, tmp_path) -> None:
    cfg = base_cfg
    metrics = {
        "solver_status": "regular",
        "equilibrium_error_N": 0.0,
        "min_fs_primary": 1.2,
        "min_fs_all": 1.1,
        "predicted_breaking_load_kgf": 90.0,
        "quarter_model_used": False,
        "quarter_model_fallback_reason": "symmetry_validation_failed",
    }
    report_path = ReportService().write_markdown(
        cfg,
        metrics,
        recommendations={"summary": "ok", "suggestions": []},
        out_path=tmp_path / "report.md",
        detailed={"summary": {"competition_mass_g": 900.0, "mass_margin_g": 100.0}},
    )
    text = report_path.read_text(encoding="utf-8")
    assert "Análise executada no modelo completo." in text
    assert "Projeto analisado por 1/4" not in text
