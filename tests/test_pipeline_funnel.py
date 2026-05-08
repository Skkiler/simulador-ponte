from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

from src.services.active_design_planner import ActiveDesignPlanner
from src.services.report_bundle_service import ReportBundleService
from src.services.staged_fidelity_funnel import StagedFidelityFunnelPlanner


def _fake_case_eval(
    self: StagedFidelityFunnelPlanner,
    cfg: Dict[str, Any],
    load_case_name: str,
    *,
    stage_name: str,
    tension_only: bool,
) -> Dict[str, Any]:
    b = cfg.get("bridge", {}) or {}
    side = str(b.get("side_truss_type", "Pratt_symmetric"))
    pattern_bonus = {
        "Pratt_symmetric": 1.3,
        "Howe_inverted": 0.7,
        "Warren_symmetric": 1.0,
        "Warren_mid_braced": 0.9,
    }.get(side, 0.6)
    h = float(b.get("center_height_mm", 300.0))
    panel = float(b.get("panel_mm", 100.0))
    width = float(b.get("width_mm", 160.0))

    case_factor = {
        "center": 1.00,
        "left_offset": 0.98,
        "right_offset": 0.98,
        "torsion_60_40": 0.92,
        "lateral_imperfection": 0.90,
        "self_weight": 0.95,
    }.get(str(load_case_name), 1.0)

    predicted = max(30.0, (70.0 + 0.08 * h + pattern_bonus * 8.0 - 0.03 * abs(width - 160.0)) * case_factor)
    fs_design = max(0.4, (0.85 + 0.0025 * h - 0.001 * abs(panel - 100.0) + 0.08 * pattern_bonus) * case_factor)
    mass_proxy = max(500.0, 640.0 + 0.45 * width + 0.10 * h)
    max_disp = max(1.0, 22.0 - 0.05 * h + 0.03 * abs(panel - 100.0))

    return {
        "case": str(load_case_name),
        "solver_status": "regular",
        "solver_regular": True,
        "equilibrium_error_N": 0.0,
        "equilibrium_ok": True,
        "equilibrium_tol_N": 1.0,
        "min_fs_primary": fs_design,
        "min_fs_design": fs_design,
        "predicted_breaking_load_proxy_kgf": predicted,
        "max_displacement_proxy_mm": max_disp,
        "max_compression_proxy_N": 30.0,
        "max_tension_proxy_N": 32.0,
        "buckling_risk_proxy": 0.32,
        "mass_proxy_g": mass_proxy,
        "load_path_score": 0.72,
        "support_reaction_balance": 0.84,
        "topology_stability_proxy": 1.0,
        "nodal_stability_proxy": 0.88,
        "member_checks": [],
        "support_checks": [],
        "member_results": [
            {"member_id": 1, "N_N": 12.0},
            {"member_id": 2, "N_N": -6.0},
            {"member_id": 3, "N_N": 0.2},
        ],
        "node_results": [],
        "near_zero_member_ids": [3],
        "quick_mass_g": mass_proxy,
        "nodes": [],
        "members": [],
        "supports": [],
        "loads": [],
    }


def _fake_trust_region_refine(
    self: StagedFidelityFunnelPlanner,
    cfg: Dict[str, Any],
    load_cases: List[str],
    *,
    stage_name: str,
    tension_only: bool = False,
) -> Dict[str, Any]:
    c = copy.deepcopy(cfg)
    c.setdefault("bridge", {})["center_height_mm"] = float(c["bridge"].get("center_height_mm", 300.0)) + 5.0
    summary = self._multi_case_summary(c, load_cases, stage_name=stage_name, tension_only=True)
    return {
        "best_cfg": c,
        "best_summary": summary,
        "trace_rows": [
            {
                "iteration": 1,
                "mutation": "height_plus",
                "delta_height_mm": 5.0,
                "delta_panel_mm": 0.0,
                "delta_width_mm": 0.0,
                "radius_height_mm": 30.0,
                "radius_panel_mm": 15.0,
                "radius_width_mm": 20.0,
                "objective": summary.get("objective"),
                "accepted": True,
            }
        ],
        "before": {
            "center_height_mm": float(cfg.get("bridge", {}).get("center_height_mm", 0.0)),
            "panel_mm": float(cfg.get("bridge", {}).get("panel_mm", 0.0)),
            "width_mm": float(cfg.get("bridge", {}).get("width_mm", 0.0)),
        },
        "after": {
            "center_height_mm": float(c.get("bridge", {}).get("center_height_mm", 0.0)),
            "panel_mm": float(c.get("bridge", {}).get("panel_mm", 0.0)),
            "width_mm": float(c.get("bridge", {}).get("width_mm", 0.0)),
        },
    }


def _fake_member_sizing_pass(
    self: StagedFidelityFunnelPlanner,
    cfg: Dict[str, Any],
    load_cases: List[str],
    *,
    stage_name: str,
    tension_only: bool = False,
) -> Dict[str, Any]:
    summary = self._multi_case_summary(cfg, load_cases, stage_name=stage_name, tension_only=True)
    row = {
        "member_id": 1,
        "group": "top_chord",
        "N_N": -12.0,
        "compression_direct_util": 0.8,
        "tension_util": None,
        "buckling_util_y": 0.5,
        "buckling_util_z": 0.4,
        "beam_column_util": 0.7,
        "governing_mode": "buckling_y",
        "FS_min": 1.4,
        "utilization": 1.0 / 1.4,
        "action": "reinforce",
        "n_sticks_current": 2,
        "n_sticks_recommended": 3,
        "delta_mass_g": 5.0,
        "reason": "critical",
        "can_be_mass_donor": False,
    }
    return {
        "best_cfg": cfg,
        "summary": summary,
        "trace_rows": [row],
        "donors": [],
        "critical": [row],
        "before_after": [
            {
                "member_id": 1,
                "group": "top_chord",
                "before_n_sticks": 2,
                "after_n_sticks": 3,
                "delta_mass_g": 5.0,
                "action": "reinforce",
            }
        ],
    }


def _fake_topology_cleanup(
    self: StagedFidelityFunnelPlanner,
    cfg: Dict[str, Any],
    load_cases: List[str],
    *,
    stage_name: str,
    tension_only: bool = False,
) -> Dict[str, Any]:
    summary = self._multi_case_summary(cfg, load_cases, stage_name=stage_name, tension_only=True)
    return {
        "best_cfg": cfg,
        "summary": summary,
        "trace_rows": [
            {
                "iteration": 1,
                "operation": "remove_member",
                "objective": summary.get("objective"),
                "member_id": 3,
            }
        ],
        "removed_members": [{"iteration": 1, "member_id": 3, "reason": "low_force_all_cases"}],
        "mixed_patterns": [{"iteration": 1, "panel_side_truss_pattern": '{"0":"Pratt_symmetric"}'}],
        "zero_force_diag": [{"iteration": 1, "member_id": 3, "threshold_N": 2.0}],
        "mass_realloc": [{"topology_freed_mass_pool_g": 8.0, "before_mass_proxy_g": 820.0, "after_mass_proxy_g": 812.0}],
    }


def _install_fast_funnel_mocks(monkeypatch, include_detail_calls: list[bool] | None = None) -> None:
    monkeypatch.setattr(StagedFidelityFunnelPlanner, "_evaluate_case_cached", _fake_case_eval)
    monkeypatch.setattr(StagedFidelityFunnelPlanner, "_trust_region_refine", _fake_trust_region_refine)
    monkeypatch.setattr(StagedFidelityFunnelPlanner, "_member_sizing_pass", _fake_member_sizing_pass)
    monkeypatch.setattr(StagedFidelityFunnelPlanner, "_topology_cleanup", _fake_topology_cleanup)

    def _fake_eval_config(self, cfg, include_detail=False, detail_dir=None):
        if include_detail_calls is not None:
            include_detail_calls.append(bool(include_detail))
        summary = {
            "installed_stick_mass_g": 810.0,
            "purchased_stick_mass_g": 960.0,
            "cutting_scrap_mass_g": 150.0,
            "wet_glue_mass_g": 92.0,
            "cured_glue_mass_g": 46.0,
            "competition_mass_g": 856.0,
            "mass_limit_effective_g": 1000.0,
            "n_weak_glue_joints": 0,
        }
        return {
            "solver_status": "regular",
            "equilibrium_ok": True,
            "min_fs_primary": 1.25,
            "min_fs_design": 1.25,
            "predicted_breaking_load_kgf": 92.0,
            "mass_g": 856.0,
            "competition_mass_g": 856.0,
            "detailed": {
                "summary": summary,
                "cutting_list": [{"cut_length_mm": 100, "quantity": 2}],
                "glue_joints": [{"joint_id": "J1", "FS_glue_shear": 2.5}],
            },
        }

    monkeypatch.setattr(ActiveDesignPlanner, "_evaluate_config", _fake_eval_config)


def _run_fast_funnel(base_cfg: dict, tmp_path: Path, monkeypatch, include_detail_calls=None, *, install_mocks: bool = True):
    cfg = copy.deepcopy(base_cfg)
    cfg["planner_pipeline"]["mode"] = "staged_fidelity_funnel"
    cfg["planner_pipeline"]["macro_candidates_count"] = 12
    cfg["planner_pipeline"]["fast_screening_keep_top_k"] = 3
    cfg["planner_pipeline"]["multi_loadcase_keep_top_k"] = 2
    cfg["planner_pipeline"]["geometry_refinement_keep_top_k"] = 1
    cfg["analysis"]["staged_fidelity_funnel_enabled"] = True

    if install_mocks:
        _install_fast_funnel_mocks(monkeypatch, include_detail_calls=include_detail_calls)

    planner = ActiveDesignPlanner()
    return planner.run(cfg, tmp_path / "optimization")


def test_pipeline_funnel_limits(base_cfg: dict, tmp_path, monkeypatch) -> None:
    result = _run_fast_funnel(base_cfg, tmp_path, monkeypatch)

    macro_limit = int(base_cfg.get("planner_pipeline", {}).get("macro_candidates_count", 12))
    assert len(result["s1_macro"]) <= macro_limit
    assert result["stage_counts"]["S2_fast_screening_top_k"] <= 3
    assert result["stage_counts"]["S3_multi_loadcase_top_k"] <= 2
    assert len(result["s7_fabrication"]) == 1

    selected_top3 = json.loads((tmp_path / "optimization" / "selected_top3.json").read_text(encoding="utf-8"))
    selected_top2 = json.loads((tmp_path / "optimization" / "selected_top2.json").read_text(encoding="utf-8"))
    assert len(selected_top3) <= 3
    assert len(selected_top2) <= 2


def test_no_fabrication_before_s7(base_cfg: dict, tmp_path, monkeypatch) -> None:
    include_detail_calls: list[bool] = []
    _run_fast_funnel(base_cfg, tmp_path, monkeypatch, include_detail_calls=include_detail_calls)

    assert include_detail_calls
    assert include_detail_calls.count(True) == 1
    assert (tmp_path / "optimization" / "fabrication_summary.csv").exists()


def test_geometry_refinement_trust_region(base_cfg: dict) -> None:
    cfg = copy.deepcopy(base_cfg)
    cfg["planner_pipeline"]["mode"] = "staged_fidelity_funnel"
    cfg["analysis"]["staged_fidelity_funnel_enabled"] = True
    cfg["bridge"]["center_height_mm"] = 250.0
    cfg["bridge"]["panel_mm"] = 120.0
    cfg["bridge"]["width_mm"] = 140.0

    planner = ActiveDesignPlanner()
    funnel = StagedFidelityFunnelPlanner(planner)

    def _mock_summary(_cfg, _load_cases, *, stage_name, tension_only):
        b = _cfg["bridge"]
        h = float(b.get("center_height_mm", 0.0))
        p = float(b.get("panel_mm", 0.0))
        w = float(b.get("width_mm", 0.0))
        # Ótimo em (300,100,160) para permitir melhoria e depois contração.
        obj = 120.0 - ((h - 300.0) ** 2) / 400.0 - ((p - 100.0) ** 2) / 300.0 - ((w - 160.0) ** 2) / 500.0
        return {
            "objective": obj,
            "predicted_breaking_load_proxy_kgf": 85.0 + 0.08 * obj,
            "min_fs_design_proxy": max(0.1, 0.6 + 0.01 * obj),
            "dead_weight_proxy_g": 800.0,
            "solver_regular": True,
            "equilibrium_ok": True,
            "multi_case_zero_force_members": 1,
            "nodal_stability_proxy": 0.9,
            "lateral_stability_proxy": 0.8,
            "support_reaction_balance": 0.8,
            "load_path_score": 0.75,
            "topology_stability_proxy": 1.0,
            "cases": [],
            "zero_force_member_ids": [3],
            "geometry_hash": "g",
            "topology_hash": "t",
            "sizing_hash": "s",
            "load_case_hash": "l",
        }

    funnel._multi_case_summary = _mock_summary  # type: ignore[assignment]

    refined = funnel._trust_region_refine(cfg, ["center", "left_offset"], stage_name="S4")
    trace_rows = refined["trace_rows"]

    assert trace_rows
    for row in trace_rows:
        assert abs(float(row["delta_height_mm"])) <= float(row["radius_height_mm"]) + 1.0e-9
        assert abs(float(row["delta_panel_mm"])) <= float(row["radius_panel_mm"]) + 1.0e-9
        assert abs(float(row["delta_width_mm"])) <= float(row["radius_width_mm"]) + 1.0e-9

    initial = float(cfg["local_geometry_refinement"]["initial_delta_height_mm"])
    radii = [float(r["radius_height_mm"]) for r in trace_rows]

    assert radii
    assert all(r > 0.0 for r in radii)
    assert max(radii) <= initial * float(cfg["local_geometry_refinement"].get("expand_factor", 1.25)) ** 2 + 1.0e-9


def test_diversity_preservation() -> None:
    rows = [
        {"candidate_id": "A", "global_pattern": "pratt", "quick_score": 100.0},
        {"candidate_id": "B", "global_pattern": "pratt", "quick_score": 99.0},
        {"candidate_id": "C", "global_pattern": "pratt", "quick_score": 98.0},
        {"candidate_id": "D", "global_pattern": "warren", "quick_score": 97.0},
        {"candidate_id": "E", "global_pattern": "howe", "quick_score": 96.0},
    ]
    selected = StagedFidelityFunnelPlanner._pick_with_diversity(rows, top_k=3, key_field="global_pattern")
    patterns = {r["global_pattern"] for r in selected}
    assert len(selected) == 3
    assert len(patterns) >= 2


def test_topology_mutation_deferred(base_cfg: dict, tmp_path, monkeypatch) -> None:
    order: List[str] = []

    _install_fast_funnel_mocks(monkeypatch)

    def _wrap_member(self, cfg, load_cases, *, stage_name, tension_only=False):
        order.append("S5")
        return _fake_member_sizing_pass(self, cfg, load_cases, stage_name=stage_name)

    def _wrap_topology(self, cfg, load_cases, *, stage_name, tension_only=False):
        order.append("S6")
        return _fake_topology_cleanup(self, cfg, load_cases, stage_name=stage_name)

    monkeypatch.setattr(StagedFidelityFunnelPlanner, "_member_sizing_pass", _wrap_member)
    monkeypatch.setattr(StagedFidelityFunnelPlanner, "_topology_cleanup", _wrap_topology)

    result = _run_fast_funnel(base_cfg, tmp_path, monkeypatch, install_mocks=False)
    assert order
    assert "S5" in order and "S6" in order
    assert order.index("S5") < order.index("S6")
    assert len(result["s6_topology"]) >= 1


def test_final_report_pipeline_trace(base_cfg: dict, tmp_path, monkeypatch) -> None:
    optimization = _run_fast_funnel(base_cfg, tmp_path, monkeypatch)

    metrics = {
        "predicted_breaking_load_kgf": 92.0,
        "competition_mass_g": 856.0,
        "mass_limit_effective_g": 1000.0,
        "mass_compliant": True,
        "competition_mass_compliant": True,
        "solver_status": "regular",
        "equilibrium_ok": True,
        "min_fs_primary": 1.25,
        "min_fs_design": 1.25,
        "min_support_fs": 1.10,
        "min_glue_fs": 2.0,
        "member_sizing_plan": [],
        "installed_stick_mass_g": 810.0,
        "wet_glue_mass_g": 92.0,
        "cured_glue_mass_g": 46.0,
    }
    detailed = {
        "summary": {
            "installed_stick_mass_g": 810.0,
            "wet_glue_mass_g": 92.0,
            "cured_glue_mass_g": 46.0,
            "evaporated_glue_water_g": 46.0,
            "competition_mass_g": 856.0,
            "competition_mass_margin_g": 144.0,
            "purchased_blank_sticks_needed": 680,
            "purchased_stick_mass_g": 960.0,
            "cutting_scrap_mass_g": 150.0,
            "assembly_procurement_mass_g": 1052.0,
            "n_weak_glue_joints": 0,
            "extra_sticks_for_waste": 24,
            "estimated_total_sticks_with_waste": 680,
        }
    }

    report_out = ReportBundleService().generate(
        cfg=base_cfg,
        metrics=metrics,
        member_checks=[],
        detailed=detailed,
        optimization=optimization,
        warnings=[],
        out_dir=tmp_path / "final_report",
    )
    text = Path(report_out["index_md"]).read_text(encoding="utf-8")
    assert "Traço do pipeline S0..S8" in text
    assert "Melhores candidatos por estágio" in text
    assert "Comparação antes/depois da fase topológica" in text
    assert (tmp_path / "final_report" / "pipeline_stage_trace.csv").exists()
