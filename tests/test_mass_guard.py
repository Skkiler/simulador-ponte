from __future__ import annotations

from src.services.active_design_planner import ActiveDesignPlanner
from src.services.mass_guard import assert_mass_compliant, effective_mass_limit_g, resolve_mass_limits
from src.services.pipeline import SimulationPipeline


def test_resolve_mass_limits_and_effective_limit(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["planner"]["max_bridge_mass_g"] = 930.0
    cfg["material"]["mass_limit_g"] = 1000.0

    limits = resolve_mass_limits(cfg)
    assert limits["nominal_limit_g"] == 1000.0
    assert limits["planner_limit_g"] == 930.0
    assert limits["material_limit_g"] == 1000.0
    assert limits["effective_limit_g"] == 930.0
    assert effective_mass_limit_g(cfg) == 930.0


def test_assert_mass_compliant_annotates_limit_fields(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["planner"]["max_bridge_mass_g"] = 900.0
    row = {"mass_g": 901.0}
    out = assert_mass_compliant(row, cfg, source="unit_test")
    assert out["mass_compliant"] is False
    assert out["mass_limit_effective_g"] == 900.0
    assert out["mass_limit_nominal_g"] == 1000.0


def test_overweight_candidate_gets_hard_negative_score(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["planner"]["max_bridge_mass_g"] = 900.0
    metrics = {
        "min_fs_primary": 2.5,
        "estimated_breaking_load_kgf": 200.0,
        "mass_g": 950.0,
        "solver_status": "regular",
        "min_support_fs": 2.0,
        "equilibrium_error_N": 0.0,
        "inactive_support_count": 0,
        "weak_glue_joints": 0,
        "detailed": {"splice_stagger_report": {"critical_clusters": 0}},
    }
    score = ActiveDesignPlanner()._score_candidate(cfg, metrics)
    assert score < -1.0e8


def test_pipeline_selects_mass_compliant_fallback_candidate(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["material"]["mass_limit_g"] = 1000.0
    cfg["bridge"]["load_total_N"] = float(cfg["bridge"]["load_total_kgf"]) * 9.80665

    optimization = {
        "stage4": [
            {
                "candidate_id": "S4-0001",
                "mass_g": 1200.0,
                "solver_status": "regular",
                "equilibrium_error_N": 0.0,
                "predicted_breaking_load_kgf": 180.0,
                "min_fs_primary": 2.2,
                "score": 10.0,
                "config": {"id": "overweight"},
            }
        ],
        "stage3": [
            {
                "candidate_id": "S3-0001",
                "mass_g": 980.0,
                "solver_status": "singular_lstsq_rank_10_of_12",
                "equilibrium_error_N": 0.0,
                "predicted_breaking_load_kgf": 210.0,
                "min_fs_primary": 2.5,
                "score": 9.0,
                "config": {"id": "irregular"},
            }
        ],
        "stage2": [
            {
                "candidate_id": "S2-0001",
                "mass_g": 995.0,
                "solver_status": "regular",
                "equilibrium_error_N": 0.1,
                "predicted_breaking_load_kgf": 160.0,
                "min_fs_primary": 2.0,
                "score": 8.0,
                "config": {"id": "mass_ok"},
            }
        ],
        "stage1": [],
    }

    picked = SimulationPipeline._select_mass_compliant_candidate(optimization, cfg)
    assert picked is not None
    assert picked.get("candidate_id") == "S2-0001"
    assert picked.get("config", {}).get("id") == "mass_ok"


def test_pipeline_mass_compliant_fallback_returns_none_when_unavailable(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["material"]["mass_limit_g"] = 900.0
    cfg["bridge"]["load_total_N"] = float(cfg["bridge"]["load_total_kgf"]) * 9.80665

    optimization = {
        "stage4": [
            {
                "candidate_id": "S4-0001",
                "mass_g": 950.0,
                "solver_status": "regular",
                "equilibrium_error_N": 0.0,
                "predicted_breaking_load_kgf": 190.0,
                "min_fs_primary": 2.2,
                "score": 10.0,
                "config": {"id": "a"},
            }
        ],
        "stage3": [
            {
                "candidate_id": "S3-0001",
                "mass_g": 880.0,
                "solver_status": "regular",
                "equilibrium_error_N": 1.0e9,
                "predicted_breaking_load_kgf": 170.0,
                "min_fs_primary": 1.8,
                "score": 7.0,
                "config": {"id": "b"},
            }
        ],
        "stage2": [],
        "stage1": [],
    }

    picked = SimulationPipeline._select_mass_compliant_candidate(optimization, cfg)
    assert picked is None
