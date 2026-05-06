from __future__ import annotations

from src.services.active_design_planner import ActiveDesignPlanner


def test_s0_candidates_respect_symmetry_when_enforced(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["analysis"]["enforce_symmetry"] = True

    planner = ActiveDesignPlanner()
    candidates = planner._build_stage1_candidates(cfg, 36)

    assert candidates
    for cand in candidates:
        ok, reason = planner._is_symmetry_compliant_candidate(cfg, cand)
        assert ok, reason
        assert cand["top_chord_truss_type"] == cand["bottom_chord_truss_type"]

        span = float(cand["span_mm"])
        panel = float(cand["panel_mm"])
        n_panels = round(span / panel)
        assert int(n_panels) % 2 == 0

