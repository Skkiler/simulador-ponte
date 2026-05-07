from __future__ import annotations

from copy import deepcopy

from src.services.active_design_planner import ActiveDesignPlanner
from src.services.geometry_service import GeometryService
from src.services.quarter_model_service import QuarterModelService


def test_quarter_model_build_and_replicate_geometry(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["analysis"]["enforce_symmetry"] = True
    cfg["analysis"]["use_quarter_model"] = True

    geom = GeometryService()
    nodes, members, supports, loads = geom.generate(cfg)
    svc = QuarterModelService()

    val = svc.validate_quarter_symmetry(cfg, nodes, members, supports, loads)
    assert val["is_valid"], val

    qm = svc.build_quarter_model(cfg, nodes, members, supports, loads)
    assert len(qm.nodes) > 0
    assert len(qm.members) > 0
    assert len(qm.members) < len(members)

    rep = svc.replicate_quarter_geometry(cfg, qm.nodes, qm.members, qm.supports, qm.loads)
    assert len(rep["nodes"]) >= len(nodes)
    assert len(rep["members"]) >= len(members) * 0.75

    # Mapeamentos de simetria com multiplicidade esperada 1/2/4.
    for qmid, mids in svc._quarter_to_full_members.items():
        assert qmid > 0
        assert len(mids) in {1, 2, 4}
        for mid in mids:
            assert svc.map_full_member_to_quarter_member(mid) == qmid

    # Conservação da carga vertical total após replicação.
    original_fz = sum(float(l.Fz) for l in loads)
    replicated_fz = sum(float(l.Fz) for l in rep["loads"])
    assert abs(original_fz - replicated_fz) < 1.0e-6


def test_planner_uses_quarter_model_when_enabled(base_cfg: dict) -> None:
    cfg = deepcopy(base_cfg)
    cfg["analysis"]["use_quarter_model"] = True
    cfg["analysis"]["enforce_symmetry"] = True

    planner = ActiveDesignPlanner()
    base = planner._solve_and_check_base(cfg)
    if bool(base.get("quarter_model_used")):
        assert int(base.get("quarter_member_count", 0)) > 0
        assert isinstance(base.get("quarter_member_map", {}), dict)
    else:
        # Se o solver do quarto-modelo ficar singular/fora de equilíbrio,
        # o planejador deve cair para o modelo completo de forma segura.
        reason = str(base.get("quarter_model_fallback_reason") or "")
        assert reason.startswith("quarter_model_solution_invalid:")


def test_planner_records_quarter_fallback_reason(base_cfg: dict, monkeypatch) -> None:
    cfg = deepcopy(base_cfg)
    cfg["analysis"]["use_quarter_model"] = True

    planner = ActiveDesignPlanner()

    def _fake_validate(*args, **kwargs):
        return {"is_valid": False, "reasons": ["forced_invalid"], "warnings": []}

    monkeypatch.setattr(planner.quarter_model, "validate_quarter_symmetry", _fake_validate)
    base = planner._solve_and_check_base(cfg)
    assert bool(base.get("quarter_model_used")) is False
    assert str(base.get("quarter_model_fallback_reason")) == "symmetry_validation_failed"
