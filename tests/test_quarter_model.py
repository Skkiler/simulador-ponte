from __future__ import annotations

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

