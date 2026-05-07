from __future__ import annotations

from copy import deepcopy

import pytest

from src.services.active_design_planner import ActiveDesignPlanner
from src.services.geometry_service import GeometryService
from src.services.topology_validator import TopologyValidator


def _incident_groups_by_node(members) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for m in members:
        out.setdefault(int(m.i), set()).add(str(m.group))
        out.setdefault(int(m.j), set()).add(str(m.group))
    return out


@pytest.mark.parametrize("mode", ["Warren", "Warren_symmetric"])
def test_warren_endpoints_have_diagonal_connection(base_cfg: dict, mode: str) -> None:
    cfg = deepcopy(base_cfg)
    cfg["bridge"]["side_truss_type"] = mode
    cfg["bridge"]["truss_type"] = mode

    nodes, members, _, _ = GeometryService().generate(cfg)
    by_node = _incident_groups_by_node(members)

    span = float(cfg["bridge"]["span_mm"])
    y_values = sorted(
        {
            float(n.y)
            for n in nodes
            if abs(float(n.x)) <= 1.0e-6 and n.level == "bottom"
        }
    )

    node_id_by_key = {
        (round(float(n.x), 6), round(float(n.y), 6), str(n.level)): int(n.id)
        for n in nodes
    }

    for x in (0.0, span):
        for y in y_values:
            for level in ("bottom", "top"):
                node_id = node_id_by_key[(round(x, 6), round(y, 6), level)]
                assert "diagonal" in by_node.get(node_id, set()), (
                    f"{mode}: nó de ponta sem diagonal em x={x}, y={y}, level={level}"
                )


@pytest.mark.parametrize("mode", ["Warren", "Warren_symmetric"])
def test_warren_has_no_floating_nodes_or_open_panels(base_cfg: dict, mode: str) -> None:
    cfg = deepcopy(base_cfg)
    cfg["bridge"]["side_truss_type"] = mode
    cfg["bridge"]["truss_type"] = mode

    nodes, members, supports, loads = GeometryService().generate(cfg)
    topo = TopologyValidator().validate(cfg, nodes, members, supports, loads)
    assert topo["is_valid"], topo
    assert not any(str(e).startswith("warren_open_panel") for e in topo.get("errors", []))
    assert not any(str(e).startswith("floating_or_weak_node") for e in topo.get("errors", []))
    assert not any(str(e).startswith("transverse_not_attached") for e in topo.get("errors", []))


def test_warren_invalid_topology_is_discarded_in_prefilter(base_cfg: dict, monkeypatch) -> None:
    planner = ActiveDesignPlanner()
    cfg = deepcopy(base_cfg)
    cfg["analysis"]["enforce_symmetry"] = True
    cfg["analysis"]["planner_prefilter_topology_check"] = True

    candidate = {
        "span_mm": float(cfg["bridge"]["span_mm"]),
        "width_mm": float(cfg["bridge"]["width_mm"]),
        "center_height_mm": 550.0,
        "panel_mm": float(cfg["bridge"]["panel_mm"]),
        "side_truss_type": "Warren_symmetric",
        "top_profile": str(cfg["bridge"]["top_profile"]),
        "internal_truss_type": str(cfg["bridge"]["internal_truss_type"]),
        "top_chord_truss_type": "X",
        "bottom_chord_truss_type": "X",
        "chord_truss_type": "none",
        "reinforcement_profile": "balanced",
        "tension_joint_model": "double_lap_reinforced",
        "compression_joint_model": "double_lap_reinforced",
        "splice_mode": "overlap",
        "overlap_length_mm": float(cfg.get("detail_model", {}).get("overlap_length_mm", 30.0)),
    }

    def _fake_validate(*args, **kwargs):
        return {"is_valid": False, "errors": ["warren_open_panel:x0=0,x1=100,y=-80"], "warnings": []}

    monkeypatch.setattr(planner.topology, "validate", _fake_validate)
    ok, reason = planner._prefilter_candidate(cfg, candidate)
    assert ok is False
    assert "PF12_topologia_invalida" in reason
