from __future__ import annotations

from copy import deepcopy

import pytest

from src.services.geometry_service import GeometryService


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
