from __future__ import annotations

import pytest

from src.domain.models import Node
from src.services.load_distribution_service import LoadDistributionService


def _deck_nodes() -> list[Node]:
    nodes: list[Node] = []
    nid = 1
    for x in (0.0, 100.0, 200.0):
        for y in (-50.0, 50.0):
            nodes.append(Node(nid, x, y, 25.0, "top", "left" if y < 0 else "right", x))
            nid += 1
    return nodes


def test_plate_footprint_uses_tributary_station_weights() -> None:
    cfg = {
        "bridge": {
            "span_mm": 200.0,
            "width_mm": 100.0,
            "load_application_level": "top",
            "load_distribution_model": "plate_surface_uniform",
            "load_distribution_x_mm": [100.0],
            "load_footprint_length_mm": 200.0,
        }
    }

    weights = LoadDistributionService.nodal_weights(cfg, _deck_nodes())

    by_station: dict[float, float] = {0.0: 0.0, 100.0: 0.0, 200.0: 0.0}
    for node in _deck_nodes():
        by_station[node.x] += weights.get(node.id, 0.0)

    assert sum(weights.values()) == pytest.approx(1.0)
    assert by_station[0.0] == pytest.approx(0.25)
    assert by_station[100.0] == pytest.approx(0.50)
    assert by_station[200.0] == pytest.approx(0.25)


def test_torsion_bias_preserves_total_vertical_load() -> None:
    cfg = {
        "bridge": {
            "span_mm": 200.0,
            "width_mm": 100.0,
            "load_application_level": "top",
            "load_distribution_model": "point_stations",
            "load_distribution_x_mm": [100.0],
            "load_footprint_length_mm": 0.0,
        }
    }

    loads = LoadDistributionService.build_nodal_loads(
        cfg,
        _deck_nodes(),
        loadcase="torsion_60_40",
        total_N=100.0,
        side_bias={"left": 0.60, "right": 0.40},
    )

    assert sum(load.Fz for load in loads) == pytest.approx(-100.0)
    left = -sum(load.Fz for load in loads if load.node_id == 3)
    right = -sum(load.Fz for load in loads if load.node_id == 4)
    assert left == pytest.approx(60.0)
    assert right == pytest.approx(40.0)


def test_plate_centroid_interpretation_does_not_create_multiple_footprints() -> None:
    cfg = {
        "bridge": {
            "span_mm": 200.0,
            "width_mm": 100.0,
            "load_application_level": "top",
            "load_distribution_model": "plate_surface_uniform",
            "load_distribution_x_mm": [50.0, 100.0, 150.0],
            "load_footprint_length_mm": 100.0,
            "load_footprint_interpretation": "centroid",
        }
    }

    weights = LoadDistributionService.nodal_weights(cfg, _deck_nodes())

    by_station: dict[float, float] = {0.0: 0.0, 100.0: 0.0, 200.0: 0.0}
    for node in _deck_nodes():
        by_station[node.x] += weights.get(node.id, 0.0)

    assert sum(weights.values()) == pytest.approx(1.0)
    assert by_station[0.0] == pytest.approx(0.0)
    assert by_station[100.0] == pytest.approx(1.0)
    assert by_station[200.0] == pytest.approx(0.0)


def test_crown_contact_loads_highest_nodes_inside_footprint() -> None:
    nodes = [
        Node(n.id, n.x, n.y, 40.0 if n.x == 100.0 else 25.0, n.level, n.side, n.x_station)
        for n in _deck_nodes()
    ]
    cfg = {
        "bridge": {
            "span_mm": 200.0,
            "width_mm": 100.0,
            "load_application_level": "top",
            "load_distribution_model": "plate_surface_uniform",
            "load_distribution_x_mm": [100.0],
            "load_footprint_length_mm": 200.0,
            "load_footprint_interpretation": "centroid",
            "load_crown_contact_z_tolerance_mm": 0.5,
        }
    }

    loads = LoadDistributionService.build_crown_contact_loads(
        cfg,
        nodes,
        loadcase="crown_contact",
        total_N=100.0,
    )

    assert sum(load.Fz for load in loads) == pytest.approx(-100.0)
    assert {load.node_id for load in loads} == {3, 4}
    assert all(load.Fz == pytest.approx(-50.0) for load in loads)
