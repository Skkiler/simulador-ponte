from __future__ import annotations

from src.domain.models import Load, Member, Node, Support
from src.services.visualization_service import VisualizationService


def _base_geometry():
    nodes = [
        Node(1, 0.0, -50.0, 0.0, "bottom", "L", 0.0),
        Node(2, 100.0, -50.0, 0.0, "bottom", "L", 100.0),
        Node(3, 0.0, 50.0, 0.0, "bottom", "R", 0.0),
        Node(4, 100.0, 50.0, 0.0, "bottom", "R", 100.0),
    ]
    members = [
        Member(1, 1, 2, "bottom_chord", 2, 10.0, 10.0, 10.0, 20.0, 20.0, 1.0, 6000.0, 500.0, 1.0, 1.0, 100.0),
        Member(2, 3, 4, "bottom_chord", 2, 10.0, 10.0, 10.0, 20.0, 20.0, 1.0, 6000.0, 500.0, 1.0, 1.0, 100.0),
        Member(3, 1, 4, "diagonal", 2, 10.0, 10.0, 10.0, 20.0, 20.0, 1.0, 6000.0, 500.0, 1.0, 1.0, 120.0),
    ]
    supports = [Support(1, 1, 1, 1, 0, 0, 0, "left", True)]
    loads = [Load("LC1", 4, 0.0, 0.0, -100.0)]
    return nodes, members, supports, loads


def test_plotly_geometry_force_mode_colors_all_members() -> None:
    nodes, members, supports, loads = _base_geometry()
    member_results = [
        {"member_id": 1, "N_N": 300.0},
        {"member_id": 2, "N_N": -450.0},
        {"member_id": 3, "N_N": 0.0},
    ]

    fig = VisualizationService().plotly_geometry(
        nodes,
        members,
        supports,
        loads,
        color_mode="force",
        member_results=member_results,
    )

    member_traces = [t for t in fig.data if str(t.name).startswith("membro_")]
    assert len(member_traces) == len(members)
    colors = [str(t.line.color) for t in member_traces]
    assert len(set(colors)) >= 2


def test_plotly_geometry_selection_only_highlights_without_recoloring() -> None:
    nodes, members, supports, loads = _base_geometry()
    member_results = [
        {"member_id": 1, "N_N": 300.0},
        {"member_id": 2, "N_N": -450.0},
        {"member_id": 3, "N_N": 0.0},
    ]

    svc = VisualizationService()
    fig_base = svc.plotly_geometry(
        nodes,
        members,
        supports,
        loads,
        color_mode="force",
        member_results=member_results,
    )
    fig_sel = svc.plotly_geometry(
        nodes,
        members,
        supports,
        loads,
        color_mode="force",
        member_results=member_results,
        selected_member_ids=[1, 3],
        highlight_selected=True,
    )

    base_colors = [str(t.line.color) for t in fig_base.data if str(t.name).startswith("membro_")]
    sel_colors = [str(t.line.color) for t in fig_sel.data if str(t.name).startswith("membro_")]
    assert base_colors == sel_colors
    assert any(str(t.name).startswith("selecionado") for t in fig_sel.data)
