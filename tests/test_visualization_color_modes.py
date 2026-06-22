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


def test_prism_view_selects_visible_mesh_without_white_hitbox_artifacts_and_preserves_camera() -> None:
    rows = [
        {
            "stick_id": "M001-L01-P01", "member_id": 1, "member_group": "top_chord", "lane": 1, "piece_index": 1,
            "x0_mm": 0.0, "y0_mm": 0.0, "z0_mm": 0.0, "x1_mm": 100.0, "y1_mm": 0.0, "z1_mm": 0.0,
            "width_mm": 7.0, "thickness_mm": 1.5, "visual_width_mm": 7.0, "visual_thickness_mm": 1.5,
            "section_global_offset_x_mm": 0.0, "section_global_offset_y_mm": 0.0, "section_global_offset_z_mm": 0.0,
            "assembly_unit_key": "top_chord|M001|L01|P01", "node_connection_ok": True,
        }
    ]
    fig = VisualizationService().plotly_stick_pieces(rows, uirevision_key="camera-test")
    meshes = [trace for trace in fig.data if trace.type == "mesh3d"]
    selectors = [trace for trace in fig.data if str(trace.name).startswith("selecionar membro")]
    assert len(meshes) == 1
    assert meshes[0].customdata[0][0] == "M001-L01-P01"
    assert meshes[0].customdata[0][1] == "1"
    assert selectors == []
    assert fig.layout.paper_bgcolor == "#0e1117"
    assert fig.layout.scene.bgcolor == "#0e1117"
    assert fig.layout.uirevision == "camera-test"
    assert fig.layout.scene.uirevision == "camera-test"
    assert fig.layout.scene.dragmode == "turntable"


def test_single_html_contains_mounted_and_exploded_buttons() -> None:
    rows = [
        {
            "stick_id": "M001-L01-P01", "member_id": 1, "member_group": "top_chord", "lane": 1, "piece_index": 1,
            "x0_mm": 0.0, "y0_mm": 2.0, "z0_mm": 0.0, "x1_mm": 100.0, "y1_mm": 2.0, "z1_mm": 0.0,
            "width_mm": 7.0, "thickness_mm": 1.5, "visual_width_mm": 7.0, "visual_thickness_mm": 1.5,
            "section_global_offset_x_mm": 0.0, "section_global_offset_y_mm": 2.0, "section_global_offset_z_mm": 0.0,
            "assembly_unit_key": "top_chord|M001|L01|P01", "node_connection_ok": True,
        }
    ]
    fig = VisualizationService().plotly_stick_pieces_mounted_exploded(rows)
    labels = [button.label for button in fig.layout.updatemenus[0].buttons]
    assert labels == ["Montada", "Explodida"]
    assert any(trace.visible is False for trace in fig.data)


def test_prism_view_highlights_and_explodes_only_selected_member() -> None:
    rows = [
        {
            "stick_id": "M001-L01-P01", "member_id": 1, "member_group": "top_chord", "lane": 1, "piece_index": 1,
            "x0_mm": 0.0, "y0_mm": 2.0, "z0_mm": 0.0, "x1_mm": 100.0, "y1_mm": 2.0, "z1_mm": 0.0,
            "width_mm": 7.0, "thickness_mm": 1.5, "visual_width_mm": 7.0, "visual_thickness_mm": 1.5,
            "section_global_offset_x_mm": 0.0, "section_global_offset_y_mm": 2.0, "section_global_offset_z_mm": 0.0,
            "assembly_unit_key": "top_chord|M001|L01|P01", "node_connection_ok": True,
        },
        {
            "stick_id": "M002-L01-P01", "member_id": 2, "member_group": "vertical", "lane": 1, "piece_index": 1,
            "x0_mm": 20.0, "y0_mm": 0.0, "z0_mm": 0.0, "x1_mm": 20.0, "y1_mm": 0.0, "z1_mm": 100.0,
            "width_mm": 7.0, "thickness_mm": 1.5, "visual_width_mm": 7.0, "visual_thickness_mm": 1.5,
            "section_global_offset_x_mm": 0.0, "section_global_offset_y_mm": 3.0, "section_global_offset_z_mm": 0.0,
            "assembly_unit_key": "vertical|M002|L01|P01", "node_connection_ok": True,
        },
    ]
    svc = VisualizationService()
    baseline = svc.prepare_stick_piece_mesh_batches(rows)
    focused = svc.prepare_stick_piece_mesh_batches(rows, focused_member_id=1, focused_section_explode_scale=5.0)
    base_r1 = next(row for row in baseline["rows"] if row["member_id"] == 1)
    focus_r1 = next(row for row in focused["rows"] if row["member_id"] == 1)
    base_r2 = next(row for row in baseline["rows"] if row["member_id"] == 2)
    focus_r2 = next(row for row in focused["rows"] if row["member_id"] == 2)
    assert base_r1["stick_id"] == focus_r1["stick_id"]
    assert base_r2["stick_id"] == focus_r2["stick_id"]
    # The generated mesh of member 1 changes; member 2 remains in its mounted position.
    assert baseline["batches"]["top_chord"]["y"] != focused["batches"]["top_chord"]["y"]
    assert baseline["batches"]["vertical"]["y"] == focused["batches"]["vertical"]["y"]
    fig = svc.plotly_stick_pieces(rows, selected_member_id=1)
    assert any(str(trace.name).startswith("membro selecionado") for trace in fig.data)


def test_prism_member_batches_expose_ctrl_click_customdata_on_visible_mesh() -> None:
    rows = [
        {
            "stick_id": "M002-L01-P01", "member_id": 2, "member_group": "top_chord", "lane": 1, "piece_index": 1,
            "x0_mm": 0.0, "y0_mm": 0.0, "z0_mm": 0.0, "x1_mm": 100.0, "y1_mm": 0.0, "z1_mm": 0.0,
            "width_mm": 7.0, "thickness_mm": 1.5, "visual_width_mm": 7.0, "visual_thickness_mm": 1.5,
            "section_global_offset_x_mm": 0.0, "section_global_offset_y_mm": 0.0, "section_global_offset_z_mm": 0.0,
            "assembly_unit_key": "top_chord|M002|L01|P01", "node_connection_ok": True,
        },
        {
            "stick_id": "M003-L01-P01", "member_id": 3, "member_group": "top_chord", "lane": 1, "piece_index": 1,
            "x0_mm": 100.0, "y0_mm": 0.0, "z0_mm": 0.0, "x1_mm": 200.0, "y1_mm": 0.0, "z1_mm": 0.0,
            "width_mm": 7.0, "thickness_mm": 1.5, "visual_width_mm": 7.0, "visual_thickness_mm": 1.5,
            "section_global_offset_x_mm": 0.0, "section_global_offset_y_mm": 0.0, "section_global_offset_z_mm": 0.0,
            "assembly_unit_key": "top_chord|M003|L01|P01", "node_connection_ok": True,
        },
    ]
    fig = VisualizationService().plotly_stick_pieces(rows, batch_by="member")
    meshes = [trace for trace in fig.data if trace.type == "mesh3d"]
    assert len(meshes) == 2
    assert {trace.meta["member_id"] for trace in meshes} == {"2", "3"}
    assert meshes[0].customdata[0][1] in {"2", "3"}


def test_load_fs_view_uses_dark_background_and_turntable_navigation() -> None:
    nodes, members, supports, loads = _base_geometry()
    fig = VisualizationService().plotly_geometry(
        nodes, members, supports, loads, color_mode="risk", member_checks=[{"member_id": 1, "FS_design": 1.1}]
    )
    assert fig.layout.paper_bgcolor == "#0e1117"
    assert fig.layout.scene.bgcolor == "#0e1117"
    assert fig.layout.scene.dragmode == "turntable"


def test_prism_piece_divisions_use_high_contrast_light_edges() -> None:
    rows = [{
        "stick_id": "M001-L01-P01", "member_id": 1, "member_group": "top_chord", "lane": 1, "piece_index": 1,
        "x0_mm": 0.0, "y0_mm": 0.0, "z0_mm": 0.0, "x1_mm": 100.0, "y1_mm": 0.0, "z1_mm": 0.0,
        "width_mm": 7.0, "thickness_mm": 1.5, "visual_width_mm": 7.0, "visual_thickness_mm": 1.5,
        "assembly_unit_key": "top_chord|M001|L01|P01", "node_connection_ok": True,
    }]
    fig = VisualizationService().plotly_stick_pieces(rows, batch_by="member")
    edges = [trace for trace in fig.data if getattr(trace, "meta", None) and trace.meta.get("trace_kind") == "edge"]
    assert len(edges) == 1
    assert float(edges[0].line.width) >= 4.0
    assert "255,246,214" in str(edges[0].line.color).replace(" ", "")


def test_real_prism_view_uses_orthographic_true_scale_to_avoid_apparent_deformation() -> None:
    rows = [{
        "stick_id": "M036-L01-P01", "member_id": 36, "member_group": "vertical", "lane": 1, "piece_index": 1,
        "x0_mm": 600.0, "y0_mm": -60.5, "z0_mm": 0.0, "x1_mm": 600.0, "y1_mm": -60.5, "z1_mm": 272.5,
        "width_mm": 7.0, "thickness_mm": 1.5, "visual_width_mm": 7.0, "visual_thickness_mm": 1.5,
        "section_global_offset_x_mm": 0.0, "section_global_offset_y_mm": -3.0, "section_global_offset_z_mm": 0.0,
        "inspection_status": "verificado pelo membro — FS 1.472 — modo: beam_column_interaction",
        "assembly_unit_key": "vertical|M036|L01|P01", "node_connection_ok": True,
    }]
    fig = VisualizationService().plotly_stick_pieces(rows, batch_by="member")
    assert fig.layout.scene.aspectmode == "data"
    assert fig.layout.scene.camera.projection.type == "orthographic"
    mesh = next(trace for trace in fig.data if trace.type == "mesh3d")
    assert "Status: verificado pelo membro" in mesh.text[0]


def test_exploded_prism_is_rigid_translation_and_preserves_long_stick_axis_length() -> None:
    rows = [{
        "stick_id": "M036-L01-P01", "member_id": 36, "member_group": "vertical", "lane": 1, "piece_index": 1,
        "x0_mm": 600.0, "y0_mm": -60.5, "z0_mm": 0.0, "x1_mm": 600.0, "y1_mm": -60.5, "z1_mm": 272.5,
        "width_mm": 7.0, "thickness_mm": 1.5, "visual_width_mm": 7.0, "visual_thickness_mm": 1.5,
        "section_global_offset_x_mm": 0.0, "section_global_offset_y_mm": -3.0, "section_global_offset_z_mm": 0.0,
        "assembly_unit_key": "vertical|M036|L01|P01", "node_connection_ok": True,
    }]
    data = VisualizationService().prepare_stick_piece_mesh_batches(
        rows, batch_by="member", section_explode_scale=7.0, connection_offset_scale=0.0
    )
    assert data["visual_dimension_error_count"] == 0
    assert data["visual_max_axis_length_error_mm"] <= 1.0e-9
    assert data["visual_max_rigid_translation_error_mm"] <= 1.0e-9


def test_hover_payload_directly_includes_status_length_and_cut_information() -> None:
    rows = [{
        "stick_id": "M036-L01-P01", "member_id": 36, "member_group": "vertical", "lane": 1, "piece_index": 1,
        "x0_mm": 600.0, "y0_mm": -60.5, "z0_mm": 0.0, "x1_mm": 600.0, "y1_mm": -60.5, "z1_mm": 272.5,
        "shop_cut_length_mm": 115.0, "installed_length_mm": 112.0,
        "width_mm": 7.0, "thickness_mm": 1.5, "visual_width_mm": 7.0, "visual_thickness_mm": 1.5,
        "miter_cut_start_shop_reference_angle_deg": 90.0, "miter_cut_end_shop_reference_angle_deg": 115.0,
        "inspection_status": "verificado pelo membro — FS 1.472 — modo: beam_column_interaction",
        "assembly_unit_key": "vertical|M036|L01|P01", "node_connection_ok": True,
    }]
    fig = VisualizationService().plotly_stick_pieces(rows, batch_by="member")
    mesh = next(trace for trace in fig.data if trace.type == "mesh3d")
    payload = list(mesh.customdata[0])
    assert payload[0] == "M036-L01-P01"
    assert payload[4].startswith("verificado pelo membro")
    assert payload[5] == "115.00"
    assert payload[6] == "112.00"
    assert payload[10] == "90.0° / 115.0°"
    assert "%{customdata[4]}" in mesh.hovertemplate
    assert "Comprimento de corte" in mesh.hovertemplate


def test_piece_mesh_mode_aggregates_edges_per_member_to_keep_view_responsive() -> None:
    rows = [
        {"stick_id": "M085-L01-P01", "member_id": 85, "member_group": "top_chord", "lane": 1, "piece_index": 1,
         "x0_mm": 0.0, "y0_mm": 0.0, "z0_mm": 0.0, "x1_mm": 100.0, "y1_mm": 0.0, "z1_mm": 0.0,
         "width_mm": 7.0, "thickness_mm": 1.5, "visual_width_mm": 7.0, "visual_thickness_mm": 1.5,
         "assembly_unit_key": "a", "node_connection_ok": True},
        {"stick_id": "M085-L02-P01", "member_id": 85, "member_group": "top_chord", "lane": 2, "piece_index": 1,
         "x0_mm": 0.0, "y0_mm": 2.0, "z0_mm": 0.0, "x1_mm": 100.0, "y1_mm": 2.0, "z1_mm": 0.0,
         "width_mm": 7.0, "thickness_mm": 1.5, "visual_width_mm": 7.0, "visual_thickness_mm": 1.5,
         "assembly_unit_key": "b", "node_connection_ok": True},
    ]
    fig = VisualizationService().plotly_stick_pieces(rows, batch_by="piece")
    meshes = [trace for trace in fig.data if trace.type == "mesh3d"]
    edges = [trace for trace in fig.data if getattr(trace, "meta", None) and trace.meta.get("trace_kind") == "edge"]
    assert len(meshes) == 2
    assert len(edges) == 1
    assert edges[0].meta["member_id"] == "85"


def test_piece_mode_uses_literal_hover_per_prism_not_interpolated_empty_payload() -> None:
    rows = [{
        "stick_id": "M085-L01-P01", "member_id": 85, "member_group": "top_chord", "lane": 1, "piece_index": 1,
        "x0_mm": 0.0, "y0_mm": 0.0, "z0_mm": 0.0, "x1_mm": 100.0, "y1_mm": 0.0, "z1_mm": 0.0,
        "shop_cut_length_mm": 100.0, "installed_length_mm": 99.65,
        "width_mm": 7.0, "thickness_mm": 1.5, "visual_width_mm": 7.0, "visual_thickness_mm": 1.5,
        "inspection_status": "sem esforço axial relevante — detalhamento de montagem registrado",
        "assembly_unit_key": "m85", "node_connection_ok": True,
    }]
    fig = VisualizationService().plotly_stick_pieces(rows, batch_by="piece")
    mesh = next(trace for trace in fig.data if trace.type == "mesh3d")
    assert "M085-L01-P01" in mesh.hovertemplate
    assert "sem esforço axial relevante" in mesh.hovertemplate
    assert "%{text}" not in mesh.hovertemplate


def test_exploded_view_fans_longitudinal_segments_transversely_without_extending_axis() -> None:
    rows = [
        {"stick_id": "M036-L01-P01", "member_id": 36, "member_group": "vertical", "lane": 1, "piece_index": 1,
         "x0_mm": 600.0, "y0_mm": -60.5, "z0_mm": 0.0, "x1_mm": 600.0, "y1_mm": -60.5, "z1_mm": 120.0,
         "width_mm": 7.0, "thickness_mm": 1.5, "visual_width_mm": 7.0, "visual_thickness_mm": 1.5,
         "assembly_unit_key": "p1", "node_connection_ok": True},
        {"stick_id": "M036-L01-P02", "member_id": 36, "member_group": "vertical", "lane": 1, "piece_index": 2,
         "x0_mm": 600.0, "y0_mm": -60.5, "z0_mm": 100.0, "x1_mm": 600.0, "y1_mm": -60.5, "z1_mm": 210.0,
         "width_mm": 7.0, "thickness_mm": 1.5, "visual_width_mm": 7.0, "visual_thickness_mm": 1.5,
         "assembly_unit_key": "p2", "node_connection_ok": True},
    ]
    data = VisualizationService().prepare_stick_piece_mesh_batches(
        rows, batch_by="piece", section_explode_scale=7.0, longitudinal_piece_explode_gap_mm=14.0
    )
    b1 = data["batches"]["M036-L01-P01"]
    b2 = data["batches"]["M036-L01-P02"]
    # A extensão axial z permanece a original; a abertura ocorre lateralmente.
    assert min(b2["z"]) < max(b1["z"])
    assert abs((max(b2["z"]) - min(b1["z"])) - 210.0) <= 2.0
    assert abs(sum(b2["x"]) / len(b2["x"]) - sum(b1["x"]) / len(b1["x"])) >= 13.5
    assert data["visual_longitudinal_explosion_strategy"] == "transverse_segment_fan_rigid_translation"
    assert data["visual_max_segment_axial_translation_mm"] <= 1.0e-9
    assert data["visual_segment_axial_translation_error_count"] == 0
    assert data["visual_dimension_error_count"] == 0


def test_piece_mesh_interactive_payload_is_not_repeated_per_vertex() -> None:
    rows = [{
        "stick_id": "M085-L01-P01", "member_id": 85, "member_group": "top_chord", "lane": 1, "piece_index": 1,
        "x0_mm": 0.0, "y0_mm": 0.0, "z0_mm": 0.0, "x1_mm": 100.0, "y1_mm": 0.0, "z1_mm": 0.0,
        "shop_cut_length_mm": 100.0, "installed_length_mm": 99.65,
        "width_mm": 7.0, "thickness_mm": 1.5, "visual_width_mm": 7.0, "visual_thickness_mm": 1.5,
        "inspection_status": "verificado pelo membro — FS 1.483 — modo: beam_column_interaction",
        "assembly_unit_key": "m85", "node_connection_ok": True,
    }]
    fig = VisualizationService().plotly_stick_pieces(rows, batch_by="piece")
    mesh = next(trace for trace in fig.data if trace.type == "mesh3d")
    assert not list(mesh.customdata or [])
    assert not list(mesh.text or [])
    assert "M085-L01-P01" in mesh.hovertemplate
    assert mesh.meta["stick_id"] == "M085-L01-P01"
    assert mesh.color
    # Regressão C17: ``facecolor=[]`` oculta as faces do Mesh3d e deixa
    # apenas contornos/textos residuais no visor.
    assert mesh.facecolor is None
