from __future__ import annotations

from src.services.piece_inspection_service import PieceInspectionService


def _pieces():
    return [
        {
            "stick_id": "M044-L01-P01", "member_id": 44, "member_group": "diagonal", "lane": 1, "piece_index": 1,
            "shop_cut_length_mm": 72.0, "installed_length_mm": 72.0, "mass_g": 0.84, "n_sticks": 1,
            "miter_cut_start_required": True, "miter_cut_start_angle_deg": 45.0, "miter_cut_start_host_group": "vertical",
            "miter_cut_end_required": False, "miter_cut_end_angle_deg": 90.0,
        },
        {
            "stick_id": "M044-L01-P02", "member_id": 44, "member_group": "diagonal", "lane": 1, "piece_index": 2,
            "shop_cut_length_mm": 70.0, "installed_length_mm": 70.0, "mass_g": 0.82, "n_sticks": 1,
        },
        {
            "stick_id": "TALA-J-M044-L01-P01-02-S01", "member_id": 44, "member_group": "diagonal_splint", "lane": 1, "piece_index": 1,
            "shop_cut_length_mm": 40.0, "installed_length_mm": 40.0, "mass_g": 0.47, "n_sticks": 1,
        },
    ]


def _joints():
    return [
        {
            "joint_id": "J-M044-L01-P01-02", "member_id": 44, "member_group": "diagonal", "piece_a": "M044-L01-P01",
            "piece_b": "M044-L01-P02", "joint_type": "butt_with_splints", "joint_model": "single_lap_tala",
            "overlap_length_mm": 40.0, "physical_glue_area_mm2": 280.0, "FS_glue_shear": 2.1, "splints_per_splice": 1,
        }
    ]


def test_selection_state_extracts_physical_stick_id() -> None:
    state = {"selection": {"points": [{"customdata": ["M044-L01-P01", "44", "diagonal", "unit"]}]}}
    assert PieceInspectionService.selected_stick_id(state) == "M044-L01-P01"


def test_piece_card_lists_overlap_cut_and_connected_splint() -> None:
    info = PieceInspectionService.inspect("M044-L01-P01", _pieces(), _joints(), [{"member_id": 44, "FS_design": 1.6, "governing_mode": "tension"}])
    assert info is not None
    assert info.nominal_overlap_total_mm == 40.0
    assert info.cuts[0]["ângulo_deg"] == 45.0
    related = {row["stick_id"] for row in info.related_pieces}
    assert "TALA-J-M044-L01-P01-02-S01" in related


def test_selecting_splint_resolves_parent_joint() -> None:
    info = PieceInspectionService.inspect("TALA-J-M044-L01-P01-02-S01", _pieces(), _joints())
    assert info is not None
    assert info.nominal_overlap_total_mm == 40.0
    assert any(row["stick_id"] == "M044-L01-P01" for row in info.related_pieces)


def test_member_wide_sandwich_bond_is_reported_for_each_lamina() -> None:
    pieces = [{
        "stick_id": "M002-L01-P01", "member_id": 2, "member_group": "top_chord", "lane": 1, "piece_index": 1,
        "shop_cut_length_mm": 100.0, "installed_length_mm": 99.0, "mass_g": 1.1, "n_sticks": 8,
    }]
    joints = [{
        "joint_id": "SANDWICH-CONT-2", "member_id": 2, "member_group": "top_chord", "lane": "all",
        "piece_a": "nucleo_capas", "piece_b": "faces_continuas", "joint_type": "continuous_face_lamination",
        "joint_model": "sandwich_continuous_face_bond_mass_accounting", "overlap_length_mm": 100.0,
        "physical_glue_area_mm2": 3300.0,
    }]
    info = PieceInspectionService.inspect("M002-L01-P01", pieces, joints)
    assert info is not None
    assert info.nominal_overlap_total_mm == 100.0
    assert info.overlaps[0]["escopo"] == "todo o membro"


def test_unchecked_member_never_displays_false_zero_fs() -> None:
    info = PieceInspectionService.inspect(
        "M044-L01-P01", _pieces(), _joints(), [{"member_id": 44, "FS_design": "", "FS_min": "", "governing_mode": "unchecked"}]
    )
    summary = {row["item"]: row["valor"] for row in PieceInspectionService.summary_rows(info)}
    assert summary["FS do membro"] == "—"


def test_selection_state_extracts_complete_member_id_from_any_lamina() -> None:
    state = {"selection": {"points": [{"customdata": ["M044-L01-P02", "44", "diagonal", "unit"]}]}}
    assert PieceInspectionService.selected_member_id(state) == "44"


def test_member_inspection_aggregates_structural_pieces_splint_cuts_and_strings() -> None:
    info = PieceInspectionService.inspect_member(
        44,
        _pieces(),
        _joints(),
        [{"member_id": 44, "FS_design": 1.6, "governing_mode": "tension"}],
        clicked_stick_id="M044-L01-P01",
    )
    assert info is not None
    assert info.member_group == "diagonal"
    assert len(info.structural_pieces) == 2
    assert len(info.auxiliary_pieces) == 1
    assert info.nominal_overlap_total_mm == 40.0
    assert any(row["ângulo_interno_CAD_deg"] == "45.00" for row in info.cuts)
    summary = PieceInspectionService.member_summary_rows(info)
    assert all(isinstance(row["valor"], str) for row in summary)
    piece_rows = PieceInspectionService.member_piece_rows(info)
    assert {row["tipo"] for row in piece_rows} == {"estrutura", "tala"}
    assert all(row["status"].strip() for row in piece_rows)
    assert any("FS 1.600" in row["status"] for row in piece_rows if row["tipo"] == "estrutura")
    assert any("FS cola 2.100" in row["status"] for row in piece_rows if row["tipo"] == "tala")


def test_generated_interactive_script_tracks_ctrl_on_pointerdown() -> None:
    source = __import__("pathlib").Path("src/services/pipeline.py").read_text(encoding="utf-8")
    assert "pointerHadModifier" in source
    assert "chart.addEventListener('pointerdown'" in source
    assert "ctrlHeld || pointerHadModifier" in source


def test_app_uses_current_streamlit_iframe_api_without_legacy_scrolling_argument() -> None:
    source = __import__("pathlib").Path("app.py").read_text(encoding="utf-8")
    assert 'native_iframe = getattr(st, "iframe", None)' in source
    assert 'native_iframe(interactive_path, height=1620, tab_index=0)' in source
    assert 'native_iframe(interactive_path, height=1620, tab_index=0, scrolling=False)' not in source
    assert 'components.html(interactive_html, height=1620, scrolling=False)' in source


def test_interactive_script_uses_incremental_render_and_non_overlapping_detail_panel() -> None:
    source = __import__("pathlib").Path("src/services/pipeline.py").read_text(encoding="utf-8")
    assert "window.__ASSEMBLY_INCREMENTAL_RENDER__ = true" in source
    assert "window.requestAnimationFrame(() =>" in source
    assert "updateSerial = updateSerial.then(() => updateVisualState())" in source
    assert "restyleChanged(visible, opacity)" in source
    assert "coords.map(c => c.x)" not in source
    assert "host.style.flexDirection = 'column'" in source
    assert "max-height:620px" in source
    assert "side.scrollIntoView" not in source
    assert "gridTemplateColumns" not in source


def test_3d_navigation_is_turntable_without_orbit_or_pan_controls() -> None:
    app = __import__("pathlib").Path("app.py").read_text(encoding="utf-8")
    viz = __import__("pathlib").Path("src/services/visualization_service.py").read_text(encoding="utf-8")
    pipeline = __import__("pathlib").Path("src/services/pipeline.py").read_text(encoding="utf-8")
    assert '"dragmode": "turntable"' in viz
    assert 'scene["dragmode"] = "turntable"' in viz
    assert '["orbitRotation", "pan3d", "zoom3d"]' in app
    assert '["orbitRotation", "pan3d", "zoom3d"]' in pipeline


def test_interactive_script_combines_isolate_and_explode_modes() -> None:
    source = __import__("pathlib").Path("src/services/pipeline.py").read_text(encoding="utf-8")
    assert "let isolateMode = false" in source
    assert "let explodeMode = false" in source
    assert "isolateMode && explodeMode ? 'Membro isolado e explodido — inspeção local'" in source
    assert "isolateMode=!isolateMode" in source
    assert "explodeMode=!explodeMode" in source


def test_near_zero_force_member_receives_explicit_non_governing_status() -> None:
    info = PieceInspectionService.inspect_member(
        44, _pieces(), _joints(), [{"member_id": 44, "N_member_N": 1.0e-12, "FS_min_global": 1.0e14, "governing_mode_global": "tension_capacity"}]
    )
    assert info is not None
    summary = {row["item"]: row["valor"] for row in PieceInspectionService.member_summary_rows(info)}
    assert summary["Status"] == "sem esforço axial relevante — detalhamento de montagem registrado"
    assert summary["FS do membro"] == "não governante (N≈0)"
    assert all(row["status"].strip() for row in PieceInspectionService.member_piece_rows(info))


def test_pipeline_propagates_status_to_prism_hover_and_writes_visual_audits() -> None:
    source = __import__("pathlib").Path("src/services/pipeline.py").read_text(encoding="utf-8")
    viz = __import__("pathlib").Path("src/services/visualization_service.py").read_text(encoding="utf-8")
    assert 'display_piece["inspection_status"]' in source
    assert 'piece_status_audit.json' in source
    assert 'visualization_dimension_audit.json' in source
    assert 'Status: {str(r.get(\'inspection_status\')' in viz


def test_exploded_member_does_not_apply_per_connection_displacements() -> None:
    source = __import__("pathlib").Path("src/services/pipeline.py").read_text(encoding="utf-8")
    assert 'exploded_view_batches = self.viz.prepare_stick_piece_mesh_batches' in source
    assert 'connection_offset_scale=mounted_scale,\n                        section_explode_scale=7.0' in source
    assert "projection:{type:'orthographic'}" in source


def test_interactive_script_refocuses_selected_member_after_incremental_update() -> None:
    source = __import__("pathlib").Path("src/services/pipeline.py").read_text(encoding="utf-8")
    assert "'scene.xaxis.autorange': false" in source
    assert "if (selectedMember && localInspection) return focusMember(selectedMember, useExploded, true)" in source
    assert "'scene.aspectmode': 'cube'" in source
    assert "button('Enquadrar alvo', () => { if (selectedMember) { highlightMode=true; isolateMode=true; scheduleVisualState(); } }" in source
    assert "window.__ASSEMBLY_LOCAL_INSPECTION_FOCUS__ = true" in source
    assert "Mantém contexto atenuado e enquadra o membro" in source
    assert "Vista local centralizada" in source


def test_interactive_scene_uses_one_visible_mesh_per_physical_piece_for_reliable_hover() -> None:
    source = __import__("pathlib").Path("src/services/pipeline.py").read_text(encoding="utf-8")
    assert 'batch_by="piece"' in source
    assert "window.__ASSEMBLY_PIECE_TRACE_HOVER__ = true" in source


def test_isolated_member_is_rigidly_translated_to_local_inspection_origin() -> None:
    source = __import__("pathlib").Path("src/services/pipeline.py").read_text(encoding="utf-8")
    assert "function centeredCoords(moveIndices, boundsIndices, coords)" in source
    assert "Number(v) - bounds.center.x" in source
    assert "const localInspection = !!selectedMember && (isolateMode || explodeMode)" in source


def test_focus_bounds_ignore_edge_null_separators_and_use_mesh_geometry_only() -> None:
    source = __import__("pathlib").Path("src/services/pipeline.py").read_text(encoding="utf-8")
    assert "const memberMeshIndices = {}" in source
    assert "function finiteCoord(value)" in source
    assert "value !== null && value !== undefined" in source
    assert "memberBounds(meshIndices, coords)" in source
    assert "centeredCoords(targetIndices, targetMeshIndices, source)" in source
    assert "window.__ASSEMBLY_NULL_SEPARATOR_BOUNDS_FIX__ = true" in source


def test_highlight_restores_global_view_after_local_inspection() -> None:
    source = __import__("pathlib").Path("src/services/pipeline.py").read_text(encoding="utf-8")
    assert "function restoreAssemblyViewport(force=false)" in source
    assert "if (!localInspection) return restoreAssemblyViewport()" in source
    assert "window.__ASSEMBLY_RESTORE_GLOBAL_VIEW_ON_HIGHLIGHT__ = true" in source


def test_interactive_hover_has_persistent_piece_status_card() -> None:
    source = __import__("pathlib").Path("src/services/pipeline.py").read_text(encoding="utf-8")
    assert "assembly-hover-piece-status" in source
    assert "chart.on('plotly_hover'" in source
    assert "pieceLookup" in source
    assert "window.__ASSEMBLY_HOVER_STATUS_CARD__ = true" in source


def test_pipeline_exports_literal_tooltip_and_longitudinal_explosion_flags() -> None:
    source = __import__("pathlib").Path("src/services/pipeline.py").read_text(encoding="utf-8")
    assert "longitudinal_piece_explode_gap_mm=14.0" in source
    assert "window.__ASSEMBLY_LITERAL_PIECE_TOOLTIP__ = true" in source
    assert "window.__ASSEMBLY_TRANSVERSE_SEGMENT_FAN__ = true" in source
    assert "window.__ASSEMBLY_HOVER_FROM_PRISM_TRACE_META__ = true" in source
    assert "window.__ASSEMBLY_ISOTROPIC_LOCAL_FOCUS__ = true" in source
    assert "detalhado para montagem — sem FS isolado aplicável" in source


def test_interactive_renderer_uses_diff_updates_and_avoids_second_plotly_figure() -> None:
    source = __import__("pathlib").Path("src/services/pipeline.py").read_text(encoding="utf-8")
    assert "function restyleChanged(nextVisible, nextOpacity)" in source
    assert "updateSerial = updateSerial.then(() => updateVisualState())" in source
    assert "exploded_edges_by_member" in source
    assert "exploded_fig = self.viz.plotly_stick_pieces" not in source


def test_cut_quantity_rows_groups_physical_cuts_by_shop_dimension() -> None:
    rows = PieceInspectionService.cut_quantity_rows(_pieces() + [{
        "stick_id": "M044-L02-P01", "member_id": 44, "member_group": "diagonal",
        "shop_cut_length_mm": 72.0, "installed_length_mm": 72.0,
    }])
    by_length = {row["tamanho do palito"]: row["quantidade"] for row in rows}
    assert by_length["72.00 mm"] == "2"
    assert by_length["70.00 mm"] == "1"
    assert by_length["40.00 mm"] == "1"


def test_bridge_overview_contains_global_metrics_and_cut_table_data() -> None:
    cfg = {
        "bridge": {"span_mm": 1200.0, "left_support_overhang_mm": 100.0, "right_support_overhang_mm": 100.0, "width_mm": 115.0, "center_height_mm": 272.5},
        "material": {"stick_width_mm": 7.0, "stick_thickness_mm": 1.5},
        "analysis": {"target_min_fs": 1.5},
    }
    rows = PieceInspectionService.bridge_summary_rows(
        cfg,
        _pieces(),
        {"competition_mass_g": 951.397, "competition_mass_margin_g": 48.603, "estimated_total_sticks_with_waste": 680, "installed_stick_mass_g": 930.0, "cured_glue_mass_g": 21.397, "wet_glue_mass_g": 112.431, "wet_glue_budget_g": 100.0, "wet_glue_budget_margin_g": -12.431},
        {"predicted_breaking_load_design_kgf": 118.607, "min_fs_member_design": 1.483, "min_fs_support": 1.650, "min_fs_glue": 1.618},
        {"interpenetration_count": 0, "node_connection_gap_piece_count": 0},
        "regular|tension_only_converged",
    )
    summary = {row["item"]: row["valor"] for row in rows}
    assert summary["Comprimento total entre extremos de apoio"] == "1400.00 mm"
    assert summary["Peso total estimado para competição"] == "951.397 g"
    assert summary["Peso de ruptura estimado"] == "118.607 kgf"
    assert summary["Interpenetrações / lacunas auditadas"] == "0 / 0"
    assert "FS membro 1.483 < alvo 1.500" in summary["Status da ponte"]
    assert "cola úmida excede orçamento" in summary["Status da ponte"]
    assert summary["Cola úmida prevista / orçamento"] == "112.431 / 100.000 g"


def test_interactive_view_uses_bridge_status_and_cut_tables_without_selection() -> None:
    source = __import__("pathlib").Path("src/services/pipeline.py").read_text(encoding="utf-8")
    assert "bridge_inspection_overview.json" in source
    assert "bridge_cut_quantity_table.csv" in source
    assert "member_cut_quantity_table.csv" in source
    assert "const bridgeRecord = __BRIDGE_INSPECTION__" in source
    assert "Status da ponte total" in source
    assert "Tabela de cortes da ponte total" in source
    assert "Tabela de cortes do membro" in source
    assert "const label = !selectedMember || !hasSelectionMode" in source
    assert "Ponte completa — selecione um membro" in source


def test_terminal_joint_area_is_capped_to_explicit_physical_face() -> None:
    config = __import__("pathlib").Path("bridge_config.json").read_text(encoding="utf-8")
    service = __import__("pathlib").Path("src/services/stick_detail_service.py").read_text(encoding="utf-8")
    assert '"terminal_joint_area_factor": 1.0' in config
    assert '"terminal_joint_area_factor_physical_cap": 1.0' in config
    assert "terminal_joint_area_factor = min(terminal_joint_area_factor_requested, terminal_joint_area_factor_physical_cap)" in service


def test_bridge_overview_reports_active_contact_supports() -> None:
    rows = PieceInspectionService.bridge_summary_rows(
        {"bridge": {"span_mm": 1200.0, "load_total_kgf": 80.0}, "material": {}, "analysis": {}},
        [], {}, {}, {}, "regular",
        [
            {"x_mm": -100.0, "support_active_vertical": False},
            {"x_mm": 0.0, "support_active_vertical": True},
            {"x_mm": 1200.0, "support_active_vertical": True},
        ],
    )
    summary = {row["item"]: row["valor"] for row in rows}
    assert summary["Carga de referência analisada"] == "80.000 kgf"
    assert "x=0.00 mm" in summary["Apoios ativos no caso analisado"]
    assert "x=-100.00 mm" in summary["Apoios sem contato no caso analisado"]
