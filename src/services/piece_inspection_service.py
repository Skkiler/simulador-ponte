from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from src.core.numeric import safe_float


def _text(value: Any, default: str = "—") -> str:
    if value is None or str(value).strip() == "":
        return default
    return str(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "sim"}


@dataclass(frozen=True)
class PieceInspection:
    """Dados auditáveis de um palito físico selecionado no CAD 3D."""

    stick_id: str
    row: Dict[str, Any]
    member_pieces: List[Dict[str, Any]]
    related_pieces: List[Dict[str, Any]]
    joints: List[Dict[str, Any]]
    overlaps: List[Dict[str, Any]]
    cuts: List[Dict[str, Any]]
    member_check: Dict[str, Any] | None
    nominal_overlap_total_mm: float


@dataclass(frozen=True)
class MemberInspection:
    """Dados de montagem do membro estrutural selecionado no CAD 3D.

    Um membro pode conter várias lâminas, segmentos e talas. A inspeção é
    agregada pelo ``member_id`` porque a unidade de montagem relevante é o
    montante, banzo ou diagonal completo, não um único palito isolado.
    """

    member_id: str
    member_group: str
    clicked_stick_id: str | None
    member_pieces: List[Dict[str, Any]]
    structural_pieces: List[Dict[str, Any]]
    auxiliary_pieces: List[Dict[str, Any]]
    joints: List[Dict[str, Any]]
    overlaps: List[Dict[str, Any]]
    cuts: List[Dict[str, Any]]
    member_check: Dict[str, Any] | None
    nominal_overlap_total_mm: float


class PieceInspectionService:
    """Converte ``stick_pieces`` e juntas em informação de montagem clicável.

    O catálogo é derivado do mesmo detalhamento utilizado para massa e CAD. A
    visualização não inventa ligações: a sobreposição exibida é sempre uma
    junta registrada em ``glue_joints``.
    """

    @staticmethod
    def _stick_id(row: Mapping[str, Any]) -> str:
        return str(row.get("stick_id") or "")

    @staticmethod
    def selected_member_id(selection_state: Any) -> str | None:
        """Extrai o membro selecionado dos hitboxes do Plotly.

        O ``customdata`` preserva o ``stick_id`` na primeira posição para
        rastreabilidade e carrega o ``member_id`` na segunda. Qualquer lâmina
        ou tala clicada seleciona o membro estrutural completo.
        """
        if not selection_state:
            return None
        if hasattr(selection_state, "selection"):
            selection = selection_state.selection
        elif isinstance(selection_state, Mapping):
            selection = selection_state.get("selection", selection_state)
        else:
            return None
        if hasattr(selection, "points"):
            points = selection.points
        elif isinstance(selection, Mapping):
            points = selection.get("points") or []
        else:
            points = []
        if not points:
            return None
        point = points[-1]
        custom = point.get("customdata") if isinstance(point, Mapping) else getattr(point, "customdata", None)
        if isinstance(custom, (list, tuple)) and len(custom) >= 2:
            return str(custom[1])
        return None

    @staticmethod
    def selected_stick_id(selection_state: Any) -> str | None:
        """Extrai o ``stick_id`` retornado por ``st.plotly_chart(on_select=...)``.

        O Streamlit pode retornar dict, objeto attribute-style ou ``None``. O
        primeiro item de ``customdata`` é sempre o identificador físico do
        palito nos traces de seleção gerados pelo ``VisualizationService``.
        """
        if not selection_state:
            return None
        if hasattr(selection_state, "selection"):
            selection = selection_state.selection
        elif isinstance(selection_state, Mapping):
            selection = selection_state.get("selection", selection_state)
        else:
            return None
        if hasattr(selection, "points"):
            points = selection.points
        elif isinstance(selection, Mapping):
            points = selection.get("points") or []
        else:
            points = []
        if not points:
            return None
        point = points[-1]
        custom = point.get("customdata") if isinstance(point, Mapping) else getattr(point, "customdata", None)
        if isinstance(custom, (list, tuple)) and custom:
            return str(custom[0])
        if custom:
            return str(custom)
        text = point.get("text") if isinstance(point, Mapping) else getattr(point, "text", None)
        return str(text).split("<br>", 1)[0] if text else None

    @staticmethod
    def _joint_mentions_stick(joint: Mapping[str, Any], stick_id: str) -> bool:
        return stick_id in {str(joint.get("piece_a") or ""), str(joint.get("piece_b") or "")}

    @staticmethod
    def _splint_joint_id(stick_id: str) -> str | None:
        if not stick_id.startswith("TALA-"):
            return None
        body = stick_id[len("TALA-"):]
        # TALA-J-M044-L01-P01-02-S01 -> J-M044-L01-P01-02
        return body.rsplit("-S", 1)[0] if "-S" in body else body

    @staticmethod
    def _is_auxiliary_piece(row: Mapping[str, Any]) -> bool:
        group = str(row.get("member_group") or "")
        return group.endswith("_splint") or str(row.get("stick_id") or "").startswith("TALA-")

    @classmethod
    def inspect_member(
        cls,
        member_id: str | int | None,
        pieces: Sequence[Mapping[str, Any]],
        glue_joints: Sequence[Mapping[str, Any]] | None = None,
        member_checks: Sequence[Mapping[str, Any]] | None = None,
        *,
        clicked_stick_id: str | None = None,
    ) -> MemberInspection | None:
        if member_id is None or str(member_id).strip() == "":
            return None
        mid = str(member_id)
        member_rows = [dict(row) for row in pieces if str(row.get("member_id")) == mid]
        if not member_rows:
            return None
        structural = [row for row in member_rows if not cls._is_auxiliary_piece(row)]
        auxiliary = [row for row in member_rows if cls._is_auxiliary_piece(row)]
        structural = structural or list(member_rows)
        primary_row = structural[0]
        member_group = str(primary_row.get("member_group") or "—")
        joints = [dict(j) for j in (glue_joints or []) if str(j.get("member_id")) == mid]

        overlaps: List[Dict[str, Any]] = []
        overlap_total = 0.0
        for joint in joints:
            length = safe_float(joint.get("overlap_length_mm"), 0.0) or 0.0
            overlap_total += max(0.0, float(length))
            overlaps.append(
                {
                    "junta": _text(joint.get("joint_id")),
                    "tipo": _text(joint.get("joint_type")),
                    "modelo": _text(joint.get("joint_model")),
                    "escopo": "todo o membro" if str(joint.get("lane")) == "all" else "junta local",
                    "sobreposição_mm": f"{float(length):.2f}",
                    "área_colada_mm²": _text(joint.get("physical_glue_area_mm2")),
                    "FS_cola": _text(joint.get("FS_glue_shear")),
                    "peça_A": _text(joint.get("piece_a")),
                    "peça_B": _text(joint.get("piece_b")),
                    "talas": _text(joint.get("splints_per_splice")),
                }
            )

        cuts: List[Dict[str, Any]] = []
        for row in structural:
            for endpoint, required_key, angle_key, host_key, relation_key, position_key in (
                ("início", "miter_cut_start_required", "miter_cut_start_angle_deg", "miter_cut_start_host_group", "miter_cut_start_relation", "miter_cut_start_position"),
                ("fim", "miter_cut_end_required", "miter_cut_end_angle_deg", "miter_cut_end_host_group", "miter_cut_end_relation", "miter_cut_end_position"),
            ):
                if not _bool(row.get(required_key)):
                    continue
                angle = safe_float(row.get(angle_key), None)
                shop_key = "miter_cut_start_shop_reference_angle_deg" if endpoint == "início" else "miter_cut_end_shop_reference_angle_deg"
                shop_angle = safe_float(row.get(shop_key), angle)
                cuts.append({
                    "palito": _text(row.get("stick_id")),
                    "extremidade": endpoint,
                    "ângulo_interno_CAD_deg": "—" if angle is None else f"{float(angle):.2f}",
                    "ângulo_de_gabarito_deg": "—" if shop_angle is None else f"{float(shop_angle):.2f}",
                    "hospedeiro": _text(row.get(host_key)),
                    "relação": _text(row.get(relation_key)),
                    "posição": _text(row.get(position_key)),
                })

        checks = [dict(c) for c in (member_checks or []) if str(c.get("member_id")) == mid]
        checks.sort(
            key=lambda c: (
                safe_float(c.get("FS_design", c.get("FS_min")), None) is None,
                safe_float(c.get("FS_design", c.get("FS_min")), 1.0e99) or 1.0e99,
            )
        )
        return MemberInspection(
            member_id=mid,
            member_group=member_group,
            clicked_stick_id=clicked_stick_id,
            member_pieces=member_rows,
            structural_pieces=structural,
            auxiliary_pieces=auxiliary,
            joints=joints,
            overlaps=overlaps,
            cuts=cuts,
            member_check=checks[0] if checks else None,
            nominal_overlap_total_mm=overlap_total,
        )

    @staticmethod
    def _member_status(info: MemberInspection) -> str:
        """Retorna um estado textual completo, nunca vazio, para a inspeção."""
        check = info.member_check or {}
        axial_force = safe_float(check.get("N_N", check.get("N_member_N")), None) if check else None
        fs = safe_float(check.get("FS_design", check.get("FS_min", check.get("FS_min_global"))), None) if check else None
        mode = str(check.get("governing_mode") or check.get("governing_mode_global") or "").strip()
        if axial_force is not None and abs(float(axial_force)) <= 1.0e-6:
            return "sem esforço axial relevante — detalhamento de montagem registrado"
        if fs is not None:
            suffix = f" — modo: {mode}" if mode and mode not in {"—", "unchecked"} else ""
            return f"verificado pelo membro — FS {float(fs):.3f}{suffix}"
        if info.structural_pieces:
            return "detalhado para montagem — sem FS isolado aplicável"
        return "componente auxiliar — verificação vinculada à junta"

    @staticmethod
    def _auxiliary_piece_status(info: MemberInspection) -> str:
        fs_values = [
            safe_float(joint.get("FS_glue_shear"), None)
            for joint in info.joints
            if safe_float(joint.get("FS_glue_shear"), None) is not None
        ]
        if fs_values:
            return f"tala auxiliar — controlada pela junta (FS cola {min(float(v) for v in fs_values):.3f})"
        return "tala auxiliar — controlada pela junta registrada"

    @staticmethod
    def cut_quantity_rows(pieces: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
        """Agrupa cortes físicos pela dimensão de preparação do palito.

        A tabela usa ``shop_cut_length_mm`` porque essa é a medida que chega
        ao gabarito de corte; inclui lâminas e talas efetivamente detalhadas.
        """
        counts: Dict[float, int] = {}
        for piece in pieces:
            length = safe_float(piece.get("shop_cut_length_mm", piece.get("cut_length_mm")), None)
            if length is None or float(length) <= 1.0e-9:
                continue
            key = round(float(length), 2)
            counts[key] = counts.get(key, 0) + 1
        return [
            {"tamanho do palito": f"{length:.2f} mm", "quantidade": str(quantity)}
            for length, quantity in sorted(counts.items(), key=lambda item: item[0], reverse=True)
        ]

    @staticmethod
    def bridge_summary_rows(
        cfg: Mapping[str, Any],
        pieces: Sequence[Mapping[str, Any]],
        detailed_summary: Mapping[str, Any] | None,
        rupture: Mapping[str, Any] | None,
        as_built_audit: Mapping[str, Any] | None,
        solver_status: str | None,
        support_checks: Sequence[Mapping[str, Any]] | None = None,
    ) -> List[Dict[str, str]]:
        """Monta a ficha padrão exibida quando nenhuma peça está selecionada."""
        bridge = dict(cfg.get("bridge", {}) or {})
        material = dict(cfg.get("material", {}) or {})
        analysis = dict(cfg.get("analysis", {}) or {})
        summary = dict(detailed_summary or {})
        rupture = dict(rupture or {})
        audit = dict(as_built_audit or {})
        span = float(safe_float(bridge.get("span_mm"), 0.0) or 0.0)
        length = span + float(safe_float(bridge.get("left_support_overhang_mm"), 0.0) or 0.0) + float(safe_float(bridge.get("right_support_overhang_mm"), 0.0) or 0.0)
        interpenetrations = int(safe_float(audit.get("interpenetration_count"), 0) or 0)
        gaps = int(safe_float(audit.get("node_connection_gap_piece_count"), 0) or 0)
        geometry_ok = interpenetrations == 0 and gaps == 0
        geometry_status = "OK — sem interpenetração/lacuna detectada" if geometry_ok else f"ATENÇÃO — {interpenetrations} interpenetração(ões), {gaps} lacuna(s)"
        member_ids = {str(row.get("member_id")) for row in pieces if str(row.get("member_id", "")).strip()}
        auxiliary_count = sum(1 for row in pieces if PieceInspectionService._is_auxiliary_piece(row))
        competitive_mass = safe_float(summary.get("competition_mass_g", summary.get("estimated_total_mass_g")), None)
        mass_margin = safe_float(summary.get("competition_mass_margin_g", summary.get("mass_margin_g")), None)
        failure_load = safe_float(rupture.get("predicted_breaking_load_design_kgf", rupture.get("predicted_breaking_load_kgf")), None)
        fs_member = safe_float(rupture.get("min_fs_member_design", rupture.get("min_fs_design")), None)
        fs_support = safe_float(rupture.get("min_fs_support"), None)
        fs_glue = safe_float(rupture.get("min_fs_glue"), None)
        target_fs = safe_float(analysis.get("target_min_fs"), None)
        wet_glue_mass = safe_float(summary.get("wet_glue_mass_g", summary.get("estimated_glue_mass_g")), None)
        wet_glue_budget = safe_float(summary.get("wet_glue_budget_g", material.get("wet_glue_budget_g")), None)
        wet_glue_margin = safe_float(summary.get("wet_glue_budget_margin_g"), None)
        warnings: List[str] = []
        if not geometry_ok:
            warnings.append("geometria com conflito")
        if fs_member is not None and target_fs is not None and float(fs_member) < float(target_fs) - 1.0e-9:
            warnings.append(f"FS membro {float(fs_member):.3f} < alvo {float(target_fs):.3f}")
        if wet_glue_margin is not None and float(wet_glue_margin) < -1.0e-9:
            warnings.append(f"cola úmida excede orçamento em {abs(float(wet_glue_margin)):.3f} g")
        global_status = geometry_status if not warnings else "ATENÇÃO — " + "; ".join(warnings)
        supports = list(support_checks or [])
        active_supports: Dict[float, int] = {}
        inactive_supports: Dict[float, int] = {}
        for support in supports:
            x = safe_float(support.get("x_mm"), None)
            if x is None:
                continue
            is_active = bool(support.get("support_active_vertical", False))
            bucket = active_supports if is_active else inactive_supports
            bucket[float(x)] = bucket.get(float(x), 0) + 1
        def _support_text(values: Dict[float, int]) -> str:
            return "—" if not values else "; ".join(f"x={x:.2f} mm ({n} nó(s))" for x, n in sorted(values.items()))
        return [
            {"item": "Status da ponte", "valor": global_status},
            {"item": "Solver", "valor": str(solver_status or "—")},
            {"item": "Carga de referência analisada", "valor": f"{float(safe_float(bridge.get('load_total_kgf'), 0.0) or 0.0):.3f} kgf"},
            {"item": "Apoios ativos no caso analisado", "valor": _support_text(active_supports)},
            {"item": "Apoios sem contato no caso analisado", "valor": _support_text(inactive_supports)},
            {"item": "Comprimento total entre extremos de apoio", "valor": f"{length:.2f} mm"},
            {"item": "Vão principal", "valor": f"{span:.2f} mm"},
            {"item": "Largura nominal entre treliças", "valor": f"{float(safe_float(bridge.get('width_mm'), 0.0) or 0.0):.2f} mm"},
            {"item": "Altura máxima nominal", "valor": f"{float(safe_float(bridge.get('center_height_mm'), 0.0) or 0.0):.2f} mm"},
            {"item": "Espessura / largura do palito-base", "valor": f"{float(safe_float(material.get('stick_thickness_mm'), 0.0) or 0.0):.2f} × {float(safe_float(material.get('stick_width_mm'), 0.0) or 0.0):.2f} mm"},
            {"item": "Membros estruturais", "valor": str(len(member_ids))},
            {"item": "Peças físicas detalhadas", "valor": str(len(pieces))},
            {"item": "Talas / peças auxiliares", "valor": str(auxiliary_count)},
            {"item": "Palitos comerciais previstos (com perda)", "valor": _text(summary.get("estimated_total_sticks_with_waste"))},
            {"item": "Massa instalada de madeira", "valor": "—" if safe_float(summary.get("installed_stick_mass_g"), None) is None else f"{float(summary['installed_stick_mass_g']):.3f} g"},
            {"item": "Massa curada de cola", "valor": "—" if safe_float(summary.get("cured_glue_mass_g"), None) is None else f"{float(summary['cured_glue_mass_g']):.3f} g"},
            {"item": "Cola úmida prevista / orçamento", "valor": ("—" if wet_glue_mass is None or wet_glue_budget is None else f"{float(wet_glue_mass):.3f} / {float(wet_glue_budget):.3f} g")},
            {"item": "Peso total estimado para competição", "valor": "—" if competitive_mass is None else f"{float(competitive_mass):.3f} g"},
            {"item": "Margem até o limite", "valor": "—" if mass_margin is None else f"{float(mass_margin):.3f} g"},
            {"item": "Peso de ruptura estimado", "valor": "—" if failure_load is None else f"{float(failure_load):.3f} kgf"},
            {"item": "FS mínimo — membros / apoios / cola", "valor": f"{'—' if fs_member is None else f'{float(fs_member):.3f}'} / {'—' if fs_support is None else f'{float(fs_support):.3f}'} / {'—' if fs_glue is None else f'{float(fs_glue):.3f}'}"},
            {"item": "FS alvo de projeto", "valor": "—" if target_fs is None else f"{float(target_fs):.3f}"},
            {"item": "Interpenetrações / lacunas auditadas", "valor": f"{interpenetrations} / {gaps}"},
        ]

    @staticmethod
    def member_summary_rows(info: MemberInspection) -> List[Dict[str, str]]:
        check = info.member_check or {}
        structural_mass = sum(float(safe_float(row.get("mass_g"), 0.0) or 0.0) for row in info.structural_pieces)
        auxiliary_mass = sum(float(safe_float(row.get("mass_g"), 0.0) or 0.0) for row in info.auxiliary_pieces)
        cut_lengths = [float(safe_float(row.get("shop_cut_length_mm"), 0.0) or 0.0) for row in info.structural_pieces]
        axial_force = safe_float(check.get("N_N", check.get("N_member_N")), None) if check else None
        fs = safe_float(check.get("FS_design", check.get("FS_min", check.get("FS_min_global"))), None) if check else None
        unloaded = axial_force is not None and abs(float(axial_force)) <= 1.0e-6
        layout = next((row.get("layout") or row.get("section_layout_effective") for row in info.structural_pieces if row.get("layout") or row.get("section_layout_effective")), "—")
        n_section = max((int(safe_float(row.get("n_sticks"), 0) or 0) for row in info.structural_pieces), default=0)
        primary = info.structural_pieces[0] if info.structural_pieces else {}
        assembled_length = safe_float(primary.get("assembled_member_length_mm", primary.get("fabrication_axis_length_mm")), None)
        assembled_width = safe_float(primary.get("assembled_member_width_mm"), None)
        assembled_thickness = safe_float(primary.get("assembled_member_thickness_mm"), None)
        longitudinal_model = _text(primary.get("longitudinal_splice_model"))
        return [
            {"item": "Membro selecionado", "valor": f"M{info.member_id} / {info.member_group}"},
            {"item": "Status", "valor": PieceInspectionService._member_status(info)},
            {"item": "Palitos resistentes na seção", "valor": str(n_section) if n_section else "—"},
            {"item": "Peças estruturais detalhadas", "valor": str(len(info.structural_pieces))},
            {"item": "Talas / auxiliares", "valor": str(len(info.auxiliary_pieces))},
            {"item": "Layout da seção", "valor": str(layout)},
            {"item": "Comprimento total montado", "valor": "—" if assembled_length is None else f"{float(assembled_length):.2f} mm"},
            {"item": "Largura total montada", "valor": "—" if assembled_width is None else f"{float(assembled_width):.2f} mm"},
            {"item": "Espessura total montada", "valor": "—" if assembled_thickness is None else f"{float(assembled_thickness):.2f} mm"},
            {"item": "Modelo longitudinal", "valor": longitudinal_model},
            {"item": "Faixa de cortes estruturais", "valor": f"{min(cut_lengths):.2f}–{max(cut_lengths):.2f} mm" if cut_lengths else "—"},
            {"item": "Sobreposição total registrada", "valor": f"{info.nominal_overlap_total_mm:.2f} mm"},
            {"item": "Massa estrutural", "valor": f"{structural_mass:.3f} g"},
            {"item": "Massa de talas/auxiliares", "valor": f"{auxiliary_mass:.3f} g"},
            {"item": "FS do membro", "valor": "não governante (N≈0)" if unloaded else ("—" if fs is None else f"{float(fs):.3f}")},
            {"item": "Modo governante", "valor": "sem esforço axial relevante" if unloaded else (str(check.get("governing_mode") or check.get("governing_mode_global") or "—") if fs is not None else "não aplicável")},
            {"item": "Palito clicado", "valor": str(info.clicked_stick_id or "—")},
        ]

    @staticmethod
    def member_piece_rows(info: MemberInspection) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        structural_status = PieceInspectionService._member_status(info)
        auxiliary_status = PieceInspectionService._auxiliary_piece_status(info)
        for piece in info.member_pieces:
            is_auxiliary = PieceInspectionService._is_auxiliary_piece(piece)
            rows.append({
                "palito": _text(piece.get("stick_id")),
                "tipo": "tala" if is_auxiliary else "estrutura",
                "status": auxiliary_status if is_auxiliary else structural_status,
                "linha": _text(piece.get("lane")),
                "segmento": _text(piece.get("piece_index")),
                "papel na seção": _text(piece.get("sandwich_lane_role") or piece.get("structural_lane_role") or piece.get("solid_laminate_role")),
                "corte_mm": f"{safe_float(piece.get('shop_cut_length_mm'), 0.0) or 0.0:.2f}",
                "instalado_mm": f"{safe_float(piece.get('installed_length_mm'), 0.0) or 0.0:.2f}",
                "massa_g": f"{safe_float(piece.get('mass_g'), 0.0) or 0.0:.3f}",
            })
        return rows

    @classmethod
    def inspect(
        cls,
        stick_id: str | None,
        pieces: Sequence[Mapping[str, Any]],
        glue_joints: Sequence[Mapping[str, Any]] | None = None,
        member_checks: Sequence[Mapping[str, Any]] | None = None,
    ) -> PieceInspection | None:
        if not stick_id:
            return None
        piece_rows = [dict(row) for row in pieces]
        piece_by_id = {cls._stick_id(row): row for row in piece_rows}
        row = piece_by_id.get(str(stick_id))
        if row is None:
            return None
        member_id = str(row.get("member_id"))
        member_rows = [p for p in piece_rows if str(p.get("member_id")) == member_id]
        joints_all = [dict(j) for j in (glue_joints or [])]
        selected_joint_id = cls._splint_joint_id(str(stick_id))
        if selected_joint_id:
            joints = [j for j in joints_all if str(j.get("joint_id")) == selected_joint_id]
        else:
            joints = [j for j in joints_all if cls._joint_mentions_stick(j, str(stick_id))]
        # A cola contínua do sanduíche é contabilizada por membro, pois cobre
        # simultaneamente todas as lâminas da seção. Ela também sobrepõe o
        # palito selecionado e precisa aparecer na ficha de montagem.
        member_wide_joints = [
            j for j in joints_all
            if str(j.get("member_id")) == member_id
            and (
                str(j.get("lane")) == "all"
                or str(j.get("joint_model")) == "sandwich_continuous_face_bond_mass_accounting"
            )
        ]
        known_joint_ids = {str(j.get("joint_id")) for j in joints}
        joints.extend(j for j in member_wide_joints if str(j.get("joint_id")) not in known_joint_ids)

        related_ids = {str(stick_id)}
        for joint in joints:
            for key in ("piece_a", "piece_b"):
                sid = str(joint.get(key) or "")
                if sid in piece_by_id:
                    related_ids.add(sid)
            joint_id = str(joint.get("joint_id") or "")
            related_ids.update(
                sid for sid in piece_by_id
                if sid.startswith(f"TALA-{joint_id}-")
            )
        related_rows = [p for p in piece_rows if cls._stick_id(p) in related_ids]
        # A vista detalhada deve mostrar a laminação do membro e, quando
        # aplicável, as talas conectadas à peça selecionada.
        displayed_ids = {cls._stick_id(p) for p in member_rows}
        related_rows = member_rows + [p for p in related_rows if cls._stick_id(p) not in displayed_ids]

        overlaps: List[Dict[str, Any]] = []
        overlap_total = 0.0
        for joint in joints:
            length = safe_float(joint.get("overlap_length_mm"), 0.0) or 0.0
            overlap_total += max(0.0, float(length))
            overlaps.append(
                {
                    "junta": _text(joint.get("joint_id")),
                    "tipo": _text(joint.get("joint_type")),
                    "modelo": _text(joint.get("joint_model")),
                    "escopo": "todo o membro" if str(joint.get("lane")) == "all" else "junta local",
                    "sobreposição_mm": length,
                    "área_colada_mm²": safe_float(joint.get("physical_glue_area_mm2"), None),
                    "FS_cola": safe_float(joint.get("FS_glue_shear"), None),
                    "peça_A": _text(joint.get("piece_a")),
                    "peça_B": _text(joint.get("piece_b")),
                    "talas": joint.get("splints_per_splice") or "—",
                }
            )
        cuts = [
            {
                "extremidade": "início",
                "exige_corte": _bool(row.get("miter_cut_start_required")),
                "ângulo_deg": safe_float(row.get("miter_cut_start_angle_deg"), None),
                "ângulo_gabarito_deg": safe_float(row.get("miter_cut_start_shop_reference_angle_deg"), safe_float(row.get("miter_cut_start_angle_deg"), None)),
                "hospedeiro": _text(row.get("miter_cut_start_host_group")),
                "relação": _text(row.get("miter_cut_start_relation")),
                "posição": _text(row.get("miter_cut_start_position")),
            },
            {
                "extremidade": "fim",
                "exige_corte": _bool(row.get("miter_cut_end_required")),
                "ângulo_deg": safe_float(row.get("miter_cut_end_angle_deg"), None),
                "ângulo_gabarito_deg": safe_float(row.get("miter_cut_end_shop_reference_angle_deg"), safe_float(row.get("miter_cut_end_angle_deg"), None)),
                "hospedeiro": _text(row.get("miter_cut_end_host_group")),
                "relação": _text(row.get("miter_cut_end_relation")),
                "posição": _text(row.get("miter_cut_end_position")),
            },
        ]
        checks = [dict(c) for c in (member_checks or []) if str(c.get("member_id")) == member_id]
        checks.sort(
            key=lambda c: (
                safe_float(c.get("FS_design", c.get("FS_min")), None) is None,
                safe_float(c.get("FS_design", c.get("FS_min")), 1.0e99) or 1.0e99,
            )
        )
        return PieceInspection(
            stick_id=str(stick_id),
            row=row,
            member_pieces=member_rows,
            related_pieces=related_rows,
            joints=joints,
            overlaps=overlaps,
            cuts=cuts,
            member_check=checks[0] if checks else None,
            nominal_overlap_total_mm=overlap_total,
        )

    @staticmethod
    def summary_rows(info: PieceInspection) -> List[Dict[str, Any]]:
        row = info.row
        check = info.member_check or {}
        return [
            {"item": "Palito selecionado", "valor": info.stick_id},
            {"item": "Membro / grupo", "valor": f"M{row.get('member_id')} / {row.get('member_group')}"},
            {"item": "Palitos na seção do membro", "valor": str(row.get("n_sticks", "—"))},
            {"item": "Palitos físicos detalhados no membro", "valor": str(len(info.member_pieces))},
            {"item": "Layout da seção", "valor": row.get("layout") or row.get("section_layout_effective") or "—"},
            {"item": "Função na seção", "valor": row.get("sandwich_lane_role") or row.get("structural_lane_role") or row.get("solid_laminate_role") or "—"},
            {"item": "Comprimento para corte", "valor": f"{safe_float(row.get('shop_cut_length_mm'), 0.0) or 0.0:.2f} mm"},
            {"item": "Comprimento instalado", "valor": f"{safe_float(row.get('installed_length_mm'), 0.0) or 0.0:.2f} mm"},
            {"item": "Comprimento total do membro montado", "valor": f"{safe_float(row.get('assembled_member_length_mm', row.get('fabrication_axis_length_mm')), 0.0) or 0.0:.2f} mm"},
            {"item": "Largura total do membro montado", "valor": f"{safe_float(row.get('assembled_member_width_mm'), 0.0) or 0.0:.2f} mm"},
            {"item": "Espessura total do membro montado", "valor": f"{safe_float(row.get('assembled_member_thickness_mm'), 0.0) or 0.0:.2f} mm"},
            {"item": "Modelo longitudinal", "valor": row.get("longitudinal_splice_model") or "—"},
            {"item": "Perda pelos chanfros", "valor": f"{safe_float(row.get('miter_cut_material_loss_length_mm'), 0.0) or 0.0:.2f} mm"},
            {"item": "Sobreposição nominal em juntas", "valor": f"{info.nominal_overlap_total_mm:.2f} mm"},
            {"item": "Massa deste palito", "valor": f"{safe_float(row.get('mass_g'), 0.0) or 0.0:.3f} g"},
            {"item": "Força axial atribuída", "valor": f"{safe_float(row.get('N_piece_N'), 0.0) or 0.0:.3f} N"},
            {
                "item": "FS do membro",
                "valor": (
                    f"{safe_float(check.get('FS_design', check.get('FS_min')), None):.3f}"
                    if check and safe_float(check.get('FS_design', check.get('FS_min')), None) is not None
                    else "—"
                ),
            },
            {"item": "Modo governante", "valor": check.get("governing_mode", "—") if check and safe_float(check.get('FS_design', check.get('FS_min')), None) is not None else "—"},
        ]
