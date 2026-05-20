from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.core.numeric import safe_float, safe_sort_key
from src.core.safety import risk_from_fs, safety_label
from src.domain.models import Member, Node
from src.services.geometry_service import GeometryService
from src.services.mass_guard import resolve_mass_limits
from src.services.section_service import SectionService
from src.services.splice_staggering_service import SpliceStaggeringService


class StickDetailService:
    """
    Modelo rápido peça-a-peça: palitos, sobreposições, cola, massa e recomendações.

    Este serviço não faz FEM. Ele expande cada membro estrutural equivalente em
    peças de palito, estima cortes, sobreposições, áreas coladas, tensões médias,
    massa e recomendações construtivas.
    """

    def __init__(self, section_service: SectionService | None = None) -> None:
        self.sections = section_service or SectionService()
        self.splice_stagger = SpliceStaggeringService()

    @staticmethod
    def floor_to_cut_increment(
        value_mm: float,
        increment_mm: float = 5.0,
        min_value_mm: float = 5.0,
    ) -> float:
        inc = max(1.0e-9, float(increment_mm))
        v = max(float(min_value_mm), float(value_mm))
        return max(float(min_value_mm), math.floor(v / inc) * inc)

    @staticmethod
    def _unit_vector(ni: Node, nj: Node) -> Tuple[float, float, float, float]:
        dx = nj.x - ni.x
        dy = nj.y - ni.y
        dz = nj.z - ni.z

        L = math.sqrt(dx * dx + dy * dy + dz * dz)

        if L <= 0:
            return 0.0, 0.0, 0.0, 0.0

        return dx / L, dy / L, dz / L, L

    @staticmethod
    def _cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    @staticmethod
    def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def _normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        if n <= 1.0e-12:
            return (0.0, 0.0, 0.0)
        return (v[0] / n, v[1] / n, v[2] / n)

    @classmethod
    def _local_section_axes(
        cls,
        ux: float,
        uy: float,
        uz: float,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return member-local y/z axes used to place each stick lane.

        The solver member line is interpreted as the centroidal axis.  The local
        section ``z`` axis is kept as close as possible to global vertical; for a
        nearly vertical member we fall back to a horizontal construction frame.
        This prevents the piece-by-piece 3D view from inventing one-sided lane
        offsets that do not exist in the calculation.
        """
        d = cls._normalize((float(ux), float(uy), float(uz)))
        if d == (0.0, 0.0, 0.0):
            return (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)

        global_z = (0.0, 0.0, 1.0)
        proj_z = (
            global_z[0] - cls._dot(global_z, d) * d[0],
            global_z[1] - cls._dot(global_z, d) * d[1],
            global_z[2] - cls._dot(global_z, d) * d[2],
        )
        local_z = cls._normalize(proj_z)
        if local_z == (0.0, 0.0, 0.0):
            global_y = (0.0, 1.0, 0.0)
            proj_y = (
                global_y[0] - cls._dot(global_y, d) * d[0],
                global_y[1] - cls._dot(global_y, d) * d[1],
                global_y[2] - cls._dot(global_y, d) * d[2],
            )
            local_y = cls._normalize(proj_y)
            if local_y == (0.0, 0.0, 0.0):
                local_y = (1.0, 0.0, 0.0)
            local_z = cls._normalize(cls._cross(d, local_y))
            return local_y, local_z

        local_y = cls._normalize(cls._cross(local_z, d))
        if local_y == (0.0, 0.0, 0.0):
            local_y = (0.0, 1.0, 0.0)
        return local_y, local_z

    @staticmethod
    def _x_bracing_layer_offset(
        member_group: str,
        ni: Node,
        nj: Node,
        *,
        stick_thickness_mm: float,
        detail: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve colisões físicas em contraventamentos em X.

        O modelo estrutural usa barras no eixo dos nós.  Em um X real feito com
        palitos, duas diagonais que se cruzam não podem ocupar o mesmo plano no
        ponto médio.  Há duas soluções montáveis: cortar uma diagonal e colar a
        ponta na face da outra, ou manter ambas contínuas em camadas diferentes
        (uma "na frente" e outra "atrás").  Para não criar uma conexão de nó
        central que o solver não calcula, adotamos a segunda opção por padrão:
        camadas alternadas, sem transferência de força no cruzamento.

        Isso não adiciona material nem resistência; apenas desloca a posição
        peça-a-peça e deixa explícito que a colagem resistente continua sendo
        nos nós/extremidades.  A separação é da ordem da espessura do palito.
        """
        group = str(member_group or "")
        layer_groups = set(str(v) for v in (detail.get("x_bracing_layered_groups") or ["bottom_bracing", "cross_frame_bracing"]))
        if group not in layer_groups:
            return {
                "offset": (0.0, 0.0, 0.0),
                "layer": 0,
                "plane": "",
                "handling": "not_x_bracing",
                "midspan_connected": False,
            }

        gap = max(0.0, float(detail.get("x_bracing_layer_clearance_mm", 0.30)))
        sep = max(0.1, float(stick_thickness_mm) + gap)
        off = 0.5 * sep

        dx = float(nj.x - ni.x)
        dy = float(nj.y - ni.y)
        dz = float(nj.z - ni.z)

        if group == "bottom_bracing":
            # Plano x-y; separar em z.  Sinal alterna entre / e \\.
            sign = 1.0 if dx * dy >= 0.0 else -1.0
            return {
                "offset": (0.0, 0.0, sign * off),
                "layer": int(sign),
                "plane": "bottom_xy",
                "handling": "alternate_front_back_layer_no_midspan_joint",
                "midspan_connected": False,
            }
        if group == "cross_frame_bracing":
            # Plano y-z; separar em x.  Sinal alterna entre / e \\.
            sign = 1.0 if dy * dz >= 0.0 else -1.0
            return {
                "offset": (sign * off, 0.0, 0.0),
                "layer": int(sign),
                "plane": "crossframe_yz",
                "handling": "alternate_front_back_layer_no_midspan_joint",
                "midspan_connected": False,
            }
        return {
            "offset": (0.0, 0.0, 0.0),
            "layer": 0,
            "plane": "",
            "handling": "not_layered_by_rule",
            "midspan_connected": False,
        }

    @staticmethod
    def _piece_intervals(
        L: float,
        stick_len: float,
        overlap: float,
    ) -> List[Tuple[float, float, float]]:
        """
        Divide um membro de comprimento L em peças de palito.

        Retorna lista de:
            s0, s1, comprimento_de_corte

        A peça seguinte começa antes do fim da anterior quando há sobreposição.
        """
        if L <= 0:
            return []

        if stick_len <= 0:
            raise ValueError("stick_length_mm precisa ser maior que zero.")

        if L <= stick_len:
            return [(0.0, L, L)]

        overlap = max(0.0, min(overlap, stick_len * 0.75))
        step = max(1.0e-6, stick_len - overlap)

        out: List[Tuple[float, float, float]] = []
        s0 = 0.0

        while s0 < L - 1.0e-9:
            s1 = min(L, s0 + stick_len)
            out.append((s0, s1, s1 - s0))

            if s1 >= L - 1.0e-9:
                break

            s0 += step

        return out


    @staticmethod
    def _split_interval_to_stock_limit(
        s0: float,
        s1: float,
        *,
        max_cut_mm: float,
        overlap_mm: float,
        cut_increment_mm: float,
    ) -> List[Tuple[float, float, float]]:
        """Reparte um intervalo físico para nenhum corte exceder o palito real."""
        a = float(s0)
        b = float(s1)
        if b <= a + 1.0e-9:
            return []
        max_cut = max(1.0, float(max_cut_mm))
        overlap = max(0.0, min(float(overlap_mm), 0.75 * max_cut))
        inc = max(1.0, float(cut_increment_mm))
        out: List[Tuple[float, float, float]] = []
        cur = a
        while cur < b - 1.0e-9:
            nxt = min(b, cur + max_cut)
            if nxt < b - 1.0e-9:
                rounded = math.floor(nxt / inc) * inc
                if rounded > cur + max(5.0, 0.30 * max_cut):
                    nxt = rounded
            out.append((cur, nxt, nxt - cur))
            if nxt >= b - 1.0e-9:
                break
            cur = max(cur + 1.0, nxt - overlap)
        return out

    @classmethod
    def _enforce_stock_limit_on_intervals(
        cls,
        intervals: List[Tuple[float, float, float]],
        *,
        max_cut_mm: float,
        overlap_mm: float,
        cut_increment_mm: float,
    ) -> tuple[List[Tuple[float, float, float]], int]:
        fixed: List[Tuple[float, float, float]] = []
        splits = 0
        for s0, s1, _cut in intervals:
            parts = cls._split_interval_to_stock_limit(
                float(s0),
                float(s1),
                max_cut_mm=max_cut_mm,
                overlap_mm=overlap_mm,
                cut_increment_mm=cut_increment_mm,
            )
            if len(parts) > 1:
                splits += 1
            fixed.extend(parts)
        return fixed, splits

    @staticmethod
    def _pack_cuts_best_fit(
        cuts: List[float],
        blank_length: float,
        kerf: float = 1.0,
    ) -> List[List[float]]:
        """
        Empacota cortes em palitos brutos usando heurística best-fit decrescente.
        Não é otimização exata, mas é rápida e suficiente para plano preliminar.
        """
        clean_cuts = [
            float(c)
            for c in cuts
            if safe_float(c, None) is not None and float(c) > 0
        ]

        clean_cuts = sorted(clean_cuts, reverse=True)

        bins: List[List[float]] = []
        remaining: List[float] = []

        for c in clean_cuts:
            best_i = None
            best_rem = None

            for i, rem in enumerate(remaining):
                need = c + (kerf if bins[i] else 0.0)

                if need <= rem + 1.0e-9:
                    nr = rem - need

                    if best_rem is None or nr < best_rem:
                        best_i = i
                        best_rem = nr

            if best_i is None:
                bins.append([c])
                remaining.append(max(0.0, blank_length - c))
            else:
                bins[best_i].append(c)
                remaining[best_i] = best_rem if best_rem is not None else remaining[best_i]

        return bins

    def analyze(
        self,
        cfg: Dict,
        nodes: List[Node],
        members: List[Member],
        member_results: List[Dict],
        member_checks: List[Dict],
        out_dir: str | Path,
    ) -> Dict:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        mat = cfg["material"]
        detail = cfg.get("detail_model", {})

        stick_len = float(mat.get("stick_length_mm", 120.0))
        stick_w = float(mat.get("stick_width_mm", 7.0))
        stick_t = float(mat.get("stick_thickness_mm", 1.5))
        stick_mass = float(mat.get("stick_mass_g", 1.4))

        overlap = float(detail.get("overlap_length_mm", 30.0))
        glue_tau = float(detail.get("glue_shear_strength_MPa", 3.5))
        glue_sf = float(detail.get("default_joint_safety_factor", 2.0))
        min_end_margin = max(0.0, float(detail.get("min_end_margin_mm", 10.0)))
        cut_increment_mm = max(0.5, float(detail.get("cut_increment_mm", 5.0)))
        allow_cut_rounding = bool(detail.get("allow_cut_rounding", True))
        min_cut_length_mm = max(1.0, float(detail.get("min_cut_length_mm", 5.0)))
        max_cut_length_mm = min(
            stick_len,
            max(1.0, float(detail.get("max_cut_length_mm", stick_len))),
        )
        strict_cut_length = bool(detail.get("strict_cut_length", True))
        stock_limit_splits = 0
        # global default joint models.  These may be overridden per member by
        # a connection planner via ``cfg['member_joint_plan']``.
        tension_joint_model = str(detail.get("tension_joint_model", "double_lap_reinforced"))
        compression_joint_model = str(detail.get("compression_joint_model", "double_lap_reinforced"))
        glue_spread = float(detail.get("glue_spread_g_per_m2", 160.0))
        glue_eff = float(detail.get("glue_mass_efficiency", 0.65))
        glue_cure_solids_fraction = max(
            0.30,
            min(0.80, float(detail.get("glue_cure_solids_fraction", 0.50))),
        )
        imperfection_e = float(detail.get("imperfection_eccentricity_mm", 2.0))
        waste = float(detail.get("construction_waste_factor", 0.08))
        kerf = float(detail.get("saw_kerf_mm", 1.0))
        reinforce_if = float(detail.get("reinforce_if_fs_lt", 2.0))
        remove_if = float(detail.get("allow_recommend_removal_if_fs_gt", 8.0))
        tension_only = bool(detail.get("tension_only_stabilizers", True))

        node_by_id = {n.id: n for n in nodes}
        res_by = {int(r["member_id"]): r for r in member_results}
        chk_by = {int(r["member_id"]): r for r in member_checks}
        sizing_map = cfg.get("member_sizing_plan_by_id", {}) or {}

        stabilizers = set(cfg.get("analysis", {}).get("stabilizer_groups", []))

        stick_rows: List[Dict] = []
        joint_rows: List[Dict] = []
        member_rows: List[Dict] = []
        reinf_rows: List[Dict] = []

        cut_counter: Counter = Counter()
        cut_lengths: List[float] = []

        total_glue_area = 0.0
        total_pieces = 0
        total_cut = 0.0

        # Determinar se estamos usando modelo de quarto de ponte.  Quando
        # `use_quarter_model` é verdadeiro e um valor de
        # `quarter_member_count` é fornecido, cada membro pode ser atribuído
        # a um dos quadrantes.  Isto permite alternar a orientação das emendas
        # por quadrante para reduzir alinhamentos contínuos.
        use_quarter_model = bool(cfg.get("analysis", {}).get("use_quarter_model", False))
        quarter_count = 0
        if use_quarter_model:
            try:
                quarter_count = int(cfg.get("analysis", {}).get("quarter_member_count", 0))
            except (TypeError, ValueError):
                quarter_count = 0

        for m in members:
            ni = node_by_id[m.i]
            nj = node_by_id[m.j]

            ux, uy, uz, L = self._unit_vector(ni, nj)

            if L <= 0:
                continue

            res = res_by.get(m.id, {})
            chk = chk_by.get(m.id, {})
            sizing = sizing_map.get(str(m.id)) or sizing_map.get(m.id) or {}
            member_plan = (cfg.get("member_joint_plan", {}) or {}).get(m.id) or (cfg.get("member_joint_plan", {}) or {}).get(str(m.id))

            N = safe_float(res.get("N_N"), 0.0) or 0.0
            n_lanes = max(1, int(m.n_sticks))
            member_overlap = overlap
            if isinstance(member_plan, dict):
                planned_overlap = safe_float(member_plan.get("required_overlap_mm"), None)
                if planned_overlap is not None:
                    member_overlap = max(8.0, min(0.85 * stick_len, float(planned_overlap)))

            layout_cfg = cfg.get("section_layout_by_group", {}).get(
                m.group,
                {"layout": "stacked"},
            )

            layout_cfg_detail = dict(layout_cfg)
            layout_cfg_detail.setdefault(
                "composite_action",
                detail.get("composite_action", {}),
            )
            sec = self.sections.composite_section(n_lanes, mat, layout_cfg_detail)

            section_positions = list(sec.get("stick_positions_yz", []) or [])
            if len(section_positions) < n_lanes:
                section_positions.extend([(0.0, 0.0)] * (n_lanes - len(section_positions)))
            cy = safe_float(sec.get("centroid_y_mm"), 0.0) or 0.0
            cz = safe_float(sec.get("centroid_z_mm"), 0.0) or 0.0
            local_y_axis, local_z_axis = self._local_section_axes(ux, uy, uz)
            x_layer = self._x_bracing_layer_offset(
                m.group,
                ni,
                nj,
                stick_thickness_mm=stick_t,
                detail=detail,
            )
            x_layer_offset = tuple(x_layer.get("offset", (0.0, 0.0, 0.0)))

            stick_orientation = str(sec.get("stick_orientation", layout_cfg_detail.get("stick_orientation", "flat"))).strip().lower()
            lane_orientations = list(sec.get("stick_orientations", []) or [])
            lane_widths = list(sec.get("stick_width_y_mm_by_lane", []) or [])
            lane_heights = list(sec.get("stick_height_z_mm_by_lane", []) or [])
            visual_width_mm = safe_float(sec.get("stick_width_y_mm"), None)
            visual_thickness_mm = safe_float(sec.get("stick_height_z_mm"), None)
            if visual_width_mm is None or visual_thickness_mm is None:
                if stick_orientation == "edge":
                    visual_width_mm = stick_t
                    visual_thickness_mm = stick_w
                else:
                    visual_width_mm = stick_w
                    visual_thickness_mm = stick_t

            per_lane = N / n_lanes
            piece_area = stick_w * stick_t
            per_sigma = per_lane / piece_area if piece_area else 0.0

            # Gera as subdivisões do membro em peças de palito.  Caso
            # seja um modelo de quarto, alternamos a orientação das
            # emendas em quadrantes ímpares para evitar alinhamento
            # perfeito de juntas nas quatro porções da ponte.  Para
            # quadrantes ímpares, invertimos a ordem de segmentação (os
            # cortes passam a ser contados a partir da extremidade oposta).
            intervals = self._piece_intervals(L, stick_len, member_overlap)
            quadrant_id = 0
            if use_quarter_model and quarter_count > 0:
                # Determinar qual quadrante este membro pertence com base
                # no número de membros no quarto.  O identificador do
                # quadrante é dado por inteiro da divisão do índice do
                # membro (começando em 0) pelo total de membros de um
                # quarto.
                try:
                    quadrant_id = (int(m.id) - 1) // int(quarter_count)
                except (TypeError, ValueError, ZeroDivisionError):
                    quadrant_id = 0
                # Inverter a orientação das emendas para quadrantes ímpares
                if quadrant_id % 2 == 1:
                    rev: List[Tuple[float, float, float]] = []
                    for s0, s1, cl in reversed(intervals):
                        # Para inverter, subtrai os limites do comprimento total
                        rev.append((L - s1, L - s0, cl))
                    intervals = rev

            base_intervals = intervals

            r_y = self.sections.radius_of_gyration(sec["Iy"], sec["A"])
            r_z = self.sections.radius_of_gyration(sec["Iz"], sec["A"])

            slender_y = m.Ky * L / r_y if r_y else None
            slender_z = m.Kz * L / r_z if r_z else None

            M_imp = abs(N) * imperfection_e if N < 0 else 0.0

            c_y = sec.get("width_mm", stick_w) / 2.0
            c_z = sec.get("thickness_mm", stick_t) / 2.0

            sig_by = M_imp * c_z / sec["Iy"] if sec["Iy"] else 0.0
            sig_bz = M_imp * c_y / sec["Iz"] if sec["Iz"] else 0.0

            sigma_axial_member = N / sec["A"] if sec["A"] else 0.0
            sig_comb = abs(sigma_axial_member) + abs(sig_by) + abs(sig_bz)

            member_glue = 0.0
            joint_fs_values: List[float] = []
            # Determine joint model.  If a connection plan is attached to
            # the configuration it overrides the global defaults on a per
            # member basis.  The plan should be a dictionary keyed by
            # member id (as int or str) containing a ``recommended_joint_model``
            # field.  When absent the global tension/compression model is used.
            if member_plan and isinstance(member_plan, dict):
                plan_model = member_plan.get("recommended_joint_model") or member_plan.get("joint_model")
            else:
                plan_model = None
            if plan_model:
                joint_model = str(plan_model)
            else:
                joint_model = tension_joint_model if N >= 0 else compression_joint_model

            joint_area_factor = {
                "butt_plain": 0.35,
                "single_lap": 1.00,
                "single_lap_tala": 1.30,
                "butt_small_splints": 1.45,
                "butt_full_splints": 1.70,
                "double_lap": 1.75,
                "double_lap_reinforced": 2.10,
                "scarf": 1.55,
                "half_lap_notched": 1.40,
            }.get(joint_model, 1.0)
            joint_secondary_bending_factor = {
                "butt_plain": 1.55,
                "single_lap": 1.25,
                "single_lap_tala": 1.12,
                "butt_small_splints": 1.05,
                "butt_full_splints": 0.98,
                "double_lap": 1.00,
                "double_lap_reinforced": 0.95,
                "scarf": 1.00,
                "half_lap_notched": 1.08,
            }.get(joint_model, 1.0)

            for lane in range(1, n_lanes + 1):
                lane_yz = section_positions[lane - 1] if lane - 1 < len(section_positions) else (0.0, 0.0)
                try:
                    lane_y = float(lane_yz[0]) - cy
                    lane_z = float(lane_yz[1]) - cz
                except (TypeError, ValueError, IndexError):
                    lane_y = 0.0
                    lane_z = 0.0
                lane_offset_vec = (
                    lane_y * local_y_axis[0] + lane_z * local_z_axis[0] + float(x_layer_offset[0]),
                    lane_y * local_y_axis[1] + lane_z * local_z_axis[1] + float(x_layer_offset[1]),
                    lane_y * local_y_axis[2] + lane_z * local_z_axis[2] + float(x_layer_offset[2]),
                )
                lane_orientation = str(lane_orientations[lane - 1]).strip().lower() if lane - 1 < len(lane_orientations) else stick_orientation
                lane_visual_width_mm = safe_float(lane_widths[lane - 1], None) if lane - 1 < len(lane_widths) else None
                lane_visual_thickness_mm = safe_float(lane_heights[lane - 1], None) if lane - 1 < len(lane_heights) else None
                if lane_visual_width_mm is None or lane_visual_thickness_mm is None:
                    lane_visual_width_mm = visual_width_mm
                    lane_visual_thickness_mm = visual_thickness_mm
                lane_intervals = list(base_intervals)
                if detail.get("splice_stagger_enabled", True):
                    lane_intervals = self.splice_stagger.offset_splice_positions(
                        lane_intervals,
                        member_length=L,
                        quadrant_id=quadrant_id,
                        lane_id=lane,
                        cfg=cfg,
                    )
                if strict_cut_length:
                    lane_intervals, split_count = self._enforce_stock_limit_on_intervals(
                        lane_intervals,
                        max_cut_mm=max_cut_length_mm,
                        overlap_mm=overlap,
                        cut_increment_mm=cut_increment_mm,
                    )
                    stock_limit_splits += split_count
                prev_id = None
                prev_end = None

                for piece_index, (s0, s1, cut_len) in enumerate(lane_intervals, 1):
                    # Arredondamento de corte para incremento de oficina (ex.: 5 mm).
                    geom_len = max(0.0, float(cut_len))
                    if strict_cut_length and geom_len > max_cut_length_mm + 1.0e-9:
                        geom_len = max_cut_length_mm
                        s1 = min(L, s0 + geom_len)
                    if allow_cut_rounding and geom_len <= max_cut_length_mm + 1.0e-9:
                        cut_len_rounded = self.floor_to_cut_increment(
                            geom_len,
                            increment_mm=cut_increment_mm,
                            min_value_mm=min_cut_length_mm,
                        )
                        # Não reduzir abaixo do necessário perto de extremidades críticas.
                        if s0 <= min_end_margin or (L - s1) <= min_end_margin:
                            cut_len_rounded = geom_len
                    else:
                        cut_len_rounded = geom_len
                    cut_rounding_delta = geom_len - cut_len_rounded
                    sid = f"M{m.id:03d}-L{lane:02d}-P{piece_index:02d}"

                    x0 = ni.x + ux * s0 + lane_offset_vec[0]
                    y0 = ni.y + uy * s0 + lane_offset_vec[1]
                    z0 = ni.z + uz * s0 + lane_offset_vec[2]

                    x1 = ni.x + ux * s1 + lane_offset_vec[0]
                    y1 = ni.y + uy * s1 + lane_offset_vec[1]
                    z1 = ni.z + uz * s1 + lane_offset_vec[2]

                    total_pieces += 1
                    total_cut += cut_len_rounded

                    cut_lengths.append(cut_len_rounded)
                    cut_counter[round(cut_len_rounded, 1)] += 1

                    stick_rows.append(
                        {
                            "stick_id": sid,
                            "member_id": m.id,
                            "member_group": m.group,
                            "lane": lane,
                            "piece_index": piece_index,
                            "s0_mm": s0,
                            "s1_mm": s1,
                            "geometric_piece_length_mm": geom_len,
                            "cut_length_mm": cut_len_rounded,
                            "cut_rounding_delta_mm": cut_rounding_delta,
                            "max_cut_length_mm": max_cut_length_mm,
                            "dimension_ok_length": bool(cut_len_rounded <= max_cut_length_mm + 1.0e-9),
                            "x0_mm": x0,
                            "y0_mm": y0,
                            "z0_mm": z0,
                            "x1_mm": x1,
                            "y1_mm": y1,
                            "z1_mm": z1,
                            "N_piece_N": per_lane,
                            "sigma_axial_piece_MPa": per_sigma,
                            "member_state": "tension" if N >= 0 else "compression",
                            "stick_orientation": lane_orientation,
                            "section_layout_effective": sec.get("layout"),
                            "section_layout_requested": sec.get("requested_layout", layout_cfg_detail.get("layout", "stacked")),
                            "section_connection_model": sec.get("section_connection_model", sec.get("layout")),
                            "width_mm": stick_w,
                            "thickness_mm": stick_t,
                            "visual_width_mm": lane_visual_width_mm,
                            "visual_thickness_mm": lane_visual_thickness_mm,
                            "dimension_ok_width": bool(max(lane_visual_width_mm, lane_visual_thickness_mm) <= max(stick_w, stick_t) + 1.0e-9),
                            "dimension_ok_thickness": bool(min(lane_visual_width_mm, lane_visual_thickness_mm) <= min(stick_w, stick_t) + 1.0e-9),
                            "section_local_y_mm": lane_y,
                            "section_local_z_mm": lane_z,
                            "section_global_offset_x_mm": lane_offset_vec[0],
                            "section_global_offset_y_mm": lane_offset_vec[1],
                            "section_global_offset_z_mm": lane_offset_vec[2],
                            "x_bracing_layer": x_layer.get("layer"),
                            "x_bracing_plane": x_layer.get("plane"),
                            "x_bracing_crossing_handling": x_layer.get("handling"),
                            "x_bracing_midspan_connected": bool(x_layer.get("midspan_connected", False)),
                            "section_axis_y_x": local_y_axis[0],
                            "section_axis_y_y": local_y_axis[1],
                            "section_axis_y_z": local_y_axis[2],
                            "section_axis_z_x": local_z_axis[0],
                            "section_axis_z_y": local_z_axis[1],
                            "section_axis_z_z": local_z_axis[2],
                            "section_centroid_y_mm": cy,
                            "section_centroid_z_mm": cz,
                            "n_sticks": n_lanes,
                            "layout": sec.get("layout"),
                            "quadrant_id": quadrant_id,
                            "mass_g": stick_mass * cut_len_rounded / stick_len,
                        }
                    )

                    if prev_id is not None and prev_end is not None:
                        overlap_actual = max(0.0, prev_end - s0)
                        glue_area = overlap_actual * stick_w * joint_area_factor

                        if glue_area > 0:
                            glue_shear = (abs(per_lane) / glue_area) * joint_secondary_bending_factor
                        else:
                            glue_shear = None

                        glue_allow = glue_tau / glue_sf if glue_sf > 0 else None

                        if glue_shear is None or glue_shear <= 0 or glue_allow is None:
                            fs_glue = None
                        else:
                            fs_glue = glue_allow / glue_shear

                        fs_glue_clean = safe_float(fs_glue, None)

                        if fs_glue_clean is not None:
                            joint_fs_values.append(fs_glue_clean)

                        member_glue += glue_area
                        total_glue_area += glue_area

                        joint_rows.append(
                            {
                                "joint_id": f"J-M{m.id:03d}-L{lane:02d}-P{piece_index-1:02d}-{piece_index:02d}",
                                "member_id": m.id,
                                "member_group": m.group,
                                "lane": lane,
                                "piece_a": prev_id,
                                "piece_b": sid,
                                "joint_type": "lap_overlap",
                                "joint_model": joint_model,
                                "overlap_length_mm": overlap_actual,
                                "splice_center_mm": 0.5 * (prev_end + s0),
                                "quadrant_id": quadrant_id,
                                "joint_area_factor": joint_area_factor,
                                "joint_secondary_bending_factor": joint_secondary_bending_factor,
                                "glue_area_mm2": glue_area,
                                "force_transfer_N": abs(per_lane),
                                "glue_shear_MPa": glue_shear,
                                "glue_allow_design_MPa": glue_allow,
                                        "FS_glue_shear": fs_glue_clean,
                                        "FS_glue_shear_label": safety_label(fs_glue_clean),
                                        "risk_flag": risk_from_fs(fs_glue_clean),
                                        "splice_pattern": self.splice_stagger.assign_splice_stagger_pattern(
                                            cfg,
                                            {"member_id": m.id},
                                            quadrant_id,
                                            lane,
                                        ).get("splice_pattern", "brick_alt"),
                                        "stagger_offset_mm": self.splice_stagger.assign_splice_stagger_pattern(
                                            cfg,
                                            {"member_id": m.id},
                                            quadrant_id,
                                            lane,
                                        ).get("stagger_offset_mm", 0.0),
                            }
                        )

                    prev_id = sid
                    prev_end = s1

            glue_mass = (member_glue / 1_000_000.0) * glue_spread / max(glue_eff, 1.0e-6)

            fs_min_global = safe_float(chk.get("FS_min"), None)
            fs_min_global_label = safety_label(fs_min_global)

            role = chk.get("member_role", "secondary")
            gov = chk.get("governing_mode", "")
            report_mode = chk.get("report_mode", gov)

            if role == "stabilizer" and tension_only:
                action = (
                    "manter como travamento/tension-only; "
                    "não dimensionar como coluna comprimida"
                )
                priority = "interpretation"
            elif fs_min_global is not None and fs_min_global < reinforce_if:
                if "buckling" in str(gov):
                    action = (
                        "reforçar: aumentar inércia, usar seção caixa/espaçada "
                        "ou reduzir comprimento livre"
                    )
                else:
                    action = (
                        "reforçar: adicionar palitos contínuos ou aumentar "
                        "sobreposição/talas"
                    )
                priority = "high"
            elif fs_min_global is not None and fs_min_global > remove_if and role != "primary":
                action = "avaliar remoção/redução: baixa solicitação relativa"
                priority = "low"
            else:
                action = "manter"
                priority = "normal"

            if priority != "normal":
                reinf_rows.append(
                    {
                        "member_id": m.id,
                        "group": m.group,
                        "role": role,
                        "N_N": N,
                        "FS_min": fs_min_global,
                        "FS_min_label": fs_min_global_label,
                        "governing_mode": gov,
                        "report_mode": report_mode,
                        "suggested_action": action,
                        "priority": priority,
                    }
                )

            fs_min_glue = min(joint_fs_values) if joint_fs_values else None

            member_rows.append(
                {
                    "member_id": m.id,
                    "group": m.group,
                    "role": role,
                    "n_sticks_current": n_lanes,
                    "n_sticks_recommended": int(sizing.get("n_sticks_recommended", n_lanes)),
                    "n_lanes_sticks": n_lanes,
                    "pieces_per_lane": len(base_intervals),
                    "total_piece_count": len(base_intervals) * n_lanes,
                    "member_length_mm": L,
                    "layout": sec.get("layout"),
                    "section_A_mm2": sec["A"],
                    "section_Iy_mm4": sec["Iy"],
                    "section_Iz_mm4": sec["Iz"],
                    "section_Iy_perfect_mm4": sec.get("Iy_perfect"),
                    "section_Iz_perfect_mm4": sec.get("Iz_perfect"),
                    "section_Iy_noncomposite_mm4": sec.get("Iy_noncomposite"),
                    "section_Iz_noncomposite_mm4": sec.get("Iz_noncomposite"),
                    "section_eta_I": sec.get("eta_I"),
                    "section_J_mm4_est": sec["J"],
                    "radius_y_mm": r_y,
                    "radius_z_mm": r_z,
                    "slenderness_y": slender_y,
                    "slenderness_z": slender_z,
                    "N_member_N": N,
                    "N_per_lane_N": per_lane,
                    "sigma_axial_member_MPa": sigma_axial_member,
                    "sigma_axial_piece_MPa": per_sigma,
                    "M_imperfection_Nmm": M_imp,
                    "sigma_bending_est_MPa": max(abs(sig_by), abs(sig_bz)),
                    "sigma_combined_est_MPa": sig_comb,
                    "joint_model": joint_model,
                    "glue_area_total_mm2": member_glue,
                    "glue_mass_est_g": glue_mass,
                    "FS_min_global": fs_min_global,
                    "FS_min_global_label": fs_min_global_label,
                    "FS_min_glue": fs_min_glue,
                    "FS_min_glue_label": safety_label(fs_min_glue),
                    "governing_mode_global": gov,
                    "report_mode_global": report_mode,
                    "suggested_action": action,
                }
            )

        # Detecta e anota alinhamentos críticos de emendas após detalhamento completo.
        joint_rows = self.splice_stagger.reduce_aligned_splices(joint_rows, cfg)
        splice_stagger_report = self.splice_stagger.validate_splice_alignment(joint_rows, cfg)

        cutting_rows = [
            {
                "cut_length_mm": k,
                "quantity": v,
                "total_length_mm": k * v,
            }
            for k, v in sorted(cut_counter.items(), reverse=True)
        ]

        bins = self._pack_cuts_best_fit(cut_lengths, stick_len, kerf)

        blank_plan: List[Dict] = []

        for idx, cuts in enumerate(bins, 1):
            used = sum(cuts) + max(0, len(cuts) - 1) * kerf

            blank_plan.append(
                {
                    "blank_stick_index": idx,
                    "cuts_mm": ";".join(f"{c:.1f}" for c in cuts),
                    "n_cuts": len(cuts),
                    "used_length_mm_including_kerf": used,
                    "waste_length_mm": max(0.0, stick_len - used),
                }
            )

        blank = len(bins)
        extra = math.ceil(blank * waste)
        total = blank + extra

        installed_stick_mass = sum(float(r["mass_g"]) for r in stick_rows)
        wet_glue_mass = (total_glue_area / 1_000_000.0) * glue_spread / max(glue_eff, 1.0e-6)
        cured_glue_mass = wet_glue_mass * glue_cure_solids_fraction
        evaporated_glue_water = max(0.0, wet_glue_mass - cured_glue_mass)

        purchased_blank_sticks_needed = total
        purchased_stick_mass = purchased_blank_sticks_needed * stick_mass
        cutting_scrap_mass = max(0.0, purchased_stick_mass - installed_stick_mass)

        competition_mass = installed_stick_mass + cured_glue_mass
        assembly_procurement_mass = purchased_stick_mass + wet_glue_mass

        mass_limits = resolve_mass_limits(cfg)
        limit = float(mass_limits["effective_limit_g"])
        mat = cfg.get("material", {}) or {}
        planner = cfg.get("planner", {}) or {}
        stick_budget_g = safe_float(mat.get("stick_budget_g"), safe_float(planner.get("target_installed_stick_mass_g"), 900.0)) or 900.0
        wet_glue_budget_g = safe_float(mat.get("wet_glue_budget_g"), safe_float(planner.get("target_wet_glue_mass_g"), 100.0)) or 100.0
        nominal_competition_limit_g = safe_float(
            mat.get("nominal_competition_limit_g"),
            safe_float(mass_limits.get("nominal_limit_g"), 1000.0),
        ) or 1000.0
        glue_acceptance_fs = safe_float(
            cfg.get("analysis", {}).get("acceptance_min_glue_fs"),
            1.5,
        ) or 1.5
        weak_glue_count = 0
        for r in (joint_rows or []):
            fs_joint = safe_float(r.get("FS_glue_shear"), None)
            if fs_joint is not None and fs_joint < glue_acceptance_fs:
                weak_glue_count += 1

        summary = {
            "total_members": len(member_rows),
            "total_piece_instances": total_pieces,
            "total_cut_length_mm": total_cut,
            "estimated_blank_sticks_needed": blank,
            "waste_factor": waste,
            "extra_sticks_for_waste": extra,
            "estimated_total_sticks_with_waste": total,
            "estimated_piece_mass_g_without_waste_scaling": installed_stick_mass,
            "installed_stick_mass_g": installed_stick_mass,
            "purchased_blank_sticks_needed": purchased_blank_sticks_needed,
            "purchased_stick_mass_g": purchased_stick_mass,
            "cutting_scrap_mass_g": cutting_scrap_mass,
            "estimated_glue_area_mm2": total_glue_area,
            "estimated_glue_mass_g": wet_glue_mass,
            "wet_glue_mass_g": wet_glue_mass,
            "glue_cure_solids_fraction": glue_cure_solids_fraction,
            "cured_glue_mass_g": cured_glue_mass,
            "evaporated_glue_water_g": evaporated_glue_water,
            "competition_mass_g": competition_mass,
            "assembly_procurement_mass_g": assembly_procurement_mass,
            # Backward compatibility: old "total mass" now maps to final competition mass.
            "estimated_total_mass_g": competition_mass,
            "mass_limit_g": limit,
            "mass_margin_g": limit - competition_mass,
            "mass_limit_nominal_g": float(mass_limits["nominal_limit_g"]),
            "mass_limit_material_g": mass_limits["material_limit_g"],
            "mass_limit_planner_g": mass_limits["planner_limit_g"],
            "mass_limit_effective_g": float(mass_limits["effective_limit_g"]),
            "mass_limit_effective_source": str(mass_limits["effective_source"]),
            "stick_budget_g": stick_budget_g,
            "wet_glue_budget_g": wet_glue_budget_g,
            "stick_budget_margin_g": stick_budget_g - installed_stick_mass,
            "wet_glue_budget_margin_g": wet_glue_budget_g - wet_glue_mass,
            "nominal_competition_limit_g": nominal_competition_limit_g,
            "competition_mass_margin_g": nominal_competition_limit_g - competition_mass,
            "n_weak_glue_joints": weak_glue_count,
            "glue_shear_strength_MPa": glue_tau,
            "glue_safety_factor": glue_sf,
            "cut_increment_mm": cut_increment_mm,
            "allow_cut_rounding": allow_cut_rounding,
            "max_cut_length_mm": max_cut_length_mm,
            "strict_cut_length": strict_cut_length,
            "stock_limit_splits": stock_limit_splits,
            "oversize_piece_count": int(sum(1 for r in stick_rows if (safe_float(r.get("cut_length_mm"), 0.0) or 0.0) > max_cut_length_mm + 1.0e-9)),
        }

        weakest = sorted(
            member_rows,
            key=lambda r: safe_sort_key(r.get("FS_min_global")),
        )[:30]

        glue_weak = sorted(
            joint_rows,
            key=lambda r: safe_sort_key(r.get("FS_glue_shear")),
        )[:30]

        exports = {
            "stick_pieces.csv": stick_rows,
            "glue_joints.csv": joint_rows,
            "member_detail_checks.csv": member_rows,
            "cutting_list.csv": cutting_rows,
            "blank_cut_plan.csv": blank_plan,
            "reinforcement_suggestions.csv": reinf_rows,
            "weakest_members.csv": weakest,
            "weakest_glue_joints.csv": glue_weak,
        }

        for filename, rows in exports.items():
            GeometryService.write_csv(out / filename, rows)

        (out / "detailed_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out / "splice_stagger_report.json").write_text(
            json.dumps(splice_stagger_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        oversize_rows = [
            r for r in stick_rows
            if (safe_float(r.get("cut_length_mm"), 0.0) or 0.0) > max_cut_length_mm + 1.0e-9
        ]
        (out / "09_auditoria_conectividade_e_cortes.md").write_text(
            "# Auditoria de cortes, dimensões e conectividade\n\n"
            f"- Comprimento máximo permitido por palito: **{max_cut_length_mm:.1f} mm**.\n"
            f"- Incremento de corte usado: **{cut_increment_mm:.1f} mm**.\n"
            f"- Peças repartidas por excederem o palito real após stagger: **{stock_limit_splits}**.\n"
            f"- Peças acima do limite após correção: **{len(oversize_rows)}**.\n\n"
            "Nenhum corte exportado deve exigir palito maior que o lote real. "
            "Quando o desencontro de emendas cria uma peça longa demais, ela é repartida com sobreposição.\n",
            encoding="utf-8",
        )

        return {
            "stick_pieces": stick_rows,
            "glue_joints": joint_rows,
            "member_detail_checks": member_rows,
            "cutting_list": cutting_rows,
            "blank_cut_plan": blank_plan,
            "reinforcement_suggestions": reinf_rows,
            "weakest_members": weakest,
            "weakest_glue_joints": glue_weak,
            "splice_stagger_report": splice_stagger_report,
            "summary": summary,
        }
