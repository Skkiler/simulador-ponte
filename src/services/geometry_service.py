from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from src.domain.models import Load, Member, Node, Support
from src.services.section_service import SectionService
from src.services.load_distribution_service import LoadDistributionService


class GeometryService:
    """Geração da geometria 3D da ponte. Não resolve estrutura."""

    def __init__(self, section_service: SectionService | None = None) -> None:
        self.sections = section_service or SectionService()

    @staticmethod
    def resolve_section_layout_for_member(
        cfg: Dict,
        group: str,
        x_mid: float,
        y_mid: float,
    ) -> Dict:
        """Resolve a seção por grupo e, quando informado, por estação física.

        Isto impede que montantes efetivamente construídos em sanduíche sejam
        substituídos por uma seção genérica de todo o grupo.
        """
        layout = dict(cfg.get("section_layout_by_group", {}).get(group, {"layout": "stacked"}))
        for rule in (cfg.get("section_layout_by_signature", []) or []):
            if str(rule.get("group", "")) != str(group):
                continue
            tol = float(rule.get("tolerance_mm", 1.0e-6) or 1.0e-6)
            if rule.get("x_mm") is not None and abs(float(x_mid) - float(rule["x_mm"])) > tol:
                continue
            if rule.get("y_mm") is not None and abs(float(y_mid) - float(rule["y_mm"])) > tol:
                continue
            override = rule.get("section_layout") or rule.get("layout_cfg") or {}
            if isinstance(override, dict):
                layout.update(override)
            break
        return layout

    def x_stations(self, cfg: Dict) -> List[float]:
        b = cfg["bridge"]
        left = -float(b["left_support_overhang_mm"])
        span = float(b["span_mm"])
        right = span + float(b["right_support_overhang_mm"])
        p = max(1.0, float(b["panel_mm"]))
        xs = [left]
        x = 0.0
        while x <= span + 1e-9:
            xs.append(round(x, 6))
            x += p
        if abs(xs[-1] - span) > 1e-6:
            xs.append(round(span, 6))
        xs.append(right)
        return sorted(set(xs))

    @staticmethod
    def _snap_value(value: float, candidates: List[float]) -> float:
        if not candidates:
            return float(value)

        return min(candidates, key=lambda c: abs(c - float(value)))

    @classmethod
    def _snap_many(
        cls,
        values: List[float],
        candidates: List[float],
    ) -> List[float]:
        if not values:
            return []

        return list(
            dict.fromkeys(
                round(cls._snap_value(float(v), candidates), 6)
                for v in values
            )
        )

    def top_height(self, cfg: Dict, x: float) -> float:
        b = cfg["bridge"]
        span = float(b["span_mm"])
        end_h = float(b["end_height_mm"])
        center_h = float(b["center_height_mm"])
        profile_raw = str(b.get("top_profile", "parker_plateau")).lower()
        profile_alias = {
            "plato": "parker_plateau",
            "platô": "parker_plateau",
            "pontiagudo/triangular": "triangular_peak",
            "triangular": "triangular_peak",
            "arco": "shallow_arch_faceted",
            "shallow_arch": "shallow_arch_faceted",
            "reto": "flat",
            "reta": "flat",
        }
        profile = profile_alias.get(profile_raw, profile_raw)
        x = max(0.0, min(span, float(x)))
        if profile == "flat":
            return center_h
        # Defesa adicional: um perfil não-plano com end_h == center_h vira
        # geometria reta.  A normalização deve impedir isso, mas esta camada
        # evita regressões quando GeometryService é chamado com cfg parcial.
        if center_h > 1.0 and end_h >= center_h - max(1.0, 0.02 * center_h):
            end_h = max(40.0, min(center_h - max(20.0, 0.18 * center_h), 0.35 * center_h))
        if profile == "triangular_peak":
            mid = span / 2.0
            return end_h + (center_h-end_h)*(x/mid) if x <= mid else center_h + (end_h-center_h)*((x-mid)/(span-mid))
        if profile in {"shallow_arch", "shallow_arch_faceted"}:
            xi = (x - span/2.0) / max(1e-9, span/2.0)
            return end_h + (center_h-end_h) * max(0.0, 1.0 - xi*xi)
        p0 = float(b["plateau_start_mm"]); p1 = float(b["plateau_end_mm"])
        if p0 <= x <= p1: return center_h
        if x < p0: return end_h + (center_h-end_h)*(x/p0 if p0 else 1.0)
        return center_h + (end_h-center_h)*((x-p1)/(span-p1) if span != p1 else 1.0)

    def fabricated_top_node_height(self, cfg: Dict, x: float) -> float:
        """Return centroid elevation of a top chord seated above montantes.

        ``top_height`` historically located the top-chord centroid at the end
        of each montante, causing both physical prisms to occupy the same
        volume. When the sandwich chord is glued *above* a mitered montante,
        the centroid of the chord is raised by its underside-to-centroid depth.
        """
        z = self.top_height(cfg, x)
        detail = cfg.get("detail_model", {}) or {}
        if bool(detail.get("top_chord_seated_above_vertical_enabled", False)):
            z += max(0.0, float(detail.get("top_chord_centroid_seat_raise_mm", 0.0) or 0.0))
        return z

    def _add_side_diagonal(self, truss_type: str, idx: int, x0: float, x1: float, y: float, mid: float, nid, add_member) -> None:
        typ = self._normalize_truss_mode(truss_type)
        c = 0.5 * (x0 + x1)
        if typ == "howe":
            add_member(nid(x0, y, "bottom") if c <= mid else nid(x0, y, "top"), nid(x1, y, "top") if c <= mid else nid(x1, y, "bottom"), "diagonal")
        elif typ == "howe_inverted":
            add_member(nid(x0, y, "top") if c <= mid else nid(x0, y, "bottom"), nid(x1, y, "bottom") if c <= mid else nid(x1, y, "top"), "diagonal")
        elif typ == "warren":
            add_member(nid(x0, y, "bottom") if idx % 2 == 0 else nid(x0, y, "top"), nid(x1, y, "top") if idx % 2 == 0 else nid(x1, y, "bottom"), "diagonal")
        elif typ == "warren_mid_braced":
            add_member(nid(x0, y, "bottom") if idx % 2 == 0 else nid(x0, y, "top"), nid(x1, y, "top") if idx % 2 == 0 else nid(x1, y, "bottom"), "diagonal")
            add_member(nid(x0, y, "top"), nid(x1, y, "bottom"), "diagonal")
        elif typ in {"warren_symmetric"}:
            add_member(nid(x0, y, "bottom") if idx % 2 == 0 else nid(x0, y, "top"), nid(x1, y, "top") if idx % 2 == 0 else nid(x1, y, "bottom"), "diagonal")
        elif typ in {"x", "duplo_x", "double_x"}:
            add_member(nid(x0, y, "bottom"), nid(x1, y, "top"), "diagonal")
            add_member(nid(x0, y, "top"), nid(x1, y, "bottom"), "diagonal")
        elif typ in {"pratt_symmetric"}:
            add_member(nid(x0, y, "top") if c <= mid else nid(x0, y, "bottom"), nid(x1, y, "bottom") if c <= mid else nid(x1, y, "top"), "diagonal")
        else:
            add_member(nid(x0, y, "top") if c <= mid else nid(x0, y, "bottom"), nid(x1, y, "bottom") if c <= mid else nid(x1, y, "top"), "diagonal")

    def _side_truss_uses_intermediate_verticals(self, truss_type: str) -> bool:
        """Treliça Warren pura evita montantes intermediários."""
        typ = self._normalize_truss_mode(truss_type)
        return typ not in {"warren", "warren_symmetric"}

    @staticmethod
    def _panel_mode_from_map(
        pattern_map: Dict[str, str] | Dict[int, str] | None,
        panel_idx: int,
        default_mode: str,
    ) -> str:
        if not pattern_map:
            return default_mode
        raw = pattern_map.get(str(panel_idx))
        if raw is None:
            raw = pattern_map.get(panel_idx)
        return str(raw) if raw is not None else default_mode



    @staticmethod
    def _physical_bracing_mode(cfg: Dict, mode: str, group: str) -> str:
        """Resolve modos de bracing para geometrias fisicamente montáveis.

        Um X feito por dois palitos contínuos no mesmo plano atravessa a outra
        diagonal no centro do painel.  Jogar uma diagonal para frente/trás também
        cria problema nas extremidades, pois a ponta deixa de coincidir com a
        face de colagem do nó.  Quando a política física está ativa, trocamos X
        de bracing secundário por diagonais simples alternadas, sem cruzamento.
        """
        raw = str(mode or "").strip()
        norm = GeometryService._normalize_truss_mode(raw)
        detail = cfg.get("detail_model", {}) or {}
        policy = str(detail.get("x_bracing_crossing_policy", "warren_no_crossing")).strip().lower()
        groups = set(str(v) for v in (detail.get("x_bracing_no_crossing_groups") or []))
        if norm == "x" and group in groups and policy in {
            "split_midpoint_lap_joint",
            "split_midpoint",
            "midpoint_lap",
            "x_midpoint_lap",
            "x_midpoint_lap_joint",
        }:
            return "x_midpoint_lap"
        if norm == "x" and group in groups and policy in {
            "single_diagonal_no_crossing",
            "single_diagonal",
            "convert_to_single_diagonal",
            "warren_no_crossing",
        }:
            return "warren_symmetric"
        return raw

    def _add_plane_bracing(
        self,
        mode: str,
        idx: int,
        x0: float,
        x1: float,
        ys: List[float],
        level: str,
        nid,
        add_member,
        group: str,
        mid_x: float | None = None,
    ) -> None:
        mode = self._normalize_truss_mode(mode)
        if mode == "none":
            return
        if mode == "x":
            add_member(nid(x0, ys[0], level), nid(x1, ys[1], level), group)
            add_member(nid(x0, ys[1], level), nid(x1, ys[0], level), group)
        elif mode == "x_midpoint_lap":
            # X físico montável sem atravessar palitos: as duas diagonais são
            # divididas no cruzamento e coladas em junta central palito-palito.
            # Isso preserva a função de contraventamento do X no solver, mas a
            # peça-a-peça deixa de ter duas barras contínuas ocupando o mesmo
            # volume. O nó central é real: deve ser colado e auditado como junta.
            xm = 0.5 * (float(x0) + float(x1))
            ym = 0.5 * (float(ys[0]) + float(ys[1]))
            c = nid(xm, ym, f"{level}_xlap")
            add_member(nid(x0, ys[0], level), c, group)
            add_member(c, nid(x1, ys[1], level), group)
            add_member(nid(x0, ys[1], level), c, group)
            add_member(c, nid(x1, ys[0], level), group)
        elif mode in {"warren", "warren_symmetric"}:
            add_member(nid(x0, ys[0], level) if idx % 2 == 0 else nid(x0, ys[1], level), nid(x1, ys[1], level) if idx % 2 == 0 else nid(x1, ys[0], level), group)
        elif mode == "warren_mid_braced":
            add_member(nid(x0, ys[0], level), nid(x1, ys[1], level), group)
            add_member(nid(x0, ys[1], level), nid(x1, ys[0], level), group)
        elif mode in {"pratt", "pratt_symmetric"}:
            # Em Pratt/N, a diagonal converge para o meio do vão.
            c = 0.5 * (float(x0) + float(x1))
            mid = float(mid_x) if mid_x is not None else c
            if c <= mid:
                add_member(nid(x0, ys[1], level), nid(x1, ys[0], level), group)
            else:
                add_member(nid(x0, ys[0], level), nid(x1, ys[1], level), group)
        elif mode == "howe":
            add_member(nid(x0, ys[1], level), nid(x1, ys[0], level), group)
        elif mode == "howe_inverted":
            add_member(nid(x0, ys[0], level), nid(x1, ys[1], level), group)
        else:
            add_member(nid(x0, ys[0], level), nid(x1, ys[1], level), group)

    def _add_cross_frame_bracing(self, mode: str, idx: int, x: float, ys: List[float], nid, add_member, mid_node=None) -> None:
        mode = self._normalize_truss_mode(mode)

        if mode == "none":
            return

        if mode == "x":
            add_member(nid(x, ys[0], "bottom"), nid(x, ys[1], "top"), "cross_frame_bracing")
            add_member(nid(x, ys[1], "bottom"), nid(x, ys[0], "top"), "cross_frame_bracing")
            return

        if mode == "x_midpoint_lap":
            if mid_node is None:
                # Fallback conservador se a geometria for chamada por código
                # legado que ainda não sabe criar nó central.
                if idx % 2 == 0:
                    add_member(nid(x, ys[0], "bottom"), nid(x, ys[1], "top"), "cross_frame_bracing")
                else:
                    add_member(nid(x, ys[1], "bottom"), nid(x, ys[0], "top"), "cross_frame_bracing")
                return
            ym = 0.5 * (float(ys[0]) + float(ys[1]))
            c = mid_node(float(x), ym, "cross_frame_xlap")
            add_member(nid(x, ys[0], "bottom"), c, "cross_frame_bracing")
            add_member(c, nid(x, ys[1], "top"), "cross_frame_bracing")
            add_member(nid(x, ys[1], "bottom"), c, "cross_frame_bracing")
            add_member(c, nid(x, ys[0], "top"), "cross_frame_bracing")
            return

        if mode in {"warren", "warren_symmetric"}:
            if idx % 2 == 0:
                add_member(nid(x, ys[0], "bottom"), nid(x, ys[1], "top"), "cross_frame_bracing")
            else:
                add_member(nid(x, ys[1], "bottom"), nid(x, ys[0], "top"), "cross_frame_bracing")
            return

        if mode == "warren_mid_braced":
            add_member(nid(x, ys[0], "bottom"), nid(x, ys[1], "top"), "cross_frame_bracing")
            add_member(nid(x, ys[1], "bottom"), nid(x, ys[0], "top"), "cross_frame_bracing")
            return

        if mode == "howe":
            add_member(nid(x, ys[1], "bottom"), nid(x, ys[0], "top"), "cross_frame_bracing")
            return

        if mode == "howe_inverted":
            add_member(nid(x, ys[0], "bottom"), nid(x, ys[1], "top"), "cross_frame_bracing")
            return

        # Pratt e demais modos usam diagonal inversa.
        add_member(nid(x, ys[0], "bottom"), nid(x, ys[1], "top"), "cross_frame_bracing")

    def _add_internal_longitudinal_zigzag(self, idx: int, x0: float, x1: float, ys: List[float], nid, add_member) -> None:
        """Contraventamento 3D interno, entre montantes consecutivos de lados opostos.

        O elemento não pertence aos planos horizontais dos banzos: ele liga o
        topo de um montante à base do próximo montante da lateral contrária,
        alternando lado a lado. Isso triangula a forma espacial e elimina o
        mecanismo lateral que ocorre quando cada quadro transversal é braceado
        isoladamente, sem continuidade longitudinal.
        """
        if idx % 2 == 0:
            add_member(nid(x0, ys[1], "top"), nid(x1, ys[0], "bottom"), "cross_frame_bracing")
        else:
            add_member(nid(x0, ys[0], "top"), nid(x1, ys[1], "bottom"), "cross_frame_bracing")

    def generate(self, cfg: Dict) -> Tuple[List[Node], List[Member], List[Support], List[Load]]:
        nodes: List[Node] = []
        node_id_by_key: Dict[Tuple[float, float, str], int] = {}
        xs = self.x_stations(cfg)
        half_width = float(cfg["bridge"]["width_mm"]) / 2.0
        ys = [-half_width, half_width]
        side_truss_type = str(
            cfg["bridge"].get(
                "side_truss_type",
                cfg["bridge"].get("truss_type", "Parker"),
            )
        )
        legacy_chord_truss_type = str(cfg["bridge"].get("chord_truss_type", "none"))
        legacy_chord_lacing_enabled = bool(
            cfg["bridge"].get("legacy_chord_truss_lacing_enabled", False)
        )
        top_chord_truss_type = str(
            cfg["bridge"].get(
                "top_chord_truss_type",
                cfg["bridge"].get("internal_truss_type", "X"),
            )
        )
        bottom_chord_truss_type = str(
            cfg["bridge"].get(
                "bottom_chord_truss_type",
                cfg["bridge"].get("internal_truss_type", "X"),
            )
        )
        # O contraventamento transversal interno tem padrão próprio. Quando
        # cross_frame_truss_type está configurado, ele prevalece sobre o campo
        # legado internal_truss_type para permitir a alternância esquerda/direita.
        internal_type = str(
            cfg["bridge"].get(
                "cross_frame_truss_type",
                cfg["bridge"].get("internal_truss_type", "X"),
            )
        )
        side_panel_pattern = cfg["bridge"].get("panel_side_truss_pattern", {}) or {}
        top_panel_pattern = cfg["bridge"].get("panel_top_chord_pattern", {}) or {}
        bottom_panel_pattern = cfg["bridge"].get("panel_bottom_chord_pattern", {}) or {}

        def add_node(x: float, y: float, z: float, level: str) -> int:
            key = (round(float(x), 6), round(float(y), 6), level)
            if key in node_id_by_key:
                return node_id_by_key[key]
            node_id = len(nodes) + 1
            n = Node(node_id, float(x), float(y), float(z), level, "L" if y < 0 else "R", float(x))
            nodes.append(n)
            node_id_by_key[key] = node_id
            return node_id

        span_for_node_filter = float(cfg["bridge"]["span_mm"])
        exclude_top_overhang_nodes = bool(cfg.get("bridge", {}).get("top_chord_exclude_support_overhang_panels", False))
        for x in xs:
            for y in ys:
                add_node(x, y, 0.0, "bottom")
                if (not exclude_top_overhang_nodes) or (-1.0e-6 <= float(x) <= span_for_node_filter + 1.0e-6):
                    add_node(x, y, self.fabricated_top_node_height(cfg, x), "top")

        # Identifica os nós que efetivamente recebem a carga principal. O
        # reforço de travessas do topo deve ocorrer somente nessas estações,
        # em vez de adicionar diagonais longitudinais ao plano do banzo.
        preliminary_loads = LoadDistributionService.build_nodal_loads(
            cfg,
            nodes,
            loadcase="LC1_carga_central_distribuida",
            total_N=float(cfg["bridge"]["load_total_N"]),
        )
        preliminary_node_by_id = {int(n.id): n for n in nodes}
        loaded_top_stations = {
            round(float(preliminary_node_by_id[int(ld.node_id)].x), 6)
            for ld in preliminary_loads
            if int(ld.node_id) in preliminary_node_by_id
            and str(preliminary_node_by_id[int(ld.node_id)].level) == "top"
            and abs(float(ld.Fz)) > 1.0e-12
        }

        node_lookup = {(n.x, n.y, n.level): n.id for n in nodes}
        node_by_id = {n.id: n for n in nodes}
        members_raw: List[Tuple[int, int, str]] = []

        def nid(x: float, y: float, level: str) -> int:
            key = (round(float(x), 6), round(float(y), 6), level)
            if key not in node_id_by_key:
                z = 0.0 if str(level).startswith("bottom") else self.fabricated_top_node_height(cfg, float(x))
                node_id = add_node(float(x), float(y), z, level)
                node_by_id[node_id] = nodes[-1]
                node_lookup[(float(x), float(y), level)] = node_id
            return node_id_by_key[key]

        def mid_node(x: float, y: float, level: str) -> int:
            z = 0.5 * self.fabricated_top_node_height(cfg, float(x))
            key = (round(float(x), 6), round(float(y), 6), level)
            if key not in node_id_by_key:
                node_id = add_node(float(x), float(y), z, level)
                node_by_id[node_id] = nodes[-1]
                node_lookup[(float(x), float(y), level)] = node_id
            return node_id_by_key[key]

        def add_member(i: int, j: int, group: str) -> None:
            if i == j:
                return
            ni, nj = node_by_id[i], node_by_id[j]
            if (ni.x, ni.y, ni.z) > (nj.x, nj.y, nj.z):
                i, j = j, i
            members_raw.append((i, j, group))

        side_mode = self._normalize_truss_mode(side_truss_type)
        span = float(cfg["bridge"]["span_mm"])
        left_overhang_x = -float(cfg["bridge"]["left_support_overhang_mm"])
        right_overhang_x = span + float(cfg["bridge"]["right_support_overhang_mm"])
        bridge_cfg = cfg.get("bridge", {}) or {}
        exclude_top_overhang = bool(bridge_cfg.get("top_chord_exclude_support_overhang_panels", False))
        exclude_vertical_overhang = bool(bridge_cfg.get("vertical_exclude_support_overhang_stations", False))
        exclude_top_transverse_overhang = bool(bridge_cfg.get("top_transverse_exclude_support_overhang_stations", exclude_top_overhang))
        exclude_side_diagonal_overhang = bool(bridge_cfg.get("side_diagonal_exclude_support_overhang_panels", False))

        def in_main_span(x: float) -> bool:
            return -1.0e-6 <= float(x) <= span + 1.0e-6

        def panel_in_main_span(x0: float, x1: float) -> bool:
            return in_main_span(x0) and in_main_span(x1)

        panel_modes = [
            self._normalize_truss_mode(
                self._panel_mode_from_map(side_panel_pattern, idx_panel, side_truss_type)
            )
            for idx_panel, _ in enumerate(zip(xs[:-1], xs[1:]))
        ]
        side_has_non_warren = any(
            pm not in {"warren", "warren_symmetric"} for pm in panel_modes
        )
        all_side_warren = bool(panel_modes) and all(
            pm in {"warren", "warren_symmetric"} for pm in panel_modes
        )

        for y in ys:
            for x0, x1 in zip(xs[:-1], xs[1:]):
                add_member(nid(x0, y, "bottom"), nid(x1, y, "bottom"), "bottom_chord")
                if (not exclude_top_overhang) or panel_in_main_span(x0, x1):
                    add_member(nid(x0, y, "top"), nid(x1, y, "top"), "top_chord")
            if self._side_truss_uses_intermediate_verticals(side_truss_type) or side_has_non_warren:
                for x in xs:
                    if exclude_vertical_overhang and not in_main_span(x):
                        continue
                    add_member(nid(x, y, "bottom"), nid(x, y, "top"), "vertical")
            else:
                # Warren puro: manter apenas postes de extremidade/apoio.
                for x in xs:
                    if exclude_vertical_overhang and not in_main_span(x):
                        continue
                    if (
                        abs(float(x) - left_overhang_x) <= 1.0e-6
                        or abs(float(x) - 0.0) <= 1.0e-6
                        or abs(float(x) - span) <= 1.0e-6
                        or abs(float(x) - right_overhang_x) <= 1.0e-6
                    ):
                        add_member(nid(x, y, "bottom"), nid(x, y, "top"), "vertical")
            mid = span / 2.0
            for idx_panel, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
                if x1 < 0 or x0 > span:
                    continue
                if exclude_side_diagonal_overhang and not panel_in_main_span(x0, x1):
                    continue
                side_mode_panel = self._panel_mode_from_map(
                    side_panel_pattern,
                    idx_panel,
                    side_truss_type,
                )
                self._add_side_diagonal(side_mode_panel, idx_panel, x0, x1, y, mid, nid, add_member)
                chord_type = legacy_chord_truss_type.lower()
                if legacy_chord_lacing_enabled and chord_type not in {"none", "sem", "nenhuma"}:
                    self._add_side_diagonal(
                        chord_type,
                        idx_panel,
                        x0,
                        x1,
                        y,
                        mid,
                        nid,
                        lambda a, b, g: add_member(a, b, "chord_lacing"),
                    )
            if side_mode in {"warren", "warren_symmetric"} or all_side_warren:
                # Garantir fechamento de ponta no Warren (topo e fundo conectados por diagonais).
                left_inner = [x for x in xs if 0.0 < float(x) <= span + 1.0e-6]
                right_inner = [x for x in xs if -1.0e-6 <= float(x) < span]
                if left_inner:
                    add_member(
                        nid(0.0, y, "bottom"),
                        nid(min(left_inner), y, "top"),
                        "diagonal",
                    )
                if right_inner:
                    add_member(
                        nid(span, y, "bottom"),
                        nid(max(right_inner), y, "top"),
                        "diagonal",
                    )

        for x in xs:
            add_member(nid(x, ys[0], "bottom"), nid(x, ys[1], "bottom"), "bottom_transverse")
            if (not exclude_top_transverse_overhang) or in_main_span(x):
                add_member(nid(x, ys[0], "top"), nid(x, ys[1], "top"), "top_transverse")

        for idx_panel, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
            if cfg["bridge"].get("include_bottom_x_bracing", True):
                bottom_mode_panel = self._panel_mode_from_map(
                    bottom_panel_pattern,
                    idx_panel,
                    bottom_chord_truss_type,
                )
                self._add_plane_bracing(
                    self._physical_bracing_mode(cfg, bottom_mode_panel, "bottom_bracing"),
                    idx_panel,
                    x0,
                    x1,
                    ys,
                    "bottom",
                    nid,
                    add_member,
                    "bottom_bracing",
                    mid_x=span / 2.0,
                )
            if cfg["bridge"].get("include_top_x_bracing", True):
                top_mode_panel = self._panel_mode_from_map(
                    top_panel_pattern,
                    idx_panel,
                    top_chord_truss_type,
                )
                self._add_plane_bracing(
                    self._physical_bracing_mode(cfg, top_mode_panel, "top_bracing"),
                    idx_panel,
                    x0,
                    x1,
                    ys,
                    "top",
                    nid,
                    add_member,
                    "top_bracing",
                    mid_x=span / 2.0,
                )

        if cfg["bridge"].get("include_cross_frame_bracing", True):
            detail_cfg = cfg.get("detail_model", {}) or {}
            if bool(detail_cfg.get("internal_cross_frame_zigzag_enabled", False)):
                # Diafragmas transversais nos planos dos montantes mantêm cada
                # quadro lateral esquadrejado; o zig-zag 3D subsequente liga
                # quadros consecutivos e impede racking/torsão longitudinal.
                if bool(detail_cfg.get("internal_cross_frame_station_diaphragms_enabled", True)):
                    diaphragm_interval = max(1, int(detail_cfg.get("internal_cross_frame_station_diaphragm_interval_panels", 1) or 1))
                    load_station_diaphragms = bool(detail_cfg.get("internal_cross_frame_station_diaphragm_at_loaded_top_nodes", True))
                    exclude_overhang = bool(detail_cfg.get("internal_bracing_exclude_support_overhang_panels", True))
                    for idx_x, x in enumerate(xs):
                        in_structural_span = (not exclude_overhang) or (-1.0e-6 <= float(x) <= float(span) + 1.0e-6)
                        keep_station = in_structural_span and ((idx_x % diaphragm_interval == 0) or round(float(x), 6) in {0.0, round(float(span), 6)})
                        if in_structural_span and load_station_diaphragms and round(float(x), 6) in loaded_top_stations:
                            keep_station = True
                        if keep_station:
                            self._add_cross_frame_bracing("Warren_symmetric", idx_x, x, ys, nid, add_member, mid_node=mid_node)
                for idx_panel, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
                    if bool(detail_cfg.get("internal_longitudinal_bracing_exclude_support_overhang_panels", False)) and (float(x0) < -1.0e-6 or float(x1) > float(span) + 1.0e-6):
                        continue
                    self._add_internal_longitudinal_zigzag(idx_panel, x0, x1, ys, nid, add_member)
            else:
                for idx_x, x in enumerate(xs):
                    self._add_cross_frame_bracing(
                        self._physical_bracing_mode(cfg, internal_type, "cross_frame_bracing"),
                        idx_x,
                        x,
                        ys,
                        nid,
                        add_member,
                        mid_node=mid_node,
                    )

        if cfg["bridge"].get("include_support_pad_members", True):
            for y in ys:
                add_member(nid(-float(cfg["bridge"]["left_support_overhang_mm"]), y, "bottom"), nid(0.0, y, "bottom"), "support_pad")
                span = float(cfg["bridge"]["span_mm"])
                add_member(nid(span, y, "bottom"), nid(span + float(cfg["bridge"]["right_support_overhang_mm"]), y, "bottom"), "support_pad")

        seen = set()
        unique = []
        for i, j, g in members_raw:
            key = (i, j, g)
            if key not in seen:
                seen.add(key)
                unique.append(key)

        mat = cfg["material"]
        members: List[Member] = []
        member_sticks_by_id = cfg.get("member_sticks_by_id", {}) or {}
        member_sticks_by_group = cfg.get("member_sticks_by_group", {}) or {}
        member_active_by_id = cfg.get("member_active_by_id", {}) or {}
        disabled_member_ids = {
            int(v)
            for v in (cfg.get("disabled_member_ids", []) or [])
            if str(v).strip()
        }
        for idx, (i, j, group) in enumerate(unique, 1):
            enabled = bool(
                member_active_by_id.get(
                    str(idx),
                    member_active_by_id.get(idx, True),
                )
            )
            if (idx in disabled_member_ids) or (not enabled):
                continue
            has_id_override = (str(idx) in member_sticks_by_id) or (idx in member_sticks_by_id)
            x_mid = 0.5 * (float(node_by_id[i].x) + float(node_by_id[j].x))
            y_mid = 0.5 * (float(node_by_id[i].y) + float(node_by_id[j].y))
            n_sticks = int(
                member_sticks_by_id.get(
                    str(idx),
                    member_sticks_by_id.get(idx, member_sticks_by_group.get(group, 1)),
                )
            )
            # Overrides semânticos sobrevivem a alterações de enumeração da
            # malha: reforçam o grupo e a estação, nunca um ID que possa passar
            # a representar uma diagonal após mudar o painel. IDs explícitos,
            # quando presentes, continuam tendo precedência para compatibilidade.
            if not has_id_override:
                for rule in (cfg.get("member_sticks_by_signature", []) or []):
                    if str(rule.get("group", "")) != str(group):
                        continue
                    tol = float(rule.get("tolerance_mm", 1.0e-6) or 1.0e-6)
                    if rule.get("x_mm") is not None and abs(x_mid - float(rule["x_mm"])) > tol:
                        continue
                    if rule.get("y_mm") is not None and abs(y_mid - float(rule["y_mm"])) > tol:
                        continue
                    n_sticks = int(rule.get("n_sticks", n_sticks))
                    break
            # No banzo superior, os travamentos são travessas simples. Nas
            # estações que recebem a placa/carga, sua seção é reforçada de modo
            # explícito no próprio membro e, portanto, entra em A/I/massa/FS.
            detail_cfg = cfg.get("detail_model", {}) or {}
            if (
                str(group) == "top_transverse"
                and bool(detail_cfg.get("loaded_top_transverse_reinforcement_enabled", False))
                and abs(float(node_by_id[i].x) - float(node_by_id[j].x)) <= 1.0e-6
                and round(float(node_by_id[i].x), 6) in loaded_top_stations
            ):
                extra = max(0, int(detail_cfg.get("loaded_top_transverse_extra_sticks", 1) or 0))
                minimum = max(1, int(detail_cfg.get("loaded_top_transverse_min_sticks", 2) or 2))
                maximum = max(minimum, int(detail_cfg.get("loaded_top_transverse_max_sticks", 3) or 3))
                n_sticks = min(maximum, max(minimum, int(n_sticks) + extra))
            layout_cfg = self.resolve_section_layout_for_member(cfg, group, x_mid, y_mid)
            sandwich_layouts = {
                "closed_sandwich_4core_2caps",
                "closed_sandwich_4core_2caps_2covers",
                "closed_face_sandwich_6",
                "closed_face_sandwich_8",
            }
            if str(layout_cfg.get("layout", "")).strip().lower() in sandwich_layouts:
                # Uma seção sanduíche construída não pode ser deslaminada pelo
                # sizing automático para 5/7 palitos: o núcleo+capas é uma
                # unidade física. Reforços adicionais só entram em pares
                # simétricos, mantendo a seção fechada e centrada.
                n_sticks = max(6, int(n_sticks))
                if n_sticks % 2 != 0:
                    n_sticks += 1
            # Projeto fabricável: seções box ímpares com 5+ palitos geram
            # arranjos difíceis de reproduzir e alteram inércia conforme a
            # montagem. Para os grupos críticos definidos no detalhamento,
            # arredonda para a seção box par imediatamente superior.
            simple_even_groups = {
                str(v)
                for v in (cfg.get("detail_model", {}) or {}).get(
                    "simple_even_box_section_groups",
                    ["top_chord", "vertical"],
                )
            }
            if (
                str(group) in simple_even_groups
                and str(layout_cfg.get("layout", "")).strip().lower() == "box"
                and int(n_sticks) >= 5
                and int(n_sticks) % 2 == 1
            ):
                n_sticks += 1
            layout_cfg.setdefault(
                "composite_action",
                cfg.get("detail_model", {}).get("composite_action", {}),
            )
            sec = self.sections.composite_section(n_sticks, mat, layout_cfg)
            L = self.sections.member_length_mm(node_by_id[i], node_by_id[j])
            k = cfg.get("effective_length_factor_by_group", {}).get(group, {})
            members.append(Member(idx, i, j, group, n_sticks, sec["A"], sec["A"], sec["A"], sec["Iy"], sec["Iz"], sec["J"], float(mat["E_MPa"]), float(mat["G_MPa"]), float(k.get("Ky", 1.0)), float(k.get("Kz", 1.0)), L, str(sec.get("layout", layout_cfg.get("layout", "stacked"))), str(sec.get("stick_orientation", layout_cfg.get("stick_orientation", "flat")))))

        left_xs_raw = [
            float(v)
            for v in cfg["bridge"].get("support_contact_x_left_mm", [])
        ]
        right_xs_raw = [
            float(v)
            for v in cfg["bridge"].get("support_contact_x_right_mm", [])
        ]
        support_ys_raw = [
            float(v)
            for v in cfg["bridge"].get("support_contact_y_mm", [])
        ]

        left_xs = set(self._snap_many(left_xs_raw, xs))
        right_xs = set(self._snap_many(right_xs_raw, xs))
        support_ys = set(self._snap_many(support_ys_raw, ys))
        left_min = min(left_xs) if left_xs else None
        right_max = max(right_xs) if right_xs else None
        y_min = min(support_ys) if support_ys else None
        y_max = max(support_ys) if support_ys else None

        supports: List[Support] = []

        for n in nodes:
            if n.level != "bottom" or n.y not in support_ys:
                continue
            if n.x in left_xs or n.x in right_xs:
                UX = UY = UZ = 0
                UZ = 1
                if left_min is not None and y_min is not None and n.x == left_min and n.y == y_min:
                    UX, UY = 1, 1
                elif left_min is not None and y_max is not None and n.x == left_min and n.y == y_max:
                    UY = 1
                elif right_max is not None and y_min is not None and n.x == right_max and n.y == y_min:
                    UY = 1
                supports.append(
                    Support(
                        n.id,
                        UX,
                        UY,
                        UZ,
                        0,
                        0,
                        0,
                        "left" if n.x in left_xs else "right",
                        True,
                    )
                )

        if not supports:
            raise ValueError(
                "Nenhum apoio foi criado. Verifique limites de contato dos apoios "
                "e a largura da ponte."
            )

        load_total = float(cfg["bridge"]["load_total_N"])
        loads = list(preliminary_loads)
        if not loads:
            # Defensive fallback: preserve a central nodal load if the selected
            # level has no nodes after an aggressive topology mutation.
            load_level = LoadDistributionService.load_level(cfg)
            mid_x = self._snap_value(float(cfg["bridge"]["span_mm"]) / 2.0, xs)
            loaded_nodes = sorted(nid(mid_x, y, load_level) for y in ys)
            fz_each = -load_total / max(1, len(loaded_nodes))
            loads = [
                Load("LC1_carga_central_distribuida", node_id, 0.0, 0.0, fz_each)
                for node_id in loaded_nodes
            ]

        return nodes, members, supports, loads

    @staticmethod
    def rows(items: list) -> list[dict]:
        return [item.__dict__.copy() for item in items]

    @classmethod
    def write_csv(cls, path: str | Path, rows: list[dict]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            p.write_text("", encoding="utf-8")
            return
        with open(p, "w", newline="", encoding="utf-8") as f:
            fieldnames = []
            seen_fields = set()
            for row in rows:
                for key in row.keys():
                    if key not in seen_fields:
                        seen_fields.add(key)
                        fieldnames.append(key)
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def export_csvs(self, cfg: Dict, out_dir: str | Path) -> None:
        nodes, members, supports, loads = self.generate(cfg)
        out = Path(out_dir)
        self.write_csv(out / "nodes.csv", self.rows(nodes))
        self.write_csv(out / "members.csv", self.rows(members))
        self.write_csv(out / "supports.csv", self.rows(supports))
        self.write_csv(out / "loads.csv", self.rows(loads))
    @staticmethod
    def _normalize_truss_mode(mode: str) -> str:
        raw = str(mode or "").strip().lower()
        alias = {
            "parker": "pratt",
            "baltimore": "pratt",
            "k": "x",
            "n": "pratt",
            "duplo_x": "x",
            "double_x": "x",
            "pratt_symmetric": "pratt_symmetric",
            "pratt simétrica": "pratt_symmetric",
            "warren_symmetric": "warren_symmetric",
            "warren simétrica": "warren_symmetric",
            "warren_mid_braced": "warren_mid_braced",
            "warren intermediária": "warren_mid_braced",
            "warren intermedia": "warren_mid_braced",
            "howe_inverted": "howe_inverted",
            "howe invertida": "howe_inverted",
            "k_symmetric": "x",
            "k simétrica": "x",
            "split_midpoint_lap_joint": "x_midpoint_lap",
            "x_midpoint_lap_joint": "x_midpoint_lap",
            "midpoint_lap": "x_midpoint_lap",
            "sem": "none",
            "nenhuma": "none",
        }
        return alias.get(raw, raw)
