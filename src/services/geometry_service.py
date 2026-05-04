from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from src.domain.models import Load, Member, Node, Support
from src.services.section_service import SectionService


class GeometryService:
    """Geração da geometria 3D da ponte. Não resolve estrutura."""

    def __init__(self, section_service: SectionService | None = None) -> None:
        self.sections = section_service or SectionService()

    def x_stations(self, cfg: Dict) -> List[float]:
        b = cfg["bridge"]
        left = -float(b["left_support_overhang_mm"])
        span = float(b["span_mm"])
        right = span + float(b["right_support_overhang_mm"])
        p = float(b["panel_mm"])
        xs = [left]
        x = 0.0
        while x <= span + 1e-9:
            xs.append(round(x, 6))
            x += p
        xs.append(right)
        return sorted(set(xs))

    def top_height(self, cfg: Dict, x: float) -> float:
        b = cfg["bridge"]
        span = float(b["span_mm"])
        end_h = float(b["end_height_mm"])
        center_h = float(b["center_height_mm"])
        profile = str(b.get("top_profile", "parker_plateau")).lower()
        x = max(0.0, min(span, float(x)))
        if profile == "flat":
            return center_h
        if profile == "triangular_peak":
            mid = span / 2.0
            return end_h + (center_h-end_h)*(x/mid) if x <= mid else center_h + (end_h-center_h)*((x-mid)/(span-mid))
        if profile == "shallow_arch":
            xi = (x - span/2.0) / max(1e-9, span/2.0)
            return end_h + (center_h-end_h) * max(0.0, 1.0 - xi*xi)
        p0 = float(b["plateau_start_mm"]); p1 = float(b["plateau_end_mm"])
        if p0 <= x <= p1: return center_h
        if x < p0: return end_h + (center_h-end_h)*(x/p0 if p0 else 1.0)
        return center_h + (end_h-center_h)*((x-p1)/(span-p1) if span != p1 else 1.0)

    def _add_side_diagonal(self, cfg: Dict, idx: int, x0: float, x1: float, y: float, mid: float, nid, add_member) -> None:
        typ = str(cfg["bridge"].get("side_truss_type", cfg["bridge"].get("truss_type", "Parker"))).lower()
        if typ == "parker": typ = "pratt"
        c = 0.5 * (x0 + x1)
        if typ == "howe":
            add_member(nid(x0, y, "bottom") if c <= mid else nid(x0, y, "top"), nid(x1, y, "top") if c <= mid else nid(x1, y, "bottom"), "diagonal")
        elif typ == "warren":
            add_member(nid(x0, y, "bottom") if idx % 2 == 0 else nid(x0, y, "top"), nid(x1, y, "top") if idx % 2 == 0 else nid(x1, y, "bottom"), "diagonal")
        elif typ in {"x", "duplo_x", "double_x"}:
            add_member(nid(x0, y, "bottom"), nid(x1, y, "top"), "diagonal")
            add_member(nid(x0, y, "top"), nid(x1, y, "bottom"), "diagonal")
        else:
            add_member(nid(x0, y, "top") if c <= mid else nid(x0, y, "bottom"), nid(x1, y, "bottom") if c <= mid else nid(x1, y, "top"), "diagonal")


    def _add_plane_bracing(self, mode: str, idx: int, x0: float, x1: float, ys: List[float], level: str, nid, add_member, group: str) -> None:
        mode = str(mode).lower()
        if mode in {"none", "sem", "nenhuma"}:
            return
        if mode in {"x", "duplo_x", "double_x"}:
            add_member(nid(x0, ys[0], level), nid(x1, ys[1], level), group)
            add_member(nid(x0, ys[1], level), nid(x1, ys[0], level), group)
        elif mode == "warren":
            add_member(nid(x0, ys[0], level) if idx % 2 == 0 else nid(x0, ys[1], level), nid(x1, ys[1], level) if idx % 2 == 0 else nid(x1, ys[0], level), group)
        elif mode == "howe":
            add_member(nid(x0, ys[1], level), nid(x1, ys[0], level), group)
        else:
            add_member(nid(x0, ys[0], level), nid(x1, ys[1], level), group)

    def generate(self, cfg: Dict) -> Tuple[List[Node], List[Member], List[Support], List[Load]]:
        nodes: List[Node] = []
        node_id_by_key: Dict[Tuple[float, float, str], int] = {}
        xs = self.x_stations(cfg)
        half_width = float(cfg["bridge"]["width_mm"]) / 2.0
        ys = [-half_width, half_width]

        def add_node(x: float, y: float, z: float, level: str) -> int:
            key = (round(float(x), 6), round(float(y), 6), level)
            if key in node_id_by_key:
                return node_id_by_key[key]
            node_id = len(nodes) + 1
            n = Node(node_id, float(x), float(y), float(z), level, "L" if y < 0 else "R", float(x))
            nodes.append(n)
            node_id_by_key[key] = node_id
            return node_id

        for x in xs:
            for y in ys:
                add_node(x, y, 0.0, "bottom")
                add_node(x, y, self.top_height(cfg, x), "top")

        node_lookup = {(n.x, n.y, n.level): n.id for n in nodes}
        node_by_id = {n.id: n for n in nodes}
        members_raw: List[Tuple[int, int, str]] = []

        def nid(x: float, y: float, level: str) -> int:
            return node_lookup[(float(x), float(y), level)]

        def add_member(i: int, j: int, group: str) -> None:
            if i == j:
                return
            ni, nj = node_by_id[i], node_by_id[j]
            if (ni.x, ni.y, ni.z) > (nj.x, nj.y, nj.z):
                i, j = j, i
            members_raw.append((i, j, group))

        for y in ys:
            for x0, x1 in zip(xs[:-1], xs[1:]):
                add_member(nid(x0, y, "bottom"), nid(x1, y, "bottom"), "bottom_chord")
                add_member(nid(x0, y, "top"), nid(x1, y, "top"), "top_chord")
            for x in xs:
                add_member(nid(x, y, "bottom"), nid(x, y, "top"), "vertical")
            mid = float(cfg["bridge"]["span_mm"]) / 2.0
            for idx_panel, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
                if x1 < 0 or x0 > float(cfg["bridge"]["span_mm"]):
                    continue
                self._add_side_diagonal(cfg, idx_panel, x0, x1, y, mid, nid, add_member)
                chord_type = str(cfg["bridge"].get("chord_truss_type", "none")).lower()
                if chord_type not in {"none", "sem", "nenhuma"}:
                    old_type = cfg["bridge"].get("side_truss_type", cfg["bridge"].get("truss_type", "Parker"))
                    cfg["bridge"]["side_truss_type"] = chord_type
                    self._add_side_diagonal(cfg, idx_panel, x0, x1, y, mid, nid, lambda a,b,g: add_member(a,b,"chord_lacing"))
                    cfg["bridge"]["side_truss_type"] = old_type

        for x in xs:
            add_member(nid(x, ys[0], "bottom"), nid(x, ys[1], "bottom"), "bottom_transverse")
            add_member(nid(x, ys[0], "top"), nid(x, ys[1], "top"), "top_transverse")

        internal_type = cfg["bridge"].get("internal_truss_type", "X")
        for idx_panel, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
            if cfg["bridge"].get("include_bottom_x_bracing", True):
                self._add_plane_bracing(internal_type, idx_panel, x0, x1, ys, "bottom", nid, add_member, "bottom_bracing")
            if cfg["bridge"].get("include_top_x_bracing", True):
                self._add_plane_bracing(internal_type, idx_panel, x0, x1, ys, "top", nid, add_member, "top_bracing")

        if cfg["bridge"].get("include_cross_frame_bracing", True):
            for x in xs:
                add_member(nid(x, ys[0], "bottom"), nid(x, ys[1], "top"), "cross_frame_bracing")
                add_member(nid(x, ys[1], "bottom"), nid(x, ys[0], "top"), "cross_frame_bracing")

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
        for idx, (i, j, group) in enumerate(unique, 1):
            n_sticks = int(cfg["member_sticks_by_group"].get(group, 1))
            layout_cfg = cfg.get("section_layout_by_group", {}).get(group, {"layout": "stacked"})
            sec = self.sections.composite_section(n_sticks, mat, layout_cfg)
            L = self.sections.member_length_mm(node_by_id[i], node_by_id[j])
            k = cfg.get("effective_length_factor_by_group", {}).get(group, {})
            members.append(Member(idx, i, j, group, n_sticks, sec["A"], sec["A"], sec["A"], sec["Iy"], sec["Iz"], sec["J"], float(mat["E_MPa"]), float(mat["G_MPa"]), float(k.get("Ky", 1.0)), float(k.get("Kz", 1.0)), L))

        left_xs = set(float(v) for v in cfg["bridge"]["support_contact_x_left_mm"])
        right_xs = set(float(v) for v in cfg["bridge"]["support_contact_x_right_mm"])
        support_ys = set(float(v) for v in cfg["bridge"]["support_contact_y_mm"])
        supports: List[Support] = []
        for n in nodes:
            if n.level != "bottom" or n.y not in support_ys:
                continue
            if n.x in left_xs or n.x in right_xs:
                UX = UY = UZ = 0
                UZ = 1
                if n.x == min(left_xs) and n.y == min(support_ys):
                    UX, UY = 1, 1
                elif n.x == min(left_xs) and n.y == max(support_ys):
                    UY = 1
                elif n.x == max(right_xs) and n.y == min(support_ys):
                    UY = 1
                supports.append(Support(n.id, UX, UY, UZ, 0, 0, 0, "left" if n.x in left_xs else "right", True))

        load_total = float(cfg["bridge"]["load_total_N"])
        load_xs = [float(v) for v in cfg["bridge"]["load_distribution_x_mm"]]
        loaded_nodes = []
        for x in load_xs:
            # ajusta para estação existente mais próxima
            x_near = min(xs, key=lambda xv: abs(xv - x))
            for y in ys:
                loaded_nodes.append(nid(x_near, y, "top"))
        fz_each = -load_total / len(loaded_nodes)
        loads = [Load("LC1_carga_central_distribuida", node_id, 0.0, 0.0, fz_each) for node_id in loaded_nodes]
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
