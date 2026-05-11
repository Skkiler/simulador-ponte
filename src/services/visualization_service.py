from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.core.numeric import safe_float, safe_sort_key
from src.domain.models import Load, Member, Node, Support


def safe_abs_float(value: Any, default: float = 0.0) -> float:
    """
    Valor absoluto seguro.
    """
    v = safe_float(value, None)
    return default if v is None else abs(v)


class VisualizationService:
    """Geração de figuras estáticas e interativas."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _setup_3d(ax, title: str, nodes: List[Node] | None = None) -> None:
        ax.set_title(title)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        ax.view_init(elev=20, azim=-60)

        try:
            if nodes:
                xs = [n.x for n in nodes]
                ys = [n.y for n in nodes]
                zs = [n.z for n in nodes]

                x_span = max(max(xs) - min(xs), 1.0)
                y_span = max(max(ys) - min(ys), 1.0)
                z_span = max(max(zs) - min(zs), 1.0)

                # Mantém proporção física aproximada nas figuras estáticas.
                ax.set_box_aspect((x_span, y_span, z_span))
            else:
                ax.set_box_aspect((1400, 260, 340))
        except (TypeError, ValueError):
            ax.set_box_aspect((1400, 260, 340))

    def save_all(
        self,
        nodes: List[Node],
        members: List[Member],
        supports: List[Support],
        loads: List[Load],
        node_results: List[Dict],
        member_results: List[Dict],
        member_checks: List[Dict],
        support_checks: List[Dict],
        out_dir: str | Path,
        deformed_scale: float = 30.0,
    ) -> List[Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: List[Path] = []

        paths.append(
            self.plot_geometry_3d(
                nodes,
                members,
                supports,
                loads,
                out / "01_geometria_3d.png",
            )
        )

        paths.append(
            self.plot_side(
                nodes,
                members,
                "L",
                out / "02_trelica_lateral_esquerda.png",
            )
        )

        paths.append(
            self.plot_side(
                nodes,
                members,
                "R",
                out / "03_trelica_lateral_direita.png",
            )
        )

        paths.append(
            self.plot_plan(
                nodes,
                members,
                "top",
                out / "04_plano_superior.png",
            )
        )

        paths.append(
            self.plot_plan(
                nodes,
                members,
                "bottom",
                out / "05_plano_inferior.png",
            )
        )

        paths.append(
            self.plot_axial(
                nodes,
                members,
                member_results,
                out / "06_esforcos_axiais_todos.png",
                primary_only=False,
            )
        )

        paths.append(
            self.plot_axial(
                nodes,
                members,
                member_results,
                out / "07_esforcos_axiais_principais.png",
                primary_only=True,
            )
        )

        paths.append(
            self.plot_deformed(
                nodes,
                members,
                node_results,
                out / "08_forma_deformada.png",
                scale=deformed_scale,
            )
        )

        paths.append(
            self.plot_failure_ranking(
                member_checks,
                out / "09_ranking_falha_principal.png",
            )
        )

        paths.append(
            self.plot_supports(
                support_checks,
                out / "10_reacoes_apoio.png",
            )
        )

        html = self.plotly_geometry(nodes, members, supports, loads)
        html.write_html(out / "geometria_3d_interativa.html")
        paths.append(out / "geometria_3d_interativa.html")

        return paths

    def plot_geometry_3d(self, nodes, members, supports, loads, path):
        node_by_id = {n.id: n for n in nodes}

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection="3d")

        for m in members:
            ni, nj = node_by_id[m.i], node_by_id[m.j]

            ax.plot(
                [ni.x, nj.x],
                [ni.y, nj.y],
                [ni.z, nj.z],
                linewidth=0.8,
                alpha=0.75,
            )

        support_ids = {s.node_id for s in supports if s.active_vertical}
        load_ids = {l.node_id for l in loads}

        if support_ids:
            ax.scatter(
                [node_by_id[i].x for i in support_ids],
                [node_by_id[i].y for i in support_ids],
                [node_by_id[i].z for i in support_ids],
                marker="^",
                s=50,
                label="apoios ativos",
            )

        if load_ids:
            ax.scatter(
                [node_by_id[i].x for i in load_ids],
                [node_by_id[i].y for i in load_ids],
                [node_by_id[i].z for i in load_ids],
                marker="v",
                s=50,
                label="cargas",
            )

        self._setup_3d(ax, "Geometria 3D", nodes)
        ax.legend()

        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)

        return Path(path)

    def plot_side(self, nodes, members, side, path):
        node_by_id = {n.id: n for n in nodes}

        fig, ax = plt.subplots(figsize=(12, 5))

        y_target = min(n.y for n in nodes) if side == "L" else max(n.y for n in nodes)

        for m in members:
            ni, nj = node_by_id[m.i], node_by_id[m.j]

            if abs(ni.y - y_target) < 1.0e-9 and abs(nj.y - y_target) < 1.0e-9:
                ax.plot(
                    [ni.x, nj.x],
                    [ni.z, nj.z],
                    linewidth=1.0,
                )

        ax.set_title("Treliça lateral esquerda" if side == "L" else "Treliça lateral direita")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        ax.axis("equal")

        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)

        return Path(path)

    def plot_plan(self, nodes, members, level, path):
        node_by_id = {n.id: n for n in nodes}

        fig, ax = plt.subplots(figsize=(12, 5))

        for m in members:
            ni, nj = node_by_id[m.i], node_by_id[m.j]

            if ni.level == level and nj.level == level:
                ax.plot(
                    [ni.x, nj.x],
                    [ni.y, nj.y],
                    linewidth=1.0,
                )

        ax.set_title("Plano superior" if level == "top" else "Plano inferior")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.axis("equal")

        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)

        return Path(path)

    def plot_axial(self, nodes, members, results, path, primary_only=False):
        node_by_id = {n.id: n for n in nodes}
        res = {int(r["member_id"]): r for r in results if "member_id" in r}

        maxN = max([safe_abs_float(r.get("N_N"), 0.0) for r in results] or [1.0])
        maxN = max(maxN, 1.0e-9)

        primary = {
            "bottom_chord",
            "top_chord",
            "vertical",
            "diagonal",
            "top_transverse",
            "bottom_transverse",
            "support_pad",
        }

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection="3d")

        for m in members:
            if primary_only and m.group not in primary:
                continue

            r = res.get(m.id, {})
            N = safe_float(r.get("N_N"), 0.0) or 0.0

            ni, nj = node_by_id[m.i], node_by_id[m.j]

            lw = 0.5 + 3.5 * abs(N) / maxN

            ax.plot(
                [ni.x, nj.x],
                [ni.y, nj.y],
                [ni.z, nj.z],
                linewidth=lw,
                linestyle="-" if N >= 0 else "--",
                alpha=0.85,
            )

        self._setup_3d(
            ax,
            "Esforços axiais principais" if primary_only else "Esforços axiais - todos os membros",
            nodes,
        )

        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)

        return Path(path)

    def plot_deformed(self, nodes, members, node_results, path, scale=30.0):
        node_by_id = {n.id: n for n in nodes}
        disp = {int(r["node_id"]): r for r in node_results if "node_id" in r}

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection="3d")

        for m in members:
            ni, nj = node_by_id[m.i], node_by_id[m.j]

            ax.plot(
                [ni.x, nj.x],
                [ni.y, nj.y],
                [ni.z, nj.z],
                linewidth=0.7,
                alpha=0.25,
            )

            ri = disp.get(m.i, {})
            rj = disp.get(m.j, {})

            xi = ni.x + scale * (safe_float(ri.get("Ux_mm"), 0.0) or 0.0)
            yi = ni.y + scale * (safe_float(ri.get("Uy_mm"), 0.0) or 0.0)
            zi = ni.z + scale * (safe_float(ri.get("Uz_mm"), 0.0) or 0.0)

            xj = nj.x + scale * (safe_float(rj.get("Ux_mm"), 0.0) or 0.0)
            yj = nj.y + scale * (safe_float(rj.get("Uy_mm"), 0.0) or 0.0)
            zj = nj.z + scale * (safe_float(rj.get("Uz_mm"), 0.0) or 0.0)

            ax.plot(
                [xi, xj],
                [yi, yj],
                [zi, zj],
                linewidth=1.0,
                alpha=0.9,
            )

        self._setup_3d(ax, f"Forma deformada - escala {scale:g}x", nodes)

        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)

        return Path(path)

    def plot_failure_ranking(self, checks, path, top_n=25):
        """
        Plota ranking de falha sem quebrar quando FS_min vier None/vazio/texto.

        Prioriza membros primários. Se não houver primários com FS válido,
        usa todos os membros com FS válido.
        """
        rows_all = list(checks or [])

        primary_rows = [r for r in rows_all if r.get("member_role") == "primary"]
        candidate_rows = primary_rows or rows_all

        valid_rows = [
            r
            for r in candidate_rows
            if safe_float(r.get("FS_min"), None) is not None
        ]

        # Se a filtragem por primários removeu tudo, tenta todos os membros.
        if not valid_rows and primary_rows:
            valid_rows = [
                r
                for r in rows_all
                if safe_float(r.get("FS_min"), None) is not None
            ]

        rows = sorted(
            valid_rows,
            key=lambda r: safe_sort_key(r.get("FS_min")),
        )[:top_n]

        fig, ax = plt.subplots(figsize=(12, 8))

        if not rows:
            ax.text(
                0.5,
                0.5,
                "Sem fatores de segurança numéricos para plotar.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(path, dpi=220)
            plt.close(fig)
            return Path(path)

        labels = [
            f"{r.get('member_id', '?')} {r.get('group', '')}"
            for r in rows
        ]

        values = [
            safe_float(r.get("FS_min"), 0.0) or 0.0
            for r in rows
        ]

        ax.barh(labels, values)
        ax.axvline(1.0, linestyle="--", linewidth=1.0)
        ax.axvline(2.0, linestyle=":", linewidth=1.0)

        ax.set_xlabel("Fator de segurança mínimo")
        ax.set_title("Ranking de membros principais críticos")
        ax.invert_yaxis()

        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)

        return Path(path)

    def plot_supports(self, checks, path):
        rows = list(checks or [])

        fig, ax = plt.subplots(figsize=(10, 5))

        if not rows:
            ax.text(
                0.5,
                0.5,
                "Sem reações de apoio para plotar.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(path, dpi=220)
            plt.close(fig)
            return Path(path)

        labels = [
            f"{int(safe_float(r.get('node_id'), 0) or 0)}\nx={safe_float(r.get('x_mm'), 0.0) or 0.0:.0f}"
            for r in rows
        ]

        values = [
            safe_float(r.get("reaction_Z_kgf"), 0.0) or 0.0
            for r in rows
        ]

        ax.bar(labels, values)

        ax.set_ylabel("Reação vertical [kgf]")
        ax.set_title("Reações verticais nos apoios ativos")

        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)

        return Path(path)

    def plotly_geometry(
        self,
        nodes,
        members,
        supports,
        loads,
        highlight_member_ids: Iterable[int] | None = None,
        *,
        highlight_member_colors: Dict[int, str] | None = None,
        scale_mode: str = "real",
        color_mode: str = "group",
        member_results: List[Dict] | None = None,
        member_checks: List[Dict] | None = None,
        selected_member_ids: Iterable[int] | None = None,
        highlight_selected: bool = True,
    ) -> go.Figure:
        """
        Gera visualização 3D interativa.

        scale_mode:
            "real"      -> proporção real: x muito maior que y/z.
            "didactic"  -> leve exagero de y/z para leitura visual.
            "cube"      -> cubo Plotly, útil só para inspecionar conexões.
        """
        node_by_id = {n.id: n for n in nodes}
        selected_ids = {int(v) for v in (selected_member_ids or [])}
        selected_ids |= {int(v) for v in (highlight_member_ids or [])}
        color_mode = str(color_mode or "group").strip().lower()
        res_map = {
            int(r.get("member_id")): r
            for r in (member_results or [])
            if r.get("member_id") is not None
        }
        chk_map = {
            int(r.get("member_id")): r
            for r in (member_checks or [])
            if r.get("member_id") is not None
        }

        xs_all = [n.x for n in nodes]
        ys_all = [n.y for n in nodes]
        zs_all = [n.z for n in nodes]

        x_min, x_max = min(xs_all), max(xs_all)
        y_min, y_max = min(ys_all), max(ys_all)
        z_min, z_max = min(zs_all), max(zs_all)

        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)
        z_span = max(z_max - z_min, 1.0)

        if scale_mode == "real":
            aspectmode = "manual"
            aspectratio = {
                "x": 1.0,
                "y": y_span / x_span,
                "z": z_span / x_span,
            }
        elif scale_mode == "didactic":
            aspectmode = "manual"
            aspectratio = {
                "x": 1.0,
                "y": max(y_span / x_span, 0.22),
                "z": max(z_span / x_span, 0.32),
            }
        else:
            aspectmode = "cube"
            aspectratio = None

        fig = go.Figure()

        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        groups = sorted(set(m.group for m in members))
        group_color = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

        max_abs_force = max(
            [safe_abs_float((res_map.get(int(m.id), {}) or {}).get("N_N"), 0.0) for m in members] or [1.0]
        )
        max_abs_force = max(max_abs_force, 1.0e-9)
        max_util = max(
            [safe_float((chk_map.get(int(m.id), {}) or {}).get("utilization"), 0.0) or 0.0 for m in members] or [1.0]
        )
        max_util = max(max_util, 1.0e-9)

        def _interp(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> str:
            tt = max(0.0, min(1.0, float(t)))
            r = int(round(c1[0] + (c2[0] - c1[0]) * tt))
            g = int(round(c1[1] + (c2[1] - c1[1]) * tt))
            b = int(round(c1[2] + (c2[2] - c1[2]) * tt))
            return f"rgb({r},{g},{b})"

        def _force_color(value: float) -> str:
            abs_v = abs(float(value))
            if abs_v <= max(2.0, 0.02 * max_abs_force):
                return "rgb(156,163,175)"  # neutro para quase zero
            t = min(1.0, abs_v / max_abs_force)
            if value >= 0.0:
                return _interp((147, 197, 253), (30, 64, 175), t)   # tração (azul)
            return _interp((254, 178, 76), (185, 28, 28), t)        # compressão (vermelho)

        def _util_color(value: float) -> str:
            u = max(0.0, float(value))
            if u <= 0.8:
                return _interp((34, 197, 94), (251, 191, 36), u / 0.8)
            if u <= 1.0:
                return _interp((251, 191, 36), (249, 115, 22), (u - 0.8) / 0.2)
            return _interp((249, 115, 22), (185, 28, 28), min(1.0, (u - 1.0) / 0.7))

        def _fs_color(value: float | None) -> str:
            if value is None:
                return "rgb(156,163,175)"  # cinza: sem FS global relevante

            fs = float(value)

            if fs < 0.75:
                return _interp((255, 205, 210), (127, 29, 29), min(1.0, (0.75 - fs) / 0.75))
            if fs < 1.00:
                return _interp((249, 115, 22), (185, 28, 28), min(1.0, (1.00 - fs) / 0.25))
            if fs < 1.15:
                return _interp((251, 191, 36), (249, 115, 22), min(1.0, (1.15 - fs) / 0.15))
            if fs < 1.50:
                return _interp((132, 204, 22), (251, 191, 36), min(1.0, (1.50 - fs) / 0.35))
            if fs < 2.00:
                return _interp((34, 197, 94), (132, 204, 22), min(1.0, (2.00 - fs) / 0.50))

            return "rgb(21,128,61)"

        def _member_fs_for_risk(mid: int, *, allow_local: bool = False) -> float | None:
            chk = chk_map.get(int(mid), {}) or {}

            fs_design = safe_float(chk.get("FS_design"), None)
            if fs_design is not None:
                return float(fs_design)

            design_relevant = chk.get("design_relevant", True)

            if design_relevant is False and not allow_local:
                return None

            fs_min = safe_float(chk.get("FS_min"), None)
            return float(fs_min) if fs_min is not None else None

        def _member_util_for_width(mid: int) -> float:
            chk = chk_map.get(int(mid), {}) or {}

            util = safe_float(chk.get("utilization_design"), None)
            if util is not None:
                return max(0.0, float(util))

            util = safe_float(chk.get("utilization"), None)
            if util is not None:
                return max(0.0, float(util))

            fs = _member_fs_for_risk(mid, allow_local=True)
            if fs is not None and fs > 1.0e-9:
                return max(0.0, 1.0 / fs)

            return 0.0

        def _risk_color(mid: int) -> str:
            return _fs_color(_member_fs_for_risk(mid, allow_local=False))

        def _line_width_for_member(mid: int) -> float:
            if color_mode == "force":
                n_val = safe_abs_float((res_map.get(int(mid), {}) or {}).get("N_N"), 0.0)
                return 2.0 + 5.0 * min(1.0, n_val / max_abs_force)

            util = _member_util_for_width(mid)
            return 2.0 + 6.0 * min(1.0, util / 1.25)

        member_line_color: Dict[int, str] = {}

        def _line_color_for_member(mid: int, group: str) -> str:
            if color_mode == "force":
                val = safe_float((res_map.get(mid, {}) or {}).get("N_N"), 0.0) or 0.0
                return _force_color(float(val))

            if color_mode == "utilization":
                val = _member_util_for_width(mid)
                return _util_color(float(val))

            if color_mode == "safety_factor":
                fs = _member_fs_for_risk(mid, allow_local=True)
                return _fs_color(fs)

            if color_mode == "risk":
                return _risk_color(mid)

            return group_color.get(group, "#4b5563")

        if color_mode == "group":
            for g in groups:
                xs, ys, zs = [], [], []
                for m in [m for m in members if m.group == g]:
                    ni, nj = node_by_id[m.i], node_by_id[m.j]
                    xs += [ni.x, nj.x, None]
                    ys += [ni.y, nj.y, None]
                    zs += [ni.z, nj.z, None]
                    member_line_color[int(m.id)] = group_color.get(g, "#4b5563")

                if xs:
                    fig.add_trace(
                        go.Scatter3d(
                            x=xs,
                            y=ys,
                            z=zs,
                            mode="lines",
                            name=g,
                            line={"width": 3, "color": group_color.get(g, "#4b5563")},
                            opacity=0.75,
                        )
                    )
        else:
            for m in members:
                ni, nj = node_by_id[m.i], node_by_id[m.j]
                mid = int(m.id)

                color = _line_color_for_member(mid, m.group)
                member_line_color[mid] = color

                res = res_map.get(mid, {}) or {}
                chk = chk_map.get(mid, {}) or {}

                n_val = safe_float(res.get("N_N"), 0.0) or 0.0
                util = safe_float(chk.get("utilization"), None)
                util_design = safe_float(chk.get("utilization_design"), None)
                fs_min = safe_float(chk.get("FS_min"), None)
                fs_design = safe_float(chk.get("FS_design"), None)

                hover_parts = [
                    f"Membro {mid}",
                    f"Grupo: {m.group}",
                    f"N = {n_val:.2f} N",
                ]

                if fs_min is not None:
                    hover_parts.append(f"FS_min = {fs_min:.3f}")

                if fs_design is not None:
                    hover_parts.append(f"FS_design = {fs_design:.3f}")

                if util is not None:
                    hover_parts.append(f"Utilização = {util:.3f}")

                if util_design is not None:
                    hover_parts.append(f"Utilização design = {util_design:.3f}")

                if chk.get("governing_mode"):
                    hover_parts.append(f"Modo: {chk.get('governing_mode')}")

                if chk.get("design_relevant") is False:
                    hover_parts.append("Apenas verificação local / travamento")

                hover = "<br>".join(hover_parts)
                fig.add_trace(
                    go.Scatter3d(
                        x=[ni.x, nj.x],
                        y=[ni.y, nj.y],
                        z=[ni.z, nj.z],
                        mode="lines",
                        name=f"membro_{mid}",
                        hovertext=hover,
                        hoverinfo="text",
                        line={
                            "width": _line_width_for_member(mid),
                            "color": color,
                        },
                        opacity=0.85,
                        showlegend=False,
                    )
                )
            if color_mode == "force":
                fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line={"color": "rgb(30,64,175)", "width": 5}, name="Tração"))
                fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line={"color": "rgb(185,28,28)", "width": 5}, name="Compressão"))
                fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line={"color": "rgb(156,163,175)", "width": 5}, name="Quase zero"))
            elif color_mode == "utilization":
                fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line={"color": "rgb(34,197,94)", "width": 5}, name="Baixa utilização"))
                fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line={"color": "rgb(185,28,28)", "width": 5}, name="Utilização crítica"))
            elif color_mode == "safety_factor":
                fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line={"color": "rgb(185,28,28)", "width": 5}, name="FS < 1,0 crítico"))
                fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line={"color": "rgb(249,115,22)", "width": 5}, name="FS 1,0–1,15 baixo"))
                fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line={"color": "rgb(34,197,94)", "width": 5}, name="FS seguro"))
            elif color_mode == "risk":
                fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line={"color": "rgb(185,28,28)", "width": 8}, name="Risco alto: FS < 1"))
                fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line={"color": "rgb(249,115,22)", "width": 6}, name="Risco médio: FS ≈ 1"))
                fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line={"color": "rgb(21,128,61)", "width": 4}, name="Baixo risco"))
                fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line={"color": "rgb(156,163,175)", "width": 4}, name="Local/travamento"))

        if selected_ids and highlight_selected:
            for mid in sorted(selected_ids):
                m = next((mm for mm in members if int(mm.id) == int(mid)), None)
                if m is None:
                    continue
                ni, nj = node_by_id[m.i], node_by_id[m.j]
                color = (
                    highlight_member_colors.get(int(mid), member_line_color.get(int(mid), "black"))
                    if highlight_member_colors
                    else member_line_color.get(int(mid), "black")
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=[ni.x, nj.x],
                        y=[ni.y, nj.y],
                        z=[ni.z, nj.z],
                        mode="lines+markers+text",
                        text=[f"M{mid}", ""],
                        textposition="top center",
                        hoverinfo="text",
                        hovertext=f"Membro {mid}<br>Grupo: {m.group}<br>{m.i} → {m.j}",
                        line={"width": 10, "color": color},
                        marker={"size": 5, "color": color, "symbol": "circle"},
                        name=f"selecionado {mid}",
                        showlegend=False,
                    )
                )

        support_ids = [s.node_id for s in supports if s.active_vertical]

        if support_ids:
            fig.add_trace(
                go.Scatter3d(
                    x=[node_by_id[i].x for i in support_ids],
                    y=[node_by_id[i].y for i in support_ids],
                    z=[node_by_id[i].z for i in support_ids],
                    mode="markers",
                    name="apoios ativos",
                    marker={"size": 6, "symbol": "diamond", "color": "cyan"},
                )
            )

        load_ids = [l.node_id for l in loads]

        if load_ids:
            fig.add_trace(
                go.Scatter3d(
                    x=[node_by_id[i].x for i in load_ids],
                    y=[node_by_id[i].y for i in load_ids],
                    z=[node_by_id[i].z for i in load_ids],
                    mode="markers",
                    name="cargas",
                    marker={"size": 6, "symbol": "x", "color": "orange"},
                )
            )

        scene = {
            "xaxis": {
                "title": "x [mm]",
                "range": [x_min - 50, x_max + 50],
            },
            "yaxis": {
                "title": "y [mm]",
                "range": [y_min - 40, y_max + 40],
            },
            "zaxis": {
                "title": "z [mm]",
                "range": [min(-20, z_min - 20), z_max + 40],
            },
            "aspectmode": aspectmode,
            "camera": {
                "eye": {"x": 1.65, "y": -1.25, "z": 0.75},
                "center": {"x": 0.0, "y": 0.0, "z": -0.08},
            },
        }

        if aspectratio is not None:
            scene["aspectratio"] = aspectratio

        fig.update_layout(
            scene=scene,
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
            height=650,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "left",
                "x": 0.0,
            },
        )

        return fig

    @staticmethod
    def make_oriented_stick_prism(
        p0: tuple[float, float, float],
        p1: tuple[float, float, float],
        width_mm: float,
        thickness_mm: float,
        local_rotation: float = 0.0,
        offset: tuple[float, float, float] | None = None,
    ) -> Dict[str, List[float]]:
        """Gera vértices/faces de um prisma orientado entre p0 e p1."""
        import numpy as _np

        p0v = _np.array(p0, dtype=float)
        p1v = _np.array(p1, dtype=float)
        if offset is not None:
            off = _np.array(offset, dtype=float)
            p0v = p0v + off
            p1v = p1v + off
        d = p1v - p0v
        L = _np.linalg.norm(d)
        if L <= 1e-9:
            return {"x": [float(p0v[0])] * 8, "y": [float(p0v[1])] * 8, "z": [float(p0v[2])] * 8, "i": [], "j": [], "k": []}
        d_unit = d / L
        aux = _np.array([0.0, 0.0, 1.0])
        if abs(_np.dot(aux, d_unit)) > 0.9:
            aux = _np.array([0.0, 1.0, 0.0])
        u = _np.cross(d_unit, aux)
        un = _np.linalg.norm(u)
        u = _np.array([1.0, 0.0, 0.0]) if un <= 1e-9 else (u / un)
        v = _np.cross(d_unit, u)
        vn = _np.linalg.norm(v)
        v = _np.array([0.0, 1.0, 0.0]) if vn <= 1e-9 else (v / vn)

        if abs(float(local_rotation)) > 1.0e-12:
            ang = float(local_rotation)
            c = math.cos(ang)
            s = math.sin(ang)
            u2 = c * u + s * v
            v2 = -s * u + c * v
            u, v = u2, v2

        half_w = float(width_mm) * 0.5
        half_t = float(thickness_mm) * 0.5
        offsets = [
            -half_w * u - half_t * v,
            half_w * u - half_t * v,
            half_w * u + half_t * v,
            -half_w * u + half_t * v,
        ]
        verts = []
        for end in [p0v, p1v]:
            for off in offsets:
                verts.append(end + off)
        xs = [float(pt[0]) for pt in verts]
        ys = [float(pt[1]) for pt in verts]
        zs = [float(pt[2]) for pt in verts]
        faces = [
            (0, 1, 2), (0, 2, 3),
            (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4),
            (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6),
            (3, 0, 4), (3, 4, 7),
        ]
        return {
            "x": xs,
            "y": ys,
            "z": zs,
            "i": [f[0] for f in faces],
            "j": [f[1] for f in faces],
            "k": [f[2] for f in faces],
        }

    def plotly_stick_pieces(
        self,
        stick_pieces,
        member_id: int | None = None,
        max_pieces: int = 1500,
        lane_offset_mm: float = 5.0,
        render_mode: str = "prismas reais",
    ):
        rows = list(stick_pieces or [])
        if member_id is not None:
            # Filtra apenas o membro solicitado
            rows = [r for r in rows if int(safe_float(r.get("member_id"), -1) or -1) == int(member_id)]
        # Limita número máximo de peças para evitar travamentos na renderização
        rows = rows[:max_pieces]

        fig = go.Figure()
        if not rows:
            fig.update_layout(title="Sem peças para mostrar", height=500)
            return fig

        # Agrupa por grupo de membros para colorir de forma consistente
        groups = sorted({str(r.get("member_group", "sem_grupo")) for r in rows})
        # Paleta simples de cores discretas
        base_colors = [
            "#1f77b4",  # azul
            "#ff7f0e",  # laranja
            "#2ca02c",  # verde
            "#d62728",  # vermelho
            "#9467bd",  # roxo
            "#8c564b",  # marrom
            "#e377c2",  # rosa
            "#7f7f7f",  # cinza
            "#bcbd22",  # oliva
            "#17becf",  # ciano
        ]

        render_mode_norm = str(render_mode or "prismas reais").strip().lower()
        use_lines = render_mode_norm in {"linhas", "linhas leves", "light_lines"}
        exaggeration = 1.0 if render_mode_norm in {"prismas reais", "prismas_reais"} else 2.0

        for gi, g in enumerate(groups):
            color = base_colors[gi % len(base_colors)]
            # Desenha cada peça como um prisma 3D
            for r in [rr for rr in rows if str(rr.get("member_group", "sem_grupo")) == g]:
                lane = int(safe_float(r.get("lane"), 1) or 1)
                pidx = int(safe_float(r.get("piece_index"), 1) or 1)
                # Offsets para separar visualmente lanes e peças alternadas
                off_y = (lane - 1) * lane_offset_mm
                off_z = (0.35 * lane_offset_mm) * ((pidx % 2) - 0.5)
                x0 = safe_float(r.get("x0_mm"), 0.0) or 0.0
                y0 = (safe_float(r.get("y0_mm"), 0.0) or 0.0) + off_y
                z0 = (safe_float(r.get("z0_mm"), 0.0) or 0.0) + off_z
                x1 = safe_float(r.get("x1_mm"), 0.0) or 0.0
                y1 = (safe_float(r.get("y1_mm"), 0.0) or 0.0) + off_y
                z1 = (safe_float(r.get("z1_mm"), 0.0) or 0.0) + off_z
                # Recupera dimensões físicas e orientação do palito.
                # width_mm/thickness_mm permanecem a dimensão nominal do blank;
                # visual_width_mm/visual_thickness_mm representam a orientação construtiva
                # no prisma renderizado (edge = lateral para cima).
                try:
                    nominal_wmm = float(r.get("width_mm"))
                    nominal_tmm = float(r.get("thickness_mm"))
                except (TypeError, ValueError):
                    nominal_wmm = 7.0
                    nominal_tmm = 1.5

                stick_orientation = str(r.get("stick_orientation", "flat") or "flat").strip().lower()
                try:
                    wmm = float(r.get("visual_width_mm"))
                    tmm = float(r.get("visual_thickness_mm"))
                except (TypeError, ValueError):
                    if stick_orientation == "edge":
                        wmm = nominal_tmm
                        tmm = nominal_wmm
                    else:
                        wmm = nominal_wmm
                        tmm = nominal_tmm
                # Monta label para hover
                label_parts = [
                    f"{r.get('stick_id', '')}",
                    f"Membro {r.get('member_id', '?')} — {g}",
                    f"Linha {lane}, peça {pidx}",
                    f"Corte {safe_float(r.get('cut_length_mm'), 0.0) or 0.0:.1f} mm",
                    f"N peça {safe_float(r.get('N_piece_N'), 0.0) or 0.0:.2f} N",
                ]
                label_parts.append(f"Blank nominal {nominal_wmm:.1f}×{nominal_tmm:.1f} mm")
                label_parts.append(f"Render/orientação {wmm:.1f}×{tmm:.1f} mm — {stick_orientation}")
                label = "<br>".join(label_parts)
                if use_lines:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[x0, x1],
                            y=[y0, y1],
                            z=[z0, z1],
                            mode="lines",
                            line={"width": 3, "color": color},
                            name=g,
                            hovertext=label,
                            hoverinfo="text",
                            showlegend=False,
                        )
                    )
                else:
                    prism = self.make_oriented_stick_prism(
                        (x0, y0, z0),
                        (x1, y1, z1),
                        width_mm=wmm * exaggeration,
                        thickness_mm=tmm * exaggeration,
                    )
                    fig.add_trace(
                        go.Mesh3d(
                            x=prism["x"],
                            y=prism["y"],
                            z=prism["z"],
                            i=prism["i"],
                            j=prism["j"],
                            k=prism["k"],
                            name=g,
                            opacity=0.85,
                            color=color,
                            hovertext=label,
                            hoverinfo="text",
                            showscale=False,
                        )
                    )

        # Ajusta aspectos da figura para melhor visualização
        xs_all = [safe_float(r.get("x0_mm"), 0.0) or 0.0 for r in rows] + [safe_float(r.get("x1_mm"), 0.0) or 0.0 for r in rows]
        ys_all = [safe_float(r.get("y0_mm"), 0.0) or 0.0 for r in rows] + [safe_float(r.get("y1_mm"), 0.0) or 0.0 for r in rows]
        zs_all = [safe_float(r.get("z0_mm"), 0.0) or 0.0 for r in rows] + [safe_float(r.get("z1_mm"), 0.0) or 0.0 for r in rows]
        x_span = max(max(xs_all) - min(xs_all), 1.0)
        y_span = max(max(ys_all) - min(ys_all), 1.0)
        z_span = max(max(zs_all) - min(zs_all), 1.0)
        fig.update_layout(
            title=f"Modelo peça‑a‑peça ({'linhas leves' if use_lines else render_mode})",
            scene={
                "xaxis": {"title": "x [mm]"},
                "yaxis": {"title": "y [mm]"},
                "zaxis": {"title": "z [mm]"},
                "aspectmode": "manual",
                "aspectratio": {
                    "x": 1.0,
                    "y": max(y_span / x_span, 0.20),
                    "z": max(z_span / x_span, 0.25),
                },
            },
            height=650,
            margin={"l": 0, "r": 0, "t": 45, "b": 0},
        )
        return fig
