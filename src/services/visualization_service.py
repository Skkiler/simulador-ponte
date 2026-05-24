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

        # Pacote visual enxuto: mantemos apenas a geometria 3D interativa
        # com cargas/apoios e coloração por FS/uso.  As vistas 2D/3D de
        # montagem peça-a-peça são geradas no DetailVisualizationService e no
        # bloco específico de detalhamento.
        paths: List[Path] = []
        fig_fs = self.plotly_geometry(
            nodes,
            members,
            supports,
            loads,
            color_mode="risk",
            member_results=member_results,
            member_checks=member_checks,
        )
        p_fs = out / "01_geometria_3d_fs_uso.html"
        fig_fs.write_html(p_fs)
        paths.append(p_fs)

        # Alias histórico usado por partes antigas do relatório.
        legacy = out / "geometria_3d_interativa.html"
        fig_fs.write_html(legacy)
        paths.append(legacy)
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
        width_axis: tuple[float, float, float] | None = None,
        thickness_axis: tuple[float, float, float] | None = None,
        start_miter_angle_deg: float | None = None,
        end_miter_angle_deg: float | None = None,
        start_miter_skew_sign: float | None = None,
        end_miter_skew_sign: float | None = None,
        start_miter_trim_axis: str | None = None,
        end_miter_trim_axis: str | None = None,
    ) -> Dict[str, List[float]]:
        """Gera vértices/faces de um prisma orientado entre p0 e p1.

        Quando ``width_axis``/``thickness_axis`` são informados, o prisma usa os
        mesmos eixos locais da seção que foram usados para posicionar as lanes de
        palitos no detalhamento.  Sem isso, cada membro escolhia eixos visuais a
        partir de um vetor auxiliar arbitrário; em uma ponte com muitos membros
        inclinados, prismas de uma mesma seção podiam parecer em leque mesmo
        quando o cálculo os mantinha colados no centroide correto.
        """
        import numpy as _np

        def _unit_or_none(v):
            if v is None:
                return None
            vv = _np.array(v, dtype=float)
            n = _np.linalg.norm(vv)
            if n <= 1e-9:
                return None
            return vv / n

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

        u = _unit_or_none(width_axis)
        v = _unit_or_none(thickness_axis)
        if u is not None:
            # Remove componente axial por segurança; o detalhamento já deve
            # fornecer eixos perpendiculares ao membro, mas CSVs antigos não têm
            # essa garantia.
            u = u - _np.dot(u, d_unit) * d_unit
            un = _np.linalg.norm(u)
            u = None if un <= 1e-9 else (u / un)
        if v is not None:
            v = v - _np.dot(v, d_unit) * d_unit
            vn = _np.linalg.norm(v)
            v = None if vn <= 1e-9 else (v / vn)

        if u is None or v is None or abs(float(_np.dot(u, v))) > 0.20:
            aux = _np.array([0.0, 0.0, 1.0])
            if abs(_np.dot(aux, d_unit)) > 0.9:
                aux = _np.array([0.0, 1.0, 0.0])
            u = _np.cross(d_unit, aux)
            un = _np.linalg.norm(u)
            u = _np.array([1.0, 0.0, 0.0]) if un <= 1e-9 else (u / un)
            v = _np.cross(d_unit, u)
            vn = _np.linalg.norm(v)
            v = _np.array([0.0, 1.0, 0.0]) if vn <= 1e-9 else (v / vn)
        else:
            # Ortonormaliza preservando o eixo de largura informado; isso evita
            # pequenos erros numéricos acumulados em prismas longos.
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

        def _miter_shift(angle_deg: float | None) -> float:
            if angle_deg is None:
                return 0.0
            try:
                a = float(angle_deg)
            except (TypeError, ValueError):
                return 0.0
            # 90° = corte quadrado. Ângulos menores desenham a face inclinada
            # sobre a espessura do palito. O comprimento de corte continua vindo
            # do CSV em múltiplos de 5 mm; isto muda apenas a malha visual.
            if a >= 89.5:
                return 0.0
            a = max(15.0, min(89.0, a))
            # Desenho deliberadamente simples: uma única face inclinada curta
            # na ponta do palito.  O visual não tenta representar entalhes
            # palito-a-palito nem cortes em zigue-zague, porque isso foi lido
            # como corte aleatório e não corresponde ao plano de montagem.
            cut_depth = min(abs(float(width_mm)), abs(float(thickness_mm)))
            return min(0.10 * L, max(0.0, cut_depth / math.tan(math.radians(a))))

        start_shift = _miter_shift(start_miter_angle_deg)
        end_shift = _miter_shift(end_miter_angle_deg)

        def _sign(value: float | None, default: float = 1.0) -> float:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return float(default)
            return 1.0 if v >= 0.0 else -1.0

        start_skew = _sign(start_miter_skew_sign, 1.0)
        end_skew = _sign(end_miter_skew_sign, 1.0)

        def _axis_side(idx: int, axis_name: str | None) -> float:
            axis = str(axis_name or "z").strip().lower()
            if axis in {"y", "width", "face_y"}:
                return 1.0 if idx in {1, 2} else -1.0
            # default: trim across local z/thickness face
            return 1.0 if idx in {2, 3} else -1.0

        verts = []
        for idx, off in enumerate(offsets):
            # The signed skew indicates which local side must be shortened to
            # match the host face.  Without this sign the bevel may be mirrored.
            side = _axis_side(idx, start_miter_trim_axis)
            should_trim = start_shift > 0.0 and (
                (start_skew >= 0.0 and side > 0.0)
                or (start_skew < 0.0 and side < 0.0)
            )
            verts.append(p0v + off + (start_shift if should_trim else 0.0) * d_unit)
        for idx, off in enumerate(offsets):
            side = _axis_side(idx, end_miter_trim_axis)
            should_trim = end_shift > 0.0 and (
                (end_skew >= 0.0 and side < 0.0)
                or (end_skew < 0.0 and side > 0.0)
            )
            verts.append(p1v + off - (end_shift if should_trim else 0.0) * d_unit)
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

    @staticmethod
    def _prism_edge_polyline(prism: Dict[str, List[float]]) -> tuple[List[float], List[float], List[float]]:
        """Retorna uma polyline única com as 12 arestas de um prisma."""
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        px = prism.get("x", [])
        py = prism.get("y", [])
        pz = prism.get("z", [])
        if len(px) < 8 or len(py) < 8 or len(pz) < 8:
            return xs, ys, zs
        for a, b in edges:
            xs.extend([px[a], px[b], None])
            ys.extend([py[a], py[b], None])
            zs.extend([pz[a], pz[b], None])
        return xs, ys, zs

    @staticmethod
    def _stable_color_index(key: str, modulo: int) -> int:
        if modulo <= 0:
            return 0
        total = 0
        for i, ch in enumerate(str(key)):
            total += (i + 1) * ord(ch)
        return total % modulo

    @staticmethod
    def _obb_from_prism(prism: Dict[str, List[float]]) -> Dict[str, Any] | None:
        """Caixa orientada do prisma renderizado.

        A auditoria antiga usava apenas AABB; diagonais longas ficavam com
        envelopes cartesianos enormes e geravam falsos positivos. O OBB/SAT
        usa a geometria local do próprio palito.
        """
        try:
            import numpy as _np

            pts = _np.array(list(zip(prism["x"], prism["y"], prism["z"])), dtype=float)
        except Exception:
            return None
        if pts.shape[0] < 8:
            return None
        center = pts.mean(axis=0)
        raw_axes = [pts[4] - pts[0], pts[1] - pts[0], pts[3] - pts[0]]
        axes = []
        extents = []
        for vec in raw_axes:
            n = float(_np.linalg.norm(vec))
            if n <= 1.0e-9:
                return None
            axes.append(vec / n)
            extents.append(0.5 * n)
        return {"center": center, "axes": axes, "extents": extents}

    @staticmethod
    def _obb_intersects(a: Dict[str, Any], b: Dict[str, Any], tol: float = 0.05) -> bool:
        """Teste SAT entre dois prismas retangulares orientados."""
        import numpy as _np

        axes = list(a["axes"]) + list(b["axes"])
        for ax in a["axes"]:
            for bx in b["axes"]:
                c = _np.cross(ax, bx)
                n = float(_np.linalg.norm(c))
                if n > 1.0e-8:
                    axes.append(c / n)
        delta = b["center"] - a["center"]
        for axis in axes:
            dist = abs(float(_np.dot(delta, axis)))
            ra = sum(abs(float(_np.dot(a["axes"][i], axis))) * float(a["extents"][i]) for i in range(3))
            rb = sum(abs(float(_np.dot(b["axes"][i], axis))) * float(b["extents"][i]) for i in range(3))
            if dist > ra + rb - tol:
                return False
        return True

    def prepare_stick_piece_mesh_batches(
        self,
        stick_pieces,
        member_id: int | None = None,
        max_pieces: int = 1500,
        lane_offset_mm: float = 5.0,
        color_by: str = "assembly_unit",
        connection_offset_scale: float = 0.0,
    ) -> Dict[str, Any]:
        """Pré-calcula prismas reais antes de montar o Plotly.

        O gargalo anterior era criar milhares de traces, um por palito.  Aqui
        os prismas são convertidos para vértices/faces uma única vez e agrupados
        em poucos Mesh3d por grupo estrutural, mantendo cor e hover por peça por
        meio de facecolor/text.  Isso reduz o tempo de renderização sem voltar a
        representar a ponte como grupos sólidos.
        """
        rows = list(stick_pieces or [])
        if member_id is not None:
            rows = [r for r in rows if int(safe_float(r.get("member_id"), -1) or -1) == int(member_id)]
        rows = rows[:max(1, int(max_pieces))]

        structural_colors = {
            "bottom_chord": "#1f77b4",
            "top_chord": "#d62728",
            "vertical": "#2ca02c",
            "diagonal": "#ff7f0e",
            "top_bracing": "#9467bd",
            "bottom_bracing": "#17becf",
            "cross_frame_bracing": "#bcbd22",
            "top_transverse": "#8c564b",
            "bottom_transverse": "#e377c2",
            "support_pad": "#7f7f7f",
            "chord_lacing": "#aec7e8",
        }
        # Paleta longa e alternada por matiz/luminância.  A versão anterior
        # usava hash sobre apenas 20 cores, causando colisões visuais: grupos
        # diferentes saíam com o mesmo laranja/roxo no HTML.
        base_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#637939",
            "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#31a354", "#756bb1",
            "#636363", "#e6550d", "#6baed6", "#fd8d3c", "#74c476", "#fb6a4a",
            "#9e9ac8", "#a55194", "#969696", "#bdb76b", "#00a6a6", "#b15928",
            "#a6cee3", "#fdbf6f", "#b2df8a", "#fb9a99", "#cab2d6", "#ffff99",
            "#33a02c", "#e31a1c", "#1f78b4", "#ff9896", "#98df8a", "#c5b0d5",
            "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5", "#ffbb78",
        ]
        has_real_section_offsets = any(
            safe_float(r.get("section_global_offset_y_mm"), None) is not None
            or safe_float(r.get("section_global_offset_z_mm"), None) is not None
            for r in rows
        )
        color_by_norm = str(color_by or "assembly_unit").strip().lower()

        def _row_color_key(row: Dict[str, Any]) -> str:
            group_name = str(row.get("member_group", "sem_grupo"))
            lane_i = int(safe_float(row.get("lane"), 1) or 1)
            pidx_i = int(safe_float(row.get("piece_index"), 1) or 1)
            assembly_key_i = str(row.get("assembly_unit_key") or row.get("stick_id") or f"M{row.get('member_id')}-L{lane_i}-P{pidx_i}")
            if color_by_norm in {"piece", "stick", "assembly_unit", "assembly"}:
                return assembly_key_i
            if color_by_norm == "member":
                return f"M{row.get('member_id')}"
            return group_name

        sorted_color_keys = sorted({_row_color_key(r) for r in rows})
        color_map = {key: base_colors[i % len(base_colors)] for i, key in enumerate(sorted_color_keys)}
        for group_name, color in structural_colors.items():
            if color_by_norm in {"group", "member_group", "structural_group"}:
                color_map[group_name] = color

        batches: Dict[str, Dict[str, Any]] = {}
        bounds = {"x": [], "y": [], "z": []}
        as_built_boxes: List[Dict[str, Any]] = []
        as_built_gap_piece_count = 0

        for r in rows:
            group = str(r.get("member_group", "sem_grupo"))
            lane = int(safe_float(r.get("lane"), 1) or 1)
            pidx = int(safe_float(r.get("piece_index"), 1) or 1)
            if has_real_section_offsets:
                off_y = 0.0
                off_z = 0.0
            else:
                off_y = (lane - 1) * lane_offset_mm
                off_z = (0.35 * lane_offset_mm) * ((pidx % 2) - 0.5)
            try:
                offset_scale = float(connection_offset_scale)
            except (TypeError, ValueError):
                offset_scale = 0.0
            vx = offset_scale * (safe_float(r.get("visual_connection_offset_x_mm"), 0.0) or 0.0)
            vy = offset_scale * (safe_float(r.get("visual_connection_offset_y_mm"), 0.0) or 0.0)
            vz = offset_scale * (safe_float(r.get("visual_connection_offset_z_mm"), 0.0) or 0.0)
            x0 = (safe_float(r.get("x0_mm"), 0.0) or 0.0) + vx
            y0 = (safe_float(r.get("y0_mm"), 0.0) or 0.0) + off_y + vy
            z0 = (safe_float(r.get("z0_mm"), 0.0) or 0.0) + off_z + vz
            x1 = (safe_float(r.get("x1_mm"), 0.0) or 0.0) + vx
            y1 = (safe_float(r.get("y1_mm"), 0.0) or 0.0) + off_y + vy
            z1 = (safe_float(r.get("z1_mm"), 0.0) or 0.0) + off_z + vz
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

            axis_y = (
                safe_float(r.get("section_axis_y_x"), None),
                safe_float(r.get("section_axis_y_y"), None),
                safe_float(r.get("section_axis_y_z"), None),
            )
            axis_z = (
                safe_float(r.get("section_axis_z_x"), None),
                safe_float(r.get("section_axis_z_y"), None),
                safe_float(r.get("section_axis_z_z"), None),
            )
            if any(v is None for v in axis_y) or any(v is None for v in axis_z):
                axis_y = None
                axis_z = None
            use_bevel = bool(r.get("miter_cut_required", False))
            prism = self.make_oriented_stick_prism(
                (x0, y0, z0),
                (x1, y1, z1),
                width_mm=wmm,
                thickness_mm=tmm,
                width_axis=axis_y,
                thickness_axis=axis_z,
                start_miter_angle_deg=safe_float(r.get("miter_cut_start_angle_deg"), 90.0) if use_bevel else 90.0,
                end_miter_angle_deg=safe_float(r.get("miter_cut_end_angle_deg"), 90.0) if use_bevel else 90.0,
                start_miter_skew_sign=safe_float(r.get("miter_cut_start_skew_sign"), 1.0),
                end_miter_skew_sign=safe_float(r.get("miter_cut_end_skew_sign"), 1.0),
                start_miter_trim_axis=str(r.get("miter_cut_start_trim_axis", "z") or "z"),
                end_miter_trim_axis=str(r.get("miter_cut_end_trim_axis", "z") or "z"),
            )
            assembly_key = str(r.get("assembly_unit_key") or r.get("stick_id") or f"M{r.get('member_id')}-L{lane}-P{pidx}")
            color_key = _row_color_key(r)
            color = color_map.get(color_key, base_colors[self._stable_color_index(color_key, len(base_colors))])
            label_parts = [
                f"{r.get('stick_id', '')}",
                f"Membro {r.get('member_id', '?')} — {group}",
                f"Linha {lane}, peça {pidx}",
                f"Unidade: {assembly_key}",
                f"Corte {safe_float(r.get('cut_length_mm'), 0.0) or 0.0:.1f} mm",
                f"Comprimento instalado {safe_float(r.get('installed_length_mm'), 0.0) or 0.0:.1f} mm",
                f"Perda por corte em grau {safe_float(r.get('miter_cut_material_loss_length_mm'), 0.0) or 0.0:.2f} mm",
                f"N peça {safe_float(r.get('N_piece_N'), 0.0) or 0.0:.2f} N",
                f"Blank nominal {nominal_wmm:.1f}×{nominal_tmm:.1f} mm",
                f"Render/orientação {wmm:.1f}×{tmm:.1f} mm — {stick_orientation}",
                f"Junta início: {r.get('connection_start_mode', 'axis_centroid')}",
                f"Junta fim: {r.get('connection_end_mode', 'axis_centroid')}",
            ]
            if bool(r.get("miter_cut_required", False)):
                label_parts.append(
                    f"Corte em grau: {safe_float(r.get('miter_cut_start_angle_deg'), 90.0) or 90.0:.0f}°/{safe_float(r.get('miter_cut_end_angle_deg'), 90.0) or 90.0:.0f}° "
                    f"(skew {safe_float(r.get('miter_cut_start_skew_sign'), 1.0) or 1.0:.0f}/{safe_float(r.get('miter_cut_end_skew_sign'), 1.0) or 1.0:.0f}; "
                    f"eixo {r.get('miter_cut_start_trim_axis', '') or '-'}/{r.get('miter_cut_end_trim_axis', '') or '-'})"
                )
                if r.get("miter_cut_start_host_group") or r.get("miter_cut_end_host_group"):
                    label_parts.append(
                        f"Host corte: {r.get('miter_cut_start_host_group', '') or '-'} / {r.get('miter_cut_end_host_group', '') or '-'}"
                    )
            sy = safe_float(r.get("section_local_y_mm"), None)
            sz = safe_float(r.get("section_local_z_mm"), None)
            if sy is not None and sz is not None:
                label_parts.append(f"Posição seção local y/z = {sy:.1f}/{sz:.1f} mm")
            label = "<br>".join(label_parts)

            batch = batches.setdefault(group, {
                "x": [], "y": [], "z": [], "i": [], "j": [], "k": [],
                "text": [], "facecolor": [], "edge_x": [], "edge_y": [], "edge_z": [],
                "hover_x": [], "hover_y": [], "hover_z": [], "hover_text": [],
            })
            base_idx = len(batch["x"])
            batch["x"].extend(prism["x"])
            batch["y"].extend(prism["y"])
            batch["z"].extend(prism["z"])
            batch["i"].extend([base_idx + int(v) for v in prism["i"]])
            batch["j"].extend([base_idx + int(v) for v in prism["j"]])
            batch["k"].extend([base_idx + int(v) for v in prism["k"]])
            batch["text"].extend([label] * len(prism["x"]))
            batch["facecolor"].extend([color] * len(prism["i"]))
            batch["hover_x"].append(0.5 * (x0 + x1))
            batch["hover_y"].append(0.5 * (y0 + y1))
            batch["hover_z"].append(0.5 * (z0 + z1))
            batch["hover_text"].append(label)

            # Arestas reais do prisma.  Não desenhamos diagonais internas dos
            # triângulos da malha; isso evita a falsa impressão de que um palito
            # foi repartido em mais peças do que existe no CSV.
            ex, ey, ez = self._prism_edge_polyline(prism)
            batch["edge_x"].extend(ex)
            batch["edge_y"].extend(ey)
            batch["edge_z"].extend(ez)
            bounds["x"].extend([x0, x1])
            bounds["y"].extend([y0, y1])
            bounds["z"].extend([z0, z1])
            if abs(float(offset_scale)) < 1.0e-9:
                as_built_boxes.append(
                    {
                        "stick_id": r.get("stick_id"),
                        "member_id": r.get("member_id"),
                        "member_group": group,
                        "lane": r.get("lane"),
                        "piece_index": r.get("piece_index"),
                        "s0_mm": safe_float(r.get("s0_mm"), None),
                        "s1_mm": safe_float(r.get("s1_mm"), None),
                        "x0": min(prism["x"]),
                        "x1": max(prism["x"]),
                        "y0": min(prism["y"]),
                        "y1": max(prism["y"]),
                        "z0": min(prism["z"]),
                        "z1": max(prism["z"]),
                        "obb": self._obb_from_prism(prism),
                    }
                )
                if not bool(r.get("node_connection_ok", True)):
                    as_built_gap_piece_count += 1

        as_built_interpenetration_samples: List[Dict[str, Any]] = []
        if as_built_boxes:
            tol = 1.0e-6
            boxes_sorted = sorted(as_built_boxes, key=lambda b: float(b["x0"]))
            active: List[Dict[str, Any]] = []
            for box in boxes_sorted:
                bx0 = float(box["x0"])
                bx1 = float(box["x1"])
                by0 = float(box["y0"])
                by1 = float(box["y1"])
                bz0 = float(box["z0"])
                bz1 = float(box["z1"])
                active = [a for a in active if float(a["x1"]) >= bx0 - tol]
                for a in active:
                    ax0 = float(a["x0"])
                    ax1 = float(a["x1"])
                    ay0 = float(a["y0"])
                    ay1 = float(a["y1"])
                    az0 = float(a["z0"])
                    az1 = float(a["z1"])
                    ox = min(ax1, bx1) - max(ax0, bx0)
                    oy = min(ay1, by1) - max(ay0, by0)
                    oz = min(az1, bz1) - max(az0, bz0)
                    if ox > tol and oy > tol and oz > tol:
                        # Ignore exact butt contact between consecutive pieces of
                        # the same member/lane.  They intentionally share a face;
                        # the continuity is supplied by splints, not by overlap.
                        if (
                            str(a.get("member_id")) == str(box.get("member_id"))
                            and str(a.get("lane")) == str(box.get("lane"))
                        ):
                            a0 = safe_float(a.get("s0_mm"), None)
                            a1 = safe_float(a.get("s1_mm"), None)
                            b0 = safe_float(box.get("s0_mm"), None)
                            b1 = safe_float(box.get("s1_mm"), None)
                            if None not in (a0, a1, b0, b1):
                                param_overlap = min(float(a1), float(b1)) - max(float(a0), float(b0))
                                if param_overlap <= 1.0e-6:
                                    continue
                        obb_a = a.get("obb")
                        obb_b = box.get("obb")
                        if obb_a is not None and obb_b is not None and not self._obb_intersects(obb_a, obb_b, tol=tol):
                            continue
                        as_built_interpenetration_samples.append(
                            {
                                "stick_a": a.get("stick_id"),
                                "stick_b": box.get("stick_id"),
                                "member_a": a.get("member_id"),
                                "member_b": box.get("member_id"),
                                "group_a": a.get("member_group"),
                                "group_b": box.get("member_group"),
                                "overlap_x_mm": ox,
                                "overlap_y_mm": oy,
                                "overlap_z_mm": oz,
                                "collision_test": "oriented_box_sat",
                            }
                        )
                        if len(as_built_interpenetration_samples) >= 200:
                            break
                if len(as_built_interpenetration_samples) >= 200:
                    break
                active.append(box)

        return {
            "rows": rows,
            "batches": batches,
            "bounds": bounds,
            "has_real_section_offsets": has_real_section_offsets,
            "color_by": color_by_norm,
            "as_built_interpenetration_count": len(as_built_interpenetration_samples),
            "as_built_interpenetration_samples": as_built_interpenetration_samples,
            "as_built_gap_piece_count": int(as_built_gap_piece_count),
        }

    def plotly_stick_pieces(
        self,
        stick_pieces,
        member_id: int | None = None,
        max_pieces: int = 1500,
        lane_offset_mm: float = 5.0,
        render_mode: str = "prismas reais",
        precomputed_mesh_batches: Dict[str, Any] | None = None,
        color_by: str = "assembly_unit",
        connection_offset_scale: float = 0.0,
    ):
        data = precomputed_mesh_batches or self.prepare_stick_piece_mesh_batches(
            stick_pieces,
            member_id=member_id,
            max_pieces=max_pieces,
            lane_offset_mm=lane_offset_mm,
            color_by=color_by,
            connection_offset_scale=connection_offset_scale,
        )
        rows = list(data.get("rows", []) or [])
        fig = go.Figure()
        if not rows:
            fig.update_layout(title="Sem peças para mostrar", height=500)
            return fig

        batches = data.get("batches", {}) or {}
        for group, batch in sorted(batches.items(), key=lambda kv: kv[0]):
            fig.add_trace(
                go.Mesh3d(
                    x=batch.get("x", []),
                    y=batch.get("y", []),
                    z=batch.get("z", []),
                    i=batch.get("i", []),
                    j=batch.get("j", []),
                    k=batch.get("k", []),
                    text=batch.get("text", []),
                    hovertemplate="%{text}<extra></extra>",
                    name=str(group),
                    opacity=1.0,
                    facecolor=batch.get("facecolor", []),
                    flatshading=True,
                    lighting={"ambient": 0.85, "diffuse": 0.55, "roughness": 1.0, "specular": 0.05},
                    showscale=False,
                    showlegend=True,
                )
            )
            if batch.get("edge_x"):
                fig.add_trace(
                    go.Scatter3d(
                        x=batch.get("edge_x", []),
                        y=batch.get("edge_y", []),
                        z=batch.get("edge_z", []),
                        mode="lines",
                        line={"width": 1.15, "color": "rgba(0,0,0,0.42)"},
                        name=f"arestas reais — {group}",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            # Mesh3d nem sempre dispara hover no centro da face. Estes marcadores
            # quase invisíveis ficam no centro geométrico de cada peça física e
            # garantem que a auditoria mostre dados ao passar o mouse.
            if batch.get("hover_x"):
                fig.add_trace(
                    go.Scatter3d(
                        x=batch.get("hover_x", []),
                        y=batch.get("hover_y", []),
                        z=batch.get("hover_z", []),
                        mode="markers",
                        # Marcadores de hover devem existir para capturar o mouse,
                        # mas não podem aparecer como "pontos brancos" de montagem.
                        # Usar cor quase transparente e tamanho mínimo mantém o
                        # hover peça-a-peça sem poluir a leitura geométrica.
                        marker={"size": 2, "color": "rgba(0,0,0,0.001)"},
                        text=batch.get("hover_text", []),
                        hovertemplate="%{text}<extra></extra>",
                        name=f"dados das peças — {group}",
                        showlegend=False,
                    )
                )

        # Linhas de travamento local foram removidas da vista peça-a-peça.
        # No Plotly elas eram percebidas como cortes/arestas no meio dos palitos,
        # embora fossem apenas linhas auxiliares. A auditoria geométrica deve vir
        # do CSV e do hover individual de cada prisma físico.

        bounds = data.get("bounds", {}) or {}
        xs_all = list(bounds.get("x", []) or [0.0])
        ys_all = list(bounds.get("y", []) or [0.0])
        zs_all = list(bounds.get("z", []) or [0.0])
        x_span = max(max(xs_all) - min(xs_all), 1.0)
        y_span = max(max(ys_all) - min(ys_all), 1.0)
        z_span = max(max(zs_all) - min(zs_all), 1.0)
        fig.update_layout(
            title=(
                "Modelo peça‑a‑peça — posição de encaixe"
                if abs(float(connection_offset_scale or 0.0)) < 1.0e-9
                else "Modelo peça‑a‑peça — vista explodida/auditável"
            ),
            scene={
                "xaxis": {"title": "x [mm]"},
                "yaxis": {"title": "y [mm]"},
                "zaxis": {"title": "z [mm]"},
                "aspectmode": "manual",
                "aspectratio": {
                    "x": 1.0,
                    "y": y_span / x_span,
                    "z": z_span / x_span,
                },
            },
            height=650,
            margin={"l": 0, "r": 0, "t": 45, "b": 0},
        )
        return fig
