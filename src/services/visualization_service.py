from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.domain.models import Load, Member, Node, Support


def safe_float(value: Any, default: float | None = None) -> float | None:
    """
    Converte valor para float sem quebrar com None, string vazia, NaN,
    infinito ou texto.

    Use em qualquer campo que venha de CSV, pós-processamento ou JSON.
    """
    try:
        if value is None:
            return default

        if isinstance(value, str) and value.strip() == "":
            return default

        v = float(value)

        if math.isnan(v) or math.isinf(v):
            return default

        return v
    except Exception:
        return default


def safe_sort_key(value: Any, default: float = 1.0e99) -> float:
    """
    Chave de ordenação segura.
    Valores vazios/inválidos vão para o final.
    """
    v = safe_float(value, None)
    return default if v is None else v


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
        except Exception:
            pass

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
        scale_mode: str = "real",
    ):
        """
        Gera visualização 3D interativa.

        scale_mode:
            "real"      -> proporção real: x muito maior que y/z.
            "didactic"  -> leve exagero de y/z para leitura visual.
            "cube"      -> cubo Plotly, útil só para inspecionar conexões.
        """
        node_by_id = {n.id: n for n in nodes}
        highlight_ids = {int(v) for v in (highlight_member_ids or [])}

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

        groups = sorted(set(m.group for m in members))

        for g in groups:
            xs, ys, zs = [], [], []

            for m in [m for m in members if m.group == g and m.id not in highlight_ids]:
                ni, nj = node_by_id[m.i], node_by_id[m.j]
                xs += [ni.x, nj.x, None]
                ys += [ni.y, nj.y, None]
                zs += [ni.z, nj.z, None]

            if xs:
                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        name=g,
                        line={"width": 3},
                        opacity=0.75,
                    )
                )

        if highlight_ids:
            hx, hy, hz = [], [], []
            htext = []

            for m in [m for m in members if m.id in highlight_ids]:
                ni, nj = node_by_id[m.i], node_by_id[m.j]

                hx += [ni.x, nj.x, None]
                hy += [ni.y, nj.y, None]
                hz += [ni.z, nj.z, None]

                htext += [
                    f"Membro {m.id}<br>Grupo: {m.group}<br>{m.i} → {m.j}",
                    f"Membro {m.id}<br>Grupo: {m.group}<br>{m.i} → {m.j}",
                    None,
                ]

            fig.add_trace(
                go.Scatter3d(
                    x=hx,
                    y=hy,
                    z=hz,
                    mode="lines+markers",
                    name="membro destacado",
                    text=htext,
                    hoverinfo="text",
                    line={"width": 10, "color": "red"},
                    marker={"size": 5, "color": "red"},
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

    def plotly_stick_pieces(self, stick_pieces, member_id: int | None = None, max_pieces: int = 1500, lane_offset_mm: float = 5.0):
        rows = list(stick_pieces or [])
        if member_id is not None:
            rows = [r for r in rows if int(safe_float(r.get("member_id"), -1) or -1) == int(member_id)]
        rows = rows[:max_pieces]
        fig = go.Figure()
        if not rows:
            fig.update_layout(title="Sem peças para mostrar", height=500)
            return fig
        groups = sorted({str(r.get("member_group", "sem_grupo")) for r in rows})
        for g in groups:
            xs=[]; ys=[]; zs=[]; texts=[]
            for r in [rr for rr in rows if str(rr.get("member_group", "sem_grupo")) == g]:
                lane=int(safe_float(r.get("lane"),1) or 1); pidx=int(safe_float(r.get("piece_index"),1) or 1)
                off_y=(lane-1)*lane_offset_mm; off_z=(0.35*lane_offset_mm)*((pidx%2)-0.5)
                x0=safe_float(r.get("x0_mm"),0.0) or 0.0; y0=(safe_float(r.get("y0_mm"),0.0) or 0.0)+off_y; z0=(safe_float(r.get("z0_mm"),0.0) or 0.0)+off_z
                x1=safe_float(r.get("x1_mm"),0.0) or 0.0; y1=(safe_float(r.get("y1_mm"),0.0) or 0.0)+off_y; z1=(safe_float(r.get("z1_mm"),0.0) or 0.0)+off_z
                label=f"{r.get('stick_id','')}<br>Membro {r.get('member_id','?')} — {g}<br>Linha {lane}, peça {pidx}<br>Corte {safe_float(r.get('cut_length_mm'),0.0) or 0.0:.1f} mm<br>N peça {safe_float(r.get('N_piece_N'),0.0) or 0.0:.2f} N"
                xs += [x0,x1,None]; ys += [y0,y1,None]; zs += [z0,z1,None]; texts += [label,label,None]
            fig.add_trace(go.Scatter3d(x=xs,y=ys,z=zs,mode="lines+markers",name=g,text=texts,hoverinfo="text",line={"width":6},marker={"size":3}))
        xs_all=[safe_float(r.get("x0_mm"),0.0) or 0.0 for r in rows]+[safe_float(r.get("x1_mm"),0.0) or 0.0 for r in rows]
        ys_all=[safe_float(r.get("y0_mm"),0.0) or 0.0 for r in rows]+[safe_float(r.get("y1_mm"),0.0) or 0.0 for r in rows]
        zs_all=[safe_float(r.get("z0_mm"),0.0) or 0.0 for r in rows]+[safe_float(r.get("z1_mm"),0.0) or 0.0 for r in rows]
        x_span=max(max(xs_all)-min(xs_all),1.0); y_span=max(max(ys_all)-min(ys_all),1.0); z_span=max(max(zs_all)-min(zs_all),1.0)
        fig.update_layout(title="Modelo simplificado peça-a-peça",scene={"xaxis":{"title":"x [mm]"},"yaxis":{"title":"y [mm]"},"zaxis":{"title":"z [mm]"},"aspectmode":"manual","aspectratio":{"x":1.0,"y":max(y_span/x_span,0.20),"z":max(z_span/x_span,0.25)}},height=650,margin={"l":0,"r":0,"t":45,"b":0})
        return fig
