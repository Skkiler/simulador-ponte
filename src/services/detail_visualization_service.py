from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt

from src.core.numeric import safe_float

class DetailVisualizationService:
    def save_all(self, detailed: Dict, out_dir: str | Path, top_n: int = 12) -> List[Path]:
        out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        # Saídas gráficas enxutas: apenas vistas que ajudam a montar a ponte
        # peça-a-peça.  Gráficos de massa, ranking e cortes continuam nos CSVs
        # e no relatório textual, mas deixam de poluir a pasta de plots.
        paths: List[Path] = []
        pieces = detailed.get('stick_pieces', []) or []
        if pieces:
            paths.append(self.plot_piece_orthographic_overview(pieces, out/'16_vistas_cad_peca_a_peca.png'))
            paths.extend(self.plot_piece_orthographic_by_group(pieces, out/'cad_subconjuntos'))
        return paths
    def plot_cutting_list(self, rows:List[Dict], path):
        p=Path(path); fig,ax=plt.subplots(figsize=(12,7)); top=sorted(rows,key=lambda r:float(r.get('quantity',0)),reverse=True)[:25]
        if top:
            ax.barh([f"{float(r['cut_length_mm']):.1f} mm" for r in top],[float(r['quantity']) for r in top]); ax.invert_yaxis()
        ax.set_title('Lista de cortes — comprimentos mais frequentes'); ax.set_xlabel('Quantidade de peças'); fig.tight_layout(); fig.savefig(p,dpi=220); plt.close(fig); return p
    def plot_glue_joints(self, rows:List[Dict], path):
        p=Path(path); fig,ax=plt.subplots(figsize=(12,7)); top=rows[:25]
        if top:
            ax.barh(
                [str(r.get("joint_id", ""))[-18:] for r in top],
                [safe_float(r.get("FS_glue_shear"), 0.0) or 0.0 for r in top],
            )
            ax.axvline(1, linestyle="--")
            ax.axvline(2, linestyle=":")
            ax.invert_yaxis()
        ax.set_title('Juntas coladas — menor FS ao cisalhamento'); ax.set_xlabel('FS cola'); fig.tight_layout(); fig.savefig(p,dpi=220); plt.close(fig); return p
    def plot_mass_summary(self, summary:Dict, path):
        p=Path(path); fig,ax=plt.subplots(figsize=(8,5)); vals=[float(summary.get('estimated_total_mass_g',0)),float(summary.get('mass_limit_g',1000))]
        ax.bar(['massa estimada','limite'],vals); ax.set_ylabel('massa [g]'); ax.set_title('Massa estimada vs limite')
        for i,v in enumerate(vals): ax.text(i,v,f'{v:.1f} g',ha='center',va='bottom')
        ax.text(0.5,max(vals)*0.85 if vals else 0,f"margem: {float(summary.get('mass_margin_g',0)):.1f} g",ha='center')
        fig.tight_layout(); fig.savefig(p,dpi=220); plt.close(fig); return p
    def plot_member_templates(self, sticks:List[Dict], weakest:List[Dict], path, top_n:int=12):
        p=Path(path); ids=[int(r['member_id']) for r in weakest[:top_n]] or sorted({int(r['member_id']) for r in sticks})[:top_n]
        rows_by=defaultdict(list)
        for r in sticks:
            mid=int(r['member_id'])
            if mid in ids: rows_by[mid].append(r)
        n=max(1,len(ids)); fig,axes=plt.subplots(n,1,figsize=(13,max(3,1.25*n)),sharex=False)
        if n==1: axes=[axes]
        for ax,mid in zip(axes,ids):
            rows=sorted(rows_by.get(mid,[]),key=lambda r:(int(r['lane']),int(r['piece_index']))); lanes=sorted({int(r['lane']) for r in rows})
            for lane in lanes:
                for r in [rr for rr in rows if int(rr['lane'])==lane]:
                    ax.plot([float(r['s0_mm']),float(r['s1_mm'])],[lane,lane],linewidth=5,solid_capstyle='butt')
                    ax.text((float(r['s0_mm'])+float(r['s1_mm']))/2,lane+0.13,f"P{int(r['piece_index'])}",ha='center',fontsize=7)
            group=rows[0]['member_group'] if rows else ''
            ax.set_title(f'Membro {mid} — {group}'); ax.set_ylabel('linha'); ax.grid(True,axis='x',alpha=0.25)
        axes[-1].set_xlabel('posição ao longo do membro [mm]'); fig.tight_layout(); fig.savefig(p,dpi=220); plt.close(fig); return p
    def plot_piece_orthographic_overview(self, pieces: List[Dict], path):
        """Desenha vistas 2D tipo CAD do detalhamento peça-a-peça.

        A imagem usa os mesmos pontos globais do ``stick_pieces.csv``.  Assim,
        se a visualização 3D peça-a-peça estiver filtrada por grupo, a relação
        geométrica é consistente com a ponte completa.
        """
        p = Path(path)
        rows = list(pieces or [])
        fig, axes = plt.subplots(3, 1, figsize=(14, 11))
        titles = [
            "Elevação lateral — x/z [mm]",
            "Planta superior — x/y [mm]",
            "Seções/projeções transversais — y/z [mm]",
        ]
        for ax, title in zip(axes, titles):
            ax.set_title(title)
            ax.grid(True, alpha=0.22)
        if rows:
            # Amostragem determinística para manter o arquivo leve quando há
            # milhares de peças.
            step = max(1, int(len(rows) / 1800))
            for r in rows[::step]:
                x0 = safe_float(r.get("x0_mm"), 0.0) or 0.0
                y0 = safe_float(r.get("y0_mm"), 0.0) or 0.0
                z0 = safe_float(r.get("z0_mm"), 0.0) or 0.0
                x1 = safe_float(r.get("x1_mm"), 0.0) or 0.0
                y1 = safe_float(r.get("y1_mm"), 0.0) or 0.0
                z1 = safe_float(r.get("z1_mm"), 0.0) or 0.0
                axes[0].plot([x0, x1], [z0, z1], linewidth=1.2, alpha=0.55)
                axes[1].plot([x0, x1], [y0, y1], linewidth=1.2, alpha=0.55)
                axes[2].plot([y0, y1], [z0, z1], linewidth=1.2, alpha=0.45)
            axes[0].set_xlabel("x [mm]"); axes[0].set_ylabel("z [mm]")
            axes[1].set_xlabel("x [mm]"); axes[1].set_ylabel("y [mm]")
            axes[2].set_xlabel("y [mm]"); axes[2].set_ylabel("z [mm]")
            axes[0].axis("equal"); axes[1].axis("equal"); axes[2].axis("equal")
        else:
            for ax in axes:
                ax.text(0.5, 0.5, "Sem dados peça-a-peça", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(p, dpi=220)
        plt.close(fig)
        return p

    def plot_piece_orthographic_by_group(self, pieces: List[Dict], out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows = list(pieces or [])
        paths: List[Path] = []
        if not rows:
            return paths
        preferred = [
            "top_chord",
            "bottom_chord",
            "vertical",
            "diagonal",
            "top_transverse",
            "bottom_transverse",
            "top_bracing",
            "bottom_bracing",
            "cross_frame_bracing",
            "support_pad",
        ]
        groups = [g for g in preferred if any(str(r.get("member_group")) == g for r in rows)]
        for group in groups:
            safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in group)
            paths.append(self.plot_piece_orthographic_overview(
                [r for r in rows if str(r.get("member_group")) == group],
                out / f"{safe_name}_vistas_2d.png",
            ))
        return paths

    def plot_piece_map_by_group(self, rows:List[Dict], path):
        p=Path(path); counts=defaultdict(float)
        for r in rows: counts[r.get('group','')]+=float(r.get('total_piece_count',0))
        items=sorted(counts.items(),key=lambda kv:kv[1],reverse=True); fig,ax=plt.subplots(figsize=(12,7))
        if items:
            ax.barh([k for k,v in items],[v for k,v in items]); ax.invert_yaxis()
        ax.set_title('Quantidade de peças por grupo estrutural'); ax.set_xlabel('peças individuais estimadas'); fig.tight_layout(); fig.savefig(p,dpi=220); plt.close(fig); return p
