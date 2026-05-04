from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt

import math
from typing import Any

def safe_float(value: Any, default: float = 0.0) -> float:
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

class DetailVisualizationService:
    def save_all(self, detailed: Dict, out_dir: str | Path, top_n: int = 12) -> List[Path]:
        out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        return [
            self.plot_cutting_list(detailed.get('cutting_list',[]), out/'11_lista_de_cortes.png'),
            self.plot_glue_joints(detailed.get('weakest_glue_joints',[]), out/'12_juntas_coladas_criticas.png'),
            self.plot_mass_summary(detailed.get('summary',{}), out/'13_resumo_massa.png'),
            self.plot_member_templates(detailed.get('stick_pieces',[]), detailed.get('weakest_members',[]), out/'14_gabaritos_membros_criticos.png', top_n),
            self.plot_piece_map_by_group(detailed.get('member_detail_checks',[]), out/'15_pecas_por_grupo.png'),
        ]
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
                [safe_float(r.get("FS_glue_shear"), 0.0) for r in top],
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
    def plot_piece_map_by_group(self, rows:List[Dict], path):
        p=Path(path); counts=defaultdict(float)
        for r in rows: counts[r.get('group','')]+=float(r.get('total_piece_count',0))
        items=sorted(counts.items(),key=lambda kv:kv[1],reverse=True); fig,ax=plt.subplots(figsize=(12,7))
        if items:
            ax.barh([k for k,v in items],[v for k,v in items]); ax.invert_yaxis()
        ax.set_title('Quantidade de peças por grupo estrutural'); ax.set_xlabel('peças individuais estimadas'); fig.tight_layout(); fig.savefig(p,dpi=220); plt.close(fig); return p
