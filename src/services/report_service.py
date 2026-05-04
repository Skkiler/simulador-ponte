from __future__ import annotations
from pathlib import Path
from typing import Dict

class ReportService:
    def write_markdown(self,cfg:Dict,metrics:Dict,recommendations:Dict,out_path:str|Path,detailed:Dict|None=None)->Path:
        p=Path(out_path); p.parent.mkdir(parents=True,exist_ok=True)
        suggestions='\n'.join(f'- {s}' for s in recommendations['suggestions']); detailed=detailed or {}; dsum=detailed.get('summary',{})
        detail_block=''
        if dsum:
            detail_block=f"""
## Modelo peça-a-peça e massa

- Palitos estimados com perdas: {dsum.get('estimated_total_sticks_with_waste')}
- Peças individuais: {dsum.get('total_piece_instances')}
- Palitos brutos antes de perdas: {dsum.get('estimated_blank_sticks_needed')}
- Área total de cola estimada: {float(dsum.get('estimated_glue_area_mm2',0.0)):.1f} mm²
- Massa de cola estimada: {float(dsum.get('estimated_glue_mass_g',0.0)):.1f} g
- Massa total estimada: {float(dsum.get('estimated_total_mass_g',0.0)):.1f} g
- Margem de massa: {float(dsum.get('mass_margin_g',0.0)):.1f} g
- Resistência de cisalhamento da cola adotada: {float(dsum.get('glue_shear_strength_MPa',0.0)):.2f} MPa

Arquivos detalhados: `outputs/details/stick_pieces.csv`, `glue_joints.csv`, `cutting_list.csv`, `blank_cut_plan.csv`, `member_detail_checks.csv`, `reinforcement_suggestions.csv`.
"""
        md=f"""# Relatório automático da simulação

## Configuração analisada

- Vão livre: {cfg['bridge']['span_mm']:.0f} mm
- Largura: {cfg['bridge']['width_mm']:.0f} mm
- Altura central: {cfg['bridge']['center_height_mm']:.0f} mm
- Carga aplicada: {cfg['bridge']['load_total_kgf']:.2f} kgf
- Módulo de elasticidade adotado: {cfg['material']['E_MPa']:.1f} MPa

## Métricas principais

- Nós: {metrics.get('n_nodes')}
- Membros: {metrics.get('n_members')}
- Apoios ativos: {metrics.get('n_active_supports')}
- Apoios com perda de contato: {metrics.get('n_uplift_supports')}
- Erro de equilíbrio vertical: {metrics.get('equilibrium_error_N'):.3e} N
- Menor FS em membros principais: {metrics.get('min_fs_primary')}
- Menor FS em todos os membros: {metrics.get('min_fs_all')}

## Análise automática

{recommendations['summary']}

## Sugestões de melhoria

{suggestions}

{detail_block}
## Imagens geradas

Veja a pasta `outputs/plots/`.

## Observações

Este relatório é preliminar. O modelo axial em NumPy calcula forças axiais, reações e deslocamentos sob hipótese linear. Flambagem é verificada por Euler. O detalhamento peça-a-peça estima cortes, emendas e cola; não substitui ensaio físico nem FEM de contato/cola.
"""
        p.write_text(md,encoding='utf-8'); return p
