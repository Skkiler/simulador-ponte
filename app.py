from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.services.cache_cleanup_service import CacheCleanupService
from src.services.config_service import ConfigService
from src.services.pipeline import SimulationPipeline
from src.services.visualization_service import VisualizationService


PROJECT_ROOT = Path(__file__).resolve().parent


def _cleanup_on_app_exit() -> None:
    cleaner = CacheCleanupService(PROJECT_ROOT)
    cleaner.cleanup_filesystem_cache()

    try:
        st.cache_data.clear()
    except Exception:
        pass

    try:
        st.cache_resource.clear()
    except Exception:
        pass


if os.environ.get("PONTE_APP_CACHE_CLEANUP_REGISTERED") != "1":
    atexit.register(_cleanup_on_app_exit)
    os.environ["PONTE_APP_CACHE_CLEANUP_REGISTERED"] = "1"


st.set_page_config(
    page_title="Planejador Ativo de Ponte de Palitos",
    layout="wide",
)

st.markdown(
    """
<style>
:root {
  --card-bg: #f4f8fb;
  --card-border: #d8e4ec;
  --accent: #0e5a6d;
  --accent-soft: #cde6ee;
}

.block-container {
  max-width: 1540px;
  padding-top: 1.0rem;
  padding-bottom: 2.0rem;
}

div[data-testid="stMetric"] {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 12px;
  padding: 12px;
}

.hero {
  border: 1px solid var(--card-border);
  border-radius: 14px;
  background: linear-gradient(110deg, #edf6fa 0%, #f9fcfd 60%, #ffffff 100%);
  padding: 16px 18px;
  margin-bottom: 14px;
}

.hero h1 {
  color: #153848;
  margin: 0 0 4px 0;
  font-size: 1.55rem;
}

.hero p {
  margin: 0;
  color: #3d5a67;
  font-size: 0.95rem;
}

.section-title {
  color: var(--accent);
  font-weight: 600;
  margin-top: 0.2rem;
}

.small-note {
  color: #4e6570;
  font-size: 0.85rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>Planejador Ativo de Ponte de Palitos</h1>
  <p>
    Você define limites geométricos, massa máxima e propriedades dos materiais.
    O sistema gera propostas, filtra em múltiplas etapas e retorna o modelo mais viável com análise estrutural completa.
  </p>
</div>
""",
    unsafe_allow_html=True,
)


def _as_dataframe(data: Any) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()

    if isinstance(data, pd.DataFrame):
        return data.copy()

    try:
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()

    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _prepare_member_checks(data: Any) -> pd.DataFrame:
    df = _as_dataframe(data)

    if df.empty:
        return df

    numeric_cols = [
        "member_id",
        "N_N",
        "L_mm",
        "A_mm2",
        "Pcr_min_N",
        "FS_tension",
        "FS_compression_direct",
        "FS_buckling_y",
        "FS_buckling_z",
        "FS_min",
    ]
    df = _coerce_numeric(df, numeric_cols)

    if "member_id" in df.columns:
        df["member_id"] = df["member_id"].fillna(-1).astype(int)
    else:
        df["member_id"] = -1

    df["FS_min_num"] = pd.to_numeric(df.get("FS_min"), errors="coerce")
    df["FS_min_sort"] = df["FS_min_num"].fillna(1.0e99)

    if "member_role" not in df.columns:
        df["member_role"] = "unknown"

    if "risk_flag" not in df.columns:
        df["risk_flag"] = "—"

    if "group" not in df.columns:
        df["group"] = "—"

    return df.sort_values("FS_min_sort", ascending=True)


def _prepare_table(data: Any) -> pd.DataFrame:
    df = _as_dataframe(data)

    if df.empty:
        return df

    for col in df.columns:
        if col.startswith("FS") or col.endswith("_N") or col.endswith("_MPa") or col.endswith("_mm") or col.endswith("_g"):
            df[col] = pd.to_numeric(df[col], errors="ignore")

    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df = df.sort_values("score", ascending=False)

    if "FS_min" in df.columns:
        df["FS_min_num"] = pd.to_numeric(df["FS_min"], errors="coerce")

    return df


def _safe_metric(value: Any, default: str = "—") -> str:
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    return str(value)


def _format_float(value: Any, decimals: int = 2, default: str = "—") -> str:
    try:
        v = float(value)

        if pd.isna(v):
            return default

        return f"{v:.{decimals}f}"
    except Exception:
        return default


def _download_text_button(label: str, text: str, file_name: str) -> None:
    st.download_button(
        label,
        text,
        file_name=file_name,
        mime="application/json" if file_name.endswith(".json") else "text/plain",
    )


cs = ConfigService()
base_cfg = cs.load()
planner_cfg = base_cfg.get("planner", {})
analysis_cfg = base_cfg.get("analysis", {})
material_cfg = base_cfg.get("material", {})
bridge_cfg = base_cfg.get("bridge", {})
detail_cfg = base_cfg.get("detail_model", {})

with st.form("planner_form", clear_on_submit=False):
    st.markdown("<div class='section-title'>Objetivos e Restrições</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        target_load_kgf = st.number_input(
            "Carga de projeto [kgf]",
            min_value=1.0,
            max_value=1000.0,
            value=float(planner_cfg.get("target_load_kgf", bridge_cfg.get("load_total_kgf", 120.0))),
            step=5.0,
            help="Carga aplicada no modelo durante as simulações.",
        )
        target_breaking_load_kgf = st.number_input(
            "Carga de ruptura alvo [kgf]",
            min_value=1.0,
            max_value=2000.0,
            value=float(planner_cfg.get("target_breaking_load_kgf", bridge_cfg.get("load_total_kgf", 120.0))),
            step=5.0,
            help="Meta de capacidade de ruptura estimada (aprox. carga_projeto x FS).",
        )
        target_min_fs = st.number_input(
            "FS mínimo alvo",
            min_value=1.0,
            max_value=10.0,
            value=float(analysis_cfg.get("target_min_fs", 2.0)),
            step=0.1,
        )

    with c2:
        max_bridge_mass_g = st.number_input(
            "Massa máxima da ponte [g]",
            min_value=50.0,
            max_value=4000.0,
            value=float(planner_cfg.get("max_bridge_mass_g", material_cfg.get("mass_limit_g", 1000.0))),
            step=20.0,
        )
        target_bridge_mass_g = st.number_input(
            "Massa alvo da ponte [g]",
            min_value=50.0,
            max_value=4000.0,
            value=float(planner_cfg.get("target_bridge_mass_g", material_cfg.get("mass_limit_g", 1000.0) * 0.85)),
            step=20.0,
            help="Usada na função de score para buscar um compromisso entre resistência e peso.",
        )
        stage1_variants = st.number_input(
            "Quantidade de propostas iniciais",
            min_value=40,
            max_value=1200,
            value=int(analysis_cfg.get("planner_stage1_variants", 220)),
            step=20,
            help="Mais propostas aumentam cobertura de busca e tempo de processamento.",
        )
        adaptive_iterations = st.number_input(
            "Iterações de refinamento adaptativo",
            min_value=1,
            max_value=20,
            value=int(analysis_cfg.get("planner_stage4_iterations", 8)),
            step=1,
            help="Quantidade máxima de ajustes automáticos orientados por membros críticos.",
        )

    with c3:
        run_frame3dd = st.checkbox(
            "Rodar validação Frame3DD",
            value=bool(analysis_cfg.get("run_frame3dd_if_available", True)),
        )
        generate_piece_views = st.checkbox(
            "Gerar visualizações peça-a-peça",
            value=bool(detail_cfg.get("generate_piece_views", True)),
        )
        optimize_variants = st.checkbox(
            "Ativar planejamento multiestágio",
            value=bool(analysis_cfg.get("optimize_variants", True)),
        )
        adaptive_refinement = st.checkbox(
            "Ativar refinamento adaptativo (S4)",
            value=bool(analysis_cfg.get("planner_adaptive_refinement", True)),
        )
        objective_profile = st.selectbox(
            "Estratégia de objetivo",
            ["balanced", "max_strength", "min_mass"],
            index=["balanced", "max_strength", "min_mass"].index(
                str(analysis_cfg.get("planner_objective_profile", "balanced"))
                if str(analysis_cfg.get("planner_objective_profile", "balanced")) in {"balanced", "max_strength", "min_mass"}
                else "balanced"
            ),
            help=(
                "balanced: equilíbrio entre segurança e massa; "
                "max_strength: prioriza atingir carga/FS; "
                "min_mass: prioriza leveza com segurança mínima."
            ),
        )

        with st.expander("Pesos da função objetivo (avançado)", expanded=False):
            w_fs = st.slider(
                "Peso FS",
                min_value=0.0,
                max_value=1.0,
                value=float(analysis_cfg.get("planner_objective_weight_fs", 0.52)),
                step=0.01,
            )
            w_break = st.slider(
                "Peso carga de ruptura",
                min_value=0.0,
                max_value=1.0,
                value=float(analysis_cfg.get("planner_objective_weight_break", 0.28)),
                step=0.01,
            )
            w_mass_target = st.slider(
                "Peso massa alvo",
                min_value=0.0,
                max_value=1.0,
                value=float(analysis_cfg.get("planner_objective_weight_mass_target", 0.12)),
                step=0.01,
            )
            w_mass_limit = st.slider(
                "Peso massa limite",
                min_value=0.0,
                max_value=1.0,
                value=float(analysis_cfg.get("planner_objective_weight_mass_limit", 0.08)),
                step=0.01,
            )

    st.markdown("<div class='section-title'>Envelope Geométrico</div>", unsafe_allow_html=True)
    g1, g2, g3, g4 = st.columns(4)

    with g1:
        span_min_mm = st.number_input(
            "Vão mínimo [mm]",
            min_value=300.0,
            max_value=6000.0,
            value=float(planner_cfg.get("span_min_mm", bridge_cfg.get("span_mm", 1200.0) * 0.85)),
            step=10.0,
        )
        span_max_mm = st.number_input(
            "Vão máximo [mm]",
            min_value=300.0,
            max_value=6000.0,
            value=float(planner_cfg.get("span_max_mm", bridge_cfg.get("span_mm", 1200.0) * 1.15)),
            step=10.0,
        )

    with g2:
        width_min_mm = st.number_input(
            "Largura mínima [mm]",
            min_value=60.0,
            max_value=600.0,
            value=float(planner_cfg.get("width_min_mm", bridge_cfg.get("width_mm", 180.0) * 0.85)),
            step=5.0,
        )
        width_max_mm = st.number_input(
            "Largura máxima [mm]",
            min_value=60.0,
            max_value=600.0,
            value=float(planner_cfg.get("width_max_mm", bridge_cfg.get("width_mm", 180.0) * 1.2)),
            step=5.0,
        )

    with g3:
        height_min_mm = st.number_input(
            "Altura mínima [mm]",
            min_value=50.0,
            max_value=1200.0,
            value=float(planner_cfg.get("height_min_mm", bridge_cfg.get("center_height_mm", 300.0) * 0.75)),
            step=5.0,
        )
        height_max_mm = st.number_input(
            "Altura máxima [mm]",
            min_value=50.0,
            max_value=1200.0,
            value=float(planner_cfg.get("height_max_mm", bridge_cfg.get("center_height_mm", 300.0) * 1.25)),
            step=5.0,
        )

    with g4:
        panel_min_mm = st.number_input(
            "Painel mínimo [mm]",
            min_value=30.0,
            max_value=300.0,
            value=float(planner_cfg.get("panel_min_mm", bridge_cfg.get("panel_mm", 100.0) * 0.85)),
            step=5.0,
        )
        panel_max_mm = st.number_input(
            "Painel máximo [mm]",
            min_value=30.0,
            max_value=300.0,
            value=float(planner_cfg.get("panel_max_mm", bridge_cfg.get("panel_mm", 100.0) * 1.2)),
            step=5.0,
        )

    st.markdown("<div class='section-title'>Material dos Palitos</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        E_MPa = st.number_input(
            "Módulo E [MPa]",
            min_value=500.0,
            max_value=50000.0,
            value=float(material_cfg.get("E_MPa", 6000.0)),
            step=100.0,
        )

    with m2:
        stick_length_mm = st.number_input(
            "Comprimento do palito [mm]",
            min_value=60.0,
            max_value=400.0,
            value=float(material_cfg.get("stick_length_mm", 120.0)),
            step=1.0,
        )
        stick_mass_g = st.number_input(
            "Massa por palito [g]",
            min_value=0.1,
            max_value=10.0,
            value=float(material_cfg.get("stick_mass_g", 1.4)),
            step=0.1,
        )

    with m3:
        stick_width_mm = st.number_input(
            "Largura do palito [mm]",
            min_value=2.0,
            max_value=30.0,
            value=float(material_cfg.get("stick_width_mm", 7.0)),
            step=0.1,
        )
        stick_thickness_mm = st.number_input(
            "Espessura do palito [mm]",
            min_value=0.3,
            max_value=10.0,
            value=float(material_cfg.get("stick_thickness_mm", 1.5)),
            step=0.1,
        )

    with m4:
        tension_capacity_per_stick_kgf = st.number_input(
            "Tração por palito [kgf]",
            min_value=0.5,
            max_value=300.0,
            value=float(material_cfg.get("tension_capacity_per_stick_kgf", 72.0)),
            step=0.5,
        )
        compression_capacity_one_stick_kgf = st.number_input(
            "Compressão 1 palito [kgf]",
            min_value=0.1,
            max_value=200.0,
            value=float(material_cfg.get("compression_capacity_one_stick_kgf", 4.0)),
            step=0.1,
        )
        compression_capacity_two_sticks_kgf = st.number_input(
            "Compressão 2 palitos colados [kgf]",
            min_value=0.1,
            max_value=400.0,
            value=float(material_cfg.get("compression_capacity_two_sticks_kgf", 11.0)),
            step=0.1,
        )

    st.markdown("<div class='section-title'>Cola e Busca de Topologias</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)

    with d1:
        overlap_length_mm = st.number_input(
            "Sobreposição de emenda [mm]",
            min_value=5.0,
            max_value=max(200.0, float(stick_length_mm) * 0.9),
            value=float(detail_cfg.get("overlap_length_mm", 30.0)),
            step=1.0,
        )
        glue_shear_strength_MPa = st.number_input(
            "Resistência da cola ao cisalhamento [MPa]",
            min_value=0.2,
            max_value=50.0,
            value=float(detail_cfg.get("glue_shear_strength_MPa", 3.5)),
            step=0.1,
        )

    with d2:
        side_trusses = st.multiselect(
            "Treliças laterais candidatas",
            ["Parker", "Pratt", "Howe", "Warren"],
            default=list(planner_cfg.get("consider_side_trusses", ["Parker", "Pratt", "Howe", "Warren"])),
        )
        top_profiles = st.multiselect(
            "Perfis de topo candidatos",
            ["parker_plateau", "triangular_peak", "shallow_arch", "flat"],
            default=list(planner_cfg.get("consider_top_profiles", ["parker_plateau", "triangular_peak", "shallow_arch", "flat"])),
        )

    with d3:
        internal_trusses = st.multiselect(
            "Treliças internas candidatas",
            ["X", "Warren", "Pratt", "Howe", "none"],
            default=list(planner_cfg.get("consider_internal_trusses", ["X", "Warren", "Pratt", "Howe", "none"])),
        )
        chord_trusses = st.multiselect(
            "Lacing de banzo candidato",
            ["none", "Warren", "X"],
            default=list(planner_cfg.get("consider_chord_trusses", ["none", "Warren", "X"])),
        )

    st.markdown(
        "<p class='small-note'>O algoritmo usa busca em etapas: varredura ampla, refinamento de palitos por grupo e validação detalhada com massa/cola.</p>",
        unsafe_allow_html=True,
    )

    run = st.form_submit_button(
        "Planejar e analisar modelo ideal",
        type="primary",
        use_container_width=True,
    )

if run:
    if span_min_mm > span_max_mm or width_min_mm > width_max_mm or height_min_mm > height_max_mm or panel_min_mm > panel_max_mm:
        st.error("Limites geométricos inválidos: cada mínimo deve ser menor ou igual ao máximo.")
        st.stop()

    if not side_trusses or not top_profiles or not internal_trusses or not chord_trusses:
        st.error("Selecione ao menos uma opção em cada lista de topologias candidatas.")
        st.stop()

    cfg = cs.from_planner_inputs(
        base_cfg,
        target_load_kgf=target_load_kgf,
        span_min_mm=span_min_mm,
        span_max_mm=span_max_mm,
        width_min_mm=width_min_mm,
        width_max_mm=width_max_mm,
        height_min_mm=height_min_mm,
        height_max_mm=height_max_mm,
        panel_min_mm=panel_min_mm,
        panel_max_mm=panel_max_mm,
        max_bridge_mass_g=max_bridge_mass_g,
        target_bridge_mass_g=target_bridge_mass_g,
        E_MPa=E_MPa,
        stick_length_mm=stick_length_mm,
        stick_width_mm=stick_width_mm,
        stick_thickness_mm=stick_thickness_mm,
        stick_mass_g=stick_mass_g,
        tension_capacity_per_stick_kgf=tension_capacity_per_stick_kgf,
        compression_capacity_one_stick_kgf=compression_capacity_one_stick_kgf,
        compression_capacity_two_sticks_kgf=compression_capacity_two_sticks_kgf,
        glue_shear_strength_MPa=glue_shear_strength_MPa,
        overlap_length_mm=overlap_length_mm,
        target_min_fs=target_min_fs,
        stage1_variants=stage1_variants,
        objective_profile=objective_profile,
        adaptive_refinement=adaptive_refinement,
        adaptive_iterations=int(adaptive_iterations),
    )

    cfg["planner"]["target_breaking_load_kgf"] = float(target_breaking_load_kgf)
    cfg["planner"]["consider_side_trusses"] = side_trusses
    cfg["planner"]["consider_top_profiles"] = top_profiles
    cfg["planner"]["consider_internal_trusses"] = internal_trusses
    cfg["planner"]["consider_chord_trusses"] = chord_trusses

    cfg["analysis"]["optimize_variants"] = bool(optimize_variants)
    cfg["analysis"]["active_planner_enabled"] = bool(optimize_variants)
    cfg["analysis"]["run_frame3dd_if_available"] = bool(run_frame3dd)
    cfg["analysis"]["planner_stage1_variants"] = int(stage1_variants)
    cfg["analysis"]["planner_stage4_iterations"] = int(adaptive_iterations)
    cfg["analysis"]["planner_adaptive_refinement"] = bool(adaptive_refinement)
    cfg["analysis"]["planner_objective_profile"] = str(objective_profile)
    cfg["analysis"]["planner_objective_weight_fs"] = float(w_fs)
    cfg["analysis"]["planner_objective_weight_break"] = float(w_break)
    cfg["analysis"]["planner_objective_weight_mass_target"] = float(w_mass_target)
    cfg["analysis"]["planner_objective_weight_mass_limit"] = float(w_mass_limit)
    cfg["analysis"]["final_variants_enabled"] = True
    cfg["detail_model"]["generate_piece_views"] = bool(generate_piece_views)

    with st.spinner("Executando busca multiestágio e análise estrutural completa..."):
        st.session_state["last_result"] = SimulationPipeline("outputs").run(cfg)

r = st.session_state.get("last_result")

if not r:
    st.info("Preencha os limites e clique em Planejar e analisar modelo ideal.")
    st.stop()

metrics = r.get("metrics", {})
dsum = r.get("detailed", {}).get("summary", {})
recommendations = r.get("recommendations", {})
frame3dd_result = r.get("frame3dd_result", {})
opt = r.get("optimization") or {}
best = opt.get("best") or {}
stage_counts = opt.get("stage_counts") or {}
final_variants = opt.get("final_variants") or {}

model_label = (
    f"{best.get('side_truss_type', r.get('cfg', {}).get('bridge', {}).get('side_truss_type', '—'))} / "
    f"{best.get('top_profile', r.get('cfg', {}).get('bridge', {}).get('top_profile', '—'))}"
)

cols = st.columns(8)
cols[0].metric("Modelo", model_label)
cols[1].metric("FS principal", _format_float(metrics.get("min_fs_primary"), 2))
cols[2].metric("Ruptura estimada", f"{_format_float(metrics.get('predicted_breaking_load_kgf'), 1, '—')} kgf")
cols[3].metric("Massa estimada", f"{_format_float(dsum.get('estimated_total_mass_g'), 0, '0')} g")
cols[4].metric("Margem de massa", f"{_format_float(dsum.get('mass_margin_g'), 0, '0')} g")
cols[5].metric("Palitos", _safe_metric(dsum.get("estimated_total_sticks_with_waste")))
cols[6].metric("Solver", _safe_metric(metrics.get("solver_status")))
cols[7].metric("Frame3DD", _safe_metric(frame3dd_result.get("status")))

st.caption(
    f"Busca: S1={stage_counts.get('stage1', 0)} | S2={stage_counts.get('stage2', 0)} | "
    f"S3={stage_counts.get('stage3', 0)} | S4-trace={stage_counts.get('stage4_trace', 0)} | "
    f"S4={stage_counts.get('stage4', 0)} | finais={stage_counts.get('final_variants', 0)}"
)

summary = recommendations.get("summary")
if summary:
    st.write(summary)

viz = VisualizationService()

tab_resumo, tab_variantes, tab_visual, tab_criticos, tab_detalhes, tab_export = st.tabs(
    [
        "Resumo e Recomendações",
        "Etapas de Busca",
        "Visualizações 3D/2D",
        "Membros Críticos",
        "Montagem e Cola",
        "Frame3DD e Downloads",
    ]
)

with tab_resumo:
    suggestions = recommendations.get("suggestions", [])

    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            st.write(f"**{i}.** {suggestion}")
    else:
        st.info("Nenhuma recomendação textual foi gerada.")

    st.subheader("Configuração escolhida")
    st.json(r.get("cfg", {}), expanded=False)

    if final_variants:
        st.subheader("Versões finais da sugestão")
        final_rows = []
        for label in ("ideal", "min", "max"):
            row = final_variants.get(label)
            if not row:
                continue
            final_rows.append(
                {
                    "versao": label,
                    "FS_min_principal": row.get("min_fs_primary"),
                    "carga_ruptura_estimada_kgf": row.get("predicted_breaking_load_kgf"),
                    "massa_g": row.get("mass_g"),
                    "solver_status": row.get("solver_status"),
                    "viavel": row.get("feasible"),
                    "span_mm": row.get("span_mm"),
                    "width_mm": row.get("width_mm"),
                    "center_height_mm": row.get("center_height_mm"),
                    "panel_mm": row.get("panel_mm"),
                }
            )

        if final_rows:
            st.dataframe(_prepare_table(final_rows), use_container_width=True)

with tab_variantes:
    stage_map = {
        "Stage 1 - varredura": opt.get("stage1", []),
        "Stage 2 - refinamento": opt.get("stage2", []),
        "Stage 3 - validação detalhada": opt.get("stage3", []),
        "Stage 4 trace - ajustes iterativos": opt.get("stage4_trace", []),
        "Stage 4 - candidatos adaptados validados": opt.get("stage4", []),
        "Final - ideal/min/max": list(final_variants.values()),
    }

    stage_pick = st.selectbox("Etapa", list(stage_map))
    stage_rows = stage_map.get(stage_pick, [])

    if stage_rows:
        table = _prepare_table([{k: v for k, v in row.items() if k != "config"} for row in stage_rows])
        st.dataframe(table, use_container_width=True)
    else:
        st.info("Sem dados para a etapa selecionada.")

with tab_visual:
    v1, v2 = st.columns([2, 1])

    with v1:
        scale_mode = st.radio(
            "Escala do 3D",
            ["didactic", "real", "cube"],
            index=0,
            horizontal=True,
        )

        st.plotly_chart(
            viz.plotly_geometry(
                r["nodes"],
                r["members"],
                r["supports"],
                r["loads"],
                scale_mode=scale_mode,
            ),
            use_container_width=True,
        )

    with v2:
        st.markdown("**Planos e treliças (2D)**")
        plot_dir = Path("outputs/plots")
        preferred = [
            "02_trelica_lateral_esquerda.png",
            "03_trelica_lateral_direita.png",
            "04_plano_superior.png",
            "05_plano_inferior.png",
        ]

        shown = 0
        for name in preferred:
            p = plot_dir / name
            if p.exists():
                st.image(str(p), caption=name)
                shown += 1

        if shown == 0:
            st.info("Imagens 2D não encontradas em outputs/plots.")

with tab_criticos:
    member_df = _prepare_member_checks(r.get("member_checks", []))

    if member_df.empty:
        st.info("Nenhum membro disponível.")
    else:
        show_cols = [
            c
            for c in [
                "member_id",
                "group",
                "member_role",
                "state",
                "N_N",
                "L_mm",
                "FS_min",
                "FS_min_label",
                "governing_mode",
                "report_mode",
                "risk_flag",
            ]
            if c in member_df.columns
        ]

        st.dataframe(member_df[show_cols].head(80), use_container_width=True)

        critical_first = (
            member_df[member_df["member_id"] >= 0]
            .sort_values("FS_min_sort", ascending=True)["member_id"]
            .drop_duplicates()
            .tolist()
        )

        if critical_first:
            member_id = st.selectbox("Destacar membro", critical_first, index=0)

            st.plotly_chart(
                viz.plotly_geometry(
                    r["nodes"],
                    r["members"],
                    r["supports"],
                    r["loads"],
                    highlight_member_ids=[int(member_id)],
                    scale_mode="didactic",
                ),
                use_container_width=True,
            )

            st.dataframe(
                member_df[member_df["member_id"] == int(member_id)][show_cols],
                use_container_width=True,
            )

with tab_detalhes:
    pieces_all = r.get("detailed", {}).get("stick_pieces", [])

    if pieces_all:
        only_member = st.checkbox("Mostrar apenas membro selecionado", value=False)
        member_selected = None

        if only_member:
            mdf = _prepare_member_checks(r.get("member_checks", []))
            mids = mdf["member_id"].drop_duplicates().tolist() if not mdf.empty else []

            if mids:
                member_selected = st.selectbox("Membro", mids, index=0, key="detail_member_pick")

        st.plotly_chart(
            viz.plotly_stick_pieces(
                pieces_all,
                member_id=int(member_selected) if member_selected is not None else None,
                max_pieces=2000,
            ),
            use_container_width=True,
        )
    else:
        st.info("Sem dados peça-a-peça para visualizar.")

    glue_df = _prepare_table(r.get("detailed", {}).get("weakest_glue_joints", []))
    if not glue_df.empty:
        st.markdown("**Juntas coladas mais críticas**")
        st.dataframe(glue_df.head(40), use_container_width=True)

    cuts_df = _prepare_table(r.get("detailed", {}).get("cutting_list", []))
    if not cuts_df.empty:
        st.markdown("**Lista de cortes**")
        st.dataframe(cuts_df.head(40), use_container_width=True)

with tab_export:
    st.subheader("Frame3DD")
    st.json(frame3dd_result)

    out_file = Path("outputs/frame3dd/ponte_palitos.out")
    if out_file.exists():
        st.text_area(
            "Saída Frame3DD",
            out_file.read_text(encoding="utf-8", errors="ignore")[:40000],
            height=350,
        )

    st.subheader("Downloads")

    zp = Path(r.get("zip_path", ""))
    if zp.exists():
        st.download_button(
            "Baixar pacote de resultados",
            zp.read_bytes(),
            file_name="resultados_simulacao.zip",
            mime="application/zip",
        )

    _download_text_button(
        "Baixar configuração analisada",
        json.dumps(r.get("cfg", {}), indent=2, ensure_ascii=False),
        "config_used.json",
    )

    _download_text_button(
        "Baixar configuração solicitada",
        json.dumps(r.get("input_cfg", {}), indent=2, ensure_ascii=False),
        "config_requested.json",
    )

    if opt.get("recommended_config_path"):
        rec_path = Path(opt["recommended_config_path"])
        if rec_path.exists():
            _download_text_button(
                "Baixar configuração recomendada",
                rec_path.read_text(encoding="utf-8"),
                "recommended_config.json",
            )

    if opt.get("recommended_config_ideal_path"):
        p = Path(opt["recommended_config_ideal_path"])
        if p.exists():
            _download_text_button(
                "Baixar versão IDEAL",
                p.read_text(encoding="utf-8"),
                "recommended_config_ideal.json",
            )

    if opt.get("recommended_config_min_path"):
        p = Path(opt["recommended_config_min_path"])
        if p.exists():
            _download_text_button(
                "Baixar versão MIN",
                p.read_text(encoding="utf-8"),
                "recommended_config_min.json",
            )

    if opt.get("recommended_config_max_path"):
        p = Path(opt["recommended_config_max_path"])
        if p.exists():
            _download_text_button(
                "Baixar versão MAX",
                p.read_text(encoding="utf-8"),
                "recommended_config_max.json",
            )
