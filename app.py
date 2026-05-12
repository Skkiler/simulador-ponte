from __future__ import annotations

import atexit
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from src.core.numeric import safe_float
from src.services.cache_cleanup_service import CacheCleanupService
from src.services.config_service import ConfigService
from src.services.mass_guard import resolve_mass_limits
from src.services.pipeline import SimulationPipeline
from src.services.visualization_service import VisualizationService


PROJECT_ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)

THEMES = {
    "Oceano Claro": {
        "card_bg": "#f4f8fb",
        "card_border": "#d8e4ec",
        "accent": "#0e5a6d",
        "accent_soft": "#cde6ee",
        "hero_a": "#edf6fa",
        "hero_b": "#f9fcfd",
        "hero_c": "#ffffff",
        "hero_title": "#153848",
        "hero_text": "#3d5a67",
        "metric_label": "#4b6474",
        "metric_value": "#153848",
        "tab_active": "#0e5a6d",
        "tab_text": "#243b47",
    },
    "Areia Técnica": {
        "card_bg": "#fbf7ef",
        "card_border": "#e7dcc7",
        "accent": "#875f1a",
        "accent_soft": "#efe2c6",
        "hero_a": "#fff8ea",
        "hero_b": "#fffefb",
        "hero_c": "#ffffff",
        "hero_title": "#4d3c1f",
        "hero_text": "#69553a",
        "metric_label": "#7b684b",
        "metric_value": "#4d3c1f",
        "tab_active": "#875f1a",
        "tab_text": "#4d3c1f",
    },
    "Concreto Frio": {
        "card_bg": "#f3f5f7",
        "card_border": "#d9dee4",
        "accent": "#375366",
        "accent_soft": "#d7e3ec",
        "hero_a": "#eff3f7",
        "hero_b": "#f8fbfd",
        "hero_c": "#ffffff",
        "hero_title": "#273a47",
        "hero_text": "#4e616f",
        "metric_label": "#516472",
        "metric_value": "#273a47",
        "tab_active": "#375366",
        "tab_text": "#273a47",
    },
    "Noite Carbono": {
        "card_bg": "#141b22",
        "card_border": "#273442",
        "accent": "#7fc3ff",
        "accent_soft": "#1f2f3f",
        "hero_a": "#0f151c",
        "hero_b": "#111c28",
        "hero_c": "#152534",
        "hero_title": "#d9ecff",
        "hero_text": "#a9c4dc",
        "metric_label": "#9ebad1",
        "metric_value": "#e7f3ff",
        "tab_active": "#7fc3ff",
        "tab_text": "#dbefff",
    },
}

LENGTH_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
KGF_TO_FORCE_UNIT = {"gf": 1000.0, "kgf": 1.0, "tgf": 0.001}
FORCE_UNIT_TO_KGF = {"gf": 0.001, "kgf": 1.0, "tgf": 1000.0}

TOP_PROFILE_LABEL_TO_CODE = {
    "pontiagudo/triangular": "triangular_peak",
    "platô": "parker_plateau",
    "arco": "shallow_arch",
    "reto": "flat",
}
TOP_PROFILE_CODE_TO_LABEL = {v: k for k, v in TOP_PROFILE_LABEL_TO_CODE.items()}

SIDE_TRUSS_OPTIONS = [
    "Parker",
    "Pratt",
    "Howe",
    "Warren",
    "K",
    "Baltimore",
    "Howe_inverted",
    "Warren_mid_braced",
    "Pratt_symmetric",
    "Warren_symmetric",
    "K_symmetric",
]
PLANE_TRUSS_OPTIONS = [
    "X",
    "Warren",
    "Pratt",
    "Howe",
    "K",
    "N",
    "none",
    "Howe_inverted",
    "Warren_mid_braced",
    "Pratt_symmetric",
    "Warren_symmetric",
    "K_symmetric",
]

JOINT_MODEL_LABEL_TO_CODE = {
    "ponta a ponta simples": "butt_plain",
    "sobreposto simples": "single_lap",
    "ponta a ponta com talas curtas": "butt_small_splints",
    "ponta a ponta com talas longas": "butt_full_splints",
    "sobreposto com tala": "single_lap_tala",
    "duplo sobreposto": "double_lap",
    "duplo sobreposto reforçado": "double_lap_reinforced",
    "emenda chanfrada (scarf)": "scarf",
    "meia-madeira recortada": "half_lap_notched",
}
JOINT_MODEL_CODE_TO_LABEL = {v: k for k, v in JOINT_MODEL_LABEL_TO_CODE.items()}

OBJECTIVE_LABEL_TO_CODE = {
    "Balanceado": "balanced",
    "Máxima resistência": "max_strength",
    "Mínima massa": "min_mass",
}
OBJECTIVE_CODE_TO_LABEL = {v: k for k, v in OBJECTIVE_LABEL_TO_CODE.items()}

COL_PT = {
    "stage": "etapa",
    "candidate_id": "id_candidato",
    "side_truss_type": "treliça_lateral",
    "top_profile": "perfil_topo",
    "internal_truss_type": "treliça_interna",
    "top_chord_truss_type": "treliça_banzo_superior",
    "bottom_chord_truss_type": "treliça_banzo_inferior",
    "reinforcement_profile": "perfil_reforço",
    "tension_joint_model": "emenda_tração",
    "compression_joint_model": "emenda_compressão",
    "splice_mode": "modo_emenda",
    "score": "pontuação",
    "feasible": "viável",
    "solver_status": "status_solver",
    "equilibrium_error_N": "erro_equilíbrio_N",
    "min_fs_primary": "FS_mín_principal",
    "min_fs_all": "FS_mín_geral",
    "mass_g": "massa_g",
    "quick_mass_g": "massa_rápida_g",
    "estimated_sticks": "palitos_estimados",
    "predicted_breaking_load_kgf": "carga_ruptura_prevista_kgf",
    "inactive_support_count": "apoios_sem_contato",
    "critical_members": "membros_críticos",
    "adaptive_seed": "semente_adaptativa",
    "adaptive_iteration": "iteração_adaptativa",
    "adaptive_actions": "ações_adaptativas",
    "discard_reason": "motivo_descarte",
}


st.set_page_config(
    page_title="Planejador Ativo de Ponte de Palitos",
    layout="wide",
)


def _cleanup_on_app_exit() -> None:
    cleaner = CacheCleanupService(PROJECT_ROOT)
    cleaner.cleanup_filesystem_cache()

    try:
        st.cache_data.clear()
    except RuntimeError as exc:
        LOGGER.warning("Falha ao limpar st.cache_data: %r", exc)

    try:
        st.cache_resource.clear()
    except RuntimeError as exc:
        LOGGER.warning("Falha ao limpar st.cache_resource: %r", exc)


if os.environ.get("PONTE_APP_CACHE_CLEANUP_REGISTERED") != "1":
    atexit.register(_cleanup_on_app_exit)
    os.environ["PONTE_APP_CACHE_CLEANUP_REGISTERED"] = "1"


if "tema_ui" not in st.session_state:
    st.session_state["tema_ui"] = "Oceano Claro"
if "unidade_comprimento" not in st.session_state:
    st.session_state["unidade_comprimento"] = "mm"
if "unidade_forca" not in st.session_state:
    st.session_state["unidade_forca"] = "kgf"


def _apply_theme(theme_name: str) -> None:
    theme = THEMES.get(theme_name, THEMES["Oceano Claro"])
    st.markdown(
        f"""
<style>
:root {{
  --card-bg: {theme['card_bg']};
  --card-border: {theme['card_border']};
  --accent: {theme['accent']};
  --accent-soft: {theme['accent_soft']};
}}

.block-container {{
  max-width: 1560px;
  padding-top: 0.9rem;
  padding-bottom: 2rem;
}}

div[data-testid="stMetric"] {{
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 12px;
  padding: 12px;
}}

div[data-testid="stMetricLabel"] {{
  color: {theme['metric_label']} !important;
}}

div[data-testid="stMetricValue"] {{
  color: {theme['metric_value']} !important;
}}

.hero {{
  border: 1px solid var(--card-border);
  border-radius: 14px;
  background: linear-gradient(110deg, {theme['hero_a']} 0%, {theme['hero_b']} 60%, {theme['hero_c']} 100%);
  padding: 16px 18px;
  margin-bottom: 14px;
}}

.hero h1 {{
  color: {theme['hero_title']};
  margin: 0 0 4px 0;
  font-size: 1.55rem;
}}

.hero p {{
  margin: 0;
  color: {theme['hero_text']};
  font-size: 0.95rem;
}}

.section-title {{
  color: var(--accent);
  font-weight: 600;
  margin-top: 0.2rem;
}}

.small-note {{
  color: {theme['hero_text']};
  font-size: 0.85rem;
}}

button[data-baseweb="tab"] {{
  color: {theme['tab_text']} !important;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
  color: {theme['tab_active']} !important;
  border-bottom-color: {theme['tab_active']} !important;
  font-weight: 700 !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )


_apply_theme(st.session_state["tema_ui"])

st.markdown(
    """
<div class="hero">
  <h1>Planejador Ativo de Ponte de Palitos</h1>
  <p>
    Defina limites, materiais e metas. O sistema gera propostas, aplica filtros rápidos e detalhados,
    realiza cálculo estrutural e retorna uma solução final com relatório técnico completo.
  </p>
</div>
""",
    unsafe_allow_html=True,
)


def _to_display_length(value_mm: float, unit: str) -> float:
    return float(value_mm) / LENGTH_TO_MM[unit]


def _from_display_length(value: float, unit: str) -> float:
    return float(value) * LENGTH_TO_MM[unit]


def _to_display_force_from_kgf(value_kgf: float, unit: str) -> float:
    return float(value_kgf) * KGF_TO_FORCE_UNIT[unit]


def _from_display_force_to_kgf(value: float, unit: str) -> float:
    return float(value) * FORCE_UNIT_TO_KGF[unit]


def _to_display_force_from_N(value_N: float, unit: str) -> float:
    return _to_display_force_from_kgf(float(value_N) / 9.80665, unit)


def _format_force_kgf(value_kgf: Any, unit: str, decimals: int = 2, default: str = "—") -> str:
    try:
        v = float(value_kgf)
        if pd.isna(v):
            return default
        return f"{_to_display_force_from_kgf(v, unit):.{decimals}f} {unit}"
    except (TypeError, ValueError):
        return default


def _as_dataframe(data: Any) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()

    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        try:
            df = pd.DataFrame(data)
        except (TypeError, ValueError):
            return pd.DataFrame()

    # Evita falhas de serialização Arrow em colunas object com tipos mistos
    # (ex.: int + string vazia). Nestes casos forçamos string homogênea.
    for col in df.columns:
        if df[col].dtype != "object":
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        if len({type(v) for v in non_null}) > 1:
            df[col] = df[col].astype(str)
    return df


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
        df["member_role"] = "desconhecido"

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
        if col.startswith("FS") or col.endswith("_N") or col.endswith("_MPa") or col.endswith("_mm") or col.endswith("_g") or col.endswith("_kgf"):
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().any():
                df[col] = numeric.where(numeric.notna(), df[col])

    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df = df.sort_values("score", ascending=False)

    if "FS_min" in df.columns:
        df["FS_min_num"] = pd.to_numeric(df["FS_min"], errors="coerce")

    return df


def _translate_top_profile(code: Any) -> str:
    raw = str(code or "").strip().lower()
    return TOP_PROFILE_CODE_TO_LABEL.get(raw, str(code or "—"))


def _safe_metric(value: Any, default: str = "—") -> str:
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        return default

    return str(value)


def _solver_status_pt(value: Any) -> str:
    raw = str(value or "").strip()
    if raw == "":
        return "—"
    if raw == "regular":
        return "regular"
    if raw.startswith("singular"):
        return raw.replace("singular", "singular (solver)")
    if raw == "skipped":
        return "não executado"
    if raw == "ok":
        return "ok"
    return raw


def _describe_variant(row: Dict[str, Any], length_unit: str, force_unit: str) -> str:
    span = _to_display_length(float(row.get("span_mm", 0.0) or 0.0), length_unit)
    width = _to_display_length(float(row.get("width_mm", 0.0) or 0.0), length_unit)
    height = _to_display_length(float(row.get("center_height_mm", 0.0) or 0.0), length_unit)
    panel = _to_display_length(float(row.get("panel_mm", 0.0) or 0.0), length_unit)
    ruptura = _format_force_kgf(row.get("predicted_breaking_load_kgf"), force_unit, 1)
    fs = _format_float(row.get("min_fs_primary"), 2)
    massa = _format_float(row.get("mass_g"), 1)
    emenda_t = JOINT_MODEL_CODE_TO_LABEL.get(str(row.get("tension_joint_model", "")), str(row.get("tension_joint_model", "—")))
    emenda_c = JOINT_MODEL_CODE_TO_LABEL.get(str(row.get("compression_joint_model", "")), str(row.get("compression_joint_model", "—")))
    overlap = _format_float(_to_display_length(float(row.get("overlap_length_mm", 0.0) or 0.0), length_unit), 1)
    return (
        f"Treliça lateral {row.get('side_truss_type', '—')}, topo {_translate_top_profile(row.get('top_profile', '—'))}, "
        f"interna {row.get('internal_truss_type', '—')}, banzo sup. {row.get('top_chord_truss_type', '—')} "
        f"e banzo inf. {row.get('bottom_chord_truss_type', '—')}. "
        f"Emendas: tração {emenda_t}, compressão {emenda_c}, sobreposição {overlap} {length_unit}. "
        f"Dimensões: vão {span:.2f} {length_unit}, largura {width:.2f} {length_unit}, altura {height:.2f} {length_unit}, "
        f"painel {panel:.2f} {length_unit}. Desempenho: FS={fs}, ruptura prevista={ruptura}, massa={massa} g."
    )


def _format_float(value: Any, decimals: int = 2, default: str = "—") -> str:
    try:
        v = float(value)

        if pd.isna(v):
            return default

        return f"{v:.{decimals}f}"
    except (TypeError, ValueError):
        return default


def _download_text_button(label: str, text: str, file_name: str) -> None:
    st.download_button(
        label,
        text,
        file_name=file_name,
        mime="application/json" if file_name.endswith(".json") else "text/plain",
    )


def _df_to_display_units(df: pd.DataFrame, length_unit: str, force_unit: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    rename: dict[str, str] = {}

    for col in list(out.columns):
        col_lower = col.lower()

        if col.endswith("_mm"):
            num = pd.to_numeric(out[col], errors="coerce")
            if num.notna().any():
                out[col] = num / LENGTH_TO_MM[length_unit]
            rename[col] = col[:-3] + f"_{length_unit}"
            continue

        if col.endswith("_kgf"):
            num = pd.to_numeric(out[col], errors="coerce")
            if num.notna().any():
                out[col] = num * KGF_TO_FORCE_UNIT[force_unit]
            rename[col] = col[:-4] + f"_{force_unit}"
            continue

        if col.endswith("_N") or "reaction" in col_lower:
            num = pd.to_numeric(out[col], errors="coerce")
            if num.notna().any():
                out[col] = (num / 9.80665) * KGF_TO_FORCE_UNIT[force_unit]
                rename[col] = col[:-2] + f"_{force_unit}" if col.endswith("_N") else col

    out = out.rename(columns=rename)
    out = out.rename(columns={k: v for k, v in COL_PT.items() if k in out.columns})

    if "perfil_topo" in out.columns:
        out["perfil_topo"] = out["perfil_topo"].map(_translate_top_profile)

    if "emenda_tração" in out.columns:
        out["emenda_tração"] = out["emenda_tração"].map(lambda v: JOINT_MODEL_CODE_TO_LABEL.get(str(v), str(v)))
    if "emenda_compressão" in out.columns:
        out["emenda_compressão"] = out["emenda_compressão"].map(lambda v: JOINT_MODEL_CODE_TO_LABEL.get(str(v), str(v)))

    if "status_solver" in out.columns:
        out["status_solver"] = out["status_solver"].map(_solver_status_pt)
    elif "solver_status" in out.columns:
        out["solver_status"] = out["solver_status"].map(_solver_status_pt)

    return out


def _number_input_length(
    label: str,
    unit: str,
    value_mm: float,
    min_mm: float,
    max_mm: float,
    step_mm: float,
    help_text: str | None = None,
) -> float:
    factor = LENGTH_TO_MM[unit]
    val = st.number_input(
        f"{label} [{unit}]",
        min_value=float(min_mm / factor),
        max_value=float(max_mm / factor),
        value=float(value_mm / factor),
        step=float(step_mm / factor),
        help=help_text,
    )
    return float(val * factor)


def _number_input_force(
    label: str,
    unit: str,
    value_kgf: float,
    min_kgf: float,
    max_kgf: float,
    step_kgf: float,
    help_text: str | None = None,
) -> float:
    factor = KGF_TO_FORCE_UNIT[unit]
    val = st.number_input(
        f"{label} [{unit}]",
        min_value=float(min_kgf * factor),
        max_value=float(max_kgf * factor),
        value=float(value_kgf * factor),
        step=float(step_kgf * factor),
        help=help_text,
    )
    return float(val / factor)


cs = ConfigService()
base_cfg = cs.load()
planner_cfg = base_cfg.get("planner", {})
analysis_cfg = base_cfg.get("analysis", {})
material_cfg = base_cfg.get("material", {})
bridge_cfg = base_cfg.get("bridge", {})
detail_cfg = base_cfg.get("detail_model", {})

aba_simulacao, aba_config = st.tabs(["Simulação", "Configurações"])

with aba_config:
    st.markdown("<div class='section-title'>Configurações de Interface e Unidades</div>", unsafe_allow_html=True)
    cfg1, cfg2, cfg3 = st.columns(3)

    with cfg1:
        tema_ui = st.selectbox("Tema visual", list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state["tema_ui"]))

    with cfg2:
        unidade_comprimento = st.selectbox("Unidade de comprimento", ["mm", "cm", "m"], index=["mm", "cm", "m"].index(st.session_state["unidade_comprimento"]))

    with cfg3:
        unidade_forca = st.selectbox("Unidade de força", ["gf", "kgf", "tgf"], index=["gf", "kgf", "tgf"].index(st.session_state["unidade_forca"]))

    if (
        tema_ui != st.session_state["tema_ui"]
        or unidade_comprimento != st.session_state["unidade_comprimento"]
        or unidade_forca != st.session_state["unidade_forca"]
    ):
        st.session_state["tema_ui"] = tema_ui
        st.session_state["unidade_comprimento"] = unidade_comprimento
        st.session_state["unidade_forca"] = unidade_forca
        st.info("Configurações atualizadas. A interface será recarregada para aplicar o tema e as unidades.")
        st.rerun()

    st.caption(
        "Os cálculos internos seguem em mm/N/kgf para consistência numérica. "
        "As unidades selecionadas afetam entrada, tabelas, gráficos e textos exibidos."
    )

with aba_simulacao:
    length_unit = st.session_state["unidade_comprimento"]
    force_unit = st.session_state["unidade_forca"]

    with st.form("planner_form", clear_on_submit=False):
        st.markdown("<div class='section-title'>Objetivos e Restrições</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            target_load_kgf = _number_input_force(
                "Carga de projeto",
                force_unit,
                float(planner_cfg.get("target_load_kgf", bridge_cfg.get("load_total_kgf", 120.0))),
                1.0,
                1000.0,
                5.0,
                "Carga aplicada durante as simulações.",
            )
            target_breaking_load_kgf = _number_input_force(
                "Carga de ruptura alvo",
                force_unit,
                float(planner_cfg.get("target_breaking_load_kgf", bridge_cfg.get("load_total_kgf", 120.0))),
                1.0,
                2000.0,
                5.0,
                "Meta de capacidade de ruptura aproximada (carga x FS).",
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
            )
            stage1_variants = st.number_input(
                "Quantidade de propostas iniciais",
                min_value=40,
                max_value=1200,
                value=int(analysis_cfg.get("planner_stage1_variants", 220)),
                step=20,
            )
            adaptive_iterations = st.number_input(
                "Iterações de refinamento adaptativo",
                min_value=1,
                max_value=20,
                value=int(analysis_cfg.get("planner_stage4_iterations", 8)),
                step=1,
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

            objective_labels = list(OBJECTIVE_LABEL_TO_CODE.keys())
            objective_default_code = str(analysis_cfg.get("planner_objective_profile", "balanced"))
            objective_default_label = OBJECTIVE_CODE_TO_LABEL.get(objective_default_code, "Balanceado")
            objective_profile_label = st.selectbox(
                "Estratégia de objetivo",
                objective_labels,
                index=objective_labels.index(objective_default_label),
                help=(
                    "Balanceado: equilíbrio entre segurança e massa; "
                    "Máxima resistência: prioriza capacidade; "
                    "Mínima massa: prioriza leveza com segurança mínima."
                ),
            )
            objective_profile = OBJECTIVE_LABEL_TO_CODE[objective_profile_label]

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
                planner_threads = st.number_input(
                    "Threads do planejador (0=automático)",
                    min_value=0,
                    max_value=32,
                    value=int(analysis_cfg.get("planner_threads", 0)),
                    step=1,
                )

        st.markdown("<div class='section-title'>Envelope Geométrico</div>", unsafe_allow_html=True)
        g1, g2, g3, g4 = st.columns(4)

        with g1:
            span_min_mm = _number_input_length(
                "Vão mínimo",
                length_unit,
                float(planner_cfg.get("span_min_mm", 1200.0)),
                300.0,
                6000.0,
                10.0,
            )
            span_max_mm = _number_input_length(
                "Vão máximo",
                length_unit,
                float(planner_cfg.get("span_max_mm", 1200.0)),
                300.0,
                6000.0,
                10.0,
            )

        with g2:
            width_min_mm = _number_input_length(
                "Largura mínima",
                length_unit,
                float(planner_cfg.get("width_min_mm", 100.0)),
                60.0,
                600.0,
                5.0,
            )
            width_max_mm = _number_input_length(
                "Largura máxima",
                length_unit,
                float(planner_cfg.get("width_max_mm", 200.0)),
                60.0,
                600.0,
                5.0,
            )

        with g3:
            height_min_mm = _number_input_length(
                "Altura mínima",
                length_unit,
                float(planner_cfg.get("height_min_mm", 50.0)),
                50.0,
                1200.0,
                5.0,
            )
            height_max_mm = _number_input_length(
                "Altura máxima",
                length_unit,
                float(planner_cfg.get("height_max_mm", bridge_cfg.get("center_height_mm", 300.0) * 1.25)),
                50.0,
                1200.0,
                5.0,
            )

        with g4:
            panel_min_mm = _number_input_length(
                "Painel mínimo",
                length_unit,
                float(planner_cfg.get("panel_min_mm", bridge_cfg.get("panel_mm", 100.0) * 0.85)),
                30.0,
                300.0,
                5.0,
            )
            panel_max_mm = _number_input_length(
                "Painel máximo",
                length_unit,
                float(planner_cfg.get("panel_max_mm", bridge_cfg.get("panel_mm", 100.0) * 1.2)),
                30.0,
                300.0,
                5.0,
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
            stick_length_mm = _number_input_length(
                "Comprimento do palito",
                length_unit,
                float(material_cfg.get("stick_length_mm", 115.0)),
                60.0,
                400.0,
                1.0,
            )
            stick_mass_g = st.number_input(
                "Massa por palito [g]",
                min_value=0.1,
                max_value=10.0,
                value=float(material_cfg.get("stick_mass_g", 1.4)),
                step=0.1,
            )

        with m3:
            # Defaults for stick dimensions updated to 7.0 mm × 1.5 mm.  The ranges remain the same.
            stick_width_mm = _number_input_length(
                "Largura do palito",
                length_unit,
                float(material_cfg.get("stick_width_mm", 7.0)),
                2.0,
                30.0,
                0.1,
            )
            stick_thickness_mm = _number_input_length(
                "Espessura do palito",
                length_unit,
                float(material_cfg.get("stick_thickness_mm", 1.5)),
                0.3,
                10.0,
                0.1,
            )

        with m4:
            tension_capacity_per_stick_kgf = _number_input_force(
                "Tração por palito",
                force_unit,
                float(material_cfg.get("tension_capacity_per_stick_kgf", 72.0)),
                0.5,
                300.0,
                0.5,
            )
            compression_capacity_one_stick_kgf = _number_input_force(
                "Compressão 1 palito",
                force_unit,
                float(material_cfg.get("compression_capacity_one_stick_kgf", 4.0)),
                0.1,
                200.0,
                0.1,
            )
            compression_capacity_two_sticks_kgf = _number_input_force(
                "Compressão 2 palitos colados",
                force_unit,
                float(material_cfg.get("compression_capacity_two_sticks_kgf", 11.0)),
                0.1,
                400.0,
                0.1,
            )

        st.markdown("<div class='section-title'>Cola e Busca de Topologias</div>", unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)

        with d1:
            overlap_length_mm = float(detail_cfg.get("overlap_length_mm", 30.0))
            st.caption(
                "Sobreposição de emenda: calculada automaticamente pelo planejador a partir "
                "do comprimento do palito, resistência da cola e força por membro."
            )
            glue_shear_strength_MPa = st.number_input(
                "Resistência da cola ao cisalhamento [MPa]",
                min_value=0.2,
                max_value=50.0,
                value=float(detail_cfg.get("glue_shear_strength_MPa", 3.5)),
                step=0.1,
            )
            # Definições globais de emenda para tração/compressão foram removidas da UI.
            # O planejador de conexões escolherá automaticamente o melhor modelo de emenda
            # para cada membro.  Mantemos variáveis locais para compatibilidade mas fixamos
            # em duplo sobreposto reforçado, que serve como fallback quando o plano de conexões
            # não está disponível.
            tension_joint_model = "double_lap_reinforced"
            compression_joint_model = "double_lap_reinforced"

        top_profile_labels = list(TOP_PROFILE_LABEL_TO_CODE.keys())
        default_top_profiles = [
            TOP_PROFILE_CODE_TO_LABEL.get(code, "platô")
            for code in planner_cfg.get("consider_top_profiles", ["triangular_peak", "parker_plateau", "shallow_arch", "flat"])
            if code in TOP_PROFILE_CODE_TO_LABEL
        ]
        if not default_top_profiles:
            default_top_profiles = ["pontiagudo/triangular", "platô", "arco", "reto"]

        with d2:
            side_trusses = st.multiselect(
                "Treliças laterais candidatas",
                SIDE_TRUSS_OPTIONS,
                default=list(planner_cfg.get("consider_side_trusses", SIDE_TRUSS_OPTIONS)),
            )
            top_profile_labels_selected = st.multiselect(
                "Perfis de topo candidatos",
                top_profile_labels,
                default=default_top_profiles,
            )
            top_profiles = [TOP_PROFILE_LABEL_TO_CODE[v] for v in top_profile_labels_selected]

        with d3:
            internal_trusses = st.multiselect(
                "Treliças internas candidatas",
                PLANE_TRUSS_OPTIONS,
                default=list(planner_cfg.get("consider_internal_trusses", PLANE_TRUSS_OPTIONS)),
            )
            top_chord_trusses = st.multiselect(
                "Treliças do banzo superior",
                PLANE_TRUSS_OPTIONS,
                default=list(planner_cfg.get("consider_top_chord_trusses", PLANE_TRUSS_OPTIONS)),
            )
            bottom_chord_trusses = st.multiselect(
                "Treliças do banzo inferior",
                PLANE_TRUSS_OPTIONS,
                default=list(planner_cfg.get("consider_bottom_chord_trusses", PLANE_TRUSS_OPTIONS)),
            )

        st.markdown(
            "<p class='small-note'>Busca em múltiplas etapas com filtros rápidos (pré-cálculo), "
            "refinamento local e validação detalhada com massa, emendas e cola.</p>",
            unsafe_allow_html=True,
        )

        run = st.form_submit_button(
            "Planejar e analisar modelo ideal",
            type="primary",
            width="stretch",
        )

    if run:
        if span_min_mm > span_max_mm or width_min_mm > width_max_mm or height_min_mm > height_max_mm or panel_min_mm > panel_max_mm:
            st.error("Limites geométricos inválidos: cada mínimo deve ser menor ou igual ao máximo.")
            st.stop()

        if not side_trusses or not top_profiles or not internal_trusses or not top_chord_trusses or not bottom_chord_trusses:
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
            top_chord_truss_type=str(top_chord_trusses[0]),
            bottom_chord_truss_type=str(bottom_chord_trusses[0]),
            objective_profile=objective_profile,
            adaptive_refinement=adaptive_refinement,
            adaptive_iterations=int(adaptive_iterations),
        )

        cfg["planner"]["target_breaking_load_kgf"] = float(target_breaking_load_kgf)
        cfg["planner"]["consider_side_trusses"] = side_trusses
        cfg["planner"]["consider_top_profiles"] = top_profiles
        cfg["planner"]["consider_internal_trusses"] = internal_trusses
        cfg["planner"]["consider_top_chord_trusses"] = top_chord_trusses
        cfg["planner"]["consider_bottom_chord_trusses"] = bottom_chord_trusses

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
        cfg["analysis"]["planner_threads"] = int(planner_threads)
        cfg["analysis"]["strict_mass_acceptance"] = True
        cfg["analysis"]["final_variants_enabled"] = True
        cfg["detail_model"]["generate_piece_views"] = bool(generate_piece_views)
        cfg["detail_model"]["tension_joint_model"] = str(tension_joint_model)
        cfg["detail_model"]["compression_joint_model"] = str(compression_joint_model)

        progress = st.progress(0, text="Inicializando análise...")
        log_box = st.empty()
        ui_logs: list[str] = []

        def on_progress(value: float, text: str) -> None:
            progress.progress(int(max(0.0, min(1.0, value)) * 100), text=text)

        def on_log(msg: str) -> None:
            ui_logs.append(str(msg))
            log_box.code("\n".join(ui_logs[-30:]), language="text")

        with st.spinner("Executando busca multiestágio e análise estrutural completa..."):
            result = SimulationPipeline("outputs").run(cfg, progress_callback=on_progress, log_callback=on_log)
            st.session_state["last_result"] = result
            st.session_state["last_run_logs"] = result.get("execution_logs", ui_logs)

        progress.progress(100, text="Análise concluída")

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
    mass_limits = resolve_mass_limits(r.get("cfg", {}))
    strict_mass_ui = bool(r.get("cfg", {}).get("analysis", {}).get("strict_mass_acceptance", True))

    model_label = (
        f"{best.get('side_truss_type', r.get('cfg', {}).get('bridge', {}).get('side_truss_type', '—'))} / "
        f"{_translate_top_profile(best.get('top_profile', r.get('cfg', {}).get('bridge', {}).get('top_profile', '—')))}"
    )
    if strict_mass_ui and not bool(metrics.get("mass_constraint_passed", True)):
        model_label = "Sem solução conforme massa"

    cols = st.columns(8)
    cols[0].metric("Modelo", model_label)
    cols[1].metric("FS principal", _format_float(metrics.get("min_fs_primary"), 2))
    cols[2].metric("Ruptura estimada", _format_force_kgf(metrics.get("predicted_breaking_load_kgf"), force_unit, 1))
    cols[3].metric("Massa estimada", f"{_format_float(dsum.get('estimated_total_mass_g'), 0, '0')} g")
    cols[4].metric("Limite efetivo", f"{_format_float(mass_limits.get('effective_limit_g'), 0, '0')} g")
    cols[5].metric("Palitos", _safe_metric(dsum.get("estimated_total_sticks_with_waste")))
    cols[6].metric("Solver", _solver_status_pt(metrics.get("solver_status")))
    cols[7].metric("Frame3DD", _solver_status_pt(frame3dd_result.get("status")))
    st.caption(
        "Massa: "
        f"estimada={_format_float(dsum.get('estimated_total_mass_g'), 1)} g | "
        f"nominal={_format_float(mass_limits.get('nominal_limit_g'), 1)} g | "
        f"efetivo={_format_float(mass_limits.get('effective_limit_g'), 1)} g "
        f"({mass_limits.get('effective_source', '—')})."
    )

    st.caption(
        f"Busca: S0-gerados={stage_counts.get('stage0_generated', 0)} | "
        f"S0-aprovados={stage_counts.get('stage0_prefilter_passed', 0)} | "
        f"S1-avaliados={stage_counts.get('stage1_evaluated', 0)} | "
        f"S1-válidos={stage_counts.get('stage1', 0)} | "
        f"S2-gerados={stage_counts.get('stage2_generated', 0)} | "
        f"S2A={stage_counts.get('stage2a_selected', 0)} | "
        f"S2B={stage_counts.get('stage2b_evaluated', 0)} | "
        f"S2-únicos={stage_counts.get('stage2_unique', stage_counts.get('stage2', 0))} | "
        f"S3={stage_counts.get('stage3', 0)} | S4-trace={stage_counts.get('stage4_trace', 0)} | "
        f"S4={stage_counts.get('stage4', 0)} | finais={stage_counts.get('final_variants', 0)}"
    )

    if best and not bool(best.get("feasible")):
        st.warning(
            "Nenhuma proposta atendeu simultaneamente todos os critérios de viabilidade "
            "(FS, massa, apoio e equilíbrio). A solução exibida é a melhor aproximação "
            "encontrada dentro da busca."
        )
    elif not best and bool(r.get("cfg", {}).get("analysis", {}).get("optimize_variants", True)):
        st.warning(
            "O planejador não retornou candidato final. O modelo exibido é diagnóstico "
            "da configuração executada, não uma proposta final aprovada."
        )
    max_mass_ui = float(mass_limits.get("effective_limit_g", 1000.0))
    best_mass_ui = float(best.get("mass_g", dsum.get("estimated_total_mass_g", 0.0)) or 0.0) if best else 0.0
    if not bool(metrics.get("mass_constraint_passed", True)):
        st.error(
            "A massa estimada do modelo exibido excede o limite efetivo configurado. "
            "Este resultado não pode ser aceito como proposta final."
        )
    if best and best_mass_ui > max_mass_ui + 1e-6:
        st.error(
            "A proposta retornada excede o limite de massa configurado. "
            "Revise os limites e execute novamente."
        )

    summary = recommendations.get("summary")
    if summary:
        for line in str(summary).split("\n"):
            if line.strip():
                st.write(line.strip())

    if bool(metrics.get("quarter_model_used")):
        st.info("Projeto analisado por 1/4 e replicado por simetria.")
    if bool(r.get("cfg", {}).get("analysis", {}).get("enforce_symmetry", True)):
        st.caption("Simetria estrutural: ativa. Simetria construtiva: emendas podem ser desalinhadas intencionalmente.")

    viz = VisualizationService()

    result_sections = [
        "Resumo",
        "Por que esta treliça?",
        "Cargas e membros críticos",
        "Geometria 3D",
        "Montagem real",
        "Detalhamento técnico",
        "Logs e auditoria",
        "Frame3DD e downloads",
    ]
    result_view = st.radio(
        "Navegação dos resultados",
        result_sections,
        horizontal=True,
        key="result_view",
    )
    st.caption("Modo leve: apenas a seção selecionada é renderizada, reduzindo recálculo de UI.")

    if result_view == "Resumo":
        suggestions = recommendations.get("suggestions", [])
        proposal_desc = recommendations.get("proposal_description", "")
        comparison_notes = recommendations.get("comparison_notes", [])

        if proposal_desc:
            st.markdown("**Descrição objetiva da proposta gerada**")
            st.info(proposal_desc)

        if comparison_notes:
            st.markdown("**Comparação rápida das versões finais**")
            for note in comparison_notes:
                st.write(f"- {note}")

        if suggestions:
            st.markdown("**Recomendações de engenharia**")
            for i, suggestion in enumerate(suggestions, 1):
                st.write(f"**{i}.** {suggestion}")
        else:
            st.info("Nenhuma recomendação textual foi gerada.")

        with st.expander("Mostrar configuração completa", expanded=False):
            st.json(r.get("cfg", {}), expanded=False)

        if final_variants:
            st.subheader("Versões finais da sugestão")
            final_rows = []
            for label in ("ideal", "min", "max"):
                row = final_variants.get(label)
                if not row:
                    continue
                cfg_mass_limit = float(mass_limits.get("effective_limit_g", 1000.0))
                final_rows.append(
                    {
                        "versão": label,
                        "FS_min_principal": row.get("min_fs_primary"),
                        "carga_ruptura_estimada_kgf": row.get("predicted_breaking_load_kgf"),
                        "massa_g": row.get("mass_g"),
                        "solver_status": row.get("solver_status"),
                        "viável": row.get("feasible"),
                        "aceita_massa": (float(row.get("mass_g", 0.0) or 0.0) <= cfg_mass_limit),
                        "span_mm": row.get("span_mm"),
                        "width_mm": row.get("width_mm"),
                        "center_height_mm": row.get("center_height_mm"),
                        "panel_mm": row.get("panel_mm"),
                    }
                )

            if final_rows:
                table = _prepare_table(final_rows)
                with st.expander("Mostrar tabela detalhada das versões finais", expanded=False):
                    st.dataframe(_df_to_display_units(table, length_unit, force_unit), width="stretch")

                labels = [row["versão"] for row in final_rows]
                chosen = st.radio("Comparar versão em destaque", labels, horizontal=True)
                chosen_row = next((row for row in final_rows if row["versão"] == chosen), None)
                chosen_full = final_variants.get(chosen) if chosen in final_variants else None
                if chosen_full:
                    st.success(_describe_variant(chosen_full, length_unit, force_unit))

    if result_view == "Por que esta treliça?":
        proposal_desc = recommendations.get("proposal_description", "")
        comparison_notes = recommendations.get("comparison_notes", [])
        by_reason = stage_counts.get("stage0_prefilter_discarded_by_reason", {}) or {}
        by_reason_all = stage_counts.get("discarded_by_reason", {}) or {}
        mat_cfg = r.get("cfg", {}).get("material", {}) or {}
        t_cap = float(mat_cfg.get("tension_capacity_per_stick_kgf", 72.0))
        c_cap = max(1.0e-6, float(mat_cfg.get("compression_capacity_two_sticks_kgf", 11.0)) / 2.0)
        tc_ratio = t_cap / c_cap

        st.markdown("**Critério de escolha da solução**")
        if proposal_desc:
            st.info(proposal_desc)
        else:
            st.info("A seleção priorizou melhor equilíbrio entre fator de segurança, carga de ruptura prevista e massa limite.")

        st.markdown(
            "A influência do material foi considerada na triagem: "
            f"relação tração/compressão aproximada = **{tc_ratio:.2f}**."
        )
        if tc_ratio >= 6.0:
            st.caption(
                "Como a tração está muito acima da compressão, a busca tende a favorecer famílias tipo Pratt "
                "e reduzir famílias Howe, que penalizam diagonais comprimidas longas."
            )

        if comparison_notes:
            st.markdown("**Comparação final (ideal/min/max)**")
            for note in comparison_notes:
                st.write(f"- {note}")

        if by_reason:
            st.markdown("**Topologias descartadas antes da S0/S1 (motivos principais)**")
            top_rows = sorted(by_reason.items(), key=lambda kv: kv[1], reverse=True)[:12]
            st.dataframe(
                pd.DataFrame([{"motivo": k, "quantidade": v} for k, v in top_rows]),
                width="stretch",
            )
        if by_reason_all:
            with st.expander("Mostrar todos os descartes por motivo", expanded=False):
                all_rows = sorted(by_reason_all.items(), key=lambda kv: kv[1], reverse=True)
                st.dataframe(
                    pd.DataFrame([{"motivo": k, "quantidade": v} for k, v in all_rows]),
                    width="stretch",
                )

    if result_view == "Detalhamento técnico":
        stage_map = {
            "S1 - varredura": opt.get("stage1", []),
            "S2 - refinamento": opt.get("stage2", []),
            "S3 - validação detalhada": opt.get("stage3", []),
            "S4 trace - ajustes iterativos": opt.get("stage4_trace", []),
            "S4 - candidatos adaptados validados": opt.get("stage4", []),
            "Descarte por filtros": opt.get("discarded", []),
            "Final - ideal/min/max": list(final_variants.values()),
        }

        stage_pick = st.selectbox("Etapa", list(stage_map))
        stage_rows = stage_map.get(stage_pick, [])

        if stage_rows:
            table = _prepare_table([{k: v for k, v in row.items() if k != "config"} for row in stage_rows])
            table = _df_to_display_units(table, length_unit, force_unit)
            with st.expander("Mostrar tabela completa da etapa", expanded=False):
                st.dataframe(table, width="stretch")

            if {"pontuação", "massa_g", "FS_mín_principal"}.issubset(set(table.columns)):
                plot_df = table.copy()
                plot_df["massa_g"] = pd.to_numeric(plot_df["massa_g"], errors="coerce")
                plot_df["FS_mín_principal"] = pd.to_numeric(plot_df["FS_mín_principal"], errors="coerce")
                plot_df["pontuação"] = pd.to_numeric(plot_df["pontuação"], errors="coerce")
                plot_df = plot_df.dropna(subset=["massa_g", "FS_mín_principal", "pontuação"])
                if not plot_df.empty:
                    st.plotly_chart(
                        px.scatter(
                            plot_df,
                            x="massa_g",
                            y="FS_mín_principal",
                            color="pontuação",
                            hover_data=[c for c in ["id_candidato", "treliça_lateral", "perfil_topo"] if c in plot_df.columns],
                            title="Comparação da etapa: massa x FS mínimo (cor = pontuação)",
                        ),
                        width="stretch",
                    )

            if stage_pick != "Descarte por filtros":
                candidate_labels = []
                for i, row in enumerate(stage_rows[:120]):
                    cid = row.get("candidate_id", f"{stage_pick}-{i+1}")
                    score = _format_float(row.get("score"), 1, "—")
                    fs = _format_float(row.get("min_fs_primary"), 2, "—")
                    candidate_labels.append(f"{cid} | score={score} | FS={fs}")
                if candidate_labels:
                    idx_pick = st.selectbox("Detalhar candidato da etapa", list(range(len(candidate_labels))), format_func=lambda i: candidate_labels[i])
                    row = stage_rows[int(idx_pick)]
                    st.info(_describe_variant(row, length_unit, force_unit))
        else:
            st.info("Sem dados para a etapa selecionada.")

    if result_view == "Geometria 3D":
        v1, v2 = st.columns([2, 1])

        with v1:
            scale_label = st.radio(
                "Escala do 3D",
                ["Didática", "Real", "Cubo unitário"],
                index=0,
                horizontal=True,
            )
            scale_mode = {
                "Didática": "didactic",
                "Real": "real",
                "Cubo unitário": "cube",
            }[scale_label]
            member_df_visual = _prepare_member_checks(r.get("member_checks", []))
            rupture_df = member_df_visual[
                (member_df_visual["member_id"] >= 0)
                & (member_df_visual["FS_min_num"] < 1.0)
            ].sort_values("FS_min_sort", ascending=True)
            rupture_ids = rupture_df["member_id"].drop_duplicates().astype(int).head(20).tolist()
            rupture_details = metrics.get("rupture_details", {}) or {}
            gov_member_raw = rupture_details.get("governing_member_id")
            gov_member_id = int(gov_member_raw) if safe_float(gov_member_raw, None) is not None else None
            if gov_member_id is not None and gov_member_id > 0 and gov_member_id not in rupture_ids:
                rupture_ids = [gov_member_id] + rupture_ids

            rupture_colors = {}
            for _, row in rupture_df.head(40).iterrows():
                mid = int(row.get("member_id", -1))
                fs_val = safe_float(row.get("FS_min_num"), 10.0) or 10.0
                if fs_val < 0.4:
                    rupture_colors[mid] = "#ff2d2d"
                elif fs_val < 0.7:
                    rupture_colors[mid] = "#ff6b2c"
                else:
                    rupture_colors[mid] = "#ffb347"
            if gov_member_id is not None and gov_member_id > 0:
                rupture_colors[gov_member_id] = "#ff00ff"

            st.plotly_chart(
                viz.plotly_geometry(
                    r["nodes"],
                    r["members"],
                    r["supports"],
                    r["loads"],
                    highlight_member_ids=rupture_ids,
                    highlight_member_colors=rupture_colors,
                    scale_mode=scale_mode,
                ),
                width="stretch",
            )
            if gov_member_id is not None and gov_member_id > 0:
                st.info(
                    "Membro governante da previsão de ruptura: "
                    f"`M{gov_member_id}` | modo={rupture_details.get('governing_rupture_mode', '—')} "
                    f"| FS governante={_format_float(rupture_details.get('governing_fs'), 2, '—')}."
                )
            if rupture_ids:
                st.caption(
                    "Membros destacados: possível ruptura (FS<1,0). "
                    "Magenta = membro governante da estimativa de ruptura; tons quentes = criticidade por FS. "
                    "A análise detalhada está na seção Membros Críticos."
                )
            else:
                st.caption("Nenhum membro com FS < 1,0 nesta configuração.")

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

            if rupture_ids:
                node_by_id = {n.id: n for n in r.get("nodes", [])}
                member_by_id = {m.id: m for m in r.get("members", [])}
                rupture_rows = []
                for _, row in rupture_df.head(12).iterrows():
                    mid = int(row.get("member_id", -1))
                    m = member_by_id.get(mid)
                    if not m:
                        continue
                    ni = node_by_id.get(m.i)
                    nj = node_by_id.get(m.j)
                    x_med = None
                    if ni and nj:
                        x_med = 0.5 * (float(ni.x) + float(nj.x))
                    rupture_rows.append(
                        {
                            "member_id": mid,
                            "group": row.get("group"),
                            "FS_min": row.get("FS_min"),
                            "governing_mode": row.get("governing_mode"),
                            "x_centro_mm": x_med,
                            "força_axial_N": row.get("N_N"),
                        }
                    )
                if rupture_rows:
                    st.markdown("**Pontos críticos esperados de ruptura (top 12)**")
                    st.dataframe(
                        _df_to_display_units(_prepare_table(rupture_rows), length_unit, force_unit),
                        width="stretch",
                    )

    if result_view == "Cargas e membros críticos":
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

            show_df = member_df[show_cols].copy()
            show_df = _df_to_display_units(show_df, length_unit, force_unit)
            with st.expander("Mostrar tabela de membros (detalhada)", expanded=False):
                st.dataframe(show_df.head(200), width="stretch")

            critical_first = (
                member_df[member_df["member_id"] >= 0]
                .sort_values("FS_min_sort", ascending=True)["member_id"]
                .drop_duplicates()
                .tolist()
            )

            if critical_first:
                # Allow multi-selection of members to highlight.  By default
                # preselect the top 5 most critical members (lowest FS).  The
                # list of options is sorted by ascending FS_min.
                default_sel = critical_first[:5]
                selected = st.multiselect(
                    "Destacar membros",
                    options=critical_first,
                    default=default_sel,
                    help="Selecione um ou mais IDs de membro para destacá-los na visualização 3D.",
                )

                color_mode_critical = st.radio(
                    "Cor da visualização crítica",
                    [
                        "risco estrutural (FS + utilização)",
                        "fator de segurança",
                        "utilização",
                        "força axial",
                    ],
                    horizontal=True,
                    index=0,
                    help=(
                        "Risco estrutural usa FS_design/FS_min como cor principal "
                        "e utilização como espessura da barra. Força axial isolada "
                        "serve para diagnóstico de caminho de carga, não para risco."
                    ),
                )

                color_mode_map = {
                    "risco estrutural (FS + utilização)": "risk",
                    "fator de segurança": "safety_factor",
                    "utilização": "utilization",
                    "força axial": "force",
                }
                st.plotly_chart(
                    viz.plotly_geometry(
                        r["nodes"],
                        r["members"],
                        r["supports"],
                        r["loads"],
                        color_mode=color_mode_map[color_mode_critical],
                        member_results=r.get("solver_result").member_results if r.get("solver_result") else [],
                        member_checks=r.get("member_checks", []),
                        selected_member_ids=[int(m) for m in selected] if selected else None,
                        highlight_selected=True,
                        scale_mode="didactic",
                    ),
                    width="stretch",
                )

                # Show detailed table for selected members.  If multiple
                # members are selected, show them all; otherwise show none.
                if selected:
                    st.dataframe(
                        _df_to_display_units(
                            member_df[member_df["member_id"].isin([int(m) for m in selected])][show_cols],
                            length_unit,
                            force_unit,
                        ),
                        width="stretch",
                    )

    if result_view == "Montagem real":
        pieces_all = r.get("detailed", {}).get("stick_pieces", [])
        assembly_groups = r.get("detailed", {}).get("assembly_groups", [])
        assembly_summary = r.get("detailed", {}).get("assembly_summary", {})
        assembly_tutorial = r.get("detailed", {}).get("assembly_tutorial", {}) or {}
        render_mode = st.radio(
            "Render dos palitos",
            ["prismas reais", "prismas exagerados", "linhas leves"],
            horizontal=True,
            index=0,
            key="stick_render_mode",
        )

        tutorial_steps = assembly_tutorial.get("assembly_steps", []) or []
        if tutorial_steps:
            st.markdown("**Tutorial de montagem por etapas**")
            for step in tutorial_steps:
                st.markdown(
                    f"**{step.get('step_id', '')} - {step.get('title', '')}**  \n"
                    f"Palitos estimados: {step.get('stick_count', 0)} | "
                    f"Massa estimada: {_format_float(step.get('estimated_mass_g'), 2, '0')} g  \n"
                    f"{step.get('instruction', '')}"
                )
            with st.expander("Mostrar listas de corte e emenda do tutorial", expanded=False):
                cut_df = _prepare_table(assembly_tutorial.get("cut_list", []))
                joint_df = _prepare_table(assembly_tutorial.get("joint_list", []))
                by_member_df = _prepare_table(
                    [
                        {"member_id": k, "stick_count": v}
                        for k, v in (assembly_tutorial.get("stick_count_by_member", {}) or {}).items()
                    ]
                )
                if not cut_df.empty:
                    st.markdown("**Lista de cortes**")
                    st.dataframe(_df_to_display_units(cut_df, length_unit, force_unit), width="stretch")
                if not joint_df.empty:
                    st.markdown("**Lista de juntas**")
                    st.dataframe(_df_to_display_units(joint_df, length_unit, force_unit), width="stretch")
                if not by_member_df.empty:
                    st.markdown("**Palitos por membro**")
                    st.dataframe(by_member_df, width="stretch")

        # Show assembly group overview first if available
        if assembly_groups:
            st.markdown("**Resumo de grupos de montagem**")
            # Convert to DataFrame for display.  Select key columns to keep the table narrow.
            grp_df = pd.DataFrame(assembly_groups)
            key_cols = [
                c
                for c in [
                    "friendly_name",
                    "member_group",
                    "orientation",
                    "approx_length_mm",
                    "n_pieces",
                    "n_members",
                    "mass_g",
                ]
                if c in grp_df.columns
            ]
            st.dataframe(grp_df[key_cols].head(50), width="stretch")

            # Provide summary metrics in a small table
            if assembly_summary:
                asm_rows = [
                    {"item": "Total de peças", "valor": assembly_summary.get("total_pieces")},
                    {"item": "Membros únicos", "valor": assembly_summary.get("unique_members")},
                    {"item": "Massa total", "valor": f"{assembly_summary.get('total_mass_g', 0):.1f} g" if assembly_summary.get("total_mass_g") is not None else ""},
                    {"item": "Faixa de comprimento", "valor": f"{assembly_summary.get('length_range_mm', (0,0))[0]:.1f}–{assembly_summary.get('length_range_mm', (0,0))[1]:.1f} mm"},
                ]
                # Ensure the 'valor' column is consistently treated as string to avoid
                # Arrow conversion errors when values mix numbers and units.
                asm_df = pd.DataFrame(asm_rows)
                if "valor" in asm_df.columns:
                    asm_df["valor"] = asm_df["valor"].astype(str)
                st.dataframe(asm_df, width="stretch")

            # Optionally allow the user to filter the piece view by selecting a group
            group_keys = [g.get("friendly_name") for g in assembly_groups]
            selected_group_name = None
            if group_keys:
                selected_group_name = st.selectbox(
                    "Visualizar detalhes do grupo",
                    options=["(todos)"] + group_keys,
                    index=0,
                    help="Selecione um grupo de montagem para filtrar as peças."
                )
            # Filter pieces by selected group if any
            pieces_to_show = pieces_all
            if selected_group_name and selected_group_name != "(todos)":
                # find group by friendly name
                selected_keys = [g for g in assembly_groups if g.get("friendly_name") == selected_group_name]
                if selected_keys:
                    gsel = selected_keys[0]
                    # match pieces that belong to this group by comparing grouping attributes
                    mg = gsel.get("member_group")
                    orient = gsel.get("orientation")
                    length_bin = gsel.get("approx_length_mm")
                    n_sticks = gsel.get("n_sticks")
                    filtered = []
                    for r in pieces_all:
                        if str(r.get("member_group")) != str(mg):
                            continue
                        # orientation
                        p0 = (
                            float(r.get("x0_mm", 0.0) or 0.0),
                            float(r.get("y0_mm", 0.0) or 0.0),
                            float(r.get("z0_mm", 0.0) or 0.0),
                        )
                        p1 = (
                            float(r.get("x1_mm", 0.0) or 0.0),
                            float(r.get("y1_mm", 0.0) or 0.0),
                            float(r.get("z1_mm", 0.0) or 0.0),
                        )
                        from src.services.assembly_grouping_service import _orientation_from_endpoints  # type: ignore
                        ori = _orientation_from_endpoints(p0, p1)
                        if ori != orient:
                            continue
                        # length bin
                        try:
                            L = float(r.get("cut_length_mm"))
                        except (TypeError, ValueError):
                            L = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2) ** 0.5
                        length_bin_r = round(L / 10.0) * 10.0
                        if abs(length_bin_r - length_bin) > 1e-3:
                            continue
                        # n_sticks (optional)
                        if n_sticks is not None:
                            try:
                                ns = int(r.get("n_sticks"))
                            except (TypeError, ValueError):
                                ns = None
                            if ns != n_sticks:
                                continue
                        filtered.append(r)
                    pieces_to_show = filtered

            # Show 3D piece chart (simplified) for selected group or all
            if pieces_to_show:
                st.plotly_chart(
                    viz.plotly_stick_pieces(
                        pieces_to_show,
                        member_id=None,
                        max_pieces=2000,
                        render_mode=render_mode,
                    ),
                    width="stretch",
                )
            else:
                st.info("Nenhuma peça encontrada para o grupo selecionado.")
        else:
            # Fallback to original per-piece view when grouping is not available
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
                        render_mode=render_mode,
                    ),
                    width="stretch",
                )
            else:
                st.info("Sem dados peça-a-peça para visualizar.")

        # Show weakest glue joints and cutting list in expandable sections
        glue_df = _prepare_table(r.get("detailed", {}).get("weakest_glue_joints", []))
        if not glue_df.empty:
            with st.expander("Juntas coladas mais críticas", expanded=False):
                st.dataframe(
                    _df_to_display_units(glue_df.head(40), length_unit, force_unit),
                    width="stretch",
                )

        cuts_df = _prepare_table(r.get("detailed", {}).get("cutting_list", []))
        if not cuts_df.empty:
            with st.expander("Lista de cortes", expanded=False):
                st.dataframe(
                    _df_to_display_units(cuts_df.head(40), length_unit, force_unit),
                    width="stretch",
                )

    if result_view == "Detalhamento técnico":
        st.subheader("Verificação contra critérios eliminatórios (edital)")
        checks_df = _prepare_table(r.get("edital_checks", []))
        if not checks_df.empty:
            checks_df = checks_df.rename(columns={"criterio": "critério", "valor_obtido": "valor obtido", "regra": "regra", "conforme": "conforme"})
            st.dataframe(checks_df, width="stretch")
        else:
            st.info("Sem dados de critérios do edital.")

        st.subheader("Memorial de cálculo (síntese)")
        carga_projeto = float(r.get("cfg", {}).get("bridge", {}).get("load_total_kgf", 0.0))
        fs_min = float(metrics.get("min_fs_primary", 0.0) or 0.0)
        ruptura_prevista = float(metrics.get("predicted_breaking_load_kgf", carga_projeto * fs_min) or 0.0)
        st.markdown(
            f"""
- Carga de projeto adotada: **{_format_force_kgf(carga_projeto, force_unit, 2)}**.
- FS mínimo principal obtido: **{_format_float(fs_min, 3)}**.
- Estimativa de carga de colapso/ruptura: **{_format_force_kgf(ruptura_prevista, force_unit, 2)}**.
- Relação usada no resumo: `carga_ruptura ≈ carga_projeto × FS_min_principal`.
"""
        )

        st.subheader("Gráficos de apoio à decisão")
        member_df = _prepare_member_checks(r.get("member_checks", []))
        if not member_df.empty:
            grp = (
                member_df.groupby("group", as_index=False)["FS_min_num"]
                .min()
                .rename(columns={"group": "Grupo", "FS_min_num": "FS mínimo"})
                .sort_values("FS mínimo", ascending=True)
            )
            st.plotly_chart(
                px.bar(grp, x="Grupo", y="FS mínimo", title="FS mínimo por grupo estrutural"),
                width="stretch",
            )

            top_axial = member_df.copy()
            top_axial["Força axial"] = pd.to_numeric(top_axial.get("N_N"), errors="coerce").abs()
            top_axial = top_axial.sort_values("Força axial", ascending=False).head(25)
            top_axial["Força axial"] = (top_axial["Força axial"] / 9.80665) * KGF_TO_FORCE_UNIT[force_unit]
            top_axial["membro"] = top_axial["member_id"].astype(str)
            st.plotly_chart(
                px.bar(top_axial, x="membro", y="Força axial", title=f"Top 25 forças axiais absolutas [{force_unit}]"),
                width="stretch",
            )

        # Additional charts: axial force distribution along span and support reactions
        # Compute axial force vs. x-position by averaging member endpoints
        try:
            # Build mapping of node id to x coordinate
            node_map = {n.id: n for n in r.get("nodes", [])}
            axial_rows = []
            # Use member_results from solver (if available)
            mres = r.get("solver_result").member_results if r.get("solver_result") else []
            if not mres:
                mres = []
                for m in r.get("members", []):
                    # fallback: use member_df values
                    row = member_df[member_df["member_id"] == m.id]
                    if not row.empty:
                        N_val = float(row.iloc[0]["N_N"])
                    else:
                        N_val = 0.0
                    mres.append({"member_id": m.id, "i": m.i, "j": m.j, "N_N": N_val})
            for mr in mres:
                mi = node_map.get(int(mr.get("i")))
                mj = node_map.get(int(mr.get("j")))
                if mi and mj:
                    x_mid = 0.5 * (float(mi.x) + float(mj.x))
                    N_val = float(mr.get("N_N", 0.0))
                    axial_rows.append({"x_mm": x_mid, "N_N": N_val})
            if axial_rows:
                axial_df = pd.DataFrame(axial_rows)
                axial_df = axial_df.sort_values("x_mm")
                axial_df["Força axial"] = (axial_df["N_N"].abs() / 9.80665) * KGF_TO_FORCE_UNIT[force_unit]
                # Convert x coordinate to user selected unit
                try:
                    axial_df["posição"] = axial_df["x_mm"] / LENGTH_TO_MM[length_unit]
                    x_label = f"posição x [{length_unit}]"
                except KeyError:
                    axial_df["posição"] = axial_df["x_mm"]
                    x_label = "x [mm]"
                st.plotly_chart(
                    px.line(
                        axial_df,
                        x="posição",
                        y="Força axial",
                        title=f"Distribuição da força axial ao longo do vão (|N|) [{force_unit}]",
                        labels={"posição": x_label},
                    ),
                    width="stretch",
                )
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            st.warning(f"Não foi possível gerar o gráfico de distribuição axial: {exc!r}")

        # Support reaction graph
        try:
            sp_df = _prepare_table(r.get("support_checks", []))
            if not sp_df.empty:
                # Only show active supports with reactions
                sp_df = sp_df[sp_df.get("support_active_vertical") == True].copy()
                if not sp_df.empty:
                    sp_df["reação Z"] = (pd.to_numeric(sp_df.get("reaction_Z_N"), errors="coerce").abs() / 9.80665) * KGF_TO_FORCE_UNIT[force_unit]
                    sp_df["apoio"] = sp_df["node_id"].astype(str)
                    st.plotly_chart(
                        px.bar(
                            sp_df,
                            x="apoio",
                            y="reação Z",
                            title=f"Reações de apoio verticais [{force_unit}]",
                        ),
                        width="stretch",
                    )
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            st.warning(f"Não foi possível gerar o gráfico de reações de apoio: {exc!r}")

        st.subheader("Resumo dimensional e construtivo")
        resumo = [
            {
                "item": "Vão final",
                "valor": f"{_to_display_length(float(r.get('cfg', {}).get('bridge', {}).get('span_mm', 0.0)), length_unit):.2f} {length_unit}",
            },
            {
                "item": "Largura final",
                "valor": f"{_to_display_length(float(r.get('cfg', {}).get('bridge', {}).get('width_mm', 0.0)), length_unit):.2f} {length_unit}",
            },
            {
                "item": "Altura final",
                "valor": f"{_to_display_length(float(r.get('cfg', {}).get('bridge', {}).get('center_height_mm', 0.0)), length_unit):.2f} {length_unit}",
            },
            {
                "item": "Peso estimado",
                "valor": f"{_format_float(dsum.get('estimated_total_mass_g'), 1)} g",
            },
            {
                "item": "Consumo estimado de palitos",
                "valor": f"{_safe_metric(dsum.get('estimated_total_sticks_with_waste'))}",
            },
        ]
        # Convert valor column to string to ensure consistent typing for Arrow tables
        resumo_df = pd.DataFrame(resumo)
        if "valor" in resumo_df.columns:
            resumo_df["valor"] = resumo_df["valor"].astype(str)
        st.dataframe(resumo_df, width="stretch")

    if result_view == "Logs e auditoria":
        st.markdown("**Resumo numérico da busca**")
        stage_resume = pd.DataFrame(
            [
                {"etapa": "S0 geradas", "quantidade": stage_counts.get("stage0_generated", 0)},
                {"etapa": "S0 aprovadas no prefilter", "quantidade": stage_counts.get("stage0_prefilter_passed", 0)},
                {"etapa": "S0 descartadas no prefilter", "quantidade": stage_counts.get("stage0_prefilter_discarded", 0)},
                {"etapa": "S1 avaliadas no solver", "quantidade": stage_counts.get("stage1_evaluated", 0)},
                {"etapa": "S1 descartadas pós-solver", "quantidade": stage_counts.get("stage1_discarded_post_solver", 0)},
                {"etapa": "S1 válidas", "quantidade": stage_counts.get("stage1", 0)},
                {"etapa": "S2 variantes geradas", "quantidade": stage_counts.get("stage2_generated", 0)},
                {"etapa": "S2A aprovadas no filtro rápido", "quantidade": stage_counts.get("stage2a_selected", 0)},
                {"etapa": "S2B avaliadas no solver", "quantidade": stage_counts.get("stage2b_evaluated", 0)},
                {"etapa": "S2 variantes únicas", "quantidade": stage_counts.get("stage2_unique", stage_counts.get("stage2", 0))},
                {"etapa": "S3 validadas", "quantidade": stage_counts.get("stage3", 0)},
                {"etapa": "S4 rastros", "quantidade": stage_counts.get("stage4_trace", 0)},
                {"etapa": "S4 validadas", "quantidade": stage_counts.get("stage4", 0)},
                {"etapa": "Versões finais", "quantidade": stage_counts.get("final_variants", 0)},
            ]
        )
        st.dataframe(stage_resume, width="stretch")

        by_reason = stage_counts.get("stage0_prefilter_discarded_by_reason", {}) or {}
        if by_reason:
            st.markdown("**Descarte por motivo (prefilter)**")
            by_reason_df = pd.DataFrame(
                [{"motivo": k, "quantidade": v} for k, v in by_reason.items()]
            ).sort_values("quantidade", ascending=False)
            st.dataframe(by_reason_df, width="stretch")

        by_reason_all = stage_counts.get("discarded_by_reason", {}) or {}
        if by_reason_all:
            st.markdown("**Descarte por motivo (todas as etapas de filtro)**")
            by_reason_all_df = pd.DataFrame(
                [{"motivo": k, "quantidade": v} for k, v in by_reason_all.items()]
            ).sort_values("quantidade", ascending=False)
            st.dataframe(by_reason_all_df, width="stretch")

        logs_combined = []
        logs_combined.extend(st.session_state.get("last_run_logs", []))
        logs_combined.extend(opt.get("logs", []))
        warnings_list = r.get("warnings", []) or []

        if logs_combined:
            st.markdown("**Logs detalhados da execução**")
            st.code("\n".join(logs_combined[-300:]), language="text")
        else:
            st.info("Nenhum log detalhado disponível.")
        if warnings_list:
            st.markdown("**Warnings estruturados**")
            st.dataframe(pd.DataFrame(warnings_list), width="stretch")

        debug_jsonl = Path(r.get("planner_debug_jsonl", ""))
        debug_summary = Path(r.get("planner_debug_summary", ""))
        if debug_jsonl.exists():
            st.download_button(
                "Baixar planner_debug.jsonl",
                debug_jsonl.read_bytes(),
                file_name="planner_debug.jsonl",
                mime="application/json",
            )
        if debug_summary.exists():
            st.download_button(
                "Baixar planner_debug_summary.md",
                debug_summary.read_bytes(),
                file_name="planner_debug_summary.md",
                mime="text/markdown",
            )

        discarded_df = _prepare_table(opt.get("discarded", []))
        if not discarded_df.empty:
            st.markdown("**Propostas descartadas (auditoria de filtros)**")
            st.dataframe(_df_to_display_units(discarded_df, length_unit, force_unit), width="stretch")

    if result_view == "Frame3DD e downloads":
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

        fzp = Path(r.get("focused_zip_path", ""))
        if fzp.exists():
            st.download_button(
                "Baixar pacote focado: cálculo + fabricação",
                fzp.read_bytes(),
                file_name="pacote_focado_fabricacao_e_calculo.zip",
                mime="application/zip",
                help="Pacote menor e mais direto: resumo, memorial, guia de fabricação, listas essenciais e auditorias.",
            )

        zp = Path(r.get("zip_path", ""))
        if zp.exists():
            st.download_button(
                "Baixar pacote completo de depuração",
                zp.read_bytes(),
                file_name="resultados_simulacao_completo.zip",
                mime="application/zip",
                help="Pacote completo com todos os CSVs, plots e saídas intermediárias.",
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
