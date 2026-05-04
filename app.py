from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.services.config_service import ConfigService
from src.services.pipeline import SimulationPipeline
from src.services.visualization_service import VisualizationService


st.set_page_config(page_title="Ponte de Palitos — Simulador v4", layout="wide")
st.title("Simulador de Ponte de Palitos")
st.caption(
    "Configuração simples, análise completa, comparação de propostas "
    "e visualização prática de montagem."
)


def _as_dataframe(data: Any) -> pd.DataFrame:
    """Converte listas/dicts em DataFrame sem quebrar quando vier vazio."""
    if data is None:
        return pd.DataFrame()

    if isinstance(data, pd.DataFrame):
        return data.copy()

    try:
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Converte colunas para numérico quando existirem."""
    out = df.copy()

    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _prepare_member_checks(data: Any) -> pd.DataFrame:
    """Prepara a tabela de membros para ordenação, exibição e seleção."""
    df = _as_dataframe(data)

    if df.empty:
        return df

    numeric_cols = [
        "member_id",
        "N_N",
        "L_mm",
        "A_mm2",
        "Iy_mm4",
        "Iz_mm4",
        "Pcr_y_N",
        "Pcr_z_N",
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

    if "FS_min" in df.columns:
        df["FS_min_num"] = pd.to_numeric(df["FS_min"], errors="coerce")
    else:
        df["FS_min_num"] = pd.NA

    df["FS_min_sort"] = df["FS_min_num"].fillna(1.0e99)

    if "member_role" not in df.columns:
        df["member_role"] = "unknown"

    if "risk_flag" not in df.columns:
        df["risk_flag"] = "—"

    if "group" not in df.columns:
        df["group"] = "—"

    return df.sort_values("FS_min_sort", ascending=True)


def _prepare_table(data: Any) -> pd.DataFrame:
    """Prepara tabelas gerais para exibição."""
    df = _as_dataframe(data)

    if df.empty:
        return df

    for col in df.columns:
        if col.startswith("FS") or col.endswith("_N") or col.endswith("_MPa") or col.endswith("_mm"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "FS_min" in df.columns:
        df["FS_min_num"] = pd.to_numeric(df["FS_min"], errors="coerce")
        df["FS_min_sort"] = df["FS_min_num"].fillna(1.0e99)
        df = df.sort_values("FS_min_sort", ascending=True)

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

with st.sidebar:
    st.header("Objetivo")

    load_kgf = st.number_input(
        "Carga alvo [kgf]",
        1.0,
        300.0,
        float(base_cfg["bridge"].get("load_total_kgf", 120.0)),
        5.0,
        help="Carga vertical total que a ponte deve suportar.",
    )

    mass_limit_g = st.number_input(
        "Massa limite [g]",
        100.0,
        2000.0,
        float(base_cfg["material"].get("mass_limit_g", 1000.0)),
        50.0,
        help="Limite de massa total da ponte pronta.",
    )

    st.header("Forma")

    truss_type = st.selectbox(
        "Treliça lateral",
        ["Parker", "Pratt", "Howe", "Warren"],
        0,
        help="Controla as diagonais das duas treliças laterais.",
    )

    internal_truss_type = st.selectbox(
        "Treliça interna / contraventamento",
        ["X", "Warren", "Pratt", "Howe", "none"],
        0,
        help="Controla os contraventamentos dos planos superior/inferior e quadros transversais.",
    )

    chord_truss_type = st.selectbox(
        "Treliça/lacing dos banzos",
        ["none", "Warren", "Pratt", "Howe", "X"],
        0,
        help="Adiciona lacing/redundância junto aos banzos laterais. Aumenta rigidez e massa.",
    )

    top_profile = st.selectbox(
        "Perfil do topo",
        ["parker_plateau", "triangular_peak", "shallow_arch", "flat"],
        0,
        help="Define a forma geométrica do banzo superior.",
    )

    span_mm = st.number_input(
        "Vão livre [mm]",
        300.0,
        3000.0,
        float(base_cfg["bridge"].get("span_mm", 1200.0)),
        50.0,
    )

    stick_length_mm = st.number_input(
        "Comprimento do palito [mm]",
        80.0,
        200.0,
        float(base_cfg["material"].get("stick_length_mm", 120.0)),
        1.0,
        help="Comprimento real médio do palito. Para seu lote, use 120 mm.",
    )

    panel_mm = st.number_input(
        "Painel [mm]",
        50.0,
        200.0,
        float(base_cfg["bridge"].get("panel_mm", 100.0)),
        10.0,
        help="Distância entre nós sucessivos no comprimento da ponte.",
    )

    width_mm = st.number_input(
        "Largura [mm]",
        80.0,
        220.0,
        float(base_cfg["bridge"].get("width_mm", 180.0)),
        10.0,
    )

    center_height_mm = st.number_input(
        "Altura central [mm]",
        50.0,
        600.0,
        float(base_cfg["bridge"].get("center_height_mm", 300.0)),
        10.0,
    )

    st.header("Material")

    E_MPa = st.number_input(
        "Módulo E [MPa]",
        500.0,
        20000.0,
        float(base_cfg["material"].get("E_MPa", 6000.0)),
        100.0,
    )

    stick_width_mm = st.number_input(
        "Largura palito [mm]",
        3.0,
        15.0,
        float(base_cfg["material"].get("stick_width_mm", 7.0)),
        0.5,
    )

    stick_thickness_mm = st.number_input(
        "Espessura palito [mm]",
        0.5,
        5.0,
        float(base_cfg["material"].get("stick_thickness_mm", 1.5)),
        0.1,
    )

    stick_mass_g = st.number_input(
        "Massa/palito [g]",
        0.2,
        5.0,
        float(base_cfg["material"].get("stick_mass_g", 1.4)),
        0.1,
    )

    tension_capacity_per_stick_kgf = st.number_input(
        "Resistência tração/palito [kgf]",
        1.0, 200.0,
        float(base_cfg["material"].get("tension_capacity_per_stick_kgf", 72.0)),
        1.0,
        help="Carga máxima de ruptura à tração. Não é o módulo E.",
    )
    compression_capacity_one_stick_kgf = st.number_input(
        "Compressão 1 palito [kgf]",
        0.1, 100.0,
        float(base_cfg["material"].get("compression_capacity_one_stick_kgf", 4.0)),
        0.5,
    )
    compression_capacity_two_sticks_kgf = st.number_input(
        "Compressão 2 palitos colados [kgf]",
        0.1, 200.0,
        float(base_cfg["material"].get("compression_capacity_two_sticks_kgf", 11.0)),
        0.5,
    )

    overlap_length_mm = st.number_input(
        "Sobreposição [mm]",
        5.0,
        min(100.0, stick_length_mm * 0.8),
        float(base_cfg["detail_model"].get("overlap_length_mm", 30.0)),
        5.0,
        help="Comprimento de sobreposição usado nas emendas coladas.",
    )

    glue_shear_strength_MPa = st.number_input(
        "Cola cisalhamento [MPa]",
        0.5,
        30.0,
        float(base_cfg["detail_model"].get("glue_shear_strength_MPa", 3.5)),
        0.5,
        help="Resistência estimada da cola ao cisalhamento.",
    )

    optimize = st.checkbox("Comparar Pratt/Parker/Howe/Warren", True)

    run = st.button(
        "Rodar análise completa",
        type="primary",
        width="stretch",
    )


cfg = cs.from_minimal_inputs(
    base_cfg,
    load_kgf=load_kgf,
    span_mm=span_mm,
    width_mm=width_mm,
    center_height_mm=center_height_mm,
    panel_mm=panel_mm,
    truss_type=truss_type,
    top_profile=top_profile,
    internal_truss_type=internal_truss_type,
    chord_truss_type=chord_truss_type,
    E_MPa=E_MPa,
    stick_length_mm=stick_length_mm,
    stick_width_mm=stick_width_mm,
    stick_thickness_mm=stick_thickness_mm,
    stick_mass_g=stick_mass_g,
    glue_shear_strength_MPa=glue_shear_strength_MPa,
    overlap_length_mm=overlap_length_mm,
    mass_limit_g=mass_limit_g,
    tension_capacity_per_stick_kgf=tension_capacity_per_stick_kgf,
    compression_capacity_one_stick_kgf=compression_capacity_one_stick_kgf,
    compression_capacity_two_sticks_kgf=compression_capacity_two_sticks_kgf,
)

cfg["analysis"]["optimize_variants"] = optimize

if run:
    with st.spinner("Rodando análise completa..."):
        st.session_state["last_result"] = SimulationPipeline("outputs").run(cfg)

r = st.session_state.get("last_result")

if not r:
    st.info("Configure os dados e clique em Rodar análise completa.")
    st.stop()


metrics = r.get("metrics", {})
dsum = r.get("detailed", {}).get("summary", {})
frame3dd_result = r.get("frame3dd_result", {})
recommendations = r.get("recommendations", {})

cols = st.columns(6)

cols[0].metric(
    "Treliça",
    _safe_metric(r.get("cfg", {}).get("bridge", {}).get("truss_type")),
)

cols[1].metric(
    "FS principal",
    _format_float(metrics.get("min_fs_primary"), 2, "—"),
)

cols[2].metric(
    "Massa",
    f"{_format_float(dsum.get('estimated_total_mass_g'), 0, '0')} g",
)

cols[3].metric(
    "Margem",
    f"{_format_float(dsum.get('mass_margin_g'), 0, '0')} g",
)

cols[4].metric(
    "Palitos",
    _safe_metric(dsum.get("estimated_total_sticks_with_waste")),
)

cols[5].metric(
    "Frame3DD",
    _safe_metric(frame3dd_result.get("status")),
)

summary = recommendations.get("summary")

if summary:
    st.write(summary)
else:
    st.info("Análise concluída. Veja as abas abaixo para detalhes.")


tabs = [
    st.expander("👁️ Visual 3D da estrutura", expanded=True),
    st.expander("🛠️ Melhorias e diagnóstico", expanded=True),
    st.expander("🧪 Proposta automática — comparação com a atual", expanded=True),
    st.expander("🔎 Localizar membro e palitos", expanded=True),
    st.expander("🪵 Montagem e peça-a-peça", expanded=False),
    st.expander("📋 Tabelas", expanded=False),
    st.expander("🏗️ Frame3DD", expanded=False),
    st.expander("⬇️ Downloads", expanded=False),
]

viz = VisualizationService()


with tabs[0]:
    scale_mode = st.radio(
        "Escala da visualização",
        ["didactic", "real", "cube"],
        index=0,
        horizontal=True,
        help=(
            "didactic: exagera levemente altura/largura para leitura; "
            "real: respeita proporção geométrica real; "
            "cube: distorce tudo para caber em cubo."
        ),
        key="visual_scale_mode",
    )

    st.plotly_chart(
        viz.plotly_geometry(
            r["nodes"],
            r["members"],
            r["supports"],
            r["loads"],
            scale_mode=scale_mode,
        ),
        width="stretch",
    )


with tabs[1]:
    suggestions = recommendations.get("suggestions", [])

    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            st.write(f"**{i}.** {suggestion}")
    else:
        st.info("Nenhuma sugestão automática foi gerada.")

    member_df = _prepare_member_checks(r.get("member_checks", []))

    if not member_df.empty:
        st.subheader("Membros mais críticos")
        show_cols = [
            c
            for c in [
                "member_id",
                "group",
                "member_role",
                "state",
                "N_N",
                "FS_min",
                "FS_min_label",
                "FS_min_num",
                "governing_mode",
                "report_mode",
                "risk_flag",
            ]
            if c in member_df.columns
        ]

        st.dataframe(
            member_df[show_cols].head(40),
            width="stretch",
        )
    else:
        st.info("Nenhum resultado de membro disponível.")


with tabs[2]:
    opt = r.get("optimization") or {}

    if opt.get("best"):
        best = opt["best"]

        st.success(
            f"Recomendado: {best.get('truss_type', best.get('side_truss_type', '—'))} / "
            f"{best.get('top_profile', '—')} | interna {best.get('internal_truss_type','—')} | banzos {best.get('chord_truss_type','—')}"
        )
        st.write(
            f"FS={_format_float(best.get('min_fs_primary'),2,'—')} | "
            f"massa={_format_float(best.get('mass_g'),0,'—')} g | "
            f"margem={_format_float(best.get('mass_margin_g'),0,'—')} g | "
            f"palitos={best.get('estimated_sticks','—')} | "
            f"viável={best.get('feasible', False)} | variantes={opt.get('tried_variants','—')}"
        )

        variants = [
            {k: v for k, v in q.items() if k != "config"}
            for q in opt.get("variants", [])[:50]
        ]

        opt_df = _prepare_table(variants)

        if not opt_df.empty:
            st.dataframe(opt_df, width="stretch")
        else:
            st.info("Nenhuma variação foi retornada.")
    else:
        if opt.get("error"):
            st.error("Otimizador retornou erro.")
            st.code(str(opt.get("error")))
        else:
            st.info("Sem otimização. Verifique se a opção de comparação está ativa e se há limites viáveis.")


with tabs[3]:
    df = _prepare_member_checks(r.get("member_checks", []))

    if df.empty:
        st.info("Nenhum membro disponível para destacar.")
    else:
        critical_first = (
            df[df["member_id"] >= 0]
            .sort_values("FS_min_sort", ascending=True)["member_id"]
            .drop_duplicates()
            .tolist()
        )

        if not critical_first:
            st.warning("Nenhum membro válido disponível para destaque.")
            st.stop()

        mid = st.selectbox(
            "Membro para destacar",
            critical_first,
            index=0,
            format_func=lambda x: f"Membro {x}",
        )
        st.session_state["selected_member_id"] = int(mid)

        scale_mode = st.radio(
            "Escala visual",
            ["didactic", "real", "cube"],
            index=0,
            horizontal=True,
            help=(
                "didactic: mantém leitura clara; "
                "real: mostra proporção real da ponte; "
                "cube: distorce para inspecionar conexões."
            ),
            key="member_scale_mode",
        )

        st.plotly_chart(
            viz.plotly_geometry(
                r["nodes"],
                r["members"],
                r["supports"],
                r["loads"],
                highlight_member_ids=[int(mid)],
                scale_mode=scale_mode,
            ),
            width="stretch",
        )

        st.subheader("Diagnóstico do membro")

        show_cols = [
            c
            for c in [
                "member_id",
                "group",
                "member_role",
                "state",
                "N_N",
                "L_mm",
                "A_mm2",
                "FS_min",
                "FS_min_label",
                "FS_min_num",
                "governing_mode",
                "report_mode",
                "risk_flag",
            ]
            if c in df.columns
        ]

        st.dataframe(
            df[df["member_id"] == int(mid)][show_cols],
            width="stretch",
        )

        pieces = _as_dataframe(r.get("detailed", {}).get("stick_pieces", []))

        if not pieces.empty:
            if "member_id" in pieces.columns:
                pieces["member_id"] = pd.to_numeric(
                    pieces["member_id"],
                    errors="coerce",
                ).fillna(-1).astype(int)

                st.subheader("Peças/palitos que compõem esse membro")
                st.dataframe(
                    pieces[pieces["member_id"] == int(mid)],
                    width="stretch",
                )
            else:
                st.info("A tabela de peças não contém member_id.")


with tabs[4]:
    pieces_all = r.get("detailed", {}).get("stick_pieces", [])
    member_for_piece = st.session_state.get("selected_member_id")
    only_selected_piece = st.checkbox("Mostrar apenas membro destacado", value=True)
    if pieces_all:
        st.plotly_chart(
            viz.plotly_stick_pieces(
                pieces_all,
                member_id=member_for_piece if only_selected_piece else None,
                max_pieces=1800,
            ),
            width="stretch",
        )
    else:
        st.info("Sem dados peça-a-peça. Rode a análise completa.")

    plot_dir = Path("outputs/plots")
    imgs = sorted(plot_dir.glob("*.png"))

    if not imgs:
        st.info("Nenhuma imagem de montagem foi encontrada.")
    else:
        pick = st.selectbox("Imagem", [p.name for p in imgs])

        if pick:
            st.image(str(plot_dir / pick), width="stretch")


with tabs[5]:
    options = {
        "membros": "outputs/opensees/member_failure_checks.csv",
        "apoios": "outputs/opensees/support_reaction_checks.csv",
        "palitos": "outputs/details/stick_pieces.csv",
        "colagens": "outputs/details/glue_joints.csv",
        "cortes": "outputs/details/cutting_list.csv",
        "reforços": "outputs/details/reinforcement_suggestions.csv",
    }

    k = st.selectbox("Tabela", list(options))
    p = Path(options[k])

    if p.exists():
        table_df = _prepare_table(pd.read_csv(p))
        st.dataframe(table_df, width="stretch")
    else:
        st.info(f"Tabela não encontrada: {p}")


with tabs[6]:
    st.json(frame3dd_result)

    p = Path("outputs/frame3dd/ponte_palitos.out")

    if p.exists():
        st.text_area(
            "Saída Frame3DD",
            p.read_text(encoding="utf-8", errors="ignore")[:40000],
            height=420,
        )
    else:
        st.info("Arquivo de saída do Frame3DD ainda não encontrado.")


with tabs[7]:
    zp = Path(r.get("zip_path", ""))

    if zp.exists():
        st.download_button(
            "Baixar resultados",
            zp.read_bytes(),
            file_name="resultados_simulacao.zip",
            mime="application/zip",
        )
    else:
        st.info("ZIP de resultados ainda não encontrado.")

    _download_text_button(
        "Baixar configuração usada",
        json.dumps(r.get("cfg", {}), indent=2, ensure_ascii=False),
        "config_used.json",
    )