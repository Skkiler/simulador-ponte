from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

from src.core.numeric import safe_float


class ConfigService:
    """Responsabilidade única: ler, salvar e normalizar configurações."""

    def __init__(self, default_path: str | Path = "bridge_config.json") -> None:
        self.default_path = Path(default_path)

    def load(self, path: str | Path | None = None) -> Dict[str, Any]:
        p = Path(path) if path else self.default_path
        with open(p, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return self.normalize(cfg)

    def save(self, cfg: Dict[str, Any], path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.normalize(cfg), f, indent=2, ensure_ascii=False)
        return p

    @staticmethod
    def _build_default_load_distribution_x_mm(bridge: Dict[str, Any]) -> list[float]:
        span = float(bridge.get("span_mm", 1200.0))
        panel = max(1.0, float(bridge.get("panel_mm", 100.0)))

        start = float(bridge.get("plateau_start_mm", span / 3.0))
        end = float(bridge.get("plateau_end_mm", 2.0 * span / 3.0))

        lo = max(0.0, min(start, end))
        hi = min(span, max(start, end))

        xs: list[float] = []
        x = lo

        while x <= hi + 1.0e-9:
            xs.append(round(x, 6))
            x += panel

        return xs or [round(span / 2.0, 6)]

    @classmethod
    def _normalize_load_distribution_x_mm(cls, bridge: Dict[str, Any]) -> list[float]:
        raw = bridge.get("load_distribution_x_mm")
        span = float(bridge.get("span_mm", 1200.0))

        if not isinstance(raw, list) or not raw:
            return cls._build_default_load_distribution_x_mm(bridge)

        cleaned: list[float] = []

        for value in raw:
            try:
                x = float(value)
            except (TypeError, ValueError):
                continue

            x = max(0.0, min(span, x))
            cleaned.append(round(x, 6))

        cleaned = sorted(set(cleaned))
        return cleaned or cls._build_default_load_distribution_x_mm(bridge)

    @staticmethod
    def _validate_normalized(cfg: Dict[str, Any]) -> None:
        bridge = cfg.get("bridge", {}) or {}
        material = cfg.get("material", {}) or {}
        planner = cfg.get("planner", {}) or {}

        positive_bridge = (
            "span_mm",
            "panel_mm",
            "width_mm",
            "center_height_mm",
            "load_total_kgf",
        )

        for key in positive_bridge:
            val = float(bridge.get(key, 0.0))
            if val <= 0.0:
                raise ValueError(
                    f"Config inválida: bridge.{key} deve ser > 0 "
                    f"(recebido {val})."
                )

        positive_material = (
            "stick_length_mm",
            "stick_width_mm",
            "stick_thickness_mm",
            "stick_mass_g",
            "mass_limit_g",
        )

        for key in positive_material:
            val = float(material.get(key, 0.0))
            if val <= 0.0:
                raise ValueError(
                    f"Config inválida: material.{key} deve ser > 0 "
                    f"(recebido {val})."
                )

        range_pairs = (
            ("span_min_mm", "span_max_mm"),
            ("width_min_mm", "width_max_mm"),
            ("height_min_mm", "height_max_mm"),
            ("panel_min_mm", "panel_max_mm"),
        )

        for min_key, max_key in range_pairs:
            mn = float(planner.get(min_key, 0.0))
            mx = float(planner.get(max_key, 0.0))

            if mn > mx:
                raise ValueError(
                    f"Config inválida: planner.{min_key} ({mn}) não pode ser "
                    f"maior que planner.{max_key} ({mx})."
                )

    def normalize(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        cfg = copy.deepcopy(cfg)

        bridge = cfg.setdefault("bridge", {})
        mat = cfg.setdefault("material", {})
        planner = cfg.setdefault("planner", {})
        analysis = cfg.setdefault("analysis", {})
        compat_warnings = cfg.setdefault("compatibility_warnings", [])

        def add_compat_warning(message: str) -> None:
            if message not in compat_warnings:
                compat_warnings.append(message)

        # ---------------------------------------------------------------------
        # Analysis / quarter model
        # ---------------------------------------------------------------------
        analysis.setdefault("enforce_symmetry", True)

        if "use_quarter_model" not in analysis:
            analysis["use_quarter_model"] = bool(analysis.get("enforce_symmetry", True))

        analysis.setdefault("quarter_model_mode", "strict")
        analysis.setdefault("quarter_model_debug", False)
        analysis.setdefault("quarter_model_finalize_with_full", True)

        # ---------------------------------------------------------------------
        # Bridge: top profile aliases
        # ---------------------------------------------------------------------
        top_profile_legacy = str(bridge.get("top_profile", "parker_plateau")).strip().lower()

        top_profile_aliases = {
            "parker_plateau": "parker_plateau",
            "plateau": "parker_plateau",
            "platô": "parker_plateau",
            "plato": "parker_plateau",
            "triangular_peak": "triangular_peak",
            "triangular": "triangular_peak",
            "pontiagudo/triangular": "triangular_peak",
            "shallow_arch": "shallow_arch",
            "arch": "shallow_arch",
            "arco": "shallow_arch",
            "flat": "flat",
            "reto": "flat",
            "reta": "flat",
        }

        bridge["top_profile"] = top_profile_aliases.get(
            top_profile_legacy,
            "parker_plateau",
        )

        # ---------------------------------------------------------------------
        # Bridge: truss defaults and legacy chord migration
        # ---------------------------------------------------------------------
        bridge.setdefault("truss_type", bridge.get("side_truss_type", "Parker"))
        bridge.setdefault("side_truss_type", bridge.get("truss_type", "Parker"))
        bridge.setdefault("internal_truss_type", "X")
        bridge.setdefault("chord_truss_type", "none")

        legacy_chord = str(bridge.get("chord_truss_type", "none"))
        legacy_mode = legacy_chord.strip().lower()
        legacy_enabled = legacy_mode not in {"", "none", "sem", "nenhuma"}

        bridge.setdefault("legacy_chord_truss_lacing_enabled", False)

        if "top_chord_truss_type" not in bridge:
            bridge["top_chord_truss_type"] = legacy_chord if legacy_enabled else "X"

            if legacy_enabled:
                add_compat_warning(
                    "bridge.chord_truss_type legado migrado para "
                    "bridge.top_chord_truss_type."
                )
        else:
            bridge["top_chord_truss_type"] = str(
                bridge.get("top_chord_truss_type", "X")
            )

        if "bottom_chord_truss_type" not in bridge:
            bridge["bottom_chord_truss_type"] = legacy_chord if legacy_enabled else "X"

            if legacy_enabled:
                add_compat_warning(
                    "bridge.chord_truss_type legado migrado para "
                    "bridge.bottom_chord_truss_type."
                )
        else:
            bridge["bottom_chord_truss_type"] = str(
                bridge.get("bottom_chord_truss_type", "X")
            )

        if legacy_enabled and not bool(bridge.get("legacy_chord_truss_lacing_enabled", False)):
            add_compat_warning(
                "bridge.chord_truss_type legado detectado; lacing legado "
                "desativado por padrão. Ative "
                "bridge.legacy_chord_truss_lacing_enabled=true para manter "
                "comportamento antigo."
            )

        bridge.setdefault("cross_frame_truss_type", bridge.get("internal_truss_type", "X"))

        unsupported_bridge_trusses = {
            "k",
            "k_symmetric",
            "k simétrica",
            "k simetrica",
        }

        def sanitize_bridge_truss_field(key: str, fallback: str) -> None:
            value = str(bridge.get(key, fallback)).strip()

            if value.lower() in unsupported_bridge_trusses:
                bridge[key] = fallback
                add_compat_warning(
                    f"bridge.{key}='{value}' removido: K-truss ainda não é "
                    f"implementado com nós intermediários reais. Usando "
                    f"'{fallback}'."
                )
            else:
                bridge[key] = value

        sanitize_bridge_truss_field("side_truss_type", "Pratt_symmetric")
        sanitize_bridge_truss_field(
            "truss_type",
            bridge.get("side_truss_type", "Pratt_symmetric"),
        )
        sanitize_bridge_truss_field("internal_truss_type", "X")
        sanitize_bridge_truss_field(
            "cross_frame_truss_type",
            bridge.get("internal_truss_type", "X"),
        )
        sanitize_bridge_truss_field("top_chord_truss_type", "X")
        sanitize_bridge_truss_field("bottom_chord_truss_type", "X")

        # ---------------------------------------------------------------------
        # Bridge: numeric defaults
        # ---------------------------------------------------------------------
        bridge.setdefault("span_mm", 1200.0)
        bridge.setdefault("panel_mm", 100.0)
        bridge.setdefault("width_mm", 160.0)
        bridge.setdefault("left_support_overhang_mm", 100.0)
        bridge.setdefault("right_support_overhang_mm", 100.0)
        bridge.setdefault("end_height_mm", 100.0)
        bridge.setdefault("center_height_mm", 300.0)
        bridge.setdefault("plateau_start_mm", float(bridge["span_mm"]) / 3.0)
        bridge.setdefault("plateau_end_mm", 2.0 * float(bridge["span_mm"]) / 3.0)
        bridge.setdefault("load_total_kgf", 120.0)

        bridge.setdefault("tension_only_bracing_solver_enabled", True)
        bridge.setdefault("tension_only_bracing_interpretation", True)

        bridge["load_total_N"] = float(bridge["load_total_kgf"]) * 9.80665

        bridge["load_application_level"] = str(
            bridge.get("load_application_level", "top")
        ).strip().lower()

        if bridge["load_application_level"] not in {"top", "bottom"}:
            bridge["load_application_level"] = "top"

        bridge["load_distribution_x_mm"] = self._normalize_load_distribution_x_mm(bridge)

        half_w = float(bridge["width_mm"]) / 2.0

        bridge.setdefault("support_contact_y_mm", [-half_w, half_w])
        bridge.setdefault(
            "support_contact_x_left_mm",
            [-float(bridge["left_support_overhang_mm"]), 0.0],
        )
        bridge.setdefault(
            "support_contact_x_right_mm",
            [
                float(bridge["span_mm"]),
                float(bridge["span_mm"]) + float(bridge["right_support_overhang_mm"]),
            ],
        )

        # ---------------------------------------------------------------------
        # Material
        # ---------------------------------------------------------------------
        mat.setdefault("E_MPa", 6000.0)
        mat.setdefault("G_MPa", 500.0)

        mat.setdefault("stick_length_mm", 115.0)
        mat.setdefault("stick_width_mm", 7.0)
        mat.setdefault("stick_thickness_mm", 1.5)
        mat.setdefault("stick_mass_g", 1.4)

        mat.setdefault("mass_limit_g", 1000.0)
        mat.setdefault("nominal_competition_limit_g", 1000.0)
        mat.setdefault("stick_budget_g", 900.0)
        mat.setdefault("wet_glue_budget_g", 100.0)
        mat.setdefault("glue_reserved_g", 100.0)

        vol = max(
            1.0e-9,
            float(mat["stick_length_mm"])
            * float(mat["stick_width_mm"])
            * float(mat["stick_thickness_mm"]),
        )

        mat["density_g_per_mm3"] = float(
            mat.get("density_g_per_mm3", float(mat["stick_mass_g"]) / vol)
        )

        mat.setdefault("tension_capacity_per_stick_kgf", 72.0)
        mat.setdefault("compression_capacity_one_stick_kgf", 4.0)
        mat.setdefault("compression_capacity_two_sticks_kgf", 11.0)
        mat.setdefault(
            "compression_capacity_table_kgf",
            {
                "1": 4.0,
                "2": 11.0,
            },
        )
        mat.setdefault("compression_capacity_model", "experimental_table_with_efficiency")
        mat.setdefault("bending_strength_MPa", 55.0)
        mat.setdefault("compression_strength_MPa", 32.0)

        mat["tension_capacity_per_stick_N"] = float(
            mat.get(
                "tension_capacity_per_stick_N",
                float(mat["tension_capacity_per_stick_kgf"]) * 9.80665,
            )
        )
        mat["compression_capacity_one_stick_N"] = float(
            mat.get(
                "compression_capacity_one_stick_N",
                float(mat["compression_capacity_one_stick_kgf"]) * 9.80665,
            )
        )
        mat["compression_capacity_two_sticks_N"] = float(
            mat.get(
                "compression_capacity_two_sticks_N",
                float(mat["compression_capacity_two_sticks_kgf"]) * 9.80665,
            )
        )

        # ---------------------------------------------------------------------
        # Competition rules
        # ---------------------------------------------------------------------
        rules = cfg.setdefault("competition_rules", {})
        rules.setdefault("enforce_nominal_stick_dimensions", False)
        rules.setdefault("required_stick_length_mm", None)
        rules.setdefault("required_stick_width_mm", None)
        rules.setdefault("required_stick_thickness_mm", None)
        rules.setdefault("stick_length_tolerance_mm", 0.5)
        rules.setdefault("stick_width_tolerance_mm", 0.2)
        rules.setdefault("stick_thickness_tolerance_mm", 0.2)

        # ---------------------------------------------------------------------
        # Support checks
        # ---------------------------------------------------------------------
        support_check = cfg.setdefault("support_check", {})
        support_check.setdefault("contact_length_per_support_node_mm", 50.0)
        support_check.setdefault("n_contact_sticks_per_support_node", 4)
        support_check.setdefault("allowable_reaction_per_support_node_kgf", 22.0)
        support_check.setdefault("negative_reaction_means_uplift", True)

        # Apoios não devem ser um gargalo artificial quando o projeto adiciona
        # palitos de sapata. A capacidade nominal base continua sendo 22 kgf
        # para 4 linhas de contato, mas pode crescer proporcionalmente ao
        # número real de linhas de palito na sapata.
        support_check.setdefault("adaptive_support_capacity_from_pad", True)
        support_check.setdefault("baseline_contact_sticks_per_support_node", 4)
        support_check.setdefault("baseline_support_pad_sticks", 3)
        support_check.setdefault("capacity_per_contact_stick_kgf", None)
        support_check.setdefault("max_effective_contact_sticks_per_node", 8)

        # ---------------------------------------------------------------------
        # Detail model / joints / fabrication
        # ---------------------------------------------------------------------
        detail = cfg.setdefault("detail_model", {})
        detail.setdefault("enabled", True)
        detail.setdefault("splice_mode", "overlap")
        detail.setdefault("overlap_length_mm", 30.0)
        detail.setdefault("min_end_margin_mm", 10.0)
        detail.setdefault("reinforcement_length_mm", 55.0)
        detail.setdefault("reinforcement_sticks_per_splice", 2)
        detail.setdefault("glue_shear_strength_MPa", 3.5)
        detail.setdefault("glue_spread_g_per_m2", 160.0)
        detail.setdefault("glue_mass_efficiency", 0.65)
        detail.setdefault("glue_cure_solids_fraction", 0.50)
        detail.setdefault("default_joint_safety_factor", 2.0)

        joint_alias = {
            "single_lap": "single_lap",
            "lap": "single_lap",
            "butt_plain": "butt_plain",
            "ponta_a_ponta": "butt_plain",
            "overlap": "single_lap",
            "single_lap_tala": "single_lap_tala",
            "butt_small_splints": "butt_small_splints",
            "butt_full_splints": "butt_full_splints",
            "double_lap": "double_lap",
            "double_lap_reinforced": "double_lap_reinforced",
            "scarf": "scarf",
            "half_lap_notched": "half_lap_notched",
        }

        tension_joint_model = str(
            detail.get("tension_joint_model", "double_lap_reinforced")
        ).strip().lower()

        compression_joint_model = str(
            detail.get("compression_joint_model", "double_lap_reinforced")
        ).strip().lower()

        detail["tension_joint_model"] = joint_alias.get(
            tension_joint_model,
            "double_lap_reinforced",
        )
        detail["compression_joint_model"] = joint_alias.get(
            compression_joint_model,
            "double_lap_reinforced",
        )

        detail.setdefault(
            "joint_efficiency_tension_by_model",
            {
                "butt_plain": 0.58,
                "single_lap": 0.86,
                "single_lap_tala": 0.94,
                "butt_small_splints": 0.98,
                "butt_full_splints": 1.04,
                "double_lap": 1.00,
                "double_lap_reinforced": 1.08,
                "scarf": 0.98,
                "half_lap_notched": 0.95,
            },
        )
        detail.setdefault(
            "joint_efficiency_compression_by_model",
            {
                "butt_plain": 0.52,
                "single_lap": 0.80,
                "single_lap_tala": 0.90,
                "butt_small_splints": 0.93,
                "butt_full_splints": 1.00,
                "double_lap": 0.98,
                "double_lap_reinforced": 1.04,
                "scarf": 0.95,
                "half_lap_notched": 0.92,
            },
        )
        detail.setdefault(
            "joint_model_rank_tension",
            [
                "double_lap_reinforced",
                "butt_full_splints",
                "double_lap",
                "butt_small_splints",
                "scarf",
                "single_lap_tala",
                "single_lap",
                "butt_plain",
            ],
        )
        detail.setdefault(
            "joint_model_rank_compression",
            [
                "double_lap_reinforced",
                "double_lap",
                "butt_full_splints",
                "scarf",
                "butt_small_splints",
                "single_lap_tala",
                "single_lap",
                "half_lap_notched",
                "butt_plain",
            ],
        )
        detail.setdefault("joint_efficiency_decay_per_splice_tension", 0.03)
        detail.setdefault("joint_efficiency_decay_per_splice_compression", 0.04)
        detail.setdefault("construction_waste_factor", 0.08)
        detail.setdefault("fast_mass_scale", 1.0)
        detail.setdefault("saw_kerf_mm", 1.0)
        detail.setdefault("imperfection_eccentricity_mm", 2.0)
        # Excentricidade global de 2 mm é conservadora para membros simples.
        # Em membros box/double_stack com juntas reforçadas, a imperfeição efetiva
        # deve ser menor, mas nunca nula. O pós-processador aplica estes fatores.
        detail.setdefault("adaptive_imperfection_eccentricity", True)
        detail.setdefault("min_imperfection_eccentricity_mm", 0.55)
        detail.setdefault(
            "imperfection_eccentricity_factor_by_layout",
            {
                "box": 0.55,
                "double_stack": 0.70,
                "side_by_side": 0.85,
                "stacked": 1.00,
                "single": 1.00,
            },
        )
        detail.setdefault(
            "imperfection_eccentricity_factor_by_group",
            {
                "top_chord": 0.85,
                "bottom_chord": 0.90,
                "vertical": 0.85,
                "diagonal": 0.90,
                "support_pad": 0.85,
            },
        )
        detail.setdefault("splice_stagger_enabled", True)
        detail.setdefault("splice_stagger_step_mm", 5.0)
        detail.setdefault("splice_stagger_max_offset_mm", 30.0)
        detail.setdefault("splice_alignment_tolerance_mm", 10.0)
        detail.setdefault("cut_increment_mm", 5.0)
        detail.setdefault("allow_cut_rounding", True)
        detail.setdefault("min_cut_length_mm", 5.0)
        detail.setdefault("allow_recommend_removal_if_fs_gt", 8.0)
        detail.setdefault("reinforce_if_fs_lt", 2.0)
        detail.setdefault("tension_only_stabilizers", True)

        detail.setdefault(
            "composite_action",
            {
                "enabled": True,
                "default_eta_I": 0.70,
                "eta_I_by_joint_quality": {
                    "weak": 0.55,
                    "normal": 0.70,
                    "laced": 0.85,
                    "continuous_box": 0.90,
                },
                "eta_A": 1.00,
            },
        )

        detail.setdefault("generate_member_templates", True)
        detail.setdefault("generate_piece_views", True)

        glue_cure = float(detail.get("glue_cure_solids_fraction", 0.50))

        if not (0.30 <= glue_cure <= 0.80):
            raise ValueError(
                "Config inválida: detail_model.glue_cure_solids_fraction "
                "deve estar entre 0.30 e 0.80."
            )

        # ---------------------------------------------------------------------
        # Section layout defaults
        # ---------------------------------------------------------------------
        layout = cfg.setdefault("section_layout_by_group", {})

        defaults_layout = {
            "bottom_chord": {
                "layout": "box",
                "stick_orientation": "edge",
                "spacing_y_mm": 10.0,
                "spacing_z_mm": 8.0,
            },
            "top_chord": {
                "layout": "box",
                "stick_orientation": "edge",
                "spacing_y_mm": 14.0,
                "spacing_z_mm": 14.0,
            },
            "vertical": {
                "layout": "box",
                "spacing_y_mm": 10.0,
                "spacing_z_mm": 10.0,
            },
            "diagonal": {
                "layout": "double_stack",
                "columns": 2,
                "spacing_y_mm": 8.0,
                "spacing_z_mm": 6.0,
            },
            "top_transverse": {
                "layout": "stacked",
                "spacing_y_mm": 0.0,
                "spacing_z_mm": 0.0,
            },
            "bottom_transverse": {
                "layout": "stacked",
                "spacing_y_mm": 0.0,
                "spacing_z_mm": 0.0,
            },
            "top_bracing": {
                "layout": "single",
                "spacing_y_mm": 0.0,
                "spacing_z_mm": 0.0,
            },
            "bottom_bracing": {
                "layout": "single",
                "spacing_y_mm": 0.0,
                "spacing_z_mm": 0.0,
            },
            "cross_frame_bracing": {
                "layout": "single",
                "spacing_y_mm": 0.0,
                "spacing_z_mm": 0.0,
            },
            "support_pad": {
                "layout": "side_by_side",
                "spacing_y_mm": 0.0,
                "spacing_z_mm": 0.0,
            },
            "chord_lacing": {
                "layout": "single",
                "spacing_y_mm": 0.0,
                "spacing_z_mm": 0.0,
            },
        }

        for key, val in defaults_layout.items():
            layout.setdefault(key, val)

        # Banzos trabalham axialmente, mas o banzo superior é governado por
        # flambagem/interação. Por padrão, palitos dos banzos ficam "de lado"
        # (lateral para cima) para aumentar Iy da seção individual. Em seção box,
        # isso melhora o eixo vertical sem sacrificar o afastamento lateral.
        for chord_group in ("top_chord", "bottom_chord"):
            chord_layout = layout.setdefault(chord_group, {})
            chord_layout.setdefault("stick_orientation", "edge")

        # ---------------------------------------------------------------------
        # Basic group defaults
        # ---------------------------------------------------------------------
        cfg.setdefault("member_sticks_by_group", {})
        cfg["member_sticks_by_group"].setdefault("chord_lacing", 1)

        cfg.setdefault("effective_length_factor_by_group", {})
        cfg["effective_length_factor_by_group"].setdefault(
            "chord_lacing",
            {
                "Ky": 1.0,
                "Kz": 1.0,
            },
        )

        # ---------------------------------------------------------------------
        # Analysis defaults
        # ---------------------------------------------------------------------
        analysis.setdefault("max_optimizer_variants", 180)
        analysis.setdefault("active_planner_enabled", True)
        analysis.setdefault("staged_fidelity_funnel_enabled", True)

        analysis.setdefault("target_min_fs", 2.0)
        analysis.setdefault("acceptance_min_primary_fs", 1.05)
        analysis.setdefault("acceptance_min_support_fs", 1.00)
        analysis.setdefault("acceptance_min_glue_fs", 1.50)
        analysis.setdefault("acceptance_min_design_breaking_load_kgf", 80.0)
        analysis.setdefault("use_target_min_fs_as_hard_acceptance", False)

        # Ponte espacial com X-bracing/top-bottom bracing não deve usar K=1.0
        # para todos os membros comprimidos. Estes valores continuam moderados
        # e só reduzem K quando o usuário não especificou valor menor.
        analysis.setdefault("auto_braced_effective_lengths", True)
        analysis.setdefault(
            "braced_effective_length_defaults",
            {
                # Estrutura com X-bracing superior/inferior e quadros transversais
                # não deve tratar o banzo comprimido como totalmente destravado.
                # Estes K ainda são moderados; apenas reduzem o excesso de punição
                # sobre membros efetivamente travados em intervalos regulares.
                "top_chord": {"Ky": 0.68, "Kz": 0.68},
                "vertical": {"Ky": 0.80, "Kz": 0.80},
                "diagonal": {"Ky": 0.85, "Kz": 0.85},
                "bottom_chord": {"Ky": 0.85, "Kz": 0.85},
            },
        )

        if bool(analysis.get("auto_braced_effective_lengths", True)):
            k_by_group = cfg.setdefault("effective_length_factor_by_group", {})
            for group, kv in (analysis.get("braced_effective_length_defaults") or {}).items():
                entry = k_by_group.setdefault(str(group), {})
                for axis in ("Ky", "Kz"):
                    target = safe_float((kv or {}).get(axis), None)
                    if target is None:
                        continue
                    current = safe_float(entry.get(axis), None)
                    entry[axis] = float(target if current is None else min(float(current), float(target)))

        analysis.setdefault("enable_tension_only_solver_globally", False)
        analysis.setdefault("enable_tension_only_solver_in_funnel", False)
        analysis.setdefault("tension_only_groups", [])
        analysis.setdefault(
            "tension_only_stages",
            {
                "S3": False,
                "S4": False,
                "S5": False,
                "S6": False,
                "S8": False,
            },
        )
        analysis.setdefault(
            "tension_only_forbidden_groups",
            [
                "cross_frame_bracing",
                "top_transverse",
                "bottom_transverse",
                "vertical",
                "top_chord",
                "bottom_chord",
                "support_pad",
                "diagonal",
            ],
        )

        enable_tension_only_global = bool(
            analysis.get("enable_tension_only_solver_globally", False)
        )
        enable_tension_only_funnel = bool(
            analysis.get("enable_tension_only_solver_in_funnel", False)
        )

        if not (enable_tension_only_global or enable_tension_only_funnel):
            analysis["tension_only_groups"] = []
            bridge["tension_only_bracing_solver_enabled"] = False
            bridge["tension_only_bracing_interpretation"] = False
        else:
            forbidden_tension_only = {
                "cross_frame_bracing",
                "top_transverse",
                "bottom_transverse",
                "vertical",
                "top_chord",
                "bottom_chord",
                "support_pad",
                "diagonal",
                "top_bracing",
                "bottom_bracing",
                "chord_lacing",
            }

            forbidden_tension_only.update(
                str(g)
                for g in (analysis.get("tension_only_forbidden_groups") or [])
            )

            analysis["tension_only_groups"] = [
                str(g)
                for g in (analysis.get("tension_only_groups") or [])
                if str(g) not in forbidden_tension_only
            ]

            if not analysis["tension_only_groups"]:
                bridge["tension_only_bracing_solver_enabled"] = False
                bridge["tension_only_bracing_interpretation"] = False

        analysis.setdefault("planner_stage1_variants", 160)
        analysis.setdefault("planner_stage1_top_k", 30)
        analysis.setdefault("planner_stage2_top_k", 12)
        analysis.setdefault("planner_stage3_top_k", 5)
        analysis.setdefault("planner_stage2a_top_k", 100)
        analysis.setdefault("planner_stage2b_top_k", 40)
        analysis.setdefault("planner_stage2_seed_cap", 12)
        analysis.setdefault("planner_stage1_eval_cap", 180)
        analysis.setdefault("planner_stage2b_eval_cap", 60)
        analysis.setdefault("planner_fallback_validation_cap", 24)
        analysis.setdefault("planner_fallback_min_fs_ratio", 0.00)
        analysis.setdefault("planner_fallback_min_break_ratio", 0.00)
        analysis.setdefault("planner_prefilter_topology_check", False)
        analysis.setdefault("planner_prefilter_mass_factor", 1.18)
        analysis.setdefault("planner_adaptive_refinement", True)
        analysis.setdefault("planner_stage4_seed_top_k", 4)
        analysis.setdefault("planner_stage4_iterations", 10)
        analysis.setdefault("planner_auto_section_layout", True)
        analysis.setdefault("planner_max_sticks_per_group", 16)
        analysis.setdefault("planner_min_sticks_per_group", 1)

        analysis.setdefault(
            "planner_max_sticks_per_group_by_group",
            {
                "top_chord": 20,
                "vertical": 16,
                "diagonal": 14,
                "bottom_transverse": 12,
                "top_transverse": 12,
                "bottom_chord": 12,
                "support_pad": 10,
                "top_bracing": 6,
                "bottom_bracing": 6,
                "cross_frame_bracing": 6,
                "chord_lacing": 4,
            },
        )

        analysis.setdefault("planner_threads", 0)
        analysis.setdefault("strict_mass_acceptance", True)
        analysis.setdefault("planner_objective_profile", "max_strength_per_competition_mass")
        analysis.setdefault("planner_objective_weight_fs", 0.65)
        analysis.setdefault("planner_objective_weight_break", 0.25)
        analysis.setdefault("planner_objective_weight_mass_target", 0.07)
        analysis.setdefault("planner_objective_weight_mass_limit", 0.03)
        analysis.setdefault("planner_allow_infeasible_recommendation", False)
        analysis.setdefault("planner_debug_enabled", True)
        analysis.setdefault("final_variants_enabled", True)
        analysis.setdefault("final_round_step_length_mm", 5.0)
        analysis.setdefault("final_round_step_section_mm", 0.1)
        analysis.setdefault("final_round_step_mass_g", 0.1)

        analysis.setdefault(
            "global_failure_groups",
            [
                "bottom_chord",
                "top_chord",
                "vertical",
                "diagonal",
                "support_pad",
            ],
        )

        analysis.setdefault(
            "local_check_only_groups",
            [
                "top_transverse",
                "bottom_transverse",
                "top_bracing",
                "bottom_bracing",
                "cross_frame_bracing",
                "chord_lacing",
            ],
        )

        analysis["primary_groups"] = list(analysis["global_failure_groups"])

        analysis.setdefault(
            "stabilizer_groups",
            [
                "top_transverse",
                "bottom_transverse",
                "top_bracing",
                "bottom_bracing",
                "cross_frame_bracing",
                "chord_lacing",
            ],
        )

        # ---------------------------------------------------------------------
        # Planner pipeline S0..S8
        # ---------------------------------------------------------------------
        pipeline_cfg = cfg.setdefault("planner_pipeline", {})
        pipeline_cfg.setdefault("mode", "staged_fidelity_funnel")
        pipeline_cfg.setdefault("macro_candidates_count", 12)
        pipeline_cfg.setdefault("fast_screening_keep_top_k", 3)
        pipeline_cfg.setdefault("multi_loadcase_keep_top_k", 2)
        pipeline_cfg.setdefault("geometry_refinement_keep_top_k", 1)
        pipeline_cfg.setdefault("allow_top2_full_detailing", False)
        pipeline_cfg.setdefault("preserve_diversity_in_fast_screening", True)
        pipeline_cfg.setdefault("defer_fabrication_detailing_until_stage", "S7")
        pipeline_cfg.setdefault("defer_topology_mutation_until_stage", "S6")
        pipeline_cfg.setdefault("defer_member_position_fine_tuning_until_stage", "S4")
        pipeline_cfg.setdefault("defer_glue_and_cut_calculation_until_stage", "S7")

        # Mantidos conforme solicitado.
        pipeline_cfg.setdefault("s2_preferred_mass_ratio", 0.95)
        pipeline_cfg.setdefault("s2_soft_mass_factor", 1.00)
        pipeline_cfg.setdefault("s2_hard_mass_reject_factor", 1.40)
        pipeline_cfg.setdefault("s2_overweight_min_break_ratio", 0.45)

        # ---------------------------------------------------------------------
        # Fast screening
        # ---------------------------------------------------------------------
        fast_screening = cfg.setdefault("fast_screening", {})
        fast_screening.setdefault("enabled", True)
        fast_screening.setdefault("use_single_load_case", True)
        fast_screening.setdefault("use_simplified_sections", True)
        fast_screening.setdefault("compute_mass_proxy_only", True)
        fast_screening.setdefault("compute_glue", False)
        fast_screening.setdefault("compute_cut_list", False)
        fast_screening.setdefault("compute_detailed_joint_model", False)

        # ---------------------------------------------------------------------
        # Multi-loadcase screening
        # ---------------------------------------------------------------------
        multi_loadcase = cfg.setdefault("multi_loadcase_screening", {})
        multi_loadcase.setdefault("enabled", True)
        multi_loadcase.setdefault(
            "strength_governing_cases",
            [
                "center",
                "torsion_60_40",
                "lateral_imperfection",
            ],
        )
        multi_loadcase.setdefault(
            "robustness_cases",
            [
                "left_offset",
                "right_offset",
            ],
        )
        multi_loadcase.setdefault(
            "service_cases",
            [
                "self_weight",
            ],
        )
        multi_loadcase.setdefault("offset_fraction_of_span", 0.05)
        multi_loadcase.setdefault(
            "load_cases",
            [
                "center",
                "left_offset",
                "right_offset",
                "torsion_60_40",
                "lateral_imperfection",
                "self_weight",
            ],
        )
        multi_loadcase.setdefault("compute_preliminary_buckling", True)
        multi_loadcase.setdefault("compute_preliminary_tension_only", True)
        multi_loadcase.setdefault("compute_zero_force_diagnostics", True)

        # ---------------------------------------------------------------------
        # Local geometry refinement
        # ---------------------------------------------------------------------
        geom_refine = cfg.setdefault("local_geometry_refinement", {})
        geom_refine.setdefault("enabled", True)
        geom_refine.setdefault("method", "trust_region_local_search")
        geom_refine.setdefault("max_iterations", 2)
        geom_refine.setdefault("patience", 1)
        geom_refine.setdefault("initial_delta_height_mm", 30.0)
        geom_refine.setdefault("initial_delta_panel_x_mm", 15.0)
        geom_refine.setdefault("initial_delta_width_mm", 20.0)
        geom_refine.setdefault("shrink_factor", 0.5)
        geom_refine.setdefault("expand_factor", 1.25)
        geom_refine.setdefault("min_delta_mm", 2.0)
        geom_refine.setdefault("max_candidates_per_iteration", 3)
        geom_refine.setdefault("load_cases", ["center", "torsion_60_40"])

        # ---------------------------------------------------------------------
        # Member sizing
        # ---------------------------------------------------------------------
        member_sizing_cfg = cfg.setdefault("member_sizing", {})
        member_sizing_cfg.setdefault("enabled", True)
        member_sizing_cfg.setdefault("method", "utilization_based_discrete_sizing")
        member_sizing_cfg.setdefault("never_reinforce_if_fs_above", 3.0)
        member_sizing_cfg.setdefault("never_reinforce_tension_if_fs_above", 3.0)
        member_sizing_cfg.setdefault("donor_fs_threshold", 4.0)
        member_sizing_cfg.setdefault("use_mass_donor_pass", True)
        member_sizing_cfg.setdefault("reinforce_by_gain_per_gram", True)

        # Alvo competitivo: não usar os 1000 g como meta real.
        # A fabricação/densidade real dos palitos varia; manter folga melhora robustez.
        member_sizing_cfg.setdefault("competitive_mass_target_ratio", 0.98)

        # Reserva maior força o S5 a escolher reforços de maior eficiência,
        # em vez de encostar no limite e depender de resgate posterior.
        member_sizing_cfg.setdefault("mass_reserve_for_fabrication_g", 20.0)

        # 12 reforços/rodada foi pouco: o trace mostra que novos gargalos
        # aparecem depois da 2ª rodada.
        member_sizing_cfg.setdefault("max_budgeted_reinforcements_per_round", 24)

        # Permite mais rodadas de redistribuição de gargalos.
        member_sizing_cfg.setdefault("max_sizing_rounds", 6)

        # O ganho global pode ficar quase plano enquanto o gargalo migra.
        # 1.02 estava exigente demais.
        member_sizing_cfg.setdefault("min_strength_gain_ratio", 1.005)

        # Não aceitar passar do limite real de massa no funil.
        member_sizing_cfg.setdefault("max_mass_overrun_ratio", 1.00)

        # Permite aceitar algumas rodadas quase planas antes da meta.
        member_sizing_cfg.setdefault("allow_flat_pre_target_rounds", 2)
        member_sizing_cfg.setdefault("critical_budget_first_fs", 1.05)
        member_sizing_cfg.setdefault("enable_post_topology_reinvestment", True)
        # Reinvestimento só pode usar massa realmente disponível abaixo do alvo competitivo.
        # Antes estava em 1.015 e reengordava a ponte para ~999 g.
        # Após o S6, reinveste parte da massa removida nos gargalos primários,
        # mas mira massa final abaixo de ~980 g. Como a massa detalhada tende a
        # ficar alguns gramas acima do proxy, usamos alvo proxy de 0.980 e reserva 4 g.
        member_sizing_cfg.setdefault("reinvest_target_proxy_mass_ratio", 0.980)
        member_sizing_cfg.setdefault("reinvest_final_mass_reserve_g", 4.0)
        member_sizing_cfg.setdefault("reinvest_max_members", 16)
        member_sizing_cfg.setdefault("reinvest_max_sticks_per_member", 1)
        member_sizing_cfg.setdefault("reinvest_fs_threshold", 1.05)
        member_sizing_cfg.setdefault("reinvest_strength_cases_only", True)
        member_sizing_cfg.setdefault("reinvest_min_abs_force_N", 25.0)

        # Rebalanceamento pós-reinvestimento: troca 1 palito de órbitas folgadas
        # para órbitas críticas equivalentes, preservando simetria e massa quase neutra.
        member_sizing_cfg.setdefault("enable_post_reinvest_rebalance", True)
        member_sizing_cfg.setdefault("rebalance_fs_threshold", 1.05)
        member_sizing_cfg.setdefault("rebalance_donor_fs_threshold", 1.16)
        member_sizing_cfg.setdefault("rebalance_max_swaps", 4)
        member_sizing_cfg.setdefault("rebalance_max_net_mass_g", 3.0)
        member_sizing_cfg.setdefault("rebalance_min_break_retention", 0.995)
        member_sizing_cfg.setdefault("rebalance_min_fs_retention", 0.995)
        member_sizing_cfg.setdefault(
            "rebalance_groups",
            ["top_chord", "vertical", "diagonal"],
        )
        member_sizing_cfg.setdefault("enable_symmetry_audit", True)

        # Mutação de eficiência de seção: melhora a inércia de banzos/montantes
        # sem adicionar palitos. Útil quando a falha é beam_column/buckling.
        member_sizing_cfg.setdefault("enable_section_efficiency_mutation", True)
        member_sizing_cfg.setdefault("section_efficiency_groups", ["top_chord", "vertical"])
        member_sizing_cfg.setdefault("section_efficiency_top_chord_spacing_candidates_mm", [16.0, 18.0, 20.0, 22.0])
        member_sizing_cfg.setdefault("section_efficiency_vertical_spacing_candidates_mm", [11.0, 12.0, 13.0, 14.0])
        member_sizing_cfg.setdefault("section_efficiency_top_chord_K_candidates", [0.62, 0.58])
        member_sizing_cfg.setdefault("section_efficiency_vertical_K_candidates", [0.76, 0.72])
        member_sizing_cfg.setdefault("section_efficiency_diagonal_K_candidates", [0.82])
        member_sizing_cfg.setdefault("section_efficiency_max_proxy_mass_ratio", 0.985)
        member_sizing_cfg.setdefault("section_efficiency_min_break_gain", 1.001)
        member_sizing_cfg.setdefault("section_efficiency_min_fs_gain", 1.001)
        member_sizing_cfg.setdefault("section_efficiency_require_bracing_for_K", True)

        # Empurrão final de resistência: usa pequena margem de massa restante
        # para reforçar órbitas primárias críticas, sem quebrar simetria.
        # Este passo é deliberadamente pós-eficiência de seção: primeiro tenta
        # ganhar por inércia/travamento; depois usa massa só onde ainda governa.
        member_sizing_cfg.setdefault("enable_final_strength_reserve_push", True)
        member_sizing_cfg.setdefault("final_strength_push_groups", ["top_chord", "vertical", "diagonal"])
        # O alvo mínimo é 80 kgf, mas o otimizador deve tentar reserva real de
        # competição. Para mirar 100 kgf, o gargalo primário precisa chegar
        # perto de FS 1,25 sob carga de 80 kgf.
        member_sizing_cfg.setdefault("ultimate_strength_target_kgf", 100.0)
        member_sizing_cfg.setdefault("final_strength_push_fs_threshold", 1.25)
        member_sizing_cfg.setdefault("final_strength_push_max_orbits", 5)
        member_sizing_cfg.setdefault("final_strength_push_max_trials", 16)
        member_sizing_cfg.setdefault("final_strength_push_max_increment_per_orbit", 1)
        member_sizing_cfg.setdefault("final_strength_push_max_proxy_mass_ratio", 0.990)
        member_sizing_cfg.setdefault("final_strength_push_min_abs_force_N", 40.0)
        member_sizing_cfg.setdefault("final_strength_push_min_break_gain", 1.001)
        member_sizing_cfg.setdefault("final_strength_push_min_fs_gain", 1.001)
        member_sizing_cfg.setdefault("final_strength_push_allow_if_below_target", True)
        member_sizing_cfg.setdefault("enable_support_pad_capacity_push", True)
        member_sizing_cfg.setdefault("support_pad_push_target_kgf", 100.0)
        member_sizing_cfg.setdefault("support_pad_push_max_group_sticks", 6)
        member_sizing_cfg.setdefault("support_pad_push_max_proxy_mass_ratio", 0.988)
        member_sizing_cfg.setdefault("support_pad_push_min_break_retention", 0.995)
        member_sizing_cfg.setdefault("support_pad_push_min_fs_retention", 0.995)
        member_sizing_cfg.setdefault("support_pad_push_proxy_mass_margin_g", 12.0)
        # Passo final: quando o reforço estrutural passou um pouco da massa,
        # reduzir apenas órbitas simétricas com FS folgado para voltar ao limite.
        member_sizing_cfg.setdefault("enable_final_mass_symmetry_trim", True)
        member_sizing_cfg.setdefault("final_mass_trim_target_proxy_mass_ratio", 0.990)
        member_sizing_cfg.setdefault("final_mass_trim_fs_threshold", 1.22)
        member_sizing_cfg.setdefault("final_mass_trim_min_break_retention", 0.985)
        member_sizing_cfg.setdefault("final_mass_trim_min_fs_retention", 0.985)
        member_sizing_cfg.setdefault("final_mass_trim_max_trials", 12)
        member_sizing_cfg.setdefault("final_mass_trim_groups", ["top_chord", "vertical", "diagonal"])
        member_sizing_cfg.setdefault("longitudinal_symmetry_for_flat_top_chord", True)
        member_sizing_cfg.setdefault("longitudinal_symmetry_flat_top_tol_mm", 3.0)


        member_sizing_cfg.setdefault(
            "sizing_load_cases",
            [
                "center",
                "torsion_60_40",
                "lateral_imperfection",
            ],
        )

        # ---------------------------------------------------------------------
        # Topology cleanup
        # ---------------------------------------------------------------------
        topology_cleanup = cfg.setdefault("topology_cleanup", {})
        topology_cleanup.setdefault("enabled", True)
        topology_cleanup.setdefault("run_after_stage", "S5")
        topology_cleanup.setdefault("allow_mixed_truss_patterns", True)
        topology_cleanup.setdefault("allow_member_removal", True)
        topology_cleanup.setdefault("allow_panel_pattern_mutation", True)
        topology_cleanup.setdefault("require_all_load_cases_for_removal", True)
        topology_cleanup.setdefault("max_topology_iterations", 6)
        topology_cleanup.setdefault("patience", 2)
        topology_cleanup.setdefault("max_remove_candidates_per_iteration", 4)
        topology_cleanup.setdefault("skip_if_break_below_ratio", 0.65)
        # Mira massa abaixo de 980 g, não apenas abaixo do limite eliminatório.
        topology_cleanup.setdefault("mass_rescue_target_ratio", 0.955)
        topology_cleanup.setdefault("mass_rescue_min_break_retention", 0.985)
        topology_cleanup.setdefault("mass_rescue_min_fs_retention", 0.985)
        topology_cleanup.setdefault(
            "preserve_member_groups",
            [
                "bottom_chord",
                "top_chord",
                "vertical",
                "diagonal",
                "support_pad",
            ],
        )
        topology_cleanup.setdefault(
            "removable_member_groups",
            [
                "top_bracing",
                "bottom_bracing",
                "cross_frame_bracing",
                "chord_lacing",
                "top_transverse",
                "bottom_transverse",
            ],
        )
        topology_cleanup.setdefault("preserve_symmetry_on_removal", True)
        topology_cleanup.setdefault("near_zero_force_threshold_N", 2.0)
        topology_cleanup.setdefault("near_zero_force_relative_threshold", 0.01)
        topology_cleanup.setdefault("preserve_stability", True)
        topology_cleanup.setdefault("preserve_load_paths", True)
        topology_cleanup.setdefault("preserve_lateral_bracing_or_update_K", True)

        # ---------------------------------------------------------------------
        # Fabrication detailing
        # ---------------------------------------------------------------------
        fabrication_detailing = cfg.setdefault("fabrication_detailing", {})
        fabrication_detailing.setdefault("run_after_stage", "S6")
        fabrication_detailing.setdefault("compute_cut_list", True)
        fabrication_detailing.setdefault("compute_glue", True)
        fabrication_detailing.setdefault("compute_cured_glue_mass", True)
        fabrication_detailing.setdefault("compute_competition_mass", True)

        # ---------------------------------------------------------------------
        # Legacy planner ranges / local sizing
        # ---------------------------------------------------------------------
        span = float(bridge["span_mm"])
        width = float(bridge["width_mm"])
        center_height = float(bridge["center_height_mm"])
        panel = float(bridge["panel_mm"])
        mass_limit = float(mat.get("mass_limit_g", 1000.0))
        load_kgf = float(bridge.get("load_total_kgf", 120.0))

        planner.setdefault("span_min_mm", 1200.0)
        planner.setdefault("span_max_mm", 1200.0)
        planner.setdefault("width_min_mm", 100.0)
        planner.setdefault("width_max_mm", 200.0)
        planner.setdefault("height_min_mm", 50.0)
        planner.setdefault("height_max_mm", min(700.0, max(420.0, center_height * 2.0)))
        planner.setdefault("panel_min_mm", max(40.0, min(140.0, panel * 0.60)))
        planner.setdefault("panel_max_mm", min(280.0, max(200.0, panel * 2.50)))
        planner.setdefault("target_load_kgf", load_kgf)
        planner.setdefault("target_breaking_load_kgf", max(80.0, load_kgf))
        planner.setdefault("stretch_breaking_load_kgf", 120.0)
        planner.setdefault("max_bridge_mass_g", mass_limit)
        planner.setdefault(
            "target_bridge_mass_g",
            min(mass_limit, max(200.0, mass_limit * 0.85)),
        )
        planner.setdefault("target_installed_stick_mass_g", float(mat.get("stick_budget_g", 900.0)))
        planner.setdefault("target_wet_glue_mass_g", float(mat.get("wet_glue_budget_g", 100.0)))

        local_sizing = planner.setdefault("local_sizing", {})
        local_sizing.setdefault("zero_force_tolerance_N", 8.0)
        local_sizing.setdefault("low_force_ratio", 0.12)
        local_sizing.setdefault("moderate_force_ratio", 0.35)
        local_sizing.setdefault("high_force_ratio", 0.65)
        local_sizing.setdefault("allow_optional_member_removal", True)
        local_sizing.setdefault("min_sticks_structural_member", 1)
        local_sizing.setdefault("min_sticks_primary_member", 2)
        local_sizing.setdefault("structural_floor_ratio_primary", 1.00)
        local_sizing.setdefault("max_local_iterations", 6)
        local_sizing.setdefault("donor_fs_threshold", 3.0)
        local_sizing.setdefault("sizing_target_fs", 1.15)
        local_sizing.setdefault("never_reinforce_if_fs_above", 3.0)
        local_sizing.setdefault("never_reinforce_tension_if_fs_above", 3.0)
        local_sizing.setdefault("mass_reserve_for_fabrication_g", 25.0)
        local_sizing.setdefault("max_budgeted_reinforcements_per_round", 12)

        for key in (
            "donor_fs_threshold",
            "never_reinforce_if_fs_above",
            "never_reinforce_tension_if_fs_above",
            "mass_reserve_for_fabrication_g",
            "max_budgeted_reinforcements_per_round",
        ):
            if key in member_sizing_cfg:
                local_sizing[key] = member_sizing_cfg[key]

        local_sizing.setdefault("allow_primary_member_lightening_if_topology_ok", False)
        local_sizing.setdefault("require_topology_validation_after_removal", True)
        local_sizing.setdefault(
            "min_sticks_primary_member_by_group",
            {
                # Guardas de forma: banzos, montantes e diagonais não podem
                # ser "emagrecidos" até quebrar a continuidade visual/estrutural.
                # O dimensionamento pode reforçar acima disso, mas não reduzir abaixo.
                "top_chord": 4,
                "bottom_chord": 2,
                "vertical": 2,
                "diagonal": 2,
                "top_transverse": 1,
                "bottom_transverse": 1,
                "support_pad": 2,
            },
        )
        local_sizing.setdefault(
            "required_groups",
            [
                "top_chord",
                "bottom_chord",
                "support_pad",
            ],
        )
        local_sizing.setdefault(
            "optional_groups",
            [
                "top_bracing",
                "bottom_bracing",
                "cross_frame_bracing",
                "chord_lacing",
            ],
        )

        # ---------------------------------------------------------------------
        # Legacy planner search spaces
        # ---------------------------------------------------------------------
        planner.setdefault(
            "consider_top_profiles",
            [
                "parker_plateau",
                "triangular_peak",
                "shallow_arch",
                "flat",
            ],
        )
        planner.setdefault(
            "consider_side_trusses",
            [
                "Parker",
                "Pratt",
                "Howe",
                "Warren",
                "Baltimore",
                "Howe_inverted",
                "Warren_mid_braced",
                "Pratt_symmetric",
                "Warren_symmetric",
            ],
        )
        planner.setdefault(
            "consider_internal_trusses",
            [
                "X",
                "Warren",
                "Pratt",
                "Howe",
                "N",
                "none",
                "Howe_inverted",
                "Warren_mid_braced",
                "Pratt_symmetric",
                "Warren_symmetric",
            ],
        )
        planner.setdefault("consider_chord_trusses", ["none", "Warren", "X"])
        planner.setdefault(
            "consider_top_chord_trusses",
            [
                "X",
                "Warren",
                "Pratt",
                "Howe",
                "N",
                "none",
                "Howe_inverted",
                "Warren_mid_braced",
                "Pratt_symmetric",
                "Warren_symmetric",
            ],
        )
        planner.setdefault(
            "consider_bottom_chord_trusses",
            [
                "X",
                "Warren",
                "Pratt",
                "Howe",
                "N",
                "none",
                "Howe_inverted",
                "Warren_mid_braced",
                "Pratt_symmetric",
                "Warren_symmetric",
            ],
        )

        unsupported_truss_names = {"k", "k_symmetric"}

        def _drop_unsupported_trusses(values: Any) -> List[Any]:
            out: List[Any] = []

            for value in list(values or []):
                key = str(value).strip().lower()

                if key in unsupported_truss_names:
                    continue

                out.append(value)

            return out

        for key in (
            "consider_side_trusses",
            "consider_internal_trusses",
            "consider_chord_trusses",
            "consider_top_chord_trusses",
            "consider_bottom_chord_trusses",
        ):
            planner[key] = _drop_unsupported_trusses(planner.get(key) or [])

        if not planner["consider_side_trusses"]:
            planner["consider_side_trusses"] = [
                "Pratt_symmetric",
                "Warren_symmetric",
                "Warren_mid_braced",
                "Howe_inverted",
                "X",
            ]

        if not planner["consider_internal_trusses"]:
            planner["consider_internal_trusses"] = [
                "X",
                "Warren_symmetric",
                "Pratt_symmetric",
                "Howe_inverted",
                "Warren_mid_braced",
            ]

        if not planner["consider_top_chord_trusses"]:
            planner["consider_top_chord_trusses"] = [
                "X",
                "Warren_symmetric",
                "Pratt_symmetric",
                "Howe_inverted",
                "Warren_mid_braced",
            ]

        if not planner["consider_bottom_chord_trusses"]:
            planner["consider_bottom_chord_trusses"] = [
                "X",
                "Warren_symmetric",
                "Pratt_symmetric",
                "Howe_inverted",
                "Warren_mid_braced",
            ]

        planner.setdefault("prefer_truss_by_material", True)

        self._validate_normalized(cfg)
        return cfg

    def from_minimal_inputs(
        self,
        base: Dict[str, Any],
        *,
        load_kgf: float,
        span_mm: float,
        width_mm: float,
        center_height_mm: float,
        panel_mm: float,
        truss_type: str = "Parker",
        top_profile: str = "parker_plateau",
        internal_truss_type: str = "X",
        chord_truss_type: str = "none",
        E_MPa: float = 6000.0,
        stick_length_mm: float = 115.0,
        stick_width_mm: float = 7.0,
        stick_thickness_mm: float = 1.5,
        stick_mass_g: float = 1.4,
        glue_shear_strength_MPa: float | None = None,
        overlap_length_mm: float | None = None,
        mass_limit_g: float | None = None,
        tension_capacity_per_stick_kgf: float | None = None,
        compression_capacity_one_stick_kgf: float | None = None,
        compression_capacity_two_sticks_kgf: float | None = None,
    ) -> Dict[str, Any]:
        cfg = copy.deepcopy(base)

        cfg.setdefault("bridge", {}).update(
            {
                "truss_type": truss_type,
                "top_profile": top_profile,
                "internal_truss_type": internal_truss_type,
                "chord_truss_type": chord_truss_type,
                "top_chord_truss_type": chord_truss_type,
                "bottom_chord_truss_type": chord_truss_type,
                "cross_frame_truss_type": internal_truss_type,
                "load_total_kgf": load_kgf,
                "span_mm": span_mm,
                "width_mm": width_mm,
                "center_height_mm": center_height_mm,
                "panel_mm": panel_mm,
                "left_support_overhang_mm": 100.0,
                "right_support_overhang_mm": 100.0,
                "end_height_mm": max(50.0, center_height_mm / 3.0),
                "plateau_start_mm": span_mm / 3.0,
                "plateau_end_mm": 2.0 * span_mm / 3.0,
                "load_distribution_x_mm": [],
                "support_contact_y_mm": [-width_mm / 2.0, width_mm / 2.0],
                "support_contact_x_left_mm": [-100.0, 0.0],
                "support_contact_x_right_mm": [span_mm, span_mm + 100.0],
            }
        )

        cfg.setdefault("material", {}).update(
            {
                "E_MPa": E_MPa,
                "stick_length_mm": stick_length_mm,
                "stick_width_mm": stick_width_mm,
                "stick_thickness_mm": stick_thickness_mm,
                "stick_mass_g": stick_mass_g,
            }
        )

        if tension_capacity_per_stick_kgf is not None:
            cfg["material"]["tension_capacity_per_stick_kgf"] = tension_capacity_per_stick_kgf
            cfg["material"].pop("tension_capacity_per_stick_N", None)

        if compression_capacity_one_stick_kgf is not None:
            cfg["material"]["compression_capacity_one_stick_kgf"] = compression_capacity_one_stick_kgf
            cfg["material"].pop("compression_capacity_one_stick_N", None)

        if compression_capacity_two_sticks_kgf is not None:
            cfg["material"]["compression_capacity_two_sticks_kgf"] = compression_capacity_two_sticks_kgf
            cfg["material"].pop("compression_capacity_two_sticks_N", None)

        if glue_shear_strength_MPa is not None:
            cfg.setdefault("detail_model", {})["glue_shear_strength_MPa"] = glue_shear_strength_MPa

        if overlap_length_mm is not None:
            cfg.setdefault("detail_model", {})["overlap_length_mm"] = overlap_length_mm

        if mass_limit_g is not None:
            cfg.setdefault("material", {})["mass_limit_g"] = mass_limit_g

        return self.normalize(cfg)

    def from_planner_inputs(
        self,
        base: Dict[str, Any],
        *,
        target_load_kgf: float,
        span_min_mm: float,
        span_max_mm: float,
        width_min_mm: float,
        width_max_mm: float,
        height_min_mm: float,
        height_max_mm: float,
        panel_min_mm: float,
        panel_max_mm: float,
        max_bridge_mass_g: float,
        target_bridge_mass_g: float | None,
        E_MPa: float,
        stick_length_mm: float,
        stick_width_mm: float,
        stick_thickness_mm: float,
        stick_mass_g: float,
        tension_capacity_per_stick_kgf: float,
        compression_capacity_one_stick_kgf: float,
        compression_capacity_two_sticks_kgf: float,
        glue_shear_strength_MPa: float,
        overlap_length_mm: float,
        target_min_fs: float,
        stage1_variants: int,
        top_chord_truss_type: str = "X",
        bottom_chord_truss_type: str = "X",
        objective_profile: str = "balanced",
        adaptive_refinement: bool = True,
        adaptive_iterations: int = 8,
    ) -> Dict[str, Any]:
        cfg = copy.deepcopy(base)

        span_mid = 0.5 * (float(span_min_mm) + float(span_max_mm))
        width_mid = 0.5 * (float(width_min_mm) + float(width_max_mm))
        height_mid = 0.5 * (float(height_min_mm) + float(height_max_mm))
        panel_mid = 0.5 * (float(panel_min_mm) + float(panel_max_mm))

        cfg.setdefault("bridge", {}).update(
            {
                "load_total_kgf": float(target_load_kgf),
                "span_mm": span_mid,
                "width_mm": width_mid,
                "center_height_mm": height_mid,
                "panel_mm": panel_mid,
                "left_support_overhang_mm": 100.0,
                "right_support_overhang_mm": 100.0,
                "end_height_mm": max(50.0, height_mid / 3.0),
                "plateau_start_mm": span_mid / 3.0,
                "plateau_end_mm": 2.0 * span_mid / 3.0,
                "load_distribution_x_mm": [],
                "support_contact_y_mm": [-width_mid / 2.0, width_mid / 2.0],
                "support_contact_x_left_mm": [-100.0, 0.0],
                "support_contact_x_right_mm": [span_mid, span_mid + 100.0],
                "truss_type": "Parker",
                "side_truss_type": "Parker",
                "internal_truss_type": "X",
                "cross_frame_truss_type": "X",
                "chord_truss_type": "none",
                "top_chord_truss_type": str(top_chord_truss_type),
                "bottom_chord_truss_type": str(bottom_chord_truss_type),
                "top_profile": "parker_plateau",
            }
        )

        cfg.setdefault("material", {}).update(
            {
                "E_MPa": float(E_MPa),
                "stick_length_mm": float(stick_length_mm),
                "stick_width_mm": float(stick_width_mm),
                "stick_thickness_mm": float(stick_thickness_mm),
                "stick_mass_g": float(stick_mass_g),
                "mass_limit_g": float(max_bridge_mass_g),
                "tension_capacity_per_stick_kgf": float(tension_capacity_per_stick_kgf),
                "compression_capacity_one_stick_kgf": float(compression_capacity_one_stick_kgf),
                "compression_capacity_two_sticks_kgf": float(compression_capacity_two_sticks_kgf),
            }
        )

        cfg.setdefault("detail_model", {}).update(
            {
                "glue_shear_strength_MPa": float(glue_shear_strength_MPa),
                "overlap_length_mm": float(overlap_length_mm),
            }
        )

        cfg.setdefault("analysis", {}).update(
            {
                "target_min_fs": float(target_min_fs),
                "active_planner_enabled": True,
                "planner_stage1_variants": int(stage1_variants),
                "planner_objective_profile": str(objective_profile),
                "planner_adaptive_refinement": bool(adaptive_refinement),
                "planner_stage4_iterations": int(adaptive_iterations),
            }
        )

        target_mass = (
            float(target_bridge_mass_g)
            if target_bridge_mass_g is not None
            else float(max_bridge_mass_g) * 0.85
        )

        cfg.setdefault("planner", {}).update(
            {
                "span_min_mm": float(span_min_mm),
                "span_max_mm": float(span_max_mm),
                "width_min_mm": float(width_min_mm),
                "width_max_mm": float(width_max_mm),
                "height_min_mm": float(height_min_mm),
                "height_max_mm": float(height_max_mm),
                "panel_min_mm": float(panel_min_mm),
                "panel_max_mm": float(panel_max_mm),
                "target_load_kgf": float(target_load_kgf),
                "target_breaking_load_kgf": float(target_load_kgf),
                "max_bridge_mass_g": float(max_bridge_mass_g),
                "target_bridge_mass_g": float(target_mass),
            }
        )

        return self.normalize(cfg)