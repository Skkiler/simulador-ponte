from __future__ import annotations
import copy
import json
from pathlib import Path
from typing import Any, Dict


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

    def normalize(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        cfg = copy.deepcopy(cfg)
        bridge = cfg.setdefault("bridge", {})
        mat = cfg.setdefault("material", {})
        planner = cfg.setdefault("planner", {})

        bridge.setdefault("truss_type", bridge.get("side_truss_type", "Parker"))
        bridge.setdefault("side_truss_type", bridge.get("truss_type", "Parker"))
        bridge.setdefault("internal_truss_type", "X")
        bridge.setdefault("chord_truss_type", "none")
        bridge.setdefault("cross_frame_truss_type", bridge.get("internal_truss_type", "X"))
        bridge.setdefault("top_profile", "parker_plateau")
        bridge.setdefault("span_mm", 1200.0)
        bridge.setdefault("panel_mm", 100.0)
        bridge.setdefault("width_mm", 180.0)
        bridge.setdefault("left_support_overhang_mm", 100.0)
        bridge.setdefault("right_support_overhang_mm", 100.0)
        bridge.setdefault("end_height_mm", 100.0)
        bridge.setdefault("center_height_mm", 300.0)
        bridge.setdefault("plateau_start_mm", bridge["span_mm"] / 3.0)
        bridge.setdefault("plateau_end_mm", 2.0 * bridge["span_mm"] / 3.0)
        bridge.setdefault("load_total_kgf", 120.0)
        bridge["load_total_N"] = float(bridge["load_total_kgf"]) * 9.80665

        if "load_distribution_x_mm" not in bridge or not bridge["load_distribution_x_mm"]:
            p = float(bridge["panel_mm"])
            xs = []
            x = float(bridge["plateau_start_mm"])
            while x <= float(bridge["plateau_end_mm"]) + 1e-9:
                xs.append(round(x, 6))
                x += p
            bridge["load_distribution_x_mm"] = xs or [bridge["span_mm"] / 2.0]

        half_w = float(bridge["width_mm"]) / 2.0
        bridge.setdefault("support_contact_y_mm", [-half_w, half_w])
        bridge.setdefault("support_contact_x_left_mm", [-bridge["left_support_overhang_mm"], 0.0])
        bridge.setdefault("support_contact_x_right_mm", [bridge["span_mm"], bridge["span_mm"] + bridge["right_support_overhang_mm"]])

        mat.setdefault("E_MPa", 6000.0)
        mat.setdefault("G_MPa", 500.0)
        mat.setdefault("stick_length_mm", 120.0)
        mat.setdefault("stick_width_mm", 7.0)
        mat.setdefault("stick_thickness_mm", 1.5)
        mat.setdefault("stick_mass_g", 1.4)
        mat.setdefault("mass_limit_g", 1000.0)
        mat.setdefault("glue_reserved_g", 100.0)
        vol = max(1e-9, float(mat["stick_length_mm"]) * float(mat["stick_width_mm"]) * float(mat["stick_thickness_mm"]))
        mat["density_g_per_mm3"] = float(mat.get("density_g_per_mm3", float(mat["stick_mass_g"]) / vol))
        mat.setdefault("tension_capacity_per_stick_kgf", 72.0)
        mat.setdefault("compression_capacity_one_stick_kgf", 4.0)
        mat.setdefault("compression_capacity_two_sticks_kgf", 11.0)

        support_check = cfg.setdefault("support_check", {})
        support_check.setdefault("contact_length_per_support_node_mm", 50.0)
        support_check.setdefault("n_contact_sticks_per_support_node", 4)
        support_check.setdefault("allowable_reaction_per_support_node_kgf", 22.0)
        support_check.setdefault("negative_reaction_means_uplift", True)

        detail = cfg.setdefault("detail_model", {})
        detail.setdefault("enabled", True)
        detail.setdefault("splice_mode", "overlap")
        detail.setdefault("overlap_length_mm", 30.0)
        detail.setdefault("min_end_margin_mm", 15.0)
        detail.setdefault("reinforcement_length_mm", 55.0)
        detail.setdefault("reinforcement_sticks_per_splice", 2)
        detail.setdefault("glue_shear_strength_MPa", 3.5)
        detail.setdefault("glue_spread_g_per_m2", 160.0)
        detail.setdefault("glue_mass_efficiency", 0.65)
        detail.setdefault("default_joint_safety_factor", 2.0)
        detail.setdefault("construction_waste_factor", 0.08)
        detail.setdefault("saw_kerf_mm", 1.0)
        detail.setdefault("imperfection_eccentricity_mm", 2.0)
        detail.setdefault("allow_recommend_removal_if_fs_gt", 8.0)
        detail.setdefault("reinforce_if_fs_lt", 2.0)
        detail.setdefault("tension_only_stabilizers", True)
        detail.setdefault("generate_member_templates", True)
        detail.setdefault("generate_piece_views", True)

        cfg.setdefault("section_layout_by_group", {})
        layout = cfg["section_layout_by_group"]
        defaults_layout = {
            "bottom_chord": {"layout": "stacked", "spacing_y_mm": 0.0, "spacing_z_mm": 0.0},
            "top_chord": {"layout": "box", "spacing_y_mm": 12.0, "spacing_z_mm": 12.0},
            "vertical": {"layout": "stacked", "spacing_y_mm": 0.0, "spacing_z_mm": 0.0},
            "diagonal": {"layout": "stacked", "spacing_y_mm": 0.0, "spacing_z_mm": 0.0},
            "top_transverse": {"layout": "stacked", "spacing_y_mm": 0.0, "spacing_z_mm": 0.0},
            "bottom_transverse": {"layout": "stacked", "spacing_y_mm": 0.0, "spacing_z_mm": 0.0},
            "top_bracing": {"layout": "single", "spacing_y_mm": 0.0, "spacing_z_mm": 0.0},
            "bottom_bracing": {"layout": "single", "spacing_y_mm": 0.0, "spacing_z_mm": 0.0},
            "cross_frame_bracing": {"layout": "single", "spacing_y_mm": 0.0, "spacing_z_mm": 0.0},
            "support_pad": {"layout": "side_by_side", "spacing_y_mm": 0.0, "spacing_z_mm": 0.0},
            "chord_lacing": {"layout": "single", "spacing_y_mm": 0.0, "spacing_z_mm": 0.0},
        }
        for key, val in defaults_layout.items():
            layout.setdefault(key, val)

        mat["tension_capacity_per_stick_N"] = float(mat.get("tension_capacity_per_stick_N", mat["tension_capacity_per_stick_kgf"] * 9.80665))
        mat["compression_capacity_one_stick_N"] = float(mat.get("compression_capacity_one_stick_N", mat["compression_capacity_one_stick_kgf"] * 9.80665))
        mat["compression_capacity_two_sticks_N"] = float(mat.get("compression_capacity_two_sticks_N", mat["compression_capacity_two_sticks_kgf"] * 9.80665))

        cfg.setdefault("member_sticks_by_group", {})
        cfg["member_sticks_by_group"].setdefault("chord_lacing", 1)
        cfg.setdefault("effective_length_factor_by_group", {})
        cfg["effective_length_factor_by_group"].setdefault("chord_lacing", {"Ky": 1.0, "Kz": 1.0})
        cfg.setdefault("analysis", {})
        cfg["analysis"].setdefault("max_optimizer_variants", 180)
        cfg["analysis"].setdefault("active_planner_enabled", True)
        cfg["analysis"].setdefault("target_min_fs", 2.0)
        cfg["analysis"].setdefault("planner_stage1_variants", 220)
        cfg["analysis"].setdefault("planner_stage1_top_k", 42)
        cfg["analysis"].setdefault("planner_stage2_top_k", 14)
        cfg["analysis"].setdefault("planner_stage3_top_k", 6)
        cfg["analysis"].setdefault("planner_adaptive_refinement", True)
        cfg["analysis"].setdefault("planner_stage4_seed_top_k", 4)
        cfg["analysis"].setdefault("planner_stage4_iterations", 8)
        cfg["analysis"].setdefault("planner_max_sticks_per_group", 12)
        cfg["analysis"].setdefault("planner_min_sticks_per_group", 1)
        cfg["analysis"].setdefault("planner_objective_profile", "balanced")
        cfg["analysis"].setdefault("planner_objective_weight_fs", 0.52)
        cfg["analysis"].setdefault("planner_objective_weight_break", 0.28)
        cfg["analysis"].setdefault("planner_objective_weight_mass_target", 0.12)
        cfg["analysis"].setdefault("planner_objective_weight_mass_limit", 0.08)
        cfg["analysis"].setdefault("final_variants_enabled", True)
        cfg["analysis"].setdefault("final_round_step_length_mm", 5.0)
        cfg["analysis"].setdefault("final_round_step_section_mm", 0.1)
        cfg["analysis"].setdefault("final_round_step_mass_g", 0.1)
        cfg["analysis"].setdefault("primary_groups", ["bottom_chord", "top_chord", "vertical", "diagonal", "top_transverse", "bottom_transverse", "support_pad", "chord_lacing"])
        cfg["analysis"].setdefault("stabilizer_groups", ["top_bracing", "bottom_bracing", "cross_frame_bracing"])

        span = float(bridge["span_mm"])
        width = float(bridge["width_mm"])
        center_height = float(bridge["center_height_mm"])
        panel = float(bridge["panel_mm"])
        mass_limit = float(mat.get("mass_limit_g", 1000.0))
        load_kgf = float(bridge.get("load_total_kgf", 120.0))

        planner.setdefault("span_min_mm", max(400.0, span * 0.85))
        planner.setdefault("span_max_mm", min(3000.0, span * 1.15))
        planner.setdefault("width_min_mm", max(80.0, width * 0.85))
        planner.setdefault("width_max_mm", min(300.0, width * 1.2))
        planner.setdefault("height_min_mm", max(80.0, center_height * 0.75))
        planner.setdefault("height_max_mm", min(700.0, center_height * 1.25))
        planner.setdefault("panel_min_mm", max(40.0, panel * 0.85))
        planner.setdefault("panel_max_mm", min(220.0, panel * 1.2))
        planner.setdefault("target_load_kgf", load_kgf)
        planner.setdefault("target_breaking_load_kgf", load_kgf)
        planner.setdefault("max_bridge_mass_g", mass_limit)
        planner.setdefault("target_bridge_mass_g", min(mass_limit, max(200.0, mass_limit * 0.85)))
        planner.setdefault("consider_top_profiles", ["parker_plateau", "triangular_peak", "shallow_arch", "flat"])
        planner.setdefault("consider_side_trusses", ["Parker", "Pratt", "Howe", "Warren"])
        planner.setdefault("consider_internal_trusses", ["X", "Warren", "Pratt", "Howe", "none"])
        planner.setdefault("consider_chord_trusses", ["none", "Warren", "X"])
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
        stick_length_mm: float = 120.0,
        stick_width_mm: float = 7.0,
        stick_thickness_mm: float,
        stick_mass_g: float,
        glue_shear_strength_MPa: float | None = None,
        overlap_length_mm: float | None = None,
        mass_limit_g: float | None = None,
        tension_capacity_per_stick_kgf: float | None = None,
        compression_capacity_one_stick_kgf: float | None = None,
        compression_capacity_two_sticks_kgf: float | None = None,
    ) -> Dict[str, Any]:
        cfg = copy.deepcopy(base)
        cfg["bridge"].update(
            {
                "truss_type": truss_type,
                "top_profile": top_profile,
                "internal_truss_type": internal_truss_type,
                "chord_truss_type": chord_truss_type,
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
        cfg["material"].update(
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
