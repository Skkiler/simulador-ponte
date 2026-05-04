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
        cfg["analysis"].setdefault("primary_groups", ["bottom_chord", "top_chord", "vertical", "diagonal", "top_transverse", "bottom_transverse", "support_pad", "chord_lacing"])
        cfg["analysis"].setdefault("stabilizer_groups", ["top_bracing", "bottom_bracing", "cross_frame_bracing"])
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
