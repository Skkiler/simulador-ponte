from __future__ import annotations

import math
from typing import Dict, List, Tuple

from src.core.numeric import safe_float


class SectionService:
    """Section properties and simplified strength models for stick members."""

    @staticmethod
    def _contact_box_sticks(
        n: int,
        stick_width_mm: float,
        stick_thickness_mm: float,
        *,
        strategy: str = "balanced",
        spacing_y_mm: float | None = None,
        spacing_z_mm: float | None = None,
    ) -> List[Dict[str, float | str]]:
        """Return per-stick geometry for a buildable contact box/tee section.

        Local axes: y = bridge/transverse width of the member section, z =
        vertical/depth of the member section.  A realistic four-stick box is not
        four equal sticks floating at the corners.  It is two vertical webs
        (edge orientation) and two horizontal caps/flanges (flat orientation),
        touching side/face around the perimeter.  That geometry is less stiff
        than the old artificial spaced box, but it is buildable and keeps the
        centroid on the solver member axis.

        Underfilled boxes are not boxes:
        - n=2: compact face-contact lamination;
        - n=3: compact mixed T/I-like lamination, dynamically checked by
          Euler/Johnson through its real I_y/I_z;
        - n>=4: contact box with balanced extra sticks added at the centroid,
          alternating flat/edge orientation.
        """
        n = max(1, int(n))
        b = float(stick_width_mm)
        t = float(stick_thickness_mm)
        # flat: broad face in y, thickness in z. edge: thickness in y, broad face in z.
        def rec(y: float, z: float, orient: str) -> Dict[str, float | str]:
            if orient == "edge":
                return {"y": y, "z": z, "ydim": t, "zdim": b, "orientation": "edge"}
            return {"y": y, "z": z, "ydim": b, "zdim": t, "orientation": "flat"}

        mode = str(strategy or "balanced").strip().lower()
        if mode in {"corner_cycle", "legacy", "legacy_corner_cycle"}:
            # Modo legado explícito para reproduzir estudos antigos: percorre os
            # cantos e pode gerar centroide excêntrico quando n é ímpar.
            sy = max(float(spacing_y_mm if spacing_y_mm is not None else 2.0 * b), b + t)
            sz = max(float(spacing_z_mm if spacing_z_mm is not None else 2.0 * b), b + t)
            corners = [
                (-sy / 2.0, -sz / 2.0),
                (sy / 2.0, -sz / 2.0),
                (-sy / 2.0, sz / 2.0),
                (sy / 2.0, sz / 2.0),
            ]
            return [rec(corners[k % 4][0], corners[k % 4][1], "edge") for k in range(n)]

        if mode in {"spaced", "spaced_box", "laced", "laced_box"} and n >= 4:
            # Espaçado só é aceito para 4+ palitos, e deve ser interpretado como
            # seção built-up com battens/lacing/talas.  O espaçamento vem do
            # layout porque é parte da geometria resistente; não pode ser inferido
            # silenciosamente.  Para n=2/3 caímos nas laminações compactas abaixo.
            sy = max(float(spacing_y_mm if spacing_y_mm is not None else 2.0 * b), b + t)
            sz = max(float(spacing_z_mm if spacing_z_mm is not None else 2.0 * b), b + t)
            corners = [
                (-sy / 2.0, -sz / 2.0),
                (sy / 2.0, -sz / 2.0),
                (-sy / 2.0, sz / 2.0),
                (sy / 2.0, sz / 2.0),
            ]
            sticks = [rec(corners[k % 4][0], corners[k % 4][1], "edge") for k in range(4)]
            extras = n - 4
            # Extras balanceados: centro ou pares simétricos.
            for k in range(extras):
                if extras % 2 == 1 and k == 0:
                    sticks.append(rec(0.0, 0.0, "edge"))
                else:
                    pair = [(-sy / 2.0, 0.0), (sy / 2.0, 0.0), (0.0, -sz / 2.0), (0.0, sz / 2.0)]
                    y, z = pair[k % len(pair)]
                    sticks.append(rec(y, z, "edge"))
            return sticks

        if n == 1:
            return [rec(0.0, 0.0, "flat")]
        if n == 2:
            # Face-to-face cap lamination; no diagonal floating pair.
            return [rec(0.0, -t / 2.0, "flat"), rec(0.0, t / 2.0, "flat")]
        if n == 3:
            # Mixed compact T/I-like lamination: one web plus two caps.  This is
            # a connected 3-stick section and usually improves the weak axis
            # relative to a pure stack without pretending to be a box.
            return [rec(0.0, 0.0, "edge"), rec(0.0, -t, "flat"), rec(0.0, t, "flat")]

        d = max(0.0, 0.5 * b - 0.5 * t)
        sticks: List[Dict[str, float | str]] = [
            rec(-d, 0.0, "edge"),
            rec(d, 0.0, "edge"),
            rec(0.0, -d, "flat"),
            rec(0.0, d, "flat"),
        ]
        # Add balanced extra material at the centroid.  Alternating orientations
        # keeps I_y and I_z similar and avoids a hidden eccentric section.
        extras = n - 4
        for k in range(extras):
            sticks.append(rec(0.0, 0.0, "flat" if k % 2 == 0 else "edge"))
        return sticks


    @staticmethod
    def rectangular_section(width_mm: float, thickness_mm: float) -> Dict[str, float]:
        b = float(width_mm)
        h = float(thickness_mm)
        A = b * h
        Iy = b * h**3 / 12.0
        Iz = h * b**3 / 12.0
        J = (1.0 / 3.0) * min(b, h) * max(b, h) ** 3
        return {"A": A, "Iy": Iy, "Iz": Iz, "J": J, "width_mm": b, "thickness_mm": h}

    @classmethod
    def equivalent_laminated_section(
        cls,
        n_sticks: int,
        material: Dict[str, float],
    ) -> Dict[str, float]:
        return cls.composite_section(n_sticks, material, {"layout": "stacked"})

    @staticmethod
    def _resolve_composite_action(
        material: Dict[str, float],
        layout_cfg: Dict | None,
    ) -> tuple[float, float]:
        cfg = layout_cfg or {}
        comp = cfg.get("composite_action")
        if not isinstance(comp, dict):
            comp = material.get("composite_action", {}) or {}

        enabled = bool(comp.get("enabled", False))
        eta_A = safe_float(comp.get("eta_A"), 1.0) or 1.0
        eta_A = max(0.50, min(1.00, eta_A))

        if not enabled:
            return 1.0, eta_A

        eta_default = safe_float(comp.get("default_eta_I"), 0.70) or 0.70
        eta_map = comp.get("eta_I_by_joint_quality", {}) or {}
        quality = str(cfg.get("joint_quality", "normal")).strip().lower()
        eta_quality = safe_float(eta_map.get(quality), None)
        eta_I = eta_default if eta_quality is None else eta_quality
        eta_I = max(0.40, min(1.00, float(eta_I)))
        return eta_I, eta_A

    @classmethod
    def composite_section(
        cls,
        n_sticks: int,
        material: Dict[str, float],
        layout_cfg: Dict | None = None,
    ) -> Dict[str, float]:
        n = max(1, int(n_sticks))
        layout_cfg = layout_cfg or {"layout": "stacked"}
        layout = str(layout_cfg.get("layout", "stacked")).lower()
        orientation_raw = str(
            layout_cfg.get(
                "stick_orientation",
                layout_cfg.get("orientation", "flat"),
            )
        ).strip().lower()
        edge_orientations = {
            "edge",
            "on_edge",
            "edge_up",
            "lateral_up",
            "side_up",
            "lado",
            "lateral",
            "em_pe",
            "em_pé",
            "vertical",
        }
        stick_orientation = "edge" if orientation_raw in edge_orientations else "flat"

        b = float(material["stick_width_mm"])
        t = float(material["stick_thickness_mm"])

        # Eixos locais da seção do membro: y = largura lateral, z = altura vertical.
        # flat: face larga "deitada" (largura em y, espessura em z).
        # edge: palito "de lado"/lateral para cima (espessura em y, largura em z).
        # Para banzos comprimidos, edge aumenta fortemente Iy, pois I = b*h^3/12.
        stick_y_mm = t if stick_orientation == "edge" else b
        stick_z_mm = b if stick_orientation == "edge" else t

        A1 = stick_y_mm * stick_z_mm
        Iy1 = stick_y_mm * stick_z_mm**3 / 12.0
        Iz1 = stick_z_mm * stick_y_mm**3 / 12.0

        positions: List[Tuple[float, float]] = []
        stick_defs: List[Dict[str, float | str]] | None = None
        if layout == "single":
            positions = [(0.0, 0.0)]
        elif layout == "side_by_side":
            start = -0.5 * (n - 1) * stick_y_mm
            positions = [(start + k * stick_y_mm, 0.0) for k in range(n)]
        elif layout == "double_stack":
            cols = max(1, int(layout_cfg.get("columns", 2)))
            rows = int(math.ceil(n / cols))
            sy = max(float(layout_cfg.get("spacing_y_mm", stick_y_mm)), stick_y_mm)
            sz = max(float(layout_cfg.get("spacing_z_mm", stick_z_mm)), stick_z_mm)
            y0 = -0.5 * (cols - 1) * sy
            z0 = -0.5 * (rows - 1) * sz
            for idx in range(n):
                c = idx % cols
                r = idx // cols
                positions.append((y0 + c * sy, z0 + r * sz))
        elif layout == "box":
            stick_defs = cls._contact_box_sticks(
                n,
                b,
                t,
                strategy=str(layout_cfg.get("box_extra_stick_strategy", "balanced")),
                spacing_y_mm=safe_float(layout_cfg.get("spacing_y_mm"), None),
                spacing_z_mm=safe_float(layout_cfg.get("spacing_z_mm"), None),
            )
            positions = [(float(v["y"]), float(v["z"])) for v in stick_defs]
        elif layout == "custom":
            raw = layout_cfg.get("stick_positions_yz", []) or []
            for yz in raw[:n]:
                if isinstance(yz, (list, tuple)) and len(yz) >= 2:
                    positions.append((float(yz[0]), float(yz[1])))
            if not positions:
                positions = [(0.0, 0.0) for _ in range(n)]
            while len(positions) < n:
                positions.append(positions[-1])
        else:
            start = -0.5 * (n - 1) * stick_z_mm
            positions = [(0.0, start + k * stick_z_mm) for k in range(n)]

        eta_I, eta_A = cls._resolve_composite_action(material, layout_cfg)

        if stick_defs is None:
            stick_defs = [
                {
                    "y": y,
                    "z": z,
                    "ydim": stick_y_mm,
                    "zdim": stick_z_mm,
                    "orientation": stick_orientation,
                }
                for y, z in positions
            ]

        A_perfect = sum(float(v["ydim"]) * float(v["zdim"]) for v in stick_defs)
        cy = (
            sum(float(v["y"]) * float(v["ydim"]) * float(v["zdim"]) for v in stick_defs) / A_perfect
            if A_perfect > 0
            else 0.0
        )
        cz = (
            sum(float(v["z"]) * float(v["ydim"]) * float(v["zdim"]) for v in stick_defs) / A_perfect
            if A_perfect > 0
            else 0.0
        )

        Iy_perfect = 0.0
        Iz_perfect = 0.0
        Iy_noncomp = 0.0
        Iz_noncomp = 0.0
        for v in stick_defs:
            y = float(v["y"])
            z = float(v["z"])
            yd = float(v["ydim"])
            zd = float(v["zdim"])
            Ai = yd * zd
            Iyi = yd * zd**3 / 12.0
            Izi = zd * yd**3 / 12.0
            Iy_noncomp += Iyi
            Iz_noncomp += Izi
            Iy_perfect += Iyi + Ai * (z - cz) ** 2
            Iz_perfect += Izi + Ai * (y - cy) ** 2

        Iy = Iy_noncomp + eta_I * (Iy_perfect - Iy_noncomp)
        Iz = Iz_noncomp + eta_I * (Iz_perfect - Iz_noncomp)

        A = A_perfect * eta_A
        J = max(1e-9, 0.35 * (Iy + Iz))
        if stick_defs:
            y_min = min(float(v["y"]) - 0.5 * float(v["ydim"]) for v in stick_defs)
            y_max = max(float(v["y"]) + 0.5 * float(v["ydim"]) for v in stick_defs)
            z_min = min(float(v["z"]) - 0.5 * float(v["zdim"]) for v in stick_defs)
            z_max = max(float(v["z"]) + 0.5 * float(v["zdim"]) for v in stick_defs)
            width = y_max - y_min
            height = z_max - z_min
        else:
            width = stick_y_mm
            height = stick_z_mm

        orientations = [str(v.get("orientation", stick_orientation)) for v in stick_defs]
        ydims = [float(v.get("ydim", stick_y_mm)) for v in stick_defs]
        zdims = [float(v.get("zdim", stick_z_mm)) for v in stick_defs]
        effective_layout = layout
        box_strategy = str(layout_cfg.get("box_extra_stick_strategy", "balanced")).strip().lower()
        spaced_strategies = {"corner_cycle", "legacy", "legacy_corner_cycle", "spaced", "spaced_box", "laced", "laced_box"}
        if layout == "box" and n < 4:
            effective_layout = "tee3" if n == 3 else ("laminated2" if n == 2 else "single")
        elif layout == "box" and box_strategy in spaced_strategies:
            effective_layout = "laced_box"
        elif layout == "box":
            effective_layout = "contact_box"
        section_connection_model = {
            "single": "single_stick",
            "laminated2": "face_to_face_lamination",
            "tee3": "mixed_T_contact_lamination",
            "contact_box": "four_side_contact_box_with_face_side_glue",
            "laced_box": "spaced_built_up_box_requires_battens_lacing",
        }.get(effective_layout, effective_layout)

        return {
            "A": A,
            "Iy": Iy,
            "Iz": Iz,
            "J": J,
            "n_sticks": n,
            "width_mm": width,
            "thickness_mm": height,
            "centroid_y_mm": cy,
            "centroid_z_mm": cz,
            "layout": effective_layout,
            "requested_layout": layout,
            "stick_orientation": stick_orientation if len(set(orientations)) == 1 else "mixed",
            "stick_orientations": orientations,
            "section_connection_model": section_connection_model,
            "stick_width_y_mm": stick_y_mm,
            "stick_height_z_mm": stick_z_mm,
            "stick_width_y_mm_by_lane": ydims,
            "stick_height_z_mm_by_lane": zdims,
            "stick_positions_yz": positions,
            "buckling_I_critical_mm4": min(Iy, Iz),
            "Iy_perfect": Iy_perfect,
            "Iz_perfect": Iz_perfect,
            "Iy_noncomposite": Iy_noncomp,
            "Iz_noncomposite": Iz_noncomp,
            "eta_I": eta_I,
            "eta_A": eta_A,
        }

    @staticmethod
    def _layout_efficiency(layout: str, n_sticks: int) -> float:
        base = max(0.65, 1.0 - 0.035 * max(0, n_sticks - 2))
        l = str(layout or "").lower()
        if l in {"box", "double_stack"}:
            base += 0.08
        if l in {"laced", "continuous_box"}:
            base += 0.12
        return max(0.65, min(1.00, base))

    @staticmethod
    def _effective_compression_stress_MPa(material: Dict[str, float]) -> float:
        """Material-level direct-compression stress used after table anchors.

        The edital gives rupture loads for one isolated stick and for a two-stick
        glued composition.  Those values are indispensable anchors, but they are
        not enough to extrapolate multi-stick box sections.  For n >= 3 this
        helper estimates a conservative *direct* compression stress from the
        measured anchors and from a user-visible fraction of the configured wood
        compression strength.  Euler/Johnson remains checked separately and may
        still govern long or weak-axis members.
        """
        b = float(material.get("stick_width_mm", 0.0) or 0.0)
        t = float(material.get("stick_thickness_mm", 0.0) or 0.0)
        A1 = max(1.0e-9, b * t)
        c1 = float(material.get("compression_capacity_one_stick_N", 0.0) or 0.0)
        c2 = float(material.get("compression_capacity_two_sticks_N", 0.0) or 0.0)
        anchor_sigma = max(c1 / A1, c2 / (2.0 * A1), 1.0e-9)
        wood_sigma = safe_float(material.get("compression_strength_MPa"), None)
        if wood_sigma is None or wood_sigma <= 0:
            return float(anchor_sigma)
        factor = safe_float(material.get("compression_area_strength_factor"), 0.22) or 0.22
        factor = max(0.05, min(0.60, float(factor)))
        return float(max(anchor_sigma, min(float(wood_sigma), float(wood_sigma) * factor)))

    @staticmethod
    def compression_capacity_N(
        n_sticks: int,
        material: Dict[str, float],
        *,
        layout: str = "stacked",
    ) -> float:
        n = max(1, int(n_sticks))
        c1 = float(material["compression_capacity_one_stick_N"])
        c2 = float(material["compression_capacity_two_sticks_N"])
        model = str(
            material.get(
                "compression_capacity_model",
                "linear_by_two_stick_capacity",
            )
        ).strip().lower()

        table = material.get("compression_capacity_table_kgf", {}) or {}
        if model in {"experimental_table_with_efficiency", "experimental_table_with_area_cap"}:
            table_val_kgf = safe_float(table.get(str(n)), None)
            if table_val_kgf is not None:
                return float(table_val_kgf) * 9.80665

        if n <= 1:
            return c1
        if n == 2:
            return c2

        eta = SectionService._layout_efficiency(layout, n)
        if model in {"experimental_table_with_area_cap", "table_anchor_area_cap", "experimental_table_with_efficiency_and_area_cap"}:
            b = float(material.get("stick_width_mm", 0.0) or 0.0)
            t = float(material.get("stick_thickness_mm", 0.0) or 0.0)
            A = max(1.0e-9, n * b * t)
            sigma_eff = SectionService._effective_compression_stress_MPa(material)
            area_cap = A * sigma_eff * eta

            # Preserve the edital anchors while avoiding an unconstrained jump for
            # many sticks.  The multiplier is intentionally conservative and
            # documented in the report/config, because adhesive quality and
            # eccentricity dominate real popsicle-stick compression tests.
            linear_anchor = (n / 2.0) * c2 * eta
            max_mult = safe_float(material.get("compression_area_cap_max_multiplier_vs_table", 1.50), 1.50) or 1.50
            max_mult = max(1.0, min(2.0, float(max_mult)))
            return max(linear_anchor, min(area_cap, linear_anchor * max_mult))

        if model in {"experimental_table_with_efficiency", "one_or_two_then_linear_by_two_stick_capacity"}:
            base = (n / 2.0) * c2
            return base * eta
        return n * (c2 / 2.0)

    @staticmethod
    def tension_capacity_N(n_sticks: int, material: Dict[str, float]) -> float:
        return max(1, int(n_sticks)) * float(material["tension_capacity_per_stick_N"])

    @staticmethod
    def euler_buckling_N(E_MPa: float, I_mm4: float, K: float, L_mm: float) -> float:
        if L_mm <= 0:
            return float("inf")
        return (math.pi**2 * float(E_MPa) * float(I_mm4)) / ((float(K) * float(L_mm)) ** 2)

    @staticmethod
    def radius_of_gyration(I_mm4: float, A_mm2: float) -> float:
        return math.sqrt(max(0.0, I_mm4) / A_mm2) if A_mm2 > 0 else 0.0

    @staticmethod
    def slenderness_ratio(K: float, L_mm: float, I_mm4: float, A_mm2: float) -> float | None:
        r = SectionService.radius_of_gyration(I_mm4, A_mm2)
        if r <= 0 or L_mm <= 0:
            return None
        return float(K) * float(L_mm) / r

    @staticmethod
    def johnson_buckling_N(
        E_MPa: float,
        A_mm2: float,
        r_mm: float,
        K: float,
        L_mm: float,
        sigma_c_MPa: float,
    ) -> float:
        if r_mm <= 0 or L_mm <= 0 or A_mm2 <= 0:
            return 0.0
        slender = (float(K) * float(L_mm)) / float(r_mm)
        coeff = float(sigma_c_MPa) / (4.0 * math.pi**2 * float(E_MPa))
        sigma_allow = float(sigma_c_MPa) * max(0.0, 1.0 - coeff * slender**2)
        return max(0.0, sigma_allow * float(A_mm2))

    @staticmethod
    def column_capacity_N(
        *,
        E_MPa: float,
        A_mm2: float,
        I_mm4: float,
        K: float,
        L_mm: float,
        sigma_c_MPa: float,
        method: str = "auto",
        eccentricity_mm: float = 0.0,
    ) -> Dict[str, float | str | None]:
        r = SectionService.radius_of_gyration(I_mm4, A_mm2)
        slender = SectionService.slenderness_ratio(K, L_mm, I_mm4, A_mm2)
        euler = SectionService.euler_buckling_N(E_MPa, I_mm4, K, L_mm)
        johnson = SectionService.johnson_buckling_N(E_MPa, A_mm2, r, K, L_mm, sigma_c_MPa)

        m = str(method or "auto").strip().lower()
        if m == "euler":
            cap = euler
            used = "euler"
        elif m == "johnson":
            cap = johnson
            used = "johnson"
        else:
            cc = math.sqrt(max(1.0e-9, 2.0 * math.pi**2 * float(E_MPa) / max(1.0e-9, float(sigma_c_MPa))))
            if slender is not None and slender <= cc:
                cap = max(0.0, min(euler, johnson if johnson > 0 else euler))
                used = "johnson"
            else:
                cap = euler
                used = "euler"

        if abs(float(eccentricity_mm or 0.0)) > 1.0e-9:
            ecc_ratio = abs(float(eccentricity_mm)) / max(1.0e-6, r)
            reduction = 1.0 / (1.0 + 0.25 * ecc_ratio)
            cap *= max(0.35, min(1.0, reduction))
            used = f"{used}_secant_approx"

        return {
            "capacity_N": float(max(0.0, cap)),
            "method": used,
            "slenderness": slender,
            "euler_N": float(euler),
            "johnson_N": float(johnson),
            "r_mm": float(r),
        }

    @staticmethod
    def member_length_mm(ni, nj) -> float:
        dx = ni.x - nj.x
        dy = ni.y - nj.y
        dz = ni.z - nj.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def splice_efficiency_factor(
        *,
        L_mm: float,
        stick_length_mm: float,
        overlap_length_mm: float,
        model_efficiency: float,
        decay_per_splice: float,
        min_factor: float = 0.55,
        max_factor: float = 1.20,
    ) -> float:
        """Simple member splice efficiency factor."""
        L = max(0.0, float(L_mm))
        stick = max(1.0e-6, float(stick_length_mm))
        overlap = max(0.0, min(float(overlap_length_mm), stick * 0.85))
        step = max(1.0e-6, stick - overlap)

        if L <= stick:
            splices = 0
        else:
            pieces = int(math.ceil((L - stick) / step)) + 1
            splices = max(0, pieces - 1)

        if splices <= 0:
            return 1.0

        eta = float(model_efficiency) * (1.0 - float(decay_per_splice) * splices)
        return max(float(min_factor), min(float(max_factor), eta))
