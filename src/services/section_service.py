from __future__ import annotations

import math
from typing import Dict, List, Tuple

from src.core.numeric import safe_float


class SectionService:
    """Section properties and simplified strength models for stick members."""

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
        b = float(material["stick_width_mm"])
        t = float(material["stick_thickness_mm"])

        A1 = b * t
        Iy1 = b * t**3 / 12.0
        Iz1 = t * b**3 / 12.0

        positions: List[Tuple[float, float]] = []
        if layout == "single":
            positions = [(0.0, 0.0)]
        elif layout == "side_by_side":
            start = -0.5 * (n - 1) * b
            positions = [(start + k * b, 0.0) for k in range(n)]
        elif layout == "double_stack":
            cols = max(1, int(layout_cfg.get("columns", 2)))
            rows = int(math.ceil(n / cols))
            sy = max(float(layout_cfg.get("spacing_y_mm", b)), b)
            sz = max(float(layout_cfg.get("spacing_z_mm", t)), t)
            y0 = -0.5 * (cols - 1) * sy
            z0 = -0.5 * (rows - 1) * sz
            for idx in range(n):
                c = idx % cols
                r = idx // cols
                positions.append((y0 + c * sy, z0 + r * sz))
        elif layout == "box":
            sy = max(float(layout_cfg.get("spacing_y_mm", b + 2.0)), b)
            sz = max(float(layout_cfg.get("spacing_z_mm", t + 2.0)), t)
            if n == 1:
                positions = [(0.0, 0.0)]
            elif n == 2:
                positions = [(-sy / 2, -sz / 2), (sy / 2, sz / 2)]
            elif n == 3:
                positions = [(-sy / 2, -sz / 2), (sy / 2, -sz / 2), (0.0, sz / 2)]
            else:
                base = [(-sy / 2, -sz / 2), (sy / 2, -sz / 2), (-sy / 2, sz / 2), (sy / 2, sz / 2)]
                positions = [base[k % 4] for k in range(n)]
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
            start = -0.5 * (n - 1) * t
            positions = [(0.0, start + k * t) for k in range(n)]

        eta_I, eta_A = cls._resolve_composite_action(material, layout_cfg)

        A_perfect = n * A1
        cy = sum(y * A1 for y, _ in positions) / A_perfect if A_perfect > 0 else 0.0
        cz = sum(z * A1 for _, z in positions) / A_perfect if A_perfect > 0 else 0.0

        Iy_perfect = sum(Iy1 + A1 * (z - cz) ** 2 for y, z in positions)
        Iz_perfect = sum(Iz1 + A1 * (y - cy) ** 2 for y, z in positions)
        Iy_noncomp = n * Iy1
        Iz_noncomp = n * Iz1
        Iy = Iy_noncomp + eta_I * (Iy_perfect - Iy_noncomp)
        Iz = Iz_noncomp + eta_I * (Iz_perfect - Iz_noncomp)

        A = A_perfect * eta_A
        J = max(1e-9, 0.35 * (Iy + Iz))
        width = (max(y for y, _ in positions) - min(y for y, _ in positions) + b) if positions else b
        height = (max(z for _, z in positions) - min(z for _, z in positions) + t) if positions else t

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
            "layout": layout,
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
        if model == "experimental_table_with_efficiency":
            table_val_kgf = safe_float(table.get(str(n)), None)
            if table_val_kgf is not None:
                return float(table_val_kgf) * 9.80665

        if n <= 1:
            return c1
        if n == 2:
            return c2

        eta = SectionService._layout_efficiency(layout, n)
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
