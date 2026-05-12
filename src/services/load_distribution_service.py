from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List

from src.domain.models import Load


class LoadDistributionService:
    """Build nodal loads for point, line and plate-footprint load models.

    The structural solver only accepts nodal forces.  This service converts a
    physically loaded area on the deck into equivalent nodal forces using tributary
    station weights in x and side weights in y.  With a zero footprint length it
    preserves the old point-station behaviour.
    """

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(value)))

    @staticmethod
    def load_level(cfg: Dict[str, Any]) -> str:
        level = str((cfg.get("bridge", {}) or {}).get("load_application_level", "top")).strip().lower()
        return level if level in {"top", "bottom"} else "top"

    @staticmethod
    def configured_targets(cfg: Dict[str, Any]) -> List[float]:
        bridge = cfg.get("bridge", {}) or {}
        span = float(bridge.get("span_mm", 1200.0))
        raw = bridge.get("load_distribution_x_mm") or []
        targets: List[float] = []

        for value in raw:
            try:
                targets.append(LoadDistributionService._clamp(float(value), 0.0, span))
            except (TypeError, ValueError):
                continue

        if not targets:
            targets = [0.5 * span]

        return sorted(set(round(v, 6) for v in targets))

    @staticmethod
    def shifted_targets(cfg: Dict[str, Any], delta_mm: float) -> List[float]:
        span = float((cfg.get("bridge", {}) or {}).get("span_mm", 1200.0))
        return [
            LoadDistributionService._clamp(x + float(delta_mm), 0.0, span)
            for x in LoadDistributionService.configured_targets(cfg)
        ]

    @staticmethod
    def _station_influence_intervals(xs: List[float], span: float) -> Dict[float, tuple[float, float]]:
        if not xs:
            return {}
        if len(xs) == 1:
            return {xs[0]: (0.0, span)}

        out: Dict[float, tuple[float, float]] = {}
        for i, x in enumerate(xs):
            left = 0.0 if i == 0 else 0.5 * (xs[i - 1] + x)
            right = span if i == len(xs) - 1 else 0.5 * (x + xs[i + 1])
            out[x] = (max(0.0, left), min(span, right))
        return out

    @staticmethod
    def _interpolated_station_weights(x_target: float, xs: List[float], span: float) -> Dict[float, float]:
        if not xs:
            return {}
        x = LoadDistributionService._clamp(float(x_target), 0.0, span)
        if x <= xs[0]:
            return {xs[0]: 1.0}
        if x >= xs[-1]:
            return {xs[-1]: 1.0}

        for x0, x1 in zip(xs[:-1], xs[1:]):
            if x0 - 1.0e-9 <= x <= x1 + 1.0e-9:
                if abs(x1 - x0) <= 1.0e-9:
                    return {x0: 1.0}
                w1 = (x - x0) / (x1 - x0)
                w0 = 1.0 - w1
                out: Dict[float, float] = {}
                if w0 > 1.0e-12:
                    out[x0] = float(w0)
                if w1 > 1.0e-12:
                    out[x1] = float(w1)
                return out or {x0: 1.0}
        return {min(xs, key=lambda xv: abs(xv - x)): 1.0}

    @classmethod
    def station_weights(
        cls,
        cfg: Dict[str, Any],
        x_targets: Iterable[float] | None,
        xs: List[float],
    ) -> Dict[float, float]:
        bridge = cfg.get("bridge", {}) or {}
        span = float(bridge.get("span_mm", 1200.0))
        xs = sorted({round(float(x), 6) for x in xs if -1.0e-6 <= float(x) <= span + 1.0e-6})
        if not xs:
            return {}

        targets = list(x_targets or cls.configured_targets(cfg))
        footprint = max(0.0, float(bridge.get("load_footprint_length_mm", 0.0) or 0.0))
        model = str(bridge.get("load_distribution_model", "point_stations")).strip().lower()

        weights: Dict[float, float] = defaultdict(float)
        if footprint <= 1.0e-9 or model in {"point", "point_station", "point_stations"}:
            for target in targets:
                for station, w in cls._interpolated_station_weights(float(target), xs, span).items():
                    weights[station] += w
        else:
            intervals = cls._station_influence_intervals(xs, span)
            for target in targets:
                center = cls._clamp(float(target), 0.0, span)
                lo = cls._clamp(center - 0.5 * footprint, 0.0, span)
                hi = cls._clamp(center + 0.5 * footprint, 0.0, span)
                if hi <= lo + 1.0e-9:
                    for station, w in cls._interpolated_station_weights(center, xs, span).items():
                        weights[station] += w
                    continue
                for station, (a, b) in intervals.items():
                    overlap = max(0.0, min(hi, b) - max(lo, a))
                    if overlap > 1.0e-12:
                        weights[station] += overlap

        total = sum(max(0.0, v) for v in weights.values())
        if total <= 1.0e-12:
            return cls._interpolated_station_weights(0.5 * span, xs, span)
        return {float(k): float(v) / total for k, v in weights.items() if v > 1.0e-12}

    @staticmethod
    def _side_factor(y: float, side_bias: Dict[str, float] | None) -> float:
        if not side_bias:
            return 1.0
        left = float(side_bias.get("left", 0.5))
        right = float(side_bias.get("right", 0.5))
        return left if float(y) < 0.0 else right

    @classmethod
    def nodal_weights(
        cls,
        cfg: Dict[str, Any],
        nodes: List[Any],
        *,
        x_targets: Iterable[float] | None = None,
        side_bias: Dict[str, float] | None = None,
    ) -> Dict[int, float]:
        bridge = cfg.get("bridge", {}) or {}
        span = float(bridge.get("span_mm", 1200.0))
        level = cls.load_level(cfg)
        load_nodes = [
            n for n in nodes
            if getattr(n, "level", "") == level
            and -1.0e-6 <= float(getattr(n, "x", 0.0)) <= span + 1.0e-6
        ]
        if not load_nodes:
            load_nodes = [n for n in nodes if getattr(n, "level", "") == level]
        if not load_nodes:
            return {}

        xs = sorted({round(float(n.x), 6) for n in load_nodes})
        x_weights = cls.station_weights(cfg, x_targets, xs)
        by_x: Dict[float, List[Any]] = defaultdict(list)
        for n in load_nodes:
            by_x[round(float(n.x), 6)].append(n)

        raw_node_weights: Dict[int, float] = defaultdict(float)
        for x_station, wx in x_weights.items():
            station_nodes = by_x.get(round(float(x_station), 6), [])
            if not station_nodes:
                continue
            side_factors = [max(0.0, cls._side_factor(float(n.y), side_bias)) for n in station_nodes]
            side_total = sum(side_factors)
            if side_total <= 1.0e-12:
                side_factors = [1.0 for _ in station_nodes]
                side_total = float(len(station_nodes))
            for n, sy in zip(station_nodes, side_factors):
                raw_node_weights[int(n.id)] += float(wx) * float(sy) / side_total

        total = sum(max(0.0, v) for v in raw_node_weights.values())
        if total <= 1.0e-12:
            return {}
        return {int(k): float(v) / total for k, v in raw_node_weights.items() if v > 1.0e-12}

    @classmethod
    def build_nodal_loads(
        cls,
        cfg: Dict[str, Any],
        nodes: List[Any],
        *,
        loadcase: str,
        total_N: float,
        x_targets: Iterable[float] | None = None,
        side_bias: Dict[str, float] | None = None,
        lateral_factor: float = 0.0,
    ) -> List[Load]:
        weights = cls.nodal_weights(cfg, nodes, x_targets=x_targets, side_bias=side_bias)
        if not weights:
            return []

        node_by_id = {int(n.id): n for n in nodes}
        loads: List[Load] = []
        for nid, w in weights.items():
            n = node_by_id.get(int(nid))
            lateral_sign = -1.0 if n is not None and float(getattr(n, "y", 0.0)) < 0.0 else 1.0
            loads.append(
                Load(
                    str(loadcase),
                    int(nid),
                    0.0,
                    lateral_sign * float(lateral_factor) * abs(float(total_N)) * float(w),
                    -abs(float(total_N)) * float(w),
                )
            )
        return loads
