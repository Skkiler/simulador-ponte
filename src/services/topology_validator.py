from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Set, Tuple

from src.domain.models import Load, Member, Node, Support


class TopologyValidator:
    """Valida conectividade e integridade topológica básica da treliça."""

    @staticmethod
    def _normalize_truss_mode(mode: str) -> str:
        raw = str(mode or "").strip().lower()
        alias = {
            "parker": "pratt",
            "baltimore": "pratt",
            "pratt_symmetric": "pratt_symmetric",
            "pratt simétrica": "pratt_symmetric",
            "warren_symmetric": "warren_symmetric",
            "warren simétrica": "warren_symmetric",
            "warren_mid_braced": "warren_mid_braced",
            "warren intermediária": "warren_mid_braced",
            "warren intermedia": "warren_mid_braced",
            "howe_inverted": "howe_inverted",
            "howe invertida": "howe_inverted",
            "k_symmetric": "k_symmetric",
            "k simétrica": "k_symmetric",
            "duplo_x": "x",
            "double_x": "x",
            "sem": "none",
            "nenhuma": "none",
        }
        return alias.get(raw, raw)

    @staticmethod
    def _incident_groups(members: Iterable[Member]) -> Dict[int, Set[str]]:
        out: Dict[int, Set[str]] = {}
        for m in members:
            out.setdefault(int(m.i), set()).add(str(m.group))
            out.setdefault(int(m.j), set()).add(str(m.group))
        return out

    @staticmethod
    def _member_adjacency(members: Iterable[Member]) -> Dict[int, Set[int]]:
        graph: Dict[int, Set[int]] = {}
        for m in members:
            i = int(m.i)
            j = int(m.j)
            graph.setdefault(i, set()).add(j)
            graph.setdefault(j, set()).add(i)
        return graph

    @staticmethod
    def _main_component_nodes(graph: Dict[int, Set[int]]) -> Set[int]:
        if not graph:
            return set()
        start = next(iter(graph))
        vis: Set[int] = set([start])
        q: deque[int] = deque([start])
        while q:
            u = q.popleft()
            for v in graph.get(u, set()):
                if v in vis:
                    continue
                vis.add(v)
                q.append(v)
        return vis

    @classmethod
    def _check_warren_endpoint_diagonals(
        cls,
        cfg: Dict,
        nodes: List[Node],
        members: List[Member],
    ) -> List[str]:
        bridge = cfg.get("bridge", {}) or {}
        side_mode = cls._normalize_truss_mode(
            bridge.get("side_truss_type", bridge.get("truss_type", ""))
        )
        if side_mode not in {"warren", "warren_symmetric"}:
            return []

        span = float(bridge.get("span_mm", 0.0))
        by_node = cls._incident_groups(members)
        node_id_by_key = {
            (round(float(n.x), 6), round(float(n.y), 6), str(n.level)): int(n.id)
            for n in nodes
        }
        y_values = sorted(
            {
                float(n.y)
                for n in nodes
                if n.level == "bottom" and abs(float(n.x)) <= 1.0e-6
            }
        )
        errors: List[str] = []

        for x in (0.0, span):
            for y in y_values:
                for level in ("bottom", "top"):
                    key = (round(float(x), 6), round(float(y), 6), level)
                    node_id = node_id_by_key.get(key)
                    if node_id is None:
                        errors.append(
                            f"warren_endpoint_node_missing:x={x:.3f},y={y:.3f},level={level}"
                        )
                        continue
                    if "diagonal" not in by_node.get(node_id, set()):
                        errors.append(
                            f"warren_endpoint_without_diagonal:x={x:.3f},y={y:.3f},level={level}"
                        )
        return errors

    @classmethod
    def _check_warren_panels_closed(
        cls,
        cfg: Dict,
        nodes: List[Node],
        members: List[Member],
    ) -> List[str]:
        bridge = cfg.get("bridge", {}) or {}
        side_mode = cls._normalize_truss_mode(
            bridge.get("side_truss_type", bridge.get("truss_type", ""))
        )
        if side_mode not in {"warren", "warren_symmetric", "warren_mid_braced"}:
            return []

        span = float(bridge.get("span_mm", 0.0))
        xs = sorted(
            {
                round(float(n.x), 6)
                for n in nodes
                if n.level == "bottom" and -1.0e-6 <= float(n.x) <= span + 1.0e-6
            }
        )
        ys = sorted(
            {
                round(float(n.y), 6)
                for n in nodes
                if n.level == "bottom" and abs(float(n.x)) <= 1.0e-6
            }
        )
        if len(xs) < 2 or not ys:
            return []

        node_id_by_key = {
            (round(float(n.x), 6), round(float(n.y), 6), str(n.level)): int(n.id)
            for n in nodes
        }
        edges = set()
        for m in members:
            if str(m.group) != "diagonal":
                continue
            a = min(int(m.i), int(m.j))
            b = max(int(m.i), int(m.j))
            edges.add((a, b))

        errors: List[str] = []
        for y in ys:
            for x0, x1 in zip(xs[:-1], xs[1:]):
                k00 = (round(float(x0), 6), y, "bottom")
                k01 = (round(float(x0), 6), y, "top")
                k10 = (round(float(x1), 6), y, "bottom")
                k11 = (round(float(x1), 6), y, "top")
                n00 = node_id_by_key.get(k00)
                n01 = node_id_by_key.get(k01)
                n10 = node_id_by_key.get(k10)
                n11 = node_id_by_key.get(k11)
                if None in {n00, n01, n10, n11}:
                    continue
                d1 = (min(n00, n11), max(n00, n11)) in edges
                d2 = (min(n01, n10), max(n01, n10)) in edges
                if not (d1 or d2):
                    errors.append(
                        f"warren_open_panel:x0={x0:.3f},x1={x1:.3f},y={y:.3f}"
                    )
        return errors

    @staticmethod
    def _check_transverses_attached(
        graph: Dict[int, Set[int]],
        members: Iterable[Member],
    ) -> List[str]:
        errors: List[str] = []
        for m in members:
            if str(m.group) not in {"top_transverse", "bottom_transverse"}:
                continue
            di = len(graph.get(int(m.i), set()))
            dj = len(graph.get(int(m.j), set()))
            if di <= 1 or dj <= 1:
                errors.append(f"transverse_not_attached:member={int(m.id)}")
        return errors

    def validate(
        self,
        cfg: Dict,
        nodes: List[Node],
        members: List[Member],
        supports: List[Support],
        loads: List[Load],
        *,
        solver: Any | None = None,
    ) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []

        node_ids = {int(n.id) for n in nodes}
        if not members:
            errors.append("no_active_members")

        for m in members:
            if int(m.i) not in node_ids or int(m.j) not in node_ids:
                errors.append(f"member_with_missing_node:{int(m.id)}")

        graph = self._member_adjacency(members)
        used_nodes = set(graph.keys())
        if not used_nodes:
            errors.append("no_structural_nodes")
        else:
            main_component = self._main_component_nodes(graph)
            if used_nodes - main_component:
                errors.append("disconnected_components")

            for s in supports:
                nid = int(s.node_id)
                if nid not in main_component:
                    errors.append(f"support_outside_main_component:{nid}")
            for l in loads:
                nid = int(l.node_id)
                if nid not in main_component:
                    errors.append(f"load_outside_main_component:{nid}")

            # nós estruturais internos com grau muito baixo
            node_by_id = {int(n.id): n for n in nodes}
            span = float(cfg.get("bridge", {}).get("span_mm", 0.0))
            tol = 1.0e-6
            for nid in used_nodes:
                n = node_by_id.get(nid)
                if n is None:
                    continue
                deg = len(graph.get(nid, set()))
                on_support_line = abs(float(n.x)) <= tol or abs(float(n.x) - span) <= tol
                if deg <= 1 and not on_support_line:
                    errors.append(f"floating_or_weak_node:{nid}")

        errors.extend(self._check_warren_endpoint_diagonals(cfg, nodes, members))
        errors.extend(self._check_warren_panels_closed(cfg, nodes, members))
        errors.extend(self._check_transverses_attached(graph, members))

        if solver is not None and not errors:
            try:
                sol = solver.solve(
                    nodes,
                    members,
                    supports,
                    loads,
                    unilateral_supports=bool(cfg.get("bridge", {}).get("unilateral_supports", True)),
                    tension_only_solver_enabled=bool(
                        cfg.get("bridge", {}).get("tension_only_bracing_solver_enabled", False)
                    ),
                    tension_only_groups=cfg.get("analysis", {}).get(
                        "tension_only_groups",
                        ["top_bracing", "bottom_bracing", "cross_frame_bracing", "chord_lacing"],
                    ),
                    tension_only_compression_tolerance_N=float(
                        cfg.get("analysis", {}).get("tension_only_compression_tolerance_N", 1.0e-6)
                    ),
                )
                if str(sol.status).startswith("singular"):
                    errors.append("solver_singular_after_topology_check")
            except (TypeError, ValueError, RuntimeError) as exc:
                warnings.append(f"topology_solver_check_failed:{exc!r}")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


__all__ = ["TopologyValidator"]
