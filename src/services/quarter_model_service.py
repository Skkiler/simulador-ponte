from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.core.numeric import safe_float
from src.domain.models import Load, Member, Node, Support
from src.solvers.linear_truss_solver import SolverResult


@dataclass
class QuarterModel:
    nodes: List[Node]
    members: List[Member]
    supports: List[Support]
    loads: List[Load]
    source_member_by_quarter_member: Dict[int, int]
    mirror_maps: Dict[str, Any]


@dataclass
class FullModelResult:
    nodes: List[Node]
    members: List[Member]
    supports: List[Support]
    loads: List[Load]
    result: SolverResult
    quarter_member_count: int
    mirror_maps: Dict[str, Any]
    used_quarter_model: bool
    fallback_reason: str | None = None


class QuarterModelService:
    def __init__(self) -> None:
        self._quarter_to_full_members: Dict[int, List[int]] = {}
        self._full_to_quarter_member: Dict[int, int] = {}

    @staticmethod
    def is_quarter_model_enabled(cfg: Dict) -> bool:
        a = cfg.get("analysis", {}) or {}
        if "use_quarter_model" in a:
            return bool(a.get("use_quarter_model"))
        return bool(a.get("enforce_symmetry", True))

    @staticmethod
    def _is_on_x_plane(x: float, span: float, tol: float = 1e-6) -> bool:
        return abs(float(x) - float(span) / 2.0) <= tol

    @staticmethod
    def _is_on_y_plane(y: float, tol: float = 1e-6) -> bool:
        return abs(float(y)) <= tol

    def symmetry_multiplicity(self, point: Tuple[float, float], span: float) -> int:
        x, y = point
        mx = 1 if self._is_on_x_plane(x, span) else 2
        my = 1 if self._is_on_y_plane(y) else 2
        return mx * my

    def validate_quarter_symmetry(
        self,
        cfg: Dict,
        nodes: List[Node],
        members: List[Member],
        supports: List[Support],
        loads: List[Load],
    ) -> Dict[str, Any]:
        span = float(cfg.get("bridge", {}).get("span_mm", 0.0))
        if span <= 0.0:
            return {"is_valid": False, "reasons": ["span_invalid"], "warnings": []}

        ys = [float(n.y) for n in nodes]
        min_y = min(ys) if ys else 0.0
        max_y = max(ys) if ys else 0.0
        reasons: List[str] = []
        warnings: List[str] = []
        if abs(abs(min_y) - abs(max_y)) > 1.0e-6:
            reasons.append("y_extents_not_symmetric")

        has_left = any(float(n.x) < span / 2.0 - 1e-6 for n in nodes)
        has_right = any(float(n.x) > span / 2.0 + 1e-6 for n in nodes)
        if not (has_left and has_right):
            reasons.append("missing_nodes_on_both_sides_of_midspan")

        if not members:
            reasons.append("no_members")
        if not supports:
            reasons.append("no_supports")
        if not loads:
            warnings.append("no_loads")

        return {
            "is_valid": len(reasons) == 0,
            "reasons": reasons,
            "warnings": warnings,
        }

    @staticmethod
    def _lerp(na: Node, nb: Node, t: float) -> tuple[float, float, float]:
        return (
            float(na.x) + t * (float(nb.x) - float(na.x)),
            float(na.y) + t * (float(nb.y) - float(na.y)),
            float(na.z) + t * (float(nb.z) - float(na.z)),
        )

    def _split_member_in_quarter(
        self,
        member: Member,
        node_by_id: Dict[int, Node],
        span: float,
        *,
        tol: float = 1e-6,
    ) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        mid_x = span / 2.0
        ni = node_by_id[member.i]
        nj = node_by_id[member.j]
        ts = [0.0, 1.0]

        dx = float(nj.x) - float(ni.x)
        dy = float(nj.y) - float(ni.y)
        if abs(dx) > tol:
            tx = (mid_x - float(ni.x)) / dx
            if tol < tx < 1.0 - tol:
                ts.append(tx)
        if abs(dy) > tol:
            ty = (0.0 - float(ni.y)) / dy
            if tol < ty < 1.0 - tol:
                ts.append(ty)
        ts = sorted(set(round(t, 12) for t in ts))

        out: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
        for t0, t1 in zip(ts[:-1], ts[1:]):
            p0 = self._lerp(ni, nj, t0)
            p1 = self._lerp(ni, nj, t1)
            pm = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5, (p0[2] + p1[2]) * 0.5)
            if pm[0] <= mid_x + tol and pm[1] >= -tol:
                out.append((p0, p1))
        return out

    def build_quarter_model(
        self,
        cfg: Dict,
        nodes: List[Node],
        members: List[Member],
        supports: List[Support],
        loads: List[Load],
    ) -> QuarterModel:
        span = float(cfg["bridge"]["span_mm"])
        mid_x = span / 2.0
        node_by_id = {n.id: n for n in nodes}

        q_nodes: List[Node] = []
        q_node_by_key: Dict[tuple[float, float, float, str], int] = {}

        def add_q_node(x: float, y: float, z: float, level: str) -> int:
            key = (round(x, 6), round(y, 6), round(z, 6), str(level))
            if key in q_node_by_key:
                return q_node_by_key[key]
            nid = len(q_nodes) + 1
            q_nodes.append(Node(nid, float(x), float(y), float(z), str(level), "R" if y >= 0 else "L", float(x)))
            q_node_by_key[key] = nid
            return nid

        q_members: List[Member] = []
        source_member_by_quarter_member: Dict[int, int] = {}
        for m in members:
            segments = self._split_member_in_quarter(m, node_by_id, span)
            for p0, p1 in segments:
                ni = node_by_id[m.i]
                i_new = add_q_node(p0[0], p0[1], p0[2], ni.level)
                j_new = add_q_node(p1[0], p1[1], p1[2], ni.level)
                if i_new == j_new:
                    continue
                qid = len(q_members) + 1
                L = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2) ** 0.5
                q_members.append(
                    Member(
                        qid,
                        i_new,
                        j_new,
                        m.group,
                        m.n_sticks,
                        m.A,
                        m.Asy,
                        m.Asz,
                        m.Iy,
                        m.Iz,
                        m.J,
                        m.E,
                        m.G,
                        m.Ky,
                        m.Kz,
                        L,
                        getattr(m, "layout", "stacked"),
                        getattr(m, "stick_orientation", "flat"),
                    )
                )
                source_member_by_quarter_member[qid] = int(m.id)

        q_supports_map: Dict[int, Support] = {}
        for s in supports:
            n = node_by_id.get(int(s.node_id))
            if n is None:
                continue
            xq = float(n.x) if float(n.x) <= mid_x else 2.0 * mid_x - float(n.x)
            yq = abs(float(n.y))
            nid = add_q_node(xq, yq, float(n.z), n.level)
            prev = q_supports_map.get(nid)
            if prev is None:
                q_supports_map[nid] = Support(nid, s.UX, s.UY, s.UZ, s.RX, s.RY, s.RZ, s.support_group, s.active_vertical)
            else:
                q_supports_map[nid] = Support(
                    nid,
                    max(prev.UX, s.UX),
                    max(prev.UY, s.UY),
                    max(prev.UZ, s.UZ),
                    max(prev.RX, s.RX),
                    max(prev.RY, s.RY),
                    max(prev.RZ, s.RZ),
                    prev.support_group,
                    prev.active_vertical or s.active_vertical,
                )
        q_supports = list(q_supports_map.values())
        q_supports = self.apply_symmetry_boundary_conditions(cfg, q_nodes, q_supports)

        q_loads_acc: Dict[Tuple[str, int], List[float]] = {}
        for ld in loads:
            n = node_by_id.get(int(ld.node_id))
            if n is None:
                continue
            xq = float(n.x) if float(n.x) <= mid_x else 2.0 * mid_x - float(n.x)
            yq = abs(float(n.y))
            nid = add_q_node(xq, yq, float(n.z), n.level)
            qld = self.distribute_load_to_quarter(ld, (float(n.x), float(n.y), float(n.z)), cfg)
            key = (str(ld.loadcase), nid)
            if key not in q_loads_acc:
                q_loads_acc[key] = [0.0] * 6
            q_loads_acc[key][0] += qld.Fx
            q_loads_acc[key][1] += qld.Fy
            q_loads_acc[key][2] += qld.Fz
            q_loads_acc[key][3] += qld.Mx
            q_loads_acc[key][4] += qld.My
            q_loads_acc[key][5] += qld.Mz
        q_loads: List[Load] = []
        for (lc, nid), vals in q_loads_acc.items():
            q_loads.append(Load(lc, nid, vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]))

        mirror_maps = {
            "quarter_node_to_full": {},
            "quarter_member_to_full": {},
            "full_member_to_quarter": {},
            "node_replications": [],
            "member_replications": [],
        }
        return QuarterModel(q_nodes, q_members, q_supports, q_loads, source_member_by_quarter_member, mirror_maps)

    def apply_symmetry_boundary_conditions(
        self,
        cfg: Dict,
        quarter_nodes: List[Node],
        quarter_supports: List[Support],
    ) -> List[Support]:
        span = float(cfg["bridge"]["span_mm"])
        mid_x = span / 2.0
        out = {int(s.node_id): s for s in quarter_supports}
        for n in quarter_nodes:
            ux = 0
            uy = 0
            group = None
            if abs(float(n.x) - mid_x) <= 1e-6:
                ux = 1
                group = "symmetry_x"
            if abs(float(n.y)) <= 1e-6:
                uy = 1
                group = "symmetry_y" if group is None else "symmetry_xy"
            if ux == 0 and uy == 0:
                continue
            prev = out.get(n.id)
            if prev is None:
                out[n.id] = Support(n.id, ux, uy, 0, 0, 0, 0, str(group), True)
            else:
                out[n.id] = Support(
                    n.id,
                    max(prev.UX, ux),
                    max(prev.UY, uy),
                    prev.UZ,
                    prev.RX,
                    prev.RY,
                    prev.RZ,
                    prev.support_group if prev.support_group else str(group),
                    prev.active_vertical,
                )
        return list(out.values())

    def distribute_load_to_quarter(
        self,
        load: Load,
        node_position: Tuple[float, float, float],
        cfg: Dict,
    ) -> Load:
        span = float(cfg["bridge"]["span_mm"])
        x, y, _ = node_position
        m = self.symmetry_multiplicity((x, y), span)
        f = 1.0 / float(max(1, m))
        return Load(
            load.loadcase,
            load.node_id,
            load.Fx * f,
            load.Fy * f,
            load.Fz * f,
            load.Mx * f,
            load.My * f,
            load.Mz * f,
        )

    def replicate_quarter_geometry(
        self,
        cfg: Dict,
        quarter_nodes: List[Node],
        quarter_members: List[Member],
        quarter_supports: List[Support],
        quarter_loads: List[Load],
    ) -> Dict[str, Any]:
        span = float(cfg["bridge"]["span_mm"])
        mid_x = span / 2.0
        node_by_id = {n.id: n for n in quarter_nodes}

        full_nodes: List[Node] = []
        node_key_to_id: Dict[Tuple[float, float, float, str], int] = {}
        quarter_node_to_full: Dict[int, List[int]] = {}
        node_replications: List[Dict[str, Any]] = []

        def add_node(x: float, y: float, z: float, level: str) -> int:
            key = (round(x, 6), round(y, 6), round(z, 6), str(level))
            if key in node_key_to_id:
                return node_key_to_id[key]
            nid = len(full_nodes) + 1
            full_nodes.append(Node(nid, x, y, z, level, "R" if y >= 0 else "L", x))
            node_key_to_id[key] = nid
            return nid

        mirror_ops = [(False, False), (True, False), (False, True), (True, True)]
        node_map_by_op: Dict[Tuple[bool, bool], Dict[int, int]] = {(mx, my): {} for mx, my in mirror_ops}
        for qn in quarter_nodes:
            full_ids: List[int] = []
            for mx, my in mirror_ops:
                x = float(qn.x) if not mx else 2.0 * mid_x - float(qn.x)
                y = float(qn.y) if not my else -float(qn.y)
                nid = add_node(x, y, float(qn.z), qn.level)
                node_map_by_op[(mx, my)][qn.id] = nid
                if nid not in full_ids:
                    full_ids.append(nid)
                node_replications.append(
                    {
                        "quarter_node_id": qn.id,
                        "full_node_id": nid,
                        "mirror_x": mx,
                        "mirror_y": my,
                    }
                )
            quarter_node_to_full[qn.id] = sorted(full_ids)

        full_members: List[Member] = []
        member_key_to_id: Dict[Tuple[int, int, str], int] = {}
        quarter_member_to_full: Dict[int, List[int]] = {}
        full_member_to_quarter: Dict[int, int] = {}
        member_replications: List[Dict[str, Any]] = []
        for qm in quarter_members:
            ids: List[int] = []
            for mx, my in mirror_ops:
                i = node_map_by_op[(mx, my)][qm.i]
                j = node_map_by_op[(mx, my)][qm.j]
                a, b = (i, j) if i <= j else (j, i)
                key = (a, b, qm.group)
                if key in member_key_to_id:
                    mid = member_key_to_id[key]
                else:
                    n1 = full_nodes[a - 1]
                    n2 = full_nodes[b - 1]
                    L = ((n2.x - n1.x) ** 2 + (n2.y - n1.y) ** 2 + (n2.z - n1.z) ** 2) ** 0.5
                    mid = len(full_members) + 1
                    full_members.append(
                        Member(
                            mid,
                            a,
                            b,
                            qm.group,
                            qm.n_sticks,
                            qm.A,
                            qm.Asy,
                            qm.Asz,
                            qm.Iy,
                            qm.Iz,
                            qm.J,
                            qm.E,
                            qm.G,
                            qm.Ky,
                            qm.Kz,
                            L,
                            getattr(qm, "layout", "stacked"),
                            getattr(qm, "stick_orientation", "flat"),
                        )
                    )
                    member_key_to_id[key] = mid
                    full_member_to_quarter[mid] = qm.id
                if mid not in ids:
                    ids.append(mid)
                member_replications.append(
                    {
                        "quarter_member_id": qm.id,
                        "full_member_id": mid,
                        "mirror_x": mx,
                        "mirror_y": my,
                    }
                )
            quarter_member_to_full[qm.id] = sorted(ids)

        full_supports: List[Support] = []
        support_seen = set()
        for qs in quarter_supports:
            if str(qs.support_group).startswith("symmetry"):
                continue
            for mx, my in mirror_ops:
                nid = node_map_by_op[(mx, my)].get(qs.node_id)
                if nid is None:
                    continue
                key = (nid, qs.UX, qs.UY, qs.UZ, qs.RX, qs.RY, qs.RZ, qs.support_group)
                if key in support_seen:
                    continue
                support_seen.add(key)
                full_supports.append(
                    Support(
                        nid,
                        qs.UX,
                        qs.UY,
                        qs.UZ,
                        qs.RX,
                        qs.RY,
                        qs.RZ,
                        qs.support_group,
                        qs.active_vertical,
                    )
                )

        full_loads_acc: Dict[Tuple[str, int], List[float]] = {}
        for ql in quarter_loads:
            target_nodes = set()
            for mx, my in mirror_ops:
                nid = node_map_by_op[(mx, my)].get(ql.node_id)
                if nid is not None:
                    target_nodes.add(int(nid))
            if not target_nodes:
                continue
            for nid in sorted(target_nodes):
                key = (str(ql.loadcase), int(nid))
                if key not in full_loads_acc:
                    full_loads_acc[key] = [0.0] * 6
                full_loads_acc[key][0] += float(ql.Fx)
                full_loads_acc[key][1] += float(ql.Fy)
                full_loads_acc[key][2] += float(ql.Fz)
                full_loads_acc[key][3] += float(ql.Mx)
                full_loads_acc[key][4] += float(ql.My)
                full_loads_acc[key][5] += float(ql.Mz)
        full_loads: List[Load] = [
            Load(lc, nid, vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])
            for (lc, nid), vals in sorted(full_loads_acc.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        ]

        self._quarter_to_full_members = quarter_member_to_full
        self._full_to_quarter_member = full_member_to_quarter
        return {
            "nodes": full_nodes,
            "members": full_members,
            "supports": full_supports,
            "loads": full_loads,
            "mirror_maps": {
                "quarter_node_to_full": {str(k): v for k, v in quarter_node_to_full.items()},
                "quarter_member_to_full": {str(k): v for k, v in quarter_member_to_full.items()},
                "full_member_to_quarter": {str(k): v for k, v in full_member_to_quarter.items()},
                "node_replications": node_replications,
                "member_replications": member_replications,
            },
        }

    def replicate_quarter_results(
        self,
        cfg: Dict,
        quarter_result: SolverResult,
        mirror_maps: Dict[str, Any],
    ) -> Dict[str, Any]:
        span = float(cfg["bridge"]["span_mm"])
        mid_x = span / 2.0

        node_reps = mirror_maps.get("node_replications", []) or []
        member_reps = mirror_maps.get("member_replications", []) or []
        qn_res_by_id = {int(r["node_id"]): r for r in quarter_result.node_results}
        qm_res_by_id = {int(r["member_id"]): r for r in quarter_result.member_results}

        full_node_results: Dict[int, Dict[str, Any]] = {}
        for rep in node_reps:
            qid = int(rep["quarter_node_id"])
            fid = int(rep["full_node_id"])
            mx = bool(rep.get("mirror_x"))
            my = bool(rep.get("mirror_y"))
            base = dict(qn_res_by_id.get(qid, {}))
            if not base:
                continue
            ux = safe_float(base.get("Ux_mm"), 0.0) or 0.0
            uy = safe_float(base.get("Uy_mm"), 0.0) or 0.0
            uz = safe_float(base.get("Uz_mm"), 0.0) or 0.0
            if mx:
                ux = -ux
            if my:
                uy = -uy
            x = safe_float(base.get("x"), 0.0) or 0.0
            y = safe_float(base.get("y"), 0.0) or 0.0
            z = safe_float(base.get("z"), 0.0) or 0.0
            if mx:
                x = 2.0 * mid_x - x
            if my:
                y = -y
            nr = dict(base)
            nr["node_id"] = fid
            nr["x"] = x
            nr["y"] = y
            nr["z"] = z
            nr["Ux_mm"] = ux
            nr["Uy_mm"] = uy
            nr["Uz_mm"] = uz
            full_node_results[fid] = nr

        full_member_results: Dict[int, Dict[str, Any]] = {}
        for rep in member_reps:
            qmid = int(rep["quarter_member_id"])
            fmid = int(rep["full_member_id"])
            base = dict(qm_res_by_id.get(qmid, {}))
            if not base:
                continue
            mr = dict(base)
            mr["member_id"] = fmid
            full_member_results[fmid] = mr

        active_nodes = set()
        inactive_nodes = set()
        inactive_tension_only_members = set()
        q_active = set(int(v) for v in quarter_result.active_support_node_ids)
        q_inactive = set(int(v) for v in quarter_result.inactive_support_node_ids)
        q_tension_inactive = set(
            int(v) for v in getattr(quarter_result, "inactive_tension_only_member_ids", set())
        )
        q2f = mirror_maps.get("quarter_node_to_full", {}) or {}
        q2f_member = mirror_maps.get("quarter_member_to_full", {}) or {}
        for qn_str, full_ids in q2f.items():
            qn = int(qn_str)
            if qn in q_active:
                active_nodes.update(int(fid) for fid in full_ids)
            if qn in q_inactive:
                inactive_nodes.update(int(fid) for fid in full_ids)
        for qm_str, full_ids in q2f_member.items():
            qm = int(qm_str)
            if qm in q_tension_inactive:
                inactive_tension_only_members.update(int(fid) for fid in full_ids)

        return {
            "node_results": [full_node_results[k] for k in sorted(full_node_results)],
            "member_results": [full_member_results[k] for k in sorted(full_member_results)],
            "active_support_node_ids": active_nodes,
            "inactive_support_node_ids": inactive_nodes,
            "inactive_tension_only_member_ids": inactive_tension_only_members,
            "status": quarter_result.status,
            "iterations": quarter_result.iterations,
            "equilibrium_error_N": (safe_float(quarter_result.equilibrium_error_N, 0.0) or 0.0) * 4.0,
            "tension_only_iterations": int(getattr(quarter_result, "tension_only_iterations", 0) or 0),
            "tension_only_compression_released_N_total": (
                safe_float(
                    getattr(quarter_result, "tension_only_compression_released_N_total", 0.0),
                    0.0,
                )
                or 0.0
            )
            * 4.0,
            "tension_only_converged": bool(getattr(quarter_result, "tension_only_converged", True)),
            "instability_due_to_tension_only_bracing": bool(
                getattr(quarter_result, "instability_due_to_tension_only_bracing", False)
            ),
        }

    def solve_quarter_and_replicate(
        self,
        cfg: Dict,
        solver: Any,
        quarter_model: QuarterModel,
    ) -> FullModelResult:
        qres = solver.solve(
            quarter_model.nodes,
            quarter_model.members,
            quarter_model.supports,
            quarter_model.loads,
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
        geom = self.replicate_quarter_geometry(
            cfg,
            quarter_model.nodes,
            quarter_model.members,
            quarter_model.supports,
            quarter_model.loads,
        )
        rep = self.replicate_quarter_results(cfg, qres, geom["mirror_maps"])
        fres = SolverResult(
            rep["node_results"],
            rep["member_results"],
            rep["active_support_node_ids"],
            rep["inactive_support_node_ids"],
            rep.get("inactive_tension_only_member_ids", set()),
            rep["status"],
            rep["iterations"],
            rep["equilibrium_error_N"],
            int(rep.get("tension_only_iterations", 0) or 0),
            float(rep.get("tension_only_compression_released_N_total", 0.0) or 0.0),
            bool(rep.get("tension_only_converged", True)),
            bool(rep.get("instability_due_to_tension_only_bracing", False)),
        )
        return FullModelResult(
            geom["nodes"],
            geom["members"],
            geom["supports"],
            geom["loads"],
            fres,
            quarter_member_count=len(quarter_model.members),
            mirror_maps=geom["mirror_maps"],
            used_quarter_model=True,
        )

    def map_quarter_member_to_full_members(self, member_id: int) -> List[int]:
        return list(self._quarter_to_full_members.get(int(member_id), []))

    def map_full_member_to_quarter_member(self, member_id: int) -> int:
        return int(self._full_to_quarter_member.get(int(member_id), -1))
