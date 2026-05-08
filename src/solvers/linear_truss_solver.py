from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np

from src.domain.models import Load, Member, Node, Support
from src.services.geometry_service import GeometryService


@dataclass
class SolverResult:
    node_results: List[Dict]
    member_results: List[Dict]
    active_support_node_ids: Set[int]
    inactive_support_node_ids: Set[int]
    inactive_tension_only_member_ids: Set[int]
    status: str
    iterations: int
    equilibrium_error_N: float
    tension_only_iterations: int
    tension_only_compression_released_N_total: float
    tension_only_converged: bool
    instability_due_to_tension_only_bracing: bool


class LinearTrussSolver:
    """3D axial matrix solver with unilateral supports and optional tension-only release."""

    @staticmethod
    def _dof(node_id: int, comp: int) -> int:
        return 3 * (node_id - 1) + comp

    @staticmethod
    def _direction(ni: Node, nj: Node) -> tuple[float, float, float, float]:
        dx, dy, dz = nj.x - ni.x, nj.y - ni.y, nj.z - ni.z
        L = float((dx * dx + dy * dy + dz * dz) ** 0.5)
        if L <= 0:
            raise ValueError(f"Membro com comprimento zero entre nós {ni.id} e {nj.id}")
        return dx / L, dy / L, dz / L, L

    def solve(
        self,
        nodes: List[Node],
        members: List[Member],
        supports: List[Support],
        loads: List[Load],
        *,
        unilateral_supports: bool = True,
        uplift_tolerance_N: float = 1e-6,
        max_iterations: int = 12,
        tension_only_solver_enabled: bool = False,
        tension_only_groups: Iterable[str] | None = None,
        tension_only_compression_tolerance_N: float = 1e-6,
    ) -> SolverResult:
        active_vertical = {s.node_id for s in supports if s.UZ}
        inactive_vertical: Set[int] = set()

        active_member_ids: Set[int] = {int(m.id) for m in members}
        inactive_tension_only: Set[int] = set()
        removed_order: List[int] = []

        tension_groups = {
            str(g) for g in (
                tension_only_groups
                or ("top_bracing", "bottom_bracing", "cross_frame_bracing", "chord_lacing")
            )
        }

        tension_iterations = 0
        tension_released_total = 0.0
        tension_converged = not bool(tension_only_solver_enabled)
        instability_due_to_tension_only = False
        status = "regular"
        result = None

        for it in range(1, max_iterations + 1):
            result = self._solve_once(
                nodes,
                members,
                supports,
                loads,
                active_vertical,
                active_member_ids,
            )
            node_res, member_res, status = result
            changed = False

            if unilateral_supports:
                negative = {
                    int(r["node_id"])
                    for r in node_res
                    if int(r["node_id"]) in active_vertical and float(r["Rz_N"]) < -abs(uplift_tolerance_N)
                }
                if negative:
                    if len(active_vertical - negative) < 3:
                        status += "_uplift_limited"
                    else:
                        active_vertical -= negative
                        inactive_vertical |= negative
                        changed = True

            if tension_only_solver_enabled:
                tension_iterations += 1
                if status.startswith("singular") and inactive_tension_only:
                    # Fallback conservative: reactivate one previously released brace.
                    restored = False
                    while removed_order:
                        mid = int(removed_order.pop())
                        if mid in inactive_tension_only:
                            inactive_tension_only.remove(mid)
                            active_member_ids.add(mid)
                            restored = True
                            break
                    if restored:
                        instability_due_to_tension_only = True
                        changed = True
                    else:
                        tension_converged = False
                    if not changed:
                        break

                compressed_tension_only: List[tuple[int, float]] = []
                for r in member_res:
                    mid = int(r["member_id"])
                    group = str(r.get("group", ""))
                    if group not in tension_groups:
                        continue
                    n_val = float(r.get("N_N", 0.0))
                    if n_val < -abs(tension_only_compression_tolerance_N):
                        compressed_tension_only.append((mid, n_val))

                for mid, n_val in compressed_tension_only:
                    if mid in inactive_tension_only:
                        continue
                    if mid in active_member_ids:
                        active_member_ids.remove(mid)
                        inactive_tension_only.add(mid)
                        removed_order.append(mid)
                        tension_released_total += abs(float(n_val))
                        changed = True

                if not compressed_tension_only and not changed:
                    tension_converged = True

            if not changed:
                break

        assert result is not None
        node_results, member_results_active, status = result
        member_by_id = {int(m.id): m for m in members}
        for mid in sorted(inactive_tension_only):
            m = member_by_id.get(mid)
            if m is None:
                continue
            member_results_active.append(
                {
                    "member_id": m.id,
                    "i": m.i,
                    "j": m.j,
                    "group": m.group,
                    "n_sticks": m.n_sticks,
                    "L_mm": m.L,
                    "A_mm2": m.A,
                    "Iy_mm4": m.Iy,
                    "Iz_mm4": m.Iz,
                    "Ky": m.Ky,
                    "Kz": m.Kz,
                    "layout": getattr(m, "layout", "stacked"),
                    "N_N": 0.0,
                    "sigma_axial_MPa": 0.0,
                    "state": "inactive_tension_only",
                    "tension_only_released": True,
                    "inactive": True,
                }
            )

        load_sum = sum(l.Fz for l in loads)
        reaction_sum = sum(float(r["Rz_N"]) for r in node_results)
        eq_error = load_sum + reaction_sum

        if tension_only_solver_enabled:
            if tension_converged:
                status = f"{status}|tension_only_converged"
            else:
                status = f"{status}|tension_only_singular"
            if instability_due_to_tension_only:
                status = f"{status}|instability_due_to_tension_only_bracing"

        return SolverResult(
            node_results,
            member_results_active,
            active_vertical,
            inactive_vertical,
            inactive_tension_only,
            status,
            it,
            eq_error,
            tension_iterations,
            tension_released_total,
            tension_converged,
            instability_due_to_tension_only,
        )

    def _solve_once(
        self,
        nodes: List[Node],
        members: List[Member],
        supports: List[Support],
        loads: List[Load],
        active_vertical: Set[int],
        active_member_ids: Set[int],
    ) -> tuple[List[Dict], List[Dict], str]:
        n_dof = 3 * len(nodes)
        K = np.zeros((n_dof, n_dof), dtype=float)
        F = np.zeros(n_dof, dtype=float)
        node_by_id = {n.id: n for n in nodes}

        for m in members:
            if int(m.id) not in active_member_ids:
                continue
            ni, nj = node_by_id[m.i], node_by_id[m.j]
            cx, cy, cz, L = self._direction(ni, nj)
            EA_L = m.E * m.A / L
            c = np.array([cx, cy, cz], dtype=float)
            k3 = EA_L * np.outer(c, c)
            di = [self._dof(m.i, k) for k in range(3)]
            dj = [self._dof(m.j, k) for k in range(3)]
            for a in range(3):
                for b in range(3):
                    K[di[a], di[b]] += k3[a, b]
                    K[di[a], dj[b]] -= k3[a, b]
                    K[dj[a], di[b]] -= k3[a, b]
                    K[dj[a], dj[b]] += k3[a, b]

        for l in loads:
            F[self._dof(l.node_id, 0)] += l.Fx
            F[self._dof(l.node_id, 1)] += l.Fy
            F[self._dof(l.node_id, 2)] += l.Fz

        fixed_dofs = set()
        support_by_node = {s.node_id: s for s in supports}
        for s in supports:
            if s.UX:
                fixed_dofs.add(self._dof(s.node_id, 0))
            if s.UY:
                fixed_dofs.add(self._dof(s.node_id, 1))
            if s.UZ and s.node_id in active_vertical:
                fixed_dofs.add(self._dof(s.node_id, 2))

        all_dofs = np.arange(n_dof)
        free_dofs = np.array([i for i in all_dofs if i not in fixed_dofs], dtype=int)
        Kff = K[np.ix_(free_dofs, free_dofs)]
        Ff = F[free_dofs]
        rank = np.linalg.matrix_rank(Kff)
        U = np.zeros(n_dof, dtype=float)
        if rank == Kff.shape[0]:
            U[free_dofs] = np.linalg.solve(Kff, Ff)
            status = "regular"
        else:
            U[free_dofs] = np.linalg.lstsq(Kff, Ff, rcond=None)[0]
            status = f"singular_lstsq_rank_{rank}_of_{Kff.shape[0]}"
        R = K @ U - F

        node_results = []
        for n in nodes:
            is_support = n.id in support_by_node
            is_active_vertical = n.id in active_vertical
            node_results.append(
                {
                    "node_id": n.id,
                    "x": n.x,
                    "y": n.y,
                    "z": n.z,
                    "Ux_mm": U[self._dof(n.id, 0)],
                    "Uy_mm": U[self._dof(n.id, 1)],
                    "Uz_mm": U[self._dof(n.id, 2)],
                    "Rx_N": R[self._dof(n.id, 0)] if is_support else 0.0,
                    "Ry_N": R[self._dof(n.id, 1)] if is_support else 0.0,
                    "Rz_N": R[self._dof(n.id, 2)] if is_support and is_active_vertical else 0.0,
                    "support_active_vertical": bool(is_active_vertical) if is_support else False,
                }
            )

        member_results = []
        for m in members:
            if int(m.id) not in active_member_ids:
                continue
            ni, nj = node_by_id[m.i], node_by_id[m.j]
            cx, cy, cz, L = self._direction(ni, nj)
            ui = np.array([U[self._dof(m.i, k)] for k in range(3)])
            uj = np.array([U[self._dof(m.j, k)] for k in range(3)])
            ext = float((uj - ui) @ np.array([cx, cy, cz]))
            N = m.E * m.A / L * ext
            member_results.append(
                {
                    "member_id": m.id,
                    "i": m.i,
                    "j": m.j,
                    "group": m.group,
                    "n_sticks": m.n_sticks,
                    "L_mm": m.L,
                    "A_mm2": m.A,
                    "Iy_mm4": m.Iy,
                    "Iz_mm4": m.Iz,
                    "Ky": m.Ky,
                    "Kz": m.Kz,
                    "layout": getattr(m, "layout", "stacked"),
                    "N_N": N,
                    "sigma_axial_MPa": N / m.A if m.A else 0.0,
                    "state": "tension" if N >= 0 else "compression",
                    "tension_only_released": False,
                    "inactive": False,
                }
            )
        return node_results, member_results, status

    def export(self, result: SolverResult, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        GeometryService.write_csv(out / "opensees_truss_nodes.csv", result.node_results)
        GeometryService.write_csv(out / "opensees_truss_members.csv", result.member_results)
        GeometryService.write_csv(
            out / "solver_summary.csv",
            [
                {
                    "status": result.status,
                    "iterations": result.iterations,
                    "equilibrium_error_N": result.equilibrium_error_N,
                    "active_supports": ";".join(map(str, sorted(result.active_support_node_ids))),
                    "inactive_supports_uplift": ";".join(map(str, sorted(result.inactive_support_node_ids))),
                    "inactive_tension_only_members": ";".join(map(str, sorted(result.inactive_tension_only_member_ids))),
                    "tension_only_iterations": result.tension_only_iterations,
                    "tension_only_converged": result.tension_only_converged,
                    "tension_only_compression_released_N_total": result.tension_only_compression_released_N_total,
                    "instability_due_to_tension_only_bracing": result.instability_due_to_tension_only_bracing,
                }
            ],
        )
