from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from src.domain.models import Load, Member, Node, Support

class Frame3DDAdapter:
    """Exporta e, quando possível, executa Frame3DD. Sem acoplar a UI."""
    def find_executable(self, configured_path: str | None = None) -> Optional[Path]:
        if configured_path and configured_path not in {"", "auto"}:
            p = Path(configured_path)
            if p.exists(): return p
        # Prefer the native binary for the current OS.  The previous order picked
        # the Windows executable first; on Linux this can fail with Exec format
        # error or rely on Wine/binfmt side effects.
        import os
        import platform

        system = platform.system().lower()
        if system.startswith("win"):
            candidates = [
                Path("Frame3DD/windows/frame3dd.exe"),
                Path("frame3dd.exe"),
                Path("Frame3DD/linux/frame3dd"),
                Path("Frame3DD/osx/frame3dd"),
                Path("frame3dd"),
            ]
        elif system == "darwin":
            candidates = [
                Path("Frame3DD/osx/frame3dd"),
                Path("frame3dd"),
                Path("Frame3DD/linux/frame3dd"),
                Path("Frame3DD/windows/frame3dd.exe"),
            ]
        else:
            candidates = [
                Path("Frame3DD/linux/frame3dd"),
                Path("frame3dd"),
                Path("Frame3DD/osx/frame3dd"),
                Path("Frame3DD/windows/frame3dd.exe"),
            ]

        for p in candidates:
            if not p.exists():
                continue
            if str(p).endswith(".exe") and not system.startswith("win"):
                continue
            if not system.startswith("win") and not os.access(p, os.X_OK):
                try:
                    p.chmod(p.stat().st_mode | 0o111)
                except OSError:
                    continue
            return p
        found = shutil.which("frame3dd") or (shutil.which("frame3dd.exe") if system.startswith("win") else None)
        return Path(found) if found else None
    def _density(self,cfg:Dict)->float:
        gmm3=float(cfg.get('material',{}).get('density_g_per_mm3',1e-6))
        return max(gmm3*1e-6,1e-18)  # tonne/mm3

    def _stabilized_reactions(self, cfg: Dict, nodes: List[Node], active: List[Support]) -> List[tuple[int, int, int, int, int, int, int]]:
        """Return a numerically stable support set for Frame3DD.

        The truss solver keeps the competition boundary condition: vertical
        reactions on the table-contact nodes, with unilateral uplift handling.
        Frame3DD, however, also needs the six global rigid-body modes removed.
        The previous exporter restrained only UZ and fixed rotations at every
        contact node, leaving UX/UY rigid translations free; Frame3DD therefore
        aborted with a non positive-definite stiffness matrix.

        This method preserves the vertical support pattern but adds the minimum
        in-plane restraints normally used for a 3D simply-supported validation
        model: one pinned node (UX, UY, UZ) and one laterally guided roller
        (UY, UZ) on the opposite support line.  Other contact nodes keep UZ
        only.  Rotational restraints are free by default because the bridge is
        supported on tables, not clamped.
        """
        if not active:
            return []

        analysis = cfg.get("analysis", {}) or {}
        stabilize = bool(analysis.get("frame3dd_stabilize_rigid_body_modes", True))
        fix_rot = int(bool(analysis.get("frame3dd_fix_support_rotations", False)))
        node_by_id = {int(n.id): n for n in nodes}

        if not stabilize:
            rows = []
            for s in active:
                rows.append((int(s.node_id), int(s.UX), int(s.UY), int(s.UZ), int(s.RX), int(s.RY), int(s.RZ)))
            return rows

        active_sorted = sorted(
            active,
            key=lambda sp: (
                float(getattr(node_by_id.get(int(sp.node_id)), "x", 0.0)),
                float(getattr(node_by_id.get(int(sp.node_id)), "y", 0.0)),
                int(sp.node_id),
            ),
        )
        pin = active_sorted[0]
        pin_node = node_by_id.get(int(pin.node_id))
        pin_x = float(getattr(pin_node, "x", 0.0))
        pin_y = float(getattr(pin_node, "y", 0.0))

        # Pick a second node as far as possible from the pin in plan view.
        guide = max(
            active_sorted,
            key=lambda sp: (
                (float(getattr(node_by_id.get(int(sp.node_id)), "x", 0.0)) - pin_x) ** 2
                + (float(getattr(node_by_id.get(int(sp.node_id)), "y", 0.0)) - pin_y) ** 2,
                int(sp.node_id),
            ),
        )

        rows = []
        for s in active_sorted:
            nid = int(s.node_id)
            ux = 1 if nid == int(pin.node_id) else 0
            uy = 1 if nid in {int(pin.node_id), int(guide.node_id)} else 0
            uz = 1
            rows.append((nid, ux, uy, uz, fix_rot, fix_rot, fix_rot))
        return rows
    def write_input(self,cfg:Dict,nodes:List[Node],members:List[Member],supports:List[Support],loads:List[Load],out_path:str|Path)->Path:
        out=Path(out_path); out.parent.mkdir(parents=True,exist_ok=True); lines=[]
        lines.append("Ponte de palitos - Frame3DD linear estabilizado (N mm tonne)")
        lines.append("# Frame3DD é validação linear; flambagem/ruptura são pós-processadas.")
        lines.append(f"{len(nodes)} # number of nodes")
        lines.append("# node x y z rj")
        for n in nodes: lines.append(f"{n.id:5d} {n.x:12.6f} {n.y:12.6f} {n.z:12.6f} {0.0:12.6f}")
        active=[s for s in supports if s.active_vertical]
        frame_reactions = self._stabilized_reactions(cfg, nodes, active)
        lines.append(f"{len(frame_reactions)} # number of nodes with reactions")
        lines.append("# node x y z xx yy zz")
        for node_id, ux, uy, uz, rx, ry, rz in frame_reactions:
            lines.append(f"{node_id:5d} {ux:d} {uy:d} {uz:d} {rx:d} {ry:d} {rz:d}")
        dens=self._density(cfg)
        # Frame3DD exige que os números dos elementos fiquem dentro do intervalo
        # 1..nE. Os IDs internos do modelo podem ter lacunas depois de remoções
        # topológicas; por isso exportamos uma numeração local contígua.
        frame_members = list(members)
        lines.append(f"{len(frame_members)} # number of frame elements")
        lines.append("# e n1 n2 Ax Asy Asz Jxx Iyy Izz E G roll density")
        for eid, m in enumerate(frame_members, start=1):
            lines.append(f"{eid:5d} {m.i:5d} {m.j:5d} {m.A:12.6f} {m.Asy:12.6f} {m.Asz:12.6f} {m.J:12.6f} {m.Iy:12.6f} {m.Iz:12.6f} {m.E:12.6f} {m.G:12.6f} {0.0:8.3f} {dens:12.6e}")
        include_shear = int(float(cfg.get("analysis", {}).get("frame3dd_include_shear_deformation", 1)))
        include_geo = int(float(cfg.get("analysis", {}).get("frame3dd_include_geometric_stiffness", 0)))
        lines += [
            f"{include_shear} # include shear deformation",
            f"{include_geo} # include geometric stiffness",
            "20.0 # exaggerate",
            "1.0 # zoom",
            f"{float(cfg['analysis'].get('frame3dd_internal_force_dx_mm',25.0)):12.6f} # internal force increment",
        ]
        lines += ["1 # number of static load cases", "0.0 0.0 0.0 # gravity", f"{len(loads)} # number of loaded nodes", "# node Fx Fy Fz Mxx Myy Mzz"]
        for l in loads: lines.append(f"{l.node_id:5d} {l.Fx:12.6f} {l.Fy:12.6f} {l.Fz:12.6f} {l.Mx:12.6f} {l.My:12.6f} {l.Mz:12.6f}")
        lines += ["0 # uniform loads", "0 # trapezoidal loads", "0 # internal point loads", "0 # temperature loads", "0 # prescribed displacements", "0 # number of desired dynamic modes", "0 # matrix condensation method: none", "0 # number of condensed nodes"]
        out.write_text("\n".join(lines)+"\n",encoding='utf-8'); return out
    def _equivalence_status(self, cfg: Dict) -> str:
        if bool(cfg.get("analysis", {}).get("frame3dd_assume_truss_equivalent", False)):
            return "run_truss_equivalent"
        return "run_frame_model_not_equivalent_to_truss"

    def run(self,cfg:Dict,input_path:str|Path,output_path:str|Path)->Dict[str,str]:
        exe=self.find_executable(cfg['analysis'].get('frame3dd_path','auto'))
        if exe is None:
            return {
                'status':'not_run',
                'message':'Frame3DD não encontrado.',
                'mode':'linear_stabilized',
                'classification':'not_run',
            }
        out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True)
        try:
            proc=subprocess.run([str(exe),str(input_path),str(out)],capture_output=True,text=True,timeout=15)
            converged='** converged **' in (proc.stdout or '')
            ok=proc.returncode==0 and 'error' not in (proc.stderr or '').lower()
            status=self._equivalence_status(cfg) if (ok or converged) else 'failed'
            return {
                'status':status,
                'returncode':str(proc.returncode),
                'stdout':proc.stdout,
                'stderr':proc.stderr,
                'output_path':str(out),
                'exe':str(exe),
                'mode':'linear_stabilized',
                'classification':status,
                'note':'Frame3DD é usado como validação/visualização linear. Esforços de pórtico não substituem checagens de treliça sem equivalência explícita.'
            }
        except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
            return {
                'status':'failed',
                'message':repr(exc),
                'exe':str(exe),
                'mode':'linear_stabilized',
                'classification':'failed',
            }
