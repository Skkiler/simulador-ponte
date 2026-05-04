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
        for p in [Path("Frame3DD/windows/frame3dd.exe"), Path("Frame3DD/linux/frame3dd"), Path("Frame3DD/osx/frame3dd"), Path("frame3dd.exe"), Path("frame3dd")]:
            if p.exists(): return p
        found = shutil.which("frame3dd") or shutil.which("frame3dd.exe")
        return Path(found) if found else None
    def _density(self,cfg:Dict)->float:
        gmm3=float(cfg.get('material',{}).get('density_g_per_mm3',1e-6))
        return max(gmm3*1e-6,1e-18)  # tonne/mm3
    def write_input(self,cfg:Dict,nodes:List[Node],members:List[Member],supports:List[Support],loads:List[Load],out_path:str|Path)->Path:
        out=Path(out_path); out.parent.mkdir(parents=True,exist_ok=True); lines=[]
        lines.append("Ponte de palitos - Frame3DD linear estabilizado (N mm tonne)")
        lines.append("# Frame3DD é validação linear; flambagem/ruptura são pós-processadas.")
        lines.append(f"{len(nodes)} # number of nodes")
        lines.append("# node x y z rj")
        for n in nodes: lines.append(f"{n.id:5d} {n.x:12.6f} {n.y:12.6f} {n.z:12.6f} {0.0:12.6f}")
        active=[s for s in supports if s.active_vertical]
        lines.append(f"{len(active)} # number of nodes with reactions")
        lines.append("# node x y z xx yy zz")
        for s in active:
            lines.append(f"{s.node_id:5d} {int(s.UX):d} {int(s.UY):d} {int(s.UZ):d} 1 1 1")
        dens=self._density(cfg)
        lines.append(f"{len(members)} # number of frame elements")
        lines.append("# e n1 n2 Ax Asy Asz Jxx Iyy Izz E G roll density")
        for m in members:
            lines.append(f"{m.id:5d} {m.i:5d} {m.j:5d} {m.A:12.6f} {m.Asy:12.6f} {m.Asz:12.6f} {m.J:12.6f} {m.Iy:12.6f} {m.Iz:12.6f} {m.E:12.6f} {m.G:12.6f} {0.0:8.3f} {dens:12.6e}")
        lines += ["1 # include shear deformation", "0 # include geometric stiffness", "20.0 # exaggerate", "1.0 # zoom", f"{float(cfg['analysis'].get('frame3dd_internal_force_dx_mm',25.0)):12.6f} # internal force increment"]
        lines += ["1 # number of static load cases", "0.0 0.0 0.0 # gravity", f"{len(loads)} # number of loaded nodes", "# node Fx Fy Fz Mxx Myy Mzz"]
        for l in loads: lines.append(f"{l.node_id:5d} {l.Fx:12.6f} {l.Fy:12.6f} {l.Fz:12.6f} {l.Mx:12.6f} {l.My:12.6f} {l.Mz:12.6f}")
        lines += ["0 # uniform loads", "0 # trapezoidal loads", "0 # internal point loads", "0 # temperature loads", "0 # prescribed displacements", "0 # number of desired dynamic modes", "0 # matrix condensation method: none", "0 # number of condensed nodes"]
        out.write_text("\n".join(lines)+"\n",encoding='utf-8'); return out
    def run(self,cfg:Dict,input_path:str|Path,output_path:str|Path)->Dict[str,str]:
        exe=self.find_executable(cfg['analysis'].get('frame3dd_path','auto'))
        if exe is None: return {'status':'not_found','message':'Frame3DD não encontrado.','mode':'linear_stabilized'}
        out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True)
        try:
            proc=subprocess.run([str(exe),str(input_path),str(out)],capture_output=True,text=True,timeout=15)
            converged='** converged **' in (proc.stdout or '')
            ok=proc.returncode==0 and 'error' not in (proc.stderr or '').lower()
            status='ok' if ok else ('ok_with_warnings' if converged else 'error')
            return {'status':status,'returncode':str(proc.returncode),'stdout':proc.stdout,'stderr':proc.stderr,'output_path':str(out),'exe':str(exe),'mode':'linear_stabilized','note':'Frame3DD é usado como validação/visualização linear; flambagem é avaliada no pós-processador.'}
        except Exception as exc: return {'status':'error','message':repr(exc),'exe':str(exe),'mode':'linear_stabilized'}
