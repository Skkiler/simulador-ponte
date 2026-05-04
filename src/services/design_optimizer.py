from __future__ import annotations
import copy, json, math
from pathlib import Path
from typing import Any, Dict, List
from src.services.config_service import ConfigService
from src.services.geometry_service import GeometryService
from src.solvers.linear_truss_solver import LinearTrussSolver
from src.services.postprocessor import PostProcessor

def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or (isinstance(value,str) and value.strip()==""): return default
        v=float(value)
        if math.isnan(v) or math.isinf(v): return default
        return v
    except Exception: return default

class DesignOptimizer:
    def __init__(self):
        self.config=ConfigService(); self.geometry=GeometryService(); self.solver=LinearTrussSolver(); self.post=PostProcessor()
    def _mass(self,cfg,members):
        Ls=float(cfg['material']['stick_length_mm']); mg=float(cfg['material']['stick_mass_g']); waste=float(cfg.get('detail_model',{}).get('construction_waste_factor',0.08))
        sticks=sum(max(1,math.ceil(m.L/Ls))*m.n_sticks for m in members)
        return sticks*(1+waste)*mg, int(math.ceil(sticks*(1+waste)))
    def _primary_fs(self,checks):
        vals=[safe_float(r.get('FS_min')) for r in checks if r.get('member_role')=='primary']; vals=[v for v in vals if v is not None]
        return min(vals) if vals else 0.0
    def _reinforce(self,cfg,profile):
        s=cfg.setdefault('member_sticks_by_group',{})
        maps={
          'light': {'top_chord':3,'bottom_chord':3,'diagonal':2,'vertical':2,'top_transverse':1,'bottom_transverse':1,'support_pad':3,'chord_lacing':1},
          'balanced': {'top_chord':4,'bottom_chord':3,'diagonal':2,'vertical':2,'top_transverse':1,'bottom_transverse':1,'support_pad':4,'chord_lacing':1},
          'strong_top': {'top_chord':5,'bottom_chord':3,'diagonal':3,'vertical':2,'top_transverse':2,'bottom_transverse':1,'support_pad':4,'chord_lacing':1},
          'strong': {'top_chord':5,'bottom_chord':4,'diagonal':3,'vertical':3,'top_transverse':2,'bottom_transverse':2,'support_pad':4,'chord_lacing':1},
        }
        s.update(maps.get(profile,maps['balanced']))
        for g in ['top_bracing','bottom_bracing','cross_frame_bracing']: s[g]=1
    def run(self,cfg:Dict,out_dir:str|Path)->Dict:
        out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); base=self.config.normalize(cfg)
        H=float(base['bridge']['center_height_mm']); P=float(base['bridge']['panel_mm']); W=float(base['bridge']['width_mm']); target=float(base['analysis'].get('target_min_fs',2)); limit=float(base['material'].get('mass_limit_g',1000)); maxv=int(base['analysis'].get('max_optimizer_variants',180))
        side=list(dict.fromkeys([base['bridge'].get('side_truss_type',base['bridge'].get('truss_type','Parker')),'Parker','Pratt','Howe','Warren']))
        top=list(dict.fromkeys([base['bridge'].get('top_profile','parker_plateau'),'parker_plateau','triangular_peak','shallow_arch','flat']))
        internal=list(dict.fromkeys([base['bridge'].get('internal_truss_type','X'),'X','Warren','Pratt','Howe']))
        chord=list(dict.fromkeys([base['bridge'].get('chord_truss_type','none'),'none','Warren','X']))
        heights=sorted(set([max(120,H*.85),H,min(450,H*1.15),min(500,H*1.3)])); panels=sorted(set([max(80,P*.85),P,min(140,P*1.1)])); widths=sorted(set([max(120,W*.9),W,min(220,W*1.1)])); profiles=['light','balanced','strong_top','strong']
        rows=[]; tried=0
        for a in side:
          for b in top:
           for c in internal:
            for d in chord:
             for h in heights:
              for p in panels:
               for w in widths:
                for prof in profiles:
                 if tried>=maxv: break
                 tried+=1; v=copy.deepcopy(base); v['bridge'].update({'truss_type':a,'side_truss_type':a,'top_profile':b,'internal_truss_type':c,'cross_frame_truss_type':c,'chord_truss_type':d,'center_height_mm':h,'end_height_mm':h if b=='flat' else max(50,h/3),'panel_mm':p,'width_mm':w,'load_distribution_x_mm':[]}); self._reinforce(v,prof); v=self.config.normalize(v)
                 try:
                  nodes,members,supports,loads=self.geometry.generate(v); sol=self.solver.solve(nodes,members,supports,loads,unilateral_supports=True); checks=self.post.check_members(v,sol.member_results); fs=self._primary_fs(checks); mass,sticks=self._mass(v,members); margin=limit-mass; feasible=(fs>=target and margin>=0 and sol.status=='regular'); score=min(fs/target,2)*60+max(-1,min(1,margin/max(limit,1)))*30+(10 if sol.status=='regular' else -25)+(25 if feasible else 0)
                  rows.append({'truss_type':a,'side_truss_type':a,'top_profile':b,'internal_truss_type':c,'chord_truss_type':d,'reinforcement_profile':prof,'center_height_mm':h,'panel_mm':p,'width_mm':w,'min_fs_primary':fs,'mass_g':mass,'mass_margin_g':margin,'estimated_sticks':sticks,'solver_status':sol.status,'feasible':feasible,'score':score,'config':v})
                 except Exception as exc:
                  rows.append({'truss_type':a,'side_truss_type':a,'top_profile':b,'internal_truss_type':c,'chord_truss_type':d,'reinforcement_profile':prof,'score':-999,'feasible':False,'error':repr(exc),'config':v})
                if tried>=maxv: break
               if tried>=maxv: break
              if tried>=maxv: break
             if tried>=maxv: break
            if tried>=maxv: break
           if tried>=maxv: break
          if tried>=maxv: break
        rows=sorted(rows,key=lambda r:safe_float(r.get('score'),-999) or -999,reverse=True); feasible=[r for r in rows if r.get('feasible')]; best=feasible[0] if feasible else (rows[0] if rows else None)
        GeometryService.write_csv(out/'variant_results.csv',[{k:v for k,v in r.items() if k!='config'} for r in rows])
        if best: (out/'recommended_config.json').write_text(json.dumps(best['config'],indent=2,ensure_ascii=False),encoding='utf-8')
        return {'variants':rows,'best':best,'best_is_feasible':bool(best and best.get('feasible')),'csv_path':str(out/'variant_results.csv'),'recommended_config_path':str(out/'recommended_config.json'),'tried_variants':len(rows)}
