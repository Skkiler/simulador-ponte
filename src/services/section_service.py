from __future__ import annotations
import math
from typing import Dict, List, Tuple

class SectionService:
    """Cálculos de seções compostas, inércia e capacidades resistentes."""
    @staticmethod
    def rectangular_section(width_mm: float, thickness_mm: float) -> Dict[str, float]:
        b=float(width_mm); h=float(thickness_mm); A=b*h
        Iy=b*h**3/12.0; Iz=h*b**3/12.0
        J=(1.0/3.0)*min(b,h)*max(b,h)**3
        return {"A":A,"Iy":Iy,"Iz":Iz,"J":J,"width_mm":b,"thickness_mm":h}
    @classmethod
    def equivalent_laminated_section(cls,n_sticks:int,material:Dict[str,float])->Dict[str,float]:
        return cls.composite_section(n_sticks,material,{"layout":"stacked"})
    @classmethod
    def composite_section(cls,n_sticks:int,material:Dict[str,float],layout_cfg:Dict|None=None)->Dict[str,float]:
        n=max(1,int(n_sticks)); layout_cfg=layout_cfg or {"layout":"stacked"}
        layout=str(layout_cfg.get("layout","stacked")).lower()
        b=float(material["stick_width_mm"]); t=float(material["stick_thickness_mm"])
        A1=b*t; Iy1=b*t**3/12.0; Iz1=t*b**3/12.0
        positions:List[Tuple[float,float]]=[]
        if layout=="side_by_side":
            start=-0.5*(n-1)*b; positions=[(start+k*b,0.0) for k in range(n)]
        elif layout=="box" and n>=4:
            sy=max(float(layout_cfg.get("spacing_y_mm",b+2.0)),b)
            sz=max(float(layout_cfg.get("spacing_z_mm",t+2.0)),t)
            base=[(-sy/2,-sz/2),(sy/2,-sz/2),(-sy/2,sz/2),(sy/2,sz/2)]
            positions=[base[k%4] for k in range(n)]
        else:
            start=-0.5*(n-1)*t; positions=[(0.0,start+k*t) for k in range(n)]
        A=n*A1; cy=sum(y*A1 for y,z in positions)/A; cz=sum(z*A1 for y,z in positions)/A
        Iy=sum(Iy1+A1*(z-cz)**2 for y,z in positions); Iz=sum(Iz1+A1*(y-cy)**2 for y,z in positions)
        J=max(1e-9,0.35*(Iy+Iz))
        width=(max(y for y,z in positions)-min(y for y,z in positions)+b) if positions else b
        height=(max(z for y,z in positions)-min(z for y,z in positions)+t) if positions else t
        return {"A":A,"Iy":Iy,"Iz":Iz,"J":J,"n_sticks":n,"width_mm":width,"thickness_mm":height,"centroid_y_mm":cy,"centroid_z_mm":cz,"layout":layout,"stick_positions_yz":positions}
    @staticmethod
    def compression_capacity_N(n_sticks:int,material:Dict[str,float])->float:
        n=int(n_sticks); c1=float(material["compression_capacity_one_stick_N"]); c2=float(material["compression_capacity_two_sticks_N"])
        if n<=1: return c1
        if n==2: return c2
        return n*(c2/2.0)
    @staticmethod
    def tension_capacity_N(n_sticks:int,material:Dict[str,float])->float:
        return int(n_sticks)*float(material["tension_capacity_per_stick_N"])
    @staticmethod
    def euler_buckling_N(E_MPa:float,I_mm4:float,K:float,L_mm:float)->float:
        if L_mm<=0: return float("inf")
        return (math.pi**2*float(E_MPa)*float(I_mm4))/((float(K)*float(L_mm))**2)
    @staticmethod
    def radius_of_gyration(I_mm4:float,A_mm2:float)->float:
        return math.sqrt(max(0.0,I_mm4)/A_mm2) if A_mm2>0 else 0.0
    @staticmethod
    def member_length_mm(ni,nj)->float:
        dx=nj.x-ni.x; dy=nj.y-ni.y; dz=nj.z-ni.z
        return math.sqrt(dx*dx+dy*dy+dz*dz)
