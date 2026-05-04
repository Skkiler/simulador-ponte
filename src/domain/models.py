from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    id: int
    x: float
    y: float
    z: float
    level: str
    side: str
    x_station: float


@dataclass(frozen=True)
class Member:
    id: int
    i: int
    j: int
    group: str
    n_sticks: int
    A: float
    Asy: float
    Asz: float
    Iy: float
    Iz: float
    J: float
    E: float
    G: float
    Ky: float
    Kz: float
    L: float


@dataclass(frozen=True)
class Support:
    node_id: int
    UX: int
    UY: int
    UZ: int
    RX: int
    RY: int
    RZ: int
    support_group: str
    active_vertical: bool = True


@dataclass(frozen=True)
class Load:
    loadcase: str
    node_id: int
    Fx: float
    Fy: float
    Fz: float
    Mx: float = 0.0
    My: float = 0.0
    Mz: float = 0.0
