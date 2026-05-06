from __future__ import annotations

"""
assembly_grouping_service
=========================

This module provides a simple grouping heuristic for stick‐level pieces
produced by the ``StickDetailService``. The goal is to collapse thousands
of individual stick rows into a handful of logical assembly groups.  Each
group attempts to represent a class of members or sub‐assemblies that
share similar geometry, length, orientation and cross‐section.  The
grouping is intentionally conservative and does not attempt to solve
hard clustering problems; instead it buckets pieces by user‐visible
attributes so that the UI can present a concise summary of the
construction effort.

Each group contains:

* ``group_key`` – A unique key identifying the group.
* ``member_group`` – The original structural group (e.g. ``banzo_inferior``).
* ``orientation`` – One of ``horizontal``, ``vertical`` or ``diagonal``.
* ``approx_length_mm`` – The average cut length rounded to the nearest 10 mm.
* ``n_pieces`` – Total count of pieces in the group.
* ``n_members`` – Number of distinct structural members represented.
* ``n_sticks`` – Number of sticks per piece (if available) or None.
* ``length_range_mm`` – Min/max cut length within the group.
* ``mass_g`` – Sum of piece masses for the group.
* ``width_mm`` – Nominal stick width (from piece metadata) or None.
* ``thickness_mm`` – Nominal stick thickness (from piece metadata) or None.
* ``sample_member_ids`` – Example member identifiers included in the group.
* ``friendly_name`` – A human friendly label combining the above.

The service exposes one main function ``group_stick_pieces`` that
ingests a list of piece dictionaries and returns a list of grouped
rows.  It can also write JSON/CSV files if desired.

Note
----
This grouping is heuristic and does not attempt to map exactly to
the structural classifications requested by the user (e.g. distinguishing
between different types of trusses). Those refinements can be added
later.  The intent is to provide a pragmatic reduction that makes the
"Montagem e Cola" page readable without overwhelming the user.
"""

from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

from src.core.numeric import safe_float, safe_int

def _orientation_from_endpoints(p0: Tuple[float, float, float], p1: Tuple[float, float, float]) -> str:
    """Classify the orientation of a piece based on its end point differences.

    Parameters
    ----------
    p0, p1 : tuple of floats
        (x, y, z) coordinates of the piece endpoints.

    Returns
    -------
    str
        One of ``"horizontal"``, ``"vertical"`` or ``"diagonal"``.  The
        classification is based on which axis exhibits the largest
        difference.  Vertical pieces have dominant z differences; horizontal
        pieces have dominant x or y differences; anything else is
        considered diagonal.
    """
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)

    # Compare dominant axis
    max_axis = max(dx, dy, dz)
    if max_axis <= 1e-9:
        return "horizontal"
    if dz >= dx and dz >= dy:
        return "vertical"
    if dx >= dy and dx >= dz:
        return "horizontal"
    # fall back
    return "diagonal"


class AssemblyGroupingService:
    """Service to collapse stick level pieces into assembly groups."""

    def __init__(self) -> None:
        pass

    def group_stick_pieces(self, pieces: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate stick pieces into logical groups.

        Parameters
        ----------
        pieces : iterable of dict
            Raw piece dictionaries from ``StickDetailService``.  It is
            expected that each piece contains at least the following
            keys: ``member_id``, ``member_group``, ``cut_length_mm``,
            ``x0_mm``, ``y0_mm``, ``z0_mm``, ``x1_mm``, ``y1_mm``, ``z1_mm``,
            ``mass_g``.  Optionally, ``width_mm``, ``thickness_mm``, and
            ``n_sticks`` may be present.

        Returns
        -------
        list of dict
            A list of summary rows representing each group.  See module
            documentation for field descriptions.
        """
        # Bucket pieces by (member_group, orientation, approx_length_bin, n_sticks)
        buckets: Dict[Tuple[str, str, float, int | None], List[Dict[str, Any]]] = defaultdict(list)
        for r in pieces:
            mg = str(r.get("member_group", "sem_grupo"))
            # Compute orientation from endpoints
            p0 = (
                float(r.get("x0_mm", 0.0) or 0.0),
                float(r.get("y0_mm", 0.0) or 0.0),
                float(r.get("z0_mm", 0.0) or 0.0),
            )
            p1 = (
                float(r.get("x1_mm", 0.0) or 0.0),
                float(r.get("y1_mm", 0.0) or 0.0),
                float(r.get("z1_mm", 0.0) or 0.0),
            )
            orientation = _orientation_from_endpoints(p0, p1)
            # Use cut length if available, otherwise distance between endpoints
            l_clean = safe_float(r.get("cut_length_mm"), None)
            if l_clean is None:
                L = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2) ** 0.5
            else:
                L = l_clean
            # Bin length to nearest 10 mm for grouping
            length_bin = round(L / 10.0) * 10.0
            n_sticks = safe_int(r.get("n_sticks"), None)  # not always present
            key = (mg, orientation, length_bin, n_sticks)
            buckets[key].append(r)

        groups: List[Dict[str, Any]] = []
        for key, rows in buckets.items():
            member_group, orientation, length_bin, n_sticks = key
            cut_lengths = []
            masses = []
            widths = []
            thks = []
            member_ids = set()
            for r in rows:
                # capture cut length
                cut_len = safe_float(r.get("cut_length_mm", 0.0), None)
                if cut_len is not None:
                    cut_lengths.append(cut_len)
                # capture mass
                mass_val = safe_float(r.get("mass_g", 0.0), None)
                if mass_val is not None:
                    masses.append(mass_val)
                # width and thickness
                width = safe_float(r.get("width_mm"), None)
                if width is not None:
                    widths.append(width)
                thk = safe_float(r.get("thickness_mm"), None)
                if thk is not None:
                    thks.append(thk)
                # member id
                mid = safe_int(r.get("member_id"), None)
                if mid is not None:
                    member_ids.add(mid)
            n_pieces = len(rows)
            n_members = len(member_ids)
            total_mass = sum(masses) if masses else None
            avg_length = mean(cut_lengths) if cut_lengths else 0.0
            length_range = (min(cut_lengths) if cut_lengths else 0.0, max(cut_lengths) if cut_lengths else 0.0)
            width = mean(widths) if widths else None
            thickness = mean(thks) if thks else None
            friendly_name = f"{member_group} {orientation} ≈{length_bin:.0f} mm"
            if n_sticks:
                friendly_name += f" ({n_sticks}× palitos)"
            groups.append({
                "group_key": "|".join([str(member_group), orientation, f"{length_bin:.0f}", str(n_sticks or '')]),
                "member_group": member_group,
                "orientation": orientation,
                "approx_length_mm": length_bin,
                "n_pieces": n_pieces,
                "n_members": n_members,
                "n_sticks": n_sticks,
                "length_range_mm": length_range,
                "average_length_mm": avg_length,
                "mass_g": total_mass,
                "width_mm": width,
                "thickness_mm": thickness,
                "sample_member_ids": sorted(member_ids)[:5],
                "friendly_name": friendly_name,
            })

        # Sort groups by member group then descending number of pieces
        groups.sort(key=lambda g: (-g["n_pieces"], g["member_group"], g["approx_length_mm"]))
        return groups

    def summarize(self, pieces: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary dictionary with high level stats.

        Parameters
        ----------
        pieces : iterable of dict
            Piece rows as in ``group_stick_pieces``.

        Returns
        -------
        dict
            Summary values such as total piece count, total mass, unique
            member count and overall length range.
        """
        p_list = list(pieces)
        count = len(p_list)
        masses: List[float] = []
        lengths: List[float] = []
        member_ids: set[int] = set()
        for r in p_list:
            mass_val = safe_float(r.get("mass_g", 0.0), None)
            if mass_val is not None:
                masses.append(mass_val)
            length_val = safe_float(r.get("cut_length_mm", 0.0), None)
            if length_val is not None:
                lengths.append(length_val)
            mid = safe_int(r.get("member_id"), None)
            if mid is not None:
                member_ids.add(mid)
        total_mass = sum(masses) if masses else None
        length_range = (min(lengths) if lengths else 0.0, max(lengths) if lengths else 0.0)
        return {
            "total_pieces": count,
            "total_mass_g": total_mass,
            "unique_members": len(member_ids),
            "length_range_mm": length_range,
        }
