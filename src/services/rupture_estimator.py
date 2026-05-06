"""
Service for estimating the ultimate breaking load of a popsicle stick bridge.

This module collects approximate limit states from member and support checks and
produces a best estimate of when the structure will fail.  It always returns
a dictionary with the key ``predicted_breaking_load_kgf`` and additional
diagnostic fields.  When no valid limit state exists the predicted load is
``None`` but the dictionary still contains the fields so that the UI can
render an informative message instead of omitting the value entirely.

The current implementation is deliberately conservative and simple:

* It examines the minimum safety factor of primary members, supports and
  glue joints (if detailed information is available).
* It multiplies the design load by each minimum safety factor to obtain a
  candidate breaking load and returns the smallest of these values.
* The governing mode is reported as one of ``member``, ``support`` or
  ``glue`` depending on which produced the minimum value.

Future versions could incorporate more complex reliability models and take
additional factors such as solver irregularities into account.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional


def estimate_rupture_load(
    cfg: Dict[str, Any],
    member_checks: List[Dict[str, Any]],
    support_checks: List[Dict[str, Any]],
    detailed: Optional[Dict[str, Any]],
    load_kgf: float,
) -> Dict[str, Any]:
    """Compute an estimated breaking load in kgf for the given bridge.

    Returns a dictionary with the following keys:

      * ``predicted_breaking_load_kgf`` – the minimum estimated breaking load or
        ``None`` if indeterminable.
      * ``governing_rupture_mode`` – one of ``member``, ``support``, ``glue`` or
        ``unknown``.
      * ``governing_member_id`` – the id of the critical member if applicable.
      * ``governing_group`` – the group of the critical member if applicable.
      * ``governing_fs`` – the governing safety factor that limited the estimate.
      * ``rupture_basis`` – short description of the calculation basis.
      * ``confidence_level`` – qualitative indicator of reliability (``low``,
        ``medium`` or ``high``).
      * ``included_limit_states`` – list of limit state values considered.

    The implementation is intentionally simple to guarantee a value is always
    available.  Consumers should treat it as a heuristic indicator rather than
    a guarantee of performance.
    """
    load_kgf = float(load_kgf or 0.0)
    included: List[float] = []
    governing_mode = "unknown"
    governing_id = None
    governing_group = None
    governing_fs = None

    # Primary members: use minimum safety factor
    min_fs_primary = None
    for chk in member_checks or []:
        if chk.get("member_role") == "primary":
            fs = chk.get("FS_min")
            try:
                fs_val = float(fs)
            except (TypeError, ValueError):
                continue
            if min_fs_primary is None or fs_val < min_fs_primary:
                min_fs_primary = fs_val
                governing_id = chk.get("member_id")
                governing_group = chk.get("group")
                governing_fs = fs_val
                governing_mode = "member"
    if min_fs_primary is not None:
        included.append(load_kgf * min_fs_primary)

    # Supports: reaction safety factors may be provided
    min_fs_support = None
    for sup in support_checks or []:
        fs = sup.get("FS_support_reaction")
        if fs is None:
            continue
        try:
            fs_val = float(fs)
        except (TypeError, ValueError):
            continue
        if min_fs_support is None or fs_val < min_fs_support:
            min_fs_support = fs_val
            governing_mode = "support"
            governing_fs = fs_val
            governing_id = sup.get("node_id") or sup.get("support_id")
            governing_group = "support"
    if min_fs_support is not None:
        included.append(load_kgf * min_fs_support)

    # Glue joints: use detailed analysis if available
    min_fs_glue = None
    if detailed:
        weak = detailed.get("weakest_glue_joints") or []
        for w in weak:
            fs = w.get("FS_glue") or w.get("FS" )
            try:
                fs_val = float(fs)
            except (TypeError, ValueError):
                continue
            if min_fs_glue is None or fs_val < min_fs_glue:
                min_fs_glue = fs_val
                governing_mode = "glue"
                governing_fs = fs_val
                governing_id = w.get("joint_id")
                governing_group = w.get("member_group") or "glue"
        if min_fs_glue is not None:
            included.append(load_kgf * min_fs_glue)

    if included:
        predicted = min(included)
    else:
        predicted = None

    # Qualitative confidence level.  If solver status is irregular the
    # confidence is low; otherwise it depends on how many limit states were
    # available.
    n_states = len(included)
    if n_states >= 3:
        confidence = "high"
    elif n_states == 2:
        confidence = "medium"
    elif n_states == 1:
        confidence = "low"
    else:
        confidence = "low"

    return {
        "predicted_breaking_load_kgf": predicted,
        "governing_rupture_mode": governing_mode,
        "governing_member_id": governing_id,
        "governing_group": governing_group,
        "governing_fs": governing_fs,
        "rupture_basis": "min(FS) * load",
        "confidence_level": confidence,
        "included_limit_states": included,
    }
