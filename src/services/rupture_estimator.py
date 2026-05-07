from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.numeric import safe_float


def _fs_and_id_from_member_checks(
    member_checks: List[Dict[str, Any]],
    *,
    role_filter: str | None = None,
    fs_key: str = "FS_min",
    design_only: bool = False,
) -> tuple[float | None, Any, str | None, str | None]:
    min_fs = None
    governing_id = None
    governing_group = None
    governing_mode = None
    for chk in member_checks or []:
        if role_filter and chk.get("member_role") != role_filter:
            continue
        if design_only and (chk.get("design_relevant") is False):
            continue
        fs = safe_float(chk.get(fs_key), None)
        if fs is None:
            continue
        if min_fs is None or fs < min_fs:
            min_fs = fs
            governing_id = chk.get("member_id")
            governing_group = chk.get("group")
            governing_mode = chk.get("governing_mode")
    return min_fs, governing_id, governing_group, governing_mode


def estimate_rupture_load(
    cfg: Dict[str, Any],
    member_checks: List[Dict[str, Any]],
    support_checks: List[Dict[str, Any]],
    detailed: Optional[Dict[str, Any]],
    load_kgf: float,
) -> Dict[str, Any]:
    """Estimate rupture loads from primary/all/design limit-state sets."""
    load_kgf = float(load_kgf or 0.0)

    min_fs_primary, primary_member_id, primary_group, primary_mode = _fs_and_id_from_member_checks(
        member_checks,
        role_filter="primary",
        fs_key="FS_min",
        design_only=False,
    )
    min_fs_all_raw, all_member_id, all_group, all_mode = _fs_and_id_from_member_checks(
        member_checks,
        role_filter=None,
        fs_key="FS_min_all_raw",
        design_only=False,
    )
    if min_fs_all_raw is None:
        min_fs_all_raw, all_member_id, all_group, all_mode = _fs_and_id_from_member_checks(
            member_checks,
            role_filter=None,
            fs_key="FS_min",
            design_only=False,
        )
    min_fs_design, design_member_id, design_group, design_mode = _fs_and_id_from_member_checks(
        member_checks,
        role_filter=None,
        fs_key="FS_design",
        design_only=True,
    )
    if min_fs_design is None:
        min_fs_design, design_member_id, design_group, design_mode = _fs_and_id_from_member_checks(
            member_checks,
            role_filter="primary",
            fs_key="FS_min",
            design_only=False,
        )

    min_fs_support = None
    governing_support_node = None
    for sup in support_checks or []:
        fs_val = safe_float(sup.get("FS_support_reaction"), None)
        if fs_val is None:
            continue
        if min_fs_support is None or fs_val < min_fs_support:
            min_fs_support = fs_val
            governing_support_node = sup.get("node_id") or sup.get("support_id")

    min_fs_glue = None
    governing_joint_id = None
    governing_glue_group = None
    if detailed:
        glue_rows = (
            detailed.get("glue_joints")
            or detailed.get("weakest_glue_joints")
            or []
        )
        for row in glue_rows:
            fs_joint = safe_float(
                row.get("FS_glue_shear", row.get("FS_glue", row.get("FS"))),
                None,
            )
            if fs_joint is None:
                continue
            if min_fs_glue is None or fs_joint < min_fs_glue:
                min_fs_glue = fs_joint
                governing_joint_id = row.get("joint_id")
                governing_glue_group = row.get("member_group") or "glue"

    predicted_primary = load_kgf * min_fs_primary if min_fs_primary is not None else None

    all_candidates: List[tuple[str, float, Dict[str, Any]]] = []
    if min_fs_all_raw is not None:
        all_candidates.append(
            (
                "member_all_raw",
                load_kgf * min_fs_all_raw,
                {
                    "governing_mode": all_mode,
                    "member_id": all_member_id,
                    "member_group": all_group,
                    "fs": min_fs_all_raw,
                },
            )
        )
    if min_fs_support is not None:
        all_candidates.append(
            (
                "support",
                load_kgf * min_fs_support,
                {
                    "support_node_id": governing_support_node,
                    "fs": min_fs_support,
                },
            )
        )
    if min_fs_glue is not None:
        all_candidates.append(
            (
                "glue",
                load_kgf * min_fs_glue,
                {
                    "joint_id": governing_joint_id,
                    "member_group": governing_glue_group,
                    "fs": min_fs_glue,
                },
            )
        )

    design_candidates: List[tuple[str, float, Dict[str, Any]]] = []
    if min_fs_design is not None:
        design_candidates.append(
            (
                "member_design",
                load_kgf * min_fs_design,
                {
                    "governing_mode": design_mode,
                    "member_id": design_member_id,
                    "member_group": design_group,
                    "fs": min_fs_design,
                },
            )
        )
    if min_fs_support is not None:
        design_candidates.append(
            (
                "support",
                load_kgf * min_fs_support,
                {
                    "support_node_id": governing_support_node,
                    "fs": min_fs_support,
                },
            )
        )
    if min_fs_glue is not None:
        design_candidates.append(
            (
                "glue",
                load_kgf * min_fs_glue,
                {
                    "joint_id": governing_joint_id,
                    "member_group": governing_glue_group,
                    "fs": min_fs_glue,
                },
            )
        )

    predicted_all = None
    governing_all = {}
    if all_candidates:
        name, predicted_all, meta = min(all_candidates, key=lambda t: t[1])
        governing_all = {"limit_state": name, **meta}

    predicted_design = None
    governing_design = {}
    if design_candidates:
        name, predicted_design, meta = min(design_candidates, key=lambda t: t[1])
        governing_design = {"limit_state": name, **meta}

    predicted_main = predicted_design if predicted_design is not None else predicted_all

    governing_mode = "unknown"
    if governing_design:
        ls = str(governing_design.get("limit_state", ""))
        if ls.startswith("member"):
            governing_mode = "member"
        elif ls == "support":
            governing_mode = "support"
        elif ls == "glue":
            governing_mode = "glue"

    included_limit_states = [v for _, v, _ in all_candidates]
    n_states = len(included_limit_states)
    confidence = "high" if n_states >= 3 else ("medium" if n_states == 2 else "low")

    return {
        "predicted_breaking_load_kgf": predicted_main,
        "predicted_breaking_load_primary_kgf": predicted_primary,
        "predicted_breaking_load_all_kgf": predicted_all,
        "predicted_breaking_load_design_kgf": predicted_design,
        "governing_rupture_mode": governing_mode,
        "governing_limit_state": governing_design.get("limit_state"),
        "governing_mode": governing_design.get("governing_mode"),
        "governing_member_id": governing_design.get("member_id"),
        "governing_group": governing_design.get("member_group"),
        "governing_joint_id": governing_design.get("joint_id"),
        "governing_support_node_id": governing_design.get("support_node_id"),
        "governing_fs": governing_design.get("fs"),
        "min_fs_primary": min_fs_primary,
        "min_fs_all_raw": min_fs_all_raw,
        "min_fs_design": min_fs_design,
        "min_fs_support": min_fs_support,
        "min_fs_glue": min_fs_glue,
        "rupture_basis": "min(limit_state_fs) * load",
        "confidence_level": confidence,
        "included_limit_states": included_limit_states,
        "governing_all": governing_all,
        "governing_design": governing_design,
    }
