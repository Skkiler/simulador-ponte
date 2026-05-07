from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.core.numeric import safe_float
from src.domain.models import Member, Node
from src.services.geometry_service import GeometryService


class AssemblyTutorialService:
    """Monta um roteiro prático de fabricação com base no detalhamento."""

    @staticmethod
    def _symmetry_map(cfg: Dict, nodes: List[Node], members: List[Member]) -> Dict[str, List[int]]:
        span = float(cfg.get("bridge", {}).get("span_mm", 0.0))
        x_mid = 0.5 * span
        node_by_id = {int(n.id): n for n in nodes}

        def edge_key(a: tuple[float, float, float], b: tuple[float, float, float], group: str) -> tuple:
            p0 = (round(float(a[0]), 6), round(float(a[1]), 6), round(float(a[2]), 6))
            p1 = (round(float(b[0]), 6), round(float(b[1]), 6), round(float(b[2]), 6))
            e0, e1 = (p0, p1) if p0 <= p1 else (p1, p0)
            return (e0, e1, str(group))

        edge_to_id: Dict[tuple, int] = {}
        for m in members:
            ni = node_by_id.get(int(m.i))
            nj = node_by_id.get(int(m.j))
            if ni is None or nj is None:
                continue
            edge_to_id[edge_key((ni.x, ni.y, ni.z), (nj.x, nj.y, nj.z), m.group)] = int(m.id)

        out: Dict[str, List[int]] = {}
        ops = [(False, False), (True, False), (False, True), (True, True)]
        for m in members:
            ni = node_by_id.get(int(m.i))
            nj = node_by_id.get(int(m.j))
            if ni is None or nj is None:
                continue
            orbit = set([int(m.id)])
            for mx, my in ops:
                p1x = (2.0 * x_mid - float(ni.x)) if mx else float(ni.x)
                p2x = (2.0 * x_mid - float(nj.x)) if mx else float(nj.x)
                p1y = (-float(ni.y)) if my else float(ni.y)
                p2y = (-float(nj.y)) if my else float(nj.y)
                key = edge_key((p1x, p1y, float(ni.z)), (p2x, p2y, float(nj.z)), m.group)
                hit = edge_to_id.get(key)
                if hit is not None:
                    orbit.add(int(hit))
            oid = str(min(orbit))
            out.setdefault(oid, sorted(orbit))
        return out

    def build(
        self,
        cfg: Dict,
        nodes: List[Node],
        members: List[Member],
        detailed: Dict[str, Any],
        out_dir: str | Path,
    ) -> Dict[str, Any]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        pieces = list((detailed or {}).get("stick_pieces", []) or [])
        joints = list((detailed or {}).get("glue_joints", []) or [])
        cuts = list((detailed or {}).get("cutting_list", []) or [])

        member_by_id = {int(m.id): m for m in members}
        stick_count_by_member: Dict[int, int] = defaultdict(int)
        mass_by_member: Dict[int, float] = defaultdict(float)
        for r in pieces:
            mid = int(safe_float(r.get("member_id"), -1) or -1)
            if mid < 0:
                continue
            stick_count_by_member[mid] += 1
            mass_by_member[mid] += safe_float(r.get("mass_g"), 0.0) or 0.0

        step_defs = [
            ("Separar e cortar palitos", []),
            ("Montar banzos inferiores", ["bottom_chord"]),
            ("Montar banzos superiores", ["top_chord"]),
            ("Colar diagonais laterais", ["diagonal"]),
            ("Colar montantes", ["vertical"]),
            ("Montar transversais", ["top_transverse", "bottom_transverse"]),
            ("Montar contraventamentos", ["top_bracing", "bottom_bracing", "cross_frame_bracing", "chord_lacing"]),
            ("Reforçar apoios e pontos de carga", ["support_pad"]),
            ("Conferência final de simetria e massa", []),
        ]

        members_by_group: Dict[str, List[int]] = defaultdict(list)
        for m in members:
            members_by_group[str(m.group)].append(int(m.id))

        assembly_steps: List[Dict[str, Any]] = []
        stick_count_by_step: Dict[str, int] = {}
        for idx, (title, groups) in enumerate(step_defs, 1):
            mids: List[int] = []
            if groups:
                for g in groups:
                    mids.extend(members_by_group.get(g, []))
            mids = sorted(set(mids))
            step_sticks = sum(int(stick_count_by_member.get(mid, 0)) for mid in mids)
            step_mass = sum(float(mass_by_member.get(mid, 0.0)) for mid in mids)
            key = f"S{idx:02d}"
            stick_count_by_step[key] = step_sticks
            assembly_steps.append(
                {
                    "step_id": key,
                    "step_index": idx,
                    "title": title,
                    "target_groups": groups,
                    "member_ids": mids,
                    "stick_count": step_sticks,
                    "estimated_mass_g": round(step_mass, 3),
                    "instruction": (
                        "Use a visualização 3D para conferir alinhamento e aplique cola apenas após pré-posicionamento a seco."
                        if groups
                        else "Conferir simetria, massa estimada e continuidade das juntas antes da cura final."
                    ),
                }
            )

        joint_list: List[Dict[str, Any]] = []
        for j in joints:
            joint_list.append(
                {
                    "joint_id": j.get("joint_id"),
                    "member_id": j.get("member_id"),
                    "joint_model": j.get("joint_model"),
                    "overlap_length_mm": j.get("overlap_length_mm"),
                    "splice_center_mm": j.get("splice_center_mm"),
                }
            )

        cut_list = [
            {
                "cut_length_mm": r.get("cut_length_mm"),
                "quantity": r.get("quantity"),
                "total_length_mm": r.get("total_length_mm"),
            }
            for r in cuts
        ]

        symmetry_repetition_map = self._symmetry_map(cfg, nodes, members)

        result = {
            "assembly_steps": assembly_steps,
            "cut_list": cut_list,
            "joint_list": joint_list,
            "stick_count_by_step": stick_count_by_step,
            "stick_count_by_member": {str(k): int(v) for k, v in sorted(stick_count_by_member.items())},
            "symmetry_repetition_map": symmetry_repetition_map,
        }

        (out / "assembly_tutorial.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        md_lines = ["# Tutorial de montagem", ""]
        for step in assembly_steps:
            md_lines.append(f"## {step['step_id']} - {step['title']}")
            md_lines.append(f"- Grupos: {', '.join(step['target_groups']) if step['target_groups'] else 'geral'}")
            md_lines.append(f"- Membros: {', '.join(map(str, step['member_ids'])) if step['member_ids'] else '—'}")
            md_lines.append(f"- Palitos estimados nesta etapa: {step['stick_count']}")
            md_lines.append(f"- Instrução: {step['instruction']}")
            md_lines.append("")
        (out / "assembly_tutorial.md").write_text("\n".join(md_lines), encoding="utf-8")

        csv_rows = [
            {
                "step_id": step["step_id"],
                "step_index": step["step_index"],
                "title": step["title"],
                "target_groups": ";".join(step.get("target_groups", [])),
                "member_ids": ";".join(str(v) for v in step.get("member_ids", [])),
                "stick_count": step.get("stick_count"),
                "estimated_mass_g": step.get("estimated_mass_g"),
                "instruction": step.get("instruction"),
            }
            for step in assembly_steps
        ]
        GeometryService.write_csv(out / "assembly_steps.csv", csv_rows)

        return result


__all__ = ["AssemblyTutorialService"]
