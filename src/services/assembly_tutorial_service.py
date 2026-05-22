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
            ("Desenhar gabarito 1:1", []),
            ("Montar banzo inferior", ["bottom_chord", "support_pad"]),
            ("Montar banzo superior", ["top_chord"]),
            ("Colar montantes", ["vertical"]),
            ("Colar diagonais", ["diagonal"]),
            ("Montar segunda lateral igual", ["bottom_chord", "top_chord", "vertical", "diagonal"]),
            ("Unir laterais com transversais", ["bottom_transverse", "top_transverse", "support_pad"]),
            ("Aplicar travamentos superior/inferior", ["cross_frame_bracing", "top_bracing", "bottom_bracing", "chord_lacing"]),
            ("Montar deck/plataforma de carga", ["top_transverse", "support_pad"]),
            ("Cura e inspeção final", []),
        ]

        members_by_group: Dict[str, List[int]] = defaultdict(list)
        for m in members:
            members_by_group[str(m.group)].append(int(m.id))

        assembly_steps: List[Dict[str, Any]] = []
        stick_count_by_step: Dict[str, int] = {}
        piece_by_member: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for p in pieces:
            mid = int(safe_float(p.get("member_id"), -1) or -1)
            if mid >= 0:
                piece_by_member[mid].append(p)
        joints_by_member: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for j in joints:
            mid = int(safe_float(j.get("member_id"), -1) or -1)
            if mid >= 0:
                joints_by_member[mid].append(j)
        cure_stage_default_min = int(safe_float(cfg.get("detail_model", {}).get("joint_cure_stage_minutes", 30), 30) or 30)
        cure_final_min = int(safe_float(cfg.get("detail_model", {}).get("joint_cure_final_minutes", 720), 720) or 720)
        for idx, (title, groups) in enumerate(step_defs, 1):
            mids: List[int] = []
            if groups:
                for g in groups:
                    mids.extend(members_by_group.get(g, []))
            mids = sorted(set(mids))
            step_pieces: List[Dict[str, Any]] = []
            step_joints: List[Dict[str, Any]] = []
            for mid in mids:
                step_pieces.extend(piece_by_member.get(mid, []))
                step_joints.extend(joints_by_member.get(mid, []))
            step_sticks = sum(int(stick_count_by_member.get(mid, 0)) for mid in mids)
            step_mass = sum(float(mass_by_member.get(mid, 0.0)) for mid in mids)
            key = f"S{idx:02d}"
            stick_count_by_step[key] = step_sticks
            if idx == 1:
                instruction = "Traçar o vão e as estações dos painéis no gabarito plano; marcar posição dos apoios e dos nós para repetição das duas laterais."
            elif idx == 2:
                instruction = "Montar o banzo inferior com palitos inteiros/sobrepostos e emendas desencontradas; manter overlap mínimo e alinhamento longitudinal."
            elif idx == 3:
                instruction = "Montar o banzo superior em gabarito, respeitando perfil escolhido e orientação dos palitos."
            elif idx == 4:
                instruction = "Colar montantes entre banzos com esquadro, sem desalinhamento torsional."
            elif idx == 5:
                instruction = "Colar diagonais no padrão Pratt/Warren sem cruzamento lateral no mesmo plano."
            elif idx == 6:
                instruction = "Repetir a lateral oposta com as mesmas peças e mesmas regras de emenda."
            elif idx == 7:
                instruction = "Unir as duas laterais com transversais para travar largura e geometria global."
            elif idx == 8:
                instruction = "Aplicar travamentos superior/inferior, com X apenas em camadas/documentação física."
            elif idx == 9:
                instruction = "Montar mesa/plataforma de carga e verificar distribuição conforme modelo de aplicação."
            else:
                instruction = "Respeitar tempo de cura final, inspecionar juntas críticas, massa e contato de apoio."

            step_lengths = sorted(
                {
                    round(float(safe_float(p.get("cut_length_mm"), 0.0) or 0.0), 3)
                    for p in step_pieces
                    if safe_float(p.get("cut_length_mm"), None) is not None
                }
            )
            overlap_vals = [
                float(safe_float(j.get("overlap_length_mm"), 0.0) or 0.0)
                for j in step_joints
                if safe_float(j.get("overlap_length_mm"), None) is not None
            ]
            overlap_mm = (min(overlap_vals), max(overlap_vals)) if overlap_vals else (None, None)
            if "top_chord" in groups or "vertical" in groups:
                orientation_note = "preferir palito em pé (edge) quando indicado na seção"
            elif "bottom_chord" in groups:
                orientation_note = "priorizar continuidade e cola de face; palito em pé/deitado conforme seção"
            else:
                orientation_note = "seguir orientação de seção exportada (palito em pé/deitado)"
            has_angle_cut = any(bool(p.get("miter_cut_required", False)) for p in step_pieces)
            cure_time_min = cure_final_min if idx == 10 else cure_stage_default_min
            assembly_steps.append(
                {
                    "step_id": key,
                    "step_index": idx,
                    "title": title,
                    "target_groups": groups,
                    "member_ids": mids,
                    "stick_count": step_sticks,
                    "quantity": len(step_pieces),
                    "estimated_mass_g": round(step_mass, 3),
                    "instruction": instruction,
                    "piece_ids": [str(p.get("stick_id")) for p in step_pieces if p.get("stick_id")][:120],
                    "lengths_mm": step_lengths,
                    "glue_face": "sobreposição de face",
                    "overlap_mm_min": overlap_mm[0],
                    "overlap_mm_max": overlap_mm[1],
                    "cure_time_min": cure_time_min,
                    "orientation_note": orientation_note,
                    "angle_cut_alert": has_angle_cut,
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

        # Instruções compactas de corte em ângulo por peça real. Mantemos esta
        # lista separada do CSV completo para o manual continuar legível: ela
        # contém somente peças cuja ponta realmente precisa de corte terminal.
        miter_cut_instructions: List[Dict[str, Any]] = []
        for r in pieces:
            if not bool(r.get("miter_cut_required", False)):
                continue
            start_required = bool(r.get("miter_cut_start_required", False))
            end_required = bool(r.get("miter_cut_end_required", False))
            item: Dict[str, Any] = {
                "stick_id": r.get("stick_id"),
                "member_id": r.get("member_id"),
                "member_group": r.get("member_group"),
                "lane": r.get("lane"),
                "piece_index": r.get("piece_index"),
                "cut_length_mm": r.get("cut_length_mm"),
                "instruction": "corte terminal; não cortar emendas internas",
            }
            if start_required:
                item["start_angle_deg"] = r.get("miter_cut_start_angle_deg")
                item["start_position"] = r.get("miter_cut_start_position", "ponta inicial") or "ponta inicial"
                item["start_host"] = r.get("miter_cut_start_host_group")
                item["start_relation"] = r.get("miter_cut_start_relation")
                item["start_skew_sign"] = r.get("miter_cut_start_skew_sign")
                item["start_trim_axis"] = r.get("miter_cut_start_trim_axis")
            if end_required:
                item["end_angle_deg"] = r.get("miter_cut_end_angle_deg")
                item["end_position"] = r.get("miter_cut_end_position", "ponta final") or "ponta final"
                item["end_host"] = r.get("miter_cut_end_host_group")
                item["end_relation"] = r.get("miter_cut_end_relation")
                item["end_skew_sign"] = r.get("miter_cut_end_skew_sign")
                item["end_trim_axis"] = r.get("miter_cut_end_trim_axis")
            miter_cut_instructions.append(item)

        symmetry_repetition_map = self._symmetry_map(cfg, nodes, members)

        result = {
            "assembly_steps": assembly_steps,
            "cut_list": cut_list,
            "joint_list": joint_list,
            "miter_cut_instructions": miter_cut_instructions,
            "stick_count_by_step": stick_count_by_step,
            "stick_count_by_member": {str(k): int(v) for k, v in sorted(stick_count_by_member.items())},
            "symmetry_repetition_map": symmetry_repetition_map,
        }

        (out / "assembly_tutorial.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        md_lines = [
            "# Manual de montagem",
            "",
            "Este roteiro assume montagem em gabarito plano, colagem por sobreposição de faces e cura completa antes de retirar as laterais do gabarito.",
            "",
            "## Regras gerais",
            "- Usar somente palitos nas dimensões configuradas e cortes em múltiplos de 5 mm.",
            "- Corte em ângulo só é permitido na ponta real da peça, quando o CSV marcar `miter_cut_required=true`.",
            "- Emendas internas entre pedaços do mesmo membro devem permanecer retas e sobrepostas, não em diagonal.",
            "- Nos X, usar apenas uma das soluções documentadas: preferir Warren/Pratt sem cruzamento; se X for mantido manualmente, dividir no centro e colar com tala curta indicada no CSV.",
            "",
        ]
        for step in assembly_steps:
            md_lines.append(f"## {step['step_id']} - {step['title']}")
            md_lines.append(f"- Grupos: {', '.join(step['target_groups']) if step['target_groups'] else 'geral'}")
            md_lines.append(f"- Membros: {', '.join(map(str, step['member_ids'])) if step['member_ids'] else '—'}")
            md_lines.append(f"- Palitos estimados nesta etapa: {step['stick_count']}")
            md_lines.append(f"- Quantidade de peças nesta etapa: {step.get('quantity', 0)}")
            md_lines.append(f"- IDs de peças (amostra): {', '.join(step.get('piece_ids', [])[:24]) if step.get('piece_ids') else '—'}")
            md_lines.append(
                f"- Comprimentos [mm]: {', '.join(f'{v:.1f}' for v in (step.get('lengths_mm') or [])[:16]) if step.get('lengths_mm') else '—'}"
            )
            md_lines.append(
                f"- Overlap [mm]: min={step.get('overlap_mm_min') if step.get('overlap_mm_min') is not None else '—'} / "
                f"max={step.get('overlap_mm_max') if step.get('overlap_mm_max') is not None else '—'}"
            )
            md_lines.append(f"- Face de cola: {step.get('glue_face', 'sobreposição de face')}")
            md_lines.append(f"- Tempo de cura recomendado: {step.get('cure_time_min', '—')} min")
            md_lines.append(f"- Orientação: {step.get('orientation_note', '—')}")
            md_lines.append(f"- Alerta de corte angular: {'sim' if step.get('angle_cut_alert') else 'não'}")
            md_lines.append(f"- Procedimento: {step['instruction']}")
            md_lines.append("- Controle: medir alinhamento, colar a seco primeiro, aplicar cola fina, prensar e aguardar cura antes de avançar para a próxima etapa crítica.")
            md_lines.append("")
        if miter_cut_instructions:
            md_lines.append("## Tabela compacta de cortes em ângulo")
            md_lines.append("")
            md_lines.append("Use esta tabela junto ao `stick_pieces.csv`. O corte é sempre na ponta indicada, nunca no meio da peça nem em emenda sobreposta.")
            md_lines.append("")
            md_lines.append("| Peça | Grupo | Compr. [mm] | Corte inicial | Corte final | Host |")
            md_lines.append("|---|---|---:|---|---|---|")
            for item in miter_cut_instructions[:80]:
                start_txt = "—"
                end_txt = "—"
                hosts = []
                if item.get("start_angle_deg") is not None:
                    start_txt = f"{item.get('start_angle_deg')}° ({item.get('start_position')}, eixo {item.get('start_trim_axis')}, skew {item.get('start_skew_sign')})"
                    if item.get("start_host"):
                        hosts.append(str(item.get("start_host")))
                if item.get("end_angle_deg") is not None:
                    end_txt = f"{item.get('end_angle_deg')}° ({item.get('end_position')}, eixo {item.get('end_trim_axis')}, skew {item.get('end_skew_sign')})"
                    if item.get("end_host"):
                        hosts.append(str(item.get("end_host")))
                md_lines.append(
                    f"| {item.get('stick_id')} | {item.get('member_group')} | {item.get('cut_length_mm')} | {start_txt} | {end_txt} | {', '.join(hosts) or '—'} |"
                )
            if len(miter_cut_instructions) > 80:
                md_lines.append(f"\nTabela truncada no manual: consultar `stick_pieces.csv` para as {len(miter_cut_instructions)} peças com corte em ângulo.")
            md_lines.append("")

        (out / "assembly_tutorial.md").write_text("\n".join(md_lines), encoding="utf-8")

        GeometryService.write_csv(out / "miter_cut_instructions.csv", miter_cut_instructions)

        csv_rows = [
            {
                "step_id": step["step_id"],
                "step_index": step["step_index"],
                "title": step["title"],
                "target_groups": ";".join(step.get("target_groups", [])),
                "member_ids": ";".join(str(v) for v in step.get("member_ids", [])),
                "stick_count": step.get("stick_count"),
                "quantity": step.get("quantity"),
                "estimated_mass_g": step.get("estimated_mass_g"),
                "piece_ids_sample": ";".join(step.get("piece_ids", [])[:60]),
                "lengths_mm": ";".join(str(v) for v in step.get("lengths_mm", [])),
                "glue_face": step.get("glue_face"),
                "overlap_mm_min": step.get("overlap_mm_min"),
                "overlap_mm_max": step.get("overlap_mm_max"),
                "cure_time_min": step.get("cure_time_min"),
                "orientation_note": step.get("orientation_note"),
                "angle_cut_alert": step.get("angle_cut_alert"),
                "instruction": step.get("instruction"),
            }
            for step in assembly_steps
        ]
        GeometryService.write_csv(out / "assembly_steps.csv", csv_rows)

        return result


__all__ = ["AssemblyTutorialService"]
