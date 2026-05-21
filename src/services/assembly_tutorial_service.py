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
            ("Separar, conferir e cortar palitos", []),
            ("Montar banzos inferiores no gabarito", ["bottom_chord", "support_pad"]),
            ("Montar banzos superiores em pares espelhados", ["top_chord"]),
            ("Instalar montantes entre banzo inferior e superior", ["vertical"]),
            ("Instalar diagonais laterais e resolver cruzamentos X", ["diagonal"]),
            ("Fechar travessas inferiores e superiores", ["bottom_transverse", "top_transverse"]),
            ("Instalar travamentos centrais e contraventamentos", ["cross_frame_bracing", "top_bracing", "bottom_bracing", "chord_lacing"]),
            ("Instalar plataforma/sela de carga e conferir apoios", ["top_transverse", "support_pad"]),
            ("Cura, inspeção dimensional e pesagem", []),
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
            if not groups and idx == 1:
                instruction = (
                    "Cortar por comprimento da lista final, identificar cada peça por membro/lane/peça e separar os cortes em ângulo apenas quando o CSV marcar miter_cut_required=true."
                )
            elif "bottom_chord" in groups:
                instruction = (
                    "Fixar o banzo inferior primeiro em gabarito reto; colar sapatas/support_pad por sobreposição de face e manter emendas desencontradas entre lanes."
                )
            elif "top_chord" in groups:
                instruction = (
                    "Montar o banzo superior fora da ponte como par espelhado; usar seções box pares e simples, sem caixa ímpar; conferir paralelismo antes de ligar aos montantes."
                )
            elif "vertical" in groups:
                instruction = (
                    "Pré-posicionar cada montante a seco entre banzos; se miter_cut_start/end_required=true, fazer um único corte de ponta no ângulo indicado para encostar na face do banzo."
                )
            elif "diagonal" in groups:
                instruction = (
                    "Instalar diagonais em pares simétricos. Em painéis X, uma diagonal fica na camada frente e a outra na camada fundo, sem junta central resistente; não colar como nó estrutural no cruzamento."
                )
            elif "cross_frame_bracing" in groups or "top_bracing" in groups or "bottom_bracing" in groups:
                instruction = (
                    "Instalar travamentos depois que as duas laterais estiverem rígidas; respeitar camada frente/fundo dos X e usar cola apenas nas extremidades ou na tala indicada."
                )
            elif "top_transverse" in groups or "bottom_transverse" in groups:
                instruction = (
                    "Fechar transversais com esquadro, mantendo as faces de cola limpas; eles definem a largura e distribuem a carga da plataforma."
                )
            else:
                instruction = (
                    "Conferir simetria, massa estimada, contato dos apoios e continuidade das juntas antes da cura final."
                )
            assembly_steps.append(
                {
                    "step_id": key,
                    "step_index": idx,
                    "title": title,
                    "target_groups": groups,
                    "member_ids": mids,
                    "stick_count": step_sticks,
                    "estimated_mass_g": round(step_mass, 3),
                    "instruction": instruction,
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
                "estimated_mass_g": step.get("estimated_mass_g"),
                "instruction": step.get("instruction"),
            }
            for step in assembly_steps
        ]
        GeometryService.write_csv(out / "assembly_steps.csv", csv_rows)

        return result


__all__ = ["AssemblyTutorialService"]
