from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.core.numeric import safe_float, safe_sort_key
from src.core.safety import risk_from_fs, safety_label
from src.domain.models import Member, Node
from src.services.geometry_service import GeometryService
from src.services.mass_guard import resolve_mass_limits
from src.services.section_service import SectionService
from src.services.splice_staggering_service import SpliceStaggeringService


class StickDetailService:
    """
    Modelo rápido peça-a-peça: palitos, sobreposições, cola, massa e recomendações.

    Este serviço não faz FEM. Ele expande cada membro estrutural equivalente em
    peças de palito, estima cortes, sobreposições, áreas coladas, tensões médias,
    massa e recomendações construtivas.
    """

    def __init__(self, section_service: SectionService | None = None) -> None:
        self.sections = section_service or SectionService()
        self.splice_stagger = SpliceStaggeringService()

    @staticmethod
    def floor_to_cut_increment(
        value_mm: float,
        increment_mm: float = 5.0,
        min_value_mm: float = 5.0,
    ) -> float:
        inc = max(1.0e-9, float(increment_mm))
        v = max(float(min_value_mm), float(value_mm))
        return max(float(min_value_mm), math.floor(v / inc) * inc)

    @staticmethod
    def ceil_to_cut_increment(
        value_mm: float,
        increment_mm: float = 5.0,
        min_value_mm: float = 5.0,
        max_value_mm: float | None = None,
    ) -> float:
        """Comprimento fabricado em escala de oficina, nunca menor que a geometria.

        Versões anteriores arredondavam para baixo. Isso fazia o relatório
        dizer que uma peça geométrica de, por exemplo, 108,7 mm seria cortada
        com 105 mm, economizando massa e palito no papel, mas criando uma peça
        fisicamente curta. Para fabricação, o blank deve ser arredondado para
        cima em múltiplos de 5 mm e depois aparado/lixado se necessário.
        """
        inc = max(1.0e-9, float(increment_mm))
        v = max(float(min_value_mm), float(value_mm))
        rounded = math.ceil((v - 1.0e-9) / inc) * inc
        if max_value_mm is not None:
            rounded = min(float(max_value_mm), rounded)
        return max(float(min_value_mm), rounded)

    @staticmethod
    def _unit_vector(ni: Node, nj: Node) -> Tuple[float, float, float, float]:
        dx = nj.x - ni.x
        dy = nj.y - ni.y
        dz = nj.z - ni.z

        L = math.sqrt(dx * dx + dy * dy + dz * dz)

        if L <= 0:
            return 0.0, 0.0, 0.0, 0.0

        return dx / L, dy / L, dz / L, L

    @staticmethod
    def _cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    @staticmethod
    def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def _normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        if n <= 1.0e-12:
            return (0.0, 0.0, 0.0)
        return (v[0] / n, v[1] / n, v[2] / n)

    @classmethod
    def _local_section_axes(
        cls,
        ux: float,
        uy: float,
        uz: float,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return member-local y/z axes used to place each stick lane.

        The solver member line is interpreted as the centroidal axis.  The local
        section ``z`` axis is kept as close as possible to global vertical; for a
        nearly vertical member we fall back to a horizontal construction frame.
        This prevents the piece-by-piece 3D view from inventing one-sided lane
        offsets that do not exist in the calculation.
        """
        d = cls._normalize((float(ux), float(uy), float(uz)))
        if d == (0.0, 0.0, 0.0):
            return (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)

        global_z = (0.0, 0.0, 1.0)
        proj_z = (
            global_z[0] - cls._dot(global_z, d) * d[0],
            global_z[1] - cls._dot(global_z, d) * d[1],
            global_z[2] - cls._dot(global_z, d) * d[2],
        )
        local_z = cls._normalize(proj_z)
        if local_z == (0.0, 0.0, 0.0):
            global_y = (0.0, 1.0, 0.0)
            proj_y = (
                global_y[0] - cls._dot(global_y, d) * d[0],
                global_y[1] - cls._dot(global_y, d) * d[1],
                global_y[2] - cls._dot(global_y, d) * d[2],
            )
            local_y = cls._normalize(proj_y)
            if local_y == (0.0, 0.0, 0.0):
                local_y = (1.0, 0.0, 0.0)
            local_z = cls._normalize(cls._cross(d, local_y))
            return local_y, local_z

        local_y = cls._normalize(cls._cross(local_z, d))
        if local_y == (0.0, 0.0, 0.0):
            local_y = (0.0, 1.0, 0.0)
        return local_y, local_z

    @staticmethod
    def _x_bracing_layer_offset(
        member_group: str,
        ni: Node,
        nj: Node,
        *,
        stick_width_mm: float = 7.0,
        stick_thickness_mm: float = 1.5,
        detail: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve colisões físicas em contraventamentos em X.

        O modelo estrutural usa barras no eixo dos nós.  Em um X real feito com
        palitos, duas diagonais que se cruzam não podem ocupar o mesmo plano no
        ponto médio.  Há duas soluções montáveis: cortar uma diagonal e colar a
        ponta na face da outra, ou manter ambas contínuas em camadas diferentes
        (uma "na frente" e outra "atrás").  Para não criar uma conexão de nó
        central que o solver não calcula, adotamos a segunda opção por padrão:
        camadas alternadas, sem transferência de força no cruzamento.

        Isso não adiciona material nem resistência; apenas desloca a posição
        peça-a-peça e deixa explícito que a colagem resistente continua sendo
        nos nós/extremidades.  A separação é da ordem da espessura do palito.
        """
        group = str(member_group or "")
        policy = str(detail.get("x_bracing_crossing_policy", "layered_x_no_interpenetration")).strip().lower()
        secondary_x_groups = ["diagonal", "bottom_bracing", "top_bracing", "cross_frame_bracing"]
        if "x_bracing_no_crossing_groups" in detail:
            no_cross_source = detail.get("x_bracing_no_crossing_groups") or []
        else:
            no_cross_source = (
                secondary_x_groups
                if policy in {
                    "single_diagonal_no_crossing",
                    "single_diagonal",
                    "convert_to_single_diagonal",
                    "warren_no_crossing",
                    "split_midpoint_lap_joint",
                    "split_midpoint",
                    "midpoint_lap",
                    "x_midpoint_lap",
                }
                else []
            )
        no_cross_groups = set(str(v) for v in no_cross_source)
        if group in no_cross_groups and policy in {
            "split_midpoint_lap_joint",
            "split_midpoint",
            "midpoint_lap",
            "x_midpoint_lap",
        }:
            return {
                "offset": (0.0, 0.0, 0.0),
                "layer": 0,
                "plane": "midpoint_lap",
                "handling": "split_midpoint_lap_joint",
                "midspan_connected": True,
            }
        if group in no_cross_groups and policy in {
            "single_diagonal_no_crossing",
            "single_diagonal",
            "convert_to_single_diagonal",
            "warren_no_crossing",
        }:
            return {
                "offset": (0.0, 0.0, 0.0),
                "layer": 0,
                "plane": "single_diagonal",
                "handling": "single_diagonal_no_crossing",
                "midspan_connected": False,
            }

        layer_groups = set(str(v) for v in (detail.get("x_bracing_layered_groups") or ["diagonal", "bottom_bracing", "top_bracing", "cross_frame_bracing"]))
        if group not in layer_groups:
            return {
                "offset": (0.0, 0.0, 0.0),
                "layer": 0,
                "plane": "",
                "handling": "not_x_bracing",
                "midspan_connected": False,
            }

        gap = max(0.0, float(detail.get("x_bracing_layer_clearance_mm", 0.30)))
        # A separação precisa considerar a maior dimensão transversal real do
        # palito.  A versão anterior usava apenas a espessura (1,5 mm); em
        # diagonais flat a largura de 7 mm continuava se interpenetrando no
        # cruzamento em X.
        physical_depth = max(float(stick_width_mm or 0.0), float(stick_thickness_mm or 0.0))
        sep = max(0.1, physical_depth + gap)
        off = 0.5 * sep

        dx = float(nj.x - ni.x)
        dy = float(nj.y - ni.y)
        dz = float(nj.z - ni.z)

        if group == "diagonal":
            # Treliças laterais em X ficam no plano x-z de cada face.  Separar
            # as diagonais por camadas em y evita que dois palitos ocupem o
            # mesmo volume no cruzamento do painel sem inventar um nó central
            # que o solver estrutural não calcula.
            sign = 1.0 if dx * dz >= 0.0 else -1.0
            side = -1.0 if ((float(ni.y) + float(nj.y)) * 0.5) < 0.0 else 1.0
            return {
                "offset": (0.0, side * sign * off, 0.0),
                "layer": int(sign),
                "plane": "side_xz",
                "handling": "alternate_front_back_layer_no_midspan_joint",
                "midspan_connected": False,
            }
        if group == "bottom_bracing":
            # Plano x-y; separar em z.  Sinal alterna entre / e \\.
            sign = 1.0 if dx * dy >= 0.0 else -1.0
            return {
                "offset": (0.0, 0.0, sign * off),
                "layer": int(sign),
                "plane": "bottom_xy",
                "handling": "alternate_front_back_layer_no_midspan_joint",
                "midspan_connected": False,
            }
        if group == "top_bracing":
            # Plano x-y no banzo superior; separar também em z, mas com sinal
            # invertido em relação ao fundo para não empilhar todas as camadas
            # no mesmo lado visual da estrutura.
            sign = -1.0 if dx * dy >= 0.0 else 1.0
            return {
                "offset": (0.0, 0.0, sign * off),
                "layer": int(sign),
                "plane": "top_xy",
                "handling": "alternate_front_back_layer_no_midspan_joint",
                "midspan_connected": False,
            }
        if group == "cross_frame_bracing":
            # Plano y-z; separar em x.  Sinal alterna entre / e \\.
            sign = 1.0 if dy * dz >= 0.0 else -1.0
            return {
                "offset": (sign * off, 0.0, 0.0),
                "layer": int(sign),
                "plane": "crossframe_yz",
                "handling": "alternate_front_back_layer_no_midspan_joint",
                "midspan_connected": False,
            }
        return {
            "offset": (0.0, 0.0, 0.0),
            "layer": 0,
            "plane": "",
            "handling": "not_layered_by_rule",
            "midspan_connected": False,
        }

    @staticmethod
    def _piece_intervals(
        L: float,
        stick_len: float,
        overlap: float,
        min_constructive_piece_length_mm: float = 0.0,
    ) -> List[Tuple[float, float, float]]:
        """Divide um membro de comprimento ``L`` em peças fabricáveis.

        Se ``overlap`` for positivo, mantém a emenda sobreposta tradicional.
        Se ``overlap`` for zero, usa emenda topo-a-topo com talas: as peças
        principais ficam adjacentes e não ocupam o mesmo volume físico. Esse
        caso é essencial para ``splice_mode=butt_with_splints``.
        """
        if L <= 0:
            return []

        if stick_len <= 0:
            raise ValueError("stick_length_mm precisa ser maior que zero.")

        if L <= stick_len:
            return [(0.0, L, L)]

        overlap = max(0.0, min(overlap, stick_len * 0.75))

        if overlap <= 1.0e-9:
            n = max(1, int(math.ceil(L / stick_len)))
            base = L / n
            out: List[Tuple[float, float, float]] = []
            s0 = 0.0
            for idx in range(n):
                s1 = L if idx == n - 1 else min(L, s0 + base)
                if s1 > s0 + 1.0e-9:
                    out.append((s0, s1, s1 - s0))
                s0 = s1
            return out

        step = max(1.0e-6, stick_len - overlap)

        out: List[Tuple[float, float, float]] = []
        s0 = 0.0

        while s0 < L - 1.0e-9:
            s1 = min(L, s0 + stick_len)
            out.append((s0, s1, s1 - s0))

            if s1 >= L - 1.0e-9:
                break

            s0 += step

        return StickDetailService._enforce_min_constructive_piece_length(
            out,
            min_constructive_piece_length_mm=min_constructive_piece_length_mm,
            domain_start_mm=0.0,
            domain_end_mm=L,
            stock_limit_mm=stick_len,
        )

    @staticmethod
    def _enforce_min_constructive_piece_length(
        intervals: List[Tuple[float, float, float]],
        *,
        min_constructive_piece_length_mm: float,
        domain_start_mm: float,
        domain_end_mm: float,
        stock_limit_mm: float,
    ) -> List[Tuple[float, float, float]]:
        """Avoid terminal stick fragments that are too short to fabricate reliably.

        The geometry can generate a final remainder of only a few millimetres
        when a member is just longer than one stock stick.  Instead of adding a
        fragile sliver, grow the short terminal piece back into the previous
        piece, increasing the overlap.  The structural member axis is unchanged;
        only the fabrication intervals are made constructible.
        """
        if not intervals:
            return []

        min_len = max(0.0, float(min_constructive_piece_length_mm or 0.0))
        stock_limit = max(1.0, float(stock_limit_mm))
        if min_len <= 1.0e-9 or len(intervals) <= 1:
            return [(float(a), float(b), max(0.0, float(b) - float(a))) for a, b, _ in intervals]

        min_len = min(min_len, stock_limit)
        lo = float(domain_start_mm)
        hi = float(domain_end_mm)
        out = [(float(a), float(b), max(0.0, float(b) - float(a))) for a, b, _ in intervals]

        def resize_first(row: Tuple[float, float, float]) -> Tuple[float, float, float]:
            a, b, _ = row
            if b - a >= min_len - 1.0e-9:
                return row
            b2 = min(hi, a + min_len)
            if b2 - a > stock_limit + 1.0e-9:
                b2 = a + stock_limit
            return (a, b2, max(0.0, b2 - a))

        def resize_last(row: Tuple[float, float, float]) -> Tuple[float, float, float]:
            a, b, _ = row
            if b - a >= min_len - 1.0e-9:
                return row
            a2 = max(lo, b - min_len)
            if b - a2 > stock_limit + 1.0e-9:
                a2 = b - stock_limit
            return (a2, b, max(0.0, b - a2))

        out[0] = resize_first(out[0])
        out[-1] = resize_last(out[-1])

        # Rare intermediate fragments can appear after midpoint splitting. Grow
        # them symmetrically within the allowed domain.
        fixed: List[Tuple[float, float, float]] = []
        for idx, (a, b, _c) in enumerate(out):
            length = b - a
            if 0 < idx < len(out) - 1 and length < min_len - 1.0e-9:
                need = min_len - length
                a = max(lo, a - 0.5 * need)
                b = min(hi, b + 0.5 * need)
                if b - a < min_len - 1.0e-9:
                    if a <= lo + 1.0e-9:
                        b = min(hi, a + min_len)
                    elif b >= hi - 1.0e-9:
                        a = max(lo, b - min_len)
                if b - a > stock_limit + 1.0e-9:
                    b = a + stock_limit
            fixed.append((a, b, max(0.0, b - a)))

        return fixed


    @staticmethod
    def _force_butt_nonoverlap_intervals(
        intervals: List[Tuple[float, float, float]],
        *,
        domain_start_mm: float,
        domain_end_mm: float,
    ) -> List[Tuple[float, float, float]]:
        """Clamp butt-splice intervals so adjacent main sticks never overlap.

        Minimum-length repair and lane staggering may move a terminal boundary
        backwards.  That is acceptable for a lap-splice model, but in
        butt-with-splints it creates two real prisms in the same volume and a
        false tiny glue overlap.  This cleanup preserves ordering and removes
        the overlap; splints carry the continuity instead of the main sticks.
        """
        if not intervals:
            return []
        lo = float(domain_start_mm)
        hi = float(domain_end_mm)
        cleaned: List[Tuple[float, float, float]] = []
        prev_end = lo
        for a, b, _c in sorted((float(a), float(b), float(c)) for a, b, c in intervals):
            aa = max(lo, min(hi, max(a, prev_end)))
            bb = max(lo, min(hi, b))
            if bb <= aa + 1.0e-9:
                continue
            cleaned.append((aa, bb, bb - aa))
            prev_end = bb
        return cleaned

    @staticmethod
    def _joint_setback_groups(detail: Dict[str, Any]) -> set[str]:
        raw = detail.get(
            "joint_setback_groups",
            [
                "top_chord",
                "bottom_chord",
                "diagonal",
                "vertical",
                "top_transverse",
                "bottom_transverse",
                "top_bracing",
                "bottom_bracing",
                "cross_frame_bracing",
                "chord_lacing",
            ],
        )
        return {str(v) for v in (raw or [])}

    @staticmethod
    def _section_envelope_mm(sec: Dict[str, Any], stick_w: float, stick_t: float) -> float:
        width = safe_float(sec.get("width_mm"), None)
        thickness = safe_float(sec.get("thickness_mm"), None)
        return max(
            float(stick_w),
            float(stick_t),
            float(width) if width is not None else 0.0,
            float(thickness) if thickness is not None else 0.0,
        )

    @staticmethod
    def _joint_face_setbacks(
        member: Member,
        member_length_mm: float,
        *,
        node_member_envelopes: Dict[int, List[Tuple[int, float]]],
        detail: Dict[str, Any],
        stick_w: float,
        stick_t: float,
        min_constructive_piece_length_mm: float,
    ) -> Tuple[float, float]:
        """Return start/end trims so pieces stop at the joint contact face.

        The solver keeps the member centreline from node to node, which is the
        correct truss idealisation.  The fabrication view, however, must not draw
        the physical prism all the way through the node centroid.  For non-chord
        members we trim the rendered/fabricated interval by the half-envelope of
        the other members meeting that node.
        """
        if not bool(detail.get("joint_face_setback_enabled", True)):
            return 0.0, 0.0
        group = str(member.group)
        if group not in StickDetailService._joint_setback_groups(detail):
            return 0.0, 0.0

        # Membros colados por face lateral (montantes, diagonais, transversais e
        # contraventamentos) não devem ser encurtados axialmente até antes do nó:
        # eles precisam encostar/ultrapassar a face do banzo hospedeiro para haver
        # área real de cola. O recuo axial fazia a peça parecer "flutuando" com
        # um vão visível na montagem. A separação volumétrica desses membros é
        # tratada pelo offset físico de camada, não por um gap no eixo.
        if bool(detail.get("side_lap_groups_skip_axis_setback", True)):
            no_axis_setback = {
                str(v)
                for v in detail.get(
                    "side_lap_no_axis_setback_groups",
                    [
                        "vertical",
                        "diagonal",
                        "top_transverse",
                        "bottom_transverse",
                        "top_bracing",
                        "bottom_bracing",
                        "cross_frame_bracing",
                        "chord_lacing",
                    ],
                )
            }
            if group in no_axis_setback and group in StickDetailService._node_lap_groups(detail):
                return 0.0, 0.0

        clearance = max(0.0, float(detail.get("joint_face_clearance_mm", 0.50)))
        min_setback = max(0.0, float(detail.get("joint_min_setback_mm", 0.50 * max(stick_w, stick_t))))
        max_setback_mm = max(0.0, float(detail.get("joint_max_setback_mm", 18.0)))
        max_setback_fraction = max(0.0, min(0.45, float(detail.get("joint_max_setback_fraction", 0.18))))
        per_end_cap = min(max_setback_mm, max_setback_fraction * max(0.0, float(member_length_mm)))

        def end_setback(node_id: int) -> float:
            others = [env for mid, env in node_member_envelopes.get(int(node_id), []) if int(mid) != int(member.id)]
            other_env = max(others) if others else max(float(stick_w), float(stick_t))
            # Para a vista de encaixe e a fabricação, o recuo terminal deve
            # aproximar a peça da face de contato do host, não da metade da
            # caixa composta inteira. Usar o envelope máximo do banzo superior
            # gerava lacunas visuais grandes e fazia montantes parecerem
            # flutuar. O modo padrão usa a maior face de um palito como
            # profundidade de contato; o envelope integral fica disponível para
            # diagnósticos conservadores.
            mode = str(detail.get("joint_face_contact_depth_mode", "stick_face")).strip().lower()
            if mode in {"stick_face", "single_stick_face", "min_stick_face"}:
                contact_depth = min(float(other_env), max(float(stick_w), float(stick_t)))
            elif mode in {"stick_thickness", "thin_face"}:
                contact_depth = min(float(other_env), min(float(stick_w), float(stick_t)))
            else:
                contact_depth = float(other_env)
            raw = max(min_setback, 0.5 * float(contact_depth) + clearance)
            return max(0.0, min(per_end_cap, raw))

        start = end_setback(member.i)
        end = end_setback(member.j)

        L = max(0.0, float(member_length_mm))
        # Always leave a useful clear stick span.  For very short members, scale
        # both end setbacks rather than erasing the piece.
        min_clear = min(max(1.0, float(min_constructive_piece_length_mm or 0.0)), max(1.0, 0.60 * L))
        max_total_setback = max(0.0, L - min_clear)
        if start + end > max_total_setback + 1.0e-9 and start + end > 1.0e-9:
            scale = max_total_setback / (start + end)
            start *= scale
            end *= scale
        return start, end


    @staticmethod
    def _node_lap_groups(detail: Dict[str, Any]) -> set[str]:
        raw = detail.get(
            "node_lap_groups",
            [
                "diagonal",
                "vertical",
                "top_transverse",
                "bottom_transverse",
                "top_bracing",
                "bottom_bracing",
                "cross_frame_bracing",
                "chord_lacing",
            ],
        )
        return {str(v) for v in (raw or [])}

    @staticmethod
    def _node_host_groups(detail: Dict[str, Any]) -> set[str]:
        raw = detail.get(
            "node_lap_host_groups",
            detail.get(
                "miter_cut_host_groups",
                [
                    "bottom_chord",
                    "top_chord",
                    "support_pad",
                    "vertical",
                    "diagonal",
                    "bottom_transverse",
                    "top_transverse",
                ],
            ),
        )
        return {str(v) for v in (raw or [])}

    @staticmethod
    def _member_group_by_id(members: List[Member]) -> Dict[int, str]:
        return {int(m.id): str(m.group) for m in (members or [])}

    @staticmethod
    def _has_node_host(
        node_id: int,
        member_id: int,
        *,
        node_member_envelopes: Dict[int, List[Tuple[int, float]]],
        member_group_by_id: Dict[int, str],
        host_groups: set[str],
    ) -> bool:
        for other_id, _env in node_member_envelopes.get(int(node_id), []) or []:
            if int(other_id) == int(member_id):
                continue
            if str(member_group_by_id.get(int(other_id), "")) in host_groups:
                return True
        return False

    @staticmethod
    def _round_to_increment(value: float, increment: float) -> float:
        inc = max(1.0e-9, float(increment or 1.0))
        return round(float(value) / inc) * inc

    @classmethod
    def _end_cut_angle_deg(
        cls,
        ux: float,
        uy: float,
        uz: float,
        *,
        detail: Dict[str, Any],
    ) -> float:
        """Legacy fallback for a terminal bevel angle.

        This method is intentionally conservative and is now used only when no
        host member can be identified at the node.  Earlier revisions applied
        its result to every segment of a multi-piece member; that made internal
        splice edges look like random diagonal cuts.  Terminal mitering is now
        decided per end by :meth:`_terminal_end_cut_angle_deg`.
        """
        if not bool(detail.get("angled_end_cuts_enabled", True)):
            return 90.0
        horiz = math.hypot(float(ux), float(uy))
        angle = math.degrees(math.atan2(abs(float(uz)), max(1.0e-9, horiz)))
        cut_angle = max(15.0, min(90.0, 90.0 - angle))
        inc = max(1.0, float(detail.get("end_cut_angle_increment_deg", 5.0)))
        return max(15.0, min(90.0, cls._round_to_increment(cut_angle, inc)))

    @classmethod
    def _terminal_end_cut_spec(
        cls,
        *,
        member: Member,
        node_id: int,
        nodes_by_id: Dict[int, Node],
        members_by_id: Dict[int, Member],
        node_member_envelopes: Dict[int, List[Tuple[int, float]]],
        member_group_by_id: Dict[int, str],
        detail: Dict[str, Any],
        fallback_axis: Tuple[float, float, float],
        local_bevel_axis: Tuple[float, float, float] | None = None,
        local_width_axis: Tuple[float, float, float] | None = None,
    ) -> Dict[str, Any]:
        """Return the terminal miter geometry for a real host contact.

        A miter cut is a property of the *end* of a stick glued against an
        inclined host face. It is not a property of every internal splice. The
        returned ``skew_sign`` tells the visualizer which local side of the
        rectangular section must be shortened; without this sign the numeric
        angle can be correct while the drawn cut appears inverted.
        """
        base: Dict[str, Any] = {
            "angle_deg": 90.0,
            "skew_sign": 1.0,
            "host_member_id": None,
            "host_group": "",
            "host_relation": "none",
            "trim_axis": "z",
        }
        if not bool(detail.get("angled_end_cuts_enabled", True)):
            return base

        allowed_groups = {
            str(v)
            for v in detail.get(
                "miter_cut_terminal_groups",
                ["vertical", "diagonal", "top_transverse", "bottom_transverse"],
            )
        }
        if str(member.group) not in allowed_groups:
            return base

        host_groups = cls._node_host_groups(detail)
        host_candidates: List[Tuple[float, Member]] = []
        for other_id, env in node_member_envelopes.get(int(node_id), []) or []:
            if int(other_id) == int(member.id):
                continue
            if str(member_group_by_id.get(int(other_id), "")) not in host_groups:
                continue
            other = members_by_id.get(int(other_id))
            if other is not None:
                host_candidates.append((float(env), other))

        if not host_candidates:
            base["angle_deg"] = cls._end_cut_angle_deg(*fallback_axis, detail=detail)
            base["host_relation"] = "fallback_axis"
            return base

        priority = [
            str(v)
            for v in detail.get(
                "miter_cut_primary_host_priority",
                ["top_chord", "bottom_chord", "support_pad", "vertical", "diagonal", "top_transverse", "bottom_transverse"],
            )
        ]
        prio = {g: i for i, g in enumerate(priority)}
        host = sorted(
            host_candidates,
            key=lambda item: (prio.get(str(member_group_by_id.get(int(item[1].id), "")), 999), -float(item[0])),
        )[0][1]
        hn0 = nodes_by_id.get(int(host.i))
        hn1 = nodes_by_id.get(int(host.j))
        mn0 = nodes_by_id.get(int(member.i))
        mn1 = nodes_by_id.get(int(member.j))
        if hn0 is None or hn1 is None or mn0 is None or mn1 is None:
            return base
        hux, huy, huz, hL = cls._unit_vector(hn0, hn1)
        mux, muy, muz, mL = cls._unit_vector(mn0, mn1)
        if hL <= 1.0e-9 or mL <= 1.0e-9:
            return base

        threshold = max(0.0, float(detail.get("miter_cut_min_host_slope_deg", 7.5)))
        inc = max(1.0, float(detail.get("end_cut_angle_increment_deg", 5.0)))

        dot = abs(float(mux) * float(hux) + float(muy) * float(huy) + float(muz) * float(huz))
        dot = max(0.0, min(1.0, dot))
        rel_angle = math.degrees(math.acos(dot))
        host_group = str(member_group_by_id.get(int(host.id), host.group))
        member_group = str(member.group)
        if member_group == host_group and member_group in {"top_chord", "bottom_chord"} and threshold <= rel_angle <= 84.0:
            # Dois segmentos de banzo que se encontram em quina devem usar
            # meia-esquadria simples. Aplicar o ângulo relativo inteiro gerava
            # cortes extremamente agudos e visualmente anômalos.
            angle = max(45.0, min(90.0, cls._round_to_increment(90.0 - 0.5 * rel_angle, inc)))
            relation = "same_chord_half_miter"
        elif threshold <= rel_angle <= 84.0:
            angle = max(35.0, min(90.0, cls._round_to_increment(rel_angle, inc)))
            relation = "relative_axis"
        else:
            host_slope = math.degrees(math.atan2(abs(float(huz)), max(1.0e-9, math.hypot(float(hux), float(huy)))))
            if host_slope < threshold:
                angle = 90.0
                relation = "square_contact"
            else:
                angle = max(35.0, min(90.0, cls._round_to_increment(90.0 - host_slope, inc)))
                relation = "host_slope"

        # Determine which local side of the stick should be shortened.  The host
        # line has no inherent orientation, so h and -h must produce the same
        # result; using the product of the projections preserves that invariance.
        # A single hard-coded bevel axis was the source of the wrong cuts: some
        # joints need a cut across the face (section z), others across the edge
        # (section y).  Pick the local axis most aligned with the host slope.
        axis_z = local_bevel_axis or (0.0, 0.0, 0.0)
        zx, zy, zz = cls._normalize((float(axis_z[0]), float(axis_z[1]), float(axis_z[2])))
        axis_y = local_width_axis or (0.0, 0.0, 0.0)
        yx, yy, yz = cls._normalize((float(axis_y[0]), float(axis_y[1]), float(axis_y[2])))
        if (zx, zy, zz) == (0.0, 0.0, 0.0):
            zx, zy, zz = 1.0, 0.0, 0.0
        hz_proj = hux * zx + huy * zy + huz * zz
        hy_proj = hux * yx + huy * yy + huz * yz if (yx, yy, yz) != (0.0, 0.0, 0.0) else 0.0
        if abs(hy_proj) > 1.20 * abs(hz_proj):
            bx, by, bz = yx, yy, yz
            hb = hy_proj
            trim_axis = "y"
        else:
            bx, by, bz = zx, zy, zz
            hb = hz_proj
            trim_axis = "z"
        if (bx, by, bz) == (0.0, 0.0, 0.0):
            bx, by, bz = 1.0, 0.0, 0.0
            hb = hux
            trim_axis = "z"
        hd = hux * mux + huy * muy + huz * muz
        skew_sign = 1.0
        if abs(hd) > 1.0e-9 and abs(hb) > 1.0e-9:
            skew_sign = 1.0 if (hd * hb) >= 0.0 else -1.0
        if bool(detail.get("miter_cut_visual_flip_sign", False)):
            skew_sign *= -1.0

        return {
            "angle_deg": angle,
            "skew_sign": skew_sign,
            "trim_axis": trim_axis,
            "host_member_id": int(host.id),
            "host_group": host_group,
            "host_relation": relation,
        }

    @classmethod
    def _terminal_end_cut_angle_deg(
        cls,
        *,
        member: Member,
        node_id: int,
        nodes_by_id: Dict[int, Node],
        members_by_id: Dict[int, Member],
        node_member_envelopes: Dict[int, List[Tuple[int, float]]],
        member_group_by_id: Dict[int, str],
        detail: Dict[str, Any],
        fallback_axis: Tuple[float, float, float],
    ) -> float:
        """Compatibility wrapper returning only the terminal miter angle."""
        spec = cls._terminal_end_cut_spec(
            member=member,
            node_id=node_id,
            nodes_by_id=nodes_by_id,
            members_by_id=members_by_id,
            node_member_envelopes=node_member_envelopes,
            member_group_by_id=member_group_by_id,
            detail=detail,
            fallback_axis=fallback_axis,
            local_bevel_axis=None,
        )
        return float(spec.get("angle_deg", 90.0) or 90.0)

    @staticmethod
    def _node_lap_visual_side_offset(
        *,
        member: Member,
        ni: Node,
        nj: Node,
        detail: Dict[str, Any],
    ) -> Tuple[float, float, float]:
        """Return the real assembly layer offset for terminal face-lap members.

        The solver member remains node-to-node, but the physical stick must sit
        on an external glue face instead of passing through the centroidal volume
        of the host.  This offset is therefore part of the as-built geometry, not
        merely a rendering trick.  It is deterministic by group so two generated
        runs with the same topology keep the same jig/assembly layers.
        """
        if not bool(detail.get("node_lap_visual_side_offset_enabled", True)):
            return (0.0, 0.0, 0.0)
        groups = {
            str(v)
            for v in detail.get(
                "node_lap_visual_side_offset_groups",
                [
                    "vertical",
                    "diagonal",
                    "top_transverse",
                    "bottom_transverse",
                    "top_bracing",
                    "bottom_bracing",
                    "cross_frame_bracing",
                    "support_pad",
                ],
            )
        }
        group = str(member.group)
        if group not in groups:
            return (0.0, 0.0, 0.0)
        base = max(0.0, float(detail.get("node_lap_visual_side_offset_mm", 4.0)))
        cap = max(base, float(detail.get("node_lap_visual_side_offset_max_mm", 8.0) or 8.0))
        y_mid = 0.5 * (float(ni.y) + float(nj.y))
        x_mid = 0.5 * (float(ni.x) + float(nj.x))

        g = str(member.group).strip().lower()
        if g in {"vertical", "diagonal", "top_bracing", "bottom_bracing"}:
            if abs(y_mid) <= 1.0e-6:
                return (0.0, 0.0, 0.0)
            if g == "vertical":
                mult = 0.60
            elif g == "diagonal":
                # A diagonal fica em uma camada externa ao montante. A folga
                # precisa ser maior que a espessura do palito para evitar colisão,
                # mas ainda pequena o bastante para parecer montagem encaixada.
                mult = 1.80 + (0.20 if int(getattr(member, "id", 0) or 0) % 2 else -0.20)
            else:
                mult = 1.70
            return (0.0, math.copysign(min(cap, base * mult), y_mid), 0.0)
        if g in {"top_transverse", "bottom_transverse"}:
            # Transversais precisam sair do plano do banzo, mas só o bastante
            # para representar cola em face. Valores grandes deixam a vista
            # montada parecer explodida.
            sign = 1.0 if g == "top_transverse" else -1.0
            return (0.0, 0.0, sign * min(cap, base * 0.75))
        if g == "support_pad":
            return (0.0, 0.0, -min(cap, base * 1.70))
        if g == "cross_frame_bracing":
            sign = 1.0 if x_mid >= 0.5 * float(detail.get("bridge_span_reference_mm", 0.0) or 0.0) else -1.0
            if abs(x_mid) <= 1.0e-6:
                sign = 1.0
            return (sign * min(cap, base * 1.20), 0.0, 0.0)
        return (0.0, 0.0, 0.0)

    @staticmethod
    def _miter_material_loss_length_mm(
        angle_deg: float | None,
        *,
        material_depth_mm: float,
        piece_length_mm: float,
        detail: Dict[str, Any],
    ) -> float:
        """Equivalent stick length removed by a single terminal miter cut.

        The shop cut length remains rounded to the configured 5 mm increment;
        this estimate only subtracts the small triangular wedge that is no
        longer part of the bridge after the bevel.  It prevents the mass model
        from counting material that the visual model explicitly removed.
        """
        if not bool(detail.get("miter_cut_mass_reduction_enabled", True)):
            return 0.0
        try:
            a = float(angle_deg)
        except (TypeError, ValueError):
            return 0.0
        if a >= 89.5:
            return 0.0
        a = max(15.0, min(89.0, a))
        depth = max(0.0, float(material_depth_mm or 0.0))
        if depth <= 1.0e-9:
            return 0.0
        raw_shift = depth / max(1.0e-9, math.tan(math.radians(a)))
        max_fraction = max(0.0, min(0.30, float(detail.get("miter_cut_max_visual_shift_fraction", 0.10))))
        shift = min(max_fraction * max(0.0, float(piece_length_mm)), raw_shift)
        loss_factor = max(0.0, min(1.0, float(detail.get("miter_cut_material_loss_factor", 0.50))))
        return loss_factor * shift

    @staticmethod
    def _terminal_lap_overlap_mm(detail: Dict[str, Any], stick_len: float) -> float:
        requested = float(detail.get("node_lap_overlap_mm", detail.get("overlap_length_mm", 30.0)))
        minimum = float(detail.get("min_node_lap_overlap_mm", 0.18 * float(stick_len)))
        return max(8.0, min(0.85 * float(stick_len), max(minimum, requested)))

    @classmethod
    def _terminal_connection_mode(
        cls,
        *,
        member: Member,
        node_id: int,
        detail: Dict[str, Any],
        node_member_envelopes: Dict[int, List[Tuple[int, float]]],
        member_group_by_id: Dict[int, str],
    ) -> str:
        if not bool(detail.get("node_face_lap_enabled", True)):
            return "axis_centroid"
        if str(member.group) not in cls._node_lap_groups(detail):
            return "axis_centroid"
        if cls._has_node_host(
            int(node_id),
            int(member.id),
            node_member_envelopes=node_member_envelopes,
            member_group_by_id=member_group_by_id,
            host_groups=cls._node_host_groups(detail),
        ):
            return "face_lap_to_host"
        return "tip_to_node_no_host"

    @staticmethod
    def _split_interval_to_stock_limit(
        s0: float,
        s1: float,
        *,
        max_cut_mm: float,
        overlap_mm: float,
        cut_increment_mm: float,
        min_constructive_piece_length_mm: float = 0.0,
    ) -> List[Tuple[float, float, float]]:
        """Reparte um intervalo físico para nenhum corte exceder o palito real."""
        a = float(s0)
        b = float(s1)
        if b <= a + 1.0e-9:
            return []
        max_cut = max(1.0, float(max_cut_mm))
        overlap = max(0.0, min(float(overlap_mm), 0.75 * max_cut))
        inc = max(1.0, float(cut_increment_mm))
        out: List[Tuple[float, float, float]] = []
        cur = a
        while cur < b - 1.0e-9:
            nxt = min(b, cur + max_cut)
            if nxt < b - 1.0e-9:
                rounded = math.floor(nxt / inc) * inc
                if rounded > cur + max(5.0, 0.30 * max_cut):
                    nxt = rounded
            out.append((cur, nxt, nxt - cur))
            if nxt >= b - 1.0e-9:
                break
            cur = max(cur + 1.0, nxt - overlap)
        return StickDetailService._enforce_min_constructive_piece_length(
            out,
            min_constructive_piece_length_mm=min_constructive_piece_length_mm,
            domain_start_mm=a,
            domain_end_mm=b,
            stock_limit_mm=max_cut,
        )

    @classmethod
    def _enforce_stock_limit_on_intervals(
        cls,
        intervals: List[Tuple[float, float, float]],
        *,
        max_cut_mm: float,
        overlap_mm: float,
        cut_increment_mm: float,
        min_constructive_piece_length_mm: float = 0.0,
    ) -> tuple[List[Tuple[float, float, float]], int]:
        fixed: List[Tuple[float, float, float]] = []
        splits = 0
        for s0, s1, _cut in intervals:
            parts = cls._split_interval_to_stock_limit(
                float(s0),
                float(s1),
                max_cut_mm=max_cut_mm,
                overlap_mm=overlap_mm,
                cut_increment_mm=cut_increment_mm,
                min_constructive_piece_length_mm=min_constructive_piece_length_mm,
            )
            if len(parts) > 1:
                splits += 1
            fixed.extend(parts)
        return fixed, splits

    @staticmethod
    def _pack_cuts_best_fit(
        cuts: List[float],
        blank_length: float,
        kerf: float = 1.0,
    ) -> List[List[float]]:
        """
        Empacota cortes em palitos brutos usando heurística best-fit decrescente.
        Não é otimização exata, mas é rápida e suficiente para plano preliminar.
        """
        clean_cuts = [
            float(c)
            for c in cuts
            if safe_float(c, None) is not None and float(c) > 0
        ]

        clean_cuts = sorted(clean_cuts, reverse=True)

        bins: List[List[float]] = []
        remaining: List[float] = []

        for c in clean_cuts:
            best_i = None
            best_rem = None

            for i, rem in enumerate(remaining):
                need = c + (kerf if bins[i] else 0.0)

                if need <= rem + 1.0e-9:
                    nr = rem - need

                    if best_rem is None or nr < best_rem:
                        best_i = i
                        best_rem = nr

            if best_i is None:
                bins.append([c])
                remaining.append(max(0.0, blank_length - c))
            else:
                bins[best_i].append(c)
                remaining[best_i] = best_rem if best_rem is not None else remaining[best_i]

        return bins

    def analyze(
        self,
        cfg: Dict,
        nodes: List[Node],
        members: List[Member],
        member_results: List[Dict],
        member_checks: List[Dict],
        out_dir: str | Path,
    ) -> Dict:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        mat = cfg["material"]
        detail = cfg.get("detail_model", {})

        stick_len = float(mat.get("stick_length_mm", 120.0))
        stick_w = float(mat.get("stick_width_mm", 7.0))
        stick_t = float(mat.get("stick_thickness_mm", 1.5))
        stick_mass = float(mat.get("stick_mass_g", 1.4))

        overlap = float(detail.get("overlap_length_mm", 30.0))
        glue_tau = float(detail.get("glue_shear_strength_MPa", 3.5))
        glue_sf = float(detail.get("default_joint_safety_factor", 2.0))
        min_end_margin = max(0.0, float(detail.get("min_end_margin_mm", 10.0)))
        cut_increment_mm = max(0.5, float(detail.get("cut_increment_mm", 5.0)))
        allow_cut_rounding = bool(detail.get("allow_cut_rounding", True))
        min_cut_length_mm = max(1.0, float(detail.get("min_cut_length_mm", 5.0)))
        min_constructive_piece_length_mm = max(
            min_cut_length_mm,
            float(detail.get("min_constructive_piece_length_mm", 20.0)),
        )
        max_cut_length_mm = min(
            stick_len,
            max(1.0, float(detail.get("max_cut_length_mm", stick_len))),
        )
        strict_cut_length = bool(detail.get("strict_cut_length", True))
        stock_limit_splits = 0
        # global default joint models.  These may be overridden per member by
        # a connection planner via ``cfg['member_joint_plan']``.
        tension_joint_model = str(detail.get("tension_joint_model", "double_lap_reinforced"))
        compression_joint_model = str(detail.get("compression_joint_model", "double_lap_reinforced"))
        glue_spread = float(detail.get("glue_spread_g_per_m2", 160.0))
        glue_eff = float(detail.get("glue_mass_efficiency", 0.65))
        glue_cure_solids_fraction = max(
            0.30,
            min(0.80, float(detail.get("glue_cure_solids_fraction", 0.50))),
        )
        imperfection_e = float(detail.get("imperfection_eccentricity_mm", 2.0))
        waste = float(detail.get("construction_waste_factor", 0.08))
        kerf = float(detail.get("saw_kerf_mm", 1.0))
        reinforce_if = float(detail.get("reinforce_if_fs_lt", 2.0))
        remove_if = float(detail.get("allow_recommend_removal_if_fs_gt", 8.0))
        tension_only = bool(detail.get("tension_only_stabilizers", True))
        terminal_joint_area_factor = max(0.1, float(detail.get("terminal_joint_area_factor", 1.35)))
        terminal_joint_secondary_bending_factor = max(0.5, float(detail.get("terminal_joint_secondary_bending_factor", 1.05)))
        splice_mode = str(detail.get("splice_mode", "overlap")).strip().lower()
        use_butt_splints = splice_mode in {"butt_with_splints", "butt_splints", "butt_full_splints"}
        splint_len = max(12.0, min(0.85 * stick_len, float(detail.get("reinforcement_length_mm", detail.get("overlap_length_mm", 25.0)))))
        splints_per_splice = max(1, int(detail.get("reinforcement_sticks_per_splice", 1)))

        node_by_id = {n.id: n for n in nodes}
        res_by = {int(r["member_id"]): r for r in member_results}
        chk_by = {int(r["member_id"]): r for r in member_checks}
        sizing_map = cfg.get("member_sizing_plan_by_id", {}) or {}

        stabilizers = set(cfg.get("analysis", {}).get("stabilizer_groups", []))

        node_member_envelopes: Dict[int, List[Tuple[int, float]]] = {}
        member_group_by_id = self._member_group_by_id(members)
        members_by_id = {int(mm.id): mm for mm in members}
        for mm in members:
            mm_layout_cfg = dict(
                cfg.get("section_layout_by_group", {}).get(
                    mm.group,
                    {"layout": "stacked"},
                )
            )
            mm_layout_cfg.setdefault(
                "composite_action",
                detail.get("composite_action", {}),
            )
            mm_sec = self.sections.composite_section(max(1, int(mm.n_sticks)), mat, mm_layout_cfg)
            env = self._section_envelope_mm(mm_sec, stick_w, stick_t)
            node_member_envelopes.setdefault(int(mm.i), []).append((int(mm.id), env))
            node_member_envelopes.setdefault(int(mm.j), []).append((int(mm.id), env))

        stick_rows: List[Dict] = []
        joint_rows: List[Dict] = []
        member_rows: List[Dict] = []
        reinf_rows: List[Dict] = []

        cut_counter: Counter = Counter()
        cut_lengths: List[float] = []

        total_glue_area = 0.0
        total_glue_physical_area = 0.0
        total_pieces = 0
        total_cut = 0.0
        total_splint_sticks_equiv = 0.0
        total_splint_mass_g = 0.0

        # Determinar se estamos usando modelo de quarto de ponte.  Quando
        # `use_quarter_model` é verdadeiro e um valor de
        # `quarter_member_count` é fornecido, cada membro pode ser atribuído
        # a um dos quadrantes.  Isto permite alternar a orientação das emendas
        # por quadrante para reduzir alinhamentos contínuos.
        use_quarter_model = bool(cfg.get("analysis", {}).get("use_quarter_model", False))
        quarter_count = 0
        if use_quarter_model:
            try:
                quarter_count = int(cfg.get("analysis", {}).get("quarter_member_count", 0))
            except (TypeError, ValueError):
                quarter_count = 0

        for m in members:
            ni = node_by_id[m.i]
            nj = node_by_id[m.j]

            ux, uy, uz, L = self._unit_vector(ni, nj)

            if L <= 0:
                continue

            res = res_by.get(m.id, {})
            chk = chk_by.get(m.id, {})
            sizing = sizing_map.get(str(m.id)) or sizing_map.get(m.id) or {}
            member_plan = (cfg.get("member_joint_plan", {}) or {}).get(m.id) or (cfg.get("member_joint_plan", {}) or {}).get(str(m.id))

            N = safe_float(res.get("N_N"), 0.0) or 0.0
            n_lanes = max(1, int(m.n_sticks))
            member_overlap = overlap
            if isinstance(member_plan, dict):
                planned_overlap = safe_float(member_plan.get("required_overlap_mm"), None)
                if planned_overlap is not None:
                    member_overlap = max(8.0, min(0.85 * stick_len, float(planned_overlap)))

            layout_cfg = cfg.get("section_layout_by_group", {}).get(
                m.group,
                {"layout": "stacked"},
            )

            layout_cfg_detail = dict(layout_cfg)
            layout_cfg_detail.setdefault(
                "composite_action",
                detail.get("composite_action", {}),
            )
            sec = self.sections.composite_section(n_lanes, mat, layout_cfg_detail)

            section_positions = list(sec.get("stick_positions_yz", []) or [])
            if len(section_positions) < n_lanes:
                section_positions.extend([(0.0, 0.0)] * (n_lanes - len(section_positions)))
            cy = safe_float(sec.get("centroid_y_mm"), 0.0) or 0.0
            cz = safe_float(sec.get("centroid_z_mm"), 0.0) or 0.0
            local_y_axis, local_z_axis = self._local_section_axes(ux, uy, uz)
            x_layer = self._x_bracing_layer_offset(
                m.group,
                ni,
                nj,
                stick_width_mm=stick_w,
                stick_thickness_mm=stick_t,
                detail=detail,
            )
            x_layer_offset = tuple(x_layer.get("offset", (0.0, 0.0, 0.0)))

            stick_orientation = str(sec.get("stick_orientation", layout_cfg_detail.get("stick_orientation", "flat"))).strip().lower()
            lane_orientations = list(sec.get("stick_orientations", []) or [])
            lane_widths = list(sec.get("stick_width_y_mm_by_lane", []) or [])
            lane_heights = list(sec.get("stick_height_z_mm_by_lane", []) or [])
            visual_width_mm = safe_float(sec.get("stick_width_y_mm"), None)
            visual_thickness_mm = safe_float(sec.get("stick_height_z_mm"), None)
            if visual_width_mm is None or visual_thickness_mm is None:
                if stick_orientation == "edge":
                    visual_width_mm = stick_t
                    visual_thickness_mm = stick_w
                else:
                    visual_width_mm = stick_w
                    visual_thickness_mm = stick_t

            per_lane = N / n_lanes
            piece_area = stick_w * stick_t
            per_sigma = per_lane / piece_area if piece_area else 0.0

            # Gera as subdivisões do membro em peças de palito.  Caso
            # seja um modelo de quarto, alternamos a orientação das
            # emendas em quadrantes ímpares para evitar alinhamento
            # perfeito de juntas nas quatro porções da ponte.  Para
            # quadrantes ímpares, invertimos a orientação das emendas (os
            # cortes passam a ser contados a partir da extremidade oposta).
            connection_start_mode = self._terminal_connection_mode(
                member=m,
                node_id=m.i,
                detail=detail,
                node_member_envelopes=node_member_envelopes,
                member_group_by_id=member_group_by_id,
            )
            connection_end_mode = self._terminal_connection_mode(
                member=m,
                node_id=m.j,
                detail=detail,
                node_member_envelopes=node_member_envelopes,
                member_group_by_id=member_group_by_id,
            )
            terminal_lap_overlap_mm = self._terminal_lap_overlap_mm(detail, stick_len)
            miter_cut_start_spec = self._terminal_end_cut_spec(
                member=m,
                node_id=m.i,
                nodes_by_id=node_by_id,
                members_by_id=members_by_id,
                node_member_envelopes=node_member_envelopes,
                member_group_by_id=member_group_by_id,
                detail=detail,
                fallback_axis=(ux, uy, uz),
                local_bevel_axis=local_z_axis,
            )
            miter_cut_end_spec = self._terminal_end_cut_spec(
                member=m,
                node_id=m.j,
                nodes_by_id=node_by_id,
                members_by_id=members_by_id,
                node_member_envelopes=node_member_envelopes,
                member_group_by_id=member_group_by_id,
                detail=detail,
                fallback_axis=(ux, uy, uz),
                local_bevel_axis=local_z_axis,
            )
            miter_cut_start_angle_deg = float(miter_cut_start_spec.get("angle_deg", 90.0) or 90.0)
            miter_cut_end_angle_deg = float(miter_cut_end_spec.get("angle_deg", 90.0) or 90.0)
            miter_cut_start_skew_sign = float(miter_cut_start_spec.get("skew_sign", 1.0) or 1.0)
            miter_cut_end_skew_sign = float(miter_cut_end_spec.get("skew_sign", 1.0) or 1.0)
            miter_cut_start_trim_axis = str(miter_cut_start_spec.get("trim_axis", "z") or "z")
            miter_cut_end_trim_axis = str(miter_cut_end_spec.get("trim_axis", "z") or "z")
            visual_connection_offset = self._node_lap_visual_side_offset(
                member=m,
                ni=ni,
                nj=nj,
                detail=detail,
            )

            joint_start_setback, joint_end_setback = self._joint_face_setbacks(
                m,
                L,
                node_member_envelopes=node_member_envelopes,
                detail=detail,
                stick_w=stick_w,
                stick_t=stick_t,
                min_constructive_piece_length_mm=min_constructive_piece_length_mm,
            )
            # A junta face-a-face é representada por recuo até a face do membro
            # hospedeiro, não por avanço até o centroide do nó.  A versão anterior
            # anulava o setback em face_lap_to_host para evitar lacunas visuais;
            # isso fazia o prisma entrar dentro do banzo/montante.  Agora o eixo
            # estrutural continua nó-a-nó, mas a peça física para na face e o CSV
            # contabiliza a sobreposição de cola no host.
            fabrication_L = max(0.0, L - joint_start_setback - joint_end_setback)
            interval_overlap = 0.0 if use_butt_splints else member_overlap
            intervals = self._piece_intervals(
                fabrication_L,
                stick_len,
                interval_overlap,
                min_constructive_piece_length_mm=min_constructive_piece_length_mm,
            )
            if joint_start_setback > 0.0 or joint_end_setback > 0.0:
                intervals = [
                    (
                        s0 + joint_start_setback,
                        s1 + joint_start_setback,
                        max(0.0, s1 - s0),
                    )
                    for s0, s1, _cl in intervals
                ]
            quadrant_id = 0
            if use_quarter_model and quarter_count > 0:
                # Determinar qual quadrante este membro pertence com base
                # no número de membros no quarto.  O identificador do
                # quadrante é dado por inteiro da divisão do índice do
                # membro (começando em 0) pelo total de membros de um
                # quarto.
                try:
                    quadrant_id = (int(m.id) - 1) // int(quarter_count)
                except (TypeError, ValueError, ZeroDivisionError):
                    quadrant_id = 0
                # Inverter a orientação das emendas para quadrantes ímpares
                if quadrant_id % 2 == 1:
                    rev: List[Tuple[float, float, float]] = []
                    for s0, s1, cl in reversed(intervals):
                        # Para inverter, subtrai os limites do comprimento total
                        rev.append((L - s1, L - s0, cl))
                    intervals = rev


            if str(x_layer.get("handling", "")) == "split_midpoint_lap_joint" and L > 2.0:
                # O solver mantém o X como duas barras axiais contínuas entre
                # nós de extremidade. Para a fabricação, porém, uma diagonal que
                # cruza outra no mesmo plano precisa ser cortada no cruzamento:
                # os segmentos terminam em uma junta colada palito-palito, sem
                # um prisma atravessando o outro. Isso altera só o detalhamento;
                # a rigidez global não ganha nó central artificial.
                mid_s = 0.5 * L
                split: List[Tuple[float, float, float]] = []
                for s0, s1, _cl in intervals:
                    if s0 + 1.0e-9 < mid_s < s1 - 1.0e-9:
                        split.append((s0, mid_s, max(0.0, mid_s - s0)))
                        split.append((mid_s, s1, max(0.0, s1 - mid_s)))
                    else:
                        split.append((s0, s1, max(0.0, s1 - s0)))
                intervals = [(a, b, min(stick_len, max(0.0, c))) for a, b, c in split if b - a > 1.0e-6]

            base_intervals = intervals

            r_y = self.sections.radius_of_gyration(sec["Iy"], sec["A"])
            r_z = self.sections.radius_of_gyration(sec["Iz"], sec["A"])

            slender_y = m.Ky * L / r_y if r_y else None
            slender_z = m.Kz * L / r_z if r_z else None

            M_imp = abs(N) * imperfection_e if N < 0 else 0.0

            c_y = sec.get("width_mm", stick_w) / 2.0
            c_z = sec.get("thickness_mm", stick_t) / 2.0

            sig_by = M_imp * c_z / sec["Iy"] if sec["Iy"] else 0.0
            sig_bz = M_imp * c_y / sec["Iz"] if sec["Iz"] else 0.0

            sigma_axial_member = N / sec["A"] if sec["A"] else 0.0
            sig_comb = abs(sigma_axial_member) + abs(sig_by) + abs(sig_bz)

            member_glue = 0.0
            member_glue_physical = 0.0
            joint_fs_values: List[float] = []
            # Determine joint model.  If a connection plan is attached to
            # the configuration it overrides the global defaults on a per
            # member basis.  The plan should be a dictionary keyed by
            # member id (as int or str) containing a ``recommended_joint_model``
            # field.  When absent the global tension/compression model is used.
            if member_plan and isinstance(member_plan, dict):
                plan_model = member_plan.get("recommended_joint_model") or member_plan.get("joint_model")
            else:
                plan_model = None
            if plan_model:
                joint_model = str(plan_model)
            else:
                joint_model = tension_joint_model if N >= 0 else compression_joint_model

            joint_area_factor = {
                "butt_plain": 0.35,
                "single_lap": 1.00,
                "single_lap_tala": 1.30,
                "butt_small_splints": 1.45,
                "butt_full_splints": 1.70,
                "double_lap": 1.75,
                "double_lap_reinforced": 2.10,
                "scarf": 1.55,
                "half_lap_notched": 1.40,
            }.get(joint_model, 1.0)
            joint_secondary_bending_factor = {
                "butt_plain": 1.55,
                "single_lap": 1.25,
                "single_lap_tala": 1.12,
                "butt_small_splints": 1.05,
                "butt_full_splints": 0.98,
                "double_lap": 1.00,
                "double_lap_reinforced": 0.95,
                "scarf": 1.00,
                "half_lap_notched": 1.08,
            }.get(joint_model, 1.0)

            for lane in range(1, n_lanes + 1):
                lane_yz = section_positions[lane - 1] if lane - 1 < len(section_positions) else (0.0, 0.0)
                try:
                    lane_y = float(lane_yz[0]) - cy
                    lane_z = float(lane_yz[1]) - cz
                except (TypeError, ValueError, IndexError):
                    lane_y = 0.0
                    lane_z = 0.0
                physical_face_lap_offset = visual_connection_offset if bool(detail.get("node_lap_physical_offset_enabled", True)) else (0.0, 0.0, 0.0)
                lane_offset_vec = (
                    lane_y * local_y_axis[0] + lane_z * local_z_axis[0] + float(x_layer_offset[0]) + float(physical_face_lap_offset[0]),
                    lane_y * local_y_axis[1] + lane_z * local_z_axis[1] + float(x_layer_offset[1]) + float(physical_face_lap_offset[1]),
                    lane_y * local_y_axis[2] + lane_z * local_z_axis[2] + float(x_layer_offset[2]) + float(physical_face_lap_offset[2]),
                )
                lane_orientation = str(lane_orientations[lane - 1]).strip().lower() if lane - 1 < len(lane_orientations) else stick_orientation
                lane_visual_width_mm = safe_float(lane_widths[lane - 1], None) if lane - 1 < len(lane_widths) else None
                lane_visual_thickness_mm = safe_float(lane_heights[lane - 1], None) if lane - 1 < len(lane_heights) else None
                if lane_visual_width_mm is None or lane_visual_thickness_mm is None:
                    lane_visual_width_mm = visual_width_mm
                    lane_visual_thickness_mm = visual_thickness_mm
                lane_intervals = list(base_intervals)
                if detail.get("splice_stagger_enabled", True):
                    lane_intervals = self.splice_stagger.offset_splice_positions(
                        lane_intervals,
                        member_length=L,
                        quadrant_id=quadrant_id,
                        lane_id=lane,
                        cfg=cfg,
                    )
                if strict_cut_length:
                    lane_intervals, split_count = self._enforce_stock_limit_on_intervals(
                        lane_intervals,
                        max_cut_mm=max_cut_length_mm,
                        overlap_mm=interval_overlap,
                        cut_increment_mm=cut_increment_mm,
                        min_constructive_piece_length_mm=min_constructive_piece_length_mm,
                    )
                    stock_limit_splits += split_count
                lane_intervals = self._enforce_min_constructive_piece_length(
                    lane_intervals,
                    min_constructive_piece_length_mm=min_constructive_piece_length_mm,
                    domain_start_mm=joint_start_setback,
                    domain_end_mm=max(joint_start_setback, L - joint_end_setback),
                    stock_limit_mm=max_cut_length_mm,
                )
                if use_butt_splints:
                    lane_intervals = self._force_butt_nonoverlap_intervals(
                        lane_intervals,
                        domain_start_mm=joint_start_setback,
                        domain_end_mm=max(joint_start_setback, L - joint_end_setback),
                    )
                prev_id = None
                prev_end = None

                for piece_index, (s0, s1, cut_len) in enumerate(lane_intervals, 1):
                    # Arredondamento de corte para incremento de oficina (ex.: 5 mm).
                    geom_len = max(0.0, float(cut_len))
                    if strict_cut_length and geom_len > max_cut_length_mm + 1.0e-9:
                        geom_len = max_cut_length_mm
                        s1 = min(L, s0 + geom_len)
                    if allow_cut_rounding and geom_len <= max_cut_length_mm + 1.0e-9:
                        cut_len_rounded = self.ceil_to_cut_increment(
                            geom_len,
                            increment_mm=cut_increment_mm,
                            min_value_mm=min_cut_length_mm,
                            max_value_mm=max_cut_length_mm,
                        )
                    else:
                        cut_len_rounded = geom_len
                    cut_rounding_delta = cut_len_rounded - geom_len
                    sid = f"M{m.id:03d}-L{lane:02d}-P{piece_index:02d}"

                    x0 = ni.x + ux * s0 + lane_offset_vec[0]
                    y0 = ni.y + uy * s0 + lane_offset_vec[1]
                    z0 = ni.z + uz * s0 + lane_offset_vec[2]

                    x1 = ni.x + ux * s1 + lane_offset_vec[0]
                    y1 = ni.y + uy * s1 + lane_offset_vec[1]
                    z1 = ni.z + uz * s1 + lane_offset_vec[2]

                    total_pieces += 1
                    total_cut += cut_len_rounded

                    cut_lengths.append(cut_len_rounded)
                    cut_counter[round(cut_len_rounded, 1)] += 1

                    is_first_piece = piece_index == 1
                    is_last_piece = piece_index == len(lane_intervals)
                    start_miter_relation = str(miter_cut_start_spec.get("host_relation", "") or "")
                    end_miter_relation = str(miter_cut_end_spec.get("host_relation", "") or "")
                    # Corte em grau é uma condição de encaixe terminal.  Para
                    # montantes/diagonais ele costuma coincidir com face-lap;
                    # para banzos, porém, a junta entre dois segmentos do próprio
                    # banzo pode estar em axis_centroid e ainda exigir meia
                    # esquadria para não criar quina/interpenetração.
                    start_can_miter = (
                        connection_start_mode == "face_lap_to_host"
                        or start_miter_relation in {"same_chord_half_miter", "relative_axis", "host_slope"}
                    )
                    end_can_miter = (
                        connection_end_mode == "face_lap_to_host"
                        or end_miter_relation in {"same_chord_half_miter", "relative_axis", "host_slope"}
                    )
                    start_miter_required = bool(
                        is_first_piece
                        and start_can_miter
                        and miter_cut_start_angle_deg < 89.0
                    )
                    end_miter_required = bool(
                        is_last_piece
                        and end_can_miter
                        and miter_cut_end_angle_deg < 89.0
                    )
                    piece_miter_start_angle = miter_cut_start_angle_deg if start_miter_required else 90.0
                    piece_miter_end_angle = miter_cut_end_angle_deg if end_miter_required else 90.0

                    material_cut_depth_mm = max(0.0, float(visual_thickness_mm))
                    miter_loss_start_mm = self._miter_material_loss_length_mm(
                        piece_miter_start_angle,
                        material_depth_mm=material_cut_depth_mm,
                        piece_length_mm=geom_len,
                        detail=detail,
                    ) if start_miter_required else 0.0
                    miter_loss_end_mm = self._miter_material_loss_length_mm(
                        piece_miter_end_angle,
                        material_depth_mm=material_cut_depth_mm,
                        piece_length_mm=geom_len,
                        detail=detail,
                    ) if end_miter_required else 0.0
                    miter_material_loss_length_mm = min(0.45 * max(0.0, geom_len), miter_loss_start_mm + miter_loss_end_mm)
                    net_installed_length_mm = max(0.0, geom_len - miter_material_loss_length_mm)
                    gross_piece_mass_g = stick_mass * geom_len / stick_len
                    net_piece_mass_g = stick_mass * net_installed_length_mm / stick_len

                    stick_rows.append(
                        {
                            "stick_id": sid,
                            "member_id": m.id,
                            "member_group": m.group,
                            "lane": lane,
                            "piece_index": piece_index,
                            "s0_mm": s0,
                            "s1_mm": s1,
                            "geometric_piece_length_mm": geom_len,
                            "installed_length_mm": net_installed_length_mm,
                            "gross_installed_length_mm": geom_len,
                            "miter_cut_material_loss_length_mm": miter_material_loss_length_mm,
                            "member_axis_length_mm": L,
                            "fabrication_axis_length_mm": fabrication_L,
                            "joint_start_setback_mm": joint_start_setback,
                            "joint_end_setback_mm": joint_end_setback,
                            "terminal_joint_trim_applied": bool(joint_start_setback > 1.0e-9 or joint_end_setback > 1.0e-9),
                            "cut_length_mm": cut_len_rounded,
                            "shop_cut_length_mm": cut_len_rounded,
                            "cut_rounding_delta_mm": cut_rounding_delta,
                            "min_constructive_piece_length_mm": min_constructive_piece_length_mm,
                            "constructive_piece_length_ok": bool(
                                geom_len >= min_constructive_piece_length_mm - 1.0e-9
                                or fabrication_L <= min_constructive_piece_length_mm + 1.0e-9
                            ),
                            "max_cut_length_mm": max_cut_length_mm,
                            "dimension_ok_length": bool(cut_len_rounded <= max_cut_length_mm + 1.0e-9),
                            "x0_mm": x0,
                            "y0_mm": y0,
                            "z0_mm": z0,
                            "x1_mm": x1,
                            "y1_mm": y1,
                            "z1_mm": z1,
                            "N_piece_N": per_lane,
                            "sigma_axial_piece_MPa": per_sigma,
                            "member_state": "tension" if N >= 0 else "compression",
                            "stick_orientation": lane_orientation,
                            "section_layout_effective": sec.get("layout"),
                            "section_layout_requested": sec.get("requested_layout", layout_cfg_detail.get("layout", "stacked")),
                            "section_connection_model": sec.get("section_connection_model", sec.get("layout")),
                            "width_mm": stick_w,
                            "thickness_mm": stick_t,
                            "visual_width_mm": lane_visual_width_mm,
                            "visual_thickness_mm": lane_visual_thickness_mm,
                            "dimension_ok_width": bool(max(lane_visual_width_mm, lane_visual_thickness_mm) <= max(stick_w, stick_t) + 1.0e-9),
                            "dimension_ok_thickness": bool(min(lane_visual_width_mm, lane_visual_thickness_mm) <= min(stick_w, stick_t) + 1.0e-9),
                            "section_local_y_mm": lane_y,
                            "section_local_z_mm": lane_z,
                            "section_global_offset_x_mm": lane_offset_vec[0],
                            "section_global_offset_y_mm": lane_offset_vec[1],
                            "section_global_offset_z_mm": lane_offset_vec[2],
                            "visual_connection_offset_x_mm": visual_connection_offset[0],
                            "visual_connection_offset_y_mm": visual_connection_offset[1],
                            "visual_connection_offset_z_mm": visual_connection_offset[2],
                            "physical_face_lap_offset_applied": bool(detail.get("node_lap_physical_offset_enabled", True)),
                            "physical_face_lap_offset_x_mm": physical_face_lap_offset[0],
                            "physical_face_lap_offset_y_mm": physical_face_lap_offset[1],
                            "physical_face_lap_offset_z_mm": physical_face_lap_offset[2],
                            "x_bracing_layer": x_layer.get("layer"),
                            "x_bracing_plane": x_layer.get("plane"),
                            "x_bracing_crossing_handling": x_layer.get("handling"),
                            "x_bracing_midspan_connected": bool(x_layer.get("midspan_connected", False)),
                            "section_axis_y_x": local_y_axis[0],
                            "section_axis_y_y": local_y_axis[1],
                            "section_axis_y_z": local_y_axis[2],
                            "section_axis_z_x": local_z_axis[0],
                            "section_axis_z_y": local_z_axis[1],
                            "section_axis_z_z": local_z_axis[2],
                            "section_centroid_y_mm": cy,
                            "section_centroid_z_mm": cz,
                            "n_sticks": n_lanes,
                            "layout": sec.get("layout"),
                            "quadrant_id": quadrant_id,
                            "connection_start_mode": connection_start_mode,
                            "connection_end_mode": connection_end_mode,
                            "terminal_lap_overlap_mm": terminal_lap_overlap_mm,
                            "node_connection_ok": bool(
                                connection_start_mode in {"face_lap_to_host", "axis_centroid"}
                                and connection_end_mode in {"face_lap_to_host", "axis_centroid"}
                            ),
                            "miter_cut_start_angle_deg": piece_miter_start_angle,
                            "miter_cut_end_angle_deg": piece_miter_end_angle,
                            "miter_cut_start_skew_sign": miter_cut_start_skew_sign if start_miter_required else 1.0,
                            "miter_cut_end_skew_sign": miter_cut_end_skew_sign if end_miter_required else 1.0,
                            "miter_cut_start_trim_axis": miter_cut_start_trim_axis if start_miter_required else "",
                            "miter_cut_end_trim_axis": miter_cut_end_trim_axis if end_miter_required else "",
                            "miter_cut_start_host_member_id": miter_cut_start_spec.get("host_member_id") if start_miter_required else None,
                            "miter_cut_end_host_member_id": miter_cut_end_spec.get("host_member_id") if end_miter_required else None,
                            "miter_cut_start_host_group": miter_cut_start_spec.get("host_group") if start_miter_required else "",
                            "miter_cut_end_host_group": miter_cut_end_spec.get("host_group") if end_miter_required else "",
                            "miter_cut_start_relation": miter_cut_start_spec.get("host_relation") if start_miter_required else "",
                            "miter_cut_end_relation": miter_cut_end_spec.get("host_relation") if end_miter_required else "",
                            "miter_cut_start_required": start_miter_required,
                            "miter_cut_end_required": end_miter_required,
                            "miter_cut_required": bool(start_miter_required or end_miter_required),
                            "miter_cut_start_position": "ponta inicial" if start_miter_required else "",
                            "miter_cut_end_position": "ponta final" if end_miter_required else "",
                            "assembly_unit_key": f"{m.group}|M{m.id:03d}|L{lane:02d}|P{piece_index:02d}",
                            # Massa competitiva da peça instalada: se o blank foi
                            # cortado em múltiplos de 5 mm e ajustado/lixado, a
                            # massa que fica na ponte é proporcional ao comprimento
                            # geométrico instalado, não ao palito bruto inteiro.
                            "gross_mass_g": gross_piece_mass_g,
                            "mass_g": net_piece_mass_g,
                        }
                    )

                    terminal_specs = []
                    if piece_index == 1 and connection_start_mode == "face_lap_to_host":
                        terminal_specs.append((
                            "start",
                            m.i,
                            None,
                            sid,
                            s0,
                            connection_start_mode,
                            miter_cut_start_angle_deg,
                        ))
                    if piece_index == len(lane_intervals) and connection_end_mode == "face_lap_to_host":
                        terminal_specs.append((
                            "end",
                            m.j,
                            sid,
                            None,
                            s1,
                            connection_end_mode,
                            miter_cut_end_angle_deg,
                        ))
                    for terminal_side, terminal_node_id, piece_a, piece_b, terminal_s, terminal_mode, terminal_angle in terminal_specs:
                        terminal_overlap = min(terminal_lap_overlap_mm, max(0.0, geom_len))
                        terminal_glue_area = terminal_overlap * stick_w * terminal_joint_area_factor
                        terminal_physical_glue_area = terminal_overlap * stick_w
                        if terminal_glue_area > 0:
                            terminal_glue_shear = (abs(per_lane) / terminal_glue_area) * terminal_joint_secondary_bending_factor
                        else:
                            terminal_glue_shear = None
                        terminal_allow = glue_tau / glue_sf if glue_sf > 0 else None
                        if terminal_glue_shear is None or terminal_glue_shear <= 0 or terminal_allow is None:
                            terminal_fs = None
                        else:
                            terminal_fs = terminal_allow / terminal_glue_shear
                        terminal_fs_clean = safe_float(terminal_fs, None)
                        if terminal_fs_clean is not None:
                            joint_fs_values.append(terminal_fs_clean)
                        member_glue += terminal_glue_area
                        member_glue_physical += terminal_physical_glue_area
                        total_glue_area += terminal_glue_area
                        total_glue_physical_area += terminal_physical_glue_area
                        joint_rows.append(
                            {
                                "joint_id": f"J-M{m.id:03d}-L{lane:02d}-{terminal_side.upper()}-NODE{int(terminal_node_id):03d}",
                                "member_id": m.id,
                                "member_group": m.group,
                                "lane": lane,
                                "piece_a": piece_a or f"NODE{int(terminal_node_id):03d}_HOST_FACE",
                                "piece_b": piece_b or f"NODE{int(terminal_node_id):03d}_HOST_FACE",
                                "joint_type": "terminal_face_lap",
                                "joint_model": "face_lap_to_host",
                                "connection_mode": terminal_mode,
                                "terminal_side": terminal_side,
                                "terminal_node_id": int(terminal_node_id),
                                "overlap_length_mm": terminal_overlap,
                                "splice_center_mm": terminal_s,
                                "quadrant_id": quadrant_id,
                                "joint_area_factor": terminal_joint_area_factor,
                                "joint_secondary_bending_factor": terminal_joint_secondary_bending_factor,
                                "glue_area_mm2": terminal_glue_area,
                                "physical_glue_area_mm2": terminal_physical_glue_area,
                                "force_transfer_N": abs(per_lane),
                                "glue_shear_MPa": terminal_glue_shear,
                                "glue_allow_design_MPa": terminal_allow,
                                "FS_glue_shear": terminal_fs_clean,
                                "FS_glue_shear_label": safety_label(terminal_fs_clean),
                                "risk_flag": risk_from_fs(terminal_fs_clean),
                                "miter_cut_angle_deg": terminal_angle,
                                "note": "junta terminal sobreposta em face; evita ligação ponta-a-ponta simples",
                            }
                        )

                    if prev_id is not None and prev_end is not None:
                        overlap_actual = max(0.0, prev_end - s0)
                        if use_butt_splints and overlap_actual <= 1.0e-9:
                            # Junta topo-a-topo com talas: as peças principais
                            # não se sobrepõem no mesmo volume. As talas são
                            # contabilizadas em massa equivalente e área de cola.
                            effective_overlap = splint_len
                            glue_area = effective_overlap * stick_w * max(joint_area_factor, 1.70) * splints_per_splice
                            physical_glue_area = effective_overlap * stick_w * splints_per_splice
                            joint_type = "butt_with_splints"
                            splice_note = "emenda topo-a-topo com talas laterais; sem interpenetração entre palitos principais"
                        else:
                            effective_overlap = overlap_actual
                            glue_area = overlap_actual * stick_w * joint_area_factor
                            physical_glue_area = overlap_actual * stick_w
                            joint_type = "lap_overlap"
                            splice_note = "emenda sobreposta ao longo da própria lane"

                        if glue_area > 0:
                            glue_shear = (abs(per_lane) / glue_area) * joint_secondary_bending_factor
                        else:
                            glue_shear = None

                        glue_allow = glue_tau / glue_sf if glue_sf > 0 else None

                        if glue_shear is None or glue_shear <= 0 or glue_allow is None:
                            fs_glue = None
                        else:
                            fs_glue = glue_allow / glue_shear

                        fs_glue_clean = safe_float(fs_glue, None)

                        if fs_glue_clean is not None:
                            joint_fs_values.append(fs_glue_clean)

                        member_glue += glue_area
                        member_glue_physical += physical_glue_area
                        total_glue_area += glue_area
                        total_glue_physical_area += physical_glue_area
                        splint_equiv = 0.0
                        splint_mass_g = 0.0
                        if use_butt_splints and overlap_actual <= 1.0e-9:
                            splint_equiv = splints_per_splice * splint_len / max(1.0e-9, stick_len)
                            splint_mass_g = stick_mass * splint_equiv
                            total_splint_sticks_equiv += splint_equiv
                            total_splint_mass_g += splint_mass_g

                        joint_rows.append(
                            {
                                "joint_id": f"J-M{m.id:03d}-L{lane:02d}-P{piece_index-1:02d}-{piece_index:02d}",
                                "member_id": m.id,
                                "member_group": m.group,
                                "lane": lane,
                                "piece_a": prev_id,
                                "piece_b": sid,
                                "joint_type": joint_type,
                                "joint_model": joint_model,
                                "overlap_length_mm": effective_overlap,
                                "splice_center_mm": 0.5 * (prev_end + s0),
                                "quadrant_id": quadrant_id,
                                "joint_area_factor": joint_area_factor,
                                "joint_secondary_bending_factor": joint_secondary_bending_factor,
                                "glue_area_mm2": glue_area,
                                "physical_glue_area_mm2": physical_glue_area,
                                "splint_length_mm": splint_len if joint_type == "butt_with_splints" else 0.0,
                                "splints_per_splice": splints_per_splice if joint_type == "butt_with_splints" else 0,
                                "splint_mass_g": splint_mass_g if joint_type == "butt_with_splints" else 0.0,
                                "force_transfer_N": abs(per_lane),
                                "glue_shear_MPa": glue_shear,
                                "glue_allow_design_MPa": glue_allow,
                                        "FS_glue_shear": fs_glue_clean,
                                        "FS_glue_shear_label": safety_label(fs_glue_clean),
                                        "risk_flag": risk_from_fs(fs_glue_clean),
                                        "splice_pattern": self.splice_stagger.assign_splice_stagger_pattern(
                                            cfg,
                                            {"member_id": m.id},
                                            quadrant_id,
                                            lane,
                                        ).get("splice_pattern", "brick_alt"),
                                        "stagger_offset_mm": self.splice_stagger.assign_splice_stagger_pattern(
                                            cfg,
                                            {"member_id": m.id},
                                            quadrant_id,
                                            lane,
                                        ).get("stagger_offset_mm", 0.0),
                                        "note": splice_note,
                            }
                        )

                    prev_id = sid
                    prev_end = s1

            glue_mass = (member_glue_physical / 1_000_000.0) * glue_spread / max(glue_eff, 1.0e-6)

            fs_min_global = safe_float(chk.get("FS_min"), None)
            fs_min_global_label = safety_label(fs_min_global)

            role = chk.get("member_role", "secondary")
            gov = chk.get("governing_mode", "")
            report_mode = chk.get("report_mode", gov)

            if role == "stabilizer" and tension_only:
                action = (
                    "manter como travamento/tension-only; "
                    "não dimensionar como coluna comprimida"
                )
                priority = "interpretation"
            elif fs_min_global is not None and fs_min_global < reinforce_if:
                if "buckling" in str(gov):
                    action = (
                        "reforçar: aumentar inércia, usar seção caixa/espaçada "
                        "ou reduzir comprimento livre"
                    )
                else:
                    action = (
                        "reforçar: adicionar palitos contínuos ou aumentar "
                        "sobreposição/talas"
                    )
                priority = "high"
            elif fs_min_global is not None and fs_min_global > remove_if and role != "primary":
                action = "avaliar remoção/redução: baixa solicitação relativa"
                priority = "low"
            else:
                action = "manter"
                priority = "normal"

            if priority != "normal":
                reinf_rows.append(
                    {
                        "member_id": m.id,
                        "group": m.group,
                        "role": role,
                        "N_N": N,
                        "FS_min": fs_min_global,
                        "FS_min_label": fs_min_global_label,
                        "governing_mode": gov,
                        "report_mode": report_mode,
                        "suggested_action": action,
                        "priority": priority,
                    }
                )

            fs_min_glue = min(joint_fs_values) if joint_fs_values else None

            member_rows.append(
                {
                    "member_id": m.id,
                    "group": m.group,
                    "role": role,
                    "n_sticks_current": n_lanes,
                    "n_sticks_recommended": int(sizing.get("n_sticks_recommended", n_lanes)),
                    "n_lanes_sticks": n_lanes,
                    "pieces_per_lane": len(base_intervals),
                    "total_piece_count": len(base_intervals) * n_lanes,
                    "member_length_mm": L,
                    "fabrication_axis_length_mm": fabrication_L,
                    "joint_start_setback_mm": joint_start_setback,
                    "joint_end_setback_mm": joint_end_setback,
                    "layout": sec.get("layout"),
                    "section_A_mm2": sec["A"],
                    "section_Iy_mm4": sec["Iy"],
                    "section_Iz_mm4": sec["Iz"],
                    "section_Iy_perfect_mm4": sec.get("Iy_perfect"),
                    "section_Iz_perfect_mm4": sec.get("Iz_perfect"),
                    "section_Iy_noncomposite_mm4": sec.get("Iy_noncomposite"),
                    "section_Iz_noncomposite_mm4": sec.get("Iz_noncomposite"),
                    "section_eta_I": sec.get("eta_I"),
                    "section_J_mm4_est": sec["J"],
                    "radius_y_mm": r_y,
                    "radius_z_mm": r_z,
                    "slenderness_y": slender_y,
                    "slenderness_z": slender_z,
                    "N_member_N": N,
                    "N_per_lane_N": per_lane,
                    "sigma_axial_member_MPa": sigma_axial_member,
                    "sigma_axial_piece_MPa": per_sigma,
                    "M_imperfection_Nmm": M_imp,
                    "sigma_bending_est_MPa": max(abs(sig_by), abs(sig_bz)),
                    "sigma_combined_est_MPa": sig_comb,
                    "joint_model": joint_model,
                    "glue_area_total_mm2": member_glue,
                    "glue_mass_est_g": glue_mass,
                    "FS_min_global": fs_min_global,
                    "FS_min_global_label": fs_min_global_label,
                    "FS_min_glue": fs_min_glue,
                    "FS_min_glue_label": safety_label(fs_min_glue),
                    "governing_mode_global": gov,
                    "report_mode_global": report_mode,
                    "suggested_action": action,
                }
            )

        # Detecta e anota alinhamentos críticos de emendas após detalhamento completo.
        joint_rows = self.splice_stagger.reduce_aligned_splices(joint_rows, cfg)
        splice_stagger_report = self.splice_stagger.validate_splice_alignment(joint_rows, cfg)

        cutting_rows = [
            {
                "cut_length_mm": k,
                "quantity": v,
                "total_length_mm": k * v,
            }
            for k, v in sorted(cut_counter.items(), reverse=True)
        ]

        bins = self._pack_cuts_best_fit(cut_lengths, stick_len, kerf)

        blank_plan: List[Dict] = []

        for idx, cuts in enumerate(bins, 1):
            used = sum(cuts) + max(0, len(cuts) - 1) * kerf

            blank_plan.append(
                {
                    "blank_stick_index": idx,
                    "cuts_mm": ";".join(f"{c:.1f}" for c in cuts),
                    "n_cuts": len(cuts),
                    "used_length_mm_including_kerf": used,
                    "waste_length_mm": max(0.0, stick_len - used),
                }
            )

        blank = len(bins)
        extra = math.ceil(blank * waste)
        total = blank + extra

        primary_piece_mass = sum(float(r["mass_g"]) for r in stick_rows)
        installed_stick_mass = primary_piece_mass + total_splint_mass_g
        wet_glue_mass = (total_glue_physical_area / 1_000_000.0) * glue_spread / max(glue_eff, 1.0e-6)
        cured_glue_mass = wet_glue_mass * glue_cure_solids_fraction
        evaporated_glue_water = max(0.0, wet_glue_mass - cured_glue_mass)

        purchased_blank_sticks_needed = max(total, int(math.ceil(installed_stick_mass / max(stick_mass, 1.0e-9))))
        purchased_stick_mass = purchased_blank_sticks_needed * stick_mass
        cutting_scrap_mass = max(0.0, purchased_stick_mass - installed_stick_mass)

        competition_mass = installed_stick_mass + cured_glue_mass
        assembly_procurement_mass = max(purchased_stick_mass + wet_glue_mass, competition_mass)

        mass_limits = resolve_mass_limits(cfg)
        limit = float(mass_limits["effective_limit_g"])
        mat = cfg.get("material", {}) or {}
        planner = cfg.get("planner", {}) or {}
        stick_budget_g = safe_float(mat.get("stick_budget_g"), safe_float(planner.get("target_installed_stick_mass_g"), 900.0)) or 900.0
        wet_glue_budget_g = safe_float(mat.get("wet_glue_budget_g"), safe_float(planner.get("target_wet_glue_mass_g"), 100.0)) or 100.0
        nominal_competition_limit_g = safe_float(
            mat.get("nominal_competition_limit_g"),
            safe_float(mass_limits.get("nominal_limit_g"), 1000.0),
        ) or 1000.0
        glue_acceptance_fs = safe_float(
            cfg.get("analysis", {}).get("acceptance_min_glue_fs"),
            1.5,
        ) or 1.5
        weak_glue_count = 0
        for r in (joint_rows or []):
            fs_joint = safe_float(r.get("FS_glue_shear"), None)
            if fs_joint is not None and fs_joint < glue_acceptance_fs:
                weak_glue_count += 1
        min_bottom_chord_glue_fs = min(
            (
                safe_float(r.get("FS_glue_shear"), None)
                for r in (joint_rows or [])
                if str(r.get("member_group", "")).strip() == "bottom_chord"
                and safe_float(r.get("FS_glue_shear"), None) is not None
            ),
            default=None,
        )

        summary = {
            "total_members": len(member_rows),
            "total_piece_instances": total_pieces,
            "total_cut_length_mm": total_cut,
            "estimated_blank_sticks_needed": blank,
            "waste_factor": waste,
            "extra_sticks_for_waste": extra,
            "estimated_total_sticks_with_waste": total,
            "estimated_piece_mass_g_without_waste_scaling": installed_stick_mass,
            "primary_piece_mass_g": primary_piece_mass,
            "splint_mass_g": total_splint_mass_g,
            "splint_sticks_equivalent": total_splint_sticks_equiv,
            "splice_mode": splice_mode,
            "installed_stick_mass_g": installed_stick_mass,
            "purchased_blank_sticks_needed": purchased_blank_sticks_needed,
            "purchased_stick_mass_g": purchased_stick_mass,
            "cutting_scrap_mass_g": cutting_scrap_mass,
            "estimated_glue_area_mm2": total_glue_physical_area,
            "effective_glue_area_mm2": total_glue_area,
            "estimated_glue_mass_g": wet_glue_mass,
            "wet_glue_mass_g": wet_glue_mass,
            "glue_cure_solids_fraction": glue_cure_solids_fraction,
            "cured_glue_mass_g": cured_glue_mass,
            "evaporated_glue_water_g": evaporated_glue_water,
            "competition_mass_g": competition_mass,
            "assembly_procurement_mass_g": assembly_procurement_mass,
            # Backward compatibility: old "total mass" now maps to final competition mass.
            "estimated_total_mass_g": competition_mass,
            "mass_limit_g": limit,
            "mass_margin_g": limit - competition_mass,
            "mass_limit_nominal_g": float(mass_limits["nominal_limit_g"]),
            "mass_limit_material_g": mass_limits["material_limit_g"],
            "mass_limit_planner_g": mass_limits["planner_limit_g"],
            "mass_limit_effective_g": float(mass_limits["effective_limit_g"]),
            "mass_limit_effective_source": str(mass_limits["effective_source"]),
            "stick_budget_g": stick_budget_g,
            "wet_glue_budget_g": wet_glue_budget_g,
            "stick_budget_margin_g": stick_budget_g - installed_stick_mass,
            "wet_glue_budget_margin_g": wet_glue_budget_g - wet_glue_mass,
            "nominal_competition_limit_g": nominal_competition_limit_g,
            "competition_mass_margin_g": nominal_competition_limit_g - competition_mass,
            "n_weak_glue_joints": weak_glue_count,
            "min_bottom_chord_glue_fs": min_bottom_chord_glue_fs,
            "glue_shear_strength_MPa": glue_tau,
            "glue_safety_factor": glue_sf,
            "cut_increment_mm": cut_increment_mm,
            "allow_cut_rounding": allow_cut_rounding,
            "max_cut_length_mm": max_cut_length_mm,
            "strict_cut_length": strict_cut_length,
            "min_constructive_piece_length_mm": min_constructive_piece_length_mm,
            "short_constructive_piece_count": int(
                sum(1 for r in stick_rows if not bool(r.get("constructive_piece_length_ok", True)))
            ),
            "joint_face_setback_enabled": bool(detail.get("joint_face_setback_enabled", True)),
            "joint_setback_piece_count": int(
                sum(1 for r in stick_rows if bool(r.get("terminal_joint_trim_applied", False)))
            ),
            "node_face_lap_enabled": bool(detail.get("node_face_lap_enabled", True)),
            "terminal_face_lap_joint_count": int(
                sum(1 for r in joint_rows if str(r.get("joint_type")) == "terminal_face_lap")
            ),
            "node_connection_gap_count": int(
                sum(1 for r in stick_rows if not bool(r.get("node_connection_ok", True)))
            ),
            "miter_cut_piece_count": int(
                sum(1 for r in stick_rows if bool(r.get("miter_cut_required", False)))
            ),
            "stock_limit_splits": stock_limit_splits,
            "oversize_piece_count": int(sum(1 for r in stick_rows if (safe_float(r.get("cut_length_mm"), 0.0) or 0.0) > max_cut_length_mm + 1.0e-9)),
        }

        weakest = sorted(
            member_rows,
            key=lambda r: safe_sort_key(r.get("FS_min_global")),
        )[:30]

        glue_weak = sorted(
            joint_rows,
            key=lambda r: safe_sort_key(r.get("FS_glue_shear")),
        )[:30]

        exports = {
            "stick_pieces.csv": stick_rows,
            "glue_joints.csv": joint_rows,
            "member_detail_checks.csv": member_rows,
            "cutting_list.csv": cutting_rows,
            "blank_cut_plan.csv": blank_plan,
            "reinforcement_suggestions.csv": reinf_rows,
            "weakest_members.csv": weakest,
            "weakest_glue_joints.csv": glue_weak,
        }

        for filename, rows in exports.items():
            GeometryService.write_csv(out / filename, rows)

        (out / "detailed_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out / "splice_stagger_report.json").write_text(
            json.dumps(splice_stagger_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        oversize_rows = [
            r for r in stick_rows
            if (safe_float(r.get("cut_length_mm"), 0.0) or 0.0) > max_cut_length_mm + 1.0e-9
        ]
        (out / "09_auditoria_conectividade_e_cortes.md").write_text(
            "# Auditoria de cortes, dimensões e conectividade\n\n"
            f"- Comprimento máximo permitido por palito: **{max_cut_length_mm:.1f} mm**.\n"
            f"- Incremento de corte usado: **{cut_increment_mm:.1f} mm**.\n"
            f"- Peças repartidas por excederem o palito real após stagger: **{stock_limit_splits}**.\n"
            f"- Peças acima do limite após correção: **{len(oversize_rows)}**.\n\n"
            "Nenhum corte exportado deve exigir palito maior que o lote real. "
            "Quando o desencontro de emendas cria uma peça longa demais, ela é repartida com sobreposição.\n",
            encoding="utf-8",
        )

        return {
            "stick_pieces": stick_rows,
            "glue_joints": joint_rows,
            "member_detail_checks": member_rows,
            "cutting_list": cutting_rows,
            "blank_cut_plan": blank_plan,
            "reinforcement_suggestions": reinf_rows,
            "weakest_members": weakest,
            "weakest_glue_joints": glue_weak,
            "splice_stagger_report": splice_stagger_report,
            "summary": summary,
        }
