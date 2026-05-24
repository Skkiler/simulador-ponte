from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from src.core.numeric import safe_float


class SpliceStaggeringService:
    """Apply constructive stagger on splice positions while preserving structural symmetry."""

    @staticmethod
    def _detail(cfg: Dict) -> Dict:
        return cfg.get("detail_model", {}) or {}

    def assign_splice_stagger_pattern(
        self,
        cfg: Dict,
        member: Dict | Any,
        quadrant_id: int,
        lane_id: int,
    ) -> Dict[str, Any]:
        d = self._detail(cfg)
        step = max(1.0, float(d.get("splice_stagger_step_mm", 5.0)))
        max_off = max(step, float(d.get("splice_stagger_max_offset_mm", 30.0)))
        pattern = "brick_alt"
        mag = min(max_off, step * (1 + (abs(int(lane_id)) % 3)))
        sign = -1.0 if ((int(quadrant_id) + int(lane_id)) % 2) else 1.0
        return {
            "splice_pattern": pattern,
            "stagger_offset_mm": round(sign * mag, 6),
            "quadrant_id": int(quadrant_id),
            "lane_id": int(lane_id),
        }

    def offset_splice_positions(
        self,
        intervals: List[Tuple[float, float, float]],
        member_length: float,
        quadrant_id: int,
        lane_id: int,
        cfg: Dict,
    ) -> List[Tuple[float, float, float]]:
        if len(intervals) <= 1:
            return intervals
        d = self._detail(cfg)
        min_margin = max(0.0, float(d.get("min_end_margin_mm", 10.0)))
        allow = bool(d.get("splice_stagger_enabled", True))
        if not allow:
            return intervals

        patt = self.assign_splice_stagger_pattern(cfg, {}, quadrant_id, lane_id)
        base_off = float(patt.get("stagger_offset_mm", 0.0))
        out = [(float(a), float(b), float(c)) for a, b, c in intervals]

        splice_mode = str(d.get("splice_mode", "overlap") or "overlap").strip().lower()
        butt_splints = splice_mode in {"butt_with_splints", "butt_splints", "butt_full_splints"}
        if butt_splints:
            # In butt-with-splints mode, stagger means moving the butt line, not
            # pulling the next stick backward into a lap.  The previous
            # implementation reused the overlap algorithm and introduced 1 mm
            # longitudinal overlaps even with overlap_length_mm = 0; those tiny
            # false laps then governed glue rupture and created same-lane volume
            # interpenetration in the as-built audit.  Here every internal splice
            # stays a single shared boundary: prev.end == next.start.
            min_piece = max(1.0, float(d.get("min_constructive_piece_length_mm", 40.0)))
            domain_start = float(out[0][0])
            domain_end = float(out[-1][1])
            if domain_end <= domain_start:
                domain_start = 0.0
                domain_end = max(float(member_length), max(float(b) for _a, b, _c in out))
            for i in range(1, len(out)):
                prev_s0, _prev_s1, _prev_cl = out[i - 1]
                s0, s1, _cl = out[i]
                alternating = -1.0 if (i % 2) else 1.0
                desired = s0 + base_off * alternating
                lo = max(prev_s0 + min_piece, domain_start + min_margin)
                hi = min(s1 - min_piece, domain_end - min_margin)
                if lo > hi:
                    boundary = s0
                else:
                    boundary = max(lo, min(hi, desired))
                if boundary <= prev_s0 + 1.0e-9 or boundary >= s1 - 1.0e-9:
                    boundary = max(prev_s0 + 1.0, min(s1 - 1.0, s0))
                out[i - 1] = (prev_s0, boundary, max(0.0, boundary - prev_s0))
                out[i] = (boundary, s1, max(0.0, s1 - boundary))
            return out

        min_overlap = max(1.0, float(d.get("overlap_length_mm", 30.0)) * 0.5)
        for i in range(1, len(out)):
            s0, s1, cl = out[i]
            prev_s0, prev_s1, prev_cl = out[i - 1]
            alternating = -1.0 if (i % 2) else 1.0
            shift = base_off * alternating
            new_s0 = s0 + shift
            new_s0 = max(min_margin, min(member_length - min_margin, new_s0))
            overlap = max(0.0, prev_s1 - new_s0)
            if overlap < min_overlap:
                new_s0 = prev_s1 - min_overlap
                new_s0 = max(min_margin, min(member_length - min_margin, new_s0))
            if new_s0 >= s1:
                continue
            out[i] = (new_s0, s1, max(0.0, s1 - new_s0))
        return out

    def detect_aligned_splice_clusters(
        self,
        full_glue_joints: List[Dict],
        tolerance_mm: float = 10.0,
    ) -> List[Dict]:
        tol = max(0.1, float(tolerance_mm))
        by_bin: Dict[int, List[Dict]] = defaultdict(list)
        for row in full_glue_joints or []:
            pos = safe_float(row.get("splice_center_mm"), None)
            if pos is None:
                continue
            b = int(round(pos / tol))
            by_bin[b].append(row)

        clusters: List[Dict] = []
        cluster_id = 1
        for b, rows in sorted(by_bin.items()):
            quadrants = {int(safe_float(r.get("quadrant_id"), -1) or -1) for r in rows}
            lanes = {int(safe_float(r.get("lane"), -1) or -1) for r in rows}
            if len(rows) >= 3 and len(quadrants) >= 2:
                clusters.append(
                    {
                        "cluster_id": cluster_id,
                        "bin_center_mm": b * tol,
                        "count": len(rows),
                        "quadrants": sorted(quadrants),
                        "lanes": sorted(lanes),
                    }
                )
                cluster_id += 1
        return clusters

    def validate_splice_alignment(self, full_glue_joints: List[Dict], cfg: Dict) -> Dict:
        d = self._detail(cfg)
        tol = max(
            float(d.get("splice_alignment_tolerance_mm", 10.0)),
            float(d.get("splice_min_aligned_distance_mm", 30.0)),
        )
        clusters = self.detect_aligned_splice_clusters(full_glue_joints, tol)
        critical = [c for c in clusters if int(c.get("count", 0)) >= 4]
        return {
            "alignment_tolerance_mm": tol,
            "clusters_found": len(clusters),
            "critical_clusters": len(critical),
            "clusters": clusters,
            "is_ok": len(critical) == 0,
        }

    def reduce_aligned_splices(self, full_glue_joints: List[Dict], cfg: Dict) -> List[Dict]:
        # Conservative post-process: annotate risk by detected clusters.
        d = self._detail(cfg)
        tol = max(
            float(d.get("splice_alignment_tolerance_mm", 10.0)),
            float(d.get("splice_min_aligned_distance_mm", 30.0)),
        )
        clusters = self.detect_aligned_splice_clusters(full_glue_joints, tol)
        if not clusters:
            for row in full_glue_joints or []:
                row["aligned_cluster_id"] = None
                row["alignment_risk"] = "low"
            return full_glue_joints

        for row in full_glue_joints or []:
            pos = safe_float(row.get("splice_center_mm"), None)
            row["aligned_cluster_id"] = None
            row["alignment_risk"] = "low"
            if pos is None:
                continue
            for c in clusters:
                if abs(float(c["bin_center_mm"]) - pos) <= tol:
                    row["aligned_cluster_id"] = int(c["cluster_id"])
                    row["alignment_risk"] = "critical" if int(c.get("count", 0)) >= 4 else "moderate"
                    break
        return full_glue_joints
