from src.services.config_service import ConfigService
from src.services.stick_detail_service import StickDetailService
from src.services.visualization_service import VisualizationService


def test_butt_with_splints_intervals_do_not_overlap():
    intervals = StickDetailService._piece_intervals(260.0, 120.0, 0.0)
    assert len(intervals) >= 3
    for (_, prev_end, _), (next_start, _, _) in zip(intervals, intervals[1:]):
        assert next_start >= prev_end - 1.0e-9


def test_normalize_butt_splice_forces_zero_overlap():
    cfg = {
        "detail_model": {
            "splice_mode": "butt_with_splints",
            "overlap_length_mm": 30.0,
            "auto_splice_overlap_enabled": True,
        }
    }
    out = ConfigService().normalize(cfg)
    assert out["detail_model"]["splice_mode"] == "butt_with_splints"
    assert out["detail_model"]["overlap_length_mm"] == 0.0
    assert out["detail_model"]["auto_splice_overlap_enabled"] is False


def test_obb_collision_does_not_flag_separated_sticks():
    rows = [
        {"stick_id": "A", "x0_mm": 0, "y0_mm": 0, "z0_mm": 0, "x1_mm": 100, "y1_mm": 0, "z1_mm": 0, "width_mm": 7, "thickness_mm": 1.5},
        {"stick_id": "B", "x0_mm": 100.3, "y0_mm": 0, "z0_mm": 0, "x1_mm": 200.3, "y1_mm": 0, "z1_mm": 0, "width_mm": 7, "thickness_mm": 1.5},
    ]
    batches = VisualizationService().prepare_stick_piece_mesh_batches(rows, connection_offset_scale=0.0)
    assert batches["as_built_interpenetration_count"] == 0
