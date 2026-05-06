from __future__ import annotations

import json

from src.services.planner_debug_logger import PlannerDebugLogger


def test_planner_debug_logger_writes_jsonl(tmp_path) -> None:
    logger = PlannerDebugLogger(tmp_path / "logs", enabled=True)
    logger.event("config_loaded", stage="planner", candidate_id="S0-0001", reason="unit_test")
    logger.event("s0_candidate_generated", stage="s0", candidate_id="S0-0001", metrics={"x": 1})
    logger.write_summary()

    assert logger.jsonl_path.exists()
    lines = [ln for ln in logger.jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 2

    row = json.loads(lines[0])
    for key in ("timestamp", "stage", "event_type"):
        assert key in row
    assert logger.summary_path.exists()

