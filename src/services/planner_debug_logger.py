from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class PlannerDebugLogger:
    """Structured planner logger (JSONL + markdown summary)."""

    def __init__(
        self,
        root_dir: str | Path = "outputs/logs",
        *,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.root_dir / "planner_debug.jsonl"
        self.summary_path = self.root_dir / "planner_debug_summary.md"
        self._counts: Counter[str] = Counter()
        self._stage_counts: Counter[str] = Counter()
        self._warnings: list[str] = []
        self._last_reason_by_event: Dict[str, str] = {}

    @staticmethod
    def _ts() -> str:
        return datetime.now(timezone.utc).isoformat()

    def event(
        self,
        event_type: str,
        *,
        stage: str = "",
        candidate_id: str | None = None,
        member_id: int | None = None,
        group: str | None = None,
        previous_value: Any = None,
        new_value: Any = None,
        reason: str | None = None,
        metrics: Dict[str, Any] | None = None,
        level: str = "info",
        **extra: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "timestamp": self._ts(),
            "level": str(level),
            "stage": str(stage),
            "event_type": str(event_type),
            "candidate_id": candidate_id,
            "member_id": member_id,
            "group": group,
            "previous_value": previous_value,
            "new_value": new_value,
            "reason": reason,
            "metrics": metrics or {},
        }
        if extra:
            payload.update(extra)

        self._counts[str(event_type)] += 1
        if stage:
            self._stage_counts[str(stage)] += 1
        if reason:
            self._last_reason_by_event[str(event_type)] = str(reason)
        if str(level).lower() in {"warning", "error"} and reason:
            self._warnings.append(f"[{stage}] {event_type}: {reason}")

        if self.enabled:
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def write_summary(self) -> Path:
        lines: list[str] = [
            "# Planner Debug Summary",
            "",
            "## Event Counters",
            "",
            "| event_type | count |",
            "| --- | ---: |",
        ]
        for ev, cnt in sorted(self._counts.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"| {ev} | {cnt} |")

        lines.extend(
            [
                "",
                "## Stage Counters",
                "",
                "| stage | count |",
                "| --- | ---: |",
            ]
        )
        for st, cnt in sorted(self._stage_counts.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"| {st} | {cnt} |")

        lines.extend(["", "## Main Reasons", ""])
        if self._last_reason_by_event:
            for ev, rsn in sorted(self._last_reason_by_event.items()):
                lines.append(f"- `{ev}`: {rsn}")
        else:
            lines.append("- none")

        lines.extend(["", "## Warnings", ""])
        if self._warnings:
            lines.extend(f"- {w}" for w in self._warnings[-100:])
        else:
            lines.append("- none")

        self.summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self.summary_path

