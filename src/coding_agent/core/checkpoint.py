"""Checkpoint system - save/restore agent session state.

Like Cursor's Checkpoints feature for snapshotting agent sessions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Checkpoint:
    id: str
    timestamp: float
    task: str
    iteration: int
    messages: list[dict]
    files_read: list[str]
    files_written: list[str]
    decisions: list[str]
    metrics: dict


class CheckpointManager:
    """Manages checkpoint save/restore for agent sessions."""

    def __init__(self, project_root: str = ".") -> None:
        self._dir = Path(project_root).resolve() / ".coding-agent" / "checkpoints"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints: list[Checkpoint] = []

    def save(self, task: str, iteration: int, messages: list[dict],
             files_read: list[str], files_written: list[str],
             decisions: list[str], metrics: dict | None = None) -> Checkpoint:
        cp = Checkpoint(
            id=f"cp-{int(time.time())}",
            timestamp=time.time(),
            task=task[:100],
            iteration=iteration,
            messages=messages[-20:],
            files_read=list(files_read),
            files_written=list(files_written),
            decisions=list(decisions),
            metrics=metrics or {},
        )
        path = self._dir / f"{cp.id}.json"
        try:
            path.write_text(json.dumps({
                "id": cp.id,
                "timestamp": cp.timestamp,
                "task": cp.task,
                "iteration": cp.iteration,
                "messages": cp.messages,
                "files_read": cp.files_read,
                "files_written": cp.files_written,
                "decisions": cp.decisions,
                "metrics": cp.metrics,
            }, indent=2))
        except Exception:
            pass
        self._checkpoints.append(cp)
        self._cleanup()
        return cp

    def list_checkpoints(self) -> list[dict]:
        return [
            {"id": cp.id, "timestamp": cp.timestamp, "task": cp.task, "iteration": cp.iteration}
            for cp in self._checkpoints
        ]

    def _cleanup(self, max_checkpoints: int = 10) -> None:
        while len(self._checkpoints) > max_checkpoints:
            old = self._checkpoints.pop(0)
            path = self._dir / f"{old.id}.json"
            if path.exists():
                path.unlink()
