"""Greedy sequential planner: starts at most one task per tick."""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Sequence


class SequentialPlanner:
    """Heuristic planner that executes a single ready task each tick."""

    planner_id = "sequential"

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed or 0
        self.rng = random.Random(self.seed)

    def begin_episode(self, level_config: Mapping[str, object], seed: int) -> None:
        self.seed = seed
        self.rng.seed(seed)

    def select_actions(self, observation: Mapping[str, object]) -> Sequence[str]:
        ready: List[Dict[str, object]] = list(observation.get("ready", []))
        if not ready:
            return []

        ready.sort(key=self._priority_key)
        chosen = ready[0]["task_id"]
        return [chosen]

    def _priority_key(self, task: Mapping[str, object]) -> tuple:
        # Naive heuristic: focus on quick wins even if long tasks starve.
        duration = int(task["duration"])
        outstanding = len(task.get("outstanding_dependencies", []))
        task_id = str(task["task_id"])
        return (duration, outstanding, task_id)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"SequentialPlanner(seed={self.seed})"
