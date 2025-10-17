"""Deterministic kitchen simulator for ParaCook Scoreboard."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TaskSpec:
    """Static specification for a task belonging to an order."""

    task_id: str
    order_id: str
    duration: int
    resources: Mapping[str, int]
    requires: Tuple[str, ...] = field(default_factory=tuple)

    def requires_full_ids(self) -> Tuple[str, ...]:
        """Return fully-qualified dependency ids (<order>:<task>)."""
        return tuple(f"{self.order_id}:{dep}" for dep in self.requires)


@dataclass
class TaskState:
    """Mutable execution state for a task."""

    spec: TaskSpec
    status: str = "pending"
    remaining: int = 0
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    ready_since: Optional[int] = None
    wait_time: int = 0

    def reset(self) -> None:
        self.status = "pending"
        self.remaining = self.spec.duration
        self.start_time = None
        self.end_time = None
        self.ready_since = None
        self.wait_time = 0

    @property
    def task_id(self) -> str:
        return self.spec.task_id

    @property
    def order_id(self) -> str:
        return self.spec.order_id


@dataclass
class SimulationResult:
    """Snapshot of an environment rollout for downstream metrics."""

    level_name: str
    seed: int
    history: List[MutableMapping[str, object]]
    task_reports: Dict[str, Dict[str, object]]
    resource_caps: Mapping[str, int]
    total_time: int


class KitchenEnv:
    """Discrete-time simulator with resource-constrained task execution."""

    def __init__(self, level_config: Mapping[str, object], seed: int = 0):
        self.level_config = level_config
        self.level_name = str(level_config.get("name", "unknown"))
        self.seed = seed
        self.rng = random.Random(seed)
        self.time: int = 0
        self._resource_caps: Dict[str, int] = {}
        self._resource_available: Dict[str, int] = {}
        self._tasks: Dict[str, TaskState] = {}
        self._active_tasks: Dict[str, TaskState] = {}
        self.history: List[MutableMapping[str, object]] = []
        self.reset()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset simulator state while keeping the same seed."""
        self.rng.seed(self.seed)
        self.time = 0
        self.history = []
        self._resource_caps = self._build_resource_caps(self.level_config)
        self._resource_available = dict(self._resource_caps)
        self._tasks = self._build_task_states(self.level_config)
        self._active_tasks = {}

    def _build_resource_caps(self, config: Mapping[str, object]) -> Dict[str, int]:
        hands = int(config.get("hands", 1))
        stations = {
            str(name): int(amount)
            for name, amount in dict(config.get("stations", {})).items()
        }
        resource_caps = {"hands": hands}
        resource_caps.update(stations)
        return resource_caps

    def _build_task_states(self, config: Mapping[str, object]) -> Dict[str, TaskState]:
        tasks: Dict[str, TaskState] = {}
        for order in config.get("orders", []):
            order_id = str(order["id"])
            raw_tasks: Mapping[str, object] = order.get("tasks", {})
            for task_local_id in sorted(raw_tasks):
                details = raw_tasks[task_local_id]
                spec = TaskSpec(
                    task_id=f"{order_id}:{task_local_id}",
                    order_id=order_id,
                    duration=int(details["duration"]),
                    resources=_normalise_resources(details.get("resources", {})),
                    requires=tuple(details.get("requires", [])),
                )
                state = TaskState(spec=spec)
                state.reset()
                # Store dependencies as fully-qualified ids for quick checks.
                tasks[spec.task_id] = state
        return tasks

    # ------------------------------------------------------------------
    # Environment loop
    # ------------------------------------------------------------------
    def observe(self) -> Dict[str, object]:
        """Return a planner-facing snapshot of the current state."""
        return {
            "time": self.time,
            "ready": [self._task_view(state) for state in self._ready_task_states()],
            "in_progress": [
                {
                    "task_id": tid,
                    "order_id": state.order_id,
                    "remaining": state.remaining,
                    "resources": dict(state.spec.resources),
                }
                for tid, state in sorted(self._active_tasks.items())
            ],
            "resources_available": dict(self._resource_available),
            "resource_caps": dict(self._resource_caps),
        }

    def step(self, requested: Sequence[str]) -> MutableMapping[str, object]:
        """Advance the simulation by one tick given requested task starts."""
        if self.done:
            raise RuntimeError("KitchenEnv already finished; reset() before stepping.")

        # First, close tasks that completed at the end of the previous tick.
        completed = self._advance_time()

        # Prepare bookkeeping for this decision tick.
        unique_request_order = []
        seen = set()

        for task_id in requested:
            if task_id not in seen:
                unique_request_order.append(task_id)
                seen.add(task_id)

        started: List[str] = []
        blocked: List[str] = []
        blocked_dependency: List[str] = []
        blocked_resource: List[str] = []
        invalid: List[str] = []

        for task_id in unique_request_order:
            state = self._tasks.get(task_id)
            if state is None or state.status != "pending":
                invalid.append(task_id)
                continue

            if not self._dependencies_satisfied(state):
                blocked.append(task_id)
                blocked_dependency.append(task_id)
                continue

            if not self._resources_available(state.spec.resources):
                blocked.append(task_id)
                blocked_resource.append(task_id)
                continue

            # Launch the task.
            self._start_task(state)
            started.append(task_id)

        # Tasks that remain pending but are ready and not started (e.g., no request)
        # are considered idle for this tick; no additional bookkeeping needed.

        # Progress all active tasks through this tick.
        for state in self._active_tasks.values():
            state.remaining -= 1

        resource_usage = {
            name: self._resource_caps[name] - self._resource_available[name]
            for name in self._resource_caps
        }

        record: MutableMapping[str, object] = {
            "time": self.time,
            "requested": list(unique_request_order),
            "started": list(started),
            "completed": list(completed),
            "blocked": list(blocked),
            "blocked_breakdown": {
                "dependency": list(blocked_dependency),
                "resource": list(blocked_resource),
            },
            "invalid": list(invalid),
            "resource_usage": resource_usage,
        }
        self.history.append(record)

        self.time += 1
        return record

    def _advance_time(self) -> List[str]:
        """Complete tasks whose remaining time elapsed and release resources."""
        completed: List[str] = []
        still_active: Dict[str, TaskState] = {}
        for task_id, state in sorted(self._active_tasks.items()):
            if state.remaining <= 0:
                state.status = "done"
                state.end_time = self.time
                completed.append(task_id)
                self._release_resources(state.spec.resources)
            else:
                still_active[task_id] = state
        self._active_tasks = still_active
        return completed

    def _start_task(self, state: TaskState) -> None:
        state.status = "in_progress"
        state.start_time = self.time
        state.remaining = state.spec.duration
        if state.ready_since is None:
            state.ready_since = self.time
        state.wait_time = self.time - state.ready_since
        state.ready_since = None
        self._claim_resources(state.spec.resources)
        self._active_tasks[state.spec.task_id] = state

    def _dependencies_satisfied(self, state: TaskState) -> bool:
        for dep in state.spec.requires_full_ids():
            dep_state = self._tasks.get(dep)
            if dep_state is None or dep_state.status != "done":
                return False
        return True

    def _resources_available(self, requirements: Mapping[str, int]) -> bool:
        for name, amount in requirements.items():
            available = self._resource_available.get(name, 0)
            if amount > available:
                return False
        return True

    def _claim_resources(self, requirements: Mapping[str, int]) -> None:
        for name, amount in requirements.items():
            self._resource_available[name] -= amount

    def _release_resources(self, requirements: Mapping[str, int]) -> None:
        for name, amount in requirements.items():
            self._resource_available[name] += amount

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def done(self) -> bool:
        return all(state.status == "done" for state in self._tasks.values())

    def _ready_task_states(self) -> List[TaskState]:
        ready = [
            state
            for state in self._tasks.values()
            if state.status == "pending" and self._dependencies_satisfied(state)
        ]
        for state in ready:
            if state.ready_since is None:
                state.ready_since = self.time
        ready.sort(key=lambda s: (s.spec.order_id, s.spec.task_id))
        return ready

    def _task_view(self, state: TaskState) -> Dict[str, object]:
        outstanding = [
            dep for dep in state.spec.requires_full_ids() if self._tasks[dep].status != "done"
        ]
        return {
            "task_id": state.spec.task_id,
            "order_id": state.order_id,
            "duration": state.spec.duration,
            "resources": dict(state.spec.resources),
            "outstanding_dependencies": outstanding,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def result(self) -> SimulationResult:
        """Return a summary suitable for downstream metric computation."""
        task_reports = {
            task_id: {
                "order_id": state.order_id,
                "duration": state.spec.duration,
                "resources": dict(state.spec.resources),
                "requires": list(state.spec.requires),
                "start_time": state.start_time,
                "end_time": state.end_time,
                "wait_time": state.wait_time,
                "status": state.status,
            }
            for task_id, state in self._tasks.items()
        }
        return SimulationResult(
            level_name=self.level_name,
            seed=self.seed,
            history=list(self.history),
            task_reports=task_reports,
            resource_caps=dict(self._resource_caps),
            total_time=self.time,
        )


# ----------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------
def load_levels(path: str) -> Dict[str, Mapping[str, object]]:
    """Load level definitions from a JSON-compatible YAML file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    levels = {}
    for level in data.get("levels", []):
        name = str(level["name"])
        levels[name] = level
    return levels


def load_level(path: str, name: str) -> Mapping[str, object]:
    """Convenience wrapper returning a single level definition."""
    levels = load_levels(path)
    if name not in levels:
        raise KeyError(f"Unknown level '{name}' in {path}")
    return levels[name]


def _normalise_resources(resources: Mapping[str, object]) -> Dict[str, int]:
    return {str(name): int(amount) for name, amount in dict(resources).items()}
