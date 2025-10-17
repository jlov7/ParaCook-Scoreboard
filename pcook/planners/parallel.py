"""Parallel planner with resource reservations to avoid starvation."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


class ParallelPlanner:
    """Resource-aware scheduler starting multiple tasks per tick."""

    planner_id = "parallel"

    def __init__(
        self,
        seed: Optional[int] = None,
        reservation_threshold: int = 3,
        fairness_weight: float = 0.75,
    ):
        self.seed = seed or 0
        self.reservation_threshold = reservation_threshold
        self.fairness_weight = fairness_weight
        self.rng = random.Random(self.seed)
        self.task_info: Dict[str, Dict[str, object]] = {}
        self.completed: Set[str] = set()
        self.active: Set[str] = set()
        self.order_last_assigned: Dict[str, int] = {}
        self._current_time: int = 0
        self._fairness_runtime_weight: float = fairness_weight

    def begin_episode(self, level_config: Mapping[str, object], seed: int) -> None:
        self.seed = seed
        self.rng.seed(seed)
        self.task_info = _extract_task_info(level_config)
        self.completed = set()
        self.active = set()
        self.order_last_assigned = {}
        self._current_time = 0
        self._fairness_runtime_weight = self.fairness_weight

    def select_actions(self, observation: Mapping[str, object]) -> Sequence[str]:
        ready: List[Dict[str, object]] = list(observation.get("ready", []))
        in_progress_list: List[Dict[str, object]] = list(observation.get("in_progress", []))
        available: Dict[str, int] = dict(observation.get("resources_available", {}))
        capacities: Dict[str, int] = dict(observation.get("resource_caps", {}))
        self._current_time = int(observation.get("time", 0))

        # Update completion bookkeeping from in-progress snapshot.
        current_active = {item["task_id"] for item in in_progress_list}
        newly_completed = self.active - current_active
        for task_id in newly_completed:
            self.completed.add(task_id)
        self.active = current_active

        ready_ids = {task["task_id"] for task in ready}
        near_ready = self._near_ready_tasks(ready_ids, in_progress_list)
        reservation = self._reservation_budget(ready, near_ready, capacities)
        priority_tasks = set(near_ready)
        for task in ready:
            if int(task["duration"]) >= self.reservation_threshold:
                priority_tasks.add(task["task_id"])

        plan: List[str] = []
        future_release = _resources_releasing(in_progress_list)
        available_working = dict(available)
        reservation_remaining = dict(reservation)
        for name, amount in future_release.items():
            available_working[name] = available_working.get(name, 0) + amount
            reservation_remaining[name] = reservation_remaining.get(name, 0) + amount
        scheduled_ids: Set[str] = set()

        # Deterministic priority: favor long tasks and heavier resource usage.
        tie_breakers = {task["task_id"]: self.rng.random() for task in ready}
        priority_cache: Dict[str, Tuple[object, ...]] = {}

        def priority_tuple(task: Mapping[str, object]) -> Tuple[float, ...]:
            task_id = task["task_id"]
            if task_id not in priority_cache:
                base = self._priority_key(task)
                priority_cache[task_id] = base + (tie_breakers[task_id],)
            return priority_cache[task_id]

        ready.sort(key=priority_tuple)

        # Fairness pass: ensure each order with no active work gets a slot.
        active_counts: Dict[str, int] = {}
        for item in in_progress_list:
            order_id = str(item["order_id"])
            active_counts[order_id] = active_counts.get(order_id, 0) + 1

        ready_by_order: Dict[str, List[Dict[str, object]]] = {}
        for task in ready:
            ready_by_order.setdefault(str(task["order_id"]), []).append(task)

        for tasks in ready_by_order.values():
            tasks.sort(key=priority_tuple)

        orders_to_fill = sorted(
            [
                order
                for order, tasks in ready_by_order.items()
                if tasks and active_counts.get(order, 0) == 0
            ]
        )

        resource_pressure = _resource_pressure(ready, capacities, available, future_release)

        self._fairness_runtime_weight = self._compute_fairness_weight(
            waiting_orders=len(orders_to_fill),
            ready_tasks=len(ready),
            resource_pressure=resource_pressure,
        )

        for order in orders_to_fill:
            assigned = False
            for task in ready_by_order[order]:
                task_id = task["task_id"]
                if task_id in scheduled_ids:
                    continue
                resources = dict(task.get("resources", {}))
                if not self._fits(resources, available_working, reservation_remaining, True):
                    continue
                plan.append(task_id)
                scheduled_ids.add(task_id)
                for name, amount in resources.items():
                    available_working[name] = available_working.get(name, 0) - amount
                    reservation_remaining[name] = max(
                        0, reservation_remaining.get(name, 0) - amount
                    )
                priority_tasks.add(task_id)
                self.order_last_assigned[order] = self._current_time
                assigned = True
                break
            if not assigned:
                continue

        for task in ready:
            task_id = task["task_id"]
            if task_id in scheduled_ids:
                continue
            resources = dict(task.get("resources", {}))
            is_priority = task_id in priority_tasks
            if not self._fits(resources, available_working, reservation_remaining, is_priority):
                continue
            plan.append(task_id)
            for name, amount in resources.items():
                available_working[name] = available_working.get(name, 0) - amount
                reservation_remaining[name] = max(
                    0, reservation_remaining.get(name, 0) - amount
                )
            order = str(task.get("order_id"))
            self.order_last_assigned[order] = self._current_time

        return plan

    def _priority_key(self, task: Mapping[str, object]) -> Tuple[float, int, int, str]:
        task_id = str(task["task_id"])
        duration = int(task["duration"])
        tail = int(self.task_info.get(task_id, {}).get("tail_duration", duration))
        total_resources = sum(int(v) for v in task.get("resources", {}).values())
        outstanding = len(task.get("outstanding_dependencies", []))
        order_id = str(task.get("order_id"))
        wait = self._current_time - self.order_last_assigned.get(order_id, -1)
        effective_weight = self._fairness_runtime_weight
        priority = -(tail - effective_weight * wait)
        return (priority, -total_resources, outstanding, task_id)

    def _near_ready_tasks(
        self,
        ready_ids: Set[str],
        in_progress_list: Sequence[Mapping[str, object]],
    ) -> Set[str]:
        """Estimate tasks that will unlock soon based on dependency status."""
        in_progress_remaining = {
            item["task_id"]: int(item.get("remaining", 0)) for item in in_progress_list
        }
        near_ready: Set[str] = set()

        for task_id, info in self.task_info.items():
            if task_id in self.completed or task_id in ready_ids or task_id in self.active:
                continue

            requires: Sequence[str] = info["requires_full"]
            if not requires:
                continue

            pending_deps: List[str] = []
            for dep in requires:
                if dep in self.completed:
                    continue
                remaining = in_progress_remaining.get(dep)
                if remaining is not None and remaining <= 1:
                    pending_deps.append(dep)
                else:
                    pending_deps = []
                    break

            if pending_deps:
                near_ready.add(task_id)

        return near_ready

    def _reservation_budget(
        self,
        ready_tasks: Sequence[Mapping[str, object]],
        near_ready: Set[str],
        capacities: Mapping[str, int],
    ) -> Dict[str, int]:
        """Reserve resource units for long or soon-to-unlock tasks."""
        reservation = {name: 0 for name in capacities}

        # Reserve for explicitly near-ready tasks.
        for task_id in near_ready:
            info = self.task_info.get(task_id)
            if not info:
                continue
            for res, amount in info["resources"].items():
                if capacities.get(res, 0) > 0 and amount > 0:
                    reservation[res] = min(
                        capacities[res], max(reservation.get(res, 0), min(1, amount))
                    )

        # Also reserve for long ready tasks.
        for task in ready_tasks:
            if int(task["duration"]) < self.reservation_threshold:
                continue
            for res, amount in task.get("resources", {}).items():
                if capacities.get(res, 0) > 0 and amount > 0:
                    reservation[res] = min(
                        capacities[res], max(reservation.get(res, 0), min(1, amount))
                    )

        return reservation

    def _fits(
        self,
        resources: Mapping[str, int],
        available: MutableMapping[str, int],
        reservation: Mapping[str, int],
        is_priority: bool,
    ) -> bool:
        for name, need in resources.items():
            capacity = available.get(name, 0)
            reserve = reservation.get(name, 0)
            if need > capacity:
                return False
            if not is_priority and capacity - need < reserve:
                return False
        return True

    def _compute_fairness_weight(
        self,
        waiting_orders: int,
        ready_tasks: int,
        resource_pressure: float,
    ) -> float:
        base = self.fairness_weight
        if resource_pressure > 0:
            base = base / (1.0 + resource_pressure)
        if waiting_orders <= 0:
            return base * (0.25 if ready_tasks <= 1 else 0.4)
        if waiting_orders == 1:
            return base * 0.6
        return base

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ParallelPlanner(seed={self.seed}, reservation_threshold={self.reservation_threshold})"


def _extract_task_info(level_config: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    task_info: Dict[str, Dict[str, object]] = {}
    for order in level_config.get("orders", []):
        order_id = str(order["id"])
        tasks: Mapping[str, object] = order.get("tasks", {})
        tails = _tail_durations(order)
        for task_local_id in sorted(tasks):
            details = tasks[task_local_id]
            task_id = f"{order_id}:{task_local_id}"
            task_info[task_id] = {
                "requires": list(details.get("requires", [])),
                "requires_full": [f"{order_id}:{dep}" for dep in details.get("requires", [])],
                "resources": dict(details.get("resources", {})),
                "duration": int(details["duration"]),
                "tail_duration": int(tails[task_local_id]),
            }
    return task_info


def _tail_durations(order_config: Mapping[str, object]) -> Dict[str, int]:
    tasks = order_config.get("tasks", {})
    successors: Dict[str, list] = {name: [] for name in tasks}
    for name, details in tasks.items():
        for dep in details.get("requires", []):
            successors.setdefault(dep, []).append(name)

    memo: Dict[str, int] = {}

    def dfs(task_name: str) -> int:
        if task_name in memo:
            return memo[task_name]
        duration = int(tasks[task_name]["duration"])
        children = successors.get(task_name, [])
        if not children:
            memo[task_name] = duration
        else:
            memo[task_name] = duration + max(dfs(child) for child in children)
        return memo[task_name]

    for name in tasks:
        dfs(name)

    return memo


def _resources_releasing(in_progress_list: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    releasing: Dict[str, int] = {}
    for item in in_progress_list:
        remaining = int(item.get("remaining", 0))
        if remaining > 0:
            continue
        for res, amount in item.get("resources", {}).items():
            releasing[res] = releasing.get(res, 0) + int(amount)
    return releasing


def _resource_pressure(
    ready_tasks: Sequence[Mapping[str, object]],
    capacities: Mapping[str, int],
    available: Mapping[str, int],
    future_release: Mapping[str, int],
) -> float:
    pressure = 0.0
    for res, cap in capacities.items():
        cap = int(cap)
        if cap <= 0:
            continue
        demand = sum(1 for task in ready_tasks if task.get("resources", {}).get(res, 0) > 0)
        supply = available.get(res, 0) + future_release.get(res, 0)
        if demand <= supply:
            continue
        pressure = max(pressure, (demand - supply) / cap)
    return pressure
