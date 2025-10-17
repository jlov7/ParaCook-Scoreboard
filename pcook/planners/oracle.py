"""Oracle baselines for ParaCook Scoreboard."""

from __future__ import annotations

import math
from typing import Mapping


def critical_path_lower_bound(level_config: Mapping[str, object]) -> int:
    """Compute a deterministic lower bound on the makespan."""
    order_bounds = [
        _longest_path_duration(order) for order in level_config.get("orders", [])
    ]
    critical_path = max(order_bounds) if order_bounds else 0

    resource_bound = 0
    resource_work = {}
    for order in level_config.get("orders", []):
        tasks = order.get("tasks", {})
        for details in tasks.values():
            duration = int(details["duration"])
            for name, amount in details.get("resources", {}).items():
                resource_work[name] = resource_work.get(name, 0) + duration * int(amount)

    capacities = {"hands": int(level_config.get("hands", 1))}
    capacities.update({str(k): int(v) for k, v in level_config.get("stations", {}).items()})

    for name, work in resource_work.items():
        cap = capacities.get(name, 0)
        if cap <= 0:
            continue
        resource_bound = max(resource_bound, math.ceil(work / cap))

    return max(critical_path, resource_bound)


def _longest_path_duration(order_config: Mapping[str, object]) -> int:
    tasks = order_config.get("tasks", {})
    durations = {name: int(details["duration"]) for name, details in tasks.items()}
    memo = {}

    def dfs(task_name: str) -> int:
        if task_name in memo:
            return memo[task_name]
        details = tasks[task_name]
        requires = details.get("requires", [])
        if not requires:
            memo[task_name] = durations[task_name]
        else:
            memo[task_name] = durations[task_name] + max(dfs(dep) for dep in requires)
        return memo[task_name]

    longest = 0
    for name in tasks:
        longest = max(longest, dfs(name))
    return longest
