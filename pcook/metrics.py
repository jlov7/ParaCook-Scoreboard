"""Metric computations for ParaCook Scoreboard."""

from __future__ import annotations

import statistics
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .kitchen_env import SimulationResult


def compute_metrics(result: SimulationResult) -> Dict[str, object]:
    """Aggregate metrics for a single simulator rollout."""
    order_oct: Dict[str, int] = {}
    for task_id, info in result.task_reports.items():
        order_id = str(info["order_id"])
        end_time = info.get("end_time")
        if end_time is None:
            raise ValueError(f"Task {task_id} never completed; schedule incomplete.")
        order_oct[order_id] = max(order_oct.get(order_id, 0), int(end_time))

    makespan = max(order_oct.values(), default=0)
    tick_horizon = max(1, makespan)

    resource_busy_time: Dict[str, int] = {res: 0 for res in result.resource_caps}
    for record in result.history:
        time_point = int(record["time"])
        if time_point >= tick_horizon:
            # Post-completion bookkeeping; ignore for utilization.
            continue
        usage: MutableMapping[str, int] = record.get("resource_usage", {})
        for res, value in usage.items():
            resource_busy_time[res] = resource_busy_time.get(res, 0) + int(value)

    utilization: Dict[str, float] = {}
    for res, cap in result.resource_caps.items():
        cap = int(cap)
        if cap <= 0:
            utilization[res] = 0.0
            continue
        busy = resource_busy_time.get(res, 0)
        utilization[res] = busy / (cap * tick_horizon)

    plan_requested = 0
    plan_matched = 0
    blocked_actions = 0
    blocked_resource = 0
    blocked_dependency = 0
    wait_times: List[int] = []
    for info in result.task_reports.values():
        wait_times.append(int(info.get("wait_time", 0)))

    for record in result.history:
        requested = set(record.get("requested", []))
        started = set(record.get("started", []))
        if not requested:
            continue
        plan_requested += len(requested)
        plan_matched += len(requested & started)
        blocked_actions += len(record.get("blocked", []))
        breakdown = record.get("blocked_breakdown", {})
        blocked_resource += len(breakdown.get("resource", [])) if isinstance(breakdown, dict) else 0
        blocked_dependency += len(breakdown.get("dependency", [])) if isinstance(breakdown, dict) else 0

    plan_adherence = 1.0 if plan_requested == 0 else plan_matched / plan_requested
    blocked_rate = 0.0 if plan_requested == 0 else blocked_actions / plan_requested
    blocked_resource_rate = 0.0 if plan_requested == 0 else blocked_resource / plan_requested
    blocked_dependency_rate = 0.0 if plan_requested == 0 else blocked_dependency / plan_requested
    avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0.0
    max_wait = max(wait_times) if wait_times else 0

    return {
        "level": result.level_name,
        "seed": result.seed,
        "makespan": makespan,
        "order_completion_time": order_oct,
        "order_completion_time_mean": (
            sum(order_oct.values()) / len(order_oct) if order_oct else 0.0
        ),
        "resource_utilization": utilization,
        "plan_adherence": plan_adherence,
        "blocked_actions": blocked_actions,
        "blocked_rate": blocked_rate,
        "blocked_by_resource": blocked_resource,
        "blocked_by_dependency": blocked_dependency,
        "blocked_resource_rate": blocked_resource_rate,
        "blocked_dependency_rate": blocked_dependency_rate,
        "plan_requests": plan_requested,
        "task_wait_mean": avg_wait,
        "task_wait_max": max_wait,
    }


def summarize_metrics(metrics_list: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Compute averages across multiple metric dictionaries."""
    if not metrics_list:
        return {}

    makespans = [float(item["makespan"]) for item in metrics_list]
    adherence = [float(item["plan_adherence"]) for item in metrics_list]
    oct_means = [float(item["order_completion_time_mean"]) for item in metrics_list]
    blocked_rates = [float(item.get("blocked_rate", 0.0)) for item in metrics_list]
    wait_means = [float(item.get("task_wait_mean", 0.0)) for item in metrics_list]
    wait_maxima = [float(item.get("task_wait_max", 0.0)) for item in metrics_list]
    blocked_resource_rates = [
        float(item.get("blocked_resource_rate", 0.0)) for item in metrics_list
    ]
    blocked_dependency_rates = [
        float(item.get("blocked_dependency_rate", 0.0)) for item in metrics_list
    ]

    aggregated_util: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    oracle_gap_total = 0.0
    oracle_eff_total = 0.0
    oracle_samples = 0

    for item in metrics_list:
        for res, value in item.get("resource_utilization", {}).items():
            aggregated_util[res] = aggregated_util.get(res, 0.0) + float(value)
            counts[res] = counts.get(res, 0) + 1
        if "oracle_gap" in item:
            oracle_gap_total += float(item["oracle_gap"])
            oracle_eff_total += float(item.get("oracle_efficiency", 0.0))
            oracle_samples += 1

    avg_util = {
        res: (aggregated_util[res] / counts[res]) if counts[res] else 0.0
        for res in aggregated_util
    }

    summary = {
        "makespan_mean": sum(makespans) / len(makespans),
        "makespan_std": statistics.pstdev(makespans) if len(makespans) > 1 else 0.0,
        "oct_mean_mean": sum(oct_means) / len(oct_means),
        "oct_mean_std": statistics.pstdev(oct_means) if len(oct_means) > 1 else 0.0,
        "plan_adherence_mean": sum(adherence) / len(adherence),
        "resource_utilization_mean": avg_util,
        "blocked_rate_mean": sum(blocked_rates) / len(blocked_rates),
        "blocked_resource_rate_mean": sum(blocked_resource_rates) / len(blocked_resource_rates),
        "blocked_dependency_rate_mean": sum(blocked_dependency_rates) / len(blocked_dependency_rates),
        "task_wait_mean": sum(wait_means) / len(wait_means),
        "task_wait_max_mean": sum(wait_maxima) / len(wait_maxima),
    }

    if oracle_samples:
        summary["oracle_gap_mean"] = oracle_gap_total / oracle_samples
        summary["oracle_efficiency_mean"] = oracle_eff_total / oracle_samples

    return summary
