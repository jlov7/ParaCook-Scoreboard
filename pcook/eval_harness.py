"""Evaluation harness for ParaCook Scoreboard planners."""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .kitchen_env import KitchenEnv, load_levels
from .metrics import compute_metrics, summarize_metrics
from .planners.oracle import critical_path_lower_bound


PlannerFactory = Callable[[], "PlannerProtocol"]


class PlannerProtocol:
    """Protocol describing the planner methods consumed by the harness."""

    planner_id: str

    def begin_episode(self, level_config: Mapping[str, object], seed: int) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def select_actions(self, observation: Mapping[str, object]) -> Sequence[str]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class RunResult:
    planner: str
    level: str
    seed: int
    metrics: Dict[str, object]


def run_episode(planner_factory: PlannerFactory, level_config: Mapping[str, object], seed: int) -> RunResult:
    planner = planner_factory()
    planner.begin_episode(level_config, seed)

    env = KitchenEnv(level_config, seed=seed)
    while not env.done:
        observation = env.observe()
        actions = planner.select_actions(observation)
        env.step(actions)

    result = env.result()
    metrics = compute_metrics(result)
    metrics["planner"] = planner.planner_id
    oracle_bound = critical_path_lower_bound(level_config)
    metrics["oracle_lower_bound"] = oracle_bound
    metrics["oracle_gap"] = metrics["makespan"] - oracle_bound
    if metrics["makespan"]:
        metrics["oracle_efficiency"] = oracle_bound / metrics["makespan"]
    else:
        metrics["oracle_efficiency"] = 0.0

    return RunResult(planner=planner.planner_id, level=result.level_name, seed=seed, metrics=metrics)


def run_experiments(
    levels_path: str,
    planner_factories: Mapping[str, PlannerFactory],
    level_names: Sequence[str],
    seeds: Sequence[int],
    output_dir: str = ".",
    scenario_modifier: Optional[Callable[[Mapping[str, object], int], Mapping[str, object]]] = None,
) -> List[RunResult]:
    levels = load_levels(levels_path)
    missing = [name for name in level_names if name not in levels]
    if missing:
        raise KeyError(f"Missing levels: {', '.join(missing)}")

    results: List[RunResult] = []
    for level_name in level_names:
        level_config = levels[level_name]
        for seed in seeds:
            level_variant = (
                scenario_modifier(level_config, seed) if scenario_modifier else level_config
            )
            for planner_id, factory in planner_factories.items():
                run = run_episode(factory, level_variant, seed)
                results.append(run)

    write_results_csv(results, output_dir)
    write_summary(results, output_dir)
    return results


def write_results_csv(results: Sequence[RunResult], output_dir: str) -> Path:
    output_path = Path(output_dir) / "results.csv"
    if not results:
        output_path.write_text("", encoding="utf-8")
        return output_path

    headers = set()
    rows = []
    for run in results:
        row = flatten_metrics(run)
        headers.update(row.keys())
        rows.append(row)

    ordered_headers = ["planner", "level", "seed"]
    ordered_headers.extend(sorted(h for h in headers if h not in ordered_headers))

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ordered_headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return output_path


def write_summary(results: Sequence[RunResult], output_dir: str) -> Path:
    output_path = Path(output_dir) / "summary.md"
    if not results:
        output_path.write_text("# Summary\n\nNo runs executed.\n", encoding="utf-8")
        return output_path

    by_level: Dict[str, List[RunResult]] = {}
    for run in results:
        by_level.setdefault(run.level, []).append(run)

    lines = ["# ParaCook Scoreboard Summary", ""]

    for level, runs in sorted(by_level.items()):
        lines.append(f"## Level: {level}")
        planner_groups: Dict[str, List[Mapping[str, object]]] = {}
        for run in runs:
            planner_groups.setdefault(run.planner, []).append(run.metrics)

        for planner, metrics_list in sorted(planner_groups.items()):
            aggregate = summarize_metrics(metrics_list)
            avg_oct = aggregate.get("oct_mean_mean", 0.0)
            gap = aggregate.get("oracle_gap_mean")
            eff = aggregate.get("oracle_efficiency_mean")
            text = (
                f"- **{planner}**: makespan {aggregate['makespan_mean']:.2f}±{aggregate['makespan_std']:.2f}, "
                f"OCT mean {avg_oct:.2f}±{aggregate['oct_mean_std']:.2f}, "
                f"plan adherence {aggregate['plan_adherence_mean']:.2f}, "
                f"blocked rate {aggregate['blocked_rate_mean']:.2f} "
                f"(resource {aggregate['blocked_resource_rate_mean']:.2f}, "
                f"dependency {aggregate['blocked_dependency_rate_mean']:.2f}), "
                f"avg wait {aggregate['task_wait_mean']:.2f}"
            )
            if gap is not None and eff is not None:
                text += f", oracle gap {gap:.2f}, efficiency {eff:.2f}"
            lines.append(text)

        if "sequential" in planner_groups and "parallel" in planner_groups:
            par_summary = summarize_metrics(planner_groups["parallel"])
            seq_summary = summarize_metrics(planner_groups["sequential"])
            delta_makespan = par_summary["makespan_mean"] - seq_summary["makespan_mean"]
            delta_oct = par_summary["oct_mean_mean"] - seq_summary["oct_mean_mean"]
            delta_block_res = (
                par_summary["blocked_resource_rate_mean"] - seq_summary["blocked_resource_rate_mean"]
            )
            lines.append("")
            win_rate = compute_win_rate(planner_groups["parallel"], planner_groups["sequential"])
            lines.append(
                f"Parallel beats sequential on OCT in {win_rate * 100:.1f}% of seeds."
            )
            lines.append(
                f"Δ (parallel - sequential): makespan {delta_makespan:+.2f}, "
                f"OCT {delta_oct:+.2f}, resource-block rate {delta_block_res:+.2f}."
            )

        lines.append("")

    interpretation = interpret_results(by_level)
    lines.append("## Interpretation")
    lines.append("")
    lines.append(interpretation)

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def flatten_metrics(run: RunResult) -> Dict[str, object]:
    metrics = dict(run.metrics)
    row: Dict[str, object] = {
        "planner": run.planner,
        "level": run.level,
        "seed": run.seed,
        "makespan": metrics["makespan"],
        "oct_mean": metrics["order_completion_time_mean"],
        "plan_adherence": metrics["plan_adherence"],
        "blocked": metrics["blocked_actions"],
        "blocked_rate": f"{metrics.get('blocked_rate', 0.0):.4f}",
        "blocked_by_resource": metrics.get("blocked_by_resource", 0),
        "blocked_by_dependency": metrics.get("blocked_by_dependency", 0),
        "blocked_resource_rate": f"{metrics.get('blocked_resource_rate', 0.0):.4f}",
        "blocked_dependency_rate": f"{metrics.get('blocked_dependency_rate', 0.0):.4f}",
        "plan_requests": metrics.get("plan_requests", 0),
        "task_wait_mean": f"{metrics.get('task_wait_mean', 0.0):.2f}",
        "task_wait_max": metrics.get("task_wait_max", 0),
    }
    if "oracle_lower_bound" in metrics:
        row["oracle_lower_bound"] = metrics["oracle_lower_bound"]
    if "oracle_gap" in metrics:
        row["oracle_gap"] = metrics["oracle_gap"]
    if "oracle_efficiency" in metrics:
        row["oracle_efficiency"] = f"{metrics['oracle_efficiency']:.4f}"

    for order_id, oct_value in sorted(metrics["order_completion_time"].items()):
        row[f"oct_{order_id}"] = oct_value

    for resource, value in sorted(metrics["resource_utilization"].items()):
        row[f"util_{resource}"] = f"{value:.4f}"

    return row


def compute_win_rate(
    parallel_metrics: Sequence[Mapping[str, object]],
    sequential_metrics: Sequence[Mapping[str, object]],
) -> float:
    indexed_seq = {(m["level"], m["seed"]): m for m in sequential_metrics}
    wins = 0
    total = 0
    for parallel in parallel_metrics:
        key = (parallel["level"], parallel["seed"])
        sequential = indexed_seq.get(key)
        if sequential is None:
            continue
        total += 1
        if parallel["order_completion_time_mean"] < sequential["order_completion_time_mean"]:
            wins += 1
    return wins / total if total else 0.0


def interpret_results(by_level: Mapping[str, Sequence[RunResult]]) -> str:
    if not by_level:
        return "No evidence gathered."

    fragments = []
    for level, runs in sorted(by_level.items()):
        parallel = [run.metrics for run in runs if run.planner == "parallel"]
        sequential = [run.metrics for run in runs if run.planner == "sequential"]
        if not parallel or not sequential:
            continue

        parallel_oct = statistics.mean(m["order_completion_time_mean"] for m in parallel)
        sequential_oct = statistics.mean(m["order_completion_time_mean"] for m in sequential)
        delta = sequential_oct - parallel_oct
        win_rate = compute_win_rate(parallel, sequential)
        utilization_gain = describe_utilization_delta(parallel, sequential)
        par_gap = statistics.mean(m.get("oracle_gap", 0.0) for m in parallel)
        seq_gap = statistics.mean(m.get("oracle_gap", 0.0) for m in sequential)
        par_blocked_rate = statistics.mean(m.get("blocked_rate", 0.0) for m in parallel)
        seq_blocked_rate = statistics.mean(m.get("blocked_rate", 0.0) for m in sequential)
        par_wait = statistics.mean(m.get("task_wait_mean", 0.0) for m in parallel)
        seq_wait = statistics.mean(m.get("task_wait_mean", 0.0) for m in sequential)
        par_blocked_resource = statistics.mean(m.get("blocked_resource_rate", 0.0) for m in parallel)
        seq_blocked_resource = statistics.mean(m.get("blocked_resource_rate", 0.0) for m in sequential)

        if delta > 0:
            verdict = (
                f"Parallel beats sequential by {delta:.2f} ticks on {level}, winning {win_rate * 100:.1f}% "
                f"of seeds with an average oracle gap of {par_gap:.2f} ticks (vs {seq_gap:.2f})."
            )
        elif delta < 0:
            verdict = (
                f"Sequential retains an {abs(delta):.2f}-tick edge on {level}, taking {100 - win_rate * 100:.1f}% "
                f"of matchups while both planners sit roughly {par_gap:.2f}/{seq_gap:.2f} ticks above the oracle bound."
            )
        else:
            verdict = f"Both planners tie on {level}, landing {par_gap:.2f} ticks above the oracle lower bound."

        if par_wait < seq_wait:
            wait_clause = f"and average task wait drops from {seq_wait:.2f} to {par_wait:.2f}."
        elif par_wait > seq_wait:
            wait_clause = f"and average task wait rises from {seq_wait:.2f} to {par_wait:.2f}."
        else:
            wait_clause = f"and average task wait remains at {par_wait:.2f}."

        contention_note = (
            f" Blocked rate shifts from {seq_blocked_rate:.2f} (sequential) to {par_blocked_rate:.2f} (parallel), "
            f"{wait_clause} Resource-block rate moves from {seq_blocked_resource:.2f} to {par_blocked_resource:.2f}."
        )

        fragments.append(f"{verdict} {utilization_gain}{contention_note}")

    if not fragments:
        return "Planner comparisons were inconclusive across evaluated levels."

    return " ".join(fragments)


def describe_utilization_delta(
    parallel: Sequence[Mapping[str, object]],
    sequential: Sequence[Mapping[str, object]],
) -> str:
    if not parallel or not sequential:
        return "Utilization comparison is inconclusive."

    parallel_util = average_util(parallel)
    sequential_util = average_util(sequential)
    parts = []
    for resource in sorted(set(parallel_util) | set(sequential_util)):
        p_val = parallel_util.get(resource, 0.0)
        s_val = sequential_util.get(resource, 0.0)
        delta = p_val - s_val
        parts.append(f"{resource}: Δ{delta:+.2f}")

    return "Utilization shifts " + ", ".join(parts) + "."


def average_util(metrics_seq: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    aggregate: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for metrics in metrics_seq:
        for res, value in metrics.get("resource_utilization", {}).items():
            aggregate[res] = aggregate.get(res, 0.0) + float(value)
            counts[res] = counts.get(res, 0) + 1
    return {res: aggregate[res] / counts[res] for res in aggregate}
