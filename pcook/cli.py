"""Command-line interface for ParaCook Scoreboard experiments."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .eval_harness import run_experiments
from .planners.parallel import ParallelPlanner
from .planners.sequential import SequentialPlanner
from .results import (
    aggregate_summaries,
    write_aggregated_csv,
    write_aggregated_markdown,
    write_fairness_csv,
    write_fairness_report,
)
from .scenarios import apply_duration_jitter, apply_resource_jitter
from viz.charts import plot_oct_distribution, plot_utilization, plot_win_rate


def parse_levels(spec: str) -> List[str]:
    if not spec:
        return []
    return [part for part in (item.strip() for item in spec.split(",")) if part]


def parse_weights(spec: str) -> List[float]:
    weights: List[float] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        weights.append(float(chunk))
    return weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ParaCook Scoreboard CLI")
    parser.add_argument(
        "--levels",
        default="easy,medium,hard",
        help="Comma-separated level names to evaluate (baseline run).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of seeds to evaluate (uses range(seeds)).",
    )
    parser.add_argument(
        "--duration-jitter",
        type=float,
        default=0.25,
        help="Apply ±J duration jitter (set to 0 to disable).",
    )
    parser.add_argument(
        "--duration-levels",
        default="medium",
        help="Levels to apply duration jitter to (comma-separated). Empty to skip.",
    )
    parser.add_argument(
        "--resource-jitter",
        type=float,
        default=0.25,
        help="Apply ±J resource-cap jitter (set to 0 to disable).",
    )
    parser.add_argument(
        "--resource-levels",
        default="medium",
        help="Levels to apply resource jitter to (comma-separated). Empty to skip.",
    )
    parser.add_argument(
        "--fairness-weight",
        type=float,
        default=0.75,
        help="Base fairness weight for the parallel planner.",
    )
    parser.add_argument(
        "--tune-fairness",
        action="store_true",
        help="Grid-search fairness weights before running main scenarios.",
    )
    parser.add_argument(
        "--tune-weights",
        default="0.5,0.75,0.9,1.1",
        help="Comma-separated fairness weights to evaluate when tuning.",
    )
    parser.add_argument(
        "--tune-levels",
        default="medium",
        help="Levels to use during fairness tuning (comma-separated).",
    )
    parser.add_argument(
        "--tune-scenario",
        choices=["resource", "duration"],
        default="resource",
        help="Scenario modifier applied while tuning fairness.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("."),
        help="Base directory for artifacts (baseline results written here).",
    )
    parser.add_argument(
        "--levels-file",
        type=Path,
        default=Path("tasks/levels.yaml"),
        help="Path to the levels definition file.",
    )
    return parser


def build_planner_factories(fairness_weight: float) -> Mapping[str, Callable[[], object]]:
    return {
        "sequential": SequentialPlanner,
        "parallel": lambda: ParallelPlanner(fairness_weight=fairness_weight),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    baseline_levels = parse_levels(args.levels)
    if not baseline_levels:
        raise SystemExit("No baseline levels specified.")

    duration_levels = parse_levels(args.duration_levels)
    resource_levels = parse_levels(args.resource_levels)

    seeds = list(range(max(1, args.seeds)))
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"

    fairness_weight = args.fairness_weight
    tuning_artifacts: List[Tuple[str, Path]] = []

    if args.tune_fairness:
        candidate_weights = parse_weights(args.tune_weights)
        if not candidate_weights:
            raise SystemExit("No fairness weights specified for tuning.")
        tune_levels = parse_levels(args.tune_levels) or baseline_levels
        scenario_type = args.tune_scenario
        if scenario_type == "resource":
            jitter_value = args.resource_jitter if args.resource_jitter > 0 else 0.25
            modifier = lambda level, seed, jitter=jitter_value: apply_resource_jitter(level, seed, jitter)
            scenario_label = f"resource_jitter_{jitter_value:.0%}"
        else:
            jitter_value = args.duration_jitter if args.duration_jitter > 0 else 0.25
            modifier = lambda level, seed, jitter=jitter_value: apply_duration_jitter(level, seed, jitter)
            scenario_label = f"duration_jitter_{jitter_value:.0%}"

        tuning_dir = artifacts_dir / "fairness_tuning" / scenario_label
        tuning_dir.mkdir(parents=True, exist_ok=True)
        best, evaluations = tune_fairness_weight(
            levels_path=str(args.levels_file.resolve()),
            levels=tune_levels,
            seeds=seeds,
            weights=candidate_weights,
            scenario_modifier=modifier,
            output_dir=tuning_dir,
        )
        fairness_weight = best[0]

        print("\nFairness tuning results:")
        for weight, blocked, oct_mean, eval_dir in evaluations:
            print(
                f"  weight {weight:.2f}: resource-block {blocked:.4f}, OCT {oct_mean:.2f} (results at {eval_dir})"
            )
        print(f"Selected fairness weight: {fairness_weight:.2f}")
        write_fairness_csv(evaluations, tuning_dir / 'fairness.csv')
        write_fairness_report(evaluations, tuning_dir / 'fairness.md')
        tuning_artifacts.append((scenario_label, tuning_dir))

    planner_factories = build_planner_factories(fairness_weight)
    planner_ids = list(planner_factories.keys())

    scenarios: List[ScenarioSpec] = [
        ScenarioSpec(
            title=f"ParaCook results ({len(seeds)} seeds)",
            level_names=baseline_levels,
            scenario_modifier=None,
            output_path=output_dir,
            slug='baseline',
        )
    ]

    if args.duration_jitter > 0 and duration_levels:
        duration_path = artifacts_dir / "duration_jitter"
        duration_path.mkdir(parents=True, exist_ok=True)
        scenarios.append(
            ScenarioSpec(
                title=f"Duration jitter ±{args.duration_jitter:.0%}",
                level_names=duration_levels,
                scenario_modifier=lambda level, seed, jitter=args.duration_jitter: apply_duration_jitter(
                    level, seed, jitter
                ),
                output_path=duration_path,
                slug='duration_jitter',
            )
        )

    if args.resource_jitter > 0 and resource_levels:
        resource_path = artifacts_dir / "resource_jitter"
        resource_path.mkdir(parents=True, exist_ok=True)
        scenarios.append(
            ScenarioSpec(
                title=f"Resource jitter ±{args.resource_jitter:.0%}",
                level_names=resource_levels,
                scenario_modifier=lambda level, seed, jitter=args.resource_jitter: apply_resource_jitter(
                    level, seed, jitter
                ),
                output_path=resource_path,
                slug='resource_jitter',
            )
        )

    levels_path = args.levels_file.resolve()

    charts = {}

    summary_map = {}
    for index, scenario in enumerate(scenarios):
        results = run_experiments(
            str(levels_path),
            planner_factories=planner_factories,
            level_names=scenario.level_names,
            seeds=seeds,
            output_dir=str(scenario.output_path),
            scenario_modifier=scenario.scenario_modifier,
        )
        metrics = [run.metrics for run in results]
        scenario_levels = sorted({metric["level"] for metric in metrics})
        print()
        print_tables(
            title=scenario.title,
            level_names=scenario_levels,
            metrics=metrics,
            planner_ids=planner_ids,
        )
        if index == 0:
            charts_dir = output_dir / "charts"
            charts_dir.mkdir(parents=True, exist_ok=True)
            charts["oct"] = plot_oct_distribution(metrics, output_dir=str(charts_dir))
            charts["util"] = plot_utilization(metrics, output_dir=str(charts_dir))
            parallel_metrics = [m for m in metrics if m["planner"] == "parallel"]
            sequential_metrics = [m for m in metrics if m["planner"] == "sequential"]
            charts["win"] = plot_win_rate(parallel_metrics, sequential_metrics, output_dir=str(charts_dir))

        summary_map[scenario.slug] = scenario.output_path / 'summary.md'
    print("\nArtifacts written:")
    aggregated = aggregate_summaries(summary_map)
    write_aggregated_csv(aggregated, artifacts_dir / 'aggregated.csv')
    write_aggregated_markdown(aggregated, artifacts_dir / 'aggregated.md')
    print(f"- baseline results.csv at {output_dir / 'results.csv'}")
    print(f"- baseline summary.md at {output_dir / 'summary.md'}")
    print(f"- aggregated comparison (CSV) at {artifacts_dir / 'aggregated.csv'}")
    print(f"- aggregated comparison (Markdown) at {artifacts_dir / 'aggregated.md'}")
    if charts:
        print(f"- OCT chart: {charts['oct']}")
        print(f"- Utilization chart: {charts['util']}")
        print(f"- Win-rate chart: {charts['win']}")
    for scenario in scenarios[1:]:
        print(f"- {scenario.title.lower()} results.csv at {scenario.output_path / 'results.csv'}")
        print(f"- {scenario.title.lower()} summary.md at {scenario.output_path / 'summary.md'}")
    for label, path in tuning_artifacts:
        print(f"- fairness tuning ({label}) artifacts at {path}")


def print_tables(
    title: str,
    level_names: Iterable[str],
    metrics: Sequence[Mapping[str, object]],
    planner_ids: Sequence[str],
) -> None:
    from collections import defaultdict

    print(title)
    print("-" * len(title))
    grouped = defaultdict(list)
    for metric in metrics:
        grouped[(metric["planner"], metric["level"])] += [metric]

    for level in level_names:
        print(f"\nLevel: {level}")
        print("planner    OCT mean  Makespan  Adherence  Blocked")
        for planner in planner_ids:
            planner_metrics = grouped.get((planner, level), [])
            if not planner_metrics:
                continue
            oct_mean = sum(m["order_completion_time_mean"] for m in planner_metrics) / len(planner_metrics)
            makespan = sum(m["makespan"] for m in planner_metrics) / len(planner_metrics)
            adherence = sum(m["plan_adherence"] for m in planner_metrics) / len(planner_metrics)
            blocked = sum(m.get("blocked_rate", 0.0) for m in planner_metrics) / len(planner_metrics)
            print(f"{planner:<10} {oct_mean:8.2f}  {makespan:8.2f}  {adherence:9.2f}  {blocked:7.2f}")


class ScenarioSpec:
    def __init__(
        self,
        title: str,
        level_names: Sequence[str],
        scenario_modifier: Optional[Callable[[Mapping[str, object], int], Mapping[str, object]]],
        output_path: Path,
        slug: Optional[str] = None,
    ) -> None:
        self.title = title
        self.level_names = list(level_names)
        self.scenario_modifier = scenario_modifier
        self.output_path = output_path
        self.slug = slug or _slugify(title)


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip('_')


def tune_fairness_weight(
    *,
    levels_path: str,
    levels: Sequence[str],
    seeds: Sequence[int],
    weights: Sequence[float],
    scenario_modifier: Optional[Callable[[Mapping[str, object], int], Mapping[str, object]]],
    output_dir: Path,
) -> Tuple[Tuple[float, float, float], List[Tuple[float, float, float, Path]]]:
    evaluations: List[Tuple[float, float, float, Path]] = []
    best: Optional[Tuple[float, float, float]] = None

    for weight in weights:
        scenario_path = output_dir / f"weight_{str(weight).replace('.', '_')}"
        scenario_path.mkdir(parents=True, exist_ok=True)
        runs = run_experiments(
            levels_path,
            planner_factories={"parallel": lambda w=weight: ParallelPlanner(fairness_weight=w)},
            level_names=levels,
            seeds=seeds,
            output_dir=str(scenario_path),
            scenario_modifier=scenario_modifier,
        )
        metrics = [run.metrics for run in runs]
        blocked = sum(m.get("blocked_resource_rate", 0.0) for m in metrics) / len(metrics)
        oct_mean = sum(m["order_completion_time_mean"] for m in metrics) / len(metrics)
        evaluations.append((weight, blocked, oct_mean, scenario_path))
        if best is None or blocked < best[1] or (abs(blocked - best[1]) < 1e-9 and oct_mean < best[2]):
            best = (weight, blocked, oct_mean)

    assert best is not None
    return best, evaluations


if __name__ == "__main__":  # pragma: no cover
    main()
