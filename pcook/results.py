"""Utilities for aggregating ParaCook experiment summaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

DELTA_PATTERN = re.compile(
    r"Δ \(parallel - sequential\): makespan ([+-]?\d+\.\d+), OCT ([+-]?\d+\.\d+), resource-block rate ([+-]?\d+\.\d+)."
)


@dataclass
class ScenarioSummary:
    scenario: str
    level: str
    parallel_makespan: float
    sequential_makespan: float
    parallel_oct: float
    sequential_oct: float
    parallel_block: float
    sequential_block: float
    parallel_adherence: float
    sequential_adherence: float
    delta_makespan: float
    delta_oct: float
    delta_block: float
    delta_adherence: float


def aggregate_summaries(summaries: Mapping[str, Path]) -> List[ScenarioSummary]:
    aggregated: List[ScenarioSummary] = []
    for scenario, path in summaries.items():
        if not path.exists():
            continue
        aggregated.extend(_parse_summary(path, scenario))
    return aggregated


def _parse_summary(path: Path, scenario: str) -> List[ScenarioSummary]:
    lines = path.read_text().splitlines()
    summaries: List[ScenarioSummary] = []
    current_level: Optional[str] = None
    parallel_line: Optional[str] = None
    sequential_line: Optional[str] = None

    for line in lines:
        line = line.strip()
        if line.startswith("## Level:"):
            current_level = line.split(":", 1)[1].strip()
            parallel_line = None
            sequential_line = None
        elif line.startswith("- **parallel**"):
            parallel_line = line
        elif line.startswith("- **sequential**"):
            sequential_line = line
        elif line.startswith("Δ (parallel - sequential)") and current_level and parallel_line and sequential_line:
            match = DELTA_PATTERN.search(line)
            if not match:
                continue
            delta_makespan = float(match.group(1))
            delta_oct = float(match.group(2))
            delta_block = float(match.group(3))

            pm, po, pb, pa = _extract_metrics(parallel_line)
            sm, so, sb, sa = _extract_metrics(sequential_line)

            summaries.append(
                ScenarioSummary(
                    scenario=scenario,
                    level=current_level,
                    parallel_makespan=pm,
                    sequential_makespan=sm,
                    parallel_oct=po,
                    sequential_oct=so,
                    parallel_block=pb,
                    sequential_block=sb,
                    parallel_adherence=pa,
                    sequential_adherence=sa,
                    delta_makespan=delta_makespan,
                    delta_oct=delta_oct,
                    delta_block=delta_block,
                    delta_adherence=pa - sa,
                )
            )
    return summaries


def _extract_metrics(line: str) -> tuple:
    def extract(pattern: str) -> float:
        match = re.search(pattern, line)
        if match:
            return float(match.group(1))
        raise ValueError(f"Pattern {pattern} not found in line: {line}")

    makespan = extract(r"makespan ([0-9.]+)")
    oct_mean = extract(r"OCT mean ([0-9.]+)")
    blocked = extract(r"blocked rate ([0-9.]+)")
    adherence = extract(r"plan adherence ([0-9.]+)")
    return makespan, oct_mean, blocked, adherence


def write_aggregated_csv(aggregated: Sequence[ScenarioSummary], output_path: Path) -> None:
    if not aggregated:
        output_path.write_text("", encoding="utf-8")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "scenario",
        "level",
        "parallel_makespan",
        "sequential_makespan",
        "delta_makespan",
        "parallel_oct",
        "sequential_oct",
        "delta_oct",
        "parallel_block_rate",
        "sequential_block_rate",
        "delta_block_rate",
        "parallel_adherence",
        "sequential_adherence",
        "delta_adherence",
    ]

    lines = [",".join(headers)]
    for item in aggregated:
        lines.append(
            ",".join(
                [
                    item.scenario,
                    item.level,
                    f"{item.parallel_makespan:.2f}",
                    f"{item.sequential_makespan:.2f}",
                    f"{item.delta_makespan:.2f}",
                    f"{item.parallel_oct:.2f}",
                    f"{item.sequential_oct:.2f}",
                    f"{item.delta_oct:.2f}",
                    f"{item.parallel_block:.2f}",
                    f"{item.sequential_block:.2f}",
                    f"{item.delta_block:.2f}",
                    f"{item.parallel_adherence:.2f}",
                    f"{item.sequential_adherence:.2f}",
                    f"{item.delta_adherence:.2f}",
                ]
            )
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_aggregated_markdown(aggregated: Sequence[ScenarioSummary], output_path: Path) -> None:
    if not aggregated:
        output_path.write_text("# Aggregated Summary\n\nNo results available.\n", encoding="utf-8")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Aggregated Summary", ""]
    grouped: Dict[str, List[ScenarioSummary]] = {}
    for item in aggregated:
        grouped.setdefault(item.scenario, []).append(item)

    for scenario, items in grouped.items():
        lines.append(f"## Scenario: {scenario}")
        lines.append("")
        lines.append("| Level | Δ Makespan | Δ OCT | Δ Resource-block | Δ Plan Adherence | Parallel Block | Sequential Block | Parallel Adherence | Sequential Adherence |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        items.sort(key=lambda x: x.level)
        for item in items:
            lines.append(
                f"| {item.level} | {item.delta_makespan:+.2f} | {item.delta_oct:+.2f} | "
                f"{item.delta_block:+.2f} | {item.delta_adherence:+.2f} | {item.parallel_block:.2f} | {item.sequential_block:.2f} | "
                f"{item.parallel_adherence:.2f} | {item.sequential_adherence:.2f} |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_fairness_report(
    evaluations: Iterable[tuple],
    output_path: Path,
) -> None:
    """Create a human-readable summary of fairness weight evaluations."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Fairness Tuning Summary", ""]
    lines.append("| Weight | Resource-block | OCT | Artifacts |")
    lines.append("| ---: | ---: | ---: | --- |")
    for weight, blocked, oct_mean, scenario_path in evaluations:
        lines.append(
            f"| {weight:.3f} | {blocked:.4f} | {oct_mean:.2f} | {scenario_path} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_fairness_csv(
    evaluations: Iterable[tuple],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["weight,blocked_resource_rate,oct_mean,artifacts"]
    for weight, blocked, oct_mean, scenario_path in evaluations:
        rows.append(f"{weight:.3f},{blocked:.4f},{oct_mean:.2f},{scenario_path}")
    output_path.write_text("\n".join(rows), encoding="utf-8")
