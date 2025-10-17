"""Visualization helpers for ParaCook Scoreboard."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    plt = None


def plot_oct_distribution(results: Sequence[Mapping[str, object]], output_dir: str = "charts") -> Path:
    """Plot or summarize order completion time distributions."""
    planner_to_oct = _group_by_planner(results, key="order_completion_time_mean")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if plt is None:
        summary_file = output_path / "oct_distribution.txt"
        lines = ["Matplotlib unavailable. Summary instead:", ""]
        for planner, values in sorted(planner_to_oct.items()):
            if not values:
                continue
            avg = sum(values) / len(values)
            lines.append(f"{planner}: mean OCT {avg:.2f} across {len(values)} runs.")
        summary_file.write_text("\n".join(lines), encoding="utf-8")
        return summary_file

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = max(5, min(20, len(results)))
    for planner, values in sorted(planner_to_oct.items()):
        if not values:
            continue
        ax.hist(values, bins=bins, alpha=0.6, label=planner)
    ax.set_xlabel("Order completion time (mean)")
    ax.set_ylabel("Frequency")
    ax.set_title("OCT distribution by planner")
    ax.legend()
    chart_path = output_path / "oct_distribution.png"
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)
    return chart_path


def plot_utilization(results: Sequence[Mapping[str, object]], output_dir: str = "charts") -> Path:
    """Plot average resource utilization per planner."""
    planner_to_util = _group_resource_utilization(results)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if plt is None:
        summary_file = output_path / "utilization.txt"
        lines = ["Matplotlib unavailable. Summary instead:", ""]
        for planner, util in sorted(planner_to_util.items()):
            formatted = ", ".join(f"{res}={val:.2f}" for res, val in sorted(util.items()))
            lines.append(f"{planner}: {formatted}")
        summary_file.write_text("\n".join(lines), encoding="utf-8")
        return summary_file

    resources = sorted(
        {res for util in planner_to_util.values() for res in util}
    )
    planners = sorted(planner_to_util)
    values = [[planner_to_util[p].get(res, 0.0) for res in resources] for p in planners]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(resources))
    width = 0.8 / max(1, len(planners))

    for idx, planner in enumerate(planners):
        offsets = [val + idx * width for val in x]
        ax.bar(offsets, values[idx], width=width, label=planner)

    ax.set_xticks([val + width * (len(planners) - 1) / 2 for val in x])
    ax.set_xticklabels(resources)
    ax.set_ylabel("Utilization")
    ax.set_ylim(0, 1)
    ax.set_title("Average resource utilization")
    ax.legend()
    chart_path = output_path / "utilization.png"
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)
    return chart_path


def plot_win_rate(
    parallel_results: Sequence[Mapping[str, object]],
    sequential_results: Sequence[Mapping[str, object]],
    output_dir: str = "charts",
) -> Path:
    """Plot the win-rate of parallel planner vs sequential."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    indexed_seq = {(m["level"], m["seed"]): m for m in sequential_results}
    wins = []
    labels = []
    for metrics in parallel_results:
        key = (metrics["level"], metrics["seed"])
        opponent = indexed_seq.get(key)
        if opponent is None:
            continue
        labels.append(f"{metrics['level']}-seed{metrics['seed']}")
        wins.append(
            1 if metrics["order_completion_time_mean"] < opponent["order_completion_time_mean"] else 0
        )

    if plt is None:
        summary_file = output_path / "win_rate.txt"
        win_pct = (sum(wins) / len(wins)) * 100 if wins else 0.0
        summary_file.write_text(
            f"Matplotlib unavailable. Parallel win-rate: {win_pct:.1f}% over {len(wins)} matchups.\n",
            encoding="utf-8",
        )
        return summary_file

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(wins)), wins, color="tab:green")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks(range(len(wins)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Win (1) / Loss (0)")
    ax.set_title("Parallel vs Sequential OCT wins")
    chart_path = output_path / "win_rate.png"
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)
    return chart_path


def _group_by_planner(results: Sequence[Mapping[str, object]], key: str) -> Dict[str, List[float]]:
    grouped: Dict[str, List[float]] = {}
    for metrics in results:
        planner = str(metrics["planner"])
        value = float(metrics[key])
        grouped.setdefault(planner, []).append(value)
    return grouped


def _group_resource_utilization(results: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    for metrics in results:
        planner = str(metrics["planner"])
        util = metrics.get("resource_utilization", {})
        grouped.setdefault(planner, {})
        counts.setdefault(planner, {})
        for res, value in util.items():
            grouped[planner][res] = grouped[planner].get(res, 0.0) + float(value)
            counts[planner][res] = counts[planner].get(res, 0) + 1

    for planner, util in grouped.items():
        for res, total in util.items():
            util[res] = total / counts[planner][res]

    return grouped
