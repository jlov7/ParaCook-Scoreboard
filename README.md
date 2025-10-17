# ParaCook Scoreboard

ParaCook Scoreboard is a reproducible research sandbox for comparing sequential and concurrent planning strategies inside a lightweight, fully deterministic kitchen simulator. The project focuses on **time efficiency under parallel and async constraints**, in line with ParaCook’s research agenda.

## Project layout

- `pcook/kitchen_env.py` – discrete-time simulator modelling orders as DAGs with resource contention (hands + stations).
- `pcook/planners/` – baseline planners:
  - `sequential.py` executes at most one task per tick using a greedy priority.
  - `parallel.py` applies resource-aware batching with a reservation lookahead to avoid starving long tasks.
  - `oracle.py` provides a critical-path lower bound for reference.
- `pcook/metrics.py` – computes order completion time (OCT), makespan, utilization, and plan adherence from simulator logs.
- `pcook/eval_harness.py` – runs experiments across levels and seeds, writing `results.csv` and `summary.md`.
- `tasks/levels.yaml` – reproducible scenario definitions covering easy/medium/hard kitchens.
- `viz/charts.py` – optional chart helpers for OCT distributions, utilization, and planner win-rates. Falls back to text summaries if `matplotlib` is unavailable.
- `tests/` – deterministic unit tests for the simulator, planners, and metrics.
- `examples/demo_run.py` – quick script to run both planners, print a comparison table, and emit charts.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
# Optional chart support
pip install -e .[viz]
pytest
python examples/demo_run.py
```

Both planners and the environment operate deterministically for a given seed. All scenarios in `tasks/levels.yaml` use YAML’s JSON subset so the harness parses them with the Python standard library (`json`).


### CLI

Run a full experiment suite (baseline + jitter) via:

```bash
python -m pcook.cli --levels easy,medium,hard --seeds 10 --duration-jitter 0.25 --resource-jitter 0.25
```

Use `python -m pcook.cli --help` to explore additional options such as custom fairness weights or selective jitter levels.

Aggregated comparison tables (CSV + Markdown) are written to `artifacts/` for quick planner deltas (makespan, OCT, resource-block rate, plan adherence) across baseline and jitter scenarios, and fairness tuning (if enabled) drops its own CSV/Markdown under `artifacts/fairness_tuning/`.

### Scenario perturbations

Research runs can inject controlled variability through the `scenario_modifier` hook on `pcook.eval_harness.run_experiments`. Helpers in `pcook.scenarios` support deterministic ±J% duration jitter (`apply_duration_jitter`) and station/hand capacity jitter (`apply_resource_jitter`) per seed, enabling stress-tests of scheduling heuristics without bloating the static task corpus.

## Metrics

- **Order Completion Time (OCT)** – ticks until every task in an order finishes.
- **Makespan** – total ticks from start to final completion.
- **Agent & Station Utilization** – fraction of ticks each resource remained busy.
- **Plan Adherence** – share of requested actions that execute in the same tick as planned.
- **Oracle Gap & Efficiency** – delta from the critical-path lower bound and the corresponding efficiency ratio.
- **Blocked Rate** – frequency with which planned actions are deferred due to resource or precedence conflicts (with resource/dependency breakdown).
- **Task Wait** – average latency between a task becoming ready and actually starting.

## Limitations

The simulator abstracts away stochastic cooking phenomena (temperature drift, failures) and assumes perfect execution once resources are allocated. The parallel planner implements a single-step lookahead reservation rather than full optimal scheduling. Despite these simplifications, the setup surfaces ParaCook’s core question: when do lightweight concurrency heuristics beat strictly sequential execution under tight kitchen constraints?

## Citation

> ParaCook Research, *ParaCook Scoreboard: Measuring Sequential vs. Concurrent Planning in Deterministic Kitchens*, 2024.
