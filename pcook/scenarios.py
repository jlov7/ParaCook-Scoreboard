"""Scenario perturbation helpers for ParaCook Scoreboard."""

from __future__ import annotations

import copy
import random
from typing import Mapping, MutableMapping


def apply_duration_jitter(
    level_config: Mapping[str, object],
    seed: int,
    jitter: float = 0.2,
) -> Mapping[str, object]:
    """Return a deep-copied level config with deterministic duration jitter.

    Args:
        level_config: Base scenario definition.
        seed: Evaluation seed; governs perturbation RNG.
        jitter: Maximum fractional deviation applied to each task duration.
    """
    jitter = max(0.0, float(jitter))
    cloned = copy.deepcopy(level_config)
    level_name = str(level_config.get("name", "level"))
    rng_seed = _stable_seed(level_name, seed)
    rng = random.Random(rng_seed)

    lower = max(0.0, 1.0 - jitter)
    upper = 1.0 + jitter

    for order in cloned.get("orders", []):
        tasks: MutableMapping[str, MutableMapping[str, object]] = order.get("tasks", {})
        for task in tasks.values():
            base_duration = int(task["duration"])
            if base_duration <= 0:
                continue
            factor = rng.uniform(lower, upper)
            task["duration"] = max(1, int(round(base_duration * factor)))

    cloned["name"] = f"{level_name}_jitter"
    return cloned


def apply_resource_jitter(
    level_config: Mapping[str, object],
    seed: int,
    jitter: float = 0.25,
) -> Mapping[str, object]:
    """Return a level variant with station/hand capacities jittered per seed."""
    jitter = max(0.0, float(jitter))
    cloned = copy.deepcopy(level_config)
    level_name = str(level_config.get("name", "level"))
    rng = random.Random(_stable_seed(level_name + "_resources", seed))

    hands = int(cloned.get("hands", 0))
    if hands:
        cloned["hands"] = max(1, int(round(hands * rng.uniform(1.0 - jitter, 1.0 + jitter))))

    stations = dict(cloned.get("stations", {}))
    for name, capacity in stations.items():
        new_capacity = max(1, int(round(int(capacity) * rng.uniform(1.0 - jitter, 1.0 + jitter))))
        stations[name] = new_capacity
    cloned["stations"] = stations

    cloned["name"] = f"{level_name}_resjitter"
    return cloned


def _stable_seed(level_name: str, seed: int) -> int:
    return (hash((level_name, seed)) & 0xFFFFFFFF) ^ (seed << 16 & 0xFFFFFFFF)
