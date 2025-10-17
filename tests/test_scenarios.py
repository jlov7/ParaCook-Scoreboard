import copy
import unittest

from pcook.kitchen_env import load_levels
from pcook.scenarios import apply_duration_jitter, apply_resource_jitter


class ScenarioPerturbationTest(unittest.TestCase):
    def test_duration_jitter_is_deterministic_and_non_mutating(self):
        levels = load_levels("tasks/levels.yaml")
        base = levels["medium"]
        base_copy = copy.deepcopy(base)

        jitter_a = apply_duration_jitter(base, seed=123, jitter=0.3)
        jitter_b = apply_duration_jitter(base, seed=123, jitter=0.3)
        jitter_c = apply_duration_jitter(base, seed=124, jitter=0.3)

        self.assertEqual(base, base_copy, "apply_duration_jitter must not mutate the original level")
        self.assertEqual(jitter_a, jitter_b)
        self.assertNotEqual(jitter_a, jitter_c)

    def test_resource_jitter_changes_capacities_but_not_orders(self):
        levels = load_levels("tasks/levels.yaml")
        base = levels["medium"]
        jitter = 0.3
        variant = apply_resource_jitter(base, seed=42, jitter=jitter)

        self.assertEqual(set(base["stations"]), set(variant["stations"]))
        self.assertTrue(all(val >= 1 for val in variant["stations"].values()))
        self.assertEqual(base["orders"], variant["orders"])

        min_hands = max(1, int(round(base["hands"] * (1.0 - jitter))))
        max_hands = max(1, int(round(base["hands"] * (1.0 + jitter))))
        self.assertTrue(min_hands <= variant["hands"] <= max_hands)

        for name, capacity in base["stations"].items():
            min_capacity = max(1, int(round(int(capacity) * (1.0 - jitter))))
            max_capacity = max(1, int(round(int(capacity) * (1.0 + jitter))))
            self.assertTrue(min_capacity <= variant["stations"][name] <= max_capacity)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
