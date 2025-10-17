import unittest

from pcook.kitchen_env import KitchenEnv, load_levels
from pcook.metrics import compute_metrics, summarize_metrics
from pcook.planners.sequential import SequentialPlanner


def run_once(level_config, seed):
    planner = SequentialPlanner()
    planner.begin_episode(level_config, seed)
    env = KitchenEnv(level_config, seed=seed)
    while not env.done:
        obs = env.observe()
        actions = planner.select_actions(obs)
        env.step(actions)
    return env.result()


class MetricsTest(unittest.TestCase):
    def test_metric_shapes_and_bounds(self):
        levels = load_levels("tasks/levels.yaml")
        level = levels["easy"]
        result = run_once(level, seed=1)
        metrics = compute_metrics(result)

        self.assertGreater(metrics["makespan"], 0)
        self.assertGreaterEqual(metrics["plan_adherence"], 0.0)
        self.assertLessEqual(metrics["plan_adherence"], 1.0)
        self.assertGreaterEqual(metrics["order_completion_time_mean"], 0)
        self.assertTrue({"hands", "prep", "stove", "plating"}.issubset(metrics["resource_utilization"]))
        self.assertIn("blocked_rate", metrics)
        self.assertGreaterEqual(metrics["blocked_rate"], 0.0)
        self.assertLessEqual(metrics["blocked_rate"], 1.0)
        self.assertIn("blocked_by_resource", metrics)
        self.assertIn("blocked_by_dependency", metrics)
        self.assertIn("blocked_resource_rate", metrics)
        self.assertIn("blocked_dependency_rate", metrics)
        self.assertIn("plan_requests", metrics)
        self.assertGreaterEqual(metrics["plan_requests"], 0)
        self.assertIn("task_wait_mean", metrics)
        self.assertGreaterEqual(metrics["task_wait_mean"], 0)
        self.assertIn("task_wait_max", metrics)
        self.assertGreaterEqual(metrics["task_wait_max"], 0)

        summary = summarize_metrics([metrics])
        self.assertEqual(summary["makespan_mean"], metrics["makespan"])
        self.assertEqual(summary["plan_adherence_mean"], metrics["plan_adherence"])
        self.assertIn("makespan_std", summary)
        self.assertIn("oct_mean_mean", summary)
        self.assertIn("blocked_rate_mean", summary)
        self.assertIn("blocked_resource_rate_mean", summary)
        self.assertIn("blocked_dependency_rate_mean", summary)
        self.assertIn("task_wait_mean", summary)
        self.assertIn("task_wait_max_mean", summary)

    def test_summary_handles_oracle_metrics(self):
        sample = {
            "makespan": 20,
            "plan_adherence": 0.9,
            "order_completion_time_mean": 10,
            "resource_utilization": {"hands": 0.5},
            "oracle_gap": 5,
            "oracle_efficiency": 0.75,
            "blocked_rate": 0.2,
            "blocked_resource_rate": 0.05,
            "blocked_dependency_rate": 0.15,
            "task_wait_mean": 1.5,
            "task_wait_max": 3,
        }
        summary = summarize_metrics([sample])
        self.assertAlmostEqual(summary["oracle_gap_mean"], 5)
        self.assertAlmostEqual(summary["oracle_efficiency_mean"], 0.75)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
