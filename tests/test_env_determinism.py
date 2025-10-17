import unittest

from pcook.kitchen_env import KitchenEnv, load_levels
from pcook.planners.sequential import SequentialPlanner


def run_sequence(level_config, seed):
    planner = SequentialPlanner()
    planner.begin_episode(level_config, seed)
    env = KitchenEnv(level_config, seed=seed)
    while not env.done:
        obs = env.observe()
        actions = planner.select_actions(obs)
        env.step(actions)
    return env.result()


class DeterminismTest(unittest.TestCase):
    def test_deterministic_history(self):
        levels = load_levels("tasks/levels.yaml")
        level = levels["easy"]

        first = run_sequence(level, seed=42)
        second = run_sequence(level, seed=42)

        self.assertEqual(first.total_time, second.total_time)
        self.assertEqual(first.history, second.history)
        for task_id in first.task_reports:
            self.assertEqual(
                first.task_reports[task_id]["start_time"],
                second.task_reports[task_id]["start_time"],
            )
            self.assertEqual(
                first.task_reports[task_id]["end_time"],
                second.task_reports[task_id]["end_time"],
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
