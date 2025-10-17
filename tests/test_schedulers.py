import unittest

from pcook.kitchen_env import KitchenEnv, load_levels
from pcook.metrics import compute_metrics
from pcook.planners.parallel import ParallelPlanner
from pcook.planners.sequential import SequentialPlanner


def run_planner(planner_cls, level_config, seed):
    planner = planner_cls()
    planner.begin_episode(level_config, seed)
    env = KitchenEnv(level_config, seed=seed)
    while not env.done:
        obs = env.observe()
        actions = planner.select_actions(obs)
        env.step(actions)
    return compute_metrics(env.result())


class SchedulerComparisonTest(unittest.TestCase):
    def test_parallel_beats_sequential_on_medium(self):
        levels = load_levels("tasks/levels.yaml")
        level = levels["medium"]

        seeds = list(range(10))
        parallel_wins = 0
        for seed in seeds:
            seq_metrics = run_planner(SequentialPlanner, level, seed)
            par_metrics = run_planner(ParallelPlanner, level, seed)
            if par_metrics["order_completion_time_mean"] < seq_metrics["order_completion_time_mean"]:
                parallel_wins += 1

        self.assertGreaterEqual(parallel_wins, int(0.6 * len(seeds)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
