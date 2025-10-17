
import unittest
import tempfile
from pathlib import Path

from pcook.cli import parse_levels, parse_weights, tune_fairness_weight
from pcook.scenarios import apply_resource_jitter


class CLITest(unittest.TestCase):
    def test_parse_levels(self):
        self.assertEqual(parse_levels("easy, medium ,hard"), ["easy", "medium", "hard"])
        self.assertEqual(parse_levels(""), [])

    def test_parse_weights(self):
        self.assertEqual(parse_weights("0.5, 0.75,1"), [0.5, 0.75, 1.0])
        self.assertEqual(parse_weights(""), [])

    def test_tune_fairness_weight(self):
        weights = [0.5, 0.75]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            best, evaluations = tune_fairness_weight(
                levels_path="tasks/levels.yaml",
                levels=["medium"],
                seeds=list(range(3)),
                weights=weights,
                scenario_modifier=lambda level, seed: apply_resource_jitter(level, seed, 0.25),
                output_dir=output_dir,
            )
        self.assertIn(best[0], weights)
        self.assertEqual(len(evaluations), len(weights))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
