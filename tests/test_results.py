
import tempfile
import unittest
from pathlib import Path

from pcook.results import (
    aggregate_summaries,
    write_aggregated_csv,
    write_aggregated_markdown,
    write_fairness_csv,
    write_fairness_report,
)

SUMMARY_TEXT = """# ParaCook Scoreboard Summary

## Level: medium
- **parallel**: makespan 16.00±0.00, OCT mean 14.33±0.00, plan adherence 1.00, blocked rate 0.00 (resource 0.00, dependency 0.00), avg wait 2.87, oracle gap 5.00, efficiency 0.69
- **sequential**: makespan 21.00±0.00, OCT mean 17.00±0.00, plan adherence 0.94, blocked rate 0.06 (resource 0.06, dependency 0.00), avg wait 2.80, oracle gap 10.00, efficiency 0.52

Δ (parallel - sequential): makespan -5.00, OCT -2.67, resource-block rate -0.06.

"""


class ResultsTest(unittest.TestCase):
    def test_aggregate_and_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'summary.md'
            path.write_text(SUMMARY_TEXT)
            aggregated = aggregate_summaries({'test': path})
            self.assertEqual(len(aggregated), 1)
            item = aggregated[0]
            self.assertEqual(item.level, 'medium')
            self.assertEqual(item.delta_makespan, -5.0)
            self.assertAlmostEqual(item.delta_adherence, 0.06)

            csv_path = Path(tmpdir) / 'agg.csv'
            md_path = Path(tmpdir) / 'agg.md'
            write_aggregated_csv(aggregated, csv_path)
            write_aggregated_markdown(aggregated, md_path)
            csv_text = csv_path.read_text()
            md_text = md_path.read_text()
            self.assertIn('delta_adherence', csv_text)
            self.assertIn('Scenario: test', md_text)
            self.assertIn('Δ Plan Adherence', md_text)

            fairness_csv = Path(tmpdir) / 'fairness.csv'
            fairness_md = Path(tmpdir) / 'fairness.md'
            from pcook.results import write_fairness_csv, write_fairness_report
            sample_evals = [(0.75, 0.01, 14.0, Path('tmp'))]
            write_fairness_csv(sample_evals, fairness_csv)
            write_fairness_report(sample_evals, fairness_md)
            self.assertIn('0.750', fairness_csv.read_text())
            self.assertIn('0.750', fairness_md.read_text())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
