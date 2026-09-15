import unittest
from workflows.sbi.e2e_multinoise_report import grouped, summarize


class ReportTests(unittest.TestCase):
    def test_grouped_interpolation_is_not_hardcoded(self):
        metrics = {k: [1., 2., 3., 4.] for k in ('gain', 'error_power', 'error_over_truth_power', 'noise_amplitude', 'power_ratio')}
        rows = [dict(group=g, ratio=r, metrics=metrics, velocity_mse=2.) for g in ('fit', 'transfer') for r in (.08, .5)]
        result = grouped(rows)
        self.assertEqual(set(result), {'fit/0.08', 'fit/0.5', 'transfer/0.08', 'transfer/0.5'})
        self.assertEqual(result['transfer/0.08']['gain'], [1., 2., 3., 4.])

    def test_reject_incomplete(self):
        with self.assertRaises(ValueError):
            summarize(dict(complete=False, results=[], baseline=[], fields=[], checkpoints={}))


if __name__ == '__main__':
    unittest.main()
