import unittest
from workflows.sbi.e2e_multinoise_report import grouped, summarize, validate_probes
from workflows.sbi import e2e_wide_pipeline as p


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

    def test_probe_panel_rejects_duplicate_or_wrong_seed(self):
        panel = dict(fit_anchors=['a'], transfer_anchors=['b'], seed=7, evaluation_noise_replicates=1)
        rows = [dict(anchor_id=a, group=g, ratio=.05, rep=0, seed=p.seed_for(7, a, 'evaluation-0', 'fine'))
                for a, g in (('a', 'fit'), ('b', 'transfer'))]
        validate_probes(rows, panel, [.05])
        with self.assertRaises(ValueError):
            validate_probes(rows+rows[:1], panel, [.05])
        rows[0]['seed'] += 1
        with self.assertRaises(ValueError):
            validate_probes(rows, panel, [.05])


if __name__ == '__main__':
    unittest.main()
