import tempfile
import unittest
from pathlib import Path

import numpy as np

from workflows.sbi.e2e_wholefield_pilot_readout import aggregate, check_metrics, readout


class PilotReadoutTests(unittest.TestCase):
    def test_independent_formulas_and_corruption(self):
        x = np.random.default_rng(31).normal(size=(32, 6))
        y = np.arange(6)/10
        lo, hi = np.quantile(x, [.05, .95], axis=0)
        saved = dict(truth=y, mean=x.mean(0), sd=x.std(0, ddof=1), lower90=lo, upper90=hi,
                     coverage90=(lo <= y)&(y <= hi), rank=(x < y).mean(0),
                     crps=abs(x-y).mean(0)-.5*abs(x[:, None]-x[None, :]).mean((0, 1)),
                     energy_score=np.linalg.norm(x-y, axis=-1).mean()-.5*np.linalg.norm(x[:, None]-x[None, :], axis=-1).mean())
        check_metrics(x, saved)
        saved['energy_score'] += .01
        with self.assertRaises(AssertionError):
            check_metrics(x, saved)

    def test_empty_is_missing_not_zero(self):
        self.assertIsNone(aggregate([]))
        with tempfile.TemporaryDirectory() as directory:
            report = readout(Path(directory))
        self.assertEqual(report['completed_addresses'], 0)
        self.assertFalse(report['all_addresses_complete'])
        self.assertEqual(len(report['panels']), 4)
        self.assertTrue(all(s['scores'] is None and not s['complete']
                            for p in report['panels'] for s in p['splits']))

    def test_oracle_does_not_gain_sampled_baseline(self):
        full = dict(crps=[1]*6, coverage90=[True]*6, mean=[2]*6, truth=[1]*6,
                    sd=[3]*6, rank=[.5]*6, energy_score=4)
        row = dict(result=dict(fullbox=full, reference_fullbox=None,
                               density_voxels=dict(coverage90=.5, mean_crps=2)))
        result = aggregate([row])
        self.assertNotIn('paired_reference_crps_difference', result)
        self.assertEqual(result['cases'], 1)
        self.assertEqual(result['summary_component_rmse'], [1]*6)
        self.assertEqual(result['summary_component_rank_counts'][0][16], 1)


if __name__ == '__main__':
    unittest.main()
