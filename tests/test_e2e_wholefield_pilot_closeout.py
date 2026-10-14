import unittest

from workflows.sbi.e2e_wholefield_pilot_closeout import (
    compare, paired, development_point, central_rank_coverage, reconcile_groups, SPLITS,
)


def row(pair, energy=1, crps=2, truth=0):
    return dict(phase='ph012', pair=pair, seed=17,
                result=dict(fullbox=dict(truth=[truth]*6, energy_score=energy),
                            density_voxels=dict(mean_crps=crps, coverage90=.8)))


class CloseoutTests(unittest.TestCase):
    def test_phase_split_reconciliation_and_corruption(self):
        groups = []
        for architecture in ('unet', 'transformer'):
            for coarse in ('sampled', 'oracle'):
                for split, phases in SPLITS.items():
                    for kind, value, count in [('split', split, 32*len(phases))]+[('phase', p, 32) for p in phases]:
                        scores = dict(cases=count, density_voxel_crps=2., density_voxel_coverage90=.8,
                                      energy_score=1., summary_component_crps=[2.]*6,
                                      summary_component_coverage90=[.8]*6, component_mean_bias=[.1]*6)
                        groups.append(dict(kind=kind, value=value, architecture=architecture,
                                           coarse=coarse, scores=scores))
        self.assertEqual(reconcile_groups(groups), 176)
        groups[0]['scores']['component_mean_bias'][0] = .2
        with self.assertRaises(ValueError):
            reconcile_groups(groups)

    def test_exchangeable_rank_reference(self):
        self.assertAlmostEqual(float(central_rank_coverage(list(range(33)))), 29/33)
        with self.assertRaises(ValueError):
            central_rank_coverage([.5])

    def test_missing_development_loss_remains_missing(self):
        point = development_point(dict(available=False, reason='data not ready'), 4)
        self.assertIsNone(point['fixed_noise_cfm_loss'])
        self.assertFalse(point['available'])
        with self.assertRaises(ValueError):
            development_point(dict(available=True, ids=[['ph014', 'pair']], fixed_noise_cfm_loss=1), 4)

    def test_pairing_ignores_input_order(self):
        out = paired([row('a'), row('b')], [row('b'), row('a')])
        self.assertEqual([(a['pair'], b['pair']) for a, b in out], [('a', 'a'), ('b', 'b')])

    def test_bad_pairings_rejected(self):
        for left, right in (([], []), ([row('a')], [row('b')]),
                            ([row('a'), row('a')], [row('a')])):
            with self.assertRaises(ValueError):
                paired(left, right)
        with self.assertRaises(AssertionError):
            paired([row('a')], [row('a', truth=1)])

    def test_difference_direction_and_denominator(self):
        out = compare([row('a', energy=1), row('b', energy=2)],
                      [row('b', energy=4), row('a', energy=2)])
        self.assertEqual(out['cases'], 2)
        self.assertEqual(out['energy']['mean_difference'], -1.5)
        self.assertEqual(out['energy']['fraction_negative'], 1)
        self.assertEqual(out['density_crps']['mean_difference'], 0)


if __name__ == '__main__':
    unittest.main()
