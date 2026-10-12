import unittest
import numpy as np
from workflows.sbi import e2e_wholefield_matched_screen as s


class MatchedScreenTests(unittest.TestCase):
    def test_ratio_pools_terms_not_ratios(self):
        q = np.array([[2., 6.], [9., 9.]])
        v = np.array([[1., 1.], [3., 3.]])
        self.assertEqual(s.ratio(q, v), 3.25)
        self.assertNotEqual(s.ratio(q, v), (q/v).mean())

    def test_floor_and_acceptance_are_distinct(self):
        base = np.array([1.5, 1.5, 1.5])
        candidate = np.array([[[1.44, 1.4, 1.49]]])
        np.testing.assert_array_equal(s.necessary_event(base, candidate, .08), [False, True, False])

    def test_no_per_phase_requirement(self):
        self.assertTrue(s.necessary_event(np.array([1.5]), np.array([[[1.3]]]), .01)[0])

    def test_hoeffding(self):
        result = s.bound(np.zeros(4096, bool), 120, .01)
        self.assertLess(result['upper'], .04)
        self.assertEqual(result['lower'], 0.)

    def test_full_rule_subset_and_selector(self):
        ref = dict(q=np.full((2, 6, 2), 1.5), v=np.ones((2, 6, 2)), energy=np.ones((2, 6, 2)),
                   coverage=np.full((2, 6, 2, 6), .9), crps=np.ones((2, 6, 2, 6)))
        cand = dict(q=np.full((2, 2, 4, 6, 2), 1.2), v=np.ones((2, 2, 4, 6, 2)), energy=np.full((2, 2, 4, 6, 2), .7),
                    coverage=np.full((2, 2, 4, 6, 2, 6), .9), crps=np.ones((2, 2, 4, 6, 2, 6)))
        result = s.complete_rule(ref, cand, .05, .02)
        self.assertTrue(result['passed']); self.assertEqual(result['selected'], [0, 0])
        self.assertTrue(s.necessary_event(np.array([1.5]), s.ratio(cand['q'][:, 1], cand['v'][:, 1])[..., None], .05)[0])
        cand['q'][:] = 1.49
        self.assertFalse(s.complete_rule(ref, cand, .05, .02)['passed'])

    def test_simulation_same_noise_zero_effect_identity(self):
        cfg = dict(replicates=12, batch_size=4, effects=[0., .2, .4, .6, 1.])
        ff = [dict(q=[1.5, 1.6], v=[1., 1.], eigen=[.5, .5], direction=[1., 0.]) for _ in range(6)]
        for law in ('gaussian_scale', 'biased', 'common_mode', 'heavy_tail'):
            got = s.simulate((law, 1., ff, cfg, 123))
            np.testing.assert_allclose(got['candidates'][0], np.broadcast_to(got['base'], (2, 4, 12)), atol=1e-14)
            self.assertFalse(s.necessary_event(got['base'], got['candidates'][0], .01).any())
            np.testing.assert_allclose(got['target_ratios'], 1+(1-np.array(cfg['effects']))*.55)


if __name__ == '__main__':
    unittest.main()
