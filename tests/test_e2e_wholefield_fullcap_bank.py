import unittest

import numpy as np

from workflows.sbi.e2e_wholefield_fullcap_bank import pooled, terms


class BankTests(unittest.TestCase):
    def test_independent_covariance_and_energy(self):
        rng = np.random.default_rng(127)
        x = rng.normal(size=(16, 32, 6)); y = rng.normal(size=(16, 6))
        q, v, energy = terms(x, y)
        variances = [np.trace(np.cov(case, rowvar=False, ddof=1)) for case in x]
        errors = [np.dot(case.mean(0)-target, case.mean(0)-target) for case, target in zip(x, y)]
        scores = []
        for case, target in zip(x, y):
            distances = np.linalg.norm(case[:, None, :]-case[None, :, :], axis=-1)
            scores.append(np.linalg.norm(case-target, axis=-1).mean()-distances.sum()/(2*32*31))
        self.assertAlmostEqual(float(v), np.mean(variances), places=13)
        self.assertAlmostEqual(float(q), np.mean(errors)-np.mean(variances)/32, places=13)
        self.assertAlmostEqual(float(energy), np.mean(scores), places=13)

    def test_pool_traces_not_ratios(self):
        q = np.array([[1., 4.], [9., 16.]])
        v = np.array([[1., 2.], [3., 4.]])
        result = pooled(q, v, np.zeros_like(q))
        self.assertEqual(result['ratio'], 3.)
        self.assertNotEqual(result['ratio'], (q/v).mean())

    def test_leading_phase_seed_axes(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(6, 2, 16, 32, 6)); y = rng.normal(size=(6, 2, 16, 6))
        got = terms(x, y)
        for pi in range(6):
            for si in range(2):
                expected = terms(x[pi, si], y[pi, si])
                for result, value in zip(got, expected):
                    self.assertAlmostEqual(result[pi, si], value, places=13)

    def test_invalid_shape_and_zero_variance(self):
        with self.assertRaises(ValueError):
            terms(np.ones((15, 32, 6)), np.ones((15, 6)))
        with self.assertRaises(ValueError):
            pooled(np.ones(6), np.zeros(6), np.ones(6))


if __name__ == '__main__':
    unittest.main()
