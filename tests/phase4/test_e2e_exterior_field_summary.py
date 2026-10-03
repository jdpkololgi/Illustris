import unittest
import numpy as np
from workflows.sbi.e2e_exterior_field_summary import marginal_summary, classes, LEVELS
from workflows.sbi.e2e_vdm_context_metrics import central_order_interval, calibration


class FieldSummaryTests(unittest.TestCase):
    def test_matches_frozen_interval_definition(self):
        draws = np.arange(32*8, dtype='f4').reshape(32, 2, 2, 2)/10
        intervals = [central_order_interval(32, level) for level in LEVELS]
        summary = marginal_summary(draws, intervals)
        np.testing.assert_allclose(summary['mean'], draws.mean(0, dtype='f8'), rtol=1e-7)
        np.testing.assert_allclose(summary['std'], draws.std(0, dtype='f8', ddof=1), rtol=1e-7)
        truth = draws[4]; score = calibration(draws, truth)
        for index, level in enumerate(LEVELS):
            np.testing.assert_array_equal(summary['lower'][index], draws[intervals[index][0]])
            np.testing.assert_array_equal(summary['upper'][index], draws[intervals[index][1]])
            np.testing.assert_array_equal((truth >= summary['lower'][index]) & (truth <= summary['upper'][index]), score[f'covered_{level}'])
    def test_exact_class_probabilities_entropy(self):
        eigen = np.tile(np.array([[-1., -1., -1.], [-1., -1., 1.], [-1., 1., 1.], [1., 1., 1.]], dtype='f4'), (8, 1))
        probability, entropy = classes(eigen, 0)
        np.testing.assert_array_equal(probability, np.ones(4)/4)
        self.assertEqual(entropy, 1)
        probability, entropy = classes(np.ones((32, 3)), .2)
        np.testing.assert_array_equal(probability, [0, 0, 0, 1]); self.assertEqual(entropy, 0)
    def test_rejects_partial_or_nonfinite_ensemble(self):
        intervals = [central_order_interval(32, level) for level in LEVELS]
        with self.assertRaises(ValueError): marginal_summary(np.ones((4, 2)), intervals)
        with self.assertRaises(ValueError): marginal_summary(np.full((32, 2), np.nan), intervals)


if __name__ == '__main__':
    unittest.main()
