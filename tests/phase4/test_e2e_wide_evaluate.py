import unittest
import numpy as np
from workflows.sbi.e2e_wide_evaluate import ensemble_metrics,distribution


class EvaluationTests(unittest.TestCase):
    def test_exact_constant_draws(self):
        truth=np.ones((2,2,2,3))
        result=ensemble_metrics(np.stack([truth]*4),truth,np.ones((2,2,2),bool))
        for values in result.values():
            np.testing.assert_allclose(values,0)

    def test_fair_crps_pair_normalization(self):
        truth=np.ones((1,1,1,3))
        draws=np.stack([truth*0,truth*2])
        result=ensemble_metrics(draws,truth,np.ones((1,1,1),bool))
        np.testing.assert_allclose(result['fair_marginal_crps'],0)
        np.testing.assert_allclose(result['mean_pointwise_draw_std'],np.sqrt(2))

    def test_empty_mask_rejected(self):
        with self.assertRaises(ValueError):
            ensemble_metrics(np.zeros((4,2,2,2,3)),np.zeros((2,2,2,3)),np.zeros((2,2,2),bool))

    def test_distribution_retains_components(self):
        self.assertEqual(distribution([[1,3],[3,5]])['median'],[2,4])


if __name__=='__main__':
    unittest.main()
