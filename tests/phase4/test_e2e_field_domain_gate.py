import unittest

import numpy as np

from workflows.sbi.e2e_field_domain_gate import failures, gate_values, traceless_constant
from workflows.sbi.e2e_field_build_products import tensor_from_delta


class DomainGateTests(unittest.TestCase):
    def test_thresholds_inclusive_and_no_missing_gate(self):
        self.assertEqual(failures({"a":1.},{"a":1.}),[])
        self.assertEqual(failures({"a":1.1},{"a":1.}),["a"])
        with self.assertRaises(ValueError):
            failures({"a":1.},{"b":1.})
        with self.assertRaises(ValueError):
            failures({"a":float("nan")},{"a":1.})

    def test_traceless_oracle_does_not_change_density(self):
        x = np.broadcast_to(np.array([2.,.2,.3,1.,.4,3.]),(4,4,4,6)).copy()
        correction = traceless_constant(x,np.ones((4,)*3,dtype=bool))
        self.assertAlmostEqual(correction[[0,3,5]].sum(),0.)
        np.testing.assert_allclose((x-correction)[...,[0,3,5]].sum(-1),6.)

    def test_parent_mean_and_trace_convention(self):
        delta = np.random.default_rng(61).normal(size=(8,)*3)+.7
        tensor = tensor_from_delta(delta,3.383)
        np.testing.assert_allclose(tensor[...,[0,3,5]].sum(-1),delta,atol=2e-15)
        np.testing.assert_allclose(tensor.mean(axis=(0,1,2))[[0,3,5]],delta.mean()/3,atol=2e-15)

    def test_gate_values_keep_topology_separate(self):
        metrics = {"eigen_rmse":[0.,0.,0.],"eigen_bias":[0.,0.,0.],
                   "eigen_rmse_over_truth_std":[0.,0.,0.],"four_class_disagreement":0.}
        truth = {"filling_fraction":.1,"largest_void_fraction":.2,"pair":[{"value":.01}]*3,"connections_xyz":[False]*3}
        candidate = {**truth,"connections_xyz":[True,False,False]}
        values = gate_values(metrics,candidate,truth)
        self.assertEqual(values["changed_connection_axes_max"],1)
        self.assertEqual(values["eigen_rmse_max_abs"],0.)
        with self.assertRaises(ValueError):
            gate_values(metrics,{**truth,"pair":[{"value":None}]},truth)


if __name__ == "__main__":
    unittest.main()
