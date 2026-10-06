"""Focused controls for the development closure diagnostic."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

path=Path(__file__).resolve().parents[2]/'workflows/sbi/e2e_true_density_closure.py'
spec=importlib.util.spec_from_file_location('closure_diagnostic',path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ClosureTests(unittest.TestCase):
    def test_zero_gap_preserves_rank(self):
        draws=np.random.default_rng(2).normal(size=(32,6))
        target=np.arange(6)/10
        r=m.comparison(draws,target,target)
        self.assertEqual(r['fullbox_joint'],r['matched_joint'])
        self.assertEqual(r['operator_mahalanobis'],0)
        np.testing.assert_equal(r['operator_gap_sd'],np.zeros(6))

    def test_exact_explanation_removes_residual(self):
        draws=np.random.default_rng(3).normal(size=(32,6))+2
        r=m.comparison(draws,np.zeros(6),draws.mean(0))
        self.assertAlmostEqual(r['matched_mahalanobis'],0)
        self.assertAlmostEqual(r['gap_residual_alignment'],1)
        np.testing.assert_allclose(r['operator_gap_sd'],r['original_bias_sd'])

    def test_opposite_gap_worsens_residual(self):
        draws=np.random.default_rng(4).normal(size=(32,6))+2
        r=m.comparison(draws,np.zeros(6),-draws.mean(0))
        self.assertAlmostEqual(r['gap_residual_alignment'],-1)
        self.assertAlmostEqual(r['matched_mahalanobis'],2*r['original_mahalanobis'])

    def test_core_coordinates(self):
        row=dict(center=[32,32,32],grid=dict(origin_mpc_h=[0,0,0]))
        cut=m.cuts(row,np.zeros(3))
        self.assertEqual([(s.start,s.stop) for s in cut],[(8,40),(8,24),(8,24)])
        with self.assertRaises(ValueError):
            m.cuts(row,np.ones(3))

    def test_gpu_operator_against_direct_cuboids(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('GPU control runs on the allocated GPU')
        from workflows.sbi.e2e_exterior_cap_tides import CapTidal
        rng=np.random.default_rng(11)
        shape=(3,2,2); pad=1
        rho=np.exp(rng.normal(scale=.1,size=(5,4,4)))
        operator=CapTidal(shape,pad)
        actual=np.stack(list(operator.components(rho)),axis=-1)
        expected=np.zeros_like(actual)
        for target in np.ndindex(shape):
            for source in np.ndindex(rho.shape):
                offset=np.asarray(target)+pad-np.asarray(source)
                expected[target]+=(rho[source]-1)*m.tidal.cube_tensor(*offset)
        np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=0)


if __name__=='__main__':
    unittest.main()
