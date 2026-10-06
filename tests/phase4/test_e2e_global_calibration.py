"""Focused sampler and statistical-estimand regressions."""
import unittest
import numpy as np
import torch
from workflows.sbi.e2e_global_sampler import integrate, addressed_noise
from workflows.sbi.e2e_global_mock_prepare import guard
from workflows.sbi.e2e_global_mock_evaluate import density_statistics, controls
from workflows.sbi.e2e_vdm_context_metrics import calibration


class ConstantVelocity:
    @staticmethod
    def schedule(t):
        return t

    def __call__(self, x, t, value):
        return torch.ones_like(x)*value


class GlobalCalibration(unittest.TestCase):
    def test_shared_step_averages_overlap_not_exterior(self):
        base=torch.zeros((1,1,56,2,2))
        left,right=integrate(ConstantVelocity(),'coarse',[1.,3.],base,nfe=4)
        np.testing.assert_allclose(left[:,:,0:8],1.)
        np.testing.assert_allclose(left[:,:,8:],2.)
        np.testing.assert_allclose(right[:,:,:40],2.)
        np.testing.assert_allclose(right[:,:,40:],3.)

    def test_independent_integration_preserves_local_velocities(self):
        base=[torch.zeros((1,1,48,2,2)) for _ in range(2)]
        a,b=integrate(ConstantVelocity(),'coarse',[1.,3.],base,nfe=4,fuse=False)
        np.testing.assert_allclose(a,1.)
        np.testing.assert_allclose(b,3.)

    def test_addressed_noise_batch_invariant(self):
        a=addressed_noise('coarse','test',0,2,device='cpu')
        b=addressed_noise('coarse','test',1,1,device='cpu')
        np.testing.assert_array_equal(a[1:],b)

    def test_sealed_phase_rejected(self):
        for phase in ('ph018','ph019','ph000'):
            with self.assertRaises(PermissionError):guard(phase)

    def test_seam_mass_and_contrast(self):
        rho=np.ones((64,48,48));rho[48:]=3
        value=density_statistics(rho)
        np.testing.assert_array_equal(value['seam_mass'],[0.,2.])
        np.testing.assert_array_equal(value['seam_sum_difference'],[1.,2.])
        np.testing.assert_allclose(value['seam_probes'],0.,atol=1e-12)

    def test_fair_crps_independent_pair_formula(self):
        x=np.array([[0.,3.],[1.,4.],[5.,9.]])
        y=np.array([2.,5.])
        pair=np.mean([abs(x[i]-x[j]) for i in range(3) for j in range(3) if i!=j],axis=0)
        expected=abs(x-y).mean(0)-.5*pair
        np.testing.assert_allclose(calibration(x,y)['crps'],expected)

    def test_synthetic_metric_controls(self):
        self.assertTrue(controls(32)['passed'])


if __name__=='__main__':
    unittest.main()
